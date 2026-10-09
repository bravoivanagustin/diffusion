"""Pérdida de validación por **examen fijo**: la métrica que mide si el modelo generaliza.

La pérdida de DSM es ruidosa en el ``t`` y en el ruido sorteados, así que una validación que
re-sortea en cada medición produce una curva que salta por el sorteo y no por el modelo. Este
submódulo define la métrica que evita eso: :class:`FixedValExam` **congela el examen** —las
mismas imágenes, los mismos ``t`` y el mismo ruido en todas las evaluaciones de una corrida— de
modo que la única variable entre dos puntos de la curva sea la red.

El congelamiento se logra por **re-siembra**, no materializando tensores: en cada
:meth:`FixedValExam.evaluate` se crea un ``torch.Generator`` nuevo sembrado con
:data:`VAL_EXAM_SEED` y se recorre la fuente re-iterable en orden fijo, así que
:meth:`~diffusion.training.time_sampling.TimeSampler.sample` y
:func:`~diffusion.training.losses.dsm_loss` sortean exactamente la misma secuencia. La
consecuencia útil es que no hay **nada** que persistir: reanudar una corrida reconstruye el mismo
examen con solo reconstruir la fuente.

Dos propiedades de diseño:

- **Observa y no interviene**: corre sin gradientes, no toca los pesos (restaura el modo de la red
  incluso si la evaluación revienta) y saca **todo** su azar del generator local — ni el generator
  del loop ni el RNG global se mueven, así que activar la validación no cambia los pesos que
  produce una semilla dada.
- **No conoce el loop**: recibe la SDE, la fuente de batches y el muestreador de tiempos **ya
  construidos**. Que el muestreador sea el mismo objeto que usa el entrenamiento es lo que hace
  *estructural* la comparabilidad de las dos curvas (mismo criterio de pérdida y mismo esquema de
  muestreo de ``t``), en lugar de depender de que alguien recuerde configurar ambos igual.

La clase no sabe si su fuente es de validación o de entrenamiento: la corrida instancia **dos**
exámenes con la misma semilla y el mismo muestreador —uno por carpeta—, y con la misma cantidad de
imágenes y el mismo batcheo ambos usan la misma secuencia de ``t`` y de ruido, así que la
diferencia entre las dos curvas no puede venir del examen.

**El examen es específico del device** (invariante, no defecto): un ``torch.Generator`` de CUDA
sembrado con la misma semilla que uno de CPU produce otra secuencia, y el ruido se sortea en el
device del dato. La serie es comparable **dentro de un mismo device**; por eso el registro del log
lleva el device, para poder detectar el cambio en lugar de leerlo como un salto del modelo.

La segunda pieza, :func:`evaluate_with_weights`, mide el **mismo** examen con un juego de pesos
ajeno (típicamente la sombra EMA) y devuelve la red exactamente como estaba. Recibe un ``Mapping``
genérico a propósito: no conoce :class:`~diffusion.training.ema.EmaShadow`, así que sirve para
comparar cualquier par de pesos.

Uso típico (lo orquesta el loop de :mod:`diffusion.training.trainer`)::

    exam = FixedValExam(sde, val_batches, time_sampler=time_sampler, device=device)
    val_raw = exam.evaluate(net)                                  # mismo examen en cada llamada
    val_ema = evaluate_with_weights(exam, net, ema.state_dict())  # ídem, con la sombra
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final, TypedDict

import torch

from ..models import ScoreModel
from ..sde import ForwardSDE
from .losses import dsm_loss
from .time_sampling import TimeSampler

#: Semilla del examen fijo. Es una **constante congelada**: cambiar su valor cambia el examen y
#: deja de ser comparable con todas las curvas de validación ya medidas.
VAL_EXAM_SEED: Final[int] = 12345


class ValPoint(TypedDict):
    """Un punto de la serie **dispersa** de validación (una evaluación).

    Es un ``TypedDict`` —un ``dict`` en runtime— a propósito: el punto viaja dentro de la metadata
    del checkpoint (``torch.save``) y en el log ``.jsonl``, y un dict no ata la lectura de esos
    archivos a que la clase exista o sea importable.

    Attributes:
        step: Paso en que se midió, *1-indexado* (misma convención que el contador del EMA).
        raw: Pérdida de **validación** con los pesos vivos.
        ema: Ídem con la sombra EMA; ``None`` si la corrida no mantiene sombra.
        train: Examen fijo de **entrenamiento** con los pesos vivos; ``None`` si no se inyectó la
            fuente correspondiente.
    """

    step: int
    raw: float
    ema: float | None
    train: float | None


class FixedValExam:
    """Examen fijo: la pérdida de DSM sobre un conjunto completo, congelada entre evaluaciones.

    Args:
        sde: Proceso forward de la corrida (define el kernel de perturbación y el target del
            score). Es el mismo objeto con el que se entrena: la pérdida de validación no sería
            comparable con la de entrenamiento si la SDE difiriera.
        batches: Fuente **re-iterable** de batches ``(B, *E)`` con la forma de evento de ``sde``
            (típicamente lo que devuelve ``finite_batches``, o una lista de tensores en los
            tests). Se recorre entera en cada evaluación, en orden fijo; el último batch puede ser
            parcial. Re-iterable es un requisito, no una preferencia: un iterador de un solo uso
            quedaría agotado después de la primera evaluación.
        time_sampler: Muestreador de ``t`` **ya construido** — el mismo que usa el loop, para que
            val y train compartan el criterio.
        device: Dispositivo donde se evalúa (el mismo en el que se entrena). Los batches se mueven
            ahí y el generator local se crea ahí.
        seed: Semilla del examen; por defecto :data:`VAL_EXAM_SEED`. Se expone solo para poder
            construir exámenes deliberadamente distintos (tests): en una corrida se deja el
            default.

    Attributes:
        device: El device normalizado a :class:`torch.device`.
        seed: La semilla con la que se re-siembra el generator en cada evaluación.
    """

    def __init__(
        self,
        sde: ForwardSDE,
        batches: Iterable[torch.Tensor],
        *,
        time_sampler: TimeSampler,
        device: torch.device | str = "cpu",
        seed: int = VAL_EXAM_SEED,
    ) -> None:
        self._sde = sde
        self._batches = batches
        self._time_sampler = time_sampler
        self.device = torch.device(device)
        self.seed = int(seed)

    def evaluate(self, net: ScoreModel) -> float:
        """Computa la pérdida del examen para la red recibida.

        Recorre **todos** los batches de la fuente y promedia sus pérdidas **ponderando por la
        cantidad de imágenes** (``Σ loss_batch · b / Σ b``), de modo que el valor no dependa de
        cómo se batchee el conjunto (el último batch puede ser parcial).

        El azar sale enteramente de un ``torch.Generator`` nuevo, sembrado con :attr:`seed` en
        cada llamada: dos evaluaciones de una red inalterada devuelven exactamente el mismo
        número, y ni el RNG global ni ningún generator externo cambian de estado.

        Args:
            net: Red de score ``(x, t) -> score`` con la forma de evento de la SDE. Se evalúa en
                modo ``eval()`` y sin gradientes; al retornar —o al propagar una excepción— queda
                con los mismos pesos y el mismo modo que tenía.

        Returns:
            La pérdida media por imagen, como ``float``.

        Raises:
            ValueError: Si la fuente no entregó ninguna imagen (guard defensivo: el config layer
                ya rechaza una carpeta vacía).
        """
        # Re-siembra: generator NUEVO por evaluación, en el device de la evaluación (``perturb``
        # sortea el ruido con ``device=x0.device``, y torch exige que el generator coincida).
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)

        acumulado = 0.0  # Σ loss_batch · b
        imagenes = 0  # Σ b
        modo_previo = net.training
        net.eval()
        try:
            # ``autocast(enabled=False)`` es deliberado y no redundante: fija fp32 aunque la
            # evaluación se llame desde una región con precisión mixta activa. El número tiene que
            # ser comparable entre celdas entrenadas con y sin AMP, y el costo de la evaluación es
            # marginal frente a la corrida, así que no hay nada que optimizar acá.
            with (
                torch.no_grad(),
                torch.autocast(device_type=self.device.type, enabled=False),
            ):
                for batch in self._batches:
                    x0 = batch.to(self.device, non_blocking=True)
                    b = x0.shape[0]
                    if b == 0:
                        # Un batch vacío no aporta imágenes y su media sería NaN. La fuente finita
                        # no los produce; si aparece uno, se saltea sin contaminar el promedio (y
                        # sin consumir el generator, para no desplazar el examen).
                        continue
                    t, sample_weights = self._time_sampler.sample(
                        b, generator=gen, device=self.device
                    )
                    loss = dsm_loss(
                        net, self._sde, x0, t, generator=gen, sample_weights=sample_weights
                    )
                    acumulado += loss.item() * b
                    imagenes += b
        finally:
            # En el ``finally`` para que una excepción del forward no deje la red en modo
            # evaluación: el loop seguiría entrenando con el modo equivocado.
            net.train(modo_previo)

        if imagenes == 0:
            raise ValueError(
                "el examen fijo no recibió ninguna imagen: la fuente de batches está vacía. "
                "Revisá que la carpeta del examen tenga imágenes y que la fuente sea "
                "re-iterable (un iterador de un solo uso queda agotado tras la primera "
                "evaluación)."
            )
        return acumulado / imagenes


def evaluate_with_weights(
    exam: FixedValExam, net: ScoreModel, weights: Mapping[str, torch.Tensor]
) -> float:
    """Corre el examen con un juego de pesos ajeno y deja la red **exactamente** como estaba.

    Es la mecánica del intercambio que necesita la corrida para medir la validación con la sombra
    EMA sobre el **mismo** examen que los pesos vivos (criterio 4.1), sin mantener una segunda
    instancia de red: se clonan los pesos vivos, se cargan los recibidos, se evalúa y se restauran
    los crudos (criterio 4.4).

    La función recibe un ``Mapping`` genérico y **no** conoce la clase del EMA: sirve para
    comparar cualquier par de juegos de pesos. Tampoco consume azar — todo el sorteo pasa por el
    generator local de :meth:`FixedValExam.evaluate`.

    Args:
        exam: Examen fijo ya construido (define la métrica, las imágenes y el device).
        net: Red de score a evaluar. Al retornar —o al propagar una excepción— su ``state_dict``
            es **idéntico** al de la entrada, tensor por tensor; también su modo (lo restaura
            :meth:`FixedValExam.evaluate`).
        weights: Pesos con los que medir; tienen que ser exactamente las claves de
            ``net.state_dict()`` (lo garantiza :meth:`~diffusion.training.ema.EmaShadow.state_dict`,
            incluso con la red envuelta en ``EpsilonScoreWrapper``, cuyo ``state_dict`` delega al
            interno). No se mutan.

    Returns:
        La pérdida del examen medida **bajo** ``weights``, como ``float``.

    Raises:
        RuntimeError: Si las claves de ``weights`` no coinciden con las de la red (fail-fast de
            ``load_state_dict``, que se deja propagar tal cual).
        ValueError: Lo que levante :meth:`FixedValExam.evaluate` (fuente sin imágenes).
    """
    # CLONAR, no referenciar: ``state_dict()`` devuelve los tensores VIVOS de la red, así que
    # guardarlo tal cual como respaldo lo dejaría sobreescrito in-place por el ``load_state_dict``
    # de abajo — la restauración sería un no-op y la red se quedaría con los pesos ajenos para
    # siempre. Mismo cuidado que el sidecar de resume en ``trainer.py``.
    crudos = {k: v.detach().clone() for k, v in net.state_dict().items()}
    try:
        # La carga va DENTRO del bloque protegido: con claves que no coinciden,
        # ``load_state_dict`` copia igual las que sí matchean y solo después levanta el
        # ``RuntimeError``, así que sin esto el fail-fast dejaría una mezcla de pesos.
        net.load_state_dict(weights)
        return exam.evaluate(net)
    finally:
        # En el ``finally`` para que ninguna excepción —de la carga o del forward— deje la red con
        # pesos que no son los del entrenamiento. La restauración es in-place (``copy_`` dentro de
        # ``load_state_dict``), así que las referencias del optimizador siguen siendo válidas.
        net.load_state_dict(crudos)
