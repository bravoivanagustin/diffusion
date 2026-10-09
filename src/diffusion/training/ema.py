"""Sombra EMA de los pesos: la media móvil exponencial que publican los checkpoints.

La pérdida de DSM es muy ruidosa per-step (``t`` aleatorio en cada paso; batch chico en la celda
de gatos), así que los pesos del último paso de Adam son **una foto arbitraria** de una
trayectoria que oscila. Las implementaciones de referencia (Song et al., ``score_sde_pytorch``
→ ``models/ema.py``; DDPM; EDM/Karras et al.) nunca samplean con esos pesos: mantienen durante
el entrenamiento una **media móvil exponencial** de los parámetros y samplean con ella.

Este submódulo aporta esa pieza —y solo esa— como un objeto aislado y testeable:

.. math::

    \\text{ema}_s = d_s \\cdot \\text{ema}_{s-1} + (1 - d_s)\\cdot \\theta_s,
    \\qquad d_s = \\min\\!\\left(d, \\frac{1+s}{10+s}\\right)

**Convención del contador** (fijada acá; cambiarla invalida la comparabilidad entre corridas y
la réplica cerrada de los tests): ``s`` es la cantidad de **pasos completados**, *1-indexado* —
se llama ``update(s)`` justo después del ``optimizer.step()`` número ``s``, y la sombra arranca
en ``ema_0 = θ_0`` (los pesos con los que se construyó). El factor ``(1+s)/(10+s)`` es la
**rampa de warmup** estilo Karras: con ``s=1`` vale ``2/11 ≈ 0.18`` (el arranque es casi un
promedio simple, la inicialización no domina) y crece hacia ``1``, tocando ``d = 0.999`` en
``s ≈ 8990`` — para corridas cortas (2k–19k pasos) la ventana efectiva crece durante buena
parte de la corrida, que es justo el comportamiento deseado.

Dos propiedades de diseño:

- **Pasiva**: la sombra solo *lee* los tensores del módulo (después del paso del optimizador).
  No escribe en la red, no toca el optimizador y **no consume RNG** — la trayectoria de
  optimización con EMA activo es idéntica, con la misma semilla, a la de una corrida sin EMA.
- **Agnóstica a la red**: opera sobre el ``state_dict`` del módulo recibido sin ramificar por su
  tipo (``ScoreMLP``, ``ScoreUNet``, o el ``EpsilonScoreWrapper`` — cuyo ``state_dict`` delega al
  interno, así que las claves de la sombra quedan de **red pelada** y el checkpoint publicado
  sigue siendo reconstruible por ``make_model`` + la receta).

El EMA se aplica a los tensores **entrenables**; los buffers no entrenables se copian del módulo
vivo al publicar (:meth:`EmaShadow.state_dict`). Es la convención de referencia y en este repo es
*exacta*, no aproximada: los buffers son constantes (el ``denom`` del embedding sinusoidal;
GroupNorm no tiene running stats). Si algún día aparece un buffer no constante, este supuesto
deja de ser exacto (Revalidation Trigger de la spec ``ema-weights``).

Uso típico (lo orquesta el loop de :mod:`diffusion.training.trainer`)::

    shadow = EmaShadow(net, decay=0.999)      # después de net.to(device)
    for s in range(1, num_steps + 1):
        ...                                    # forward, backward, optimizer.step()
        shadow.update(s)                       # s = pasos completados (1-indexado)
    foto = shadow.state_dict()                 # lo que se publica en el checkpoint
"""

from __future__ import annotations

import math

import torch


class EmaShadow:
    """Media móvil exponencial de los pesos de un módulo, con rampa de warmup.

    Rastrea clones de los tensores **entrenables** del módulo recibido y los actualiza en cada
    paso con ``ema = d_s·ema + (1−d_s)·θ`` (ver el docstring del módulo para la convención del
    contador y la rampa). Es un observador pasivo: nunca modifica el módulo ni consume RNG.

    Las claves de la sombra son las del ``state_dict`` del módulo (no las de
    ``named_parameters``): así :meth:`state_dict` es cargable con
    ``load_state_dict(strict=True)`` en una red reconstruida por la receta, incluso cuando el
    módulo observado es el ``EpsilonScoreWrapper`` (que delega su ``state_dict`` al interno).

    Args:
        module: Módulo a observar (cualquier :class:`~diffusion.models.ScoreModel`). La sombra
            guarda una referencia para leer sus pesos en cada :meth:`update` y sus buffers al
            publicar; no lo modifica nunca.
        decay: Decay del EMA ``d``; debe ser finito y estar en el intervalo **abierto** ``(0, 1)``
            (0.999 es el valor recomendado del estudio).

    Attributes:
        decay: El decay configurado (techo de la rampa de warmup).

    Raises:
        ValueError: Si ``decay`` no es finito o cae fuera de ``(0, 1)`` — fail-fast en
            construcción, antes de entrenar, con el valor recibido en el mensaje. También si no se
            pudo rastrear **ningún** tensor entrenable del módulo (sombra vacía: promediaría nada
            en silencio).
    """

    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        self.decay = _validar_decay(decay)
        self._module = module
        entrenables = {id(p) for p in module.parameters() if p.requires_grad}
        vivos = module.state_dict(keep_vars=True)
        # Claves del state_dict cuyo tensor ES un parámetro entrenable del módulo. Se filtra por
        # identidad (no por nombre) para no depender de cómo el módulo arme sus claves.
        self._nombres = [nombre for nombre, t in vivos.items() if id(t) in entrenables]
        if entrenables and not self._nombres:
            # El módulo tiene parámetros entrenables pero su ``state_dict`` no los expuso como
            # tales: el caso típico es un override que no reenvía ``keep_vars`` y devuelve copias
            # detachadas, así que el filtro por identidad no matchea nada. Sin este guard la
            # sombra quedaría vacía y el checkpoint publicaría pesos crudos disfrazados de EMA.
            raise ValueError(
                f"EmaShadow no pudo rastrear ningún tensor entrenable de {type(module).__name__}: "
                f"tiene {len(entrenables)} parámetro(s) entrenable(s) pero su state_dict no "
                "devolvió los tensores vivos. Revisá que el módulo reenvíe keep_vars=True a "
                "nn.Module.state_dict (la sombra los identifica por identidad)."
            )
        self._sombra = {nombre: vivos[nombre].detach().clone() for nombre in self._nombres}

    def update(self, step: int) -> None:
        """Actualiza la sombra con los pesos actuales del módulo.

        Se llama **después** del ``optimizer.step()`` número ``step``.

        Args:
            step: Cantidad de pasos completados, *1-indexado* (el primer paso es ``1``). Fija el
                peso de la rampa de warmup: ``d_s = min(decay, (1+step)/(10+step))``.

        Raises:
            ValueError: Si ``step < 1`` (el contador es 1-indexado por convención).
        """
        if step < 1:
            raise ValueError(
                "step es la cantidad de pasos completados (1-indexado) y debe ser >= 1; "
                f"se recibió step={step}"
            )
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        vivos = self._module.state_dict(keep_vars=True)
        with torch.no_grad():
            for nombre in self._nombres:
                self._sombra[nombre].mul_(d).add_(vivos[nombre].detach(), alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Foto publicable de la sombra: el ``state_dict`` completo del módulo con EMA.

        Returns:
            Un dict con **las mismas claves** que ``module.state_dict()``: los tensores
            entrenables reemplazados por su valor EMA y el resto (buffers no entrenables,
            parámetros congelados) copiado del módulo vivo. Todos los tensores son **clones**:
            la foto no cambia si la sombra sigue actualizándose, y mutarla no afecta a la sombra
            (es lo que la hace segura para snapshots y checkpoints).
        """
        vivos = self._module.state_dict(keep_vars=True)
        return {
            nombre: (self._sombra[nombre] if nombre in self._sombra else t).detach().clone()
            for nombre, t in vivos.items()
        }

    def load_state(self, shadow: dict[str, torch.Tensor]) -> None:
        """Restaura la sombra desde una foto previa (reanudación de una corrida).

        Args:
            shadow: Foto producida por :meth:`state_dict` (claves extra —p. ej. buffers— se
                ignoran: los buffers se toman siempre del módulo vivo al publicar).

        Raises:
            ValueError: Si falta alguna de las claves rastreadas por la sombra; nombra las que
                faltan en lugar de continuar con una sombra a medias.
        """
        faltantes = [nombre for nombre in self._nombres if nombre not in shadow]
        if faltantes:
            raise ValueError(
                "la foto de la sombra EMA no tiene todas las claves entrenables del módulo; "
                f"faltan: {', '.join(faltantes)}"
            )
        with torch.no_grad():
            for nombre in self._nombres:
                self._sombra[nombre].copy_(shadow[nombre])


def _validar_decay(decay: float) -> float:
    """Valida el decay del EMA: finito y en el intervalo abierto ``(0, 1)`` (fail-fast).

    Args:
        decay: Valor recibido de la configuración.

    Returns:
        El valor como ``float``.

    Raises:
        ValueError: Si no es convertible a float, no es finito, o cae fuera de ``(0, 1)``. El
            mensaje incluye el valor recibido.
    """
    mensaje = (
        "ema_decay debe ser finito y estar en el intervalo abierto (0, 1); "
        f"se recibió decay={decay}"
    )
    try:
        valor = float(decay)
    except (TypeError, ValueError):
        raise ValueError(mensaje) from None
    if not math.isfinite(valor) or not 0.0 < valor < 1.0:
        raise ValueError(mensaje)
    return valor
