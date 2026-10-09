"""Grillas temporales de integración del proceso reverso (Eje 2).

El driver de :class:`~diffusion.samplers.base.ReverseSampler` integra de ``T`` a ``t_eps``
sobre una grilla de ``n_steps + 1`` tiempos. **Dónde** se colocan esos tiempos es una
elección numérica libre —no cambia el proceso, solo su discretización—, así que vive acá
como parámetro configurable y **no** obliga a reentrenar: es parte del Eje 2, igual que la
elección del sampler (ver ``docs/project/ejes.md``).

Variantes registradas (patrón registry/factory del repo, como ``make_sde``/``make_sampler``
y :mod:`diffusion.training.time_sampling`):

- ``uniform`` — el default retrocompatible: ``torch.linspace(T, t_eps, n_steps + 1)``,
  espaciado constante en ``t``. Byte-idéntico a la grilla previa al cambio.
- ``logsnr`` — espaciado constante en el **log-SNR** ``λ(t) = log(α_t² / σ_t²)``, la escala
  en la que el kernel de perturbación cambia de forma uniforme. Concentra pasos donde
  ``λ`` se mueve rápido (``t`` chico en VP/sub-VP), que es donde la discretización uniforme
  desperdicia evaluaciones. Es la grilla "uniforme en log-SNR" usual en la literatura
  (Kingma et al. 2021; el ``rho``-schedule de EDM es una variante emparentada).

Además del nombre se acepta un **callable propio** con la firma
``fn(n_steps, t_min, t_max) -> tensor de (n_steps + 1,)`` decreciente de ``t_max`` a
``t_min`` (se valida; ver :class:`CallableTimeGrid`), para probar una grilla ad hoc en un
notebook sin registrarla.

Uso típico::

    from diffusion.samplers import make_sampler

    s = make_sampler("heun", sde, score_fn, n_steps=100, time_grid="logsnr")
    x0 = s.sample(64)

Nota sobre λ y las tres SDEs del Eje 1: ``logsnr`` deriva ``λ`` de
:meth:`~diffusion.sde.base.ForwardSDE.marginal_prob` (que las tres exponen) e invierte
``λ(t)`` por **bisección**, en vez de usar una fórmula cerrada por variante. Así la misma
implementación anda en VP, VE y sub-VP —y en cualquier SDE escalar-gaussiana futura— sin
tocar el módulo :mod:`diffusion.sde`: VP tiene inversa cerrada, VE también (``λ`` es lineal
en ``t``), pero sub-VP **no**. Consecuencia útil: en **VE** el ``σ_t`` geométrico hace que
``λ`` sea lineal en ``t``, así que ``logsnr`` coincide con ``uniform`` salvo error numérico
(la grilla nueva solo cambia algo en VP/sub-VP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

from diffusion.sde import ForwardSDE

#: Contrato del callable de grilla propia: ``(n_steps, t_min, t_max) -> (n_steps + 1,)``.
TimeGridFn = Callable[[int, float, float], torch.Tensor]

#: Iteraciones de bisección al invertir ``λ(t)``. Con el rango ``[t_eps, T] ⊆ (0, 1]``, 60
#: bisecciones reducen el intervalo por debajo de la resolución de ``float32``: la inversión
#: es exacta a precisión de máquina. Son evaluaciones de ``marginal_prob`` sobre un tensor de
#: ``(n_steps + 1,)`` —aritmética escalar, sin la red—, así que el costo es despreciable.
_BISECTION_ITERS = 60

#: Piso de los logaritmos al formar ``λ = 2(log α_t − log σ_t)``: evita ``log(0)`` en los
#: extremos (p. ej. ``σ_t → 0`` cuando ``t → 0``) sin desplazar los valores útiles.
_LOG_EPS = 1e-30


class TimeGrid(ABC):
    """Base abstracta de las grillas temporales de integración.

    Contrato: ``grid(n_steps)`` devuelve un tensor de shape ``(n_steps + 1,)`` en
    ``float32``, **estrictamente decreciente**, con extremos exactos ``T`` (primero) y
    ``t_eps`` (último) — el driver arranca del prior en ``T`` y termina en ``t_eps``.

    Args:
        sde: Proceso forward del que salen ``T`` y, si la variante lo necesita, los
            coeficientes ``(α_t, σ_t)`` del kernel.
        t_eps: Tiempo terminal; debe cumplir ``0 < t_eps < sde.T``.

    Raises:
        ValueError: Si ``t_eps`` queda fuera del rango abierto ``(0, sde.T)`` (fail-fast en
            construcción, antes de integrar).
    """

    #: Nombre con el que la variante se registra en la factory (lo define cada subclase).
    name: str

    def __init__(self, sde: ForwardSDE, t_eps: float) -> None:
        if not 0.0 < t_eps < sde.T:
            raise ValueError(
                f"t_eps debe cumplir 0 < t_eps < sde.T={sde.T}; recibí t_eps={t_eps}"
            )
        self.sde = sde
        self.T = float(sde.T)
        self.t_eps = float(t_eps)

    @abstractmethod
    def grid(self, n_steps: int) -> torch.Tensor:
        """Construye la grilla de ``n_steps + 1`` tiempos de ``T`` a ``t_eps``.

        Args:
            n_steps: Número de pasos (intervalos) de integración; ``>= 1``.

        Returns:
            Tensor ``float32`` de shape ``(n_steps + 1,)``, decreciente, con extremos ``T``
            y ``t_eps``.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"{type(self).__name__}(T={self.T}, t_eps={self.t_eps})"


class UniformTimeGrid(TimeGrid):
    """Grilla uniforme en ``t`` — el default retrocompatible.

    Es exactamente ``torch.linspace(T, t_eps, n_steps + 1)``, la fórmula que tenía el driver
    antes de que la grilla fuese configurable: byte-idéntica, así que ninguna corrida previa
    cambia de resultado.
    """

    name = "uniform"

    def grid(self, n_steps: int) -> torch.Tensor:
        return torch.linspace(self.T, self.t_eps, n_steps + 1, dtype=torch.float32)


class LogSNRTimeGrid(TimeGrid):
    """Grilla uniforme en el log-SNR ``λ(t) = log(α_t² / σ_t²)``.

    En vez de repartir los pasos parejo en ``t``, los reparte parejo en ``λ``, que es la
    escala natural del kernel ``x_t = α_t x_0 + σ_t ε``: cada paso cambia la relación
    señal-ruido en la misma cantidad. Como ``λ`` decrece rápido cerca de ``t → 0`` (en
    VP/sub-VP), la grilla acumula pasos ahí —donde la uniforme desperdicia evaluaciones— y
    los ralea en ``t`` grande, donde el estado es casi ruido puro.

    Implementación **agnóstica a la variante**: ``λ`` se arma con
    :meth:`~diffusion.sde.base.ForwardSDE.marginal_prob` y se invierte por bisección
    (:data:`_BISECTION_ITERS` iteraciones, vectorizadas sobre los ``n_steps + 1`` targets),
    aprovechando que ``λ`` es estrictamente decreciente en ``t`` para las tres SDEs del Eje
    1. No hay fórmulas cerradas por SDE: VP y VE las tienen, sub-VP no.

    Los extremos se fijan **exactos** (``T`` y ``t_eps``) en vez de dejar el residuo de la
    bisección, para que el prior y el denoising final de Tweedie caigan en los tiempos que
    el driver promete.
    """

    name = "logsnr"

    def _log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Evalúa ``λ(t) = 2(log α_t − log σ_t)`` para un tensor de tiempos ``(k,)``.

        Los coeficientes salen de ``marginal_prob`` evaluada en ``x_0 = 1`` (así ``mean``
        **es** ``α_t``), el mismo truco que usa el denoising de Tweedie del driver.

        Args:
            t: Tiempos de shape ``(k,)``.

        Returns:
            ``λ`` de shape ``(k,)`` en ``float32``, estrictamente decreciente en ``t``.
        """
        tt = t.reshape(-1, 1)
        alpha, sigma = self.sde.marginal_prob(torch.ones_like(tt), tt)
        log_snr = 2.0 * (
            torch.log(alpha.clamp_min(_LOG_EPS)) - torch.log(sigma.clamp_min(_LOG_EPS))
        )
        return log_snr.reshape(-1)

    def _invert_log_snr(self, targets: torch.Tensor) -> torch.Tensor:
        """Resuelve ``λ(t) = target`` por bisección, vectorizado sobre ``targets``.

        Args:
            targets: Valores de ``λ`` buscados, shape ``(k,)``; deben caer dentro de
                ``[λ(T), λ(t_eps)]``.

        Returns:
            Los tiempos ``t`` de shape ``(k,)`` en ``float32``.
        """
        lo = torch.full_like(targets, self.t_eps)  # λ(lo) = máximo
        hi = torch.full_like(targets, self.T)      # λ(hi) = mínimo
        for _ in range(_BISECTION_ITERS):
            mid = 0.5 * (lo + hi)
            # λ decrece en t: si λ(mid) sigue por encima del target, la raíz está a la derecha.
            go_right = self._log_snr(mid) > targets
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)
        return 0.5 * (lo + hi)

    def grid(self, n_steps: int) -> torch.Tensor:
        bounds = self._log_snr(
            torch.tensor([self.T, self.t_eps], dtype=torch.float32)
        )
        # De λ(T) (mínimo, ruido) a λ(t_eps) (máximo, señal): espaciado constante en λ.
        targets = torch.linspace(
            float(bounds[0]), float(bounds[1]), n_steps + 1, dtype=torch.float32
        )
        t = self._invert_log_snr(targets)
        t[0] = self.T
        t[-1] = self.t_eps
        return t


class CallableTimeGrid(TimeGrid):
    """Adaptador de un callable propio a :class:`TimeGrid`, con validación.

    Envuelve una función ``fn(n_steps, t_min, t_max)`` —la firma que se usa a mano para
    definir una grilla ad hoc— y valida su salida antes de dejarla entrar al driver, porque
    una grilla mal formada (largo distinto, orden invertido, ``NaN``) rompería la
    integración de formas difíciles de diagnosticar. Se acepta cualquier cosa convertible a
    tensor (p. ej. un ``ndarray`` de numpy).

    Args:
        fn: Callable ``(n_steps, t_min, t_max) -> (n_steps + 1,)``, decreciente de ``t_max``
            a ``t_min``. Se lo llama como ``fn(n_steps, t_eps, T)``.
        sde: Proceso forward (aporta ``T``).
        t_eps: Tiempo terminal; ``0 < t_eps < sde.T``.
    """

    name = "callable"

    def __init__(self, fn: TimeGridFn, sde: ForwardSDE, t_eps: float) -> None:
        super().__init__(sde, t_eps)
        if not callable(fn):
            raise ValueError(
                "time_grid debe ser el nombre de una grilla registrada o un callable "
                f"(n_steps, t_min, t_max) -> tensor; recibí {type(fn).__name__}"
            )
        self.fn = fn

    def grid(self, n_steps: int) -> torch.Tensor:
        raw = self.fn(n_steps, self.t_eps, self.T)
        try:
            t = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
        except (TypeError, RuntimeError) as exc:  # pragma: no cover - defensivo
            raise ValueError(
                f"El callable de time_grid devolvió algo no convertible a tensor: {raw!r}"
            ) from exc
        _validate_grid(t, n_steps, self.T, self.t_eps)
        return t

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        name = getattr(self.fn, "__name__", repr(self.fn))
        return f"CallableTimeGrid({name}, T={self.T}, t_eps={self.t_eps})"


def _validate_grid(
    t: torch.Tensor, n_steps: int, T: float, t_eps: float
) -> None:
    """Valida que ``t`` cumpla el contrato de grilla del driver.

    Args:
        t: Grilla candidata, ya como tensor ``float32`` 1-D.
        n_steps: Número de pasos esperado (la grilla debe tener ``n_steps + 1`` puntos).
        T: Extremo inicial esperado.
        t_eps: Extremo final esperado.

    Raises:
        ValueError: Si el largo no es ``n_steps + 1``, si hay valores no finitos, si la
            grilla no es estrictamente decreciente, o si los extremos no coinciden con
            ``(T, t_eps)`` dentro de tolerancia.
    """
    expected = n_steps + 1
    if t.numel() != expected:
        raise ValueError(
            f"La grilla temporal debe tener n_steps + 1 = {expected} puntos; "
            f"recibí {t.numel()}"
        )
    if not bool(torch.isfinite(t).all()):
        raise ValueError("La grilla temporal contiene valores no finitos (NaN/inf).")
    if expected > 1 and not bool((t[1:] < t[:-1]).all()):
        raise ValueError(
            "La grilla temporal debe ser estrictamente decreciente (se integra de T a "
            "t_eps); recibí una grilla no monótona o con puntos repetidos."
        )
    tol = 1e-5 + 1e-3 * max(abs(T), abs(t_eps))
    if abs(float(t[0]) - T) > tol or abs(float(t[-1]) - t_eps) > tol:
        raise ValueError(
            f"La grilla temporal debe ir de T={T} a t_eps={t_eps} (el prior se sortea en T "
            f"y el denoising final ocurre en t_eps); recibí extremos "
            f"({float(t[0])}, {float(t[-1])})."
        )


REGISTRY: dict[str, type[TimeGrid]] = {
    cls.name: cls for cls in (UniformTimeGrid, LogSNRTimeGrid)
}


def available_time_grids() -> list[str]:
    """Nombres de las grillas temporales registradas, ordenados."""
    return sorted(REGISTRY)


def make_time_grid(
    spec: str | TimeGridFn, sde: ForwardSDE, t_eps: float
) -> TimeGrid:
    """Crea la grilla temporal ``spec`` para ``sde``.

    Args:
        spec: Nombre registrado (``"uniform"`` o ``"logsnr"``) **o** un callable
            ``(n_steps, t_min, t_max) -> (n_steps + 1,)`` decreciente (ver
            :class:`CallableTimeGrid`).
        sde: Proceso forward del que salen ``T`` y los coeficientes del kernel.
        t_eps: Tiempo terminal; debe cumplir ``0 < t_eps < sde.T``.

    Returns:
        La :class:`TimeGrid` construida.

    Raises:
        ValueError: Si ``spec`` es un string no registrado (el mensaje lista las opciones),
            si no es ni string ni callable, o si ``t_eps`` cae fuera de ``(0, sde.T)``.
    """
    if isinstance(spec, str):
        try:
            cls = REGISTRY[spec]
        except KeyError:
            opts = ", ".join(available_time_grids())
            raise ValueError(
                f"Grilla temporal desconocida '{spec}'. Opciones: {opts} "
                "(o pasá un callable (n_steps, t_min, t_max) -> tensor)."
            ) from None
        return cls(sde, t_eps)
    return CallableTimeGrid(spec, sde, t_eps)
