"""Proceso reverso (Eje 2): samplers que integran la SDE/ODE inversa.

Expone el ABC :class:`~diffusion.samplers.base.ReverseSampler`, los cuatro samplers
concretos (Euler–Maruyama, PF-ODE, Heun, predictor–corrector) y el registry/factory por
nombre (``make_sampler``/``available_samplers``), espejando :mod:`diffusion.sde`.

Uso típico::

    from diffusion.samplers import make_sampler, available_samplers
    from diffusion.sde import make_sde

    sde = make_sde("vp")
    sampler = make_sampler("euler", sde, score_fn)   # un sampler = una celda del Eje 2
    x0 = sampler.sample(64)

La orquestación checkpoint-driven (``generate_from_checkpoint``) llega en una task posterior.
"""

from __future__ import annotations

import inspect

from .base import ReverseSampler, ScoreFn
from .euler_maruyama import EulerMaruyama
from .heun import HeunODE
from .pf_ode import ProbabilityFlowODE
from .predictor_corrector import PredictorCorrector
from .time_grid import (
    TimeGrid,
    TimeGridFn,
    available_time_grids,
    make_time_grid,
)
from diffusion.sde import ForwardSDE

REGISTRY: dict[str, type[ReverseSampler]] = {
    cls.name: cls
    for cls in (EulerMaruyama, ProbabilityFlowODE, HeunODE, PredictorCorrector)
}


def available_samplers() -> list[str]:
    """Nombres de los samplers disponibles, ordenados."""
    return sorted(REGISTRY)


def make_sampler(
    name: str, sde: ForwardSDE, score_fn: ScoreFn, **kwargs
) -> ReverseSampler:
    """Crea el sampler ``name`` configurado con ``sde`` y ``score_fn``.

    Los ``kwargs`` que no aplican al sampler elegido se descartan (se filtran según la firma
    del constructor), así un caller genérico puede pasar siempre el mismo conjunto de
    parámetros (criterio 4.4): p. ej. ``snr``/``n_corrector`` son exclusivos de
    :class:`PredictorCorrector` y los demás samplers los ignoran sin fallar.

    Args:
        name: Clave del registry (``"euler"``, ``"pf_ode"``, ``"heun"``, ``"pc"``).
        sde: Proceso forward (Eje 1) del que el sampler deriva ``(f, g)`` y el prior.
        score_fn: Función pura ``(x, t) -> score`` que aproxima ``∇_x log p_t(x)``.
        **kwargs: Parámetros opcionales del sampler (p. ej. ``n_steps``, ``t_eps``,
            ``time_grid``, y para ``pc`` también ``n_corrector``/``snr``); los no aplicables
            se descartan.

    Returns:
        La instancia del sampler correspondiente.

    Raises:
        ValueError: Si ``name`` no está en el registry; el mensaje enumera las opciones.
    """
    try:
        cls = REGISTRY[name]
    except KeyError:
        opts = ", ".join(available_samplers())
        raise ValueError(
            f"Sampler desconocido '{name}'. Opciones: {opts}"
        ) from None
    params = inspect.signature(cls).parameters
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    clean = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in params}
    return cls(sde, score_fn, **clean)


# Importado al final: ``generate`` depende de ``make_sampler`` (definido arriba) y de los
# módulos upstream (``training``/``sde``), que no importan ``samplers`` (sin ciclo).
from .generate import generate_from_checkpoint

__all__ = [
    "ReverseSampler",
    "ScoreFn",
    "REGISTRY",
    "available_samplers",
    "make_sampler",
    "generate_from_checkpoint",
    "EulerMaruyama",
    "ProbabilityFlowODE",
    "HeunODE",
    "PredictorCorrector",
    "TimeGrid",
    "TimeGridFn",
    "available_time_grids",
    "make_time_grid",
]
