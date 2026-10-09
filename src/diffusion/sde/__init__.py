"""Registry de SDEs forward y factory por nombre.

Uso típico::

    from diffusion.sde import make_sde, available_sdes

    sde = make_sde("vp")               # VPSDE con defaults
    x_t, eps = sde.perturb(x0, t)      # par de entrenamiento
    target, weight = sde.score_target(x0, t, eps)
"""

from __future__ import annotations

import inspect

from .base import ForwardSDE
from .variants import SubVPSDE, VESDE, VPSDE

REGISTRY: dict[str, type[ForwardSDE]] = {
    cls.name: cls for cls in (VPSDE, VESDE, SubVPSDE)
}


def available_sdes() -> list[str]:
    """Nombres de las SDEs disponibles, ordenados."""
    return sorted(REGISTRY)


def make_sde(name: str, **kwargs) -> ForwardSDE:
    """Crea la SDE ``name``.

    Los ``kwargs`` que no aplican a la variante elegida se descartan (se filtran según la
    firma del constructor), así un caller genérico puede pasar siempre el mismo conjunto
    de parámetros. A diferencia de ``data_generation.make_distribution``, no toma ``dim``:
    el ``data_dim`` va como kwarg del constructor (2 por defecto en todas las variantes).
    """
    try:
        cls = REGISTRY[name]
    except KeyError:
        opts = ", ".join(available_sdes())
        raise ValueError(f"SDE desconocida '{name}'. Opciones: {opts}") from None
    params = inspect.signature(cls).parameters
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    clean = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in params}
    return cls(**clean)


__all__ = [
    "ForwardSDE",
    "REGISTRY",
    "available_sdes",
    "make_sde",
    "VPSDE",
    "VESDE",
    "SubVPSDE",
]
