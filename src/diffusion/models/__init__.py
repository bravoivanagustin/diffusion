"""Redes de score: la variable de control del estudio de ablación.

Agrupa las redes que aproximan el score :math:`\\nabla_x \\log p_t(x)` y sus piezas, con
misma arquitectura en todas las celdas SDE × sampler:

- :mod:`diffusion.models.layers` — piezas compartidas entre redes (embedding sinusoidal de
  tiempo, activaciones).
- :mod:`diffusion.models.mlp` — :class:`ScoreMLP`, la red para datos de juguete 2D (Fase 1).
- :mod:`diffusion.models.unet` — :class:`ScoreUNet`, la red convolucional para imágenes (Fase 2).
- :mod:`diffusion.models.base` — el Protocol :class:`ScoreModel`, el contrato
  ``(x, t) -> score`` que toda red satisface estructuralmente.
- :mod:`diffusion.models.parametrization` — :class:`EpsilonScoreWrapper`, la
  parametrización ε del score (``score = -inner/σ_t`` con σ inyectada como callable).

Además expone un registry/factory por nombre (:func:`make_model`, :func:`available_models`,
:data:`REGISTRY`), espejo de ``make_sde`` y ``make_distribution``, para construir la red desde
una receta ``(name, kwargs)`` (config-driven y reconstrucción desde checkpoint).

Uso típico::

    from diffusion.models import ScoreMLP, make_model

    net = ScoreMLP(data_dim=2)
    score = net(x, t)

    net2 = make_model("mlp", data_dim=2)   # misma red vía registry
"""

from __future__ import annotations

import inspect

from .base import ScoreModel
from .layers import SinusoidalEmbedding
from .mlp import ResidualBlock, ScoreMLP
from .parametrization import EpsilonScoreWrapper
from .unet import ScoreUNet

REGISTRY: dict[str, type] = {
    "mlp": ScoreMLP,
    "unet": ScoreUNet,
}


def available_models() -> list[str]:
    """Nombres de las redes disponibles, ordenados."""
    return sorted(REGISTRY)


def make_model(name: str, **kwargs) -> ScoreModel:
    """Crea la red de score ``name`` (satisface el contrato :class:`ScoreModel`).

    Los ``kwargs`` que no aplican a la red elegida se descartan (se filtran según la firma
    del constructor), así un caller genérico —el config-driven o la reconstrucción desde
    checkpoint— puede pasar siempre el mismo conjunto de parámetros. A diferencia de
    ``data_generation.make_distribution``, no toma un posicional ``dim``: la dimensión del
    dato va como kwarg del constructor (``data_dim`` en :class:`ScoreMLP`, ``in_channels`` /
    ``image_size`` en :class:`ScoreUNet`).
    """
    try:
        cls = REGISTRY[name]
    except KeyError:
        opts = ", ".join(available_models())
        raise ValueError(f"Red desconocida '{name}'. Opciones: {opts}") from None
    params = inspect.signature(cls).parameters
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    clean = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in params}
    return cls(**clean)


__all__ = [
    "ScoreModel",
    "SinusoidalEmbedding",
    "ResidualBlock",
    "ScoreMLP",
    "ScoreUNet",
    "EpsilonScoreWrapper",
    "REGISTRY",
    "available_models",
    "make_model",
]
