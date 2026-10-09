"""Registry de distribuciones de puntos y factory por nombre.

Uso típico::

    from diffusion.data_generation import make_distribution, available_shapes

    dist = make_distribution("two_moons", dim=2, seed=0)
    x = dist.sample(2000)          # np.ndarray (2000, 2) float32
"""

from __future__ import annotations

import inspect

from .base import PointDistribution
from .images import count_images, finite_batches, infinite_batches, report_small_images
from .iterators import infinite_bare
from .shapes import (
    ExactGaussianMixture,
    Gaussian,
    GaussianMixture,
    Spiral,
    SwissRoll,
    TwoMoons,
)

#: Formas construibles por nombre. ``ExactGaussianMixture`` **no** está acá a
#: propósito: sus parámetros (pesos, medias, covarianzas) son obligatorios y son
#: matrices, así que no viajan por CLI ni por YAML, y :func:`make_distribution`
#: filtra los kwargs que no matchean la firma en lugar de completarlos. Se
#: exporta para importarla directamente, que es su único camino de uso.
REGISTRY: dict[str, type[PointDistribution]] = {
    cls.name: cls
    for cls in (Gaussian, GaussianMixture, TwoMoons, Spiral, SwissRoll)
}


def available_shapes() -> list[str]:
    """Nombres de las formas disponibles, ordenados."""
    return sorted(REGISTRY)


def make_distribution(name: str, dim: int, **kwargs) -> PointDistribution:
    """Crea la distribución ``name`` en dimensión ``dim``.

    Los ``kwargs`` que no aplican a la forma elegida se descartan (se filtran
    según la firma del constructor), así un caller genérico como el CLI puede
    pasar siempre el mismo conjunto de parámetros.
    """
    try:
        cls = REGISTRY[name]
    except KeyError:
        opts = ", ".join(available_shapes())
        raise ValueError(f"Forma desconocida '{name}'. Opciones: {opts}") from None
    params = inspect.signature(cls).parameters
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    clean = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in params}
    return cls(dim, **clean)


__all__ = [
    "PointDistribution",
    "REGISTRY",
    "available_shapes",
    "make_distribution",
    "infinite_bare",
    "infinite_batches",
    "finite_batches",
    "count_images",
    "report_small_images",
    "Gaussian",
    "GaussianMixture",
    "ExactGaussianMixture",
    "TwoMoons",
    "Spiral",
    "SwissRoll",
]
