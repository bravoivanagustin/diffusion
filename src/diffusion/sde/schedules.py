"""Funciones de schedule de las SDEs (matemática pura sobre tensores de tiempo).

Aisladas acá porque son la superficie de mayor valor para tests numéricos (analítico vs
diferencias finitas) y porque la integral de ``beta`` se reusa en VP y sub-VP.

- VP / sub-VP usan un schedule **lineal** de ``beta(t)`` y su integral cerrada.
- VE usa un schedule **geométrico** de ``sigma(t)``.

Todas operan sobre ``t`` ya normalizado a shape ``(B, 1)`` (ver
:meth:`diffusion.sde.base.ForwardSDE._expand_t`) y devuelven tensores de la misma shape.
"""

from __future__ import annotations

import torch


def linear_beta(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """Schedule lineal ``beta(t) = beta_min + t (beta_max - beta_min)``."""
    return beta_min + t * (beta_max - beta_min)


def linear_beta_integral(
    t: torch.Tensor, beta_min: float, beta_max: float
) -> torch.Tensor:
    """Integral cerrada ``∫_0^t beta(s) ds = beta_min t + ½ (beta_max - beta_min) t^2``."""
    return beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2


def geometric_sigma(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Schedule geométrico ``sigma(t) = sigma_min (sigma_max / sigma_min)^t``."""
    return sigma_min * (sigma_max / sigma_min) ** t
