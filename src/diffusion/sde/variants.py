"""SDEs forward de la familia escalar-gaussiana: VP, VE y sub-VP.

Las tres comparten el kernel de perturbación ``p_t(x_t | x_0) = N(alpha_t x_0, sigma_t^2 I)``
con ``alpha_t``/``sigma_t`` escalares por muestra, así que :meth:`perturb` y
:meth:`score_target` se heredan tal cual de :class:`~diffusion.sde.base.ForwardSDE`. Cada
clase solo aporta :meth:`sde`, :meth:`marginal_prob` y :meth:`prior_sampling`.

Marco unificado de Song et al. (ICLR 2021).
"""

from __future__ import annotations

import math

import torch

from .base import ForwardSDE
from .schedules import geometric_sigma, linear_beta, linear_beta_integral


class VPSDE(ForwardSDE):
    """Variance Preserving SDE — límite continuo de DDPM (Ho et al., 2020).

    ``dx = -½ beta(t) x dt + sqrt(beta(t)) dW``. El drift encoge la señal mientras la
    difusión agrega ruido, manteniendo la varianza ~constante. Prior ``p_T ≈ N(0, I)``.
    """

    name = "vp"

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        data_dim: int | tuple[int, ...] = 2,
        T: float = 1.0,
    ) -> None:
        """Inicializa la VP-SDE.

        Args:
            beta_min: ``beta(0)`` del schedule lineal.
            beta_max: ``beta(T)`` del schedule lineal.
            data_dim: Geometría del dato: entero (dato plano) o tupla ``(C, H, W)`` (forma
                de evento multidimensional). Anda en cualquier rango.
            T: Horizonte temporal.
        """
        super().__init__(data_dim=data_dim, T=T)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tt = self._expand_t(t, x)
        beta = linear_beta(tt, self.beta_min, self.beta_max)
        drift = -0.5 * beta * x
        diffusion = torch.sqrt(beta)
        return drift, diffusion

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tt = self._expand_t(t, x0)
        b_int = linear_beta_integral(tt, self.beta_min, self.beta_max)
        alpha = torch.exp(-0.5 * b_int)
        mean = alpha * x0
        std = torch.sqrt(1.0 - torch.exp(-b_int))
        return mean, std

    def prior_sampling(
        self,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)


class VESDE(ForwardSDE):
    """Variance Exploding SDE — límite continuo de NCSN (Song & Ermon, 2019).

    ``dx = sqrt(d[sigma^2(t)]/dt) dW`` (drift nulo). El dato se deja en su lugar y se le
    apila ruido con un schedule geométrico ``sigma(t)`` creciente. Prior
    ``p_T ≈ N(0, sigma_max^2 I)``.

    ``sigma_max`` por defecto es ``5.0`` (≈ escala del toy data 2D estandarizado), **no**
    el ``50`` de imágenes de Song; es argumento del constructor.
    """

    name = "ve"

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        data_dim: int | tuple[int, ...] = 2,
        T: float = 1.0,
    ) -> None:
        """Inicializa la VE-SDE.

        Args:
            sigma_min: ``sigma(0)`` del schedule geométrico.
            sigma_max: ``sigma(T)`` del schedule geométrico (escala del prior).
            data_dim: Geometría del dato: entero (dato plano) o tupla ``(C, H, W)`` (forma
                de evento multidimensional). Anda en cualquier rango.
            T: Horizonte temporal.
        """
        super().__init__(data_dim=data_dim, T=T)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        # Constante del coeficiente de difusión: g(t) = sigma(t) sqrt(2 ln(sigma_max/sigma_min)).
        self._log_ratio = math.log(self.sigma_max / self.sigma_min)

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tt = self._expand_t(t, x)
        sigma = geometric_sigma(tt, self.sigma_min, self.sigma_max)
        drift = torch.zeros_like(x)
        diffusion = sigma * math.sqrt(2.0 * self._log_ratio)
        return drift, diffusion

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tt = self._expand_t(t, x0)
        mean = x0
        std = geometric_sigma(tt, self.sigma_min, self.sigma_max)
        return mean, std

    def prior_sampling(
        self,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        z = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return z * self.sigma_max


class SubVPSDE(ForwardSDE):
    """sub-VP SDE — variante de varianza acotada por debajo de VP (Song et al., 2021).

    Mismo drift que VP (``-½ beta(t) x``) pero difusión estrictamente menor:
    ``dx = -½ beta(t) x dt + sqrt(beta(t)(1 - e^{-2∫beta})) dW``. La media coincide con la
    de VP (mismo ``alpha_t``); el desvío es ``1 - e^{-∫beta}`` (sin raíz), cuya varianza
    queda por debajo de la de VP. Prior ``p_T ≈ N(0, I)``.
    """

    name = "sub_vp"

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        data_dim: int | tuple[int, ...] = 2,
        T: float = 1.0,
    ) -> None:
        """Inicializa la sub-VP SDE.

        Args:
            beta_min: ``beta(0)`` del schedule lineal.
            beta_max: ``beta(T)`` del schedule lineal.
            data_dim: Geometría del dato: entero (dato plano) o tupla ``(C, H, W)`` (forma
                de evento multidimensional). Anda en cualquier rango.
            T: Horizonte temporal.
        """
        super().__init__(data_dim=data_dim, T=T)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tt = self._expand_t(t, x)
        beta = linear_beta(tt, self.beta_min, self.beta_max)
        b_int = linear_beta_integral(tt, self.beta_min, self.beta_max)
        drift = -0.5 * beta * x
        diffusion = torch.sqrt(beta * (1.0 - torch.exp(-2.0 * b_int)))
        return drift, diffusion

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tt = self._expand_t(t, x0)
        b_int = linear_beta_integral(tt, self.beta_min, self.beta_max)
        alpha = torch.exp(-0.5 * b_int)
        mean = alpha * x0
        std = 1.0 - torch.exp(-b_int)
        return mean, std

    def prior_sampling(
        self,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
