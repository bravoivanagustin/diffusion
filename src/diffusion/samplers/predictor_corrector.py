"""Sampler predictor–corrector: paso de SDE reversa + correcciones de Langevin (Eje 2).

Es el sampler de mayor "techo de calidad" del estudio de ablación (ver
``docs/project/ejes.md``, Song et al., ICLR 2021): alterna un **predictor** —un paso de
Euler–Maruyama de la SDE reversa— con un número configurable de **correctores** de Langevin
al nuevo nivel de ruido ``t + dt``. El predictor avanza la cadena en el tiempo; cada
corrector reequilibra la muestra hacia la marginal ``p_{t+dt}`` mediante dinámica de
Langevin, sin avanzar el tiempo.

Cada corrector da un paso

    ``x ← x + ε·s + √(2ε)·Z``,  con ``Z ~ N(0, I)`` y ``s = ∇_x log p_{t+dt}(x)``,

donde el tamaño de paso ``ε`` se deriva de un target de *signal-to-noise ratio* (SNR):

    ``ε = 2·(snr·‖Z‖ / ‖s‖)²``  (normas L2 medias por batch),

el esquema adaptativo de Song et al. que mantiene el ruido inyectado proporcional al score.
El denominador ``‖s‖`` se acota por debajo para evitar división por cero / ``NaN`` cuando el
score es ~0 (estabilidad numérica, criterio 8.2).

Como :mod:`diffusion.sde`, importa **torch directamente** (opera sobre tensores; torch es
dependencia dura).
"""

from __future__ import annotations

import math

import torch

from .base import ReverseSampler, ScoreFn
from .time_grid import TimeGridFn
from diffusion.sde import ForwardSDE

#: Piso del denominador ``‖s‖`` en la fórmula del paso ``ε`` (espíritu de ``sde._std_eps``):
#: evita división por cero / ``NaN`` cuando el score colapsa a ~0.
_NORM_EPS: float = 1e-8


class PredictorCorrector(ReverseSampler):
    """Sampler predictor–corrector — estocástico.

    Combina, en cada paso, un **predictor** de Euler–Maruyama de la SDE reversa

        ``x ← x + (f - g² s)·dt + g·√|dt|·Z``

    con ``n_corrector`` pasos **correctores** de Langevin al nuevo nivel de ruido
    ``t + dt``

        ``x ← x + ε·s + √(2ε)·Z``,  con ``ε = 2·(snr·‖Z‖ / ‖s‖)²``.

    Es el único sampler con kwargs propios (``n_corrector``, ``snr``); el filtrado por firma
    del factory permite que un llamador genérico pase siempre el mismo conjunto de
    parámetros (criterio 4.4). El ruido se sortea del ``generator`` para que dos corridas con
    la misma semilla coincidan y semillas distintas difieran (criterios 5.2, 5.3).
    """

    name = "pc"

    def __init__(
        self,
        sde: ForwardSDE,
        score_fn: ScoreFn,
        *,
        n_steps: int = 500,
        t_eps: float = 1e-3,
        time_grid: str | TimeGridFn = "uniform",
        n_corrector: int = 1,
        snr: float = 0.16,
    ) -> None:
        """Inicializa el sampler predictor–corrector.

        Args:
            sde: Proceso forward (Eje 1) del que se derivan los coeficientes ``(f, g)`` y el
                prior ``p_T``.
            score_fn: Función pura ``(x, t) -> score`` que aproxima ``∇_x log p_t(x)``.
            n_steps: Número de pasos (intervalos) de integración; ``>= 1``.
            t_eps: Tiempo terminal de la integración, un piso ``> 0`` con ``0 < t_eps < sde.T``.
            time_grid: Distribución de los tiempos de la grilla (``"uniform"`` por default,
                ``"logsnr"``, o un callable propio); ver
                :mod:`diffusion.samplers.time_grid`. Se reenvía a la base.
            n_corrector: Número de correcciones de Langevin por paso (``>= 0``); con ``0`` el
                paso se reduce al predictor de Euler–Maruyama.
            snr: Target de *signal-to-noise ratio* que fija el tamaño de paso ``ε`` del
                corrector; default ``0.16`` (Song et al., ICLR 2021).

        Raises:
            ValueError: Si ``n_steps < 1``, ``t_eps`` cae fuera de ``(0, sde.T)`` o
                ``n_corrector < 0``.
        """
        super().__init__(
            sde, score_fn, n_steps=n_steps, t_eps=t_eps, time_grid=time_grid
        )
        if n_corrector < 0:
            raise ValueError(
                f"n_corrector debe ser >= 0; recibí n_corrector={n_corrector}"
            )
        self.n_corrector = int(n_corrector)
        self.snr = float(snr)

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso predictor–corrector.

        Predictor: un paso de Euler–Maruyama de la SDE reversa de ``t`` a ``t + dt``.
        Corrector: ``n_corrector`` pasos de Langevin al nivel de ruido ``t + dt``.

        Args:
            x: Estado actual de shape ``(B, data_dim)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: tiempo decreciente).
            generator: Generador de torch del que se sortea el ruido (reproducibilidad).

        Returns:
            El nuevo estado de shape ``(B, data_dim)``.
        """
        t = self._expand_t(t)

        # --- Predictor: Euler–Maruyama de la SDE reversa.
        drift = self._reverse_drift(x, t)
        _, g = self.sde.sde(x, t)
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x = x + drift * dt + g * math.sqrt(abs(dt)) * noise

        # --- Corrector: n_corrector pasos de Langevin al nuevo nivel de ruido t + dt.
        t_next = t + dt
        for _ in range(self.n_corrector):
            grad = self.score_fn(x, t_next)
            z = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            grad_norm = grad.reshape(grad.shape[0], -1).norm(dim=1).mean()
            noise_norm = z.reshape(z.shape[0], -1).norm(dim=1).mean()
            # ε adaptativo por target de SNR; piso en ‖s‖ contra div-by-zero / NaN.
            eps = 2.0 * (self.snr * noise_norm / grad_norm.clamp_min(_NORM_EPS)) ** 2
            x = x + eps * grad + torch.sqrt(2.0 * eps) * z

        return x
