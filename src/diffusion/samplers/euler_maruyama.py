"""Sampler Euler–Maruyama: discretización estocástica de la SDE reversa (Eje 2).

Es el baseline "puro estocástico" del estudio de ablación (ver ``docs/project/ejes.md``):
discretiza directamente la ecuación reversa de Anderson (1982)

    ``dx = [f(x,t) - g(t)^2 ∇_x log p_t(x)] dt + g(t) dW̄``

con el esquema de Euler–Maruyama. Cada paso usa el drift reverso ``f - g^2 s`` (provisto
por :meth:`~diffusion.samplers.base.ReverseSampler._reverse_drift`) e **inyecta ruido**
gaussiano escalado por la difusión, tomado del ``generator`` para que el sampleo sea
reproducible por semilla.

Como :mod:`diffusion.sde`, importa **torch directamente** (opera sobre tensores; torch es
dependencia dura).
"""

from __future__ import annotations

import math

import torch

from .base import ReverseSampler


class EulerMaruyama(ReverseSampler):
    """Sampler Euler–Maruyama de la SDE reversa — estocástico.

    Discretiza ``dx = [f - g^2 s] dt + g dW̄`` con un paso de Euler–Maruyama. Con la grilla
    en tiempo decreciente (``dt < 0``), el incremento de ruido se escala por ``√|dt|``:

        ``x ← x + (f - g^2 s)·dt + g·√|dt|·Z``,  con ``Z ~ N(0, I)``.

    El ruido ``Z`` se sortea del ``generator``, de modo que dos corridas con la misma semilla
    coinciden y semillas distintas difieren (Eje 2: cambiar de sampler no reentrena la red).
    """

    name = "euler"

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso de Euler–Maruyama de la SDE reversa.

        Args:
            x: Estado actual de shape ``(B, data_dim)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: tiempo decreciente).
            generator: Generador de torch del que se sortea el ruido (reproducibilidad).

        Returns:
            El nuevo estado de shape ``(B, data_dim)``.
        """
        drift = self._reverse_drift(x, t)
        _, g = self.sde.sde(x, self._expand_t(t))
        noise = torch.randn(
            x.shape, generator=generator, device=x.device, dtype=x.dtype
        )
        return x + drift * dt + g * math.sqrt(abs(dt)) * noise
