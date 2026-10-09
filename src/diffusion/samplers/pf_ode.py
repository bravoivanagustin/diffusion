"""Sampler Probability-Flow ODE (Euler): integración determinística del reverso (Eje 2).

Es el sampler **determinístico** del estudio de ablación (ver ``docs/project/ejes.md``):
en lugar de discretizar la SDE reversa estocástica, integra la **ODE de flujo de
probabilidad** asociada (Song et al., ICLR 2021), que comparte las **mismas marginales**
``p_t(x)`` que la SDE pero **sin término de ruido**:

    ``dx = [f(x,t) - ½ g(t)^2 ∇_x log p_t(x)] dt``

Cada paso es un Euler explícito sobre el drift de PF-ODE ``f - ½ g^2 s`` (provisto por
:meth:`~diffusion.samplers.base.ReverseSampler._pfode_drift`). Al no inyectar ruido, dos
corridas con el mismo estado inicial coinciden exactamente y el ``generator`` se ignora
(solo se acepta por compatibilidad de firma).

Como :mod:`diffusion.sde`, importa **torch directamente** (opera sobre tensores; torch es
dependencia dura).
"""

from __future__ import annotations

import torch

from .base import ReverseSampler


class ProbabilityFlowODE(ReverseSampler):
    """Sampler Probability-Flow ODE de paso Euler — determinístico.

    Discretiza la ODE de flujo de probabilidad ``dx = [f - ½ g^2 s] dt`` con un paso de
    Euler explícito. Con la grilla en tiempo decreciente (``dt < 0``):

        ``x ← x + (f - ½ g^2 s)·dt``.

    No hay término de ruido: el sampler es **determinístico**. Dos corridas con el mismo
    estado inicial producen resultados idénticos, independientemente del ``generator``
    (que se acepta por compatibilidad con :meth:`ReverseSampler.step` pero no se usa). Es
    la contraparte determinística de :class:`~diffusion.samplers.euler_maruyama.EulerMaruyama`
    con las mismas marginales (Eje 2: cambiar de sampler no reentrena la red).
    """

    name = "pf_ode"

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso de Euler de la ODE de flujo de probabilidad.

        Args:
            x: Estado actual de shape ``(B, data_dim)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: tiempo decreciente).
            generator: Ignorado — el sampler es determinístico. Se acepta solo por
                compatibilidad con la firma de :meth:`ReverseSampler.step`.

        Returns:
            El nuevo estado de shape ``(B, data_dim)``.
        """
        return x + self._pfode_drift(x, t) * dt
