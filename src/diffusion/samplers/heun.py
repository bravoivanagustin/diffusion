"""Sampler Heun: integración determinística de 2º orden del reverso (Eje 2).

Es el sampler determinístico **de segundo orden** del estudio de ablación (ver
``docs/project/ejes.md``): integra la misma ODE de flujo de probabilidad que
:class:`~diffusion.samplers.pf_ode.ProbabilityFlowODE`,

    ``dx = [f(x,t) - ½ g(t)^2 ∇_x log p_t(x)] dt``,

pero con el esquema de Heun (regla del trapecio explícita), el sampler por defecto de
EDM (Karras et al., NeurIPS 2022): mejor precisión por NFE que el Euler de PF-ODE. Cada
paso hace **dos** evaluaciones del drift de PF-ODE (``f - ½ g^2 s``, provisto por
:meth:`~diffusion.samplers.base.ReverseSampler._pfode_drift`) —y por tanto dos del
score—: una predicción de Euler y una corrección que promedia el drift en ambos extremos.

Al no inyectar ruido, dos corridas con el mismo estado inicial coinciden exactamente y el
``generator`` se ignora (solo se acepta por compatibilidad de firma).

Como :mod:`diffusion.sde`, importa **torch directamente** (opera sobre tensores; torch es
dependencia dura).
"""

from __future__ import annotations

import torch

from .base import ReverseSampler


class HeunODE(ReverseSampler):
    """Sampler Heun (ODE de flujo de probabilidad, 2º orden) — determinístico.

    Discretiza ``dx = [f - ½ g^2 s] dt`` con el método de Heun (predictor de Euler +
    corrección por regla del trapecio). Con ``d = _pfode_drift`` y la grilla en tiempo
    decreciente (``dt < 0``):

        - predictor:  ``x̂ = x + d(x, t)·dt``
        - corrector:  ``x ← x + ½·[d(x, t) + d(x̂, t + dt)]·dt``

    Cada paso cuesta **dos** evaluaciones del score (una por cada llamada a ``d``), el
    costo observable del 2º orden. No hay término de ruido: el sampler es
    **determinístico**. Dos corridas con el mismo estado inicial producen resultados
    idénticos, independientemente del ``generator`` (que se acepta por compatibilidad con
    :meth:`ReverseSampler.step` pero no se usa). Comparte las mismas marginales que
    :class:`~diffusion.samplers.euler_maruyama.EulerMaruyama` y
    :class:`~diffusion.samplers.pf_ode.ProbabilityFlowODE` (Eje 2: cambiar de sampler no
    reentrena la red).
    """

    name = "heun"

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso de Heun de la ODE de flujo de probabilidad.

        Hace un predictor de Euler con el drift en ``t`` y lo corrige promediando con el
        drift evaluado en el estado predicho a tiempo ``t + dt`` (dos evaluaciones de
        score por paso).

        Args:
            x: Estado actual de shape ``(B, data_dim)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: tiempo decreciente).
            generator: Ignorado — el sampler es determinístico. Se acepta solo por
                compatibilidad con la firma de :meth:`ReverseSampler.step`.

        Returns:
            El nuevo estado de shape ``(B, data_dim)``.
        """
        t = self._expand_t(t)
        d1 = self._pfode_drift(x, t)
        x_pred = x + d1 * dt
        d2 = self._pfode_drift(x_pred, t + dt)
        return x + 0.5 * (d1 + d2) * dt
