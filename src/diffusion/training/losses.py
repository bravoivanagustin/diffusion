"""Núcleo del entrenamiento: la pérdida de denoising score matching (DSM).

El corazón del módulo es :func:`dsm_loss`, que combina las tres piezas ya entregadas para
un único batch:

1. ``data_generation`` aporta el dato limpio ``x_0``.
2. ``sde.perturb`` lo ruidea hasta ``x_t`` y devuelve el ruido estándar ``eps`` usado.
3. ``sde.score_target`` da el score real del kernel ``∇_{x_t} log p_t(x_t | x_0)`` y el peso
   ``λ(t)`` de la pérdida.

La red (:class:`diffusion.models.ScoreMLP`) predice ``s_θ(x_t, t)`` y se minimiza el error
pesado ``λ(t) · ||s_θ - score_real||²``. Esta función es **agnóstica a la SDE**: ``perturb``
y ``score_target`` ya devuelven las shapes correctas y el peso adecuado (``λ(t) = σ_t²``).

Es la única pieza donde no hay I/O ni estado: dados ``(net, sde, x_0, t)`` devuelve un escalar
diferenciable, así que se testea directamente sin loop ni archivos.
"""

from __future__ import annotations

import torch

from ..sde import ForwardSDE


def dsm_loss(
    net: torch.nn.Module,
    sde: ForwardSDE,
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pérdida de denoising score matching para un batch.

    Calcula ``mean( λ(t) · || s_θ(x_t, t) - ∇_{x_t} log p_t(x_t | x_0) ||² )`` muestreando un
    único ``x_t`` por dato (estimador de un punto del DSM, suficiente para batches grandes).

    Con ``sample_weights`` la media es ponderada: ``mean( w · λ(t) · ‖·‖² )``. Los pesos son el
    **likelihood ratio** contra el muestreo uniforme de ``t`` (``w(t) = p_unif(t)/q(t)``, con
    ``E_q[w] = 1``, ver :mod:`~diffusion.training.time_sampling`): cuando ``t`` viene de una
    distribución no uniforme ``q``, ponderar por ``w`` deja la pérdida esperada idéntica a la
    del muestreo uniforme — importance sampling puro, que solo reduce la varianza del estimador
    sin cambiar el objetivo que se optimiza.

    Args:
        net: Red de score ``s_θ`` (típicamente :class:`diffusion.models.ScoreMLP`); recibe
            ``(x_t, t)`` y devuelve un tensor de la misma shape que ``x_t``.
        sde: Proceso forward que define el kernel de perturbación y el target del score.
        x0: Dato limpio de shape ``(B, D)`` con ``D = sde.data_dim``.
        t: Tiempo de shape ``(B,)`` o ``(B, 1)``, normalmente en ``[t_eps, T]``.
        generator: Generador opcional de torch para el ruido del kernel (reproducibilidad).
        sample_weights: Pesos por muestra opcionales, ``(B,)`` o ``(B, 1)`` positivos (el
            likelihood ratio del muestreo de ``t``). Con ``None`` (default) la pérdida es la
            media sin ponderar, idéntica al comportamiento previo.

    Returns:
        Escalar (tensor 0-dim) diferenciable con la pérdida media del batch.
    """
    x_t, eps = sde.perturb(x0, t, generator=generator)
    score_real, weight = sde.score_target(x0, t, eps)
    score_pred = net(x_t, t)
    if sample_weights is None:
        # Camino idéntico al previo: weight ya viene rank-matched (B, 1, …, 1) y broadcastea
        # sobre las dimensiones de evento del score.
        return (weight * (score_pred - score_real).pow(2)).mean()
    # Reshape de (B,) / (B, 1) a (B, 1, …, 1) para broadcastear sobre las dimensiones de
    # evento, igual que weight (mismo patrón que ForwardSDE._expand_t). Una primera dimensión
    # incompatible revienta con el error de broadcast de torch: es un parámetro interno del
    # loop, no superficie de usuario.
    w = sample_weights.reshape(sample_weights.shape[0], *([1] * (x0.ndim - 1)))
    return (w * weight * (score_pred - score_real).pow(2)).mean()


def sample_timesteps(
    n: int,
    T: float,
    t_eps: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Muestrea ``n`` tiempos uniformes en ``[t_eps, T]``.

    El piso ``t_eps > 0`` evita ``t = 0``, donde el desvío del kernel ``σ_t → 0`` y el target
    del score ``-eps/σ_t`` se vuelve numéricamente inestable.

    Args:
        n: Cantidad de tiempos (típicamente el tamaño del batch).
        T: Horizonte temporal (``sde.T``).
        t_eps: Piso del muestreo (p. ej. ``1e-3``).
        generator: Generador opcional de torch para reproducibilidad.
        device: Dispositivo de la salida.
        dtype: Tipo de la salida (default ``float32``).

    Returns:
        Tensor de shape ``(n,)`` con tiempos en ``[t_eps, T]``.
    """
    u = torch.rand(n, generator=generator, device=device, dtype=dtype)
    return t_eps + (T - t_eps) * u
