"""Piezas compartidas entre las redes de score.

Agrupa lo que todas las redes del subpaquete (:class:`diffusion.models.ScoreMLP` hoy, la
U-Net de imágenes de Fase 2 a futuro) usan **sin modificar**: el registry de activaciones y
el embedding sinusoidal de tiempo. Esa es la regla de admisión de este archivo — si una pieza
necesita variantes por red (p. ej. el bloque residual: lineal en el MLP, convolucional en la
U-Net), vive en el archivo de esa red, no acá.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

#: Funciones de activación soportadas, por nombre.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _make_activation(name: str) -> nn.Module:
    """Devuelve una instancia de la activación ``name``.

    Args:
        name: Nombre de la activación (p. ej. ``"silu"`` o ``"relu"``).

    Returns:
        Una nueva instancia del ``nn.Module`` de activación.

    Raises:
        ValueError: Si ``name`` no está entre las activaciones soportadas.
    """
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError:
        opts = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(
            f"Activación desconocida '{name}'. Opciones: {opts}"
        ) from None


class SinusoidalEmbedding(nn.Module):
    """Embebe un escalar de tiempo ``t`` en un vector con senos y cosenos.

    Sigue la codificación posicional de Transformers: para cada frecuencia se
    aporta un seno y un coseno, intercalados en el vector de salida::

        embed(t)_{2i}   = sin(t · scale / 10000^{2i/d})
        embed(t)_{2i+1} = cos(t · scale / 10000^{2i/d})

    con ``i = 0, …, d/2 - 1`` y ``d = embed_dim``. Las frecuencias
    (denominadores) se precomputan en :meth:`__init__` y se guardan como buffer
    (no son parámetros: no se aprenden). Funciona para cualquier ``t`` flotante
    no negativo, sin supuestos sobre su escala (el rango depende de la SDE:
    ``[0, 1]``, ``[0, T]`` o pasos enteros).

    El factor ``scale`` multiplica al tiempo **antes** de la codificación y
    sigue la convención de las implementaciones de referencia para ``t``
    continuo en ``[0, 1]``: DDPM condiciona en pasos enteros ``0..999`` y el
    repo oficial de Song et al. (``score_sde_pytorch``) usa ``labels = t * 999``
    para VP/sub-VP continuo. Con el default ``scale=1.0`` la salida es idéntica
    bit a bit a la implementación previa (retrocompatible); ``scale=1000`` es el
    valor recomendado para que la codificación resuelva tiempos chicos
    (``t ≲ 1e-2``). No agrega parámetros entrenables ni estado aleatorio.
    """

    def __init__(self, embed_dim: int = 128, scale: float = 1.0) -> None:
        """Inicializa el embedding.

        Args:
            embed_dim: Dimensión del vector de salida. Debe ser par (cada
                frecuencia aporta un seno y un coseno).
            scale: Factor de escala temporal aplicado a ``t`` antes de la
                codificación. Debe ser finito y positivo. Default ``1.0``
                (comportamiento previo, sin escala).

        Raises:
            ValueError: Si ``embed_dim`` no es par, o si ``scale`` no es
                finito y positivo.
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim debe ser par (un seno y un coseno por frecuencia); "
                f"recibí embed_dim={embed_dim}"
            )
        scale = float(scale)
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(
                f"scale debe ser finito y positivo; recibí scale={scale}"
            )
        self.embed_dim = int(embed_dim)
        #: Factor de escala temporal (float plano, no buffer: introspección y repr).
        self.scale = scale
        # Denominadores 10000^{2i/d} para i = 0 .. d/2 - 1, shape (d/2,).
        i = torch.arange(embed_dim // 2, dtype=torch.float32)
        denom = torch.pow(10000.0, (2.0 * i) / embed_dim)
        #: Buffer (no aprendible): se mueve con .to(device) junto al módulo.
        self.register_buffer("denom", denom)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embebe el tiempo.

        Args:
            t: Tensor de tiempos de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Tensor de shape ``(B, embed_dim)`` con senos y cosenos intercalados.
        """
        t = t.reshape(-1) * self.scale  # (B, 1) o (B,) -> (B,), escalado
        # args[b, i] = t_b * scale / denom_i  ->  (B, d/2)
        args = t[:, None] / self.denom[None, :]
        # Intercalar sin/cos: stack -> (B, d/2, 2) -> reshape (B, d).
        emb = torch.stack((torch.sin(args), torch.cos(args)), dim=-1)
        return emb.reshape(t.shape[0], self.embed_dim)
