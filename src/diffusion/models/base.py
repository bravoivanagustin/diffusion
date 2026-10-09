"""El contrato de una red de score: ``(x, t) -> score`` con la misma shape que ``x``.

Define el Protocol :class:`ScoreModel`, la firma común a todas las redes del subpaquete
(:class:`~diffusion.models.ScoreMLP` para datos 2D y :class:`~diffusion.models.ScoreUNet`
para imágenes). Es
tipado **estructural** (:class:`typing.Protocol`): ninguna red lo importa ni hereda de él —
lo satisfacen por tener la firma correcta — y sirve para anotar código que recibe "una red
de score cualquiera", p. ej. ``def train(model: ScoreModel, ...)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ScoreModel(Protocol):
    """Una red de score: callable ``(x, t) -> score`` con ``score.shape == x.shape``.

    ``x`` es el dato ruidoso (``(B, data_dim)`` en 2D; ``(B, C, H, W)`` en imágenes) y ``t``
    el tiempo de shape ``(B,)`` o ``(B, 1)``. La salida vive en el mismo espacio que ``x``,
    porque el score :math:`\\nabla_x \\log p_t(x)` tiene la dimensión del dato.
    """

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...
