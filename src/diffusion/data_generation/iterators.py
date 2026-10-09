"""Adaptadores de iteración sobre las fuentes de datos de juguete.

El loop de entrenamiento (:mod:`diffusion.training`) consume una **fuente
infinita de tensores crudos** ``(B, ...)`` con ``next()``, un batch por paso.
En cambio, :meth:`PointDistribution.dataloader` devuelve un ``DataLoader`` de
torch **finito** que yield-ea 1-tuplas ``(x0,)`` (envuelve un solo tensor en un
``TensorDataset``). :func:`infinite_bare` cierra esa brecha sin tocar el
``dataloader``: lo recorre en bucle y desempaqueta la tupla.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def infinite_bare(loader) -> Iterator[torch.Tensor]:
    """Recorre un ``DataLoader`` finito en bucle y yield-ea el tensor crudo.

    Convierte un ``DataLoader`` finito de puntos (el que produce
    :meth:`PointDistribution.dataloader`) en un iterador que **nunca se agota**:
    al terminar de recorrer el loader vuelve a empezar. Cada elemento se
    desempaqueta de la 1-tupla ``(x0,)`` y se yield-ea como el tensor ``x0``
    crudo, que es lo que espera el loop de entrenamiento por pasos.

    No altera el ``loader``; solo lo envuelve.

    Args:
        loader: ``DataLoader`` (u otro iterable) que yield-ea 1-tuplas ``(x0,)``.

    Yields:
        El tensor crudo ``x0`` de cada batch, indefinidamente.
    """
    while True:
        for (x0,) in loader:
            yield x0
