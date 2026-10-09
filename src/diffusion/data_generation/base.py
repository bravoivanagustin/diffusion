"""Distribuciones de puntos de juguete (toy point distributions).

Clase base abstracta para generar datasets de puntos en R^dim con distintas
formas. Las formas concretas viven en :mod:`shapes` y se registran en
:mod:`__init__`.

La generación central usa numpy y devuelve ``float32``, así que es liviana y se
puede testear sin torch. Los helpers :meth:`sample_torch` / :meth:`dataloader`
importan torch de forma diferida (lazy) para no obligar a tenerlo instalado salvo
cuando efectivamente se va a entrenar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PointDistribution(ABC):
    """Base de todas las distribuciones de puntos.

    Para crear una forma nueva: subclasear, fijar :attr:`name` (clave del
    registry) y :attr:`supported_dims`, e implementar :meth:`_sample_raw`.
    """

    #: Clave usada en el registry y el CLI. Sobreescribir en cada subclase.
    name: str = ""

    #: Dimensiones soportadas. ``None`` => cualquier ``dim >= 1``. Si no, un
    #: ``frozenset`` con las dims válidas (p. ej. ``frozenset({2})``).
    supported_dims: frozenset[int] | None = None

    def __init__(
        self,
        dim: int,
        *,
        standardize: bool = False,
        noise: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._validate_dim(dim)
        self.dim = int(dim)
        self.standardize = bool(standardize)
        self.noise = float(noise)
        self.seed = seed
        # Stats de estandarización (se completan en sample() si standardize=True).
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        # Etiqueta opcional por punto para colorear previews (la fija _sample_raw).
        self.color_: np.ndarray | None = None

    # ------------------------------------------------------------------- API

    def sample(self, n: int) -> np.ndarray:
        """Genera ``n`` puntos como ``np.ndarray`` float32 de shape ``(n, dim)``."""
        if n <= 0:
            raise ValueError(f"n debe ser positivo; recibí n={n}")
        rng = np.random.default_rng(self.seed)
        self.color_ = None
        x = np.asarray(self._sample_raw(n, rng), dtype=np.float64)
        if x.shape != (n, self.dim):
            raise RuntimeError(
                f"{type(self).__name__}._sample_raw devolvió shape {x.shape}; "
                f"esperaba {(n, self.dim)}"
            )
        if self.standardize:
            self.mean_ = x.mean(axis=0)
            self.std_ = x.std(axis=0)
            self.std_[self.std_ == 0.0] = 1.0
            x = (x - self.mean_) / self.std_
        return x.astype(np.float32, copy=False)

    def sample_torch(self, n: int):
        """Como :meth:`sample` pero devuelve un ``torch.Tensor`` float32."""
        torch = _import_torch()
        return torch.from_numpy(self.sample(n))

    def dataloader(self, n: int, batch_size: int, *, shuffle: bool = True):
        """``DataLoader`` de torch sobre ``n`` puntos muestreados."""
        _import_torch()
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(self.sample_torch(n))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # -------------------------------------------------------- a implementar

    @abstractmethod
    def _sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Geometría cruda: devolver ``(n, dim)`` sin estandarizar ni castear.

        Puede fijar ``self.color_`` con una etiqueta por punto para el preview.
        """
        raise NotImplementedError

    # ------------------------------------------------------------- internos

    @classmethod
    def _validate_dim(cls, dim: int) -> None:
        if cls.supported_dims is not None and dim not in cls.supported_dims:
            dims = ", ".join(str(d) for d in sorted(cls.supported_dims))
            raise ValueError(
                f"{cls.__name__} solo soporta dim en {{{dims}}}; recibi dim={dim}"
            )
        if dim < 1:
            raise ValueError(f"dim debe ser >= 1; recibí dim={dim}")

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"{type(self).__name__}(dim={self.dim}, standardize={self.standardize}, "
            f"noise={self.noise}, seed={self.seed})"
        )


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
        raise ModuleNotFoundError(
            "Esta función necesita PyTorch y no está instalado. "
            "Instalalo con `pip install torch` para usar los helpers de entrenamiento."
        ) from exc
    return torch
