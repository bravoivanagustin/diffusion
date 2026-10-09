"""Formas concretas de distribuciones de puntos.

Se apoya en scikit-learn donde existe un generador adecuado (``make_moons``,
``make_blobs``, ``make_swiss_roll``) y en numpy para la espiral (sklearn no trae
generador de espiral).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll

from .base import PointDistribution


def _seed_from(rng: np.random.Generator) -> int:
    """Deriva un ``random_state`` int para sklearn, atado al rng reproducible."""
    return int(rng.integers(0, 2**31 - 1))


def _split_counts(n: int, parts: int) -> list[int]:
    """Reparte ``n`` en ``parts`` enteros lo más parejos posible (suman ``n``)."""
    base, rem = divmod(n, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _largest_remainder_counts(weights: np.ndarray, n: int) -> np.ndarray:
    """Reparte ``n`` puntos entre componentes de forma determinística.

    Cada componente se lleva la parte entera de ``w_k * n`` y el resto se
    asigna por **mayor residuo**, así el reparto suma exactamente ``n`` y la
    composición es exacta en lugar de aproximada (a diferencia de sortear la
    componente de cada punto).

    Args:
        weights: Pesos ``(K,)`` no negativos que suman uno.
        n: Cantidad total de puntos a repartir.

    Returns:
        Enteros ``(K,)`` no negativos cuya suma es ``n``.
    """
    exact = weights * n
    counts = np.floor(exact).astype(np.int64)
    missing = int(n - counts.sum())
    if missing > 0:
        # ``-residues`` con orden estable => desempate por índice de componente.
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:missing]] += 1
    return counts


class Gaussian(PointDistribution):
    """Gaussiana isotrópica N-dim centrada en el origen."""

    name = "gaussian"
    supported_dims = None  # cualquier dim >= 1

    def __init__(self, dim, *, scale=1.0, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=0.0, seed=seed)
        self.scale = float(scale)

    def _sample_raw(self, n, rng):
        return rng.normal(0.0, self.scale, size=(n, self.dim))


class GaussianMixture(PointDistribution):
    """Mezcla de gaussianas isotrópicas.

    En 2D los centros se ubican en un anillo (el clásico "8 gaussianas"); en
    otras dimensiones, en direcciones aleatorias sobre una hiperesfera de radio
    ``radius``.
    """

    name = "mixture"
    supported_dims = None

    def __init__(self, dim, *, n_components=8, cluster_std=0.3, radius=5.0,
                 standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=0.0, seed=seed)
        if n_components < 1:
            raise ValueError(f"n_components debe ser >= 1; recibí {n_components}")
        self.n_components = int(n_components)
        self.cluster_std = float(cluster_std)
        self.radius = float(radius)

    def _centers(self, rng):
        k = self.n_components
        if self.dim == 2:
            ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
            return self.radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        dirs = rng.normal(size=(k, self.dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return self.radius * dirs

    def _sample_raw(self, n, rng):
        centers = self._centers(rng)
        x, y = make_blobs(
            n_samples=n,
            centers=centers,
            cluster_std=self.cluster_std,
            random_state=_seed_from(rng),
        )
        self.color_ = y
        return x


def _validate_scale_and_anisotropy(scale: float, anisotropy: float) -> None:
    """Valida las dos palancas de forma comunes a los constructores de geometría.

    Args:
        scale: Escala global de una componente; debe ser > 0.
        anisotropy: Razón entre el autovalor mayor y el menor; debe ser >= 1.

    Raises:
        ValueError: Si ``scale <= 0`` o si ``anisotropy < 1`` (también con NaN).
            El mensaje empieza nombrando el parámetro culpable.
    """
    if not float(scale) > 0.0:
        raise ValueError(f"scale debe ser > 0; recibí {scale!r}")
    if not float(anisotropy) >= 1.0:
        raise ValueError(
            f"anisotropy debe ser >= 1: es la razón entre el autovalor mayor y "
            f"el menor de la componente, así que 1 es el caso isotrópico y "
            f"valores mayores la estiran; recibí {anisotropy!r}"
        )


def _anisotropic_covariance(
    scale: float, anisotropy: float, angle: float
) -> np.ndarray:
    """Covarianza 2x2 SPD con anisotropía ``anisotropy`` y eje mayor en ``angle``.

    Los autovalores son ``scale² · √κ`` a lo largo de ``angle`` y ``scale² / √κ``
    en la dirección perpendicular, con ``κ = anisotropy``. Dos consecuencias
    deliberadas de esa elección:

    - la **razón** entre el autovalor mayor y el menor es exactamente ``κ``, que
      es la definición de anisotropía que usan los constructores de geometría;
    - la **media geométrica** de los autovalores queda igual a ``scale²`` para
      cualquier ``κ``, así que subir la anisotropía estira la componente **sin**
      cambiar su tamaño global (el determinante —el área de la elipse de
      covarianza— se conserva). Si en cambio se escalara solo el autovalor
      mayor, mover ``κ`` cambiaría en silencio la dispersión total y el barrido
      dejaría de aislar la anisotropía.

    Args:
        scale: Escala global de la componente (> 0).
        anisotropy: Razón mayor/menor de autovalores (>= 1); 1 es isotrópica.
        angle: Ángulo en radianes del eje mayor.

    Returns:
        Matriz ``(2, 2)`` float64, simétrica y definida positiva.
    """
    root = np.sqrt(float(anisotropy))
    var = float(scale) ** 2
    major = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    minor = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float64)
    return var * root * np.outer(major, major) + (var / root) * np.outer(minor, minor)


class ExactGaussianMixture(PointDistribution):
    """Mixtura de gaussianas 2D con parámetros exactos y consultables.

    Es la forma hermana de :class:`GaussianMixture` para el laboratorio 2D
    analítico: acá los pesos, las medias y las covarianzas son **entrada
    explícita** y quedan publicados en :attr:`weights_`, :attr:`means_` y
    :attr:`covariances_`, así que la densidad de los datos se conoce en forma
    cerrada y puede usarse como patrón de referencia. La legacy, en cambio, es
    isotrópica, de pesos parejos y no expone sus centros.

    Admite covarianzas simétricas definidas positivas arbitrarias (no solo
    diagonales), de modo que una componente pueda estar rotada además de
    estirada. Solo ``dim = 2``: la exactitud barata de esta línea de trabajo
    depende de la baja dimensión.

    No se registra en el registry del módulo: sus parámetros son matrices, que
    no viajan por CLI ni por YAML, así que el uso natural es importarla y
    construirla desde código.

    Note:
        No admite la estandarización empírica de :class:`PointDistribution`:
        esa transformación se estima a partir de la muestra sorteada, así que
        los parámetros efectivos dependerían de ``n`` y de la semilla y
        dejarían de ser exactos. ``standardize`` está en la firma justamente
        para poder rechazar ese caso con un error explícito, en lugar de que
        ``make_distribution`` lo descarte en silencio.
    """

    name = "mixture_exact"
    supported_dims = frozenset({2})

    def __init__(
        self,
        dim: int = 2,
        *,
        weights: Sequence[float] | np.ndarray,
        means: Sequence[Sequence[float]] | np.ndarray,
        covariances: Sequence[Sequence[Sequence[float]]] | np.ndarray,
        standardize: bool = False,
        seed: int | None = None,
    ) -> None:
        """Construye la mixtura validando sus parámetros de entrada.

        Args:
            dim: Dimensión del espacio; solo se admite ``2``.
            weights: Pesos ``(K,)``, no negativos y que suman uno.
            means: Medias ``(K, 2)``, una por componente.
            covariances: Covarianzas ``(K, 2, 2)``, cada una simétrica
                definida positiva.
            standardize: Debe ser ``False``; existe solo para rechazar de
                forma ruidosa la estandarización empírica.
            seed: Semilla del muestreo, como en el resto de las formas.

        Raises:
            ValueError: Si ``dim`` no es 2, si se pide ``standardize=True``, si
                las shapes de ``weights``/``means``/``covariances`` declaran un
                ``K`` inconsistente, si algún peso es negativo, si los pesos no
                suman uno, o si alguna covarianza no es simétrica definida
                positiva. El mensaje nombra el parámetro culpable.
        """
        self._validate_dim(dim)
        if standardize:
            raise ValueError(
                "ExactGaussianMixture no admite standardize=True: la "
                "estandarización empírica estima media y desvío a partir de la "
                "muestra sorteada, así que los parámetros dependerían de n y de "
                "la semilla y dejarían de ser exactos. Usá standardize=False y, "
                "si querés datos centrados, declaralo en means/covariances."
            )
        super().__init__(dim, standardize=False, noise=0.0, seed=seed)

        # Se copian los tres arrays de entrada: el objeto es dueño de sus
        # parámetros, así que ni los invariantes que se validan acá abajo (pesos
        # no negativos que suman uno, covarianzas SPD) ni la factorización de
        # Cholesky que queda persistida pueden desincronizarse de lo que
        # reportan los accesores si el llamador muta sus buffers después.
        w = np.array(weights, dtype=np.float64)
        if w.ndim != 1 or w.size < 1:
            raise ValueError(
                f"weights debe ser un vector (K,) con K >= 1; recibí shape {w.shape}"
            )
        k = int(w.shape[0])

        mu = np.array(means, dtype=np.float64)
        if mu.shape != (k, 2):
            raise ValueError(
                f"means debe tener shape (K, 2) con los K={k} componentes que "
                f"declara weights; recibí shape {mu.shape}"
            )

        cov = np.array(covariances, dtype=np.float64)
        if cov.shape != (k, 2, 2):
            raise ValueError(
                f"covariances debe tener shape (K, 2, 2) con los K={k} "
                f"componentes que declara weights; recibí shape {cov.shape}"
            )

        if np.any(w < 0.0):
            bad = np.flatnonzero(w < 0.0).tolist()
            raise ValueError(
                f"weights debe tener pesos no negativos; son negativos los "
                f"índices {bad}: {w[w < 0.0].tolist()}"
            )
        total = float(w.sum())
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(f"weights debe sumar 1.0; recibí suma={total!r}")

        # La definición positiva se verifica con la misma factorización de
        # Cholesky que después reusa el muestreo: si factoriza, es SPD.
        chols = np.empty_like(cov)
        for i in range(k):
            c = cov[i]
            if not np.allclose(c, c.T, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"covariances[{i}] debe ser simétrica; recibí {c.tolist()}"
                )
            try:
                chols[i] = np.linalg.cholesky(c)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"covariances[{i}] debe ser definida positiva (falló su "
                    f"factorización de Cholesky); recibí {c.tolist()}"
                ) from exc

        self._weights = w
        self._means = mu
        self._covariances = cov
        self._chols = chols

    # ------------------------------------------------- parámetros verdaderos

    @property
    def weights_(self) -> np.ndarray:
        """Pesos verdaderos ``(K,)`` float64 (copia)."""
        return self._weights.copy()

    @property
    def means_(self) -> np.ndarray:
        """Medias verdaderas ``(K, 2)`` float64 (copia)."""
        return self._means.copy()

    @property
    def covariances_(self) -> np.ndarray:
        """Covarianzas verdaderas ``(K, 2, 2)`` float64 (copia)."""
        return self._covariances.copy()

    # ------------------------------------------- geometrías del estudio (1.7)

    @classmethod
    def two_modes(
        cls,
        *,
        separation: float,
        weights: tuple[float, float] = (0.5, 0.5),
        anisotropy: float = 1.0,
        scale: float = 0.3,
        angle: float = 0.0,
        seed: int | None = None,
    ) -> "ExactGaussianMixture":
        """Mixtura de dos modos, parametrizada por las palancas del estudio.

        Es el atajo para la geometría de dos modos del barrido: separación,
        desbalance de pesos y anisotropía se piden **por nombre** y quedan
        publicadas en los accesores, sin que el llamador arme matrices de
        covarianza a mano.

        Geometría: las dos medias quedan **simétricas respecto del origen**,
        a distancia ``separation`` entre sí, sobre la dirección rotada por
        ``angle`` (``means[0] = -½·separation·u``, ``means[1] = +½·separation·u``
        con ``u = (cos angle, sin angle)``). La elongación se aplica **a lo largo
        de la dirección de separación**: es la orientación que hace significativo
        el barrido de soporte casi degenerado, porque los modos se estiran uno
        hacia el otro. Ambas componentes comparten la misma covarianza.

        Args:
            separation: Distancia entre las dos medias (>= 0). ``0`` es el límite
                degenerado, con los dos modos superpuestos en el origen.
            weights: Pesos ``(w0, w1)``, no negativos y que suman uno. Acá vive
                la palanca de desbalance (p. ej. ``(0.99, 0.01)``).
            anisotropy: Razón entre el autovalor mayor y el menor de cada
                componente (>= 1); ``1`` es isotrópica. La media geométrica de
                los autovalores se mantiene en ``scale**2``, así que estirar no
                cambia el tamaño global de la componente (ver
                :func:`_anisotropic_covariance`).
            scale: Escala global de cada componente (> 0). Con ``anisotropy=1``
                la covarianza es ``scale**2 · I``.
            angle: Ángulo en radianes de la dirección de separación (y, por lo
                tanto, del eje mayor de las componentes).
            seed: Semilla del muestreo.

        Returns:
            La mixtura construida, con sus parámetros ya validados.

        Raises:
            ValueError: Si ``separation < 0``, si ``weights`` no tiene
                exactamente dos pesos, si ``scale <= 0`` o si
                ``anisotropy < 1``; y los mismos rechazos del constructor
                general para pesos negativos o que no suman uno. El mensaje
                empieza nombrando el parámetro culpable.
        """
        if not float(separation) >= 0.0:
            raise ValueError(
                f"separation debe ser >= 0 (es la distancia entre las dos "
                f"medias); recibí {separation!r}"
            )
        w = np.array(weights, dtype=np.float64)
        if w.shape != (2,):
            raise ValueError(
                f"weights debe tener exactamente 2 pesos, uno por modo; recibí "
                f"shape {w.shape}"
            )
        _validate_scale_and_anisotropy(scale, anisotropy)

        direction = np.array(
            [np.cos(float(angle)), np.sin(float(angle))], dtype=np.float64
        )
        half = 0.5 * float(separation)
        means = np.stack([-half * direction, half * direction])
        cov = _anisotropic_covariance(scale, anisotropy, float(angle))
        covariances = np.stack([cov, cov])
        return cls(2, weights=w, means=means, covariances=covariances, seed=seed)

    @classmethod
    def ring(
        cls,
        *,
        n_components: int = 8,
        radius: float = 5.0,
        scale: float = 0.3,
        weights: Sequence[float] | None = None,
        anisotropy: float = 1.0,
        seed: int | None = None,
    ) -> "ExactGaussianMixture":
        """Mixtura en anillo, parametrizada por las palancas del estudio.

        Es el atajo para la geometría de anillo del barrido (el clásico "8
        gaussianas", acá con pesos y anisotropía controlables): las medias van
        equiespaciadas en un círculo de radio ``radius`` **empezando en ángulo
        0**, ``radius · (cos 2πk/K, sin 2πk/K)``, que es la misma convención
        angular de :class:`GaussianMixture` en 2D, así que las geometrías del
        estudio quedan comparables con el trabajo anterior.

        La anisotropía estira cada componente **en su dirección radial** (la de
        su propio centro): es la orientación que hace significativo el barrido de
        soporte casi degenerado sobre un anillo, porque los modos se alargan
        hacia el centro y hacia afuera en lugar de tangencialmente. La dirección
        se toma del **ángulo** de la componente y no del vector de su media, así
        que sigue estando definida incluso con ``radius = 0``, donde todas las
        medias caen en el origen.

        Args:
            n_components: Cantidad de componentes ``K`` (>= 1).
            radius: Radio del anillo (>= 0). ``0`` apila todas las medias en el
                origen (límite degenerado).
            scale: Escala global de cada componente (> 0). Con ``anisotropy=1``
                la covarianza es ``scale**2 · I``.
            weights: Pesos ``(K,)`` no negativos que suman uno; ``None`` da el
                anillo parejo de ``1/K``. Acá vive la palanca de desbalance.
            anisotropy: Razón entre el autovalor mayor y el menor de cada
                componente (>= 1); ``1`` es isotrópica. La media geométrica de
                los autovalores se mantiene en ``scale**2`` (ver
                :func:`_anisotropic_covariance`).
            seed: Semilla del muestreo.

        Returns:
            La mixtura construida, con sus parámetros ya validados.

        Raises:
            ValueError: Si ``n_components < 1``, si ``radius < 0``, si
                ``scale <= 0``, si ``anisotropy < 1`` o si ``weights`` no tiene
                un peso por componente; y los mismos rechazos del constructor
                general para pesos negativos o que no suman uno. El mensaje
                empieza nombrando el parámetro culpable.
        """
        k = int(n_components)
        if k < 1:
            raise ValueError(f"n_components debe ser >= 1; recibí {n_components!r}")
        if not float(radius) >= 0.0:
            raise ValueError(
                f"radius debe ser >= 0 (es el radio del anillo); recibí {radius!r}"
            )
        _validate_scale_and_anisotropy(scale, anisotropy)

        if weights is None:
            w = np.full(k, 1.0 / k, dtype=np.float64)
        else:
            w = np.array(weights, dtype=np.float64)
            if w.shape != (k,):
                raise ValueError(
                    f"weights debe tener un peso por componente, es decir shape "
                    f"({k},) para n_components={k}; recibí shape {w.shape}"
                )

        angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
        means = float(radius) * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        covariances = np.stack(
            [_anisotropic_covariance(scale, anisotropy, a) for a in angles]
        )
        return cls(2, weights=w, means=means, covariances=covariances, seed=seed)

    # ------------------------------------------------------------- muestreo

    def _sample_raw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Muestrea ``n`` puntos por componente y publica su etiqueta.

        Reparte ``n`` entre componentes por mayor residuo (determinístico y de
        composición exacta) y transforma ruido normal estándar con la
        factorización de Cholesky ya validada en la construcción. Los puntos
        salen **agrupados por componente** (todos los de la 0, después los de la
        1, …), sin permutación final.

        Note:
            El contrato de muestreo —composición exacta contra los pesos
            declarados, agrupamiento por componente, reproducibilidad por
            semilla, dtype y shape— queda fijado en
            ``tests/test_data_generation.py`` por
            ``test_exact_mixture_composition_*``,
            ``test_exact_mixture_samples_are_grouped_by_component`` y
            ``test_exact_mixture_same_seed_gives_identical_samples``.
        """
        counts = _largest_remainder_counts(self._weights, n)
        points, labels = [], []
        for i, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            z = rng.standard_normal(size=(count, 2))
            points.append(self._means[i] + z @ self._chols[i].T)
            labels.append(np.full(count, i, dtype=np.int64))
        self.color_ = np.concatenate(labels, axis=0)
        return np.concatenate(points, axis=0)


class TwoMoons(PointDistribution):
    """Dos medias lunas entrelazadas (2D). Vía ``sklearn.make_moons``."""

    name = "two_moons"
    supported_dims = frozenset({2})

    def __init__(self, dim=2, *, noise=0.05, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=noise, seed=seed)

    def _sample_raw(self, n, rng):
        x, y = make_moons(n_samples=n, noise=self.noise, random_state=_seed_from(rng))
        self.color_ = y
        return x


class Spiral(PointDistribution):
    """Una o más espirales entrelazadas (2D). Implementada con numpy."""

    name = "spiral"
    supported_dims = frozenset({2})

    def __init__(self, dim=2, *, noise=0.02, n_arms=2, turns=1.5,
                 standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=noise, seed=seed)
        if n_arms < 1:
            raise ValueError(f"n_arms debe ser >= 1; recibí {n_arms}")
        self.n_arms = int(n_arms)
        self.turns = float(turns)

    def _sample_raw(self, n, rng):
        max_t = self.turns * 2.0 * np.pi
        arms, labels = [], []
        for i, k in enumerate(_split_counts(n, self.n_arms)):
            # sqrt(U) => densidad ~uniforme en área a lo largo del radio.
            t = np.sqrt(rng.uniform(0.0, 1.0, size=k)) * max_t
            phase = i * (2.0 * np.pi / self.n_arms)
            xy = np.stack([t * np.cos(t + phase), t * np.sin(t + phase)], axis=1)
            xy = xy / max_t  # normalizar a ~[-1, 1]
            xy = xy + rng.normal(0.0, self.noise, size=(k, 2))
            arms.append(xy)
            labels.append(np.full(k, i))
        out = np.concatenate(arms, axis=0)
        lab = np.concatenate(labels, axis=0)
        perm = rng.permutation(len(out))
        self.color_ = lab[perm]
        return out[perm]


class SwissRoll(PointDistribution):
    """Rollo suizo: variedad 2D embebida en 3D. Vía ``sklearn.make_swiss_roll``."""

    name = "swiss_roll"
    supported_dims = frozenset({3})

    def __init__(self, dim=3, *, noise=0.5, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=noise, seed=seed)

    def _sample_raw(self, n, rng):
        x, t = make_swiss_roll(
            n_samples=n, noise=self.noise, random_state=_seed_from(rng)
        )
        self.color_ = t
        return x
