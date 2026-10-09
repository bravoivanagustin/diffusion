"""Tests del módulo de generación de datos de juguete (`diffusion.data_generation`)."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest

import itertools

from diffusion.data_generation import (
    REGISTRY,
    ExactGaussianMixture,
    Gaussian,
    GaussianMixture,
    TwoMoons,
    available_shapes,
    infinite_bare,
    make_distribution,
)

ALL_SHAPES = ["gaussian", "mixture", "two_moons", "spiral", "swiss_roll"]
DEFAULT_DIM = {
    "gaussian": 2,
    "mixture": 2,
    "two_moons": 2,
    "spiral": 2,
    "swiss_roll": 3,
}

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data_generation.py"


@pytest.mark.parametrize("name", ALL_SHAPES)
def test_sample_shape_dtype_finite(name):
    dim = DEFAULT_DIM[name]
    x = make_distribution(name, dim, seed=0).sample(256)
    assert x.shape == (256, dim)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))


def test_available_shapes():
    assert set(available_shapes()) == set(ALL_SHAPES)


def test_factory_returns_right_type():
    assert isinstance(make_distribution("two_moons", 2), TwoMoons)
    assert isinstance(make_distribution("gaussian", 5), Gaussian)


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        make_distribution("does_not_exist", 2)


@pytest.mark.parametrize("name,bad_dim", [("two_moons", 3), ("spiral", 3), ("swiss_roll", 2)])
def test_unsupported_dim_raises(name, bad_dim):
    with pytest.raises(ValueError):
        make_distribution(name, bad_dim)


@pytest.mark.parametrize("name,dim", [("gaussian", 5), ("mixture", 4)])
def test_generalizable_shapes_accept_any_dim(name, dim):
    x = make_distribution(name, dim, seed=1).sample(64)
    assert x.shape == (64, dim)


@pytest.mark.parametrize("name", ALL_SHAPES)
def test_reproducibility(name):
    dim = DEFAULT_DIM[name]
    a = make_distribution(name, dim, seed=7).sample(128)
    b = make_distribution(name, dim, seed=7).sample(128)
    c = make_distribution(name, dim, seed=8).sample(128)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_standardize_gives_zero_mean_unit_std():
    x = make_distribution("gaussian", 3, scale=5.0, standardize=True, seed=0).sample(5000)
    assert np.allclose(x.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(x.std(axis=0), 1.0, atol=0.1)


def test_non_positive_n_raises():
    with pytest.raises(ValueError):
        make_distribution("gaussian", 2, seed=0).sample(0)


def test_torch_helpers():
    torch = pytest.importorskip("torch")
    dist = make_distribution("two_moons", 2, seed=0)
    t = dist.sample_torch(100)
    assert tuple(t.shape) == (100, 2)
    assert t.dtype == torch.float32
    loader = dist.dataloader(100, batch_size=32)
    batch = next(iter(loader))[0]
    assert batch.shape[1] == 2
    assert batch.shape[0] <= 32


def test_infinite_bare_does_not_exhaust():
    # Loader finito de 128 puntos en batches de 64 => 2 batches. Consumir 5
    # veces no debe agotarse (reinicia el recorrido). (4.1, 4.3)
    pytest.importorskip("torch")
    dist = make_distribution("two_moons", 2, seed=0)
    it = infinite_bare(dist.dataloader(128, batch_size=64))
    batches = list(itertools.islice(it, 5))
    assert len(batches) == 5


def test_infinite_bare_yields_bare_tensor():
    # Cada elemento es un tensor crudo (B, 2), no una tupla (x0,). (4.2)
    torch = pytest.importorskip("torch")
    dist = make_distribution("two_moons", 2, seed=0)
    it = infinite_bare(dist.dataloader(128, batch_size=64))
    for batch in itertools.islice(it, 5):
        assert isinstance(batch, torch.Tensor)
        assert not isinstance(batch, tuple)
        assert batch.ndim == 2
        assert batch.shape[1] == 2
        assert batch.shape[0] <= 64


# --------------------------------------------------------------------------
# ExactGaussianMixture: parámetros exactos, validación y accesores (1.1, 1.2,
# 1.3, 1.6, 5.1). Acá solo se prueba la **construcción**; el contrato de
# muestreo (composición, reproducibilidad, dtype/shape) lo fijan los tests de
# la sección "contrato de muestreo" más abajo.
# --------------------------------------------------------------------------

EXACT_WEIGHTS = [0.6, 0.4]
EXACT_MEANS = [[-2.0, 0.0], [2.0, 1.0]]
EXACT_COVS = [
    [[1.0, 0.0], [0.0, 1.0]],
    [[2.0, 0.5], [0.5, 0.75]],
]


def _exact_mixture(**overrides):
    """Mixtura exacta de 2 componentes, con los parámetros que se quieran pisar."""
    kwargs = {
        "weights": EXACT_WEIGHTS,
        "means": EXACT_MEANS,
        "covariances": EXACT_COVS,
    }
    kwargs.update(overrides)
    return ExactGaussianMixture(**kwargs)


def _empirical_moments(x, labels, k):
    """Media y covarianza empíricas de los puntos etiquetados como componente ``k``."""
    pts = np.asarray(x, dtype=np.float64)[np.asarray(labels) == k]
    return pts.mean(axis=0), np.cov(pts, rowvar=False)


def test_exact_mixture_exposes_parameters_before_sampling():
    # 1.1, 1.2: los tres accesores están disponibles sin haber muestreado.
    mix = _exact_mixture(seed=0)
    assert mix.color_ is None  # nunca se muestreó
    assert mix.weights_.shape == (2,)
    assert mix.means_.shape == (2, 2)
    assert mix.covariances_.shape == (2, 2, 2)
    assert mix.weights_.dtype == np.float64
    assert mix.means_.dtype == np.float64
    assert mix.covariances_.dtype == np.float64
    assert np.array_equal(mix.weights_, np.asarray(EXACT_WEIGHTS))
    assert np.array_equal(mix.means_, np.asarray(EXACT_MEANS))
    assert np.array_equal(mix.covariances_, np.asarray(EXACT_COVS))


def test_exact_mixture_accessors_return_copies():
    # 1.2: los parámetros verdaderos son inmutables desde afuera.
    mix = _exact_mixture()
    mix.weights_[0] = 99.0
    mix.means_[0, 0] = 99.0
    mix.covariances_[0, 0, 0] = 99.0
    assert np.array_equal(mix.weights_, np.asarray(EXACT_WEIGHTS))
    assert np.array_equal(mix.means_, np.asarray(EXACT_MEANS))
    assert np.array_equal(mix.covariances_, np.asarray(EXACT_COVS))


def test_exact_mixture_copies_constructor_arrays():
    # 1.2, 1.6: la mixtura es dueña de sus parámetros. Si aliaseara los arrays
    # del llamador, mutarlos después de construir rompería en silencio los
    # invariantes ya validados (pesos que suman uno, covarianzas SPD) y —peor—
    # desincronizaría covariances_ de la factorización de Cholesky persistida,
    # que es la que gobierna el muestreo.
    weights = np.array(EXACT_WEIGHTS, dtype=np.float64)
    means = np.array(EXACT_MEANS, dtype=np.float64)
    covariances = np.array(EXACT_COVS, dtype=np.float64)
    mix = ExactGaussianMixture(
        weights=weights, means=means, covariances=covariances, seed=0
    )

    # Mutaciones que la validación habría rechazado de haberlas visto.
    weights[0] = 99.0  # los pesos ya no suman uno
    means[0, 0] = 99.0
    covariances[0, 0, 0] = 100.0  # varianza inflada
    covariances[1] = [[-1.0, 0.0], [0.0, 3.0]]  # indefinida

    assert np.array_equal(mix.weights_, np.asarray(EXACT_WEIGHTS, dtype=np.float64))
    assert np.array_equal(mix.means_, np.asarray(EXACT_MEANS, dtype=np.float64))
    assert np.array_equal(mix.covariances_, np.asarray(EXACT_COVS, dtype=np.float64))
    # Los invariantes de la construcción siguen valiendo.
    assert float(mix.weights_.sum()) == pytest.approx(1.0)
    for cov in mix.covariances_:
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)
    # Y lo que reportan los accesores sigue siendo lo que usa el muestreo: los
    # momentos empíricos de cada componente reproducen los parámetros
    # publicados. Es el chequeo por la puerta pública del mismo invariante (si
    # los arrays se aliasaran, `covariances_` reportaría la varianza inflada de
    # 100 mientras el muestreo seguiría usando la factorización original).
    x = mix.sample(8000)
    for i in range(2):
        mean_emp, cov_emp = _empirical_moments(x, mix.color_, i)
        assert np.allclose(mean_emp, mix.means_[i], atol=0.2)
        assert np.allclose(cov_emp, mix.covariances_[i], atol=0.2)


def test_exact_mixture_accepts_full_spd_covariance():
    # 1.3: covarianza rotada (no diagonal) y anisotrópica: se acepta tal cual.
    ang = np.pi / 6.0
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    cov = rot @ np.diag([4.0, 0.25]) @ rot.T
    mix = ExactGaussianMixture(
        weights=[1.0], means=[[0.0, 0.0]], covariances=[cov], seed=0
    )
    assert np.allclose(mix.covariances_[0], cov)
    # Sigue siendo SPD y no diagonal: los autovalores son los pedidos.
    eigvals = np.linalg.eigvalsh(mix.covariances_[0])
    assert np.allclose(np.sort(eigvals), [0.25, 4.0])
    assert not np.isclose(mix.covariances_[0, 0, 1], 0.0)


@pytest.mark.parametrize(
    "overrides,culprit",
    [
        # Peso negativo (compensado para que sume uno: solo falla el signo).
        ({"weights": [-0.2, 1.2]}, "weights"),
        # Pesos que no suman uno.
        ({"weights": [0.5, 0.4]}, "weights"),
        # Sin componentes.
        ({"weights": []}, "weights"),
        # K inconsistente entre pesos y medias: con K=3 pesos, el que queda mal
        # dimensionado es means, y es a means a quien nombra el error.
        ({"weights": [0.5, 0.3, 0.2]}, "means"),
        ({"means": [[0.0, 0.0]]}, "means"),
        # K inconsistente entre pesos y covarianzas.
        ({"covariances": [[[1.0, 0.0], [0.0, 1.0]]]}, "covariances"),
        # Medias que no son 2D.
        ({"means": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}, "means"),
        # Covarianza no simétrica.
        ({"covariances": [EXACT_COVS[0], [[1.0, 0.3], [0.9, 1.0]]]}, "covariances"),
        # Covarianza simétrica pero indefinida.
        ({"covariances": [EXACT_COVS[0], [[1.0, 2.0], [2.0, 1.0]]]}, "covariances"),
        # Covarianza semidefinida (autovalor nulo): no es definida positiva.
        ({"covariances": [EXACT_COVS[0], [[1.0, 0.0], [0.0, 0.0]]]}, "covariances"),
    ],
)
def test_exact_mixture_rejects_invalid_parameters(overrides, culprit):
    # 1.6: cada entrada inválida nombra el parámetro culpable. El patrón va
    # anclado al principio del mensaje: los errores de K inconsistente
    # mencionan a `weights` de pasada (es quien declara el K de referencia), así
    # que sin el ancla el test no discriminaría a quién se está culpando.
    with pytest.raises(ValueError, match=f"^{culprit}"):
        _exact_mixture(**overrides)


def test_exact_mixture_rejects_non_2d_dim():
    # 1.6: la exactitud barata de esta línea de trabajo vive en dim=2.
    with pytest.raises(ValueError, match="dim"):
        _exact_mixture(dim=3)


def test_exact_mixture_rejects_empirical_standardization():
    # 5.1: la estandarización empírica estima la transformación a partir del
    # sorteo, así que rompe la exactitud. Se rechaza fuerte en lugar de dejar
    # que make_distribution la descarte en silencio.
    with pytest.raises(ValueError, match="standardize"):
        _exact_mixture(standardize=True)


def test_exact_mixture_accepts_standardize_false_explicitly():
    # 5.1: el parámetro existe solo para poder rechazar el caso peligroso.
    mix = _exact_mixture(standardize=False)
    assert mix.standardize is False


# --------------------------------------------------------------------------
# ExactGaussianMixture: contrato de muestreo (1.4, 1.5).
#
# El muestreo es determinístico en dos sentidos independientes que se prueban
# por separado: (a) la *composición* —cuántos puntos toca a cada componente— no
# se sortea, se reparte por mayor residuo, así que es exacta y no aproximada; y
# (b) la *geometría* de cada punto sí se sortea, pero de un rng sembrado, así
# que es reproducible.
# --------------------------------------------------------------------------


def _ring_mixture(weights, *, scale=0.25, seed=0):
    """Mixtura exacta con los pesos dados, medias en un anillo y covarianzas iguales.

    Sirve para los tests de composición, donde lo que importa son los pesos y no
    la geometría: las medias van bien separadas solo para que las etiquetas sean
    verificables sin ambigüedad.
    """
    weights = np.asarray(weights, dtype=np.float64)
    k = weights.size
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    means = 5.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    covs = np.broadcast_to(np.eye(2) * scale**2, (k, 2, 2))
    return ExactGaussianMixture(
        weights=weights, means=means, covariances=covs, seed=seed
    )


def test_exact_mixture_sample_returns_two_columns_float32():
    # 1.4: shape (n, 2) y precisión simple, como el resto de las formas.
    x = _exact_mixture(seed=0).sample(257)
    assert x.shape == (257, 2)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))


def test_exact_mixture_composition_matches_weights_exactly():
    # 1.4: con w_k * n entero la composición es *exacta*, no aproximada: 600/400
    # y no "600 ± ruido binomial". Sortear la componente de cada punto no
    # pasaría este test.
    mix = _exact_mixture(seed=0)  # weights = [0.6, 0.4]
    x = mix.sample(1000)
    counts = np.bincount(mix.color_, minlength=2)
    assert x.shape == (1000, 2)
    assert counts.tolist() == [600, 400]


def test_exact_mixture_composition_uses_largest_remainder():
    # 1.4: con w_k * n NO entero hay un resto que repartir. Pesos [0.7, 0.2,
    # 0.1] y n=13 dan w*n = [9.1, 2.6, 1.3] => parte entera [9, 2, 1] (suma 12)
    # y el punto que falta va al **mayor residuo** (0.6, la componente 1).
    # Discrimina las alternativas: un reparto parejo daría [5, 4, 4] y darle el
    # sobrante al peso más grande daría [10, 2, 1].
    mix = _ring_mixture([0.7, 0.2, 0.1], seed=3)
    mix.sample(13)
    assert np.bincount(mix.color_, minlength=3).tolist() == [9, 3, 1]


def test_exact_mixture_composition_ties_break_by_component_index():
    # 1.4: con residuos empatados el desempate es estable por índice, así que el
    # reparto es reproducible y no depende del orden interno del sort. Pesos
    # iguales de a tercios con n=10 dan parte entera [3, 3, 3] y el punto
    # restante va a la componente 0.
    mix = _ring_mixture([1 / 3, 1 / 3, 1 / 3], seed=4)
    mix.sample(10)
    assert np.bincount(mix.color_, minlength=3).tolist() == [4, 3, 3]


def test_exact_mixture_composition_with_fewer_points_than_components():
    # 1.4: con n < K casi toda la parte entera es cero y el reparto lo decide el
    # residuo: los 3 puntos van a las 3 componentes de mayor peso, sin perder
    # ninguno. Los pesos van en orden creciente para que el resultado esperado
    # ([0, 1, 1, 1]) no coincida con el de un reparto parejo ([1, 1, 1, 0]).
    mix = _ring_mixture([0.1, 0.2, 0.3, 0.4], seed=5)
    x = mix.sample(3)
    counts = np.bincount(mix.color_, minlength=4)
    assert x.shape == (3, 2)
    assert int(counts.sum()) == 3
    assert counts.tolist() == [0, 1, 1, 1]


def test_exact_mixture_zero_weight_component_gets_no_points():
    # 1.4: un peso nulo es un peso válido (no negativo) y su componente no
    # aporta ni un punto; la etiqueta correspondiente no aparece.
    mix = _ring_mixture([0.0, 0.5, 0.5], seed=6)
    mix.sample(100)
    assert np.bincount(mix.color_, minlength=3).tolist() == [0, 50, 50]
    assert 0 not in set(mix.color_.tolist())


def test_exact_mixture_same_seed_gives_identical_samples():
    # 1.5: dos instancias con la misma semilla dan exactamente las mismas
    # muestras (igualdad bit a bit, no "cerca").
    a = _exact_mixture(seed=7).sample(256)
    b = _exact_mixture(seed=7).sample(256)
    assert np.array_equal(a, b)


def test_exact_mixture_repeated_calls_with_same_seed_are_identical():
    # 1.5: "las mismas muestras que en la llamada anterior" también dentro de la
    # misma instancia: sample() resiembra el rng en cada llamada, así que no
    # arrastra estado del sorteo previo.
    mix = _exact_mixture(seed=7)
    a = mix.sample(256)
    labels_a = mix.color_.copy()
    b = mix.sample(256)
    assert np.array_equal(a, b)
    assert np.array_equal(labels_a, mix.color_)


def test_exact_mixture_different_seed_gives_different_samples():
    # 1.5: la reproducibilidad no es un muestreo degenerado; con otra semilla la
    # geometría cambia (la composición, en cambio, es la misma a propósito).
    a = _exact_mixture(seed=7).sample(256)
    c = _exact_mixture(seed=8).sample(256)
    assert not np.array_equal(a, c)


def test_exact_mixture_publishes_label_per_point():
    # 1.4: la etiqueta de componente se publica por punto, como la legacy, y
    # cada etiqueta es un índice de componente válido.
    mix = _exact_mixture(seed=0)
    x = mix.sample(64)
    assert mix.color_.shape == (len(x),)
    assert mix.color_.dtype.kind in "iu"
    assert set(mix.color_.tolist()) <= {0, 1}


def test_exact_mixture_color_reflects_only_the_current_sample():
    # 1.4: convención heredada de PointDistribution.sample(), que resetea
    # `color_` al inicio de cada muestreo: la etiqueta describe la muestra que
    # se acaba de devolver y nunca acumula ni arrastra la anterior.
    mix = _exact_mixture(seed=0)
    assert mix.color_ is None
    mix.sample(120)
    assert mix.color_.shape == (120,)
    mix.sample(30)
    assert mix.color_.shape == (30,)


def test_exact_mixture_label_matches_the_nearest_declared_mean():
    # 1.4: la etiqueta no es decorativa; identifica de qué componente salió cada
    # punto. Con modos bien separados (radio 5, escala 0.25) la componente más
    # cercana es la verdadera con probabilidad abrumadora.
    mix = _ring_mixture([0.5, 0.5], seed=9)
    x = mix.sample(400).astype(np.float64)
    dist = np.linalg.norm(x[:, None, :] - mix.means_[None, :, :], axis=-1)
    assert np.array_equal(dist.argmin(axis=1), mix.color_)


def test_exact_mixture_samples_are_grouped_by_component():
    # 1.4 — decisión de contrato: las muestras salen **agrupadas por
    # componente** (todos los puntos de la 0, después los de la 1, …), sin
    # permutación final, a diferencia de la legacy `mixture` (sklearn baraja) y
    # de `Spiral` (permuta explícitamente).
    #
    # Es aceptable acá y se fija a propósito: (a) la clase no entra al registry,
    # así que no puede llegar por YAML a una corrida con `shuffle: false`, que
    # es el único caso donde el orden se filtraría a los batches; (b) el consumo
    # natural es por `dataloader(..., shuffle=True)`, que reordena de todos
    # modos; (c) el oráculo analítico lee los parámetros, no el orden; y (d)
    # agrupado permite indexar los puntos de una componente sin filtrar,
    # justamente lo que necesitan las verificaciones por componente. Permutar
    # costaría un sorteo extra del rng —cambiando el mapa semilla → muestra— sin
    # agregar información.
    mix = _ring_mixture([0.5, 0.3, 0.2], seed=10)
    mix.sample(200)
    labels = mix.color_
    assert np.array_equal(labels, np.sort(labels))
    assert labels.tolist() == [0] * 100 + [1] * 60 + [2] * 40


def test_exact_mixture_per_component_moments_match_parameters():
    # 1.4: cada componente sale de aplicar la factorización de su covarianza a
    # ruido normal estándar, así que sus momentos empíricos reproducen la media
    # y la covarianza declaradas —incluida la correlación de la componente
    # rotada, que una escala solo diagonal perdería—. Tolerancia de Monte Carlo:
    # el reparto deja 7200 y 4800 puntos por componente, y el error estándar de
    # una entrada de la covarianza es ~sigma^2 * sqrt(2/n) <= 0.04 para las
    # varianzas de orden 1-2 de este caso, así que atol=0.2 son varios errores
    # estándar y además la semilla está fija.
    mix = _exact_mixture(seed=11)
    x = mix.sample(12_000)
    for i in range(2):
        mean_emp, cov_emp = _empirical_moments(x, mix.color_, i)
        assert np.allclose(mean_emp, mix.means_[i], atol=0.2), i
        assert np.allclose(cov_emp, mix.covariances_[i], atol=0.2), i
    # La segunda componente es la correlacionada: el test no pasaría si el
    # muestreo ignorara los términos fuera de la diagonal.
    assert mix.covariances_[1][0, 1] == pytest.approx(0.5)


def test_exact_mixture_sampling_does_not_touch_the_true_parameters():
    # 1.2, 1.4: el estado es inmutable tras la construcción — muestrear no
    # re-ajusta los parámetros (a diferencia de la estandarización empírica de
    # la clase base, que recalcula media y desvío en cada sample()).
    mix = _exact_mixture(seed=0)
    before = (mix.weights_, mix.means_, mix.covariances_)
    mix.sample(500)
    mix.sample(37)
    assert np.array_equal(mix.weights_, before[0])
    assert np.array_equal(mix.means_, before[1])
    assert np.array_equal(mix.covariances_, before[2])
    assert mix.mean_ is None and mix.std_ is None


# --------------------------------------------------------------------------
# ExactGaussianMixture: constructores de geometría del estudio (1.7).
#
# `two_modes` y `ring` son las palancas para armar las geometrías del barrido
# —separación entre modos, desbalance de pesos y anisotropía— sin que el
# llamador escriba matrices de covarianza a mano. Todo se verifica por la API
# pública (`weights_`, `means_`, `covariances_`): lo pedido tiene que ser
# exactamente lo que queda declarado.
#
# Convención de anisotropía fijada acá: `anisotropy` (κ) es la **razón entre el
# autovalor mayor y el menor** de una componente (κ=1 => isotrópica), y la
# **media geométrica** de los autovalores se mantiene en `scale**2`, así que
# estirar una componente no cambia su tamaño global (el determinante —el área de
# la elipse de covarianza— se conserva para cualquier κ).
# --------------------------------------------------------------------------


def _eigen_ratio(cov):
    """Razón entre el autovalor mayor y el menor de una covarianza 2x2."""
    eigvals = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))
    return float(eigvals[-1] / eigvals[0])


def _major_axis_alignment(cov, direction):
    """|cos| entre el eje mayor de ``cov`` y ``direction`` (1.0 => alineados).

    Se usa el valor absoluto porque un autovector define un eje, no un sentido:
    ``v`` y ``-v`` describen la misma elongación.
    """
    eigvals, eigvecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    major = eigvecs[:, int(np.argmax(eigvals))]
    d = np.asarray(direction, dtype=np.float64)
    return abs(float(major @ (d / np.linalg.norm(d))))


def _unit(angle):
    """Versor de dirección ``angle`` (radianes)."""
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


# ------------------------------------------------------------------ two_modes


def test_two_modes_reports_the_requested_separation():
    # 1.7: la palanca de separación se lee de vuelta en las medias declaradas,
    # que quedan simétricas respecto del origen.
    mix = ExactGaussianMixture.two_modes(separation=4.0, seed=0)
    assert isinstance(mix, ExactGaussianMixture)
    assert mix.dim == 2
    assert mix.means_.shape == (2, 2)
    assert float(np.linalg.norm(mix.means_[1] - mix.means_[0])) == pytest.approx(4.0)
    assert np.allclose(mix.means_.sum(axis=0), 0.0)


def test_two_modes_is_balanced_by_default():
    # 1.7: sin pedir desbalance, los dos modos pesan igual.
    mix = ExactGaussianMixture.two_modes(separation=3.0, seed=0)
    assert np.allclose(mix.weights_, [0.5, 0.5])


def test_two_modes_reports_the_requested_weight_imbalance():
    # 1.7: el desbalance es la segunda palanca del barrido y se declara tal cual.
    mix = ExactGaussianMixture.two_modes(separation=3.0, weights=(0.99, 0.01), seed=0)
    assert np.allclose(mix.weights_, [0.99, 0.01])
    assert float(mix.weights_.sum()) == pytest.approx(1.0)


def test_two_modes_separation_follows_the_requested_angle():
    # 1.7: la dirección de separación la fija `angle` (radianes); con angle=0 los
    # modos quedan sobre el eje x.
    horizontal = ExactGaussianMixture.two_modes(separation=2.0, seed=0)
    assert np.allclose(horizontal.means_, [[-1.0, 0.0], [1.0, 0.0]])

    angle = np.pi / 3.0
    oblique = ExactGaussianMixture.two_modes(separation=6.0, angle=angle, seed=0)
    delta = oblique.means_[1] - oblique.means_[0]
    assert np.allclose(delta / np.linalg.norm(delta), _unit(angle))
    assert np.allclose(oblique.means_, [-3.0 * _unit(angle), 3.0 * _unit(angle)])


def test_two_modes_is_isotropic_by_default():
    # 1.7: κ=1 (default) => covarianza esférica de escala `scale`, sin que el
    # llamador arme la matriz.
    mix = ExactGaussianMixture.two_modes(separation=4.0, scale=0.5, seed=0)
    for cov in mix.covariances_:
        assert np.allclose(cov, np.eye(2) * 0.25)
        assert _eigen_ratio(cov) == pytest.approx(1.0)


def test_two_modes_anisotropy_is_the_eigenvalue_ratio():
    # 1.7: κ es exactamente la razón entre el autovalor mayor y el menor, para
    # las dos componentes.
    mix = ExactGaussianMixture.two_modes(separation=4.0, anisotropy=9.0, seed=0)
    assert mix.covariances_.shape == (2, 2, 2)
    for cov in mix.covariances_:
        assert _eigen_ratio(cov) == pytest.approx(9.0)


def test_two_modes_anisotropy_preserves_the_overall_scale():
    # 1.7: la media geométrica de los autovalores se mantiene en scale**2, así
    # que subir κ estira la componente sin cambiar su tamaño global. El
    # determinante (= producto de autovalores = scale**4) es el mismo que en el
    # caso isotrópico: una convención que escalara solo el autovalor mayor lo
    # multiplicaría por κ y no pasaría este test.
    scale = 0.3
    iso = ExactGaussianMixture.two_modes(separation=4.0, scale=scale, seed=0)
    aniso = ExactGaussianMixture.two_modes(
        separation=4.0, scale=scale, anisotropy=16.0, seed=0
    )
    for cov in aniso.covariances_:
        eigvals = np.linalg.eigvalsh(cov)
        assert float(np.sqrt(eigvals.prod())) == pytest.approx(scale**2)
        assert float(np.linalg.det(cov)) == pytest.approx(
            float(np.linalg.det(iso.covariances_[0]))
        )


def test_two_modes_elongates_along_the_separation_direction():
    # 1.7: la elongación se aplica **a lo largo de la dirección de separación**,
    # que es la orientación que hace significativo el barrido de soporte casi
    # degenerado (los modos se estiran uno hacia el otro).
    angle = 0.4
    mix = ExactGaussianMixture.two_modes(
        separation=4.0, angle=angle, anisotropy=25.0, seed=0
    )
    for cov in mix.covariances_:
        assert _major_axis_alignment(cov, _unit(angle)) == pytest.approx(1.0)
        # Y el eje menor queda perpendicular: la varianza en la dirección normal
        # es la más chica.
        normal = _unit(angle + np.pi / 2.0)
        var_major = float(_unit(angle) @ cov @ _unit(angle))
        var_minor = float(normal @ cov @ normal)
        assert var_major / var_minor == pytest.approx(25.0)


def test_two_modes_covariance_is_rotated_for_an_oblique_angle():
    # 1.7, 1.3: con κ>1 y un ángulo oblicuo la covarianza deja de ser diagonal,
    # es decir la palanca produce componentes *rotadas* y no solo estiradas
    # sobre los ejes.
    mix = ExactGaussianMixture.two_modes(
        separation=4.0, angle=np.pi / 4.0, anisotropy=4.0, seed=0
    )
    for cov in mix.covariances_:
        assert not np.isclose(cov[0, 1], 0.0)
        assert np.allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_two_modes_accepts_zero_separation():
    # 1.7: separation=0 es el límite degenerado (los dos modos superpuestos) y se
    # acepta; solo se rechaza una separación negativa.
    mix = ExactGaussianMixture.two_modes(separation=0.0, seed=0)
    assert np.allclose(mix.means_, 0.0)


def test_two_modes_produces_a_sampleable_mixture():
    # 1.7 con 1.4/1.5: la geometría que arma el constructor pasa la validación
    # del constructor general y se puede muestrear; la composición respeta el
    # desbalance pedido y los momentos por componente reproducen lo declarado.
    mix = ExactGaussianMixture.two_modes(
        separation=6.0, weights=(0.75, 0.25), anisotropy=4.0, scale=0.4, seed=3
    )
    x = mix.sample(8000)
    assert x.shape == (8000, 2)
    assert x.dtype == np.float32
    assert np.bincount(mix.color_, minlength=2).tolist() == [6000, 2000]
    for i in range(2):
        mean_emp, cov_emp = _empirical_moments(x, mix.color_, i)
        assert np.allclose(mean_emp, mix.means_[i], atol=0.1), i
        assert np.allclose(cov_emp, mix.covariances_[i], atol=0.1), i
    # La semilla se propaga: dos construcciones iguales dan la misma muestra.
    other = ExactGaussianMixture.two_modes(
        separation=6.0, weights=(0.75, 0.25), anisotropy=4.0, scale=0.4, seed=3
    )
    assert np.array_equal(x, other.sample(8000))


@pytest.mark.parametrize(
    "overrides,culprit",
    [
        # Separación negativa: no es una distancia.
        ({"separation": -1.0}, "separation"),
        # κ < 1 invertiría el significado de la razón mayor/menor.
        ({"anisotropy": 0.5}, "anisotropy"),
        ({"anisotropy": 0.0}, "anisotropy"),
        # Escala no positiva: la covarianza no sería definida positiva.
        ({"scale": 0.0}, "scale"),
        ({"scale": -0.3}, "scale"),
        # `two_modes` son exactamente dos modos: los pesos tienen que ser dos.
        ({"weights": (0.5, 0.3, 0.2)}, "weights"),
        ({"weights": (1.0,)}, "weights"),
        # Pesos inválidos por sí mismos (delegado a la validación general, que
        # también nombra a weights).
        ({"weights": (0.6, 0.6)}, "weights"),
        ({"weights": (-0.1, 1.1)}, "weights"),
    ],
)
def test_two_modes_rejects_invalid_inputs(overrides, culprit):
    # 1.6, 1.7: cada entrada inválida nombra su parámetro culpable. El patrón va
    # anclado al principio del mensaje porque varios mensajes mencionan más de un
    # parámetro de pasada.
    kwargs = {"separation": 4.0, "seed": 0}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=f"^{culprit}"):
        ExactGaussianMixture.two_modes(**kwargs)


# ----------------------------------------------------------------------- ring


def test_ring_matches_the_legacy_angular_convention():
    # 1.7: las medias van equiespaciadas en un círculo de radio `radius`,
    # arrancando en ángulo 0, igual que la convención de la mixtura legacy
    # (radius * (cos 2πk/K, sin 2πk/K)); así las geometrías del estudio son
    # comparables con el trabajo anterior.
    k, radius = 8, 5.0
    mix = ExactGaussianMixture.ring(n_components=k, radius=radius, seed=0)
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    expected = radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    assert mix.means_.shape == (k, 2)
    assert np.allclose(mix.means_, expected)
    assert np.allclose(mix.means_[0], [radius, 0.0])
    assert np.allclose(np.linalg.norm(mix.means_, axis=1), radius)


def test_ring_defaults_to_uniform_weights():
    # 1.7: sin pesos explícitos el anillo es parejo, 1/K por componente.
    mix = ExactGaussianMixture.ring(n_components=5, seed=0)
    assert np.allclose(mix.weights_, 1.0 / 5.0)
    assert float(mix.weights_.sum()) == pytest.approx(1.0)


def test_ring_reports_explicit_weights_verbatim():
    # 1.7: el desbalance también es palanca en el anillo.
    weights = [0.4, 0.3, 0.2, 0.1]
    mix = ExactGaussianMixture.ring(n_components=4, weights=weights, seed=0)
    assert np.allclose(mix.weights_, weights)


def test_ring_is_isotropic_by_default():
    # 1.7: κ=1 (default) => todas las componentes esféricas de escala `scale`.
    mix = ExactGaussianMixture.ring(n_components=6, scale=0.3, seed=0)
    for cov in mix.covariances_:
        assert np.allclose(cov, np.eye(2) * 0.09)
        assert _eigen_ratio(cov) == pytest.approx(1.0)


def test_ring_anisotropy_is_the_eigenvalue_ratio_of_every_component():
    # 1.7: κ es la razón mayor/menor en **cada** componente, y la media
    # geométrica de los autovalores sigue siendo scale**2.
    scale, kappa = 0.3, 9.0
    mix = ExactGaussianMixture.ring(
        n_components=8, scale=scale, anisotropy=kappa, seed=0
    )
    assert mix.covariances_.shape == (8, 2, 2)
    for cov in mix.covariances_:
        assert _eigen_ratio(cov) == pytest.approx(kappa)
        assert float(np.sqrt(np.linalg.eigvalsh(cov).prod())) == pytest.approx(scale**2)


def test_ring_elongates_each_component_along_its_radial_direction():
    # 1.7 — decisión de orientación: cada componente se estira **en su dirección
    # radial** (alineada con su propio centro), que es la orientación que hace
    # significativo el barrido de soporte casi degenerado sobre un anillo (los
    # modos se estiran hacia el centro y hacia afuera, no de forma tangencial).
    k, kappa = 8, 16.0
    mix = ExactGaussianMixture.ring(n_components=k, anisotropy=kappa, seed=0)
    for i, cov in enumerate(mix.covariances_):
        radial = mix.means_[i] / np.linalg.norm(mix.means_[i])
        tangential = np.array([-radial[1], radial[0]])
        assert _major_axis_alignment(cov, radial) == pytest.approx(1.0), i
        var_radial = float(radial @ cov @ radial)
        var_tangential = float(tangential @ cov @ tangential)
        assert var_radial / var_tangential == pytest.approx(kappa), i


def test_ring_radial_orientation_comes_from_the_angle_not_from_the_mean():
    # 1.7: con radius=0 todas las medias caen en el origen y la dirección radial
    # del centro no está definida; la orientación se toma del ángulo de la
    # componente, así que sigue siendo la del anillo.
    k, kappa = 4, 9.0
    mix = ExactGaussianMixture.ring(
        n_components=k, radius=0.0, anisotropy=kappa, seed=0
    )
    assert np.allclose(mix.means_, 0.0)
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    for i, cov in enumerate(mix.covariances_):
        assert _major_axis_alignment(cov, _unit(ang[i])) == pytest.approx(1.0), i


def test_ring_with_a_single_component_is_a_lone_anisotropic_gaussian():
    # 1.7: K=1 es el borde del anillo: una sola componente en (radius, 0),
    # estirada sobre el eje x (su dirección radial).
    mix = ExactGaussianMixture.ring(
        n_components=1, radius=2.0, scale=0.5, anisotropy=4.0, seed=0
    )
    assert np.allclose(mix.weights_, [1.0])
    assert np.allclose(mix.means_, [[2.0, 0.0]])
    cov = mix.covariances_[0]
    assert np.allclose(cov, np.diag([0.25 * 2.0, 0.25 / 2.0]))


def test_ring_produces_a_sampleable_mixture():
    # 1.7 con 1.4/1.5: la geometría del anillo pasa la validación general y se
    # puede muestrear; la composición sigue los pesos y los momentos por
    # componente reproducen lo declarado (incluida la rotación de cada una).
    mix = ExactGaussianMixture.ring(
        n_components=4, radius=5.0, scale=0.4, anisotropy=4.0, seed=5
    )
    x = mix.sample(8000)
    assert x.shape == (8000, 2)
    assert x.dtype == np.float32
    assert np.bincount(mix.color_, minlength=4).tolist() == [2000] * 4
    for i in range(4):
        mean_emp, cov_emp = _empirical_moments(x, mix.color_, i)
        assert np.allclose(mean_emp, mix.means_[i], atol=0.1), i
        assert np.allclose(cov_emp, mix.covariances_[i], atol=0.1), i
    other = ExactGaussianMixture.ring(
        n_components=4, radius=5.0, scale=0.4, anisotropy=4.0, seed=5
    )
    assert np.array_equal(x, other.sample(8000))


@pytest.mark.parametrize(
    "overrides,culprit",
    [
        # Un anillo necesita al menos una componente.
        ({"n_components": 0}, "n_components"),
        ({"n_components": -3}, "n_components"),
        # Radio negativo: no es una distancia.
        ({"radius": -1.0}, "radius"),
        # Escala no positiva y κ < 1, igual que en two_modes.
        ({"scale": 0.0}, "scale"),
        ({"anisotropy": 0.5}, "anisotropy"),
        # Un peso por componente: la longitud tiene que coincidir con K. Sin
        # validación propia, el error lo tiraría `means` (que es quien queda mal
        # dimensionado frente al K que declaran los pesos) y culparía al
        # parámetro equivocado.
        ({"weights": [0.5, 0.5]}, "weights"),
        # Pesos inválidos por sí mismos (delegado a la validación general).
        ({"n_components": 2, "weights": [0.5, 0.4]}, "weights"),
        ({"n_components": 2, "weights": [-0.1, 1.1]}, "weights"),
    ],
)
def test_ring_rejects_invalid_inputs(overrides, culprit):
    # 1.6, 1.7: cada entrada inválida nombra su parámetro culpable, con el patrón
    # anclado al principio del mensaje.
    kwargs = {"n_components": 8, "seed": 0}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=f"^{culprit}"):
        ExactGaussianMixture.ring(**kwargs)


def test_geometry_constructors_do_not_need_hand_written_matrices():
    # 1.7: el objetivo de la palanca es que las tres cantidades del barrido
    # —separación, desbalance y anisotropía— se pidan por nombre y se lean de
    # vuelta por los accesores, sin que el llamador toque una matriz.
    two = ExactGaussianMixture.two_modes(
        separation=7.0, weights=(0.9, 0.1), anisotropy=6.0, seed=0
    )
    assert float(np.linalg.norm(two.means_[1] - two.means_[0])) == pytest.approx(7.0)
    assert float(two.weights_.max() / two.weights_.min()) == pytest.approx(9.0)
    assert _eigen_ratio(two.covariances_[0]) == pytest.approx(6.0)

    ring = ExactGaussianMixture.ring(
        n_components=3, radius=4.0, weights=[0.6, 0.3, 0.1], anisotropy=2.0, seed=0
    )
    assert np.allclose(np.linalg.norm(ring.means_, axis=1), 4.0)
    assert float(ring.weights_.max() / ring.weights_.min()) == pytest.approx(6.0)
    assert _eigen_ratio(ring.covariances_[2]) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# API pública del módulo de datos: la mixtura exacta se **exporta pero no se
# registra** (2.2, 2.4) y la mixtura **legacy** queda fijada contra valores de
# referencia (2.1, 2.3, 2.5).
#
# La forma exacta no entra al registry a propósito: sus parámetros
# (`weights`, `means`, `covariances`) son obligatorios y son matrices, y
# `make_distribution` **filtra** los kwargs que no matchean la firma en lugar de
# completarlos, así que por esa vía la clase no es construible. Se importa
# directamente desde la API pública, que es su único camino de uso.
# --------------------------------------------------------------------------

# Valores de referencia de la mixtura legacy con la configuración de la que
# dependen los notebooks y el checkpoint 2D: 8 componentes, cluster_std 0.3,
# radio 5.0 y seed=1. Se registraron desde la implementación vigente y sirven de
# ancla: cualquier cambio en el muestreo legacy (centros, escala de cluster,
# consumo del rng, estandarización) los mueve y hace fallar el test.
LEGACY_N = 1000
LEGACY_SEED = 1

# Tolerancia: las muestras son float32, así que el último dígito de los valores
# transcritos (6 decimales) no es representable exactamente; 1e-5 absoluto cubre
# ese redondeo y sigue siendo dos órdenes de magnitud más chico que cualquier
# cambio real de geometría.
LEGACY_ATOL = 1e-5

LEGACY_GOLDEN = {
    # standardize=True: la configuración exacta con la que se entrenó
    # `models/phase_1/vp_mixture.pt` y la que usan los notebooks.
    True: {
        "head": [
            [0.977878, -0.904543],
            [-0.006189, -1.402111],
            [-0.890129, 1.108003],
            [1.302082, -0.060786],
            [-0.080772, -1.281309],
            [0.907611, 1.028924],
        ],
        "tail": [0.951770, 1.012636],
        "abs_mean": 0.868430,
        "cube_mean": -0.001915,
        "norm_mean": 1.411808,
        "col_std": [1.000000, 1.000000],
    },
    # standardize=False: la geometría cruda, sin la transformación empírica, para
    # que el golden cubra también el anillo de radio 5 tal como sale.
    False: {
        "head": [
            [3.475602, -3.215929],
            [-0.017238, -4.985749],
            [-3.154688, 3.942581],
            [4.626329, -0.214732],
            [-0.281962, -4.556066],
            [3.226198, 3.661304],
        ],
        "tail": [3.382935, 3.603365],
        "abs_mean": 3.085660,
        "cube_mean": 0.031635,
        "norm_mean": 5.016388,
        "col_std": [3.549392, 3.556943],
    },
}

# Etiqueta de componente: el reparto de sklearn es parejo (125 puntos por
# componente con n=1000 y K=8) y el prefijo fija además el *orden* en que salen.
LEGACY_LABELS_HEAD = [7, 6, 3, 0, 6, 1, 2, 4, 7, 4, 2, 7]
LEGACY_LABEL_COUNTS = [125] * 8


def _legacy_mixture(*, standardize):
    """Mixtura legacy con la configuración de la que dependen notebooks y checkpoints."""
    return make_distribution(
        "mixture", 2, n_components=8, standardize=standardize, seed=LEGACY_SEED
    )


def test_exact_mixture_is_exported_from_the_public_api():
    # 2.4: la clase se importa por su nombre público desde el módulo de datos
    # (el import del encabezado ya lo ejerce) y figura en `__all__`, que es el
    # contrato de exportación del paquete.
    import diffusion.data_generation as data_generation

    assert "ExactGaussianMixture" in data_generation.__all__
    assert data_generation.ExactGaussianMixture is ExactGaussianMixture
    # Y lo exportado es usable, no un alias vacío.
    mix = ExactGaussianMixture(
        weights=[1.0], means=[[0.0, 0.0]], covariances=[np.eye(2)], seed=0
    )
    assert mix.sample(8).shape == (8, 2)


def test_exact_mixture_is_exported_but_not_registered():
    # 2.2, 2.4: exportada sí, registrada no. Con parámetros obligatorios que son
    # matrices, la factory (que filtra kwargs en vez de completarlos) no podría
    # construirla, y los tests genéricos parametrizados sobre el registry la
    # llamarían con `(name, dim, seed=...)` y nada más.
    assert ExactGaussianMixture.name == "mixture_exact"
    assert ExactGaussianMixture.name not in REGISTRY
    assert ExactGaussianMixture not in REGISTRY.values()
    assert ExactGaussianMixture.name not in available_shapes()
    with pytest.raises(ValueError, match="mixture_exact"):
        make_distribution("mixture_exact", 2)


def test_registry_keeps_exactly_the_historical_shapes():
    # 2.4: el conjunto de formas disponibles no cambia con esta feature. Se fija
    # la lista ordenada completa (no solo el conjunto) y que cada entrada siga
    # apuntando a la clase que declara ese nombre.
    assert available_shapes() == sorted(ALL_SHAPES)
    assert len(REGISTRY) == len(ALL_SHAPES)
    for name, cls in REGISTRY.items():
        assert cls.name == name


def test_legacy_mixture_is_still_constructible_by_name_with_its_defaults():
    # 2.2: la mixtura legacy sigue llegando por nombre desde el registro, con los
    # mismos parámetros y los mismos valores por defecto que antes de la feature.
    dist = make_distribution("mixture", 2)
    assert isinstance(dist, GaussianMixture)
    assert dist.name == "mixture"
    assert dist.dim == 2
    assert dist.n_components == 8
    assert dist.cluster_std == pytest.approx(0.3)
    assert dist.radius == pytest.approx(5.0)
    assert dist.standardize is False


def test_legacy_mixture_still_publishes_the_component_label():
    # 2.3: la etiqueta por punto que usan las previsualizaciones se sigue
    # publicando, con una etiqueta por punto y un índice de componente válido.
    dist = _legacy_mixture(standardize=True)
    x = dist.sample(LEGACY_N)
    assert dist.color_ is not None
    assert dist.color_.shape == (len(x),)
    assert dist.color_.dtype.kind in "iu"
    assert set(dist.color_.tolist()) == set(range(8))
    assert dist.color_[: len(LEGACY_LABELS_HEAD)].tolist() == LEGACY_LABELS_HEAD
    assert np.bincount(dist.color_, minlength=8).tolist() == LEGACY_LABEL_COUNTS


@pytest.mark.parametrize("standardize", [True, False])
def test_legacy_mixture_samples_match_the_recorded_golden_values(standardize):
    # 2.1, 2.5: las muestras de la configuración usada hasta ahora son
    # exactamente las de siempre. El golden combina muestras individuales (las
    # primeras filas y la última, que detectan cualquier corrimiento del sorteo)
    # con agregados de toda la muestra (momento absoluto, tercer momento, norma
    # media y desvío por eje), que detectan cambios que no toquen el prefijo.
    golden = LEGACY_GOLDEN[standardize]
    dist = _legacy_mixture(standardize=standardize)
    x = dist.sample(LEGACY_N)
    assert x.shape == (LEGACY_N, 2)
    assert x.dtype == np.float32

    head = np.asarray(golden["head"], dtype=np.float64)
    assert np.allclose(x[: len(head)], head, rtol=0.0, atol=LEGACY_ATOL)
    assert np.allclose(x[-1], golden["tail"], rtol=0.0, atol=LEGACY_ATOL)

    xd = x.astype(np.float64)
    assert float(np.abs(xd).mean()) == pytest.approx(golden["abs_mean"], abs=LEGACY_ATOL)
    assert float((xd**3).mean()) == pytest.approx(golden["cube_mean"], abs=LEGACY_ATOL)
    assert float(np.linalg.norm(xd, axis=1).mean()) == pytest.approx(
        golden["norm_mean"], abs=LEGACY_ATOL
    )
    assert np.allclose(xd.std(axis=0), golden["col_std"], rtol=0.0, atol=LEGACY_ATOL)


def test_cli_smoke(tmp_path):
    out = tmp_path / "d.npz"
    png = tmp_path / "d.png"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--shape", "two_moons", "--dim", "2",
         "--n-samples", "200", "--seed", "0", "--out", str(out), "--preview", str(png)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and png.exists()
    data = np.load(out)
    assert data["X"].shape == (200, 2)
    assert data["X"].dtype == np.float32
