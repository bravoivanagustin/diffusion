"""Tests de la verdad analítica del laboratorio 2D (`diffusion.analytic`)."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import diffusion.analytic
from diffusion.analytic.mixture_oracle import BiasReport, MixtureOracle
from diffusion.analytic.quadrature import QuadratureGrid, auto_grid, integrate
from diffusion.data_generation import ExactGaussianMixture, GaussianMixture
from diffusion.sde import ForwardSDE, make_sde


# --------------------------------------------------------------- paquete y fronteras


def test_paquete_importable_desde_la_ruta_publica():
    """El módulo se importa como `diffusion.analytic` y publica un `__all__`."""
    assert diffusion.analytic.__name__ == "diffusion.analytic"
    assert isinstance(diffusion.analytic.__all__, list)
    assert all(
        hasattr(diffusion.analytic, nombre) for nombre in diffusion.analytic.__all__
    )


def test_no_importa_samplers_ni_training():
    """Dirección de dependencias `data_generation → sde → analytic` (criterio 2.4).

    Se chequea en un intérprete nuevo porque `sys.modules` del proceso de pytest ya
    tiene cargados los módulos que importan las otras suites.
    """
    src = Path(diffusion.analytic.__file__).resolve().parents[2]
    codigo = (
        "import sys, diffusion.analytic; "
        "print([m for m in ('diffusion.samplers', 'diffusion.training') "
        "if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


# ------------------------------------------------------------------------ cuadratura 2D


def _identidad(escala: float) -> "torch.Tensor":
    """Covarianza isotrópica ``escala · I`` de forma ``(2, 2)``."""
    return escala * torch.eye(2, dtype=torch.float64)


def _diagonal(a: float, b: float) -> "torch.Tensor":
    """Covarianza diagonal ``diag(a, b)`` de forma ``(2, 2)``."""
    return torch.diag(torch.tensor([a, b], dtype=torch.float64))


def _rotada(a: float, b: float, angulo: float) -> "torch.Tensor":
    """Covarianza ``diag(a, b)`` rotada un ángulo dado (en radianes)."""
    c, s = torch.cos(torch.tensor(angulo)), torch.sin(torch.tensor(angulo))
    rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    return rot @ _diagonal(a, b) @ rot.T


def _gaussiana(media: "torch.Tensor", cov: "torch.Tensor"):
    """Devuelve la densidad gaussiana 2D de parámetros dados como callable de ``(N, 2)``."""
    import math

    def densidad(puntos: "torch.Tensor") -> "torch.Tensor":
        mu = media.to(puntos.dtype)
        precision = torch.linalg.inv(cov.to(puntos.dtype))
        dif = puntos - mu
        cuadratica = ((dif @ precision) * dif).sum(dim=-1)
        det = torch.linalg.det(cov.to(puntos.dtype))
        norma = 2.0 * math.pi * torch.sqrt(det)
        return torch.exp(-0.5 * cuadratica) / norma

    return densidad


def _mixtura(pesos, medias, covarianzas):
    """Devuelve la densidad de una mixtura de gaussianas como callable de ``(N, 2)``."""
    componentes = [_gaussiana(mu, cov) for mu, cov in zip(medias, covarianzas)]

    def densidad(puntos: "torch.Tensor") -> "torch.Tensor":
        total = torch.zeros(puntos.shape[0], dtype=puntos.dtype)
        for peso, componente in zip(pesos, componentes):
            total = total + peso * componente(puntos)
        return total

    return densidad


def test_la_cuadratura_no_importa_sde_ni_data_generation():
    """La cuadratura es una utilidad numérica autónoma: solo depende de ``torch``."""
    src = Path(diffusion.analytic.__file__).resolve().parents[2]
    codigo = (
        "import sys, diffusion.analytic.quadrature; "
        "print([m for m in ('diffusion.sde', 'diffusion.data_generation', "
        "'diffusion.samplers', 'diffusion.training') if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_integrar_gaussiana_isotropica_da_masa_uno():
    """El observable de la task: una gaussiana de covarianza conocida integra a uno."""
    media = torch.zeros(2, dtype=torch.float64)
    cov = _identidad(0.5)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...])
    assert integrate(_gaussiana(media, cov), grid) == pytest.approx(1.0, abs=1e-8)


def test_integrar_gaussiana_anisotropica_da_masa_uno():
    """La malla se adapta a una componente mucho más angosta en un eje."""
    media = torch.tensor([0.5, -0.5], dtype=torch.float64)
    cov = _diagonal(1.0, 0.02)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...])
    assert not grid.truncated
    assert integrate(_gaussiana(media, cov), grid) == pytest.approx(1.0, abs=1e-8)


def test_integrar_gaussiana_rotada_da_masa_uno():
    """Con covarianza rotada el paso lo fija el autovalor mínimo, no la diagonal."""
    import math

    media = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    cov = _rotada(1.0, 0.01, math.pi / 4)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...])
    assert not grid.truncated
    assert integrate(_gaussiana(media, cov), grid) == pytest.approx(1.0, abs=1e-8)


def test_integrar_mixtura_de_dos_componentes_da_masa_uno():
    """La malla cubre todas las medias a la vez, no solo una."""
    medias = torch.tensor([[-2.0, 0.0], [2.0, 1.0]], dtype=torch.float64)
    covarianzas = torch.stack([_identidad(0.3), _diagonal(0.5, 0.1)])
    grid = auto_grid(means=medias, covariances=covarianzas)
    densidad = _mixtura([0.3, 0.7], medias, covarianzas)
    assert integrate(densidad, grid) == pytest.approx(1.0, abs=1e-8)


def test_el_paso_lo_fija_la_componente_mas_angosta():
    """El paso sale de ``min_k sqrt(lambda_min)``, no de la componente ancha ni del promedio."""
    medias = torch.zeros(2, 2, dtype=torch.float64)
    covarianzas = torch.stack([_identidad(4.0), _identidad(0.01)])
    grid = auto_grid(means=medias, covariances=covarianzas, points_per_sigma=6.0)

    paso_angosta = 0.1 / 6.0  # sqrt(0.01) / points_per_sigma
    paso_ancha = 2.0 / 6.0
    paso_promedio = ((2.0 + 0.1) / 2.0) / 6.0

    assert grid.spacing <= paso_angosta
    assert grid.spacing == pytest.approx(paso_angosta, rel=1e-2)
    assert grid.spacing < paso_promedio / 2.0
    assert grid.spacing < paso_ancha / 2.0


def test_la_malla_no_colapsa_cuando_el_ruido_del_kernel_tiende_a_cero():
    """Piso de concentración: sumar un ruido diminuto no cambia la malla.

    Es la contracara de dimensionar por ``sigma_t``: si el paso saliera del desvío del
    kernel, con ``sigma_t = 1e-5`` la resolución pedida explotaría.
    """
    media = torch.zeros(1, 2, dtype=torch.float64)
    base = _identidad(0.25)[None, ...]
    sigma_t = 1e-5
    ruideada = base + (sigma_t**2) * torch.eye(2, dtype=torch.float64)[None, ...]

    grid_base = auto_grid(means=media, covariances=base)
    grid_ruideada = auto_grid(means=media, covariances=ruideada)

    assert grid_ruideada.n_points == grid_base.n_points
    assert not grid_ruideada.truncated


def test_prior_std_ensancha_el_dominio():
    """Integrar contra una distribución de referencia amplia extiende el dominio."""
    media = torch.zeros(1, 2, dtype=torch.float64)
    cov = _identidad(0.25)[None, ...]
    sin_prior = auto_grid(means=media, covariances=cov)
    con_prior = auto_grid(means=media, covariances=cov, prior_std=5.0, n_sigma=8.0)

    assert con_prior.half_width > sin_prior.half_width
    assert con_prior.half_width == pytest.approx(40.0)


def test_prior_std_chico_no_encoge_el_dominio():
    """El prior amplía el dominio, nunca lo recorta por debajo de las componentes."""
    media = torch.zeros(1, 2, dtype=torch.float64)
    cov = _identidad(1.0)[None, ...]
    sin_prior = auto_grid(means=media, covariances=cov)
    con_prior = auto_grid(means=media, covariances=cov, prior_std=0.01)
    assert con_prior.half_width == pytest.approx(sin_prior.half_width)


def test_el_dominio_cubre_las_medias_mas_varios_desvios():
    """``half_width`` = mayor ``|media| + n_sigma * sqrt(lambda_max)`` entre componentes."""
    medias = torch.tensor([[3.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    covarianzas = torch.stack([_identidad(1.0), _identidad(4.0)])
    grid = auto_grid(means=medias, covariances=covarianzas, n_sigma=4.0)
    # componente 0: 3 + 4*1 = 7 ; componente 1: 1 + 4*2 = 9
    assert grid.half_width == pytest.approx(9.0)


def test_resolucion_por_encima_del_tope_devuelve_malla_truncada():
    """Segundo observable: el tope de puntos queda marcado en la malla devuelta."""
    media = torch.zeros(1, 2, dtype=torch.float64)
    cov = _diagonal(1.0, 1e-4)[None, ...]
    grid = auto_grid(means=media, covariances=cov, max_points=64)
    assert grid.truncated
    assert grid.n_points == 64


def test_resolucion_dentro_del_tope_no_marca_la_malla():
    """Sin recorte, ``truncated`` queda en falso (su valor por defecto)."""
    media = torch.zeros(1, 2, dtype=torch.float64)
    cov = _identidad(1.0)[None, ...]
    grid = auto_grid(means=media, covariances=cov, max_points=4096)
    assert not grid.truncated
    assert grid.n_points < 4096


def test_una_malla_severamente_truncada_delata_la_perdida_de_masa():
    """La masa integrada es el detector de una malla insuficiente."""
    media = torch.zeros(2, dtype=torch.float64)
    cov = _diagonal(1.0, 1e-4)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...], max_points=8)
    assert grid.truncated
    assert integrate(_gaussiana(media, cov), grid) < 0.5


def test_spacing_consistente_con_half_width_y_n_points():
    """El paso es el ancho total dividido por la cantidad de intervalos."""
    grid = QuadratureGrid(half_width=4.0, n_points=9)
    assert grid.spacing == pytest.approx(8.0 / 8.0)
    assert not grid.truncated


def test_la_malla_es_inmutable():
    """``QuadratureGrid`` es un valor congelado: describe una malla, no la muta."""
    grid = QuadratureGrid(half_width=1.0, n_points=4)
    with pytest.raises((AttributeError, TypeError)):
        grid.n_points = 8  # type: ignore[misc]


def test_integrate_da_el_mismo_valor_sea_cual_sea_el_dtype_del_caller():
    """La suma corre en doble precisión aunque ``fn`` devuelva ``float32``."""
    media = torch.zeros(2, dtype=torch.float64)
    cov = _identidad(0.5)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...])
    densidad = _gaussiana(media, cov)

    def densidad_f32(puntos: "torch.Tensor") -> "torch.Tensor":
        return densidad(puntos).to(torch.float32)

    exacta = integrate(densidad, grid)
    degradada = integrate(densidad_f32, grid)
    assert exacta == pytest.approx(1.0, abs=1e-8)
    assert degradada == pytest.approx(exacta, abs=1e-5)
    assert isinstance(degradada, float)


def test_integrate_acumula_en_la_precision_pedida_y_no_en_la_del_caller():
    """La suma se acumula en doble precisión aunque los valores lleguen en ``float32``.

    Con decenas de miles de nodos, acumular en la precisión de ``fn`` deja un error
    relativo del orden de ``1e-7``; acumular en doble lo deja en el ruido de ``1e-15``.
    """
    grid = QuadratureGrid(half_width=1.0, n_points=201)
    valor_f32 = torch.tensor(1.0 / 3.0, dtype=torch.float32)

    def constante_f32(puntos: "torch.Tensor") -> "torch.Tensor":
        return valor_f32.expand(puntos.shape[0])

    esperado = grid.n_points**2 * float(valor_f32) * grid.spacing**2
    assert integrate(constante_f32, grid) == pytest.approx(esperado, rel=1e-12)


def test_integrate_pasa_los_puntos_en_el_dtype_pedido():
    """El dtype de la cuadratura lo fija ``integrate``, no el caller."""
    vistos: list["torch.dtype"] = []
    formas: list[tuple[int, ...]] = []

    def espia(puntos: "torch.Tensor") -> "torch.Tensor":
        vistos.append(puntos.dtype)
        formas.append(tuple(puntos.shape))
        return torch.zeros(puntos.shape[0], dtype=puntos.dtype)

    grid = QuadratureGrid(half_width=1.0, n_points=5)
    integrate(espia, grid)
    assert set(vistos) == {torch.float64}
    assert all(forma[1] == 2 for forma in formas)
    assert sum(forma[0] for forma in formas) == 25

    vistos.clear()
    integrate(espia, grid, dtype=torch.float32)
    assert set(vistos) == {torch.float32}


def test_integrate_de_una_constante_es_la_suma_de_riemann():
    """Suma de Riemann pura: ``sum(valores) * spacing**2``, sin pesos de trapecio."""
    grid = QuadratureGrid(half_width=1.0, n_points=101)
    area = integrate(
        lambda puntos: torch.ones(puntos.shape[0], dtype=puntos.dtype), grid
    )
    assert area == pytest.approx((101 * grid.spacing) ** 2, rel=1e-12)


# ------------------------------------------------- cuadratura 2D: entradas inválidas


def test_grid_rechaza_pocos_puntos():
    """Una malla necesita al menos dos puntos por eje para tener paso."""
    with pytest.raises(ValueError, match=r"^n_points"):
        QuadratureGrid(half_width=1.0, n_points=1)


@pytest.mark.parametrize("ancho", [0.0, -1.0, float("inf"), float("nan")])
def test_grid_rechaza_medio_ancho_invalido(ancho: float):
    """El medio ancho tiene que ser finito y positivo."""
    with pytest.raises(ValueError, match=r"^half_width"):
        QuadratureGrid(half_width=ancho, n_points=4)


def test_auto_grid_rechaza_medias_con_forma_equivocada():
    """Las medias son ``(K, 2)``: cualquier otra forma es un error del caller."""
    with pytest.raises(ValueError, match=r"^means"):
        auto_grid(
            means=torch.zeros(3, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
        )


def test_auto_grid_rechaza_covarianzas_con_forma_equivocada():
    """Las covarianzas son ``(K, 2, 2)``."""
    with pytest.raises(ValueError, match=r"^covariances"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0),
        )


def test_auto_grid_rechaza_cantidades_desparejas():
    """Tiene que haber una covarianza por media."""
    with pytest.raises(ValueError, match=r"^covariances"):
        auto_grid(
            means=torch.zeros(2, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
        )


def test_auto_grid_rechaza_covarianza_no_simetrica():
    """La precondición es covarianzas SPD."""
    cov = torch.tensor([[1.0, 0.5], [-0.5, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"^covariances"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=cov[None, ...],
        )


def test_auto_grid_rechaza_covarianza_singular():
    """Sin autovalor mínimo positivo no hay paso que resolver la componente."""
    cov = _diagonal(1.0, 0.0)
    with pytest.raises(ValueError, match=r"^covariances"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=cov[None, ...],
        )


def test_auto_grid_rechaza_medias_no_finitas():
    """Una media infinita haría un dominio infinito."""
    medias = torch.tensor([[0.0, float("nan")]], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"^means"):
        auto_grid(means=medias, covariances=_identidad(1.0)[None, ...])


@pytest.mark.parametrize("valor", [0.0, -1.0])
def test_auto_grid_rechaza_n_sigma_invalido(valor: float):
    """El dominio se mide en desvíos: el factor tiene que ser positivo."""
    with pytest.raises(ValueError, match=r"^n_sigma"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
            n_sigma=valor,
        )


@pytest.mark.parametrize("valor", [0.0, -3.0])
def test_auto_grid_rechaza_points_per_sigma_invalido(valor: float):
    """La resolución pedida se mide en puntos por desvío: tiene que ser positiva."""
    with pytest.raises(ValueError, match=r"^points_per_sigma"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
            points_per_sigma=valor,
        )


@pytest.mark.parametrize("valor", [1, 0, -4])
def test_auto_grid_rechaza_tope_de_puntos_invalido(valor: int):
    """El tope tiene que dejar espacio para una malla con paso."""
    with pytest.raises(ValueError, match=r"^max_points"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
            max_points=valor,
        )


@pytest.mark.parametrize("valor", [0.0, -2.0, float("inf")])
def test_auto_grid_rechaza_prior_std_invalido(valor: float):
    """La escala de la distribución de referencia tiene que ser finita y positiva."""
    with pytest.raises(ValueError, match=r"^prior_std"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
            prior_std=valor,
        )


def test_integrate_rechaza_dtype_no_flotante():
    """La cuadratura es en punto flotante; un dtype entero es un error del caller."""
    grid = QuadratureGrid(half_width=1.0, n_points=4)
    with pytest.raises(ValueError, match=r"^dtype"):
        integrate(
            lambda puntos: torch.zeros(puntos.shape[0]), grid, dtype=torch.int64
        )


def test_integrate_rechaza_fn_con_forma_de_salida_equivocada():
    """``fn`` recibe ``(N, 2)`` y devuelve ``(N,)``: cualquier otra forma se avisa."""
    grid = QuadratureGrid(half_width=1.0, n_points=4)
    with pytest.raises(ValueError, match=r"^fn"):
        integrate(lambda puntos: puntos, grid)


# ------------------------------- oráculo: admisibilidad y parámetros marginales


#: Las tres variantes escalar-gaussianas del Eje 1.
_SDES_ESCALARES = ["vp", "ve", "sub_vp"]

#: Defaults de las variantes registradas, replicados acá a propósito: la forma cerrada
#: esperada se escribe con ``math`` y sin importar ``diffusion.sde.schedules``, así que es
#: una **segunda implementación** y no una tautología contra el paquete.
_BETA_MIN, _BETA_MAX = 0.1, 20.0
_SIGMA_MIN, _SIGMA_MAX = 0.01, 5.0

#: Tiempos de referencia: el régimen casi singular, dos intermedios y el horizonte.
_TIEMPOS = [1e-4, 0.05, 0.5, 1.0]


def _beta_integral(t: float) -> float:
    """``∫_0^t beta(s) ds`` del schedule lineal de VP y sub-VP, escrito a mano."""
    return _BETA_MIN * t + 0.5 * (_BETA_MAX - _BETA_MIN) * t**2


def _alpha_sigma_cerrados(nombre: str, t: float) -> tuple[float, float]:
    """Forma cerrada de ``(alpha_t, sigma_t)`` por variante, calculada con ``math``.

    Convenciones que este helper fija de forma independiente:

    - VP: ``alpha = e^{-½∫beta}``, ``sigma = sqrt(1 - e^{-∫beta})``.
    - VE: ``alpha = 1`` (no contrae la media), ``sigma`` geométrico.
    - sub-VP: mismo ``alpha`` que VP y ``sigma = 1 - e^{-∫beta}`` **sin raíz**, que es la
      convención que publica el paquete y que el oráculo debe consumir tal cual.
    """
    if nombre == "vp":
        b = _beta_integral(t)
        return math.exp(-0.5 * b), math.sqrt(1.0 - math.exp(-b))
    if nombre == "ve":
        return 1.0, _SIGMA_MIN * (_SIGMA_MAX / _SIGMA_MIN) ** t
    if nombre == "sub_vp":
        b = _beta_integral(t)
        return math.exp(-0.5 * b), 1.0 - math.exp(-b)
    raise AssertionError(f"variante sin forma cerrada en el test: {nombre!r}")


def _mixtura_exacta() -> ExactGaussianMixture:
    """Mixtura exacta chica: pesos desbalanceados y una componente rotada y estirada."""
    return ExactGaussianMixture(
        2,
        weights=[0.3, 0.7],
        means=[[-1.5, 0.5], [2.0, -1.0]],
        covariances=[
            _rotada(1.0, 0.09, math.pi / 3).tolist(),
            _diagonal(0.25, 0.04).tolist(),
        ],
        seed=0,
    )


class _SDEEscalarInventada(ForwardSDE):
    """SDE escalar-gaussiana que no está en el registry.

    Existe para probar que el oráculo cubre la **familia** por un único camino de código y
    no las tres variantes conocidas: su kernel es ``N(alpha_t x_0, sigma_t² I)`` con
    ``alpha_t = 1 - 0.4 t`` y ``sigma_t = 0.2 + t``, que no coinciden con ningún schedule
    del paquete.
    """

    name = "escalar_inventada"

    def sde(self, x, t):  # pragma: no cover - el oráculo no usa los coeficientes
        return torch.zeros_like(x), self._expand_t(t, x)

    def marginal_prob(self, x0, t):
        tt = self._expand_t(t, x0)
        return (1.0 - 0.4 * tt) * x0, 0.2 + tt

    def prior_sampling(self, shape, **kwargs):  # pragma: no cover - fuera de alcance
        return torch.zeros(shape)


class _SDEContadora(_SDEEscalarInventada):
    """Como :class:`_SDEEscalarInventada` pero cuenta las consultas al contrato marginal."""

    def __init__(self) -> None:
        super().__init__()
        self.consultas = 0

    def marginal_prob(self, x0, t):
        self.consultas += 1
        return super().marginal_prob(x0, t)


class _SDEDiagonal(_SDEEscalarInventada):
    """SDE fuera de la familia: escala **cada coordenada** por un factor distinto.

    Su desvío sí colapsa a un escalar por muestra, así que lo único que la delata es la
    verificación de proporcionalidad de la media.
    """

    name = "diagonal_falsa"

    def marginal_prob(self, x0, t):
        tt = self._expand_t(t, x0)
        escala = torch.tensor([1.0, 0.5], dtype=x0.dtype, device=x0.device)
        return (1.0 - 0.4 * tt) * x0 * escala, 0.2 + tt


class _SDEMezcladora(_SDEEscalarInventada):
    """SDE fuera de la familia: su media **promedia** las dos coordenadas del dato.

    Es el caso que justifica sondear con coordenadas distintas entre sí: con un vector de
    unos, ``mean = alpha_t · (x + y)/2`` da exactamente ``alpha_t · x_0`` y el rechazo se
    perdería. Con ``(1, 2)`` la media sale ``(1.5, 1.5)``, que no es proporcional a ``(1, 2)``.
    """

    name = "mezcladora_falsa"

    def marginal_prob(self, x0, t):
        tt = self._expand_t(t, x0)
        promedio = x0.mean(dim=-1, keepdim=True).expand_as(x0)
        return (1.0 - 0.4 * tt) * promedio, 0.2 + tt


class _SDEPorDimension(_SDEEscalarInventada):
    """SDE fuera de la familia: su desvío es **por coordenada**, no por muestra."""

    name = "por_dimension_falsa"

    def marginal_prob(self, x0, t):
        tt = self._expand_t(t, x0)
        escala = torch.tensor([1.0, 3.0], dtype=x0.dtype, device=x0.device)
        return x0, (0.2 + tt) * escala


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_marginal_params_coincide_con_la_forma_cerrada_de_cada_variante(nombre: str):
    """Un solo camino de código da ``alpha`` y ``sigma`` en las tres variantes escalares.

    Se contrasta contra la forma cerrada escrita a mano en el test, no contra el paquete.

    Tolerancia: ``1e-9`` es holgadísima frente al ruido de la doble precisión, pero contempla
    la cancelación catastrófica de ``1 - e^{-∫beta}`` cuando ``t`` es diminuto (en
    ``t = 1e-4`` el desvío de sub-VP vale ``1e-5`` y la resta pierde unos cinco dígitos, así
    que ``math.exp`` y ``torch.exp`` pueden separarse ``~1e-11`` con solo diferir en un ulp).
    Cualquier convención equivocada —tomar una raíz, re-derivar el schedule— se aparta
    órdenes de magnitud, así que el test no pierde poder discriminante.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    t = torch.tensor(_TIEMPOS, dtype=torch.float64)

    alpha, sigma = oraculo.marginal_params(t)

    assert alpha.shape == (len(_TIEMPOS), 1)
    assert sigma.shape == (len(_TIEMPOS), 1)
    for i, valor in enumerate(_TIEMPOS):
        a, s = _alpha_sigma_cerrados(nombre, valor)
        assert float(alpha[i, 0]) == pytest.approx(a, rel=1e-9)
        assert float(sigma[i, 0]) == pytest.approx(s, rel=1e-9)


def test_ve_no_contrae_la_media():
    """En VE el kernel deja el dato en su lugar: ``alpha_t == 1`` para todo ``t``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("ve"))
    t = torch.tensor(_TIEMPOS, dtype=torch.float64)
    alpha, _ = oraculo.marginal_params(t)
    assert torch.equal(alpha, torch.ones_like(alpha))


def test_el_desvio_de_sub_vp_se_consume_sin_raiz():
    """sub-VP publica ``std = 1 - e^{-∫beta}`` (sin raíz) y el oráculo lo toma tal cual.

    Es el test que discrimina la "corrección" plausible: con ``t = 0.05`` el desvío vale
    ``0.0294`` y su raíz ``0.1716``, casi seis veces más, así que tomar la raíz no puede
    pasar inadvertido.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("sub_vp"))
    _, sigma = oraculo.marginal_params(torch.tensor([0.05], dtype=torch.float64))

    esperado = 1.0 - math.exp(-_beta_integral(0.05))
    assert float(sigma[0, 0]) == pytest.approx(esperado, rel=1e-12)
    assert float(sigma[0, 0]) < math.sqrt(esperado) / 4.0


def test_el_desvio_de_sub_vp_queda_por_debajo_del_de_vp():
    """Invariante de la variante: su varianza es estrictamente menor que la de VP."""
    t = torch.tensor([0.05, 0.5, 1.0], dtype=torch.float64)
    mixtura = _mixtura_exacta()
    _, sigma_vp = MixtureOracle(mixtura, make_sde("vp")).marginal_params(t)
    _, sigma_sub = MixtureOracle(mixtura, make_sde("sub_vp")).marginal_params(t)
    assert bool((sigma_sub < sigma_vp).all())


def test_una_sde_escalar_gaussiana_nueva_pasa_por_el_mismo_camino():
    """Sin ramificación por variante: anda una SDE de la familia ajena al registry."""
    oraculo = MixtureOracle(_mixtura_exacta(), _SDEEscalarInventada())
    t = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)

    alpha, sigma = oraculo.marginal_params(t)

    esperado_alpha = torch.tensor([[1.0], [0.88], [0.6]], dtype=torch.float64)
    esperado_sigma = torch.tensor([[0.2], [0.5], [1.2]], dtype=torch.float64)
    assert torch.allclose(alpha, esperado_alpha, rtol=1e-12, atol=1e-15)
    assert torch.allclose(sigma, esperado_sigma, rtol=1e-12, atol=1e-15)


def test_los_parametros_marginales_no_heredan_la_precision_del_caller():
    """Los pasos intermedios corren en doble aunque ``t`` llegue en simple precisión.

    Los tiempos elegidos son exactos en ``float32``, así que la conversión no aporta error:
    la única fuente posible de un error relativo ``~1e-7`` sería evaluar el schedule en
    simple precisión.
    """
    tiempos = [0.25, 0.5, 1.0]
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))

    alpha, sigma = oraculo.marginal_params(torch.tensor(tiempos, dtype=torch.float32))

    assert alpha.dtype is torch.float64
    assert sigma.dtype is torch.float64
    for i, valor in enumerate(tiempos):
        a, s = _alpha_sigma_cerrados("vp", valor)
        assert float(alpha[i, 0]) == pytest.approx(a, rel=1e-12)
        assert float(sigma[i, 0]) == pytest.approx(s, rel=1e-12)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_tiempo_plano_y_en_columna_dan_el_mismo_resultado(nombre: str):
    """Contrato de shapes del proyecto: ``t`` se acepta como ``(B,)`` y como ``(B, 1)``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    plano = torch.tensor([0.05, 0.7], dtype=torch.float64)
    columna = plano.reshape(-1, 1)

    alpha_plano, sigma_plano = oraculo.marginal_params(plano)
    alpha_columna, sigma_columna = oraculo.marginal_params(columna)

    assert torch.equal(alpha_plano, alpha_columna)
    assert torch.equal(sigma_plano, sigma_columna)
    assert torch.equal(
        oraculo.component_covariances(plano),
        oraculo.component_covariances(columna),
    )


def test_component_covariances_es_alpha2_por_sigma_por_componente():
    """``Sigma_k(t) = alpha_t² Sigma_k + sigma_t² I``, con la componente rotada incluida."""
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    tiempos = [0.05, 0.9]
    t = torch.tensor(tiempos, dtype=torch.float64)

    covs = oraculo.component_covariances(t)

    assert covs.shape == (2, 2, 2, 2)
    assert covs.dtype is torch.float64
    bases = torch.as_tensor(mixtura.covariances_, dtype=torch.float64)
    identidad = torch.eye(2, dtype=torch.float64)
    for i, valor in enumerate(tiempos):
        a, s = _alpha_sigma_cerrados("vp", valor)
        esperado = a**2 * bases + s**2 * identidad
        assert torch.allclose(covs[i], esperado, rtol=1e-10, atol=1e-14)


def test_component_covariances_conserva_la_rotacion_de_la_componente():
    """Sumar ``sigma_t² I`` no diagonaliza: la componente rotada sigue correlacionada.

    Discrimina la versión que se queda con la diagonal de ``Sigma_k`` en lugar de la matriz
    completa, que daría un término fuera de la diagonal nulo.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("ve"))
    covs = oraculo.component_covariances(torch.tensor([0.3], dtype=torch.float64))

    fuera_de_diagonal = float(covs[0, 0, 0, 1])
    assert abs(fuera_de_diagonal) > 0.1
    assert float(covs[0, 0, 1, 0]) == pytest.approx(fuera_de_diagonal, rel=1e-12)


def test_component_covariances_devuelve_un_tensor_nuevo_en_cada_llamada():
    """Invariante de pureza: escribir en la salida no contamina las llamadas siguientes."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    t = torch.tensor([0.4], dtype=torch.float64)

    primera = oraculo.component_covariances(t).clone()
    contaminada = oraculo.component_covariances(t)
    contaminada[...] = 0.0

    assert torch.equal(oraculo.component_covariances(t), primera)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_component_covariances_es_simetrica_y_definida_positiva(nombre: str):
    """``Sigma_k(t)`` hereda la definición positiva de la mixtura para todo ``t``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    covs = oraculo.component_covariances(torch.tensor(_TIEMPOS, dtype=torch.float64))

    assert torch.allclose(covs, covs.transpose(-1, -2), rtol=0.0, atol=1e-15)
    autovalores = torch.linalg.eigvalsh(covs)
    assert bool((autovalores > 0.0).all())


# ---------------------------- oráculo: inversa y determinante de forma cerrada


def test_la_forma_cerrada_2x2_coincide_con_el_algebra_lineal_generica():
    """La inversa y el determinante 2×2 se calculan a mano; acá se contrastan con el solver.

    El solver genérico es la fuente independiente: la producción **no** lo usa, justamente
    porque en 2×2 la adjunta sobre el determinante es exacta y barata.
    """
    from diffusion.analytic.mixture_oracle import _inverse_and_det_2x2

    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("sub_vp"))
    covs = oraculo.component_covariances(torch.tensor([0.05, 0.8], dtype=torch.float64))

    inv, det = _inverse_and_det_2x2(covs)

    assert inv.shape == covs.shape
    assert det.shape == covs.shape[:-2]
    assert torch.allclose(inv, torch.linalg.inv(covs), rtol=1e-10, atol=1e-14)
    assert torch.allclose(det, torch.linalg.det(covs), rtol=1e-10, atol=1e-14)
    producto = inv @ covs
    esperado = torch.eye(2, dtype=torch.float64).expand_as(producto)
    assert torch.allclose(producto, esperado, rtol=0.0, atol=1e-10)


def test_la_forma_cerrada_2x2_no_transpone_la_matriz():
    """Con una matriz **no simétrica** un error de signo o de transposición se ve.

    Las covarianzas son simétricas, así que confundir la adjunta con su transpuesta pasaría
    inadvertido si solo se probara con ellas.
    """
    from diffusion.analytic.mixture_oracle import _inverse_and_det_2x2

    m = torch.tensor([[[1.0, 2.0], [0.5, 3.0]]], dtype=torch.float64)

    inv, det = _inverse_and_det_2x2(m)

    assert float(det[0]) == pytest.approx(1.0 * 3.0 - 2.0 * 0.5, rel=1e-15)
    assert torch.allclose(inv, torch.linalg.inv(m), rtol=1e-12, atol=1e-15)


def test_la_forma_cerrada_2x2_rechaza_una_matriz_con_determinante_no_positivo():
    """Sin determinante positivo no hay covarianza que invertir: se avisa, no se da NaN."""
    from diffusion.analytic.mixture_oracle import _inverse_and_det_2x2

    singular = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"^cov"):
        _inverse_and_det_2x2(singular)


def test_la_forma_cerrada_2x2_rechaza_una_forma_que_no_es_2x2():
    """La forma cerrada es específica de 2×2: cualquier otra forma es un error del caller."""
    from diffusion.analytic.mixture_oracle import _inverse_and_det_2x2

    with pytest.raises(ValueError, match=r"^cov"):
        _inverse_and_det_2x2(torch.eye(3, dtype=torch.float64))


# ------------------------------------ oráculo: rechazos en construcción (5.1, 5.2)


def test_rechaza_una_mixtura_legacy_sin_parametros_exactos():
    """La mixtura legacy no publica pesos, medias ni covarianzas verdaderos (criterio 5.1)."""
    with pytest.raises(ValueError, match=r"^mixture debe ser una ExactGaussianMixture"):
        MixtureOracle(GaussianMixture(2, seed=1), make_sde("vp"))


def test_rechaza_una_distribucion_que_no_es_una_mixtura():
    """Cualquier otra forma de datos tampoco tiene parámetros en forma cerrada."""
    from diffusion.data_generation import make_distribution

    with pytest.raises(ValueError, match=r"^mixture debe ser una ExactGaussianMixture"):
        MixtureOracle(make_distribution("two_moons", 2, seed=0), make_sde("vp"))


def test_rechaza_la_estandarizacion_empirica_de_la_mixtura_legacy():
    """El caso real (criterio 5.2): la config legacy del laboratorio usa ``standardize=True``.

    El rechazo por estandarización se evalúa **antes** que el de exactitud, así que el
    mensaje que ve el autor explica lo que de verdad rompe la exactitud: la transformación
    se estimó del sorteo.
    """
    legacy = GaussianMixture(2, standardize=True, seed=1)
    with pytest.raises(ValueError, match=r"^mixture no puede usar la estandarización") as info:
        MixtureOracle(legacy, make_sde("vp"))
    assert "sorteo" in str(info.value)


def test_rechaza_una_mixtura_exacta_a_la_que_le_prendieron_la_estandarizacion():
    """El flag se consulta en el objeto recibido, no se asume por su tipo."""
    mixtura = _mixtura_exacta()
    mixtura.standardize = True
    with pytest.raises(ValueError, match=r"^mixture no puede usar la estandarización"):
        MixtureOracle(mixtura, make_sde("vp"))


def test_rechaza_una_sde_cuya_media_no_es_proporcional_al_dato():
    """Familia escalar-gaussiana: ``mean == alpha_t · x_0`` con un escalar por muestra."""
    with pytest.raises(NotImplementedError, match=r"^sde no pertenece") as info:
        MixtureOracle(_mixtura_exacta(), _SDEDiagonal())
    assert "proporcional" in str(info.value)


def test_rechaza_una_sde_cuya_media_mezcla_las_coordenadas():
    """El sondeo usa coordenadas distintas entre sí, así que un kernel que las promedia cae.

    Con un vector de unos este caso pasaría el chequeo de proporcionalidad sin ser de la
    familia: el test le pone dientes a la elección de los puntos de sondeo.
    """
    with pytest.raises(NotImplementedError, match=r"^sde no pertenece") as info:
        MixtureOracle(_mixtura_exacta(), _SDEMezcladora())
    assert "proporcional" in str(info.value)


def test_rechaza_una_sde_cuyo_desvio_no_colapsa_a_un_escalar_por_muestra():
    """El desvío por coordenada delata un kernel que no es ``N(alpha x_0, sigma² I)``."""
    with pytest.raises(NotImplementedError, match=r"^sde no pertenece") as info:
        MixtureOracle(_mixtura_exacta(), _SDEPorDimension())
    assert "escalar por muestra" in str(info.value)


def test_la_pertenencia_a_la_familia_no_se_consulta_por_un_atributo():
    """El chequeo es **estructural**: ``is_augmented`` ya no existe en el paquete.

    Guard de compatibilidad hacia adelante: ninguna de las tres SDEs registradas cae en el
    caso, así que la única forma de verificarlo es con una SDE inventada, y la única forma de
    detectarla es mirando lo que devuelve el contrato marginal.
    """
    for sde in (_SDEDiagonal(), _SDEPorDimension(), make_sde("vp")):
        assert not hasattr(sde, "is_augmented")


@pytest.mark.parametrize(
    "caso",
    [
        "mixtura_legacy",
        "mixtura_estandarizada",
        "sde_diagonal",
        "sde_mezcladora",
        "sde_por_dimension",
    ],
)
def test_las_entradas_inadmisibles_fallan_en_construccion_y_no_al_integrar(caso: str):
    """Fail-fast: ninguna corrida "exacta" arranca sobre premisas que no lo son.

    La lista ``construidos`` es lo que le da dientes al test: con validación diferida el
    constructor devolvería un oráculo, no habría excepción y ``pytest.raises`` fallaría.
    """
    mixtura: object = _mixtura_exacta()
    sde: ForwardSDE = make_sde("vp")
    if caso == "mixtura_legacy":
        mixtura = GaussianMixture(2, seed=1)
    elif caso == "mixtura_estandarizada":
        mixtura = GaussianMixture(2, standardize=True, seed=1)
    elif caso == "sde_diagonal":
        sde = _SDEDiagonal()
    elif caso == "sde_mezcladora":
        sde = _SDEMezcladora()
    else:
        sde = _SDEPorDimension()

    construidos: list[MixtureOracle] = []
    with pytest.raises((ValueError, NotImplementedError)):
        construidos.append(MixtureOracle(mixtura, sde))  # type: ignore[arg-type]

    assert construidos == []


def test_la_validacion_de_familia_consulta_el_contrato_marginal_en_construccion():
    """Contracara del fail-fast: el sondeo estructural ya corrió al construir el oráculo."""
    sde = _SDEContadora()
    MixtureOracle(_mixtura_exacta(), sde)
    assert sde.consultas >= 1


# --------------------------------------- oráculo: validación de la entrada de tiempo


@pytest.mark.parametrize("metodo", ["marginal_params", "component_covariances"])
@pytest.mark.parametrize(
    "malo",
    [
        torch.zeros((), dtype=torch.float64),
        torch.zeros(3, 2, dtype=torch.float64),
        torch.zeros(2, 1, 1, dtype=torch.float64),
        torch.zeros(0, dtype=torch.float64),
    ],
)
def test_rechaza_un_tiempo_con_forma_invalida(metodo: str, malo: "torch.Tensor"):
    """``t`` es un valor por punto: ``(B,)`` o ``(B, 1)`` con ``B >= 1``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        getattr(oraculo, metodo)(malo)


def test_rechaza_un_tiempo_que_no_es_tensor():
    """El contrato del proyecto es tensorial; un float suelto es un error del caller."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        oraculo.marginal_params(0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("valor", [-1e-6, float("nan"), float("inf")])
def test_rechaza_un_tiempo_negativo_o_no_finito(valor: float):
    """El proceso corre en ``[0, T]``: fuera de ahí el schedule no significa nada."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        oraculo.marginal_params(torch.tensor([valor], dtype=torch.float64))


# ------------------------------- oráculo: densidad, log-densidad y masa integrada


#: Puntos de evaluación: el origen, los dos modos y dos puntos de cola.
_PUNTOS = torch.tensor(
    [[0.0, 0.0], [-1.5, 0.5], [2.0, -1.0], [1.0, 3.0], [-3.0, -2.0]],
    dtype=torch.float64,
)


def _tiempo(valor: float, batch: int) -> "torch.Tensor":
    """Vector ``(batch,)`` con el mismo tiempo en todas las filas."""
    return torch.full((batch,), valor, dtype=torch.float64)


def _log_gaussiana_cerrada(
    punto: tuple[float, float],
    media: tuple[float, float],
    cov: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Log-densidad gaussiana 2D escrita con floats de Python.

    Tercera implementación independiente —``_gaussiana`` trabaja en el dominio lineal con el
    álgebra lineal genérica de torch, y la producción usa su propia forma cerrada—. Sirve
    justamente en el régimen en el que la densidad **no** es representable y el dominio
    lineal ya no alcanza.
    """
    dx = punto[0] - media[0]
    dy = punto[1] - media[1]
    (a, b), (c, d) = cov[0], cov[1]
    det = a * d - b * c
    cuadratica = (dx * (d * dx - b * dy) + dy * (-c * dx + a * dy)) / det
    return -math.log(2.0 * math.pi) - 0.5 * math.log(det) - 0.5 * cuadratica


def _params_ruideados(
    nombre: str, t: float, mixtura: ExactGaussianMixture
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """``(alpha_t mu_k, Sigma_k(t))`` armados con la forma cerrada escrita a mano en el test."""
    a, s = _alpha_sigma_cerrados(nombre, t)
    base = torch.as_tensor(mixtura.covariances_, dtype=torch.float64)
    medias = a * torch.as_tensor(mixtura.means_, dtype=torch.float64)
    covs = a**2 * base + (s**2) * torch.eye(2, dtype=torch.float64)
    return medias, covs


def _log_mixtura_independiente(
    mixtura: ExactGaussianMixture,
    medias: "torch.Tensor",
    covs: "torch.Tensor",
    puntos: "torch.Tensor",
) -> "torch.Tensor":
    """Log-densidad de referencia: mixtura sumada en el dominio **lineal** y recién ahí log.

    Es la fuente independiente del criterio 3.1: no usa ``logsumexp`` ni la forma cerrada
    2×2 de la producción, sino ``torch.linalg.inv``/``det`` sobre cada componente.
    """
    densidad = _mixtura(mixtura.weights_.tolist(), medias, covs)
    return torch.log(densidad(puntos))


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_la_log_densidad_coincide_con_la_formula_independiente(nombre: str):
    """Criterio 3.1/3.2: ``log p_t`` contra una mixtura gaussiana escrita aparte.

    Tolerancia: ``rtol=1e-10`` sobre log-densidades de magnitud hasta ~50, es decir unos
    ``5e-9`` absolutos, varios órdenes por encima del ruido de la doble precisión y muy por
    debajo de cualquier error estructural (olvidar los pesos, usar ``Sigma_k`` en lugar de
    ``Sigma_k(t)``, perder el factor de contracción), que se aparta en unidades enteras.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde(nombre))

    for valor in _TIEMPOS:
        medias, covs = _params_ruideados(nombre, valor, mixtura)
        esperada = _log_mixtura_independiente(mixtura, medias, covs, _PUNTOS)
        obtenida = oraculo.log_prob(_PUNTOS, _tiempo(valor, _PUNTOS.shape[0]))

        assert obtenida.shape == (_PUNTOS.shape[0],)
        assert torch.allclose(obtenida, esperada, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_la_densidad_coincide_con_la_formula_independiente(nombre: str):
    """La densidad es la de la misma mixtura ruideada, no la de otra cosa (criterio 3.1)."""
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde(nombre))

    for valor in [0.05, 0.5]:
        medias, covs = _params_ruideados(nombre, valor, mixtura)
        esperada = _mixtura(mixtura.weights_.tolist(), medias, covs)(_PUNTOS)
        obtenida = oraculo.prob(_PUNTOS, _tiempo(valor, _PUNTOS.shape[0]))
        assert torch.allclose(obtenida, esperada, rtol=1e-10, atol=1e-300)


def test_la_log_densidad_sigue_siendo_finita_donde_la_densidad_ya_no_se_representa():
    """Criterio 3.2: el dominio logarítmico es lo que salva el régimen no representable.

    El punto está tan lejos de los dos modos que **cada** componente vale ``exp(-5e5)``, es
    decir cero en doble precisión: sumar densidades y recién ahí tomar el logaritmo da
    ``-inf``. El test fija las dos caras —que el resultado es finito y coincide con la
    componente dominante, y que el camino ingenuo de verdad desborda—, así que una
    implementación que sume en el dominio lineal falla acá.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    punto = (200.0, 200.0)
    valor = 1e-4

    medias, covs = _params_ruideados("vp", valor, mixtura)
    logs = [
        math.log(peso)
        + _log_gaussiana_cerrada(
            punto,
            (float(medias[k, 0]), float(medias[k, 1])),
            (
                (float(covs[k, 0, 0]), float(covs[k, 0, 1])),
                (float(covs[k, 1, 0]), float(covs[k, 1, 1])),
            ),
        )
        for k, peso in enumerate(mixtura.weights_.tolist())
    ]
    # El camino ingenuo: exponenciar cada componente y sumar. Desborda por abajo a cero.
    assert sum(math.exp(log_componente) for log_componente in logs) == 0.0

    obtenida = oraculo.log_prob(
        torch.tensor([punto], dtype=torch.float64), _tiempo(valor, 1)
    )
    assert torch.isfinite(obtenida).all()
    assert float(obtenida[0]) == pytest.approx(max(logs), rel=1e-12)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
@pytest.mark.parametrize("valor", [1e-5, 1e-4, 1e-3])
def test_la_log_densidad_es_finita_con_el_desvio_del_kernel_diminuto(
    nombre: str, valor: float
):
    """Criterio 4.7: con ``sigma_t`` de ``1e-5`` o menos no hay desbordes ni indeterminaciones.

    sub-VP llega a ``sigma_t ≈ 1e-6`` en el extremo del barrido. La densidad **no** explota,
    porque la concentración tiene piso ``alpha_t² Sigma_k``; lo que se verifica es que ni la
    log-densidad ni la densidad producen ``nan`` ni ``inf``.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    t = _tiempo(valor, _PUNTOS.shape[0])

    log_densidad = oraculo.log_prob(_PUNTOS, t)
    densidad = oraculo.prob(_PUNTOS, t)

    assert torch.isfinite(log_densidad).all()
    assert torch.isfinite(densidad).all()
    assert bool((densidad >= 0.0).all())


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_la_densidad_es_la_exponencial_de_la_log_densidad(nombre: str):
    """Las dos cantidades son la misma: la densidad se deriva de la log-densidad."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    t = _tiempo(0.2, _PUNTOS.shape[0])

    log_densidad = oraculo.log_prob(_PUNTOS, t)
    densidad = oraculo.prob(_PUNTOS, t)

    assert torch.allclose(densidad, torch.exp(log_densidad), rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_tiempo_plano_y_en_columna_dan_la_misma_densidad(nombre: str):
    """Criterio 3.5: ``t`` se acepta como ``(B,)`` y como ``(B, 1)``, con un valor por punto."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    plano = _tiempo(0.35, _PUNTOS.shape[0])
    columna = plano.reshape(-1, 1)

    assert torch.equal(
        oraculo.log_prob(_PUNTOS, plano), oraculo.log_prob(_PUNTOS, columna)
    )
    assert torch.equal(oraculo.prob(_PUNTOS, plano), oraculo.prob(_PUNTOS, columna))


def test_cada_punto_se_evalua_en_su_propio_tiempo():
    """Un valor por punto significa eso: el lote no comparte un tiempo único.

    Discrimina la versión que toma el primer tiempo del lote (o su promedio) y lo aplica a
    todas las filas: acá los dos tiempos son extremos del horizonte.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    punto = torch.tensor([[0.4, -0.3]], dtype=torch.float64)
    juntos = torch.cat([punto, punto], dim=0)

    mezclado = oraculo.log_prob(juntos, torch.tensor([0.01, 1.0], dtype=torch.float64))
    solo_temprano = oraculo.log_prob(punto, _tiempo(0.01, 1))
    solo_tardio = oraculo.log_prob(punto, _tiempo(1.0, 1))

    assert float(mezclado[0]) == pytest.approx(float(solo_temprano[0]), rel=1e-12)
    assert float(mezclado[1]) == pytest.approx(float(solo_tardio[0]), rel=1e-12)
    assert abs(float(solo_temprano[0]) - float(solo_tardio[0])) > 0.5


@pytest.mark.parametrize("nombre", ["vp", "sub_vp"])
def test_con_el_tiempo_en_cero_la_log_densidad_es_la_mixtura_original(nombre: str):
    """Criterio 3.4 en las variantes cuyo desvío llega a cero: el límite es ``p_0`` exacta."""
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde(nombre))
    esperada = _log_mixtura_independiente(
        mixtura,
        torch.as_tensor(mixtura.means_, dtype=torch.float64),
        torch.as_tensor(mixtura.covariances_, dtype=torch.float64),
        _PUNTOS,
    )

    obtenida = oraculo.log_prob(_PUNTOS, _tiempo(0.0, _PUNTOS.shape[0]))

    assert torch.allclose(obtenida, esperada, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("nombre", ["vp", "sub_vp"])
def test_la_log_densidad_converge_a_la_mixtura_original_al_bajar_el_tiempo(nombre: str):
    """Criterio 3.4: la convergencia es monótona al acercarse a cero, no solo el valor en cero."""
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde(nombre))
    esperada = _log_mixtura_independiente(
        mixtura,
        torch.as_tensor(mixtura.means_, dtype=torch.float64),
        torch.as_tensor(mixtura.covariances_, dtype=torch.float64),
        _PUNTOS,
    )

    errores = [
        float(
            (oraculo.log_prob(_PUNTOS, _tiempo(v, _PUNTOS.shape[0])) - esperada)
            .abs()
            .max()
        )
        for v in (1e-2, 1e-4, 1e-6)
    ]

    assert errores[0] > errores[1] > errores[2]
    assert errores[-1] < 1e-3


def test_en_ve_el_limite_es_la_mixtura_suavizada_por_el_piso_del_schedule():
    """Criterio 3.4 en VE: su schedule no baja de ``sigma_min``, así que el límite no es ``p_0``.

    Es la cara por variante del invariante de diseño. El test fija las dos direcciones: que
    el límite **es** la mixtura convolucionada con ``N(0, sigma_min² I)`` y que **no** es
    ``p_0``, así que comparar las tres SDEs contra ``p_0`` falla acá.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("ve"))
    # Puntos desplazados sobre el eje angosto de la segunda componente (desvío ``0.2``),
    # donde el piso del schedule deja su huella más visible.
    puntos = torch.tensor(
        [[2.0, -1.6], [2.0, -0.4], [-1.5, 0.5], [0.0, 0.0]], dtype=torch.float64
    )
    medias = torch.as_tensor(mixtura.means_, dtype=torch.float64)
    base = torch.as_tensor(mixtura.covariances_, dtype=torch.float64)
    suavizadas = base + (_SIGMA_MIN**2) * torch.eye(2, dtype=torch.float64)

    obtenida = oraculo.log_prob(puntos, _tiempo(0.0, puntos.shape[0]))
    esperada = _log_mixtura_independiente(mixtura, medias, suavizadas, puntos)
    sin_suavizar = _log_mixtura_independiente(mixtura, medias, base, puntos)

    assert torch.allclose(obtenida, esperada, rtol=1e-12, atol=1e-14)
    assert float((obtenida - sin_suavizar).abs().max()) > 1e-3


def test_la_densidad_contrae_las_medias_por_el_factor_del_kernel():
    """Sin ``alpha_t mu_k`` la densidad quedaría centrada donde estaban los datos.

    En VP con ``t = 1`` el factor de contracción vale ``0.0064``, así que ``p_1`` está
    prácticamente centrada en el origen: la log-densidad tiene que ser mayor ahí que sobre
    la media original de una componente. Con las medias sin contraer la desigualdad se
    invierte.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    puntos = torch.tensor([[0.0, 0.0], [2.0, -1.0]], dtype=torch.float64)

    obtenida = oraculo.log_prob(puntos, _tiempo(1.0, 2))

    assert float(obtenida[0]) > float(obtenida[1]) + 1.0


def test_la_densidad_usa_la_covarianza_en_el_tiempo_y_no_la_de_los_datos():
    """``Sigma_k(t) = alpha_t² Sigma_k + sigma_t² I``: en VE con ``t = 1`` domina el ruido.

    Con ``sigma_t = 5`` la densidad es mucho más plana que ``p_0``; quedarse con ``Sigma_k``
    da valores que difieren en unidades enteras de log-densidad.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("ve"))
    medias, covs = _params_ruideados("ve", 1.0, mixtura)

    obtenida = oraculo.log_prob(_PUNTOS, _tiempo(1.0, _PUNTOS.shape[0]))
    con_tiempo = _log_mixtura_independiente(mixtura, medias, covs, _PUNTOS)
    sin_tiempo = _log_mixtura_independiente(
        mixtura,
        torch.as_tensor(mixtura.means_, dtype=torch.float64),
        torch.as_tensor(mixtura.covariances_, dtype=torch.float64),
        _PUNTOS,
    )

    assert torch.allclose(obtenida, con_tiempo, rtol=1e-10, atol=1e-12)
    assert float((obtenida - sin_tiempo).abs().max()) > 1.0


def test_un_peso_nulo_no_contamina_la_log_densidad():
    """``log 0 = -inf`` entra a ``logsumexp`` sin producir ``nan``: no hace falta recortarlo.

    Discrimina la versión que le pone un piso a los pesos: con un piso, la componente de
    peso cero seguiría aportando densidad y el resultado no coincidiría con el de la mixtura
    de una sola componente.
    """
    media_viva = [[2.0, -1.0]]
    cov_viva = [_diagonal(0.25, 0.04).tolist()]
    con_muerta = ExactGaussianMixture(
        2,
        weights=[0.0, 1.0],
        means=[[-1.5, 0.5]] + media_viva,
        covariances=[_rotada(1.0, 0.09, math.pi / 3).tolist()] + cov_viva,
        seed=0,
    )
    sola = ExactGaussianMixture(
        2, weights=[1.0], means=media_viva, covariances=cov_viva, seed=0
    )
    t = _tiempo(0.2, _PUNTOS.shape[0])

    obtenida = MixtureOracle(con_muerta, make_sde("vp")).log_prob(_PUNTOS, t)
    esperada = MixtureOracle(sola, make_sde("vp")).log_prob(_PUNTOS, t)

    assert torch.isfinite(obtenida).all()
    assert torch.allclose(obtenida, esperada, rtol=1e-12, atol=1e-14)


def test_llamadas_repetidas_dan_el_mismo_valor():
    """El oráculo no tiene estado mutable: la evaluación es pura."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("sub_vp"))
    t = _tiempo(0.6, _PUNTOS.shape[0])
    primera = oraculo.log_prob(_PUNTOS, t)
    assert torch.equal(primera, oraculo.log_prob(_PUNTOS, t))


# --------------------------- oráculo: la log-densidad es derivable respecto del estado


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_gradiente_de_la_log_densidad_atraviesa_el_oraculo(nombre: str):
    """La log-densidad se arma con operaciones tensoriales, sin cortes de grafo.

    Es la adyacencia declarada con la spec de métricas, que necesita ``d/dx log p_t``. Un
    ``.item()``, un desvío por numpy o un ``no_grad()`` interno romperían el grafo y este
    test fallaría con ``RuntimeError``.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    x = _PUNTOS.clone().requires_grad_(True)

    log_densidad = oraculo.log_prob(x, _tiempo(0.3, _PUNTOS.shape[0]))
    (gradiente,) = torch.autograd.grad(log_densidad.sum(), x)

    assert gradiente.shape == x.shape
    assert torch.isfinite(gradiente).all()
    assert float(gradiente.abs().max()) > 0.0


def test_el_gradiente_fluye_tambien_desde_un_estado_en_precision_simple():
    """No se pierde el grafo al promover el estado a doble precisión."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    x = _PUNTOS.to(torch.float32).clone().requires_grad_(True)

    log_densidad = oraculo.log_prob(x, _tiempo(0.3, _PUNTOS.shape[0]))
    (gradiente,) = torch.autograd.grad(log_densidad.sum(), x)

    assert gradiente.dtype is torch.float32
    assert torch.isfinite(gradiente).all()


def test_la_traza_del_jacobiano_por_dos_backward_es_finita():
    """Dos pasadas hacia atrás sobre la log-densidad: lo que pide la log-verosimilitud.

    Verifica que la log-densidad es dos veces derivable respecto del estado, que es la forma
    en la que la spec de métricas piensa calcular la traza exacta del jacobiano.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("ve"))
    x = _PUNTOS.clone().requires_grad_(True)

    log_densidad = oraculo.log_prob(x, _tiempo(0.4, _PUNTOS.shape[0]))
    (gradiente,) = torch.autograd.grad(log_densidad.sum(), x, create_graph=True)
    traza = torch.zeros(x.shape[0], dtype=torch.float64)
    for eje in range(2):
        (segunda,) = torch.autograd.grad(gradiente[:, eje].sum(), x, retain_graph=True)
        traza = traza + segunda[:, eje]

    assert torch.isfinite(traza).all()


def _mixtura_de_una_componente() -> ExactGaussianMixture:
    """Mixtura degenerada ``K = 1``: una sola gaussiana anisotrópica de media conocida."""
    return ExactGaussianMixture(
        2,
        weights=[1.0],
        means=[[0.3, -0.2]],
        covariances=[_diagonal(0.25, 0.04).tolist()],
        seed=0,
    )


def test_con_una_sola_componente_la_traza_del_jacobiano_es_menos_la_traza_de_la_precision():
    """Fuente independiente de la segunda derivada: con ``K = 1`` el hessiano es ``-Sigma(t)⁻¹``.

    La traza esperada se escribe a mano con ``math`` a partir de la forma cerrada de
    ``alpha_t`` y ``sigma_t``, así que el test cruza la log-densidad de la producción contra
    una identidad analítica que no depende de ella.
    """
    oraculo = MixtureOracle(_mixtura_de_una_componente(), make_sde("vp"))
    valor = 0.3
    x = torch.tensor([[0.5, 0.1], [-1.0, 2.0]], dtype=torch.float64, requires_grad=True)

    log_densidad = oraculo.log_prob(x, _tiempo(valor, 2))
    (gradiente,) = torch.autograd.grad(log_densidad.sum(), x, create_graph=True)
    traza = torch.zeros(2, dtype=torch.float64)
    for eje in range(2):
        (segunda,) = torch.autograd.grad(gradiente[:, eje].sum(), x, retain_graph=True)
        traza = traza + segunda[:, eje]

    a, s = _alpha_sigma_cerrados("vp", valor)
    esperada = -(1.0 / (a**2 * 0.25 + s**2) + 1.0 / (a**2 * 0.04 + s**2))
    assert torch.allclose(
        traza, torch.full((2,), esperada, dtype=torch.float64), rtol=1e-9
    )


# --------------------------------- oráculo: contrato de formas y precisión de salida


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("metodo", ["log_prob", "prob"])
def test_devuelve_un_valor_por_punto_en_la_precision_del_caller(
    dtype: "torch.dtype", metodo: str
):
    """Un valor por punto, en el dtype con el que llega el estado."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    x = _PUNTOS.to(dtype)
    t = _tiempo(0.3, x.shape[0]).to(dtype)

    salida = getattr(oraculo, metodo)(x, t)

    assert salida.shape == (x.shape[0],)
    assert salida.dtype is dtype


def test_el_resultado_en_simple_precision_coincide_con_el_de_doble():
    """La degradación es solo de salida: el valor es el mismo dentro del ruido de ``float32``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    t = _tiempo(0.3, _PUNTOS.shape[0])

    doble = oraculo.log_prob(_PUNTOS, t)
    simple = oraculo.log_prob(_PUNTOS.to(torch.float32), t.to(torch.float32))

    assert torch.allclose(simple.to(torch.float64), doble, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("metodo", ["log_prob", "prob"])
@pytest.mark.parametrize(
    "malo",
    [
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, 3, dtype=torch.float64),
        torch.zeros(0, 2, dtype=torch.float64),
        torch.zeros(2, 2, 2, dtype=torch.float64),
    ],
)
def test_rechaza_un_estado_con_forma_invalida(metodo: str, malo: "torch.Tensor"):
    """El estado del laboratorio 2D es ``(B, 2)`` con ``B >= 1``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^x debe"):
        getattr(oraculo, metodo)(malo, _tiempo(0.3, max(1, malo.shape[0])))


def test_rechaza_un_estado_que_no_es_tensor():
    """El contrato del proyecto es tensorial."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^x debe"):
        oraculo.log_prob([[0.0, 0.0]], _tiempo(0.3, 1))  # type: ignore[arg-type]


def test_rechaza_un_estado_que_no_es_de_punto_flotante():
    """Un estado entero no es un punto del plano: se avisa en lugar de truncar en silencio."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^x debe"):
        oraculo.log_prob(torch.zeros(2, 2, dtype=torch.int64), _tiempo(0.3, 2))


def test_rechaza_un_tiempo_con_un_lote_distinto_al_del_estado():
    """Un tiempo por punto: si los lotes no coinciden, el broadcast silencioso sería peor."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        oraculo.log_prob(_PUNTOS, _tiempo(0.3, _PUNTOS.shape[0] - 1))


@pytest.mark.parametrize("metodo", ["log_prob", "prob"])
def test_rechaza_un_tiempo_negativo_al_evaluar_la_densidad(metodo: str):
    """La validación de tiempo del módulo también protege a la densidad."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        getattr(oraculo, metodo)(_PUNTOS, _tiempo(-0.1, _PUNTOS.shape[0]))


# ------------------------------------------------- oráculo: masa integrada (3.6)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
@pytest.mark.parametrize("valor", [1e-4, 0.1, 1.0])
def test_la_masa_integrada_da_uno(nombre: str, valor: float):
    """Criterio 3.6: la densidad integra a uno sobre el plano, en las tres SDEs.

    Tolerancia: ``1e-8`` absoluto. La malla adaptativa cubre ocho desvíos con seis nodos por
    desvío y la suma de Riemann sobre gaussianas converge espectralmente, así que el error
    queda en el ruido de la doble precisión (el prototipo de la spec midió ``1.00000000``
    incluso con ``sigma_t ≈ 1e-5``). Cualquier normalización equivocada —perder el ``2 pi``,
    olvidar la raíz del determinante, no pesar las componentes— se aparta en decenas de por
    ciento.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    assert oraculo.total_mass(valor) == pytest.approx(1.0, abs=1e-8)


def test_la_masa_integrada_da_uno_en_el_tiempo_cero():
    """El extremo del horizonte: con ``t = 0`` la densidad es ``p_0`` y también integra a uno."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    assert oraculo.total_mass(0.0) == pytest.approx(1.0, abs=1e-8)


def test_la_masa_integrada_de_una_sola_componente_da_uno():
    """Caso ``K = 1``: la normalización gaussiana no depende de la mixtura."""
    oraculo = MixtureOracle(_mixtura_de_una_componente(), make_sde("sub_vp"))
    assert oraculo.total_mass(0.25) == pytest.approx(1.0, abs=1e-8)


def test_la_masa_integrada_centra_la_malla_en_las_medias_contraidas():
    """La malla se centra en ``alpha_t mu_k``, no en las medias de los datos.

    Con los modos a ``±3000`` y VP en el horizonte, el factor de contracción vale
    ``0.0066``: la densidad vive a menos de treinta unidades del origen y la malla
    automática la resuelve con 334 nodos. Una malla centrada en las medias **sin** contraer
    tendría que cubrir ``±3008`` con el mismo paso, se choca contra el tope de puntos y
    pierde masa. El test fija las dos caras, así que es la que le da dientes al ``alpha_t``
    del dimensionado.
    """
    cov = _diagonal(0.25, 0.04).tolist()
    lejana = ExactGaussianMixture(
        2,
        weights=[0.4, 0.6],
        means=[[-3000.0, 0.0], [3000.0, 1.0]],
        covariances=[cov, cov],
        seed=0,
    )
    oraculo = MixtureOracle(lejana, make_sde("vp"))
    t = torch.tensor([1.0], dtype=torch.float64)
    sin_contraer = auto_grid(
        means=torch.as_tensor(lejana.means_, dtype=torch.float64),
        covariances=oraculo.component_covariances(t)[0],
    )

    assert sin_contraer.truncated
    assert oraculo.total_mass(1.0) == pytest.approx(1.0, abs=1e-8)
    assert oraculo.total_mass(1.0, grid=sin_contraer) != pytest.approx(1.0, abs=1e-8)


def test_la_masa_integrada_dimensiona_la_malla_con_las_covarianzas_en_el_tiempo():
    """La malla usada por defecto es la que la cuadratura deriva de ``Sigma_k(t)``.

    Se reconstruye con ``auto_grid`` a partir de los parámetros que publica el oráculo y se
    exige el mismo número, así que dimensionar la malla de otra forma (por ``Sigma_k`` sin
    tiempo, o por ``sigma_t``) cambiaría el resultado.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("ve"))
    valor = 0.7
    t = torch.tensor([valor], dtype=torch.float64)
    alpha, _ = oraculo.marginal_params(t)
    medias = float(alpha[0, 0]) * torch.as_tensor(
        mixtura.means_, dtype=torch.float64
    )
    grid = auto_grid(means=medias, covariances=oraculo.component_covariances(t)[0])

    assert oraculo.total_mass(valor, grid=grid) == pytest.approx(
        oraculo.total_mass(valor), rel=1e-12
    )


def test_la_masa_integrada_sobre_una_malla_insuficiente_no_da_uno():
    """La masa es el detector: sobre un dominio que no cubre la densidad, se pierde masa.

    Le da dientes a la firma con malla explícita: si el argumento se ignorara y siempre se
    usara la malla automática, este test daría uno.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    grid = QuadratureGrid(half_width=0.5, n_points=8)
    assert oraculo.total_mass(0.1, grid=grid) < 0.5


@pytest.mark.parametrize("valor", [-1e-3, float("nan"), float("inf")])
def test_la_masa_integrada_rechaza_un_tiempo_invalido(valor: float):
    """Mismo contrato de tiempo que el resto del módulo, con ``t`` escalar."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        oraculo.total_mass(valor)


@pytest.mark.parametrize(
    "malo", ["0.3", None, torch.tensor([0.3], dtype=torch.float64)]
)
def test_la_masa_integrada_rechaza_un_tiempo_que_no_es_un_numero(malo: object):
    """``total_mass`` toma un tiempo escalar: cualquier otra cosa es un error del caller.

    La cadena ``"0.3"`` es el caso con dientes: ``float("0.3")`` la convierte sin chistar, así
    que una validación por conversión en lugar de por tipo la aceptaría en silencio.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        oraculo.total_mass(malo)  # type: ignore[arg-type]


# ------------------------------------------------- oráculo: score exacto (4.1–4.8)


def _score_mixtura_independiente(
    mixtura: ExactGaussianMixture,
    medias: "torch.Tensor",
    covs: "torch.Tensor",
    puntos: "torch.Tensor",
) -> "torch.Tensor":
    """Score de referencia: responsabilidades en el dominio **lineal** y álgebra genérica.

    Segunda implementación del criterio 4.1, escrita para no compartir nada con la
    producción: invierte con ``torch.linalg.inv`` en lugar de la forma cerrada 2×2, y calcula
    las responsabilidades **dividiendo densidades** en vez de con un ``softmax`` sobre
    log-densidades.

    Args:
        mixtura: Mixtura exacta de la que se leen los pesos verdaderos.
        medias: Medias ya contraídas ``alpha_t mu_k``, de forma ``(K, 2)``.
        covs: Covarianzas ya propagadas ``Sigma_k(t)``, de forma ``(K, 2, 2)``.
        puntos: Puntos de evaluación, de forma ``(B, 2)``.

    Returns:
        Tensor ``(B, 2)`` con el score de referencia.
    """
    pesos = torch.as_tensor(mixtura.weights_, dtype=torch.float64)
    inversas = torch.linalg.inv(covs)
    dets = torch.linalg.det(covs)
    dif = puntos.unsqueeze(1) - medias.unsqueeze(0)  # (B, K, 2)
    cuadratica = torch.einsum("bki,kij,bkj->bk", dif, inversas, dif)
    densidades = (
        pesos / (2.0 * math.pi * torch.sqrt(dets)) * torch.exp(-0.5 * cuadratica)
    )
    responsabilidades = densidades / densidades.sum(dim=1, keepdim=True)
    # Sigma_k(t)^{-1} (x - alpha_t mu_k), de forma (B, K, 2).
    empuje = torch.einsum("kij,bkj->bki", inversas, dif)
    return -(responsabilidades.unsqueeze(-1) * empuje).sum(dim=1)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_score_coincide_con_la_formula_independiente(nombre: str):
    """Criterio 4.1: el score contra una forma cerrada escrita aparte, en las tres SDEs.

    Tolerancia: ``rtol=1e-9`` sobre magnitudes de hasta ~100. Es holgada frente al ruido de la
    doble precisión y órdenes de magnitud más chica que cualquier error estructural —un signo
    invertido, usar ``Sigma_k`` en lugar de ``Sigma_k(t)``, olvidar las responsabilidades—,
    que se apartan en factores enteros.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde(nombre))

    for valor in _TIEMPOS:
        medias, covs = _params_ruideados(nombre, valor, mixtura)
        esperado = _score_mixtura_independiente(mixtura, medias, covs, _PUNTOS)
        obtenido = oraculo.score(_PUNTOS, _tiempo(valor, _PUNTOS.shape[0]))

        assert obtenido.shape == _PUNTOS.shape
        assert torch.allclose(obtenido, esperado, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_score_coincide_con_el_gradiente_de_su_log_densidad(nombre: str):
    """Criterio 4.6: el score **es** ``d/dx log p_t``, verificado con autograd.

    Cruza las dos formas cerradas del módulo entre sí, así que un signo invertido o un eje
    contraído de más en cualquiera de las dos se delata acá.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))

    for valor in _TIEMPOS:
        t = _tiempo(valor, _PUNTOS.shape[0])
        x = _PUNTOS.clone().requires_grad_(True)
        (gradiente,) = torch.autograd.grad(oraculo.log_prob(x, t).sum(), x)
        obtenido = oraculo.score(_PUNTOS, t)
        assert torch.allclose(obtenido, gradiente, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_con_una_sola_componente_el_score_es_la_forma_cerrada_de_una_gaussiana(
    nombre: str,
):
    """Criterio 7.1 en su versión ``K = 1``: ``score = -Sigma(t)^{-1} (x - alpha mu)``.

    La esperanza se arma con ``math`` y floats de Python a partir de ``alpha_t`` y ``sigma_t``
    escritos a mano, así que no depende de ninguna pieza de la producción.
    """
    oraculo = MixtureOracle(_mixtura_de_una_componente(), make_sde(nombre))
    x = torch.tensor([[0.5, 0.1], [-1.0, 2.0]], dtype=torch.float64)

    for valor in _TIEMPOS:
        a, s = _alpha_sigma_cerrados(nombre, valor)
        v1, v2 = a**2 * 0.25 + s**2, a**2 * 0.04 + s**2
        esperado = torch.tensor(
            [[-(px - a * 0.3) / v1, -(py + a * 0.2) / v2] for px, py in x.tolist()],
            dtype=torch.float64,
        )
        obtenido = oraculo.score(x, _tiempo(valor, x.shape[0]))
        assert torch.allclose(obtenido, esperado, rtol=1e-9, atol=1e-12)


def test_el_score_pondera_por_las_responsabilidades_posteriores():
    """Criterio 4.1: los pesos entran por las responsabilidades, no de forma pareja.

    Dos componentes de la **misma** covarianza y medias simétricas respecto del origen: en el
    origen las dos densidades valen lo mismo, así que la responsabilidad de cada una es
    exactamente su peso. Con pesos ``(0.8, 0.2)`` el score vale ``-0.6·alpha/v`` en la primera
    coordenada; repartir las responsabilidades por igual daría exactamente **cero**, que es lo
    que le da dientes al test.
    """
    simetrica = ExactGaussianMixture(
        2,
        weights=[0.8, 0.2],
        means=[[-1.0, 0.0], [1.0, 0.0]],
        covariances=[_identidad(0.25).tolist(), _identidad(0.25).tolist()],
        seed=0,
    )
    oraculo = MixtureOracle(simetrica, make_sde("vp"))
    valor = 0.4
    a, s = _alpha_sigma_cerrados("vp", valor)
    v = a**2 * 0.25 + s**2

    obtenido = oraculo.score(torch.zeros(1, 2, dtype=torch.float64), _tiempo(valor, 1))

    esperado = torch.tensor([[-0.6 * a / v, 0.0]], dtype=torch.float64)
    assert torch.allclose(obtenido, esperado, rtol=1e-9, atol=1e-12)
    assert float(obtenido[0, 0].abs()) > 0.1


def test_el_score_usa_la_covarianza_en_el_tiempo_y_no_la_de_los_datos():
    """El kernel propaga la covarianza: ``Sigma(t) = alpha_t² Sigma + sigma_t² I``.

    En ``t = 0.5`` el ruido del kernel domina a la covarianza de los datos, así que usar
    ``Sigma`` en lugar de ``Sigma(t)`` da un score varias veces más grande.
    """
    oraculo = MixtureOracle(_mixtura_de_una_componente(), make_sde("vp"))
    valor = 0.5
    a, s = _alpha_sigma_cerrados("vp", valor)
    x = torch.tensor([[0.5, 0.1]], dtype=torch.float64)
    dif = x - a * torch.tensor([[0.3, -0.2]], dtype=torch.float64)

    obtenido = oraculo.score(x, _tiempo(valor, 1))

    con_ruido = -dif / torch.tensor(
        [[a**2 * 0.25 + s**2, a**2 * 0.04 + s**2]], dtype=torch.float64
    )
    sin_ruido = -dif / torch.tensor([[0.25, 0.04]], dtype=torch.float64)
    assert torch.allclose(obtenido, con_ruido, rtol=1e-9, atol=1e-12)
    assert not torch.allclose(obtenido, sin_ruido, rtol=1e-2)


def test_el_score_no_recorta_el_desvio_del_kernel_como_el_target_de_entrenamiento():
    """Criterio 4.8: el oráculo devuelve el valor **exacto**, sin piso para ``sigma_t``.

    La componente es casi puntual (covarianza ``1e-18 I``), así que ``Sigma(t) ≈ sigma_t² I`` y
    el recorte se ve de lleno. En sub-VP con ``t = 1e-5`` el desvío vale ``~1e-6``, por debajo
    del piso ``1e-5`` que sí aplica el target de entrenamiento: el valor exacto queda un orden
    de magnitud por encima del recortado.
    """
    casi_puntual = ExactGaussianMixture(
        2,
        weights=[1.0],
        means=[[0.0, 0.0]],
        covariances=[_identidad(1e-18).tolist()],
        seed=0,
    )
    sde = make_sde("sub_vp")
    oraculo = MixtureOracle(casi_puntual, sde)
    valor = 1e-5
    a, s = _alpha_sigma_cerrados("sub_vp", valor)
    assert s < 1e-5, "el test necesita un desvío por debajo del piso del target"

    x = torch.tensor([[3e-6, -2e-6]], dtype=torch.float64)
    t = torch.tensor([[valor]], dtype=torch.float64)

    obtenido = oraculo.score(x, t)

    exacto = -x / (a**2 * 1e-18 + s**2)
    assert torch.allclose(obtenido, exacto, rtol=1e-9, atol=0.0)

    # El target de entrenamiento sí recorta: con ``x = alpha·0 + s·eps`` devuelve
    # ``-eps / max(s, 1e-5)``, que acá es diez veces más chico que el score verdadero.
    recortado, _ = sde.score_target(torch.zeros_like(x), t, x / s)
    assert float(obtenido.abs().max()) > 5.0 * float(recortado.abs().max())


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
@pytest.mark.parametrize("valor", [1e-5, 1e-4, 1e-3])
def test_el_score_es_finito_con_el_desvio_del_kernel_diminuto(
    nombre: str, valor: float
):
    """Criterio 4.7: sin desbordes aunque la magnitud del score crezca como ``1/sigma_t²``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))

    for dtype in (torch.float32, torch.float64):
        x = _PUNTOS.to(dtype)
        salida = oraculo.score(x, _tiempo(valor, x.shape[0]).to(dtype))
        assert torch.isfinite(salida).all()


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_tiempo_plano_y_en_columna_dan_el_mismo_score(nombre: str):
    """Criterio 3.5/4.2: ``t`` como ``(B,)`` o ``(B, 1)``; los samplers pasan la columna."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    plano = _tiempo(0.3, _PUNTOS.shape[0])

    assert torch.equal(
        oraculo.score(_PUNTOS, plano), oraculo.score(_PUNTOS, plano.reshape(-1, 1))
    )


def test_cada_punto_recibe_el_score_de_su_propio_tiempo():
    """Un tiempo por punto: el lote no se evalúa entero en el primer instante."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("ve"))
    x = _PUNTOS[:2]
    tiempos = [0.1, 0.9]

    juntos = oraculo.score(x, torch.tensor(tiempos, dtype=torch.float64))

    for fila, valor in enumerate(tiempos):
        solo = oraculo.score(x[fila : fila + 1], _tiempo(valor, 1))
        assert torch.allclose(juntos[fila : fila + 1], solo, rtol=1e-12, atol=1e-14)
    assert not torch.allclose(juntos, oraculo.score(x, _tiempo(tiempos[0], 2)))


@pytest.mark.parametrize("metodo", ["score", "__call__"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_el_score_devuelve_la_forma_el_dtype_y_el_device_del_estado(
    dtype: "torch.dtype", metodo: str
):
    """Criterio 4.1/4.5: la salida tiene la shape, la precisión y el device de ``x``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    x = _PUNTOS.to(dtype)
    t = _tiempo(0.3, x.shape[0]).to(dtype)

    salida = getattr(oraculo, metodo)(x, t)

    assert salida.shape == x.shape
    assert salida.dtype is dtype
    assert salida.device == x.device


def test_el_score_en_simple_precision_coincide_con_el_de_doble():
    """La degradación es solo de salida: los pasos intermedios corren en doble precisión."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    t = _tiempo(0.3, _PUNTOS.shape[0])

    doble = oraculo.score(_PUNTOS, t)
    simple = oraculo.score(_PUNTOS.to(torch.float32), t.to(torch.float32))

    assert torch.allclose(simple.to(torch.float64), doble, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
def test_el_score_sale_en_el_device_del_estado_cuando_hay_gpu():  # pragma: no cover
    """Criterio 4.5 en su mitad de device: los parámetros se promueven en cada llamada."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    x = _PUNTOS.to(device="cuda", dtype=torch.float32)

    salida = oraculo.score(x, _tiempo(0.3, x.shape[0]).to(x.device))

    assert salida.device == x.device
    assert salida.dtype is torch.float32


def test_el_score_no_muta_el_oraculo_ni_cachea_una_copia_por_device():
    """Criterio 4.4: dos llamadas iguales dan lo mismo y nada del oráculo cambia.

    Se compara el estado completo del objeto antes y después: ningún atributo nuevo (una
    caché por device rompería el contrato en GPU), las mismas identidades de tensor, los
    mismos valores y la doble precisión intacta.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    antes = dict(vars(oraculo))
    copias = {k: v.clone() for k, v in antes.items() if isinstance(v, torch.Tensor)}
    x = _PUNTOS.to(torch.float32)
    t = _tiempo(0.3, x.shape[0])

    primera = oraculo.score(x, t)
    segunda = oraculo.score(x, t)

    assert torch.equal(primera, segunda)
    assert set(vars(oraculo)) == set(antes)
    for clave, valor in antes.items():
        assert vars(oraculo)[clave] is valor
    for clave, copia in copias.items():
        assert torch.equal(vars(oraculo)[clave], copia)
        assert vars(oraculo)[clave].dtype is torch.float64


def test_el_score_funciona_dentro_de_no_grad():
    """El driver de los samplers evalúa el score bajo ``torch.no_grad()``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))

    with torch.no_grad():
        salida = oraculo.score(_PUNTOS, _tiempo(0.3, _PUNTOS.shape[0]))

    assert torch.isfinite(salida).all()
    assert not salida.requires_grad


def test_el_score_no_apaga_el_grafo_del_estado():
    """No hay ``no_grad`` interno: la derivabilidad del módulo se usa aguas abajo."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    x = _PUNTOS.clone().requires_grad_(True)

    salida = oraculo.score(x, _tiempo(0.3, x.shape[0]))

    assert salida.requires_grad


def test_el_invocable_es_el_mismo_score_y_entra_donde_se_espera_uno_inyectable():
    """Criterio 4.2: ``__call__`` cumple el contrato ``(x, t) -> score`` de los samplers."""
    from diffusion.samplers import ScoreFn

    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    t = _tiempo(0.3, _PUNTOS.shape[0]).reshape(-1, 1)

    def consumir(fn: ScoreFn, x: "torch.Tensor", tt: "torch.Tensor") -> "torch.Tensor":
        """Consume el score como lo hacen los samplers: un invocable y nada más."""
        return fn(x, tt)

    assert callable(oraculo)
    assert torch.equal(consumir(oraculo, _PUNTOS, t), oraculo.score(_PUNTOS, t))


@pytest.mark.parametrize("metodo", ["score", "__call__"])
@pytest.mark.parametrize(
    "malo",
    [
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, 3, dtype=torch.float64),
        torch.zeros(0, 2, dtype=torch.float64),
    ],
)
def test_el_score_rechaza_un_estado_con_forma_invalida(
    metodo: str, malo: "torch.Tensor"
):
    """El estado del laboratorio 2D es ``(B, 2)`` con ``B >= 1``, también para el score."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^x debe"):
        getattr(oraculo, metodo)(malo, _tiempo(0.3, max(1, malo.shape[0])))


@pytest.mark.parametrize("metodo", ["score", "__call__"])
def test_el_score_rechaza_un_tiempo_que_no_trae_uno_por_punto(metodo: str):
    """Un tiempo por punto: el broadcast silencioso sería peor que el error."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        getattr(oraculo, metodo)(_PUNTOS, _tiempo(0.3, _PUNTOS.shape[0] - 1))


@pytest.mark.parametrize("metodo", ["score", "__call__"])
def test_el_score_rechaza_un_tiempo_negativo(metodo: str):
    """Mismo contrato de tiempo que la densidad: el proceso corre en ``[0, T]``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        getattr(oraculo, metodo)(_PUNTOS, _tiempo(-0.1, _PUNTOS.shape[0]))


def test_un_peso_nulo_no_contamina_el_score():
    """``log 0 = -inf`` entra al ``softmax`` sin producir ``nan``: la componente aporta cero.

    Es la contraparte del mismo chequeo sobre la log-densidad, y discrimina la versión que le
    pone un piso a los pesos: con un piso, la componente muerta seguiría empujando y el score
    no coincidiría con el de la mixtura de una sola componente.
    """
    media_viva = [[2.0, -1.0]]
    cov_viva = [_diagonal(0.25, 0.04).tolist()]
    con_muerta = ExactGaussianMixture(
        2,
        weights=[0.0, 1.0],
        means=[[-1.5, 0.5]] + media_viva,
        covariances=[_rotada(1.0, 0.09, math.pi / 3).tolist()] + cov_viva,
        seed=0,
    )
    sola = ExactGaussianMixture(
        2, weights=[1.0], means=media_viva, covariances=cov_viva, seed=0
    )
    t = _tiempo(0.2, _PUNTOS.shape[0])

    obtenido = MixtureOracle(con_muerta, make_sde("vp")).score(_PUNTOS, t)
    esperado = MixtureOracle(sola, make_sde("vp")).score(_PUNTOS, t)

    assert torch.isfinite(obtenido).all()
    assert torch.allclose(obtenido, esperado, rtol=1e-12, atol=1e-14)


# ------------------------------- oráculo: sesgo de inicialización, cota cerrada (6.1, 6.5)


#: Varianza de la distribución de partida de cada variante registrada. **No** se lee de la
#: SDE: ``prior_sampling`` solo muestrea y no publica su escala, así que el número entra como
#: dato del caller. VP y sub-VP arrancan de ``N(0, I)``; VE de ``N(0, sigma_max² I)`` con el
#: ``sigma_max = 5.0`` por defecto del paquete.
_VARIANZA_DEL_PRIOR = {"vp": 1.0, "ve": 25.0, "sub_vp": 1.0}

#: Tiempos en los que se evalúa el sesgo: dos intermedios y el horizonte.
_TIEMPOS_SESGO = [0.05, 0.5, 1.0]


def _kl_gaussiana_contra_isotropica(
    media: tuple[float, float],
    cov: tuple[tuple[float, float], tuple[float, float]],
    varianza_prior: float,
) -> float:
    """``KL(N(m, S) ‖ N(0, v I))`` en dos dimensiones, escrita con floats de Python.

    Fuente independiente de la cota: ``½[tr(S)/v + mᵀm/v − 2 − log det S + 2 log v]``,
    calculada con ``math`` y sin una sola operación de torch, de modo que no sea una
    tautología contra la producción.
    """
    (a, b), (c, d) = cov[0], cov[1]
    det = a * d - b * c
    traza = a + d
    norma2 = media[0] ** 2 + media[1] ** 2
    return 0.5 * (
        traza / varianza_prior
        + norma2 / varianza_prior
        - 2.0
        - math.log(det)
        + 2.0 * math.log(varianza_prior)
    )


def _componentes_ruideadas_a_mano(
    nombre: str, t: float, mixtura: ExactGaussianMixture, *, contraer: bool = True
) -> list[
    tuple[float, tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]
]:
    """``(w_k, alpha_t mu_k, Sigma_k(t))`` de cada componente, en floats de Python.

    Args:
        nombre: Variante cuya forma cerrada de ``(alpha_t, sigma_t)`` escribe el test.
        t: Instante en el que se propaga la mixtura.
        mixtura: Mixtura de parámetros exactos.
        contraer: Si es ``False`` deja las medias **sin** el factor ``alpha_t``, para poder
            construir a propósito la variante equivocada contra la que discriminar.
    """
    a, s = _alpha_sigma_cerrados(nombre, t)
    factor = a if contraer else 1.0
    salida = []
    for w, mu, cov in zip(
        mixtura.weights_.tolist(),
        mixtura.means_.tolist(),
        mixtura.covariances_.tolist(),
    ):
        media = (factor * mu[0], factor * mu[1])
        propagada = (
            (a * a * cov[0][0] + s * s, a * a * cov[0][1]),
            (a * a * cov[1][0], a * a * cov[1][1] + s * s),
        )
        salida.append((w, media, propagada))
    return salida


def _cota_convexa_a_mano(
    nombre: str,
    t: float,
    mixtura: ExactGaussianMixture,
    varianza_prior: float,
    *,
    contraer: bool = True,
) -> float:
    """``Σ_k w_k KL(N_k(t) ‖ N(0, v I))``: la combinación convexa, sumada a mano."""
    return sum(
        w * _kl_gaussiana_contra_isotropica(media, cov, varianza_prior)
        for w, media, cov in _componentes_ruideadas_a_mano(
            nombre, t, mixtura, contraer=contraer
        )
    )


def _explota(*args: object, **kwargs: object) -> None:
    """Reemplazo que falla si alguien lo invoca: marca un camino que no debería recorrerse."""
    raise AssertionError(
        "el sesgo no debe estimar la varianza de partida; el reemplazo fue invocado."
    )


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
@pytest.mark.parametrize("valor", _TIEMPOS_SESGO)
def test_la_cota_es_la_combinacion_convexa_de_las_kl_por_componente(
    nombre: str, valor: float
):
    """Criterio 6.1: la cota es ``Σ_k w_k KL(N_k(t) ‖ prior)``, en las tres SDEs.

    La KL de una mixtura contra una gaussiana **no** tiene forma cerrada —la entropía
    diferencial de una mixtura no la tiene—, así que lo que sí se puede calcular exactamente
    es la cota por convexidad de la KL, con cada término gaussiano-gaussiano cerrado. El valor
    esperado se escribe aparte con ``math`` y con la forma cerrada de ``(alpha, sigma)`` del
    propio test, así que discrimina cualquier reescritura de la fórmula.

    Tolerancia: ``rtol=1e-12``, el ruido de la doble precisión sobre una suma de unos pocos
    términos.
    """
    mixtura = _mixtura_exacta()
    varianza = _VARIANZA_DEL_PRIOR[nombre]
    oraculo = MixtureOracle(mixtura, make_sde(nombre))

    reporte = oraculo.initialization_bias(varianza, t=valor)

    assert reporte.bound == pytest.approx(
        _cota_convexa_a_mano(nombre, valor, mixtura, varianza), rel=1e-12
    )


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
@pytest.mark.parametrize("valor", [1e-4, *_TIEMPOS_SESGO])
def test_la_cota_es_no_negativa(nombre: str, valor: float):
    """Criterio 6.6: cada término es una KL, así que la combinación convexa no es negativa."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    reporte = oraculo.initialization_bias(_VARIANZA_DEL_PRIOR[nombre], t=valor)
    assert reporte.bound >= 0.0
    assert math.isfinite(reporte.bound)


def test_la_cota_no_depende_de_la_malla_de_la_cuadratura():
    """Criterio 6.1: la cota es **forma cerrada**, no una cuadratura disfrazada.

    El reporte trae las dos mitades y una de ellas sí se integra, así que la forma de exigir que
    la cota no lo haga es que sea **idéntica bit a bit** con dos mallas de resoluciones muy
    distintas, mientras la tolerancia del valor —que sí depende de la malla— cambia. Una cota
    calculada por cuadratura, o corregida por la masa, no podría coincidir exactamente.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    automatica = _malla_del_sesgo(oraculo, mixtura, 1.0, 1.0)
    fina = QuadratureGrid(
        half_width=1.25 * automatica.half_width, n_points=3 * automatica.n_points
    )

    por_defecto = oraculo.initialization_bias(1.0)
    con_malla_fina = oraculo.initialization_bias(1.0, grid=fina)

    assert por_defecto.bound > 0.0
    assert con_malla_fina.bound == por_defecto.bound
    assert con_malla_fina.tolerance < por_defecto.tolerance


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_con_una_sola_componente_la_cota_es_la_kl_exacta(nombre: str):
    """Criterio 6.4: con ``K = 1`` la cota **es** la KL, porque ahí sí hay forma cerrada.

    La desigualdad de convexidad se satura cuando la mixtura tiene una sola componente: no
    queda entropía de mezcla que acotar. El valor se compara contra la KL gaussiana-gaussiana
    escrita a mano, que en ese caso es la divergencia exacta y no una cota.
    """
    mixtura = _mixtura_de_una_componente()
    varianza = _VARIANZA_DEL_PRIOR[nombre]
    oraculo = MixtureOracle(mixtura, make_sde(nombre))
    ((peso, media, cov),) = _componentes_ruideadas_a_mano(nombre, 1.0, mixtura)
    assert peso == pytest.approx(1.0)

    reporte = oraculo.initialization_bias(varianza, t=1.0)

    assert reporte.bound == pytest.approx(
        _kl_gaussiana_contra_isotropica(media, cov, varianza), rel=1e-12
    )


def test_la_cota_pondera_cada_componente_por_su_peso():
    """La combinación es **convexa**: sin los pesos, la cota sería el promedio simple.

    La geometría está elegida para separar las dos candidatas por más de un orden de
    magnitud: una componente que casi coincide con el prior se lleva el ``0.99`` del peso y
    una componente lejanísima, cuya KL es del orden de ``10³``, se lleva el ``0.01``. Ponderar
    da ``~10``; promediar a ciegas da ``~600``. La versión que olvida los pesos no puede
    satisfacer los dos asserts a la vez.
    """
    mixtura = ExactGaussianMixture(
        2,
        weights=[0.99, 0.01],
        means=[[0.0, 0.0], [50.0, 0.0]],
        covariances=[_identidad(1.0).tolist(), _diagonal(0.04, 0.04).tolist()],
        seed=0,
    )
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    componentes = _componentes_ruideadas_a_mano("vp", 0.05, mixtura)
    por_componente = [
        _kl_gaussiana_contra_isotropica(media, cov, 1.0)
        for _, media, cov in componentes
    ]
    ponderada = sum(w * kl for (w, _, _), kl in zip(componentes, por_componente))
    promedio_simple = sum(por_componente) / len(por_componente)

    reporte = oraculo.initialization_bias(1.0, t=0.05)

    assert ponderada < 0.05 * promedio_simple
    assert reporte.bound == pytest.approx(ponderada, rel=1e-12)


def test_la_cota_usa_la_covarianza_en_el_tiempo_y_no_la_de_los_datos():
    """La covarianza de cada término es ``Sigma_k(t) = alpha_t² Sigma_k + sigma_t² I``.

    Con VE en el horizonte el ruido del kernel aporta ``sigma_T² = 25``, que domina por dos
    órdenes de magnitud a las covarianzas de los datos: usar ``Sigma_k`` sin propagar da una
    cota decenas de veces más grande.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("ve"))
    propagada = _cota_convexa_a_mano("ve", 1.0, mixtura, 25.0)
    sin_propagar = sum(
        w
        * _kl_gaussiana_contra_isotropica(
            (mu[0], mu[1]),
            ((cov[0][0], cov[0][1]), (cov[1][0], cov[1][1])),
            25.0,
        )
        for w, mu, cov in zip(
            mixtura.weights_.tolist(),
            mixtura.means_.tolist(),
            mixtura.covariances_.tolist(),
        )
    )

    reporte = oraculo.initialization_bias(25.0, t=1.0)

    assert sin_propagar > 10.0 * propagada
    assert reporte.bound == pytest.approx(propagada, rel=1e-12)


def test_la_cota_contrae_las_medias_por_el_factor_del_kernel():
    """Las medias de cada término son ``alpha_t mu_k``, no las de los datos.

    Con los modos a ``±3000`` y VP en el horizonte el factor vale ``6.6e-3``: la cota
    correcta queda del orden de ``10²``, mientras que olvidar la contracción da ``10⁶``.
    """
    cov = _diagonal(0.25, 0.04).tolist()
    lejana = ExactGaussianMixture(
        2,
        weights=[0.4, 0.6],
        means=[[-3000.0, 0.0], [3000.0, 1.0]],
        covariances=[cov, cov],
        seed=0,
    )
    oraculo = MixtureOracle(lejana, make_sde("vp"))
    contraida = _cota_convexa_a_mano("vp", 1.0, lejana, 1.0)
    sin_contraer = _cota_convexa_a_mano("vp", 1.0, lejana, 1.0, contraer=False)

    reporte = oraculo.initialization_bias(1.0, t=1.0)

    assert sin_contraer > 1000.0 * contraida
    assert reporte.bound == pytest.approx(contraida, rel=1e-12)


def test_la_cota_resta_el_logaritmo_del_determinante():
    """El término ``− log det Sigma_k(t)`` entra con su signo.

    Con una componente muy anisotrópica el determinante propagado es bastante menor que uno,
    así que ``− log det`` aporta un término positivo grande; invertirle el signo cambia la
    cota en ``2 log det``, que acá son varias unidades.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    componentes = _componentes_ruideadas_a_mano("vp", 0.05, mixtura)
    correcta = sum(
        w * _kl_gaussiana_contra_isotropica(media, cov, 1.0)
        for w, media, cov in componentes
    )
    signo_invertido = sum(
        w
        * (
            _kl_gaussiana_contra_isotropica(media, cov, 1.0)
            + math.log(cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0])
        )
        for w, media, cov in componentes
    )

    reporte = oraculo.initialization_bias(1.0, t=0.05)

    assert abs(correcta - signo_invertido) > 1.0
    assert reporte.bound == pytest.approx(correcta, rel=1e-12)


def test_la_cota_usa_la_varianza_de_partida_y_no_su_desvio():
    """El parámetro es la **varianza** del prior, no su desvío.

    Para VE el prior es ``N(0, sigma_max² I)`` con ``sigma_max = 5``: pasar ``25`` y pasar
    ``5`` dan cotas que se separan por más de un orden de magnitud, así que una fórmula que
    le tomara raíz o cuadrado al argumento no podría coincidir con la referencia.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("ve"))
    con_varianza = _cota_convexa_a_mano("ve", 1.0, mixtura, 25.0)
    con_desvio = _cota_convexa_a_mano("ve", 1.0, mixtura, 5.0)

    reporte = oraculo.initialization_bias(25.0, t=1.0)

    assert abs(con_desvio - con_varianza) > 10.0 * con_varianza
    assert reporte.bound == pytest.approx(con_varianza, rel=1e-12)


def test_la_cota_es_cero_cuando_la_componente_coincide_con_la_partida():
    """Criterio 6.6: la cota se anula cuando ``p_T`` **es** la distribución de partida.

    Se construye el caso degenerado a propósito: una sola componente centrada en el origen y
    una varianza de partida igual a ``sigma_T² + 0.01``, que es justo la varianza propagada.
    Fija además que el resultado no se vaya del lado negativo por redondeo.
    """
    centrada = ExactGaussianMixture(
        2,
        weights=[1.0],
        means=[[0.0, 0.0]],
        covariances=[_identidad(0.01).tolist()],
        seed=0,
    )
    oraculo = MixtureOracle(centrada, make_sde("ve"))
    _, sigma = oraculo.marginal_params(torch.tensor([1.0], dtype=torch.float64))
    varianza = float(sigma[0, 0]) ** 2 + 0.01

    reporte = oraculo.initialization_bias(varianza, t=1.0)

    assert reporte.bound >= 0.0
    assert reporte.bound < 1e-12


def test_omitir_la_varianza_de_partida_es_un_error_de_llamada():
    """Criterio 6.5: la varianza de partida es obligatoria, nunca se estima en silencio.

    No hay forma exacta ni genérica de leerla de la SDE (``prior_sampling`` solo muestrea),
    así que si se pudiera omitir el camino por defecto sería una estimación y la cota dejaría
    de ser de forma cerrada sin que el caller lo note. Omitirla es un ``TypeError`` de la
    propia firma.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(TypeError):
        oraculo.initialization_bias()  # type: ignore[call-arg]


def test_la_cota_no_cae_en_la_estimacion_de_la_varianza_de_partida(
    monkeypatch: pytest.MonkeyPatch,
):
    """El camino de la cota no toca el estimador: la varianza que usa es la que le pasaron."""
    monkeypatch.setattr(MixtureOracle, "estimate_prior_variance", _explota)
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))

    reporte = oraculo.initialization_bias(1.0)

    assert reporte.prior_variance == 1.0


@pytest.mark.parametrize("valor", [0.0, -1.0, float("nan"), float("inf")])
def test_rechaza_una_varianza_de_partida_fuera_de_rango(valor: float):
    """Una varianza nula, negativa o no finita no describe ninguna distribución de partida."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^prior_variance debe"):
        oraculo.initialization_bias(valor)


@pytest.mark.parametrize(
    "malo", ["1.0", None, torch.tensor([1.0], dtype=torch.float64)]
)
def test_rechaza_una_varianza_de_partida_que_no_es_un_numero(malo: object):
    """La varianza es un escalar; la cadena ``"1.0"`` es el caso con dientes.

    ``float("1.0")`` la convierte sin chistar, así que validar por conversión en lugar de por
    tipo la aceptaría en silencio.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^prior_variance debe"):
        oraculo.initialization_bias(malo)  # type: ignore[arg-type]


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_horizonte_por_defecto_es_el_de_la_sde(nombre: str):
    """Sin ``t`` explícito el sesgo se evalúa en el horizonte que declara la SDE."""
    sde = make_sde(nombre)
    oraculo = MixtureOracle(_mixtura_exacta(), sde)

    reporte = oraculo.initialization_bias(_VARIANZA_DEL_PRIOR[nombre])

    assert reporte.horizon == pytest.approx(float(sde.T))
    explicito = oraculo.initialization_bias(
        _VARIANZA_DEL_PRIOR[nombre], t=float(sde.T)
    )
    assert reporte.bound == pytest.approx(explicito.bound, rel=1e-15)


def test_el_reporte_publica_la_varianza_de_partida_y_el_horizonte():
    """Criterio 6.5: el reporte deja a la vista con qué prior y en qué tiempo se calculó."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("ve"))
    reporte = oraculo.initialization_bias(25.0, t=0.5)
    assert isinstance(reporte, BiasReport)
    assert reporte.prior_variance == pytest.approx(25.0)
    assert reporte.horizon == pytest.approx(0.5)


def test_el_reporte_es_inmutable():
    """El reporte es un valor: no se le puede reescribir la cota después de calculada."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    reporte = oraculo.initialization_bias(1.0)
    with pytest.raises(Exception):
        reporte.bound = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("valor", [-1e-3, float("nan"), float("inf")])
def test_la_cota_rechaza_un_tiempo_invalido(valor: float):
    """Mismo contrato de tiempo escalar que la masa integrada."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        oraculo.initialization_bias(1.0, t=valor)


def test_la_cota_rechaza_un_tiempo_que_no_es_un_numero():
    """La cadena ``"0.5"`` no es un instante del proceso."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        oraculo.initialization_bias(1.0, t="0.5")  # type: ignore[arg-type]


def test_la_cota_repetida_da_el_mismo_numero_y_no_muta_el_oraculo():
    """Determinístico y sin estado: dos llamadas idénticas dan el mismo valor."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("sub_vp"))
    primera = oraculo.initialization_bias(1.0, t=0.5)
    intermedia = oraculo.initialization_bias(25.0, t=0.1)
    segunda = oraculo.initialization_bias(1.0, t=0.5)
    assert primera == segunda
    assert intermedia.bound != primera.bound


# --------------------- oráculo: estimación opt-in de la varianza de partida (6.5)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_la_estimacion_se_acerca_a_la_varianza_conocida_de_cada_prior(nombre: str):
    """El estimador de respaldo cae dentro de la tolerancia que él mismo declara.

    Las varianzas conocidas son ``1.0`` para VP y sub-VP y ``sigma_max² = 25`` para VE. El
    error relativo del estimador escala como ``sqrt(2/n)``, así que la tolerancia devuelta
    tiene que contener la desviación observada — es lo que la vuelve utilizable como cota y no
    como decorado.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    conocida = _VARIANZA_DEL_PRIOR[nombre]

    varianza, tolerancia = oraculo.estimate_prior_variance(n=100_000, seed=0)

    assert tolerancia > 0.0
    assert abs(varianza - conocida) / conocida <= tolerancia


def test_la_estimacion_es_determinista_para_una_semilla_fija():
    """Misma semilla, mismo número: el estimador no arrastra el estado global de torch."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("ve"))
    torch.manual_seed(1234)
    primera = oraculo.estimate_prior_variance(n=5_000, seed=7)
    torch.manual_seed(4321)
    segunda = oraculo.estimate_prior_variance(n=5_000, seed=7)
    otra = oraculo.estimate_prior_variance(n=5_000, seed=8)

    assert primera == segunda
    assert primera[0] != otra[0]


def test_la_tolerancia_de_la_estimacion_se_afina_al_crecer_la_muestra():
    """La tolerancia es un número computado a partir de ``n``, no una constante."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    _, tolerancia_chica = oraculo.estimate_prior_variance(n=1_000, seed=0)
    _, tolerancia_grande = oraculo.estimate_prior_variance(n=100_000, seed=0)
    assert tolerancia_grande < tolerancia_chica
    assert tolerancia_grande == pytest.approx(tolerancia_chica / 10.0, rel=1e-12)


@pytest.mark.parametrize("valor", [1, 0, -5])
def test_la_estimacion_rechaza_un_tamano_de_muestra_invalido(valor: int):
    """Con menos de dos puntos no hay estimación posible."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^n debe"):
        oraculo.estimate_prior_variance(n=valor)


@pytest.mark.parametrize("malo", [2.5, "10", None])
def test_la_estimacion_rechaza_un_tamano_de_muestra_que_no_es_entero(malo: object):
    """El tamaño de muestra es una cantidad de puntos: un flotante no lo describe."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^n debe"):
        oraculo.estimate_prior_variance(n=malo)  # type: ignore[arg-type]


@pytest.mark.parametrize("malo", [1.5, "7", None])
def test_la_estimacion_rechaza_una_semilla_que_no_es_entera(malo: object):
    """La semilla es entera: cualquier otra cosa es un error del caller."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^seed debe"):
        oraculo.estimate_prior_variance(seed=malo)  # type: ignore[arg-type]


# ------------- oráculo: sesgo de inicialización, valor por cuadratura (6.2, 6.3, 6.6)


#: Medio ancho y nodos de las mallas que los tests construyen por su cuenta para integrar la
#: divergencia sin pedirle la malla al módulo. Cubren de sobra los ocho desvíos de la mixtura
#: de prueba (y los cuarenta de VE) con un paso bastante más fino que el mínimo necesario.
_MALLA_PROPIA = {"vp": (12.0, 601), "ve": (45.0, 601), "sub_vp": (12.0, 601)}


def _malla_de_puntos(medio_ancho: float, nodos: int) -> "torch.Tensor":
    """Nodos ``(N, 2)`` de una malla cuadrada centrada en el origen, armada acá.

    No usa ``QuadratureGrid`` ni ``integrate``: es la malla del test, para que la cuadratura de
    referencia no comparta una sola línea con la de producción.
    """
    eje = torch.linspace(-medio_ancho, medio_ancho, nodos, dtype=torch.float64)
    fila, columna = torch.meshgrid(eje, eje, indexing="ij")
    return torch.stack((columna, fila), dim=-1).reshape(-1, 2)


def _log_gaussiana_isotropica(
    puntos: "torch.Tensor", varianza_prior: float
) -> "torch.Tensor":
    """``log N(x; 0, v I)`` en dos dimensiones, escrita aparte con ``math``."""
    return -math.log(2.0 * math.pi * varianza_prior) - (puntos * puntos).sum(dim=-1) / (
        2.0 * varianza_prior
    )


def _kl_por_cuadratura_independiente(
    nombre: str,
    t: float,
    mixtura: ExactGaussianMixture,
    varianza_prior: float,
    *,
    con_termino_del_prior: bool = True,
) -> float:
    """``∫ p_t (log p_t − log q)`` integrada con la malla y la mixtura del propio test.

    Segunda implementación de punta a punta: los parámetros ruideados salen de la forma cerrada
    de ``(alpha_t, sigma_t)`` escrita en el test, la densidad se suma en el dominio **lineal**
    con el álgebra lineal genérica de torch, la malla la arma el test y la suma de Riemann se
    escribe acá. No comparte código con la producción, así que no es una tautología.

    Args:
        nombre: Variante cuya forma cerrada usa el test.
        t: Instante en el que se evalúa la divergencia.
        mixtura: Mixtura de parámetros exactos.
        varianza_prior: Varianza de la distribución de partida.
        con_termino_del_prior: Si es ``False`` integra solo ``∫ p log p`` —es decir menos la
            entropía—, que es la variante equivocada contra la que se discrimina.
    """
    medio_ancho, nodos = _MALLA_PROPIA[nombre]
    puntos = _malla_de_puntos(medio_ancho, nodos)
    medias, covs = _params_ruideados(nombre, t, mixtura)
    densidad = _mixtura(mixtura.weights_.tolist(), medias, covs)(puntos)
    integrando = torch.log(densidad)
    if con_termino_del_prior:
        integrando = integrando - _log_gaussiana_isotropica(puntos, varianza_prior)
    paso = 2.0 * medio_ancho / (nodos - 1)
    return float((densidad * integrando).sum() * paso * paso)


def _tolerancia_esperada(grid: QuadratureGrid, masa: float) -> float:
    """La fórmula de la tolerancia, reescrita acá a partir de los observables de la malla.

    ``paso_relativo · (paso_relativo + |masa − 1|)`` con
    ``paso_relativo = spacing / (2 half_width)``. Se escribe en el test para que una tolerancia
    elegida a mano —una constante, o cualquier cosa que no dependa de la malla y de la masa— no
    pueda pasar.
    """
    paso_relativo = grid.spacing / (2.0 * grid.half_width)
    return paso_relativo * (paso_relativo + abs(masa - 1.0))


def _medias_contraidas(
    oraculo: MixtureOracle, mixtura: ExactGaussianMixture, t: float
) -> "torch.Tensor":
    """``alpha_t mu_k`` reconstruidas desde los accesores públicos del oráculo."""
    tt = torch.tensor([t], dtype=torch.float64)
    alpha, _ = oraculo.marginal_params(tt)
    return float(alpha[0, 0]) * torch.as_tensor(mixtura.means_, dtype=torch.float64)


def _malla_del_sesgo(
    oraculo: MixtureOracle,
    mixtura: ExactGaussianMixture,
    t: float,
    varianza_prior: float,
) -> QuadratureGrid:
    """La malla que el sesgo debe usar: covarianzas **en** ``t`` y ``prior_std = sqrt(v)``."""
    tt = torch.tensor([t], dtype=torch.float64)
    return auto_grid(
        means=_medias_contraidas(oraculo, mixtura, t),
        covariances=oraculo.component_covariances(tt)[0],
        prior_std=math.sqrt(varianza_prior),
    )


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_valor_de_referencia_coincide_con_una_cuadratura_independiente(nombre: str):
    """Criterio 6.2: el valor es ``∫ p_T (log p_T − log q)``, contra una cuadratura aparte.

    La referencia usa su propia malla (más ancha y más fina que la automática), su propia suma
    de Riemann y la mixtura sumada en el dominio lineal, así que coincidir a ``rel=1e-9`` es una
    verificación cruzada y no una tautología. La tolerancia del test es holgada respecto del
    acuerdo observado (``5e-12`` en el peor caso) y órdenes de magnitud más fina que cualquier
    error estructural: olvidar el término del prior, no contraer las medias o no propagar las
    covarianzas cambia el número por completo.
    """
    mixtura = _mixtura_exacta()
    varianza = _VARIANZA_DEL_PRIOR[nombre]
    oraculo = MixtureOracle(mixtura, make_sde(nombre))

    reporte = oraculo.initialization_bias(varianza, t=1.0)

    assert reporte.value == pytest.approx(
        _kl_por_cuadratura_independiente(nombre, 1.0, mixtura, varianza), rel=1e-9
    )


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
@pytest.mark.parametrize("valor", [0.5, 1.0])
def test_la_cota_domina_al_valor_de_referencia_y_ambos_son_no_negativos(
    nombre: str, valor: float
):
    """Criterios 6.3 y 6.6: ``bound >= value >= 0`` en las tres SDEs.

    Es la validación mutua de las dos mitades del reporte: la cota por convexidad sale de la
    aritmética de los parámetros y el valor de una cuadratura, así que dominar es una propiedad
    matemática que ninguna de las dos implementaciones puede fingir. En la mixtura de prueba la
    cota queda entre ``1.2`` y ``3.6`` veces el valor, así que la desigualdad tiene margen y no
    se apoya en el redondeo.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))

    reporte = oraculo.initialization_bias(_VARIANZA_DEL_PRIOR[nombre], t=valor)

    assert reporte.value >= 0.0
    assert math.isfinite(reporte.value)
    assert reporte.bound >= reporte.value


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_con_una_sola_componente_la_cota_y_el_valor_coinciden(nombre: str):
    """Criterio 6.4: con ``K = 1`` la desigualdad de convexidad se satura.

    Chequeo barato de consistencia entre las dos mitades del reporte —la tendencia completa
    frente al horizonte es otro trabajo—: sin entropía de mezcla que acotar, la cota **es** la
    divergencia, así que la diferencia entre las dos debe caber en la tolerancia que el propio
    reporte declara (medida: ``5e-18``, contra una tolerancia del orden de ``1e-4``).
    """
    oraculo = MixtureOracle(_mixtura_de_una_componente(), make_sde(nombre))

    reporte = oraculo.initialization_bias(_VARIANZA_DEL_PRIOR[nombre], t=1.0)

    assert abs(reporte.bound - reporte.value) <= reporte.tolerance


def test_el_valor_de_referencia_incluye_el_termino_de_la_distribucion_de_partida():
    """El integrando es ``p (log p − log q)``, no ``p log p``.

    Sin el término del prior lo que se integra es **menos la entropía**, que para esta mixtura
    da ``-2.84`` frente a una divergencia de ``2.6e-5``: no solo cambia de escala, cambia de
    signo. El test fija las dos caras, así que la versión que se olvida de ``log q`` no puede
    satisfacer los dos asserts a la vez.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    con_prior = _kl_por_cuadratura_independiente("vp", 1.0, mixtura, 1.0)
    sin_prior = _kl_por_cuadratura_independiente(
        "vp", 1.0, mixtura, 1.0, con_termino_del_prior=False
    )

    reporte = oraculo.initialization_bias(1.0, t=1.0)

    assert sin_prior < 0.0 < con_prior
    assert reporte.value == pytest.approx(con_prior, rel=1e-9)


def test_la_malla_del_valor_cubre_tambien_la_distribucion_de_partida():
    """El dominio se amplía con ``prior_std = sqrt(prior_variance)``, no con la varianza.

    Con una varianza de partida de ``100`` el desvío es ``10``: la malla correcta cubre
    ``±80`` con 960 nodos, mientras que pasarle la varianza sin raíz pediría ``±800`` y se
    chocaría contra el tope de puntos. Las dos mallas dejan tolerancias que se separan por un
    factor de casi cuatro, así que el reporte delata cuál se usó.
    """
    varianza = 100.0
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    tt = torch.tensor([1.0], dtype=torch.float64)
    medias = _medias_contraidas(oraculo, mixtura, 1.0)
    covarianzas = oraculo.component_covariances(tt)[0]
    con_raiz = auto_grid(
        means=medias, covariances=covarianzas, prior_std=math.sqrt(varianza)
    )
    sin_raiz = auto_grid(means=medias, covariances=covarianzas, prior_std=varianza)

    reporte = oraculo.initialization_bias(varianza, t=1.0)

    assert not con_raiz.truncated
    assert sin_raiz.truncated
    assert reporte.tolerance == pytest.approx(
        _tolerancia_esperada(con_raiz, reporte.mass), rel=1e-12
    )
    assert reporte.tolerance != pytest.approx(
        _tolerancia_esperada(sin_raiz, reporte.mass), rel=1e-3
    )


def test_la_malla_del_valor_se_dimensiona_con_las_covarianzas_en_el_tiempo():
    """La malla sale de ``Sigma_k(t)``, no de las covarianzas de los datos.

    Con VE en el horizonte el ruido del kernel aporta ``sigma_T² = 25``: la malla correcta se
    resuelve con 103 nodos, mientras que dimensionarla con las covarianzas sin propagar pediría
    2401. La tolerancia reportada distingue los dos casos por más de un orden de magnitud.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("ve"))
    tt = torch.tensor([1.0], dtype=torch.float64)
    medias = _medias_contraidas(oraculo, mixtura, 1.0)
    en_el_tiempo = auto_grid(
        means=medias,
        covariances=oraculo.component_covariances(tt)[0],
        prior_std=5.0,
    )
    sin_propagar = auto_grid(
        means=medias,
        covariances=torch.as_tensor(mixtura.covariances_, dtype=torch.float64),
        prior_std=5.0,
    )

    reporte = oraculo.initialization_bias(25.0, t=1.0)

    assert sin_propagar.n_points > 10 * en_el_tiempo.n_points
    assert reporte.tolerance == pytest.approx(
        _tolerancia_esperada(en_el_tiempo, reporte.mass), rel=1e-12
    )


def test_el_reporte_trae_las_seis_cantidades_con_numeros_reales():
    """Criterio 6.2: el reporte publica cota, valor, tolerancia, masa, prior y horizonte.

    Ninguna viaja vacía: el valor de referencia ya no es una promesa, y la masa integrada y la
    tolerancia son el diagnóstico que lo hace utilizable.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("ve"))

    reporte = oraculo.initialization_bias(25.0, t=1.0)

    assert isinstance(reporte, BiasReport)
    for nombre_campo in ("bound", "value", "tolerance", "mass", "prior_variance", "horizon"):
        medida = getattr(reporte, nombre_campo)
        assert isinstance(medida, float)
        assert math.isfinite(medida)
    assert reporte.mass == pytest.approx(1.0, abs=1e-8)
    assert reporte.tolerance > 0.0
    assert reporte.prior_variance == pytest.approx(25.0)
    assert reporte.horizon == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("nombre", "escala"), [("vp", 0.25), ("ve", 1.0), ("sub_vp", 4.0)]
)
def test_el_valor_de_referencia_es_cero_cuando_las_distribuciones_coinciden(
    nombre: str, escala: float
):
    """Criterio 6.6: el valor se anula —sin irse a negativo— cuando ``p_T`` **es** el prior.

    El caso degenerado se construye a propósito: una sola componente centrada en el origen y
    una varianza de partida igual a la propagada ``alpha_T² escala + sigma_T²``, de modo que
    ``p_T`` y la distribución de partida sean **la misma** gaussiana y la divergencia sea
    exactamente cero. Ahí el integrando es una cancelación entre dos log-densidades escritas con
    fórmulas distintas, así que la cuadratura devuelve ruido de redondeo con signo arbitrario: en
    estas tres combinaciones sale **negativo** (del orden de ``-1e-16``), y publicarlo sería
    reportar una divergencia imposible que además el propio reporte rechazaría.
    """
    centrada = ExactGaussianMixture(
        2,
        weights=[1.0],
        means=[[0.0, 0.0]],
        covariances=[_identidad(escala).tolist()],
        seed=0,
    )
    oraculo = MixtureOracle(centrada, make_sde(nombre))
    tt = torch.tensor([1.0], dtype=torch.float64)
    alpha, sigma = oraculo.marginal_params(tt)
    varianza = float(alpha[0, 0]) ** 2 * escala + float(sigma[0, 0]) ** 2

    reporte = oraculo.initialization_bias(varianza, t=1.0)

    assert reporte.value >= 0.0
    assert reporte.value < 1e-12


def test_el_valor_de_referencia_honra_la_malla_explicita():
    """La malla explícita se usa tal cual, con la misma convención que la masa integrada.

    Se le pasa una malla más fina que la automática: el valor tiene que coincidir con el
    automático (las dos mallas alcanzan, y la suma de Riemann sobre gaussianas converge
    espectralmente) pero la tolerancia declarada tiene que **bajar**, que es lo que delata que
    el argumento se usó y no se ignoró.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    automatica = _malla_del_sesgo(oraculo, mixtura, 1.0, 1.0)
    fina = QuadratureGrid(
        half_width=automatica.half_width, n_points=4 * automatica.n_points
    )

    por_defecto = oraculo.initialization_bias(1.0, t=1.0)
    explicita = oraculo.initialization_bias(1.0, t=1.0, grid=fina)

    assert explicita.value == pytest.approx(por_defecto.value, rel=1e-9)
    assert explicita.tolerance < por_defecto.tolerance / 10.0
    assert explicita.tolerance == pytest.approx(
        _tolerancia_esperada(fina, explicita.mass), rel=1e-12
    )


def test_la_tolerancia_del_valor_se_afina_al_refinar_la_malla():
    """La tolerancia es un número **computado** sobre la malla, no una constante elegida a mano.

    Al duplicar la resolución cae por un factor de cuatro —es de segundo orden en el paso—, así
    que una constante no puede reproducir la secuencia. Se compara además contra la fórmula
    reescrita en el test a partir del paso y de la masa reportada.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    automatica = _malla_del_sesgo(oraculo, mixtura, 1.0, 1.0)
    tolerancias = []
    for factor in (1, 2, 4):
        malla = QuadratureGrid(
            half_width=automatica.half_width,
            n_points=factor * (automatica.n_points - 1) + 1,
        )
        reporte = oraculo.initialization_bias(1.0, t=1.0, grid=malla)
        assert reporte.tolerance == pytest.approx(
            _tolerancia_esperada(malla, reporte.mass), rel=1e-12
        )
        tolerancias.append(reporte.tolerance)

    assert tolerancias[1] == pytest.approx(tolerancias[0] / 4.0, rel=1e-6)
    assert tolerancias[2] == pytest.approx(tolerancias[0] / 16.0, rel=1e-6)


@pytest.mark.parametrize(
    ("medio_ancho", "nodos"), [(0.5, 8), (2.0, 32), (20.0, 64)]
)
def test_una_malla_insuficiente_falla_en_lugar_de_devolver_un_numero(
    medio_ancho: float, nodos: int
):
    """La masa es el autochequeo: si se aparta de uno más allá de la tolerancia, se levanta.

    Un valor de referencia calculado sobre una malla que no cubre la densidad es peor que
    ninguno, porque viaja con la misma pinta que uno bueno. Los tres casos cubren el rango: un
    dominio diminuto (masa ``0.004``), uno que corta la mixtura por la mitad (masa ``0.68``) y
    uno amplio pero demasiado grueso (masa ``1.0015``, que ya excede lo que una malla de esa
    resolución debería producir).
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    malla = QuadratureGrid(half_width=medio_ancho, n_points=nodos)

    with pytest.raises(ValueError, match=r"^la masa integrada"):
        oraculo.initialization_bias(1.0, t=0.1, grid=malla)


def test_la_masa_reportada_es_la_de_la_misma_malla_que_el_valor():
    """La masa publicada es el autochequeo **de esa** cuadratura, no de otra malla.

    Se compara contra ``total_mass`` sobre la misma malla explícita, y **bit a bit**: la
    cuadratura es determinística, así que la misma malla tiene que dar el mismo número exacto.
    La comparación exacta es lo que le da dientes al test — dos mallas suficientes dan masa uno
    dentro del redondeo (acá ``0.9999999999999987`` contra ``1.0000000000000004``), así que
    cualquier tolerancia razonable dejaría pasar un reporte que integrara la masa en la malla
    automática mientras el valor sale de la explícita.
    """
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("sub_vp"))
    automatica = _malla_del_sesgo(oraculo, mixtura, 1.0, 1.0)
    fina = QuadratureGrid(
        half_width=1.5 * automatica.half_width, n_points=2 * automatica.n_points
    )

    reporte = oraculo.initialization_bias(1.0, t=1.0, grid=fina)

    assert oraculo.total_mass(1.0, grid=automatica) != oraculo.total_mass(1.0, grid=fina)
    assert reporte.mass == oraculo.total_mass(1.0, grid=fina)


# ------- oráculo: tendencias del sesgo y degradación de la cuadratura (6.4, 6.8, 6.9)


#: Variantes que **contraen la media** hacia el origen (``alpha_t`` decreciente): son las
#: únicas de las que el criterio 6.8 pide monotonía frente al horizonte. VE queda afuera a
#: propósito, porque su ``alpha_t`` es constantemente uno y su tendencia es la del 6.9.
_SDES_QUE_CONTRAEN_LA_MEDIA = ["vp", "sub_vp"]

#: Horizontes crecientes del barrido de monotonía. Cuatro puntos alcanzan para fijar el
#: sentido de la tendencia (el criterio pide monotonía, no una ley de decaimiento) y dejan
#: el barrido en décimas de segundo; un muestreo denso solo repetiría la misma información
#: pagando cuadraturas de más.
_HORIZONTES_CRECIENTES = (0.1, 0.3, 0.6, 1.0)

#: Escalas de ruido máximo con las que se construyen las VE del barrido del criterio 6.9.
#: Arrancan **por debajo** de la escala de los datos de la mixtura de prueba (medias de
#: módulo ``1.6`` y ``2.2``) y terminan una década por encima, así que el barrido cubre el
#: régimen en el que el ruido todavía no domina —donde el sesgo tiene que seguir siendo
#: visible— y el tramo en el que empieza a dominar. No se estira más arriba porque con
#: ``sigma_max = 50`` el sesgo (``2.4e-4``) ya cae al orden de la tolerancia de la
#: cuadratura y la comparación dejaría de ser significativa.
_SIGMA_MAX_CRECIENTES = (2.0, 5.0, 10.0, 20.0)

#: Horizontes en los que se comprueba la saturación de la cota con una sola componente.
#: VE se evalúa **solo en el horizonte**: en tiempos intermedios su ``sigma_t`` es chico
#: frente al desvío ``5`` del prior que ensancha el dominio, así que la malla automática se
#: vuelve enorme y cada cuadratura cuesta segundos en vez de centésimas.
_HORIZONTES_DE_SATURACION = {
    "vp": (0.2, 0.5, 1.0),
    "sub_vp": (0.2, 0.5, 1.0),
    "ve": (1.0,),
}

#: Autovalores de la componente con la que se fuerza una malla insuficiente: anisotropía
#: ``1e6``, es decir la dirección angosta con desvío ``1e-3``.
#:
#: El valor no es arbitrario. Las dos señales de degradación **no disparan juntas**: con el
#: tope de ``4096`` nodos por eje la malla ya queda marcada como truncada con
#: ``lambda_min = 1e-4`` (2.56 nodos por desvío) mientras la masa sigue dando uno con error
#: ``2e-15``, porque las sumas de Riemann sobre gaussianas convergen espectralmente y
#: aguantan mallas groseras; la masa recién se aparta en ``1e-5`` (``5e-6``) y se derrumba en
#: ``1e-6`` (``0.462``). Para que el truncamiento **y** la pérdida de masa coincidan en el
#: mismo caso —que es lo que hace fallar al sesgo por la ruta automática— hace falta
#: ``lambda_min <= 1e-6``.
_LAMBDA_MAYOR_ANGOSTA, _LAMBDA_MENOR_ANGOSTA = 1.0, 1e-6


def _mixtura_de_anisotropia_extrema() -> ExactGaussianMixture:
    """Mixtura de una componente con anisotropía ``1e6``, centrada en el origen.

    Los autovalores se escriben acá directamente (``1.0`` y ``1e-6``) en lugar de pedirlos por
    la palanca ``anisotropy`` de los constructores de geometría: lo que este caso necesita
    fijar es el **autovalor menor**, que es lo que dimensiona el paso de la malla, y con la
    palanca habría que despejarlo de la escala global. Una sola componente, además, porque el
    caso ya cuesta una cuadratura de dieciséis millones de nodos y cada componente extra la
    encarece sin agregar nada a la afirmación.
    """
    return ExactGaussianMixture(
        2,
        weights=[1.0],
        means=[[0.0, 0.0]],
        covariances=[_diagonal(_LAMBDA_MAYOR_ANGOSTA, _LAMBDA_MENOR_ANGOSTA).tolist()],
        seed=0,
    )


def _alpha_en(oraculo: MixtureOracle, t: float) -> float:
    """``alpha_t`` leído del contrato marginal que publica el oráculo."""
    alpha, _ = oraculo.marginal_params(torch.tensor([t], dtype=torch.float64))
    return float(alpha[0, 0])


@pytest.mark.parametrize("nombre", _SDES_QUE_CONTRAEN_LA_MEDIA)
def test_el_sesgo_decrece_al_crecer_el_horizonte_en_las_que_contraen_la_media(
    nombre: str,
):
    """Criterio 6.8: con la media contrayéndose, el sesgo cae al alargar el horizonte.

    Es la lectura cualitativa que el trabajo quiere poder afirmar con un número: cuanto más
    tiempo corre el forward, más se parece ``p_t`` al ruido de partida. Los asserts son
    **direccionales** —cada término tiene que ser estrictamente menor que el anterior, no
    solo distinto—, así que invertir el sentido de la dependencia con el horizonte rompe el
    test en lugar de pasarlo de casualidad.

    Se fija la tendencia de las **dos** mitades del reporte (la cota, que es aritmética
    sobre los parámetros, y el valor, que sale de la cuadratura) porque decaen por el mismo
    motivo y una sola no distinguiría un decaimiento real de una malla que se degrada. Y se
    fija además la causa: ``alpha_t`` decreciente. La caída medida es de cuatro o cinco
    órdenes de magnitud entre ``t = 0.1`` y el horizonte, así que el margen sobra sobre la
    tolerancia de la cuadratura (``1e-4``).
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))

    reportes = [
        oraculo.initialization_bias(_VARIANZA_DEL_PRIOR[nombre], t=t)
        for t in _HORIZONTES_CRECIENTES
    ]

    valores = [r.value for r in reportes]
    cotas = [r.bound for r in reportes]
    alphas = [_alpha_en(oraculo, t) for t in _HORIZONTES_CRECIENTES]
    for i in range(len(_HORIZONTES_CRECIENTES) - 1):
        assert alphas[i + 1] < alphas[i], f"alpha_t no contrae la media: {alphas}"
        assert valores[i + 1] < valores[i], f"el valor no decrece: {valores}"
        assert cotas[i + 1] < cotas[i], f"la cota no decrece: {cotas}"
    assert valores[0] > 1e3 * valores[-1]


def test_el_sesgo_de_la_que_solo_agranda_la_varianza_persiste_en_el_horizonte():
    """Criterio 6.9: VE llega al horizonte con un sesgo que **no** se apagó.

    VE no contrae la media (``alpha_T`` es exactamente uno), así que en el horizonte su
    ``p_T`` sigue teniendo las medias de los datos donde estaban y solo las tapó con ruido
    de desvío ``sigma_max = 5``: con datos de escala ``2`` eso no alcanza para que ``p_T``
    sea el prior, y el sesgo queda tres órdenes de magnitud por encima del de VP en el mismo
    horizonte (``2.8e-2`` contra ``2.6e-5``). El contraste es la afirmación: no es que VE sea
    "peor", es que su sesgo **persiste** mientras el de VP ya se apagó (el de VP, medido, cae
    por debajo de la propia tolerancia de su cuadratura). Que el de VE supere a su tolerancia
    por un factor de ``290`` es lo que hace la persistencia medible y no ruido numérico.
    """
    mixtura = _mixtura_exacta()
    solo_varianza = MixtureOracle(mixtura, make_sde("ve"))
    contrae = MixtureOracle(mixtura, make_sde("vp"))

    reporte_ve = solo_varianza.initialization_bias(_VARIANZA_DEL_PRIOR["ve"])
    reporte_vp = contrae.initialization_bias(_VARIANZA_DEL_PRIOR["vp"])

    assert _alpha_en(solo_varianza, reporte_ve.horizon) == 1.0
    assert _alpha_en(contrae, reporte_vp.horizon) < 1e-2
    assert reporte_ve.value > 10.0 * reporte_ve.tolerance
    assert reporte_ve.value > 100.0 * reporte_vp.value


def test_el_sesgo_de_la_que_solo_agranda_la_varianza_cae_al_crecer_su_ruido_maximo():
    """Criterio 6.9: el sesgo de VE decrece a medida que su escala de ruido crece.

    La otra mitad del criterio: el sesgo persiste *mientras* el ruido no domine, y la forma
    de comprobar que eso es lo que lo sostiene es hacerlo dominar. Se construyen VE con
    ``sigma_max`` creciente y se le pasa a cada una la varianza de **su** prior
    (``sigma_max²``), que es lo único que hace comparable la secuencia. Los asserts son
    direccionales y se le pide además a cada término quedar por encima de la tolerancia de
    su cuadratura, así que la caída no puede confundirse con la secuencia hundiéndose en el
    ruido numérico (el término más chico la supera por un factor de ``14``).
    """
    mixtura = _mixtura_exacta()

    reportes = [
        MixtureOracle(mixtura, make_sde("ve", sigma_max=sigma_max)).initialization_bias(
            sigma_max**2
        )
        for sigma_max in _SIGMA_MAX_CRECIENTES
    ]

    valores = [r.value for r in reportes]
    cotas = [r.bound for r in reportes]
    for reporte in reportes:
        assert reporte.value > reporte.tolerance
    for i in range(len(_SIGMA_MAX_CRECIENTES) - 1):
        assert valores[i + 1] < valores[i], f"el valor no decrece: {valores}"
        assert cotas[i + 1] < cotas[i], f"la cota no decrece: {cotas}"


def test_la_tendencia_del_ruido_creciente_exige_la_varianza_del_prior_de_cada_sde():
    """La tendencia del 6.9 compara contra ``N(0, sigma_max² I)``, no contra ``N(0, I)``.

    ``prior_variance`` es un dato del llamador y el oráculo lo toma tal cual: si al barrido
    de ``sigma_max`` se le pasara la varianza unitaria de VP, la secuencia no solo dejaría de
    decrecer sino que **crecería** monótonamente (``3.7`` → ``395``), porque cada VE se estaría
    comparando contra un prior mucho más angosto que el suyo. El test fija esa inversión para
    que el barrido de arriba no pueda "arreglarse" pasándole la varianza equivocada: las dos
    tendencias son incompatibles y solo una de las dos lecturas puede estar en verde.
    """
    mixtura = _mixtura_exacta()

    valores = [
        MixtureOracle(mixtura, make_sde("ve", sigma_max=sigma_max))
        .initialization_bias(_VARIANZA_DEL_PRIOR["vp"])
        .value
        for sigma_max in _SIGMA_MAX_CRECIENTES
    ]

    for i in range(len(_SIGMA_MAX_CRECIENTES) - 1):
        assert valores[i + 1] > valores[i], f"la secuencia equivocada no crece: {valores}"


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_con_una_sola_componente_la_cota_iguala_al_valor_en_todo_el_horizonte(
    nombre: str,
):
    """Criterio 6.4: sin entropía de mezcla que acotar, la cota **es** la divergencia.

    Con ``K = 1`` la desigualdad de convexidad no acota nada: los dos lados son la misma KL
    gaussiana-gaussiana, que sí tiene forma cerrada, así que la cota y el valor tienen que
    coincidir en **todo** el horizonte y no solo en un instante afortunado. El acuerdo medido
    es de ``2e-12`` relativo en el peor punto —el residuo de la cuadratura, no una
    coincidencia—, y se compara contra la tolerancia que el propio reporte declara.

    El segundo assert es lo que hace informativa a la igualdad: en la mixtura de dos
    componentes la brecha es real (la cota queda entre ``1.4`` y ``3.5`` veces el valor), así
    que la coincidencia con ``K = 1`` no puede explicarse por una cota que devuelva el valor
    de la cuadratura.
    """
    varianza = _VARIANZA_DEL_PRIOR[nombre]
    una_componente = MixtureOracle(_mixtura_de_una_componente(), make_sde(nombre))
    dos_componentes = MixtureOracle(_mixtura_exacta(), make_sde(nombre))

    for t in _HORIZONTES_DE_SATURACION[nombre]:
        saturada = una_componente.initialization_bias(varianza, t=t)
        con_brecha = dos_componentes.initialization_bias(varianza, t=t)

        assert saturada.bound == pytest.approx(saturada.value, rel=1e-9)
        assert abs(saturada.bound - saturada.value) <= saturada.tolerance
        assert con_brecha.bound > 1.2 * con_brecha.value


def test_una_anisotropia_alta_marca_la_malla_y_hace_fallar_el_sesgo():
    """Malla insuficiente: la degradación se delata y el sesgo se niega a devolver un número.

    Es el caso que el barrido de anisotropía del laboratorio 2D va a pisar: con la dirección
    angosta de desvío ``1e-3`` la malla automática necesitaría ``96000`` nodos por eje, el
    tope de ``4096`` la recorta a ``0.26`` nodos por desvío y la suma de Riemann se queda con
    menos de la mitad de la masa. Las tres señales que el diseño pide quedan fijadas: la
    malla marcada como truncada, la masa apartada de uno, y ``initialization_bias``
    levantando en lugar de devolver un número que viajaría con la misma pinta que uno bueno.

    Se evalúa en ``t = 0``, donde el kernel no agrega nada y la covarianza integrada es la de
    los datos: así la anisotropía que rompe la malla es exactamente la declarada y no una que
    dependa del schedule. La masa se lee del mensaje de la excepción a propósito, para no
    pagar una segunda cuadratura de dieciséis millones de nodos.
    """
    angosta = _mixtura_de_anisotropia_extrema()
    oraculo = MixtureOracle(angosta, make_sde("vp"))
    covarianzas = torch.as_tensor(angosta.covariances_, dtype=torch.float64)
    autovalores = torch.linalg.eigvalsh(covarianzas)
    malla = auto_grid(
        means=torch.as_tensor(angosta.means_, dtype=torch.float64),
        covariances=covarianzas,
        prior_std=1.0,
    )
    assert float(autovalores.max() / autovalores.min()) == pytest.approx(1e6, rel=1e-9)
    assert malla.truncated

    with pytest.raises(ValueError, match=r"^la masa integrada") as excepcion:
        oraculo.initialization_bias(1.0, t=0.0)

    reportada = re.match(r"la masa integrada dio (\S+) ", str(excepcion.value))
    assert reportada is not None, str(excepcion.value)
    assert abs(float(reportada.group(1)) - 1.0) > 0.5


# --------------- oráculo: verificación contra fuentes externas (7.1, 7.2, 7.3, 7.4)


#: Tiempos de la verificación independiente: el **mínimo admitido** (``t = 0``, donde VP y
#: sub-VP no ruidean nada y VE se queda en el piso ``sigma_min`` de su schedule), el régimen
#: casi singular, dos intermedios y el **horizonte** (criterio 7.4).
_TIEMPOS_VERIFICACION = [0.0, 1e-4, 0.05, 0.5, 1.0]


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_los_tiempos_verificados_son_los_extremos_del_horizonte(nombre: str):
    """Criterio 7.4: los extremos que barren estos tests son los del horizonte de verdad.

    Fija las dos puntas para que la parametrización no se despegue del proceso sin que nadie
    se entere: el tiempo final es el ``T`` que publica la SDE, y el mínimo es ``0`` porque es
    el borde de lo admitido —el oráculo lo acepta y rechaza cualquier cosa por debajo—.
    """
    sde = make_sde(nombre)
    oraculo = MixtureOracle(_mixtura_exacta(), sde)

    assert max(_TIEMPOS_VERIFICACION) == sde.T
    assert min(_TIEMPOS_VERIFICACION) == 0.0
    assert torch.isfinite(oraculo.log_prob(_PUNTOS, _tiempo(0.0, _PUNTOS.shape[0]))).all()
    with pytest.raises(ValueError, match=r"^t debe ser finito y no negativo"):
        oraculo.log_prob(_PUNTOS, _tiempo(-1e-12, _PUNTOS.shape[0]))


# ------------------------------------------- diferencias finitas de la log-densidad (7.2)


#: Paso de las diferencias centradas, en unidades de la escala local de la densidad.
#:
#: Es el compromiso clásico del método: la truncación del esquema centrado va como
#: ``h²·|∂³ log p|`` y el redondeo como ``eps·|log p|/h``, así que el óptimo cae cerca de
#: ``h ≈ L·eps^{1/3} ≈ 6e-6·L`` con ``L`` la escala de longitud de la densidad y ``eps`` el
#: épsilon de la doble precisión. Se toma ``1e-5·L``, del mismo orden que el óptimo y **atado
#: a la escala local**, que es lo que mantiene el paso sensato tanto en el horizonte (``L ≈ 5``
#: en VE) como en el régimen casi singular (``L ≈ 1e-3`` en sub-VP con ``t = 1e-4``).
_PASO_RELATIVO = 1e-5

#: Tolerancia **relativa** de las diferencias centradas, medida contra el mayor score del lote.
#:
#: Relativa y no absoluta porque el score exacto crece como ``1/sigma_t²`` cuando el tiempo
#: baja: en las corridas de calibración su magnitud va de ``0.15`` (VE en el horizonte) a
#: ``2e3`` (sub-VP con componentes angostas y ``t = 1e-4``), así que un ``atol`` único sería
#: laxísimo en un extremo e imposible en el otro. El acuerdo observado con este paso es de
#: ``1e-10`` relativo en la mixtura moderada y ``7e-10`` en la angosta —el residuo esperable
#: del método, contra el ``1e-15`` que da autograd—, así que ``1e-7`` deja unos dos órdenes de
#: margen y sigue siendo discriminante: cualquier error estructural del score se aparta en
#: porcentajes, no en partes por diez millones.
_RTOL_DIFERENCIAS = 1e-7


def _escala_local(oraculo: MixtureOracle, t: float) -> float:
    """Escala de longitud de ``p_t``: el desvío de la dirección más angosta de la mixtura.

    Es ``min_k sqrt(lambda_min(Sigma_k(t)))``, el mismo criterio con el que la cuadratura
    dimensiona su paso, y lo que hace que el paso de las diferencias finitas sea del orden
    correcto en todo el horizonte.
    """
    tt = torch.tensor([t], dtype=torch.float64)
    autovalores = torch.linalg.eigvalsh(oraculo.component_covariances(tt))
    return float(autovalores.min().sqrt())


def _score_por_diferencias_centradas(
    oraculo: MixtureOracle, x: "torch.Tensor", t: float, paso: float
) -> "torch.Tensor":
    """Gradiente de ``log p_t`` por diferencias centradas, coordenada por coordenada.

    Es la fuente **externa** al score del criterio 7.2: no toca ``score`` ni autograd, solo
    evalúa la log-densidad en ``x ± h e_j``.

    Args:
        oraculo: Oráculo cuya log-densidad se deriva numéricamente.
        x: Puntos de evaluación, de forma ``(B, 2)``.
        t: Tiempo escalar, el mismo para todas las filas.
        paso: Paso ``h`` de la diferencia centrada.

    Returns:
        Tensor ``(B, 2)`` con el gradiente numérico.
    """
    tiempos = torch.full((x.shape[0],), t, dtype=x.dtype)
    columnas = []
    for eje in range(x.shape[1]):
        desplazamiento = torch.zeros_like(x)
        desplazamiento[:, eje] = paso
        adelante = oraculo.log_prob(x + desplazamiento, tiempos)
        atras = oraculo.log_prob(x - desplazamiento, tiempos)
        columnas.append((adelante - atras) / (2.0 * paso))
    return torch.stack(columnas, dim=1)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_score_coincide_con_las_diferencias_finitas_de_la_log_densidad(nombre: str):
    """Criterios 4.6 y 7.2: el score contra el gradiente **numérico** de ``log p_t``.

    Es una fuente distinta de la del test de autograd: ahí el gradiente lo produce el mismo
    grafo de la log-densidad, acá sale de evaluarla en puntos desplazados, así que ninguna
    derivada analítica interviene. Cubre las tres SDEs y los cinco tiempos del horizonte
    —mínimo, casi singular, dos intermedios y final (7.4)—.

    Tolerancia: ``_RTOL_DIFERENCIAS`` relativo al mayor score del lote, justificada en el
    comentario de esa constante por el balance truncación/redondeo del esquema centrado.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))

    for valor in _TIEMPOS_VERIFICACION:
        paso = _PASO_RELATIVO * _escala_local(oraculo, valor)
        exacto = oraculo.score(_PUNTOS, _tiempo(valor, _PUNTOS.shape[0]))
        numerico = _score_por_diferencias_centradas(oraculo, _PUNTOS, valor, paso)

        escala = float(exacto.abs().max())
        error = float((numerico - exacto).abs().max())
        assert error <= _RTOL_DIFERENCIAS * escala, (
            f"{nombre} en t={valor}: las diferencias centradas (paso {paso:.3e}) se apartan "
            f"{error:.3e} del score exacto, cuya magnitud máxima es {escala:.3e}"
        )


def _mixtura_de_componentes_angostas() -> ExactGaussianMixture:
    """Dos modos casi puntuales: ``Sigma_k = 1e-6 I``, así que la escala la fija ``sigma_t``.

    Es el régimen en el que el score explota como ``1/sigma_t²``: con ``t`` chico la
    covarianza de los datos es despreciable frente al ruido del kernel, al contrario de lo que
    pasa con la mixtura moderada, donde ``Sigma_k(t) → Sigma_k`` y el score queda acotado.
    """
    cov = _identidad(1e-6).tolist()
    return ExactGaussianMixture(
        2,
        weights=[0.5, 0.5],
        means=[[-0.5, 0.0], [0.5, 0.0]],
        covariances=[cov, cov],
        seed=0,
    )


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_las_diferencias_finitas_cierran_donde_el_score_explota(nombre: str):
    """Criterios 4.7 y 7.2: la verificación numérica también vale con ``sigma_t`` diminuto.

    Con componentes de covarianza ``1e-6`` y tiempos de ``1e-4`` a ``1e-3`` el score llega a
    ``2e3``, y el test exige explícitamente que la magnitud sea grande para que el régimen no
    se pierda en silencio. El paso de las diferencias se encoge con la escala local, así que la
    comparación **relativa** cierra igual de fino que en la mixtura moderada.
    """
    oraculo = MixtureOracle(_mixtura_de_componentes_angostas(), make_sde(nombre))
    puntos = torch.tensor([[-0.5, 0.0], [0.0, 0.0], [0.4, 0.002]], dtype=torch.float64)

    for valor in [1e-4, 1e-3]:
        paso = _PASO_RELATIVO * _escala_local(oraculo, valor)
        exacto = oraculo.score(puntos, _tiempo(valor, puntos.shape[0]))
        numerico = _score_por_diferencias_centradas(oraculo, puntos, valor, paso)

        escala = float(exacto.abs().max())
        assert escala >= 10.0, f"{nombre} en t={valor}: el régimen singular no se activó"
        error = float((numerico - exacto).abs().max())
        assert error <= _RTOL_DIFERENCIAS * escala, (
            f"{nombre} en t={valor}: las diferencias centradas (paso {paso:.3e}) se apartan "
            f"{error:.3e} del score exacto, cuya magnitud máxima es {escala:.3e}"
        )


# ---------------------------------- una sola componente contra una segunda cuenta (7.1)


#: Componente única de la verificación ``K = 1``: autovalores, ángulo y media conocidos.
_K1_AUTOVALORES = (0.6, 0.05)
_K1_ANGULO = math.pi / 5
_K1_MEDIA = (0.3, -0.2)


def _rotada_en_doble(a: float, b: float, angulo: float) -> "torch.Tensor":
    """``diag(a, b)`` rotada, con la rotación armada en **doble** precisión.

    Hace falta una variante propia porque :func:`_rotada` construye el seno y el coseno
    pasando por ``torch.tensor(angulo)``, que es ``float32``: el ángulo efectivo de esa
    covarianza se aparta del pedido en unos ``1e-8``. La verificación por base propia compara
    contra un ángulo escrito con ``math``, así que necesita que los dos coincidan en doble
    precisión; para el resto de la suite la diferencia es irrelevante.
    """
    coseno, seno = math.cos(angulo), math.sin(angulo)
    rotacion = torch.tensor([[coseno, -seno], [seno, coseno]], dtype=torch.float64)
    return rotacion @ _diagonal(a, b) @ rotacion.T


def _mixtura_de_una_componente_rotada() -> ExactGaussianMixture:
    """``K = 1`` con una componente **rotada**: la base propia no es la canónica.

    La rotación es lo que le da valor a la referencia por base propia: con una covarianza
    diagonal cualquier transposición o cruce de ejes pasaría desapercibido.
    """
    cov = _rotada_en_doble(_K1_AUTOVALORES[0], _K1_AUTOVALORES[1], _K1_ANGULO)
    return ExactGaussianMixture(
        2, weights=[1.0], means=[list(_K1_MEDIA)], covariances=[cov.tolist()], seed=0
    )


def _gaussiana_en_su_base_propia(
    nombre: str, t: float, punto: tuple[float, float]
) -> tuple[float, tuple[float, float]]:
    """Log-densidad y score de una gaussiana 2D, derivados desde cero en su base propia.

    Segunda implementación **independiente** del criterio 7.1: no reordena la fórmula de la
    producción ni invierte ninguna matriz. Parte de que sumar ``sigma_t² I`` no mueve los
    autovectores, así que ``Sigma(t) = R diag(alpha²lambda_1 + sigma², alpha²lambda_2 +
    sigma²) Rᵀ``; en las coordenadas ``u = Rᵀ(x - alpha mu)`` la gaussiana **factoriza** en dos
    normales unidimensionales independientes, de las que se escriben a mano la log-densidad
    (suma de ``-½(log 2πv_i + u_i²/v_i)``) y la derivada (``-u_i/v_i``), y el score se rota de
    vuelta con ``R``. Todo con ``math`` y floats de Python.

    Args:
        nombre: Variante de SDE, para leer ``alpha_t`` y ``sigma_t`` de la forma cerrada
            escrita a mano en el test.
        t: Tiempo en el que se evalúa.
        punto: Punto ``(x, y)`` de evaluación.

    Returns:
        El par ``(log p_t, (score_x, score_y))``.
    """
    alpha, sigma = _alpha_sigma_cerrados(nombre, t)
    coseno, seno = math.cos(_K1_ANGULO), math.sin(_K1_ANGULO)

    dx = punto[0] - alpha * _K1_MEDIA[0]
    dy = punto[1] - alpha * _K1_MEDIA[1]
    # u = Rᵀ d: las coordenadas del punto en la base propia de la componente.
    u1 = coseno * dx + seno * dy
    u2 = -seno * dx + coseno * dy
    v1 = alpha * alpha * _K1_AUTOVALORES[0] + sigma * sigma
    v2 = alpha * alpha * _K1_AUTOVALORES[1] + sigma * sigma

    log_densidad = -0.5 * (math.log(2.0 * math.pi * v1) + u1 * u1 / v1)
    log_densidad += -0.5 * (math.log(2.0 * math.pi * v2) + u2 * u2 / v2)

    # Derivada de cada normal 1D y vuelta a la base canónica: score = R (-u_i/v_i).
    g1, g2 = -u1 / v1, -u2 / v2
    return log_densidad, (coseno * g1 - seno * g2, seno * g1 + coseno * g2)


#: Puntos de la verificación ``K = 1``: el origen, dos cercanos y uno de cola.
_PUNTOS_K1 = [(0.0, 0.0), (0.5, 0.1), (-1.0, 2.0), (2.0, -1.5)]


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_con_una_sola_componente_la_log_densidad_es_el_producto_de_dos_normales(nombre: str):
    """Criterio 7.1: ``log p_t`` con ``K = 1`` contra la cuenta hecha en la base propia.

    Tolerancia: ``rtol=1e-12`` sobre log-densidades de hasta ~65 (unos ``7e-11`` absolutos).
    Las dos cuentas son de doble precisión pero recorren caminos distintos —forma cerrada 2×2
    contra dos normales 1D rotadas—, y el desacuerdo observado es de ``4e-14``, así que la
    tolerancia deja tres órdenes de margen sobre el redondeo y queda muy por debajo de
    cualquier error estructural.
    """
    oraculo = MixtureOracle(_mixtura_de_una_componente_rotada(), make_sde(nombre))
    x = torch.tensor(_PUNTOS_K1, dtype=torch.float64)

    for valor in _TIEMPOS_VERIFICACION:
        esperada = torch.tensor(
            [_gaussiana_en_su_base_propia(nombre, valor, p)[0] for p in _PUNTOS_K1],
            dtype=torch.float64,
        )
        obtenida = oraculo.log_prob(x, _tiempo(valor, x.shape[0]))
        assert torch.allclose(obtenida, esperada, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_con_una_sola_componente_el_score_sale_de_dos_normales_rotadas(nombre: str):
    """Criterio 7.1: el score con ``K = 1`` contra la derivada hecha en la base propia.

    La referencia deriva cada normal 1D por separado y rota el resultado, así que no comparte
    con la producción ni la inversa de ``Sigma(t)`` ni su determinante.

    Tolerancia: ``rtol=1e-12`` sobre magnitudes de hasta ~41 (unos ``4e-11`` absolutos), contra
    un desacuerdo observado de ``3e-14``. Cubre el mínimo del horizonte y el tiempo final
    además de los intermedios (7.4).
    """
    oraculo = MixtureOracle(_mixtura_de_una_componente_rotada(), make_sde(nombre))
    x = torch.tensor(_PUNTOS_K1, dtype=torch.float64)

    for valor in _TIEMPOS_VERIFICACION:
        esperado = torch.tensor(
            [list(_gaussiana_en_su_base_propia(nombre, valor, p)[1]) for p in _PUNTOS_K1],
            dtype=torch.float64,
        )
        obtenido = oraculo.score(x, _tiempo(valor, x.shape[0]))
        assert obtenido.shape == x.shape
        assert torch.allclose(obtenido, esperado, rtol=1e-12, atol=1e-13)


def _score_marginal_isotropico(sde: ForwardSDE, mu: "torch.Tensor", sigma0: float):
    """El score analítico de ``N(mu, sigma0² I)`` tal como lo escribe la suite de samplers.

    Réplica del helper que la suite del Eje 2 ya usa para validar los cuatro samplers sobre
    las tres SDEs: lee ``alpha_t`` y ``sigma_t`` del contrato marginal sondeándolo con un
    vector de unos y devuelve ``-(x - alpha_t mu) / (alpha_t² sigma0² + sigma_t²)``. Se
    reescribe acá para no importar otra suite, y es isótropo: el denominador es un escalar por
    muestra.
    """

    def score(x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        unos = torch.ones(1, mu.shape[0], dtype=x.dtype)
        media_alpha, sigma_t = sde.marginal_prob(unos, t)
        alpha_t = media_alpha[:, :1]
        varianza_t = alpha_t**2 * sigma0**2 + sigma_t**2
        return -(x - alpha_t * mu) / varianza_t

    return score


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_con_una_sola_componente_el_score_coincide_con_el_de_la_suite_de_samplers(
    nombre: str,
):
    """Criterio 7.1 al pie de la letra: coincidir con el score gaussiano ya validado.

    El helper de la suite de samplers es el score contra el que se verificaron las doce celdas
    del Eje 2, así que hacer coincidir el oráculo con él es lo que permite reemplazar la red
    por el oráculo y seguir midiendo lo mismo. Se usa la misma gaussiana isótropa de esa suite
    (``mu = (1.5, -1)``, ``sigma0 = 0.5``).

    Tolerancia: ``rtol=1e-12``; el desacuerdo observado es de ``2e-15``, es decir redondeo de
    doble precisión, porque las dos cuentas son la misma matemática por caminos distintos.
    """
    mu = torch.tensor([1.5, -1.0], dtype=torch.float64)
    sigma0 = 0.5
    isotropica = ExactGaussianMixture(
        2,
        weights=[1.0],
        means=[mu.tolist()],
        covariances=[_identidad(sigma0**2).tolist()],
        seed=0,
    )
    sde = make_sde(nombre)
    oraculo = MixtureOracle(isotropica, sde)
    referencia = _score_marginal_isotropico(sde, mu, sigma0)
    x = torch.tensor(_PUNTOS_K1, dtype=torch.float64)

    for valor in _TIEMPOS_VERIFICACION:
        t = _tiempo(valor, x.shape[0])
        assert torch.allclose(oraculo.score(x, t), referencia(x, t), rtol=1e-12, atol=1e-13)


# ------------------------------- estadísticas empíricas del kernel forward (7.3, 7.4)


#: Tamaño de muestra de las verificaciones por Monte Carlo, según la convención del proyecto.
_N_MONTE_CARLO = 40_000

#: Cuántos errores estándar del estimador se admiten de desacuerdo.
#:
#: Es la tolerancia de Monte Carlo **declarada** de esta sección, y se expresa en errores
#: estándar en lugar de en porcentaje fijo porque las cantidades comparadas cambian de escala
#: con el tiempo: la media de ``p_t`` colapsa a ``0.006`` en VP con ``t = T`` (contra ``0.95``
#: en ``t = 0``) y un ``5%`` relativo sobre ella sería más estricto que el ruido de muestreo,
#: mientras que la varianza de VE crece hasta ``27.8`` y ese mismo ``5%`` sería laxísimo.
#: Cinco errores estándar equivalen a menos del ``5%`` relativo del convenio del proyecto en
#: todos los casos medidos, y los desacuerdos observados no pasan de ``1.6`` errores estándar
#: con las semillas fijas de estos tests.
_ERRORES_ESTANDAR = 5.0


def _momentos_propagados(
    oraculo: MixtureOracle, mixtura: ExactGaussianMixture, t: float
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Media y covarianza de ``p_t`` armadas con los parámetros que publica el oráculo.

    Son los momentos de la mixtura ruideada escritos a mano: ``m = Σ_k w_k alpha_t mu_k`` y
    ``S = Σ_k w_k (Sigma_k(t) + alpha_t² mu_k mu_kᵀ) − m mᵀ``. Al salir de
    ``marginal_params`` y ``component_covariances``, comparar contra las muestras del kernel
    forward pone a prueba esos dos accesores del oráculo.

    Args:
        oraculo: Oráculo del que se leen ``alpha_t`` y ``Sigma_k(t)``.
        mixtura: Mixtura de la que se leen los pesos verdaderos.
        t: Tiempo en el que se evalúan los momentos.

    Returns:
        El par ``(media, covarianza)``, de formas ``(2,)`` y ``(2, 2)``.
    """
    tt = torch.tensor([t], dtype=torch.float64)
    pesos = torch.as_tensor(mixtura.weights_, dtype=torch.float64)
    medias = _medias_contraidas(oraculo, mixtura, t)
    covarianzas = oraculo.component_covariances(tt)[0]

    media = (pesos[:, None] * medias).sum(dim=0)
    segundo = (
        pesos[:, None, None]
        * (covarianzas + medias.unsqueeze(-1) * medias.unsqueeze(-2))
    ).sum(dim=0)
    return media, segundo - media.unsqueeze(-1) * media.unsqueeze(-2)


def _muestra_ruideada(
    mixtura: ExactGaussianMixture, sde: ForwardSDE, t: float, semilla: int
) -> "torch.Tensor":
    """``n`` puntos de la mixtura pasados por el kernel forward de la SDE en el tiempo ``t``."""
    x0 = torch.as_tensor(mixtura.sample(_N_MONTE_CARLO), dtype=torch.float64)
    tiempos = _tiempo(t, _N_MONTE_CARLO)
    x_t, _ = sde.perturb(x0, tiempos, generator=torch.Generator().manual_seed(semilla))
    return x_t


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_los_momentos_empiricos_del_kernel_forward_coinciden_con_la_densidad(nombre: str):
    """Criterio 7.3: la media y la covarianza de puntos ruideados contra las de ``p_t``.

    Los puntos salen de la mixtura real pasados por ``perturb``, es decir del proceso forward
    de verdad y no de la densidad del oráculo; los momentos esperados salen de los parámetros
    que el oráculo publica. Se barren el tiempo mínimo, uno intermedio y el final (7.4).

    Tolerancia de Monte Carlo: ``_ERRORES_ESTANDAR`` errores estándar del estimador
    correspondiente —``sqrt(S_ii/n)`` para cada coordenada de la media y
    ``sqrt((S_ii S_jj + S_ij²)/n)`` para cada entrada de la covarianza—, que se calculan acá a
    partir de la covarianza exacta y no se eligen a mano. El muestreo de la mixtura reparte los
    puntos entre componentes en proporción **exacta** a los pesos, así que la única
    aleatoriedad es intra-componente y el error real queda por debajo de esa cota iid.
    """
    mixtura = _mixtura_exacta()
    sde = make_sde(nombre)
    oraculo = MixtureOracle(mixtura, sde)

    for valor in [0.0, 0.5, 1.0]:
        x_t = _muestra_ruideada(mixtura, sde, valor, semilla=1234)
        media, covarianza = _momentos_propagados(oraculo, mixtura, valor)

        varianzas = torch.diagonal(covarianza)
        error_media = (varianzas / _N_MONTE_CARLO).sqrt()
        error_covarianza = (
            (varianzas[:, None] * varianzas[None, :] + covarianza**2) / _N_MONTE_CARLO
        ).sqrt()

        desvio_media = (x_t.mean(dim=0) - media).abs()
        desvio_covarianza = (torch.cov(x_t.T) - covarianza).abs()
        assert (desvio_media <= _ERRORES_ESTANDAR * error_media).all(), (
            f"{nombre} en t={valor}: la media empírica se aparta {desvio_media.tolist()} de "
            f"{media.tolist()} (cota {(_ERRORES_ESTANDAR * error_media).tolist()})"
        )
        assert (desvio_covarianza <= _ERRORES_ESTANDAR * error_covarianza).all(), (
            f"{nombre} en t={valor}: la covarianza empírica se aparta "
            f"{desvio_covarianza.tolist()} de {covarianza.tolist()}"
        )


#: Centro de la observable acotada de la verificación por Monte Carlo.
_CENTRO_OBSERVABLE = (0.5, -0.25)


def _observable_acotada(puntos: "torch.Tensor") -> "torch.Tensor":
    """``f(x) = exp(-‖x - c‖²/2)``, elegida en el test y ajena al oráculo.

    Acotada en ``(0, 1]``, así que su promedio empírico tiene varianza chica y su integral
    contra la densidad converge sin cola pesada. Que la función la fije el test es lo que hace
    que la comparación no pueda cancelarse contra un error de la propia densidad.
    """
    centro = torch.tensor(_CENTRO_OBSERVABLE, dtype=puntos.dtype, device=puntos.device)
    return torch.exp(-0.5 * ((puntos - centro) ** 2).sum(dim=-1))


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_promedio_empirico_de_una_observable_coincide_con_su_integral(nombre: str):
    """Criterio 7.3: ``E_{p_t}[f]`` empírico contra ``∫ f p_t`` por cuadratura.

    Es la contracara del test de momentos: en vez de comparar dos resúmenes de la
    distribución, integra la **densidad** del oráculo contra una función fija y la contrasta
    con el promedio de esa misma función sobre puntos que salieron del kernel forward. Si
    ``prob`` estuviera mal normalizada o corrida, los dos números se separan.

    Tolerancia de Monte Carlo: ``_ERRORES_ESTANDAR`` errores estándar del promedio, estimados
    con el desvío **de la propia muestra** (``s/sqrt(n)``), de modo que la cota no es una
    constante elegida a mano. Con semillas fijas los desacuerdos observados llegan a ``1.6``
    errores estándar. La cuadratura aporta un error despreciable frente al de Monte Carlo: la
    misma malla integra la masa a ``1`` con error ``1e-8`` o menos.
    """
    mixtura = _mixtura_exacta()
    sde = make_sde(nombre)
    oraculo = MixtureOracle(mixtura, sde)

    for valor in [0.0, 0.5, 1.0]:
        x_t = _muestra_ruideada(mixtura, sde, valor, semilla=99)
        valores = _observable_acotada(x_t)
        empirico = float(valores.mean())
        error_estandar = float(valores.std(unbiased=True)) / math.sqrt(_N_MONTE_CARLO)

        malla = auto_grid(
            means=_medias_contraidas(oraculo, mixtura, valor),
            covariances=oraculo.component_covariances(
                torch.tensor([valor], dtype=torch.float64)
            )[0],
        )

        def integrando(puntos: "torch.Tensor", instante: float = valor) -> "torch.Tensor":
            """``f(x) p_t(x)`` en los nodos de un bloque de la malla."""
            tiempos = torch.full(
                (puntos.shape[0],), instante, dtype=puntos.dtype, device=puntos.device
            )
            return _observable_acotada(puntos) * oraculo.prob(puntos, tiempos)

        exacto = integrate(integrando, malla)
        assert oraculo.total_mass(valor, grid=malla) == pytest.approx(1.0, abs=1e-8)
        assert abs(empirico - exacto) <= _ERRORES_ESTANDAR * error_estandar, (
            f"{nombre} en t={valor}: el promedio empírico {empirico:.6f} se aparta "
            f"{abs(empirico - exacto):.6f} de la integral {exacto:.6f} (cota "
            f"{_ERRORES_ESTANDAR * error_estandar:.6f})"
        )


# ------------------------- API pública del paquete y smoke ejecutable (3.6, 4.6)


_API_PUBLICA = ("BiasReport", "MixtureOracle", "QuadratureGrid", "auto_grid", "integrate")


def test_el_all_del_paquete_declara_exactamente_la_api_publica():
    """El paquete publica el oráculo, el reporte de sesgo y las piezas de la cuadratura."""
    assert sorted(diffusion.analytic.__all__) == sorted(_API_PUBLICA)


def test_los_nombres_publicos_son_los_objetos_de_sus_modulos():
    """La ruta corta no duplica nada: reexporta los mismos objetos de los submódulos."""
    assert diffusion.analytic.MixtureOracle is MixtureOracle
    assert diffusion.analytic.BiasReport is BiasReport
    assert diffusion.analytic.QuadratureGrid is QuadratureGrid
    assert diffusion.analytic.auto_grid is auto_grid
    assert diffusion.analytic.integrate is integrate


def test_dir_del_paquete_incluye_los_nombres_publicos():
    """``dir()`` los lista aunque se resuelvan de forma diferida (autocompletado)."""
    listados = dir(diffusion.analytic)
    faltantes = [nombre for nombre in _API_PUBLICA if nombre not in listados]
    assert not faltantes, faltantes


def test_un_atributo_inexistente_del_paquete_levanta_attribute_error():
    """La resolución diferida no convierte un typo en un import raro: falla como siempre."""
    with pytest.raises(AttributeError, match="diffusion.analytic"):
        diffusion.analytic.NoExiste  # noqa: B018


def test_importar_la_api_con_estrella_resuelve_todos_los_nombres():
    """``from diffusion.analytic import *`` trae los cinco nombres, sin ``AttributeError``."""
    src = Path(diffusion.analytic.__file__).resolve().parents[2]
    codigo = (
        "ns = {}\n"
        "exec('from diffusion.analytic import *', ns)\n"
        "import diffusion.analytic as a\n"
        "print(sorted(n for n in a.__all__ if n not in ns))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_importar_el_paquete_no_carga_sde_hasta_pedir_el_oraculo():
    """La reexportación es **diferida**: el paquete no arrastra sus dependencias al importar.

    Es lo que mantiene viva la garantía de que ``diffusion.analytic.quadrature`` sea una
    utilidad numérica autónoma: importar cualquier submódulo ejecuta el ``__init__`` del
    paquete, así que si el ``__init__`` importara el oráculo de forma directa la cuadratura
    quedaría acoplada a ``sde`` y a ``data_generation`` por la puerta de atrás. Se mide en un
    intérprete nuevo porque el ``sys.modules`` de pytest ya los tiene cargados.
    """
    src = Path(diffusion.analytic.__file__).resolve().parents[2]
    codigo = (
        "import sys, diffusion.analytic as a; "
        "pesados = ('diffusion.sde', 'diffusion.data_generation'); "
        "antes = [m for m in pesados if m in sys.modules]; "
        "a.MixtureOracle; "
        "despues = [m for m in pesados if m in sys.modules]; "
        "print(antes, sorted(despues))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    esperado = "[] ['diffusion.data_generation', 'diffusion.sde']"
    assert proc.stdout.strip() == esperado, proc.stdout


def test_el_smoke_contrasta_el_score_contra_autograd_y_reporta_masa_unitaria():
    """Criterios 3.6 y 4.6: el observable del smoke, chequeado sin parsear stdout.

    El propio ``main`` devuelve, por SDE, el peor caso sobre los tiempos que barre: la
    discrepancia máxima entre el score en forma cerrada y el gradiente de ``log_prob`` por
    autograd, y la masa integrada más alejada de uno.

    Tolerancias del método: las dos cuentas del score son caminos de doble precisión sobre los
    mismos términos, así que el desacuerdo es error de redondeo amplificado por la magnitud
    del score (que crece como ``1/σ_t²``); lo observado llega a ``1.4e-13`` en el tiempo más
    chico del barrido, y la cota deja tres órdenes de margen. La cuadratura integra a uno con
    error ``1e-8`` o menos, igual que en el resto de la suite.
    """
    from diffusion.analytic.__main__ import main

    resumen = main()

    assert sorted(resumen) == ["sub_vp", "ve", "vp"]
    for nombre, (discrepancia, masa) in resumen.items():
        assert discrepancia < 1e-10, f"{nombre}: score vs autograd difiere en {discrepancia}"
        assert masa == pytest.approx(1.0, abs=1e-8), f"{nombre}: masa {masa}"


def test_correr_el_paquete_como_programa_imprime_y_termina_sin_error():
    """Correr ``python -m diffusion.analytic`` en CPU imprime la comparación y la masa."""
    src = Path(diffusion.analytic.__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "diffusion.analytic"],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    salida = proc.stdout.lower()
    assert "autograd" in salida, proc.stdout
    assert "masa" in salida, proc.stdout
    for nombre in ("vp", "ve", "sub_vp"):
        assert nombre in salida, proc.stdout


# ================================================================================
# Sampleo reverso con el oráculo inyectado como score (4.2, 4.3, 7.5)
#
# Es el cierre del círculo: si el oráculo es la verdad exacta y los samplers integran
# bien el reverso, entonces sampleando con el oráculo en lugar de una red hay que
# recuperar la mixtura de partida — y al refinar la grilla temporal hay que acercarse
# más. Cubre las doce celdas del Eje 2 × Eje 1 escalar (cuatro samplers × VP/VE/sub-VP)
# **sin tocar una línea de `diffusion.samplers`**: el oráculo se pasa como `score_fn` a
# `make_sampler` porque su `__call__` ya cumple el contrato `ScoreFn` (no es `nn.Module`
# ni necesita adaptador). Los samplers se importan *dentro* de las funciones para dejar
# a la vista que la dependencia va en un solo sentido: `analytic` no los conoce.
#
# --- Observable ---------------------------------------------------------------------
#
# `_discrepancia_con_la_mixtura` reduce una muestra a **un** número adimensionado.
# Asigna cada punto a su componente más probable bajo la mixtura verdadera (regla de
# Bayes con los parámetros declarados) y toma el peor de tres errores por componente:
#
#   * ``|ŵ_k − w_k|``                      — composición: detecta modos perdidos o
#                                            sobrerrepresentados;
#   * ``‖μ̂_k − μ_k‖ / escala_k``          — ubicación, en unidades del propio radio
#                                            RMS de la componente;
#   * ``‖Σ̂_k − Σ_k‖_F / ‖Σ_k‖_F``        — forma: tamaño, anisotropía y orientación.
#
# Es deliberadamente elemental (cuentas de asignación y momentos), no una distancia
# entre distribuciones: las métricas de discrepancia distribucional son materia de la
# capa de evaluación, no de esta verificación. La mixtura de prueba tiene dos modos
# separados 3.0, pesos desbalanceados (0.7, 0.3), anisotropía 3 y eje rotado π/6, así
# que el observable de forma es exigente (una orientación equivocada lo mueve) y la
# asignación es inequívoca: el punto medio queda a 3.8 desvíos de cada media sobre el
# eje mayor, con lo que la tasa de mala asignación (≈ 7e-5) no aporta sesgo medible.
#
# --- Predicado de convergencia -----------------------------------------------------
#
# La discrepancia medida es error de discretización + ruido de Monte Carlo + (en VE)
# desfasaje del prior. Al refinar, el primer término baja; los otros dos no. Así que la
# afirmación direccional que se exige es: **cada refinamiento mejora al menos un 30 % o
# ya llegó al piso irreducible**, y la grilla más fina está en el piso:
#
#   d(pasos_{i+1}) <= max(_FACTOR_DE_MEJORA · d(pasos_i), _PISO_IRREDUCIBLE)
#   d(pasos_última) <= _PISO_IRREDUCIBLE
#
# Las dos viven en `_fallas_de_la_verificacion`, que es lo que el test de las doce celdas
# exige vacío y el de discriminancia exige no vacío — el mismo predicado, sin reescribir.
#
# --- Calibración (medida, no adivinada) --------------------------------------------
#
# Barrido de las 12 celdas × 5 semillas con `n=2000`, `t_eps=1e-3` y pasos (3, 25, 200):
#
#   * peor discrepancia final observada: **0.2118** (VE con `pf_ode`; VE con `heun`
#     0.1871). VE es la celda floja y por el mismo motivo que ya documenta la suite de
#     samplers: su prior es `N(0, σ_max²I)` con `σ_max = 5`, mientras la marginal real
#     en `t = T` tiene varianza `σ_max²` + la de los datos (≈ 26.8 contra 25, un 7 % de
#     subdispersión de arranque). Los samplers estocásticos borran ese desfasaje; los
#     deterministas (`pf_ode`, `heun`) lo arrastran, porque la PF-ODE es un mapa
#     biyectivo del prior. `_PISO_IRREDUCIBLE = 0.35` deja ~1.65× de margen.
#   * peor violación del predicado escalonado con `_FACTOR_DE_MEJORA = 0.70`:
#     **−0.1382** (holgura, o sea que nunca se violó).
#   * peor razón `d(3 pasos) / d(200 pasos)`: **10.95** — la mejora al refinar es de
#     uno a dos órdenes de magnitud en la mayoría de las celdas (VE con `euler`/`pc`
#     arranca en ~5000 y termina en ~0.07).
#   * `t_eps = 1e-3` es el default de los samplers y se deja explícito: en las tres SDEs
#     el desvío del kernel ahí vale σ ≈ 0.0105, que sobre una componente de radio RMS
#     0.32 infla el desvío un 0.06 % — dos órdenes por debajo del piso, así que truncar
#     la integración en `t_eps` no es lo que limita la verificación.
#   * costo: el barrido de las 12 celdas con una semilla corre en ~20 s en CPU
#     (101 s las 5 semillas de la calibración), más ~4 s el test de discriminancia.
#
# --- Discriminancia ----------------------------------------------------------------
#
# El observable no es un colador. Aplicando el **mismo** predicado a scores rotos —los
# tres definidos acá en el test, sin tocar producción— y con el barrido completo (3, 25,
# 200), las cuatro celdas de control con el score exacto pasan y las doce combinaciones
# rotas fallan:
#
#   * signo invertido: la discrepancia final va de 7.5e4 (VE/`pf_ode`) a 1.0e16
#     (`pc`/sub-VP), y **crece** en cada refinamiento — incumple las dos condiciones;
#   * score nulo: 1.7e2 a 2.1e21, también creciente o estancado;
#   * oráculo de **un solo modo** (estructura equivocada, magnitud del score
#     perfectamente plausible): se estanca en 3.7–3.8, o sea 11× el piso, y como el
#     estancamiento ocurre *por encima* del piso también incumple la condición
#     direccional. Es el caso que muestra que el observable mira la mixtura y no solo la
#     finitud o la escala global.
#
# El test de discriminancia usa una configuración barata (pasos (3, 60), `n=1000`) que
# reproduce el mismo veredicto en ~4 s.
# ================================================================================

_SEPARACION_E2E = 3.0
_PESOS_E2E = (0.7, 0.3)
_ANISOTROPIA_E2E = 3.0
_ESCALA_E2E = 0.3
_ANGULO_E2E = math.pi / 6

_SAMPLERS_DEL_REVERSO = ("euler", "pf_ode", "heun", "pc")
# Cobertura 4×3: los cuatro samplers del Eje 2 × las tres SDEs escalares (4.3).
_CELDAS_E2E = [(s, d) for s in _SAMPLERS_DEL_REVERSO for d in _SDES_ESCALARES]

_PASOS_CRECIENTES = (3, 25, 200)
_N_MUESTRAS_E2E = 2000
_T_EPS_E2E = 1e-3
_SEMILLA_E2E = 0
_PISO_IRREDUCIBLE = 0.35
_FACTOR_DE_MEJORA = 0.70


def _mixtura_del_e2e() -> ExactGaussianMixture:
    """Mixtura de dos modos separados, desbalanceados, anisotrópicos y rotados."""
    return ExactGaussianMixture.two_modes(
        separation=_SEPARACION_E2E,
        weights=_PESOS_E2E,
        anisotropy=_ANISOTROPIA_E2E,
        scale=_ESCALA_E2E,
        angle=_ANGULO_E2E,
    )


def _asignacion_mas_probable(
    mixtura: ExactGaussianMixture, muestras: "torch.Tensor"
) -> "torch.Tensor":
    """Componente más probable de la mixtura para cada muestra (regla de Bayes).

    Se calcula con los parámetros **declarados** de la mixtura, en doble precisión y en
    el dominio logarítmico. La constante ``2π`` es común a las componentes y no cambia el
    ``argmax``, así que se omite.

    Args:
        mixtura: Mixtura verdadera, de la que se leen pesos, medias y covarianzas.
        muestras: Puntos ``(N, 2)``.

    Returns:
        Índices ``(N,)`` de la componente asignada a cada punto.
    """
    pesos = torch.as_tensor(mixtura.weights_, dtype=torch.float64)
    medias = torch.as_tensor(mixtura.means_, dtype=torch.float64)
    covarianzas = torch.as_tensor(mixtura.covariances_, dtype=torch.float64)
    x = muestras.to(torch.float64)

    logs = []
    for k in range(pesos.shape[0]):
        precision = torch.linalg.inv(covarianzas[k])
        dif = x - medias[k]
        cuadratica = ((dif @ precision) * dif).sum(dim=-1)
        det = torch.linalg.det(covarianzas[k])
        logs.append(torch.log(pesos[k]) - 0.5 * cuadratica - 0.5 * torch.log(det))
    return torch.stack(logs, dim=1).argmax(dim=1)


def _discrepancia_con_la_mixtura(
    mixtura: ExactGaussianMixture, muestras: "torch.Tensor"
) -> float:
    """Peor error adimensionado del perfil por componente de ``muestras``.

    Máximo, sobre las componentes de la mixtura, de tres errores: el del peso
    (``|ŵ_k − w_k|``), el de la media en unidades del radio RMS de la componente
    (``‖μ̂_k − μ_k‖ / escala_k``) y el relativo de la covarianza en norma de Frobenius
    (``‖Σ̂_k − Σ_k‖_F / ‖Σ_k‖_F``). Los tres son adimensionados, así que el máximo es
    comparable entre celdas.

    Args:
        mixtura: Mixtura verdadera contra la que se compara.
        muestras: Puntos ``(N, 2)`` generados por el sampler.

    Returns:
        La discrepancia, o ``inf`` si las muestras no son finitas o alguna componente
        se quedó con menos de tres puntos asignados (un modo colapsado es una falla
        total, no un número chico).
    """
    if not bool(torch.all(torch.isfinite(muestras))):
        return math.inf

    pesos = torch.as_tensor(mixtura.weights_, dtype=torch.float64)
    medias = torch.as_tensor(mixtura.means_, dtype=torch.float64)
    covarianzas = torch.as_tensor(mixtura.covariances_, dtype=torch.float64)
    x = muestras.to(torch.float64)
    asignacion = _asignacion_mas_probable(mixtura, x)
    n = x.shape[0]

    peor = 0.0
    for k in range(pesos.shape[0]):
        propios = x[asignacion == k]
        peor = max(peor, abs(propios.shape[0] / n - float(pesos[k])))
        if propios.shape[0] < 3:
            return math.inf

        escala = math.sqrt(float(torch.diagonal(covarianzas[k]).sum()) / 2.0)
        media_empirica = propios.mean(dim=0)
        peor = max(peor, float(torch.linalg.norm(media_empirica - medias[k])) / escala)

        centrado = propios - media_empirica
        cov_empirica = centrado.T @ centrado / (propios.shape[0] - 1)
        peor = max(
            peor,
            float(torch.linalg.norm(cov_empirica - covarianzas[k]))
            / float(torch.linalg.norm(covarianzas[k])),
        )
    return peor


def _barrido_de_discrepancias(
    mixtura: ExactGaussianMixture,
    sde: ForwardSDE,
    nombre_sampler: str,
    score_fn,
    pasos: tuple[int, ...],
    *,
    n: int = _N_MUESTRAS_E2E,
    semilla: int = _SEMILLA_E2E,
) -> list[float]:
    """Discrepancia con la mixtura para cada cantidad de pasos de integración.

    Args:
        mixtura: Mixtura verdadera contra la que se mide.
        sde: Proceso forward que define el reverso a integrar.
        nombre_sampler: Clave del registry de samplers.
        score_fn: Invocable ``(x, t) -> score``; acá entra el oráculo tal cual.
        pasos: Cantidades crecientes de pasos de integración.
        n: Muestras por corrida.
        semilla: Semilla del generador, la misma en todo el barrido.

    Returns:
        Una discrepancia por cada elemento de ``pasos``, en el mismo orden.
    """
    from diffusion.samplers import make_sampler

    salida = []
    for cantidad in pasos:
        sampler = make_sampler(
            nombre_sampler, sde, score_fn, n_steps=cantidad, t_eps=_T_EPS_E2E
        )
        muestras = sampler.sample(
            n, generator=torch.Generator().manual_seed(semilla)
        )
        salida.append(_discrepancia_con_la_mixtura(mixtura, muestras))
    return salida


def _fallas_de_la_verificacion(
    discrepancias: list[float], pasos: tuple[int, ...]
) -> list[str]:
    """Incumplimientos del predicado de convergencia sobre un barrido de discrepancias.

    Es **la** definición de "el sampleo recupera la mixtura", usada tal cual por el test
    de las doce celdas (que exige la lista vacía) y por el de discriminancia (que exige
    que no lo esté). Dos condiciones, calibradas en el bloque de arriba:

    1. Direccional: cada refinamiento baja la discrepancia al menos
       ``1 − _FACTOR_DE_MEJORA``, salvo que ya haya llegado a ``_PISO_IRREDUCIBLE``.
    2. Exactitud: la grilla más fina está en el piso.

    Args:
        discrepancias: Una discrepancia por cantidad de pasos, en orden creciente.
        pasos: Las cantidades de pasos correspondientes, para el mensaje.

    Returns:
        Un mensaje por condición incumplida; lista vacía si el barrido convergió.
    """
    fallas = []
    for i in range(len(discrepancias) - 1):
        exigido = max(_FACTOR_DE_MEJORA * discrepancias[i], _PISO_IRREDUCIBLE)
        if not discrepancias[i + 1] <= exigido:
            fallas.append(
                f"pasar de {pasos[i]} a {pasos[i + 1]} pasos llevó la discrepancia de "
                f"{discrepancias[i]:.4g} a {discrepancias[i + 1]:.4g}, que no mejora lo "
                f"suficiente ni llega al piso (se exigía <= {exigido:.4g})"
            )
    if not discrepancias[-1] <= _PISO_IRREDUCIBLE:
        fallas.append(
            f"con {pasos[-1]} pasos la discrepancia es {discrepancias[-1]:.4g}, por "
            f"encima del piso {_PISO_IRREDUCIBLE}"
        )
    return fallas


@pytest.mark.parametrize("nombre_sampler,nombre_sde", _CELDAS_E2E)
def test_el_oraculo_entra_como_score_en_los_cuatro_samplers(
    nombre_sampler: str, nombre_sde: str
):
    """Criterio 4.3: el oráculo se pasa donde va la red y el sampler produce muestras.

    Chequeo barato de contrato para las doce celdas: sin adaptadores, sin envolver el
    oráculo en un módulo y sin tocar el código de los samplers, ``make_sampler`` acepta
    el invocable y ``sample`` devuelve ``(n, 2)`` ``float32`` finito.
    """
    from diffusion.samplers import make_sampler

    sde = make_sde(nombre_sde)
    oraculo = diffusion.analytic.MixtureOracle(_mixtura_del_e2e(), sde)

    assert not isinstance(oraculo, torch.nn.Module)
    sampler = make_sampler(nombre_sampler, sde, oraculo, n_steps=4, t_eps=_T_EPS_E2E)
    # El sampler guarda EL MISMO objeto: no hay adaptador intermedio (criterio 4.2).
    assert sampler.score_fn is oraculo

    muestras = sampler.sample(16, generator=torch.Generator().manual_seed(_SEMILLA_E2E))
    assert muestras.shape == (16, 2)
    assert muestras.dtype == torch.float32
    assert bool(torch.all(torch.isfinite(muestras)))


@pytest.mark.parametrize("nombre_sampler,nombre_sde", _CELDAS_E2E)
def test_al_refinar_los_pasos_el_sampleo_se_acerca_a_la_mixtura(
    nombre_sampler: str, nombre_sde: str
):
    """Criterios 4.2, 4.3 y 7.5: con el score exacto, refinar acerca a la mixtura.

    Barrido de ``_PASOS_CRECIENTES`` con la misma semilla, evaluado con
    ``_fallas_de_la_verificacion``: la direccional (cada refinamiento mejora un 30 % o ya
    está en el piso) y la de exactitud (la grilla más fina está en el piso).
    """
    mixtura = _mixtura_del_e2e()
    sde = make_sde(nombre_sde)
    oraculo = diffusion.analytic.MixtureOracle(mixtura, sde)

    ds = _barrido_de_discrepancias(
        mixtura, sde, nombre_sampler, oraculo, _PASOS_CRECIENTES
    )
    fallas = _fallas_de_la_verificacion(ds, _PASOS_CRECIENTES)
    assert not fallas, (
        f"{nombre_sampler}/{nombre_sde}: {'; '.join(fallas)}; barrido completo "
        f"{[float(f'{v:.4g}') for v in ds]}"
    )


# ------------------------------------------- discriminancia del observable (7.5)


_PASOS_DISCRIMINANCIA = (3, 60)
_N_DISCRIMINANCIA = 1000
# Una celda por sampler, mezclando las tres SDEs: alcanza para mostrar que el observable
# reacciona, sin pagar las doce celdas de nuevo.
_CELDAS_DISCRIMINANCIA = [
    ("euler", "vp"),
    ("pf_ode", "ve"),
    ("heun", "vp"),
    ("pc", "sub_vp"),
]
# Cuánto por encima del piso tiene que quedar un score roto para que el test valga algo.
_MARGEN_DE_DISCRIMINANCIA = 10.0

# Las tres fábricas de score roto comparten la firma ``(oraculo, sde) -> ScoreFn`` para
# que la parametrización las trate igual; cada una usa solo lo que necesita.


def _score_de_signo_invertido(oraculo, sde):
    """Score exacto con el signo dado vuelta: el reverso empuja *lejos* de los datos."""

    def score(x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        return -oraculo(x, t)

    return score


def _score_nulo(oraculo, sde):
    """Score idénticamente nulo: el reverso integra sin ninguna atracción a los datos."""

    def score(x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        return torch.zeros_like(x)

    return score


def _score_de_un_solo_modo(oraculo, sde):
    """Score exacto pero de la mixtura equivocada: un único modo en el origen.

    Es el roto interesante: la magnitud del score es perfectamente plausible y el
    sampleo converge sin problemas — a la distribución equivocada. Un observable que
    solo mirara finitud o escala global no lo notaría.
    """
    mixtura = _mixtura_del_e2e()
    unico = ExactGaussianMixture(
        weights=[1.0],
        means=[[0.0, 0.0]],
        covariances=mixtura.covariances_[:1],
    )
    return diffusion.analytic.MixtureOracle(unico, sde)


_SCORES_ROTOS = [
    pytest.param(_score_de_signo_invertido, True, id="signo-invertido"),
    pytest.param(_score_nulo, True, id="nulo"),
    pytest.param(_score_de_un_solo_modo, False, id="un-solo-modo"),
]


@pytest.mark.parametrize("fabrica,empeora_al_refinar", _SCORES_ROTOS)
@pytest.mark.parametrize("nombre_sampler,nombre_sde", _CELDAS_DISCRIMINANCIA)
def test_un_score_roto_no_recupera_la_mixtura(
    nombre_sampler: str, nombre_sde: str, fabrica, empeora_al_refinar: bool
):
    """El observable discrimina: con el score roto la verificación de 7.5 no pasa.

    Se evalúa el **mismo** ``_fallas_de_la_verificacion`` que usa el test de las doce
    celdas, y se exige que devuelva al menos un incumplimiento. Además, para que la
    discriminancia no sea marginal, la discrepancia final tiene que quedar al menos
    ``_MARGEN_DE_DISCRIMINANCIA`` veces por encima del piso; y donde el error rompe el
    signo o la magnitud del arrastre (signo invertido, score nulo) refinar la grilla
    **empeora** el resultado, que es la negación literal del predicado direccional.
    """
    mixtura = _mixtura_del_e2e()
    sde = make_sde(nombre_sde)
    oraculo = diffusion.analytic.MixtureOracle(mixtura, sde)

    ds = _barrido_de_discrepancias(
        mixtura,
        sde,
        nombre_sampler,
        fabrica(oraculo, sde),
        _PASOS_DISCRIMINANCIA,
        n=_N_DISCRIMINANCIA,
    )
    celda = f"{nombre_sampler}/{nombre_sde}"
    barrido = [float(f"{v:.4g}") for v in ds]

    assert _fallas_de_la_verificacion(ds, _PASOS_DISCRIMINANCIA), (
        f"{celda}: el score roto pasó la verificación de convergencia; barrido {barrido}"
    )
    assert ds[-1] > _MARGEN_DE_DISCRIMINANCIA * _PISO_IRREDUCIBLE, (
        f"{celda}: con el score roto la discrepancia final es {ds[-1]:.4g}, demasiado "
        f"cerca del piso {_PISO_IRREDUCIBLE} como para que el test discrimine; "
        f"barrido {barrido}"
    )
    if empeora_al_refinar:
        assert ds[-1] >= ds[0], (
            f"{celda}: se esperaba que refinar empeorara el resultado con este score "
            f"roto, pero la discrepancia bajó de {ds[0]:.4g} a {ds[-1]:.4g}"
        )
