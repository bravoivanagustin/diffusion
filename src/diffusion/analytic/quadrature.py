"""Cuadratura 2D sobre una malla dimensionada por la escala real de la densidad.

Utilidad numérica **autónoma** del módulo analítico: construye una malla cuadrada centrada
en el origen e integra una función escalar sobre ella con una suma de Riemann. Es la pieza
que permite *verificar* la verdad analítica en vez de confiar en ella — que ``p_t`` integre a
uno sobre el plano es el autochequeo de que la densidad exacta está bien escrita, y el mismo
integrador produce el valor de referencia del sesgo de inicialización.

Solo depende de ``torch``: no importa la SDE, ni la fuente de datos, ni el oráculo. Eso la
hace testeable por separado y reusable por cualquier consumidor que traiga sus propios
parámetros.

Cómo se dimensiona la malla (la parte que importa)
--------------------------------------------------

Son **dos decisiones independientes**:

- El **dominio** (``half_width``) lo fija la extensión de la densidad: la mayor de las
  ``|media_k| + n_sigma · sqrt(λ_max(Σ_k))`` entre las componentes, ampliada a
  ``n_sigma · prior_std`` cuando se integra contra una distribución de referencia (el prior
  de la SDE) que puede ser mucho más ancha que los datos.
- El **paso** (``spacing``) lo fija la **componente más angosta**:
  ``min_k sqrt(λ_min(Σ_k))`` sobre las covarianzas **ya evaluadas en el tiempo de interés**,
  dividido por ``points_per_sigma``.

El paso **no** se deriva del desvío del kernel de difusión ``σ_t``, y esto es deliberado:
para la familia escalar-gaussiana la covarianza perturbada es
``Σ_k(t) = α_t² Σ_k + σ_t² I``, así que la concentración tiene **piso** ``α_t² λ_min(Σ_k)``
y **no colapsa** cuando ``t → 0``. Dimensionar por ``σ_t`` fallaría en los dos sentidos: con
``σ_t`` diminuto (el prototipo llegó a ``1.01e-5``) pediría una resolución absurda para una
densidad que en realidad sigue teniendo el ancho de los datos, y con ``σ_t`` grande daría un
paso más grueso del que la componente angosta necesita.

La masa obtenida —``integrate`` de la densidad— es el **detector** de una malla insuficiente:
si se aleja de uno, la resolución no alcanzó. Con componentes casi degeneradas la resolución
necesaria crece y el tope explícito ``max_points`` la recorta; cuando eso pasa la malla
devuelta queda marcada con ``truncated=True``, de modo que el recorte sea observable y no
silencioso.

Uso típico::

    grid = auto_grid(means=medias, covariances=covarianzas_en_t)
    masa = integrate(lambda x: densidad(x), grid)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

#: Cantidad máxima de puntos evaluados por llamada a ``fn``. La malla se recorre por bloques
#: de filas para que el costo de memoria no crezca con el cuadrado de la resolución.
_MAX_BLOCK_POINTS: int = 1 << 18

#: Tolerancia relativa con la que se exige simetría a las covarianzas.
_SYMMETRY_TOL: float = 1e-8


@dataclass(frozen=True)
class QuadratureGrid:
    """Malla cuadrada centrada en el origen.

    Describe ``n_points × n_points`` nodos equiespaciados que cubren
    ``[-half_width, half_width]`` en cada eje. Es un valor inmutable: no guarda los puntos,
    solo la geometría que hace falta para reconstruirlos.

    Attributes:
        half_width: Medio ancho del dominio en cada eje. Finito y positivo.
        n_points: Nodos por eje. Al menos 2, para que exista un paso.
        truncated: ``True`` si el tope de puntos limitó la resolución pedida, es decir si la
            malla es más gruesa que la que la componente más angosta requería.
    """

    half_width: float
    n_points: int
    truncated: bool = False

    def __post_init__(self) -> None:
        """Valida la geometría de la malla.

        Raises:
            ValueError: Si ``half_width`` no es finito y positivo, o si ``n_points`` es menor
                que 2.
        """
        if not math.isfinite(self.half_width) or self.half_width <= 0.0:
            raise ValueError(
                f"half_width debe ser finito y positivo; recibido {self.half_width!r}."
            )
        if self.n_points < 2:
            raise ValueError(
                f"n_points debe ser al menos 2 para que la malla tenga paso; "
                f"recibido {self.n_points!r}."
            )

    @property
    def spacing(self) -> float:
        """Paso entre nodos consecutivos: el ancho total sobre la cantidad de intervalos."""
        return 2.0 * self.half_width / (self.n_points - 1)


def auto_grid(
    *,
    means: torch.Tensor,
    covariances: torch.Tensor,
    prior_std: float | None = None,
    n_sigma: float = 8.0,
    points_per_sigma: float = 6.0,
    max_points: int = 4096,
) -> QuadratureGrid:
    """Dimensiona una malla 2D a partir de los parámetros de una mixtura gaussiana.

    El dominio cubre las medias más ``n_sigma`` desvíos totales de la componente más ancha
    (y la escala del prior, si se declara); el paso resuelve la componente **más angosta**
    con ``points_per_sigma`` nodos por desvío. Ver el docstring del módulo para el porqué de
    dimensionar por los autovalores de las covarianzas y no por ``σ_t``.

    Args:
        means: Medias de las componentes, de forma ``(K, 2)``.
        covariances: Covarianzas de las componentes, de forma ``(K, 2, 2)``, **ya escaladas
            al tiempo de interés** (es decir ``α_t² Σ_k + σ_t² I`` si se integra la densidad
            perturbada). Deben ser simétricas y definidas positivas.
        prior_std: Desvío de la distribución de referencia contra la que se integra, si hay
            alguna. Solo puede **ampliar** el dominio, nunca recortarlo.
        n_sigma: Cuántos desvíos totales cubre el dominio más allá de las medias.
        points_per_sigma: Nodos por desvío de la componente más angosta.
        max_points: Tope de nodos por eje. Si la resolución pedida lo excede, la malla se
            recorta y queda marcada con ``truncated=True``.

    Returns:
        La malla dimensionada, con ``truncated`` indicando si el tope la limitó.

    Raises:
        ValueError: Si las formas no son ``(K, 2)`` y ``(K, 2, 2)`` con el mismo ``K``, si
            hay valores no finitos, si alguna covarianza no es simétrica definida positiva,
            o si ``prior_std``, ``n_sigma``, ``points_per_sigma`` o ``max_points`` están
            fuera de rango.
    """
    if not math.isfinite(n_sigma) or n_sigma <= 0.0:
        raise ValueError(f"n_sigma debe ser finito y positivo; recibido {n_sigma!r}.")
    if not math.isfinite(points_per_sigma) or points_per_sigma <= 0.0:
        raise ValueError(
            f"points_per_sigma debe ser finito y positivo; recibido {points_per_sigma!r}."
        )
    if max_points < 2:
        raise ValueError(
            f"max_points debe ser al menos 2 para que la malla tenga paso; "
            f"recibido {max_points!r}."
        )
    if prior_std is not None and (not math.isfinite(prior_std) or prior_std <= 0.0):
        raise ValueError(
            f"prior_std debe ser finito y positivo cuando se declara; "
            f"recibido {prior_std!r}."
        )

    medias, covs = _validar_parametros(means, covariances)

    autovalores = torch.linalg.eigvalsh(covs)  # (K, 2), ascendente
    if bool((autovalores[:, 0] <= 0.0).any()):
        raise ValueError(
            "covariances deben ser definidas positivas; alguna tiene autovalor mínimo "
            f"no positivo ({float(autovalores[:, 0].min())!r})."
        )

    desvio_mayor = torch.sqrt(autovalores[:, 1])  # sqrt(lambda_max) por componente
    desvio_menor = torch.sqrt(autovalores[:, 0])  # sqrt(lambda_min) por componente

    radios = medias.abs().amax(dim=1) + n_sigma * desvio_mayor
    half_width = float(radios.max())
    if prior_std is not None:
        half_width = max(half_width, n_sigma * prior_std)

    paso_objetivo = float(desvio_menor.min()) / points_per_sigma
    intervalos = 2.0 * half_width / paso_objetivo if paso_objetivo > 0.0 else math.inf
    if not math.isfinite(intervalos) or intervalos > float(max_points):
        requeridos = max_points + 1
    else:
        requeridos = max(2, math.ceil(intervalos) + 1)

    truncated = requeridos > max_points
    return QuadratureGrid(
        half_width=half_width,
        n_points=max_points if truncated else requeridos,
        truncated=truncated,
    )


def integrate(
    fn: Callable[[torch.Tensor], torch.Tensor],
    grid: QuadratureGrid,
    *,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Integra una función escalar sobre la malla con una suma de Riemann.

    Calcula ``sum(fn(nodos)) * spacing²`` en la precisión pedida, **con independencia de la
    precisión del caller**: los nodos se construyen en ``dtype`` y los valores devueltos por
    ``fn`` se llevan a ``dtype`` antes de acumularse, así que una ``fn`` que trabaja en
    ``float32`` no degrada la suma. La malla se recorre por bloques de filas para acotar la
    memoria, lo cual no cambia el resultado.

    Args:
        fn: Función a integrar. Recibe un tensor de puntos ``(N, 2)`` y devuelve ``(N,)``.
        grid: Malla sobre la que se integra.
        dtype: Precisión de la cuadratura. Doble por defecto.

    Returns:
        El valor de la integral como ``float``.

    Raises:
        ValueError: Si ``dtype`` no es de punto flotante, o si ``fn`` no devuelve un tensor
            de forma ``(N,)`` para los ``N`` puntos que recibió.
    """
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(
            f"dtype debe ser un tipo de punto flotante de torch; recibido {dtype!r}."
        )

    eje = torch.linspace(
        -grid.half_width, grid.half_width, grid.n_points, dtype=dtype
    )
    filas_por_bloque = max(1, _MAX_BLOCK_POINTS // grid.n_points)

    total = torch.zeros((), dtype=dtype)
    for inicio in range(0, grid.n_points, filas_por_bloque):
        filas = eje[inicio : inicio + filas_por_bloque]
        # (len(filas), n_points, 2) -> (N, 2) recorriendo primero el eje x.
        malla_y, malla_x = torch.meshgrid(filas, eje, indexing="ij")
        puntos = torch.stack((malla_x, malla_y), dim=-1).reshape(-1, 2)
        valores = fn(puntos)
        if valores.ndim != 1 or valores.shape[0] != puntos.shape[0]:
            raise ValueError(
                f"fn debe devolver un tensor de forma ({puntos.shape[0]},) para los puntos "
                f"que recibe; devolvió forma {tuple(valores.shape)}."
            )
        total = total + valores.to(dtype).sum()

    paso = torch.tensor(grid.spacing, dtype=dtype)
    return float(total * paso * paso)


def _validar_parametros(
    means: torch.Tensor, covariances: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Valida medias y covarianzas y las devuelve en doble precisión.

    Args:
        means: Medias candidatas, se exige forma ``(K, 2)``.
        covariances: Covarianzas candidatas, se exige forma ``(K, 2, 2)`` simétrica.

    Returns:
        El par ``(medias, covarianzas)`` convertido a ``float64``.

    Raises:
        ValueError: Si las formas o los valores no cumplen el contrato.
    """
    if means.ndim != 2 or means.shape[-1] != 2:
        raise ValueError(
            f"means debe tener forma (K, 2); recibida {tuple(means.shape)}."
        )
    if covariances.ndim != 3 or covariances.shape[-2:] != (2, 2):
        raise ValueError(
            f"covariances debe tener forma (K, 2, 2); recibida "
            f"{tuple(covariances.shape)}."
        )
    if covariances.shape[0] != means.shape[0]:
        raise ValueError(
            f"covariances debe traer una covarianza por media; recibidas "
            f"{covariances.shape[0]} para {means.shape[0]} medias."
        )

    medias = means.detach().to(torch.float64)
    covs = covariances.detach().to(torch.float64)

    if not bool(torch.isfinite(medias).all()):
        raise ValueError("means debe tener todos sus valores finitos.")
    if not bool(torch.isfinite(covs).all()):
        raise ValueError("covariances debe tener todos sus valores finitos.")
    if not torch.allclose(covs, covs.transpose(-1, -2), rtol=_SYMMETRY_TOL, atol=1e-12):
        raise ValueError("covariances deben ser simétricas; alguna no lo es.")

    return medias, covs
