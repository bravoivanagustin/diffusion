"""Oráculo exacto de una mixtura gaussiana 2D ruideada por una SDE escalar-gaussiana.

Un kernel forward gaussiano **preserva la familia**: si los datos son
``p_0 = Σ_k w_k N(μ_k, Σ_k)`` y el kernel de perturbación es ``N(α_t x_0, σ_t² I)``, la
marginal ruideada sigue siendo una mixtura de gaussianas,

``p_t = Σ_k w_k N(α_t μ_k, Σ_k(t))``  con  ``Σ_k(t) = α_t² Σ_k + σ_t² I``,

y de ahí salen en forma cerrada la densidad, la log-densidad y el score. Este módulo
implementa esa propagación: es el **patrón de referencia** contra el que se mide el resto
del laboratorio 2D.

De dónde salen ``α_t`` y ``σ_t`` (la decisión de diseño central)
---------------------------------------------------------------

**Del contrato marginal que la SDE ya publica**, nunca re-derivando schedules ni
ramificando por variante. Las tres SDEs escalares del Eje 1 cumplen ``mean == α_t · x_0``
elementwise, así que evaluar ``marginal_prob`` sobre un vector de unos devuelve ``α_t`` en
la media y ``σ_t`` en el desvío — el mismo truco que ya usa la suite de samplers. Un único
camino de código cubre VP, VE y sub-VP (y cualquier SDE escalar-gaussiana futura), y queda
automáticamente consistente con lo que el entrenamiento realmente usó.

Dos convenciones del paquete que se consumen **tal cual**:

- **sub-VP publica su desvío sin raíz**: ``marginal_prob`` devuelve ``std = 1 - e^{-∫β}``,
  y la varianza del kernel es ese valor al cuadrado. No hay que "corregirlo" tomándole una
  raíz: con ``t = 0.05`` el desvío vale ``0.0294`` y su raíz ``0.1716``, casi seis veces
  más, así que el error sería grosero y silencioso.
- **VE no contrae la media**: su ``α_t`` es exactamente ``1``, porque su kernel deja el
  dato en su lugar y solo le apila ruido.

Familia admitida y cómo se verifica
-----------------------------------

Solo la **familia escalar-gaussiana** (kernel ``N(α_t x_0, σ_t² I)`` con ``α_t``/``σ_t``
escalares por muestra). La pertenencia se verifica **estructuralmente** sobre lo que
devuelve el contrato marginal —la media es proporcional a ``x_0`` y el desvío colapsa a un
escalar por muestra—, no consultando ningún atributo de familia: ``is_augmented`` **ya no
existe** en el paquete (era de CLD, eliminado del proyecto el 05/07/2026) y agregar uno
estaría fuera de la frontera de este módulo. Hoy ninguna de las tres SDEs registradas cae
en el rechazo: es un guard de compatibilidad hacia adelante, no una rama con usuario
actual.

Precisión: los parámetros de la mixtura se guardan en **doble** precisión y los pasos
intermedios (el armado de ``Σ_k(t)``, su inversa, las responsabilidades) se calculan ahí,
con independencia de que el caller trabaje en ``float32``. La degradación al dtype del
caller ocurre recién en la salida de las cantidades que consumen los samplers.

Sesgo de inicialización: una cota, no un número "analítico"
-----------------------------------------------------------

La divergencia ``KL(p_T ‖ q)`` entre la mixtura ruideada y la distribución de la que el
sampler efectivamente parte **no admite forma cerrada**. Descomponiéndola como
``KL = −H(p_T) − E_{p_T}[log q]``, el término cruzado sí sale de los momentos de la
mixtura, pero la **entropía diferencial de una mixtura de gaussianas no tiene expresión
cerrada** — es un resultado conocido, y por eso la literatura trabaja con cotas y
aproximaciones. Solo el caso ``K = 1`` es la excepción: ahí la KL es la gaussiana-gaussiana
de toda la vida.

Lo que este módulo calcula sin integrar nada es la **cota superior por convexidad de la
KL**::

    KL( Σ_k w_k N_k ‖ q )  ≤  Σ_k w_k · KL( N_k ‖ q )

con cada término gaussiano-gaussiano de forma cerrada. Es una cota **rigurosa** pero
**floja**: en la mixtura de prueba de la spec quedó del orden de 2.5 veces el valor real
(VP ``5.9e-5`` contra ``2.3e-5``; VE ``5.2e-2`` contra ``2.0e-2``; sub-VP ``1.2e-4`` contra
``8.4e-5``). Hay que leerla como lo que es —un techo— y no como una estimación del sesgo.

La otra mitad del reporte es el **valor numérico de referencia**, que sí es la divergencia en
sí y se obtiene por **cuadratura 2D** sobre la log-densidad exacta::

    KL(p_t ‖ q) = ∫ p_t(x) · [ log p_t(x) − log q(x) ] dx

integrada sobre la misma clase de malla que la masa —dimensionada por los parámetros ya
propagados a ``t`` y ampliada para cubrir también la distribución de partida—. En dos
dimensiones la exactitud es barata: la suma de Riemann sobre integrandos gaussianos converge
espectralmente, y en la mixtura de prueba cuadruplicar la resolución mueve el valor en
``3e-18``. Que las dos cantidades se sandwicheen (``bound ≥ value ≥ 0``) las valida
mutuamente: salen de dos caminos de cálculo sin nada en común.

El valor viaja con su **tolerancia** y con la **masa integrada**, que es el autochequeo de que
la malla alcanzó (ver :func:`_quadrature_tolerance`). Cuando la masa se aparta de uno más allá
de esa tolerancia el cálculo **falla** en lugar de devolver el número: un valor de referencia
calculado sobre una malla insuficiente es peor que ninguno, porque viaja con la misma pinta que
uno bueno.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import torch

from ..data_generation import ExactGaussianMixture
from ..sde import ForwardSDE
from .quadrature import QuadratureGrid, auto_grid, integrate

#: Término constante de la log-densidad gaussiana en dos dimensiones: ``-(d/2) log(2π)``.
_LOG_NORMALIZACION_2D: float = -math.log(2.0 * math.pi)

#: Puntos de sondeo del chequeo estructural de familia. Sus coordenadas son **no nulas**
#: (para poder despejar ``α_t`` dividiendo) y **distintas entre sí**: con un vector de unos
#: cualquier kernel que **mezclara** coordenadas —por ejemplo uno que las promediara—
#: devolvería una media que igual parece proporcional a ``x_0``, y el rechazo se perdería.
_PROBE_X0: tuple[tuple[float, float], ...] = ((1.0, 2.0), (3.0, -1.5))

#: Fracciones del horizonte en las que se sondea el contrato marginal.
_PROBE_FRACTIONS: tuple[float, ...] = (0.25, 0.75)

#: Tolerancia relativa con la que se exige ``mean == α_t · x_0`` en el sondeo. Holgada
#: respecto del ruido de la doble precisión, pero órdenes de magnitud por debajo de
#: cualquier escala por coordenada que valga la pena rechazar.
_PROPORTIONALITY_TOL: float = 1e-10

#: Dimensión del laboratorio analítico. Aparece explícito en la KL gaussiana (el término
#: ``-d`` y el ``d·log σ_q²``) para que la fórmula se lea igual que en el papel.
_DIM: int = 2

#: Cuántos desvíos del estimador cubre la tolerancia que devuelve
#: :meth:`MixtureOracle.estimate_prior_variance`. El error relativo del estimador escala
#: como ``sqrt(2/n)`` —la varianza de ``x²`` para ``x ~ N(0, v)`` es ``2v²``—, así que cuatro
#: desvíos dejan una probabilidad de excederla del orden de ``1e-4``: la tolerancia sirve
#: como cota utilizable en lugar de como un intervalo que se viola una vez de cada tres.
_TOLERANCE_SIGMAS: float = 4.0


@dataclass(frozen=True, kw_only=True)
class BiasReport:
    """Sesgo de inicialización de una SDE: cuánto se aparta ``p_T`` del ruido de partida.

    Reporta **dos cantidades complementarias** en lugar de un único número, porque la KL
    entre una mixtura y una gaussiana no tiene forma cerrada (ver el docstring del módulo):
    una cota superior rigurosa que se calcula sin integrar nada, y un valor numérico de
    referencia con su tolerancia declarada. Que la cota domine al valor las valida
    mutuamente.

    Es un valor inmutable y se construye por palabra clave, así que ningún campo puede
    confundirse con otro al leerlo.

    Attributes:
        bound: Cota superior de forma cerrada, por convexidad de la KL. No negativa. Es un
            **techo**, no una estimación: en la mixtura de prueba de la spec quedó del orden
            de 2.5 veces el valor real.
        value: Valor numérico de referencia de la misma divergencia, obtenido por cuadratura
            sobre la log-densidad exacta. No negativo, y no mayor que ``bound`` salvo en el caso
            degenerado en que las dos son cero a nivel de redondeo (ahí ambas quedan del orden
            de ``1e-17`` y el orden entre ellas es ruido, no información).
        tolerance: Cota del error de ``value``, derivada de dos observables de la propia
            cuadratura: la desviación de ``mass`` respecto de uno y el paso de la malla (ver
            :func:`_quadrature_tolerance`). Es un número **computado**, no una constante.
        mass: Masa integrada de la densidad sobre **la misma malla** con la que se calculó
            ``value``: el autochequeo de que la malla alcanzó. Vale uno salvo redondeo.
        prior_variance: Varianza de la distribución de partida con la que se calculó, tal
            como la declaró el caller. Queda publicada porque es un dato **suyo**: no se
            deriva de la SDE.
        horizon: Instante en el que se evaluó el sesgo.
    """

    bound: float
    value: float
    tolerance: float
    mass: float
    prior_variance: float
    horizon: float

    def __post_init__(self) -> None:
        """Valida que el reporte describa cantidades admisibles.

        Raises:
            ValueError: Si ``bound`` no es finito y no negativo, si ``prior_variance`` no es
                finita y positiva, si ``horizon`` no es finito y no negativo, o si alguna de
                las cantidades que salen de la cuadratura no es finita y no negativa.
        """
        if not math.isfinite(self.bound) or self.bound < 0.0:
            raise ValueError(
                "bound debe ser finito y no negativo: es una divergencia acotada por "
                f"convexidad y cada término es una KL; recibí {self.bound!r}."
            )
        if not math.isfinite(self.prior_variance) or self.prior_variance <= 0.0:
            raise ValueError(
                "prior_variance debe ser finita y estrictamente positiva; recibí "
                f"{self.prior_variance!r}."
            )
        if not math.isfinite(self.horizon) or self.horizon < 0.0:
            raise ValueError(
                "horizon debe ser finito y no negativo: el proceso corre en [0, T]; recibí "
                f"{self.horizon!r}."
            )
        for nombre, valor in (
            ("value", self.value),
            ("tolerance", self.tolerance),
            ("mass", self.mass),
        ):
            if not math.isfinite(valor) or valor < 0.0:
                raise ValueError(
                    f"{nombre} debe ser finito y no negativo: sale de una cuadratura sobre "
                    f"una densidad no negativa; recibí {valor!r}."
                )


class MixtureOracle:
    """Verdad exacta de una mixtura 2D ruideada por una SDE escalar-gaussiana.

    Determinístico, sin estado mutable y sin necesidad de gradientes: no guarda cantidades
    derivadas entre llamadas (así no hay cachés que se desincronicen si el caller cambia el
    horizonte) y no muta ni la mixtura ni la SDE que recibe.

    La admisibilidad se valida **una vez, en construcción** (ver :meth:`__init__`), de modo
    que una corrida no arranque sobre premisas falsas y falle recién al integrar.

    Note:
        El oráculo describe la mixtura **exacta**. Los modelos entrenados sobre datos con
        estandarización empírica no quedan descritos por él: en ese caso la transformación
        se estima a partir de la muestra sorteada, así que los parámetros efectivos
        dependen de ``n`` y de la semilla. Las corridas del laboratorio 2D deben usar
        mixturas de parámetros exactos.
    """

    def __init__(self, mixture: ExactGaussianMixture, sde: ForwardSDE) -> None:
        """Construye el oráculo validando que la mixtura y la SDE sean admisibles.

        Los parámetros verdaderos de la mixtura se copian a tensores de doble precisión:
        el oráculo es dueño de su copia, así que nada de lo que devuelva puede
        desincronizarse de lo que se validó acá.

        Args:
            mixture: Mixtura de parámetros exactos (pesos, medias y covarianzas conocidos
                en forma cerrada).
            sde: Proceso forward de la familia escalar-gaussiana. Solo se le consulta el
                contrato marginal (``marginal_prob``) y el horizonte (``T``).

        Raises:
            ValueError: Si ``mixture`` pide la estandarización empírica, o si no es una
                mixtura de parámetros exactos. En los dos casos el resultado no sería
                exacto, que es la premisa del módulo.
            NotImplementedError: Si ``sde`` no pertenece a la familia escalar-gaussiana,
                verificado de forma estructural sobre lo que devuelve su contrato marginal.
        """
        if getattr(mixture, "standardize", False):
            raise ValueError(
                "mixture no puede usar la estandarización empírica (standardize=True): "
                "esa transformación se estima a partir de la muestra sorteada, así que los "
                "parámetros efectivos dependen del sorteo (de n y de la semilla) y el "
                "resultado no sería exacto. Construí una ExactGaussianMixture con los "
                f"parámetros que querés y standardize=False; recibí "
                f"{type(mixture).__name__} con standardize=True."
            )
        if not isinstance(mixture, ExactGaussianMixture):
            raise ValueError(
                "mixture debe ser una ExactGaussianMixture, la única fuente de datos que "
                "publica sus pesos, medias y covarianzas verdaderos en forma cerrada; "
                f"recibí {type(mixture).__name__}, cuyos parámetros no son consultables, "
                "así que ninguna cantidad derivada de ellos sería exacta."
            )

        self._mixture = mixture
        self._sde = sde

        # Parámetros verdaderos en doble precisión: los accesores de la mixtura ya
        # devuelven copias, así que estos tensores no aliasan nada del caller.
        self._weights = torch.as_tensor(mixture.weights_, dtype=torch.float64)
        self._means = torch.as_tensor(mixture.means_, dtype=torch.float64)
        self._covariances = torch.as_tensor(mixture.covariances_, dtype=torch.float64)

        self._validate_scalar_gaussian_family()

    # ------------------------------------------------- parámetros del kernel en t

    def marginal_params(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Factor de contracción ``α_t`` y desvío del kernel ``σ_t`` en cada tiempo.

        Los lee del contrato marginal de la SDE evaluándolo sobre un vector de unos, que
        para la familia escalar-gaussiana devuelve ``mean == α_t`` y ``std == σ_t``. No
        re-deriva ningún schedule y no ramifica por variante: el mismo camino de código
        sirve para VP, VE y sub-VP.

        El sondeo se arma en doble precisión, así que el schedule se evalúa en doble
        aunque ``t`` llegue en ``float32``.

        Args:
            t: Tiempos, de forma ``(B,)`` o ``(B, 1)``, finitos y no negativos.

        Returns:
            El par ``(alpha, sigma)``, cada uno de forma ``(B, 1)`` en ``float64`` y en el
            device de ``t``. ``sigma`` es el desvío **tal como lo publica la SDE**: para
            sub-VP eso es ``1 - e^{-∫β}``, sin raíz.

        Raises:
            ValueError: Si ``t`` no es un tensor de forma ``(B,)`` o ``(B, 1)`` con
                ``B >= 1``, o si tiene valores no finitos o negativos.
        """
        tt = _normalize_time(t)
        batch = tt.shape[0]
        ones = torch.ones(batch, 2, dtype=torch.float64, device=tt.device)
        mean, std = self._sde.marginal_prob(ones, tt)
        return _as_column(mean, batch), _as_column(std, batch)

    def component_covariances(self, t: torch.Tensor) -> torch.Tensor:
        """Covarianzas de las componentes ya propagadas al tiempo ``t``.

        Es ``Σ_k(t) = α_t² Σ_k + σ_t² I``: el kernel escala la covarianza de los datos por
        ``α_t²`` y le suma ruido isotrópico. Cada llamada arma un tensor nuevo, así que
        escribir en la salida no contamina las llamadas siguientes.

        Args:
            t: Tiempos, de forma ``(B,)`` o ``(B, 1)``, finitos y no negativos.

        Returns:
            Tensor ``(B, K, 2, 2)`` en ``float64`` y en el device de ``t``, simétrico y
            definido positivo por construcción.

        Raises:
            ValueError: Si ``t`` no cumple el contrato de :meth:`marginal_params`.
        """
        alpha, sigma = self.marginal_params(t)
        return _perturbed_covariances(
            self._covariances.to(device=alpha.device), alpha, sigma
        )

    # ------------------------------------------------- densidad y log-densidad

    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Log-densidad exacta ``log p_t(x)`` de la mixtura ruideada.

        Se calcula **en el dominio logarítmico**: la log-densidad de cada componente se
        combina con ``logsumexp`` sobre ``log w_k + log N(x; α_t μ_k, Σ_k(t))``, nunca
        sumando densidades para después tomarles el logaritmo. Esa es la diferencia que
        importa cuando el desvío del kernel se hace chico: lejos de los modos cada
        componente vale ``exp(-10^5)``, que en doble precisión es cero, así que el camino
        ingenuo devolvería ``-inf`` mientras este devuelve el valor correcto.

        Está construida enteramente con operaciones tensoriales, sin cortes de grafo ni
        desvíos por fuera del tensor, así que ``torch.autograd`` puede derivarla respecto de
        ``x`` (dos veces, lo que habilita la traza exacta del jacobiano que necesitan las
        métricas). El gradiente respecto de ``t``, en cambio, **no** fluye: el tiempo se
        desconecta al normalizarse.

        Los pasos intermedios corren en doble precisión aunque ``x`` llegue en ``float32``;
        solo el resultado se degrada al dtype del caller.

        Args:
            x: Puntos del plano, de forma ``(B, 2)``, en punto flotante.
            t: Tiempos, de forma ``(B,)`` o ``(B, 1)``, finitos y no negativos: un tiempo
                por punto, evaluado punto a punto.

        Returns:
            Tensor ``(B,)`` con la log-densidad de cada punto, en el dtype y el device de
            ``x``.

        Raises:
            ValueError: Si ``x`` no es un tensor de punto flotante de forma ``(B, 2)`` con
                ``B >= 1``, si ``t`` no cumple el contrato de :meth:`marginal_params`, o si
                ``t`` no trae exactamente un tiempo por punto de ``x``.
        """
        return self._log_prob_double(x, t).to(x.dtype)

    def prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Densidad exacta ``p_t(x)`` de la mixtura ruideada.

        Se **deriva** de :meth:`log_prob` exponenciándola, de modo que las dos cantidades no
        pueden discrepar. Donde la densidad no es representable el resultado desborda a
        cero, que es el valor correcto en punto flotante; la log-densidad sigue siendo
        informativa ahí.

        Args:
            x: Puntos del plano, de forma ``(B, 2)``, en punto flotante.
            t: Tiempos, de forma ``(B,)`` o ``(B, 1)``, finitos y no negativos.

        Returns:
            Tensor ``(B,)`` no negativo con la densidad de cada punto, en el dtype y el
            device de ``x``.

        Raises:
            ValueError: Igual que :meth:`log_prob`.
        """
        return torch.exp(self._log_prob_double(x, t)).to(x.dtype)

    # ------------------------------------------------------------------- score

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score exacto ``∇_x log p_t(x)`` de la mixtura ruideada.

        Sale de derivar la log-densidad de la mixtura respecto de ``x``::

            ∇ log p_t(x) = -Σ_k r_k(x, t) · Σ_k(t)^{-1} (x - α_t μ_k)

        donde ``r_k`` son las **responsabilidades posteriores**, es decir con qué probabilidad
        la componente ``k`` explica el punto ``x`` en el tiempo ``t``. Cada componente empuja
        hacia su propia media y el score es el promedio de esos empujes ponderado por las
        responsabilidades: un solo modo cerca del punto lo domina, y en la zona intermedia
        entre dos modos el score es una mezcla.

        Las responsabilidades se calculan **en el dominio logarítmico** (un ``softmax`` sobre
        ``log w_k + log N_k``, exactamente los mismos términos que arma :meth:`log_prob`), no
        dividiendo densidades. Es lo que mantiene la exactitud cuando el desvío del kernel se
        hace chico: ahí las densidades individuales desbordan a cero y el cociente ingenuo
        daría ``0/0``, mientras que la resta de logaritmos que hace ``softmax`` da el valor
        correcto. Compartir esos términos con la log-densidad también garantiza que el score
        coincida con su gradiente, en lugar de ser dos fórmulas que podrían divergir.

        Note:
            El valor es **exacto** en todo el rango de tiempos: ``σ_t`` **no** se recorta por
            un piso mínimo. Es una diferencia deliberada respecto del target de entrenamiento
            (:meth:`diffusion.sde.ForwardSDE.score_target`), que sí lo recorta para no dividir
            por cero cuando ``t → 0``. El oráculo es el **patrón de referencia** contra el que
            se mide el error de la red, no una réplica de lo que la red aprendió: recortarlo
            escondería justamente el régimen ``t → 0`` que se quiere medir. La magnitud crece
            como ``1/σ_t²``, pero se mantiene finita porque la concentración de ``p_t`` tiene
            piso ``α_t² λ_min(Σ_k)`` y no colapsa con el ruido del kernel.

        No necesita gradientes —es forma cerrada— y no muta nada, así que funciona tal cual
        dentro del ``torch.no_grad()`` con el que los samplers integran el reverso. Tampoco
        **impone** ``no_grad``: el grafo de ``x`` sigue vivo en la salida, porque la
        derivabilidad del módulo es parte de su contrato hacia las métricas.

        Args:
            x: Puntos del plano, de forma ``(B, 2)``, en punto flotante.
            t: Tiempos, de forma ``(B,)`` o ``(B, 1)``, finitos y no negativos: un tiempo por
                punto, evaluado punto a punto.

        Returns:
            Tensor ``(B, 2)`` —la misma forma que ``x``— con el score de cada punto, en el
            dtype y el device de ``x``. Los pasos intermedios se calculan en doble precisión
            aunque ``x`` llegue en ``float32``.

        Raises:
            ValueError: Si ``x`` no es un tensor de punto flotante de forma ``(B, 2)`` con
                ``B >= 1``, si ``t`` no cumple el contrato de :meth:`marginal_params`, o si
                ``t`` no trae exactamente un tiempo por punto de ``x``.
        """
        log_pesados, empuje = self._component_terms(x, t)
        # Responsabilidades posteriores: softmax sobre log w_k + log N_k, de forma (B, K).
        responsabilidades = torch.softmax(log_pesados, dim=1)
        score = -(responsabilidades.unsqueeze(-1) * empuje).sum(dim=1)
        return score.to(x.dtype)

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Alias de :meth:`score` que cumple el contrato de score de los samplers.

        Los cuatro samplers del Eje 2 consumen el score como un **invocable puro**
        ``(x, t) -> score`` (``diffusion.samplers.ScoreFn``), sin exigir un ``nn.Module``: con
        este alias el oráculo entra donde iría la red entrenada, sin tocar los samplers. Es la
        forma de sustituir el score aprendido por el verdadero y aislar así el error de
        estimación del de discretización.

        Args:
            x: Puntos del plano, de forma ``(B, 2)``. Los samplers pasan el estado actual.
            t: Tiempos, de forma ``(B,)`` o ``(B, 1)``. Los samplers pasan ``(B, 1)``.

        Returns:
            Tensor con la forma, el dtype y el device de ``x``.

        Raises:
            ValueError: Igual que :meth:`score`.
        """
        return self.score(x, t)

    def total_mass(self, t: float, *, grid: QuadratureGrid | None = None) -> float:
        """Masa integrada de la densidad en un tiempo dado: el autochequeo del oráculo.

        Que ``∫ p_t = 1`` es la forma de *verificar* la verdad analítica en vez de confiar
        en ella. La malla por defecto la dimensiona ``auto_grid`` a partir de las covarianzas
        **ya evaluadas en** ``t`` y de las medias contraídas ``α_t μ_k``, es decir de la
        escala real de la densidad que se está integrando; el paso **no** se deriva del
        desvío del kernel, porque la concentración tiene piso ``α_t² λ_min(Σ_k)`` y no
        colapsa cuando ``t → 0``.

        Un valor que se aparta de uno delata una malla insuficiente, no un error de la
        densidad.

        Args:
            t: Tiempo escalar, finito y no negativo, en el que se integra.
            grid: Malla explícita. Si se omite, se dimensiona automáticamente.

        Returns:
            La masa integrada como ``float``.

        Raises:
            ValueError: Si ``t`` no es un número real finito y no negativo.
        """
        instante = _normalize_scalar_time(t)
        medias, covarianzas = self._propagated_components(instante)
        malla = (
            grid
            if grid is not None
            else auto_grid(means=medias, covariances=covarianzas)
        )

        def densidad(puntos: torch.Tensor) -> torch.Tensor:
            """Densidad evaluada en los nodos de un bloque de la malla."""
            tiempos = torch.full(
                (puntos.shape[0],),
                instante,
                dtype=puntos.dtype,
                device=puntos.device,
            )
            return self.prob(puntos, tiempos)

        return integrate(densidad, malla)

    # ------------------------------------------------- sesgo de inicialización

    def initialization_bias(
        self,
        prior_variance: float,
        *,
        t: float | None = None,
        grid: QuadratureGrid | None = None,
    ) -> BiasReport:
        """Sesgo de inicialización: cuánto se aparta ``p_t`` del ruido del que arranca el sampler.

        Devuelve **dos cantidades complementarias** de ``KL(p_t ‖ N(0, σ_q² I))``, porque la
        divergencia entre una mixtura y una gaussiana no tiene forma cerrada (ver el docstring
        del módulo): una **cota superior** que es pura aritmética sobre los parámetros y un
        **valor numérico de referencia** que se integra. Que se sandwicheen
        (``bound ≥ value ≥ 0``) las valida mutuamente.

        La cota sale de la convexidad de la KL::

            KL( Σ_k w_k N_k ‖ q )  ≤  Σ_k w_k · KL( N_k ‖ q )

        y con cada término gaussiano-gaussiano escrito en dos dimensiones como::

            KL( N(m, S) ‖ N(0, σ_q² I) ) = ½[ tr(S)/σ_q² + mᵀm/σ_q² − 2 − log det S
                                              + 2 log σ_q² ]

        sobre ``m = α_t μ_k`` y ``S = Σ_k(t) = α_t² Σ_k + σ_t² I``. Para la cota **no se integra
        nada**: es aritmética sobre los parámetros, y por eso no depende de la malla. Es un
        **techo** y no una estimación; con una sola componente la desigualdad se satura y la
        cota **es** la divergencia exacta.

        El valor de referencia sí se integra, sobre la log-densidad exacta del propio módulo::

            value = ∫ p_t(x) · [ log p_t(x) − log N(x; 0, σ_q² I) ] dx

        La malla por defecto la dimensiona ``auto_grid`` a partir de las covarianzas **ya
        evaluadas en** ``t`` y de las medias contraídas ``α_t μ_k`` —la escala real de la
        densidad que se integra— y se amplía con ``prior_std = sqrt(prior_variance)``, porque el
        integrando compara contra la distribución de partida y esta puede ser mucho más ancha
        que los datos (para VE lo es: desvío ``5`` contra datos de escala ``1``). Sobre esa misma
        malla se integra la masa, que viaja en el reporte como autochequeo, y de ella y del paso
        sale la ``tolerance`` (ver :func:`_quadrature_tolerance`).

        El valor se recorta a cero por abajo: la desigualdad de Gibbs garantiza que una KL no es
        negativa, y el recorte absorbe únicamente el ruido de redondeo del caso degenerado en que
        ``p_t`` **es** la distribución de partida. Ahí el integrando es una cancelación entre dos
        log-densidades escritas con fórmulas distintas, así que la cuadratura devuelve del orden
        de ``1e-16`` con signo arbitrario y sin el recorte el reporte podría publicar una
        divergencia negativa.

        Args:
            prior_variance: Varianza de la distribución de la que parte el sampler.
                **Obligatoria**: no existe forma exacta ni genérica de leerla de la SDE
                (``prior_sampling`` solo muestrea y no publica su escala, derivarla por
                variante rompería el único camino de código, y agregarle un accesor a
                :mod:`diffusion.sde` está fuera de la frontera de este módulo). Los valores
                de las variantes registradas son ``1.0`` para VP y sub-VP y ``σ_max²`` para
                VE, es decir ``25.0`` con el ``σ_max = 5.0`` por defecto. Si hace falta
                averiguarla, :meth:`estimate_prior_variance` la estima de forma explícita y
                con su tolerancia a la vista; acá no hay caída automática a la estimación,
                porque entonces la "cota de forma cerrada" dejaría de serlo en silencio.
            t: Instante en el que se evalúa el sesgo. Si se omite, el horizonte ``T`` de la
                SDE, que es de donde el sampler realmente arranca.
            grid: Malla explícita para la cuadratura. Si se omite, se dimensiona
                automáticamente. Misma convención que :meth:`total_mass`. Solo afecta al valor
                numérico: la cota es la misma con cualquier malla.

        Returns:
            El :class:`BiasReport` con las dos cantidades, la masa integrada, la tolerancia, la
            varianza de partida usada y el horizonte.

        Raises:
            TypeError: Si se omite ``prior_variance``. Es un error de llamada de la propia
                firma, deliberadamente no un valor por defecto.
            ValueError: Si ``prior_variance`` no es un número real finito y estrictamente
                positivo, si ``t`` no es un número real finito y no negativo, o si la masa
                integrada se aparta de uno más allá de la tolerancia declarada — en ese caso la
                malla no alcanzó, y devolver el número sería peor que fallar, porque viajaría
                con la misma pinta que uno bueno.

        Note:
            No se usa ``σ_t`` como sustituto de la varianza de partida aunque esté a mano: en
            VP da ``0.99998`` contra un prior de varianza ``1.0``, y esa diferencia diminuta
            *es parte del sesgo que se quiere medir*, así que lo enmascararía.
        """
        varianza = _normalize_prior_variance(prior_variance)
        horizonte = float(self._sde.T) if t is None else _normalize_scalar_time(t)

        medias, covarianzas = self._propagated_components(horizonte)
        por_componente = _kl_against_isotropic(medias, covarianzas, varianza)
        cota = float((self._weights * por_componente).sum())

        malla = (
            grid
            if grid is not None
            else auto_grid(
                means=medias,
                covariances=covarianzas,
                prior_std=math.sqrt(varianza),
            )
        )
        # La masa se integra **antes** del valor y sobre la misma malla: si la malla no alcanza,
        # se avisa sin pagar la segunda pasada.
        masa = self.total_mass(horizonte, grid=malla)
        desviacion = abs(masa - 1.0)
        tolerancia = _quadrature_tolerance(malla, masa)
        if desviacion > tolerancia:
            raise ValueError(
                f"la masa integrada dio {masa!r} y se aparta de uno en {desviacion!r}, más que "
                f"la tolerancia {tolerancia!r} que una malla de {malla.n_points} nodos por eje "
                f"sobre [-{malla.half_width}, {malla.half_width}] admite: la malla no alcanza "
                "para esta densidad. El valor de referencia calculado sobre ella no sería "
                "confiable, así que no se devuelve. Ampliá el dominio o refiná el paso (o dejá "
                "que se dimensione automáticamente omitiendo grid)."
            )

        def divergencia_puntual(puntos: torch.Tensor) -> torch.Tensor:
            """Integrando ``p_t (log p_t − log q)`` en los nodos de un bloque de la malla."""
            tiempos = torch.full(
                (puntos.shape[0],),
                horizonte,
                dtype=puntos.dtype,
                device=puntos.device,
            )
            log_p = self._log_prob_double(puntos, tiempos)
            log_q = _log_isotropic_gaussian(puntos.to(torch.float64), varianza)
            densidad = torch.exp(log_p)
            # Donde la densidad desborda a cero el integrando vale cero: es el límite correcto
            # de ``p log p`` y evita el ``0 · (-inf)`` que daría ``nan`` si el caller pidiera una
            # malla tan ancha que la forma cuadrática desbordara.
            return torch.where(
                densidad > 0.0,
                densidad * (log_p - log_q),
                torch.zeros_like(densidad),
            )

        valor = max(integrate(divergencia_puntual, malla), 0.0)

        return BiasReport(
            bound=cota,
            value=valor,
            tolerance=tolerancia,
            mass=masa,
            prior_variance=varianza,
            horizon=horizonte,
        )

    def estimate_prior_variance(
        self, *, n: int = 1_000_000, seed: int = 0
    ) -> tuple[float, float]:
        """Estima la varianza de la distribución de partida muestreándola. **Opt-in**.

        Existe como respaldo para cuando el caller no conoce la escala del prior, y devuelve
        su tolerancia junto con el número justamente para dejar constancia de que el resultado
        **ya no es de forma cerrada**. Por eso es un método aparte y no el valor por defecto
        de :meth:`initialization_bias`: si la estimación entrara sola, la cota de forma cerrada
        dejaría de serlo sin que nadie lo note.

        Estima el **segundo momento por coordenada**, ``E[x²]``, que coincide con la varianza
        porque los priors de la familia están centrados en el origen —que es además lo que la
        cota supone al comparar contra ``N(0, σ_q² I)``—. La tolerancia es
        ``_TOLERANCE_SIGMAS · sqrt(2/n)``: la varianza de ``x²`` para ``x ~ N(0, v)`` es
        ``2v²``, así que el error relativo del promedio escala como ``sqrt(2/n)``. Se cuenta
        con los ``n`` puntos y no con los ``2n`` escalares que se promedian, lo que deja la
        tolerancia del lado conservador.

        Determinístico: siembra su propio generador, así que no depende ni contamina el estado
        global de torch.

        Args:
            n: Cantidad de puntos a muestrear. Cuantos más, más fina la tolerancia.
            seed: Semilla del generador propio.

        Returns:
            El par ``(varianza, tolerancia_relativa)``.

        Raises:
            ValueError: Si ``n`` no es un entero de al menos 2, o si ``seed`` no es entero.
        """
        if not isinstance(n, numbers.Integral) or int(n) < 2:
            raise ValueError(
                "n debe ser un entero de al menos 2 (es una cantidad de puntos a "
                f"muestrear); recibí {n!r}."
            )
        if not isinstance(seed, numbers.Integral):
            raise ValueError(
                "seed debe ser un entero para sembrar el generador; recibí un objeto de "
                f"tipo {type(seed).__name__}."
            )

        cantidad = int(n)
        generador = torch.Generator().manual_seed(int(seed))
        muestra = self._sde.prior_sampling(
            (cantidad, _DIM), generator=generador, dtype=torch.float64
        )
        varianza = float((muestra.to(torch.float64) ** 2).mean())
        tolerancia = _TOLERANCE_SIGMAS * math.sqrt(2.0 / cantidad)
        return varianza, tolerancia

    # ------------------------------------------------------------------ internos

    def _propagated_components(
        self, instante: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Medias y covarianzas de la mixtura ya propagadas a un instante **escalar**.

        Es la misma propagación de la que sale todo el módulo —``α_t μ_k`` y
        ``Σ_k(t) = α_t² Σ_k + σ_t² I``— pero para un único tiempo, que es la forma en la que
        la necesitan las cantidades escalares: la masa integrada y el sesgo de inicialización.
        Vive en un solo lugar para que la malla de la cuadratura y la cota cerrada se
        construyan sobre exactamente los mismos parámetros y no puedan divergir.

        Args:
            instante: Tiempo escalar, ya validado por :func:`_normalize_scalar_time`.

        Returns:
            El par ``(medias, covarianzas)`` de formas ``(K, 2)`` y ``(K, 2, 2)``, en
            ``float64``.
        """
        tt = torch.tensor([[instante]], dtype=torch.float64)
        alpha, sigma = self.marginal_params(tt)
        covarianzas = _perturbed_covariances(self._covariances, alpha, sigma)[0]
        medias = (alpha.reshape(-1, 1, 1) * self._means)[0]
        return medias, covarianzas

    def _log_prob_double(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Log-densidad en doble precisión, sin degradar al dtype del caller.

        Combina con ``logsumexp`` los mismos términos por componente que consume el score, de
        modo que las dos cantidades salen de una sola forma cerrada. :meth:`log_prob` y
        :meth:`prob` solo la exponencian o la degradan, así que ninguna de las tres puede
        discrepar de las otras.

        Args:
            x: Puntos del plano candidatos.
            t: Tiempos candidatos.

        Returns:
            Tensor ``(B,)`` ``float64`` en el device de ``x``, conectado al grafo de ``x``.

        Raises:
            ValueError: Si ``x`` o ``t`` no cumplen su contrato, o si los lotes no coinciden.
        """
        log_pesados, _ = self._component_terms(x, t)
        return torch.logsumexp(log_pesados, dim=1)

    def _component_terms(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Los dos términos por componente de los que salen la densidad y el score.

        Es el **único** lugar donde vive la forma cerrada de la mixtura ruideada. Devuelve las
        log-densidades ya ponderadas por los pesos (``log w_k + log N_k``, lo que la densidad
        combina con ``logsumexp`` y el score con ``softmax``) y el empuje blanqueado de cada
        componente (``Σ_k(t)^{-1} (x - α_t μ_k)``, lo que el score promedia). Compartirlos es
        lo que hace que el score sea exactamente el gradiente de la log-densidad y no una
        segunda fórmula capaz de divergir de ella.

        Todo se calcula en doble precisión —la promoción de ``x`` es derivable, así que el
        grafo del caller sobrevive— y con los parámetros de la mixtura movidos al device de
        ``x`` **en cada llamada**: no se cachea ninguna copia por device, así que el oráculo
        sirve indistintamente en CPU y en GPU.

        Args:
            x: Puntos del plano candidatos.
            t: Tiempos candidatos.

        Returns:
            El par ``(log_pesados, empuje)`` de formas ``(B, K)`` y ``(B, K, 2)``, en
            ``float64`` y en el device de ``x``.

        Raises:
            ValueError: Si ``x`` o ``t`` no cumplen su contrato, o si los lotes no coinciden.
        """
        xx = _normalize_state(x)
        tt = _normalize_time(t).to(device=xx.device)
        if tt.shape[0] != xx.shape[0]:
            raise ValueError(
                "t debe traer un tiempo por punto de x (un valor por fila); recibí "
                f"{tt.shape[0]} tiempos para {xx.shape[0]} puntos."
            )

        alpha, sigma = self.marginal_params(tt)  # (B, 1) cada uno
        base = self._covariances.to(device=xx.device)  # (K, 2, 2)
        covarianzas = _perturbed_covariances(base, alpha, sigma)  # (B, K, 2, 2)
        precisiones, determinantes = _inverse_and_det_2x2(covarianzas)

        # Medias contraídas por el kernel: alpha_t mu_k, con forma (B, K, 2).
        medias = alpha.reshape(-1, 1, 1) * self._means.to(device=xx.device)
        diferencia = xx.unsqueeze(1) - medias  # (B, K, 2)
        # Empuje blanqueado Sigma_k(t)^{-1} (x - alpha_t mu_k), de forma (B, K, 2). Es lo que
        # el score promedia y, contraído contra la diferencia, la forma cuadrática de la
        # log-densidad: un solo producto sirve a las dos cantidades.
        empuje = (precisiones @ diferencia.unsqueeze(-1)).squeeze(-1)
        # Forma cuadrática (x - mu_k)^T Sigma_k(t)^{-1} (x - mu_k), de forma (B, K).
        cuadratica = (diferencia * empuje).sum(dim=-1)

        log_componentes = (
            _LOG_NORMALIZACION_2D
            - 0.5 * torch.log(determinantes)
            - 0.5 * cuadratica
        )
        # Un peso nulo entra como -inf y logsumexp lo absorbe sin producir nan, así que no
        # hace falta recortar los pesos (recortarlos haría aportar densidad a una componente
        # que la mixtura declaró inexistente).
        log_pesos = torch.log(self._weights.to(device=xx.device))  # (K,)
        return log_pesos + log_componentes, empuje

    def _validate_scalar_gaussian_family(self) -> None:
        """Verifica estructuralmente que la SDE sea de la familia escalar-gaussiana.

        Sondea el contrato marginal con puntos de coordenadas distintas y en dos tiempos, y
        exige las dos propiedades que definen al kernel ``N(α_t x_0, σ_t² I)``: que la media
        sea proporcional a ``x_0`` con **un solo escalar por muestra**, y que el desvío
        colapse a un escalar por muestra en lugar de traer un valor por coordenada o una
        matriz.

        No consulta ningún atributo de familia: ``is_augmented`` ya no existe en el paquete
        y agregarlo estaría fuera de la frontera de este módulo.

        Raises:
            NotImplementedError: Si el contrato marginal no cumple alguna de las dos
                propiedades.
        """
        x0 = torch.tensor(_PROBE_X0, dtype=torch.float64)
        horizonte = float(self._sde.T)
        t = torch.tensor(
            [f * horizonte for f in _PROBE_FRACTIONS], dtype=torch.float64
        ).reshape(-1, 1)
        mean, std = self._sde.marginal_prob(x0, t)

        if not _collapses_to_scalar_per_sample(std, x0.shape[0]):
            raise NotImplementedError(
                "sde no pertenece a la familia escalar-gaussiana: su contrato marginal "
                f"devolvió un desvío de forma {tuple(std.shape)}, que no colapsa a un "
                f"escalar por muestra para un lote de {x0.shape[0]}. Este oráculo solo "
                "describe kernels N(alpha_t x_0, sigma_t^2 I); una SDE con desvío por "
                "coordenada o con estado aumentado necesita su propia forma cerrada."
            )
        if tuple(mean.shape) != tuple(x0.shape):
            raise NotImplementedError(
                "sde no pertenece a la familia escalar-gaussiana: su contrato marginal "
                f"devolvió una media de forma {tuple(mean.shape)} para un dato de forma "
                f"{tuple(x0.shape)}."
            )
        alpha = mean[:, :1] / x0[:, :1]
        if not torch.allclose(mean, alpha * x0, rtol=_PROPORTIONALITY_TOL, atol=1e-12):
            raise NotImplementedError(
                "sde no pertenece a la familia escalar-gaussiana: su contrato marginal no "
                "devolvió una media proporcional a x_0 (se espera mean == alpha_t * x_0 "
                "con alpha_t escalar por muestra, que es lo que hace recuperable alpha_t "
                "sondeando con un vector de unos)."
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"{type(self).__name__}(K={self._weights.shape[0]}, "
            f"sde={type(self._sde).__name__})"
        )


# ---------------------------------------------------------------- helpers numéricos


def _normalize_time(t: torch.Tensor) -> torch.Tensor:
    """Normaliza ``t`` a ``(B, 1)`` en doble precisión, validando el contrato de shapes.

    Sigue la convención del proyecto (``t`` aceptado como ``(B,)`` o ``(B, 1)``) y lleva el
    tiempo a doble precisión **antes** de que la SDE evalúe su schedule, para que los pasos
    intermedios no hereden la precisión del caller.

    Args:
        t: Tiempos candidatos.

    Returns:
        ``t`` como tensor ``(B, 1)`` ``float64``, desconectado del grafo, en su device.

    Raises:
        ValueError: Si ``t`` no es un tensor de torch, si su forma no es ``(B,)`` ni
            ``(B, 1)`` con ``B >= 1``, o si tiene valores no finitos o negativos.
    """
    if not isinstance(t, torch.Tensor):
        raise ValueError(
            "t debe ser un tensor de torch de forma (B,) o (B, 1) con un tiempo por "
            f"punto; recibí un objeto de tipo {type(t).__name__}."
        )
    forma_valida = t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1)
    if not forma_valida or t.shape[0] < 1:
        raise ValueError(
            "t debe tener forma (B,) o (B, 1) con B >= 1 (un tiempo por punto); recibí "
            f"forma {tuple(t.shape)}."
        )
    tt = t.detach().to(torch.float64).reshape(t.shape[0], 1)
    if not bool(torch.isfinite(tt).all()) or bool((tt < 0.0).any()):
        raise ValueError(
            "t debe ser finito y no negativo: el proceso corre en [0, T] y fuera de ese "
            "rango el schedule de la SDE no está definido."
        )
    return tt


def _normalize_scalar_time(t: float) -> float:
    """Valida un tiempo **escalar** y lo devuelve como ``float``.

    Es la contraparte de :func:`_normalize_time` para las cantidades que se calculan en un
    solo instante (la masa integrada), donde pedir un tensor sería ceremonia inútil.

    Se valida **por tipo** y no intentando una conversión: ``float()`` acepta cadenas como
    ``"0.3"``, y colar un tiempo escrito como texto es exactamente la clase de error que
    conviene que falle fuerte.

    Args:
        t: Tiempo candidato.

    Returns:
        El tiempo como ``float``.

    Raises:
        ValueError: Si ``t`` no es un número real (un tensor tampoco lo es: acá el tiempo es
            un escalar), o si no es finito y no negativo.
    """
    if not isinstance(t, numbers.Real):
        raise ValueError(
            "t debe ser un número real que exprese un instante del proceso; recibí un "
            f"objeto de tipo {type(t).__name__}."
        )
    instante = float(t)
    if not math.isfinite(instante) or instante < 0.0:
        raise ValueError(
            "t debe ser finito y no negativo: el proceso corre en [0, T] y fuera de ese "
            f"rango el schedule de la SDE no está definido; recibí {instante!r}."
        )
    return instante


def _normalize_prior_variance(prior_variance: float) -> float:
    """Valida la varianza de la distribución de partida y la devuelve como ``float``.

    Se valida **por tipo** y no intentando una conversión, por la misma razón que el tiempo
    escalar: ``float()`` acepta cadenas como ``"1.0"``, y colar la escala del prior escrita
    como texto es exactamente la clase de error que conviene que falle fuerte. Un tensor
    tampoco pasa: acá la varianza es un escalar declarado por el caller.

    Args:
        prior_variance: Varianza candidata.

    Returns:
        La varianza como ``float``.

    Raises:
        ValueError: Si no es un número real, o si no es finita y estrictamente positiva.
    """
    if not isinstance(prior_variance, numbers.Real):
        raise ValueError(
            "prior_variance debe ser un número real con la varianza de la distribución de "
            "la que arranca el sampler (1.0 para VP y sub-VP, sigma_max**2 para VE); recibí "
            f"un objeto de tipo {type(prior_variance).__name__}."
        )
    valor = float(prior_variance)
    if not math.isfinite(valor) or valor <= 0.0:
        raise ValueError(
            "prior_variance debe ser finita y estrictamente positiva: una gaussiana de "
            f"varianza nula o negativa no describe ninguna distribución de partida; recibí "
            f"{valor!r}."
        )
    return valor


def _kl_against_isotropic(
    means: torch.Tensor, covariances: torch.Tensor, prior_variance: float
) -> torch.Tensor:
    """``KL(N(m_k, S_k) ‖ N(0, σ_q² I))`` de cada componente, en forma cerrada.

    En dos dimensiones es::

        ½[ tr(S)/σ_q² + mᵀm/σ_q² − d − log det S + d·log σ_q² ]

    donde el ``d·log σ_q²`` es ``log det(σ_q² I)``, es decir el determinante del prior. Sin
    solvers ni cuadratura: traza, norma y determinante salen de los propios parámetros.

    El determinante se toma de :func:`_inverse_and_det_2x2`, que es la misma forma cerrada
    ``2×2`` que usan la densidad y el score y que además exige determinante positivo, así que
    una covarianza inadmisible avisa acá en lugar de producir un ``nan`` en el logaritmo.

    Cada término se recorta a cero por abajo: la desigualdad de Gibbs garantiza que una KL no
    es negativa, y el recorte absorbe únicamente el ruido de redondeo del caso degenerado en
    que la componente **coincide** con la distribución de partida, donde la forma cerrada es
    una cancelación exacta entre términos de magnitud comparable.

    Args:
        means: Medias ya contraídas por el kernel, de forma ``(K, 2)``.
        covariances: Covarianzas ya propagadas al tiempo de interés, de forma ``(K, 2, 2)``.
        prior_variance: Varianza de la distribución de partida, positiva.

    Returns:
        Tensor ``(K,)`` no negativo con la divergencia de cada componente.

    Raises:
        ValueError: Si alguna covarianza no tiene determinante finito y positivo.
    """
    _, determinantes = _inverse_and_det_2x2(covariances)
    trazas = covariances.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    normas2 = (means * means).sum(dim=-1)
    divergencias = 0.5 * (
        (trazas + normas2) / prior_variance
        - float(_DIM)
        - torch.log(determinantes)
        + _DIM * math.log(prior_variance)
    )
    return divergencias.clamp_min(0.0)


def _log_isotropic_gaussian(x: torch.Tensor, variance: float) -> torch.Tensor:
    """``log N(x; 0, σ_q² I)`` en dos dimensiones: la log-densidad de la distribución de partida.

    Se escribe acá, en vez de reusar la maquinaria de la mixtura, porque es el **otro** lado de
    la comparación: la referencia contra la que se mide ``p_t``. Una gaussiana isotrópica
    centrada en el origen no necesita ni inversas ni determinantes —la forma cuadrática es
    ``‖x‖²/σ_q²`` y el logaritmo del determinante es ``d log σ_q²``—, así que la fórmula queda a
    la vista y no comparte código con lo que se está midiendo.

    Args:
        x: Puntos del plano, de forma ``(N, 2)``, en doble precisión.
        variance: Varianza de la distribución de partida, positiva.

    Returns:
        Tensor ``(N,)`` con la log-densidad de cada punto.
    """
    return (
        _LOG_NORMALIZACION_2D
        - 0.5 * _DIM * math.log(variance)
        - (x * x).sum(dim=-1) / (2.0 * variance)
    )


def _quadrature_tolerance(grid: QuadratureGrid, mass: float) -> float:
    """Cota del error de la cuadratura, derivada de dos observables de la propia malla.

    Los dos observables son el **paso relativo** —el paso de la malla medido en unidades del
    dominio que cubre, ``spacing / (2 half_width)``, que es la inversa de la resolución— y la
    **desviación de la masa integrada respecto de uno**::

        tolerance = paso_relativo · ( paso_relativo + |mass − 1| )

    - El término ``paso_relativo²`` es el error de **discretización**: una suma de Riemann cuyas
      colas se anulan es de segundo orden en el paso. Sobre integrandos gaussianos la
      convergencia es de hecho espectral, así que el término es conservador por varios órdenes
      de magnitud (medido: cuadruplicar la resolución mueve el valor en ``3e-18`` mientras la
      tolerancia cae de ``1e-4`` a ``6e-6``).
    - El término ``paso_relativo · |mass − 1|`` es la parte atribuible a lo que el **dominio** no
      cubre, amortiguada por el mismo factor. El amortiguamiento es lo que vuelve *exigible* el
      autochequeo: comparar ``|mass − 1|`` contra esta tolerancia pide una masa tanto más cercana
      a uno cuanto más fina sea la malla. Si la masa entrara con coeficiente uno, la comparación
      se cumpliría siempre y el chequeo no diría nada.

    Es **adimensional** a propósito: no depende de las unidades de ``x``, así que la misma
    fórmula sirve para una mixtura de escala ``1`` y para una de escala ``1000``. Y no hay
    ninguna constante elegida a mano: los dos números salen de la malla y de la integral.

    Args:
        grid: Malla sobre la que se integró.
        mass: Masa integrada de la densidad sobre esa misma malla.

    Returns:
        La tolerancia como ``float`` positivo.
    """
    paso_relativo = grid.spacing / (2.0 * grid.half_width)
    return paso_relativo * (paso_relativo + abs(mass - 1.0))


def _normalize_state(x: torch.Tensor) -> torch.Tensor:
    """Valida el estado y lo promueve a doble precisión **sin cortar el grafo**.

    La promoción es una operación derivable, así que el gradiente respecto de un ``x`` en
    ``float32`` sigue llegando al caller. No se hace ``detach``: es justamente ``∂/∂x`` lo
    que los consumidores necesitan.

    Args:
        x: Puntos del plano candidatos.

    Returns:
        ``x`` como tensor ``(B, 2)`` ``float64``, en su device y conectado a su grafo.

    Raises:
        ValueError: Si ``x`` no es un tensor de torch, si su forma no es ``(B, 2)`` con
            ``B >= 1``, o si su dtype no es de punto flotante.
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError(
            "x debe ser un tensor de torch de forma (B, 2) con un punto del plano por "
            f"fila; recibí un objeto de tipo {type(x).__name__}."
        )
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 1:
        raise ValueError(
            "x debe tener forma (B, 2) con B >= 1 (el laboratorio analítico es 2D); "
            f"recibí forma {tuple(x.shape)}."
        )
    if not x.dtype.is_floating_point:
        raise ValueError(
            "x debe ser un tensor de punto flotante: un estado entero no describe un punto "
            f"del plano; recibí dtype {x.dtype}."
        )
    return x.to(torch.float64)


def _perturbed_covariances(
    base: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Propaga las covarianzas de los datos al tiempo del kernel.

    Es ``Σ_k(t) = α_t² Σ_k + σ_t² I``, la única fórmula de la que dependen la densidad, el
    score y la masa integrada. Vive en un solo lugar para que no puedan divergir.

    Args:
        base: Covarianzas de los datos, de forma ``(K, 2, 2)``.
        alpha: Factor de contracción por muestra, de forma ``(B, 1)``.
        sigma: Desvío del kernel por muestra, de forma ``(B, 1)``.

    Returns:
        Tensor ``(B, K, 2, 2)`` en el dtype y el device de ``base``.
    """
    identidad = torch.eye(2, dtype=base.dtype, device=base.device)
    alpha2 = (alpha**2).reshape(-1, 1, 1, 1)  # (B, 1, 1, 1)
    sigma2 = (sigma**2).reshape(-1, 1, 1, 1)
    return alpha2 * base + sigma2 * identidad


def _as_column(value: torch.Tensor, batch: int) -> torch.Tensor:
    """Colapsa un escalar por muestra a la forma canónica ``(batch, 1)``.

    El contrato marginal devuelve escalares por muestra con las dimensiones de evento en
    uno (``(B, 1)`` en 2D), y una SDE puede devolverlos ya broadcasteados desde una sola
    fila. Esta función unifica esos casos sin asumir la forma exacta.

    Args:
        value: Tensor con un escalar por muestra: ``(B, 1, …, 1)``, ``(1, 1, …, 1)`` o de
            rango cero.
        batch: Cantidad de muestras esperada.

    Returns:
        Tensor de forma ``(batch, 1)``.
    """
    if value.ndim == 0:
        return value.reshape(1, 1).expand(batch, 1)
    columna = value.reshape(value.shape[0], -1)[:, :1]
    if columna.shape[0] != batch:
        columna = columna.expand(batch, 1)
    return columna


def _collapses_to_scalar_per_sample(value: torch.Tensor, batch: int) -> bool:
    """Indica si ``value`` es un escalar por muestra (y no un valor por coordenada).

    Es el criterio estructural con el que se reconoce el desvío de un kernel isotrópico:
    todas las dimensiones más allá de la del lote tienen que ser uno, y la del lote tiene
    que ser la del dato o uno (broadcast).

    Args:
        value: Tensor a inspeccionar.
        batch: Cantidad de muestras del sondeo.

    Returns:
        ``True`` si describe un escalar por muestra.
    """
    if value.ndim == 0:
        return True
    if value.shape[0] not in (1, batch):
        return False
    return all(d == 1 for d in value.shape[1:])


def _inverse_and_det_2x2(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inversa y determinante de matrices ``2×2`` en forma cerrada.

    En dos dimensiones no hace falta un solver genérico: el determinante es ``ad - bc`` y la
    inversa es la adjunta ``[[d, -b], [-c, a]]`` dividida por él. Es exacto, barato, se
    vectoriza sobre cualquier cantidad de dimensiones de lote y —a diferencia de una
    factorización— no introduce ramas dependientes de los datos.

    Está pensada para las covarianzas ``Σ_k(t)``, que son definidas positivas, así que
    exige determinante positivo en lugar de solamente no nulo.

    Args:
        cov: Tensor de forma ``(…, 2, 2)``.

    Returns:
        El par ``(inversa, determinante)`` con formas ``(…, 2, 2)`` y ``(…)``.

    Raises:
        ValueError: Si la forma no termina en ``(2, 2)``, o si algún determinante no es
            finito y positivo (una matriz singular o no definida positiva no es una
            covarianza válida y devolver infinitos sería peor que avisar).
    """
    if cov.ndim < 2 or cov.shape[-2:] != (2, 2):
        raise ValueError(
            f"cov debe tener forma (..., 2, 2); recibida {tuple(cov.shape)}."
        )
    a = cov[..., 0, 0]
    b = cov[..., 0, 1]
    c = cov[..., 1, 0]
    d = cov[..., 1, 1]
    det = a * d - b * c
    if not bool(torch.isfinite(det).all()) or bool((det <= 0.0).any()):
        raise ValueError(
            "cov debe tener determinante finito y positivo para ser invertible como "
            f"covarianza; el mínimo encontrado fue {float(det.min())!r}."
        )
    fila0 = torch.stack((d, -b), dim=-1)
    fila1 = torch.stack((-c, a), dim=-1)
    adjunta = torch.stack((fila0, fila1), dim=-2)
    return adjunta / det[..., None, None], det
