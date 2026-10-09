"""Base abstracta de los samplers del proceso reverso (Eje 2).

Un :class:`ReverseSampler` integra numéricamente la ecuación reversa de Anderson (1982)

    ``dx = [f(x,t) - g(t)^2 ∇_x log p_t(x)] dt + g(t) dW̄``

—o su flujo de probabilidad determinístico (PF-ODE)— para generar muestras ``x_0`` a
partir del prior de ruido ``p_T``, **reusando** el score aprendido ``s_θ(x,t)`` sin
reentrenar la red. Es el **Eje 2** del estudio de ablación (ver ``docs/project/ejes.md``):
cambiar el sampler reusa el mismo score; cambiar la SDE (Eje 1) sí obliga a reentrenar.

Patrón Template Method: este ABC fija el algoritmo de integración compartido (grilla
temporal uniforme, drifts reversos derivados de ``sde.sde`` y del score), y cada sampler
concreto define solo su :meth:`step`. La red se consume como función pura ``ScoreFn`` y la
estocasticidad vive en el sampler (EM/PC) o se anula (PF-ODE/Heun), nunca en la red.

Igual que :mod:`diffusion.sde`, este módulo importa **torch directamente** (opera sobre
tensores; torch es dependencia dura).
"""

from __future__ import annotations

import abc
from typing import Callable

import torch

from diffusion.sde import ForwardSDE

from .time_grid import TimeGridFn, make_time_grid

#: Contrato de inyección del score: ``(x: (B, *E), t: (B,) | (B,1)) -> (B, *E)`` para
#: cualquier forma de evento ``E`` (``(2,)`` en el toy 2D, ``(C, H, W)`` en imágenes).
#: Tanto una :class:`diffusion.models.ScoreMLP` entrenada como un score analítico en forma
#: cerrada encajan sin cambios en el sampler.
ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ReverseSampler(abc.ABC):
    """Base de todos los samplers del proceso reverso.

    Un sampler concreto fija :attr:`name` (clave del futuro registry) e implementa
    :meth:`step`. El ABC aporta la grilla temporal uniforme de ``T`` a ``t_eps``, los dos
    drifts reversos compartidos (:meth:`_reverse_drift` para la SDE y :meth:`_pfode_drift`
    para el flujo de probabilidad) y la normalización temporal.

    La red se consume como función pura :attr:`score_fn` y **nunca** se muta.
    """

    #: Clave del registry/factory, p. ej. ``"euler"``. Sobreescribir en cada subclase.
    name: str = ""

    def __init__(
        self,
        sde: ForwardSDE,
        score_fn: ScoreFn,
        *,
        n_steps: int = 500,
        t_eps: float = 1e-3,
        time_grid: str | TimeGridFn = "uniform",
    ) -> None:
        """Inicializa el sampler.

        Args:
            sde: Proceso forward (Eje 1) del que se derivan los coeficientes ``(f, g)`` y
                el prior ``p_T``.
            score_fn: Función pura ``(x, t) -> score`` que aproxima ``∇_x log p_t(x)``.
            n_steps: Número de pasos (intervalos) de integración; ``>= 1``.
            t_eps: Tiempo terminal de la integración, un piso ``> 0`` que evita integrar
                hasta ``t = 0`` exacto; debe cumplir ``0 < t_eps < sde.T``.
            time_grid: Cómo se distribuyen los tiempos de la grilla entre ``T`` y ``t_eps``
                (ver :mod:`diffusion.samplers.time_grid`). Nombre registrado —``"uniform"``
                (default, espaciado constante en ``t``, idéntico al comportamiento previo) o
                ``"logsnr"`` (espaciado constante en el log-SNR)— **o** un callable propio
                ``(n_steps, t_min, t_max) -> (n_steps + 1,)`` decreciente. Es una elección
                puramente numérica: **no** obliga a reentrenar (Eje 2).

        Raises:
            ValueError: Si ``n_steps < 1``, si ``t_eps`` cae fuera de ``(0, sde.T)``, o si
                ``time_grid`` no es un nombre registrado ni un callable.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps debe ser >= 1; recibí n_steps={n_steps}")
        if not (0.0 < t_eps < sde.T):
            raise ValueError(
                f"t_eps debe cumplir 0 < t_eps < sde.T={sde.T}; recibí t_eps={t_eps}"
            )
        self.sde = sde
        self.score_fn = score_fn
        self.n_steps = int(n_steps)
        self.t_eps = float(t_eps)
        self.time_grid = make_time_grid(time_grid, sde, self.t_eps)

    # --------------------------------------------------------------- a implementar

    @abc.abstractmethod
    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso de integración de ``t`` a ``t + dt`` (con ``dt < 0``).

        Cada sampler concreto define su discretización; las subclases **no** recalculan la
        grilla (la maneja el driver).

        Args:
            x: Estado actual de shape ``(B, *E)`` para cualquier forma de evento ``E``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: se integra en tiempo decreciente).
            generator: Generador de torch para los samplers estocásticos; los
                determinísticos (PF-ODE/Heun) lo ignoran.

        Returns:
            El nuevo estado de shape ``(B, *E)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------- driver

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        *,
        init: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        return_trajectory: bool = False,
        denoise: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Integra el proceso reverso de ``T`` a ``t_eps`` y devuelve las muestras ``x_0``.

        Driver compartido (Template Method): arranca del prior ``p_T`` —o del ``init``
        provisto— y recorre la grilla temporal en tiempo **decreciente** (``dt < 0``),
        delegando cada paso en :meth:`step`. Corre bajo ``torch.no_grad()`` y en
        ``float32``, sin tocar los parámetros de la red (el score se consume como función
        pura), de modo que cambiar de sampler nunca reentrena (Eje 2).

        La geometría de salida sale de la forma de evento de la SDE
        (:attr:`ForwardSDE.data_shape`): el prior se arma como ``(n_samples, *data_shape)``,
        de modo que la integración corre igual sobre el toy 2D ``(N, 2)`` que sobre una forma
        tipo-imagen ``(N, C, H, W)`` — el driver y cada :meth:`step` operan sobre ``x.shape``.

        Args:
            n_samples: Número de muestras a generar (``N``).
            init: Estado inicial ``x_T`` de shape ``(n_samples, *sde.data_shape)``. Si es
                ``None`` se sortea de ``sde.prior_sampling``; pasarlo aísla el determinismo
                del integrador del muestreo del prior.
            generator: Generador de torch para reproducibilidad; alimenta tanto el muestreo
                del prior como los pasos estocásticos (EM/PC). Los samplers determinísticos
                (PF-ODE/Heun) lo ignoran en :meth:`step`. Si se corre en GPU con ``generator``
                provisto, debe estar en el mismo device (p. ej. ``torch.Generator(device="cuda")``).
            device: Device donde generar (``"cuda"``, ``"cpu"``, …). Solo fija dónde se sortea el
                prior cuando ``init`` es ``None``; con ``init`` provisto manda ``init.device``. La
                grilla temporal y el score se computan en el device de ``x``, así que con la red en
                GPU el sampleo corre en GPU. ``None`` (default) = CPU, camino idéntico al previo.
            return_trajectory: Si es ``True``, devuelve además la trayectoria completa.
            denoise: Si es ``True`` (default), aplica un paso final de *denoising* de Tweedie
                en ``t_eps`` —``E[x_0 | x_{t_eps}] = (x + σ_t² s) / α_t``— que quita el último
                ruido residual ``σ_{t_eps}`` y devuelve la estimación limpia del dato en vez
                del estado crudo en ``t_eps``. Con ``False`` devuelve el estado crudo.

        Returns:
            El estado final ``x_0`` de shape ``(n_samples, *sde.data_shape)`` en ``float32``
            (con ``denoise=True``, la estimación de Tweedie del dato; con ``False``, el estado
            crudo en ``t_eps``). Si ``return_trajectory`` es ``True``, una tupla
            ``(x_0, trayectoria)`` donde la trayectoria tiene shape
            ``(n_steps + 1, n_samples, *sde.data_shape)`` e incluye el estado inicial ``x_T``
            (capa ``0``) y cada estado intermedio; su última capa coincide con ``x_0``.
        """
        if init is None:
            x = self.sde.prior_sampling(
                (n_samples, *self.sde.data_shape),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            x = init.to(dtype=torch.float32)

        # La grilla temporal se mueve al device de x, así t, los coeficientes de la SDE y el score
        # se computan en el mismo device (GPU si x/red están en GPU): sin esto, t quedaría en CPU y
        # chocaría contra un x en GPU. Con x en CPU es un no-op (camino previo idéntico).
        grid = self._time_grid().to(x.device)
        trajectory: list[torch.Tensor] = [x.clone()] if return_trajectory else []

        for i in range(self.n_steps):
            t_cur = grid[i]
            t_next = grid[i + 1]
            dt = (t_next - t_cur).item()  # negativo: tiempo decreciente
            t_batch = t_cur.expand(n_samples, 1)
            x = self.step(x, t_batch, dt, generator=generator)
            if return_trajectory:
                trajectory.append(x.clone())

        # Paso final de denoising (Tweedie) en t_eps: estima E[x_0 | x_{t_eps}] y quita el
        # último ruido residual σ_{t_eps}. La última capa de la trayectoria queda en esta
        # estimación limpia, preservando el invariante trajectory[-1] == x_0.
        if denoise:
            x = self._denoise(x, grid[-1].expand(n_samples, 1))
            if return_trajectory:
                trajectory[-1] = x.clone()

        if return_trajectory:
            return x, torch.stack(trajectory, dim=0)
        return x

    # ----------------------------------------------------------- helpers compartidos

    def _time_grid(self) -> torch.Tensor:
        """Grilla temporal de ``T`` a ``t_eps``, según :attr:`time_grid`.

        Delega en la :class:`~diffusion.samplers.time_grid.TimeGrid` construida en
        ``__init__``. Con el default ``"uniform"`` es ``torch.linspace(T, t_eps,
        n_steps + 1)``, idéntico al comportamiento previo a que la grilla fuese configurable.

        Returns:
            Tensor ``float32`` de shape ``(n_steps + 1,)``, decreciente, con extremos
            ``T`` (primero) y ``t_eps`` (último).
        """
        return self.time_grid.grid(self.n_steps)

    def _reverse_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift de la SDE reversa ``f - g^2 s``.

        Args:
            x: Estado de shape ``(B, *E)`` para cualquier forma de evento ``E``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Tensor de shape ``(B, *E)``.
        """
        t = self._expand_t(t)
        f, g = self.sde.sde(x, t)
        s = self.score_fn(x, t)
        return f - (g ** 2) * s

    def _pfode_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift del flujo de probabilidad (PF-ODE) ``f - ½ g^2 s``.

        Comparte las mismas marginales que la SDE reversa pero sin término de ruido.

        Args:
            x: Estado de shape ``(B, *E)`` para cualquier forma de evento ``E``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Tensor de shape ``(B, *E)``.
        """
        t = self._expand_t(t)
        f, g = self.sde.sde(x, t)
        s = self.score_fn(x, t)
        return f - 0.5 * (g ** 2) * s

    def _denoise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Paso final de *denoising* de Tweedie: estima ``E[x_0 | x_t]`` en ``t_eps``.

        Para el kernel escalar-gaussiano ``x_t = α_t x_0 + σ_t ε`` con ``s ≈ ∇_x log p_t``,
        el estimador de la media posterior del dato es

            ``E[x_0 | x_t] = (x_t + σ_t² · s) / α_t``,

        que quita el último ruido residual ``σ_{t_eps}`` (VP/sub-VP dividen por ``α_t``; VE
        tiene ``α_t = 1``, así que solo suma ``σ_t² s``). Los coeficientes ``(α_t, σ_t)`` salen
        de :meth:`ForwardSDE.marginal_prob` evaluada en ``x_0 = 1``; el denominador se acota
        por debajo (``_std_eps``) por robustez, aunque en ``t_eps`` ``α_t ≈ 1``.

        Args:
            x: Estado ``x_{t_eps}`` de shape ``(B, *E)``.
            t: Tiempo ``t_eps`` de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            La estimación limpia del dato ``x_0`` de shape ``(B, *E)``.
        """
        t = self._expand_t(t)
        mean_coef, std = self.sde.marginal_prob(torch.ones_like(x), t)  # α_t, σ_t
        s = self.score_fn(x, t)
        return (x + std ** 2 * s) / mean_coef.clamp_min(self.sde._std_eps)

    # ----------------------------------------------------------------- internos

    @staticmethod
    def _expand_t(t: torch.Tensor) -> torch.Tensor:
        """Normaliza ``t`` de shape ``(B,)`` o ``(B, 1)`` a ``(B, 1)`` para broadcast."""
        return t.reshape(-1, 1)

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"{type(self).__name__}(sde={type(self.sde).__name__}, "
            f"n_steps={self.n_steps}, t_eps={self.t_eps})"
        )
