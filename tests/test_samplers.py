"""Tests del proceso reverso (`diffusion.samplers`).

Task 1.1: paquete importable + esqueleto del ABC ``ReverseSampler`` (grilla temporal,
drifts reversos compartidos). El driver ``sample()`` y los samplers concretos llegan en
tasks posteriores; acá solo se valida el contrato base.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.samplers.base import ReverseSampler
from diffusion.sde import make_sde

B = 16


def _zero_score(x, t):
    """``ScoreFn`` trivial: score nulo, shape igual a ``x``."""
    return torch.zeros_like(x)


class _NoOpSampler(ReverseSampler):
    """Subclase mínima para instanciar el ABC: ``step`` no-op."""

    name = "noop"

    def step(self, x, t, dt, *, generator):
        return x


# --------------------------------------------------------------- importabilidad


def test_package_importable():
    # El import a tope del módulo ya prueba el observable principal de 1.1.
    assert ReverseSampler is not None


# ------------------------------------------------------------- construcción / guardas


def test_instantiate_with_scalar_sde():
    sde = make_sde("vp")
    s = _NoOpSampler(sde, _zero_score)
    assert s.sde is sde
    assert s.score_fn is _zero_score


def test_n_steps_must_be_positive():
    sde = make_sde("vp")
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, n_steps=0)


def test_t_eps_out_of_range_raises():
    sde = make_sde("vp")  # T = 1.0
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, t_eps=0.0)
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, t_eps=-1e-3)
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, t_eps=sde.T)
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, t_eps=sde.T + 1.0)


def test_abstract_cannot_instantiate_directly():
    sde = make_sde("vp")
    with pytest.raises(TypeError):
        ReverseSampler(sde, _zero_score)  # step abstracto sin implementar


# ----------------------------------------------------------------- grilla temporal


def test_time_grid_endpoints_and_length():
    sde = make_sde("vp")
    n_steps = 50
    t_eps = 1e-3
    s = _NoOpSampler(sde, _zero_score, n_steps=n_steps, t_eps=t_eps)
    grid = s._time_grid()
    assert grid.shape == (n_steps + 1,)
    assert grid.dtype == torch.float32
    assert grid[0].item() == pytest.approx(sde.T)
    assert grid[-1].item() == pytest.approx(t_eps)
    # Decreciente de T a t_eps.
    assert torch.all(grid[:-1] > grid[1:])


# ------------------------------------------------------------------ drifts reversos


@pytest.mark.parametrize("dim", [2, 5])
def test_reverse_drift_shape_and_finite(dim):
    sde = make_sde("vp", data_dim=dim)
    s = _NoOpSampler(sde, _zero_score)
    x = torch.randn(B, dim)
    t = torch.rand(B)
    d = s._reverse_drift(x, t)
    assert d.shape == (B, dim)
    assert torch.all(torch.isfinite(d))


@pytest.mark.parametrize("dim", [2, 5])
def test_pfode_drift_shape_and_finite(dim):
    sde = make_sde("vp", data_dim=dim)
    s = _NoOpSampler(sde, _zero_score)
    x = torch.randn(B, dim)
    t = torch.rand(B)
    d = s._pfode_drift(x, t)
    assert d.shape == (B, dim)
    assert torch.all(torch.isfinite(d))


def test_drifts_accept_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) deben dar el mismo resultado.
    sde = make_sde("vp")
    s = _NoOpSampler(sde, _zero_score)
    x = torch.randn(B, 2)
    t = torch.rand(B)
    assert torch.equal(s._reverse_drift(x, t), s._reverse_drift(x, t.reshape(B, 1)))
    assert torch.equal(s._pfode_drift(x, t), s._pfode_drift(x, t.reshape(B, 1)))


def test_pfode_is_half_reverse_relative_to_drift():
    # Con score nulo ambos drifts coinciden con f; con score no nulo, la corrección de
    # PF-ODE es la mitad de la del reverso completo: f - g^2 s vs f - 0.5 g^2 s.
    sde = make_sde("vp")

    def score_fn(x, t):
        return torch.ones_like(x)

    s = _NoOpSampler(sde, score_fn)
    x = torch.randn(B, 2)
    t = torch.rand(B)
    f, g = sde.sde(x, t)
    rev = s._reverse_drift(x, t)
    pf = s._pfode_drift(x, t)
    # rev = f - g^2 s ; pf = f - 0.5 g^2 s  =>  f - pf = 0.5 (f - rev)
    assert torch.allclose(f - pf, 0.5 * (f - rev), atol=1e-5)


# --------------------------------------------------------------- driver sample()


class _PFODESampler(ReverseSampler):
    """Sampler determinístico mínimo: paso de Euler sobre el drift de PF-ODE.

    Sirve para ejercitar el driver ``sample()`` sin depender de los samplers
    concretos (tasks 2.x). Ignora ``generator`` (es determinístico).
    """

    name = "pfode_test"

    def step(self, x, t, dt, *, generator):
        return x + self._pfode_drift(x, t) * dt


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito.
    sde = make_sde(sde_name)
    s = _PFODESampler(sde, _zero_score, n_steps=20)
    n = 32
    x0 = s.sample(n)
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_sample_returns_trajectory_with_correct_shape():
    # 1.5: con return_trajectory devuelve (n_steps+1, N, data_dim) y x_0 aparte.
    sde = make_sde("vp")
    n_steps = 15
    s = _PFODESampler(sde, _zero_score, n_steps=n_steps)
    n = 8
    x0, traj = s.sample(n, return_trajectory=True)
    assert x0.shape == (n, sde.data_dim)
    assert traj.shape == (n_steps + 1, n, sde.data_dim)
    assert traj.dtype == torch.float32
    # El último estado de la trayectoria es x_0.
    assert torch.equal(traj[-1], x0)


def test_trajectory_first_slice_is_start_state():
    # 1.2 / 1.5: la trayectoria incluye el estado inicial x_T como primera capa.
    sde = make_sde("vp")
    s = _PFODESampler(sde, _zero_score, n_steps=10)
    n = 8
    gen = torch.Generator().manual_seed(0)
    x_T = sde.prior_sampling((n, sde.data_dim), generator=gen)
    gen2 = torch.Generator().manual_seed(0)
    _, traj = s.sample(n, generator=gen2, return_trajectory=True)
    assert torch.equal(traj[0], x_T)


def test_sample_uses_init_as_start_state():
    # init provisto se usa como x_T (no se sortea el prior).
    sde = make_sde("vp")
    s = _PFODESampler(sde, _zero_score, n_steps=10)
    n = 8
    init = torch.full((n, sde.data_dim), 3.0)
    x0, traj = s.sample(n, init=init, return_trajectory=True)
    assert torch.equal(traj[0], init)
    # Con score nulo y drift de VP el estado evoluciona, pero arranca en init.
    assert x0.shape == (n, sde.data_dim)


def test_sample_does_not_mutate_network_params():
    # 3.3: los parámetros de una ScoreMLP no cambian tras sample().
    ScoreMLP = pytest.importorskip("diffusion.models").ScoreMLP
    sde = make_sde("vp", data_dim=2)
    net = ScoreMLP(data_dim=2)
    net.eval()
    before = [p.detach().clone() for p in net.parameters()]

    def score_fn(x, t):
        return net(x, t)

    s = _PFODESampler(sde, score_fn, n_steps=10)
    s.sample(16)
    after = list(net.parameters())
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert torch.equal(b, a)


def test_sample_t_eps_floor_respected():
    # 8.2: la integración nunca llega a t = 0 exacto (piso t_eps > 0).
    sde = make_sde("vp")
    t_eps = 5e-3
    s = _PFODESampler(sde, _zero_score, n_steps=20, t_eps=t_eps)
    grid = s._time_grid()
    assert grid[-1].item() == pytest.approx(t_eps)
    assert torch.all(grid > 0.0)


def test_sample_determinism_with_fixed_init():
    # Integrador determinístico con init fijo -> salida idéntica (aísla del prior).
    sde = make_sde("vp")
    s = _PFODESampler(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init)
    b = s.sample(16, init=init)
    assert torch.equal(a, b)


# ----------------------------------------------------- task 2.1: Euler–Maruyama


def test_euler_name():
    # 2.1: la clave del registry queda fijada en la clase (factory llega en 3.1).
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    assert EulerMaruyama.name == "euler"


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_euler_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito sobre las 3 SDEs escalares.
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde(sde_name)
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    n = 32
    x0 = s.sample(n, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_euler_reproducible_same_seed():
    # 5.2: dos corridas con generadores del MISMO seed coinciden exactamente.
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("vp")
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(123))
    b = s.sample(16, generator=torch.Generator().manual_seed(123))
    assert torch.equal(a, b)


def test_euler_differs_with_different_seed():
    # 5.3: semillas distintas -> muestras distintas.
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("vp")
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


def test_euler_step_injects_noise():
    # 2.2: el paso es genuinamente estocástico — con score y drift nulos (VE: drift 0)
    # el estado igual cambia por la inyección de ruido del término de difusión.
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("ve")  # drift nulo
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    x = torch.zeros(8, sde.data_dim)
    t = torch.full((8, 1), sde.T)
    out = s.step(x, t, dt=-0.05, generator=torch.Generator().manual_seed(0))
    # Con f=0 y s=0, out = g·√|dt|·Z, que no puede ser idénticamente x (=0).
    assert not torch.equal(out, x)
    assert torch.all(torch.isfinite(out))


def test_euler_step_accepts_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) dan el mismo resultado (mismo generador sembrado).
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("vp")
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)
    out_flat = s.step(x, t, dt=-0.05, generator=torch.Generator().manual_seed(7))
    out_col = s.step(x, t.reshape(8, 1), dt=-0.05, generator=torch.Generator().manual_seed(7))
    assert torch.equal(out_flat, out_col)


# --------------------------------------------- task 2.2: Probability-Flow ODE


def test_pf_ode_name():
    # 2.3: la clave del registry queda fijada en la clase (factory llega en 3.1).
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    assert ProbabilityFlowODE.name == "pf_ode"


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_pf_ode_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito sobre las 3 SDEs escalares.
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    sde = make_sde(sde_name)
    s = ProbabilityFlowODE(sde, _zero_score, n_steps=20)
    n = 32
    x0 = s.sample(n, init=torch.randn(n, sde.data_dim))
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_pf_ode_deterministic_same_init():
    # 5.1: dos corridas con el MISMO init producen resultados idénticos.
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    sde = make_sde("vp")
    s = ProbabilityFlowODE(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init)
    b = s.sample(16, init=init)
    assert torch.equal(a, b)


def test_pf_ode_ignores_generator():
    # 5.1 / 2.3: determinístico — con el mismo init, semillas distintas dan el MISMO
    # resultado (prueba que el sampler ignora la aleatoriedad del generator).
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    sde = make_sde("vp")
    s = ProbabilityFlowODE(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, init=init, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)


def test_pf_ode_step_is_pfode_drift_euler():
    # 2.3: el paso es exactamente x + (f - ½ g² s)·dt (Euler sobre el drift de PF-ODE).
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    def score_fn(x, t):
        return torch.ones_like(x)

    sde = make_sde("vp")
    s = ProbabilityFlowODE(sde, score_fn, n_steps=20)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    dt = -0.05
    expected = x + s._pfode_drift(x, t) * dt
    out = s.step(x, t, dt, generator=None)
    assert torch.equal(out, expected)
    assert torch.all(torch.isfinite(out))


def test_pf_ode_step_accepts_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) dan el mismo resultado (determinístico).
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    sde = make_sde("vp")
    s = ProbabilityFlowODE(sde, _zero_score, n_steps=20)
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)
    out_flat = s.step(x, t, dt=-0.05, generator=None)
    out_col = s.step(x, t.reshape(8, 1), dt=-0.05, generator=None)
    assert torch.equal(out_flat, out_col)


# ----------------------------------------------------- task 2.3: Heun (ODE 2º orden)


def test_heun_name():
    # 2.4: la clave del registry queda fijada en la clase (factory llega en 3.1).
    from diffusion.samplers.heun import HeunODE

    assert HeunODE.name == "heun"


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_heun_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito sobre las 3 SDEs escalares.
    from diffusion.samplers.heun import HeunODE

    sde = make_sde(sde_name)
    s = HeunODE(sde, _zero_score, n_steps=20)
    n = 32
    x0 = s.sample(n, init=torch.randn(n, sde.data_dim))
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_heun_deterministic_same_init():
    # 5.1: dos corridas con el MISMO init producen resultados idénticos.
    from diffusion.samplers.heun import HeunODE

    sde = make_sde("vp")
    s = HeunODE(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init)
    b = s.sample(16, init=init)
    assert torch.equal(a, b)


def test_heun_ignores_generator():
    # 5.1 / 2.4: determinístico — con el mismo init, semillas distintas dan el MISMO
    # resultado (prueba que el sampler ignora la aleatoriedad del generator).
    from diffusion.samplers.heun import HeunODE

    sde = make_sde("vp")
    s = HeunODE(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, init=init, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)


def test_heun_step_matches_trapezoidal_formula():
    # 2.4: el paso es exactamente la regla del trapecio (predictor Euler + corrección):
    #   d1 = _pfode_drift(x, t); x̂ = x + d1·dt; d2 = _pfode_drift(x̂, t+dt)
    #   out = x + ½(d1 + d2)·dt
    from diffusion.samplers.heun import HeunODE

    def score_fn(x, t):
        # Score no constante en x para que la 2ª evaluación difiera de la 1ª.
        return x

    sde = make_sde("vp")
    s = HeunODE(sde, score_fn, n_steps=20)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    dt = -0.05
    d1 = s._pfode_drift(x, t)
    x_pred = x + d1 * dt
    d2 = s._pfode_drift(x_pred, t + dt)
    expected = x + 0.5 * (d1 + d2) * dt
    out = s.step(x, t, dt, generator=None)
    assert torch.equal(out, expected)
    assert torch.all(torch.isfinite(out))


def test_heun_does_two_score_evaluations():
    # 2.4: costo observable de DOS evaluaciones de score por paso (predictor + corrector).
    from diffusion.samplers.heun import HeunODE

    calls = {"n": 0}

    def counting_score(x, t):
        calls["n"] += 1
        return torch.ones_like(x)

    sde = make_sde("vp")
    s = HeunODE(sde, counting_score, n_steps=20)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    s.step(x, t, dt=-0.05, generator=None)
    assert calls["n"] == 2


def test_heun_differs_from_single_euler_pfode_step():
    # 2.4: la corrección de 2º orden es real (no un no-op): con drift no constante,
    # un paso de Heun difiere de un único paso de Euler sobre el drift de PF-ODE.
    from diffusion.samplers.heun import HeunODE
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    def score_fn(x, t):
        return x  # drift depende de x -> d(x̂, t+dt) != d(x, t)

    sde = make_sde("vp")
    heun = HeunODE(sde, score_fn, n_steps=20)
    pf = ProbabilityFlowODE(sde, score_fn, n_steps=20)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    dt = -0.05
    out_heun = heun.step(x, t, dt, generator=None)
    out_euler = pf.step(x, t, dt, generator=None)
    assert not torch.equal(out_heun, out_euler)


def test_heun_step_accepts_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) dan el mismo resultado (determinístico).
    from diffusion.samplers.heun import HeunODE

    sde = make_sde("vp")
    s = HeunODE(sde, _zero_score, n_steps=20)
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)
    out_flat = s.step(x, t, dt=-0.05, generator=None)
    out_col = s.step(x, t.reshape(8, 1), dt=-0.05, generator=None)
    assert torch.equal(out_flat, out_col)


# ------------------------------------------- task 2.4: predictor–corrector


def _linear_score(x, t):
    """Score analítico no nulo (apunta al origen) para ejercitar el corrector."""
    return -x


def test_pc_name():
    # 2.5: la clave del registry queda fijada en la clase (factory llega en 3.1).
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    assert PredictorCorrector.name == "pc"


def test_pc_constructor_accepts_extra_kwargs():
    # 4.4 / 2.5: es el único sampler con kwargs propios (n_corrector, snr).
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _zero_score, n_steps=20, n_corrector=3, snr=0.2)
    assert s.n_corrector == 3
    assert s.snr == pytest.approx(0.2)


def test_pc_constructor_defaults():
    # 2.5: defaults documentados (Song et al.): n_corrector=1, snr=0.16.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _zero_score, n_steps=20)
    assert s.n_corrector == 1
    assert s.snr == pytest.approx(0.16)


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_pc_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito sobre las 3 SDEs escalares.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde(sde_name)
    s = PredictorCorrector(sde, _linear_score, n_steps=20, n_corrector=1)
    n = 32
    x0 = s.sample(n, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_pc_sample_finite_with_real_mlp():
    # 1.4: con una ScoreMLP real (sin entrenar) el corrector se ejercita y produce finito.
    ScoreMLP = pytest.importorskip("diffusion.models").ScoreMLP
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp", data_dim=2)
    net = ScoreMLP(data_dim=2)
    net.eval()

    def score_fn(x, t):
        return net(x, t)

    s = PredictorCorrector(sde, score_fn, n_steps=20, n_corrector=2)
    x0 = s.sample(16, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (16, 2)
    assert torch.all(torch.isfinite(x0))


def test_pc_reproducible_same_seed():
    # 5.2: dos corridas con generadores del MISMO seed coinciden exactamente.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _linear_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(123))
    b = s.sample(16, generator=torch.Generator().manual_seed(123))
    assert torch.equal(a, b)


def test_pc_differs_with_different_seed():
    # 5.3: semillas distintas -> muestras distintas.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _linear_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


def test_pc_n_corrector_zero_reduces_to_predictor():
    # 2.5: n_corrector controla el número de correcciones de Langevin. Con n_corrector=0
    # el paso es solo el predictor (Euler–Maruyama); debe diferir de n_corrector=2.
    from diffusion.samplers.predictor_corrector import PredictorCorrector
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("vp")
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    dt = -0.05

    pc0 = PredictorCorrector(sde, _linear_score, n_steps=20, n_corrector=0)
    em = EulerMaruyama(sde, _linear_score, n_steps=20)
    out0 = pc0.step(x, t, dt, generator=torch.Generator().manual_seed(5))
    out_em = em.step(x, t, dt, generator=torch.Generator().manual_seed(5))
    # Sin correcciones, el paso PC coincide con el predictor Euler–Maruyama.
    assert torch.equal(out0, out_em)

    pc2 = PredictorCorrector(sde, _linear_score, n_steps=20, n_corrector=2)
    out2 = pc2.step(x, t, dt, generator=torch.Generator().manual_seed(5))
    # Con correcciones de Langevin el resultado difiere del predictor solo.
    assert not torch.equal(out2, out0)


def test_pc_counts_score_calls():
    # 2.5: predictor usa el drift reverso (1 eval de score) + n_corrector evals de Langevin.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    calls = {"n": 0}

    def counting_score(x, t):
        calls["n"] += 1
        return -x

    sde = make_sde("vp")
    s = PredictorCorrector(sde, counting_score, n_steps=20, n_corrector=3)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    s.step(x, t, dt=-0.05, generator=torch.Generator().manual_seed(0))
    assert calls["n"] == 1 + 3  # predictor (reverse drift) + n_corrector Langevin


def test_pc_nan_safe_with_zero_score():
    # 8.2: con score nulo el denominador ‖s‖ es ~0; el piso evita div-by-zero / NaN.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _zero_score, n_steps=20, n_corrector=2)
    x0 = s.sample(16, generator=torch.Generator().manual_seed(0))
    assert torch.all(torch.isfinite(x0))


def test_pc_step_accepts_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) dan el mismo resultado (mismo generador sembrado).
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _linear_score, n_steps=20)
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)
    out_flat = s.step(x, t, dt=-0.05, generator=torch.Generator().manual_seed(7))
    out_col = s.step(x, t.reshape(8, 1), dt=-0.05, generator=torch.Generator().manual_seed(7))
    assert torch.equal(out_flat, out_col)


# ----------------------------------------------- task 3.1: registry y factory


def test_available_samplers_sorted_names():
    # 4.2: la factory expone la lista de nombres disponibles (ordenados).
    from diffusion.samplers import available_samplers

    assert available_samplers() == ["euler", "heun", "pc", "pf_ode"]


def test_registry_maps_names_to_classes():
    # 4.1/2.1: REGISTRY mapea cada nombre a su clase de sampler.
    from diffusion.samplers import REGISTRY
    from diffusion.samplers.euler_maruyama import EulerMaruyama
    from diffusion.samplers.heun import HeunODE
    from diffusion.samplers.pf_ode import ProbabilityFlowODE
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    assert REGISTRY == {
        "euler": EulerMaruyama,
        "pf_ode": ProbabilityFlowODE,
        "heun": HeunODE,
        "pc": PredictorCorrector,
    }


@pytest.mark.parametrize(
    "name,cls_name",
    [
        ("euler", "EulerMaruyama"),
        ("pf_ode", "ProbabilityFlowODE"),
        ("heun", "HeunODE"),
        ("pc", "PredictorCorrector"),
    ],
)
def test_make_sampler_returns_correct_type(name, cls_name):
    # 4.1/2.1: make_sampler instancia el sampler correcto configurado con sde y score.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler(name, sde, _zero_score)
    assert type(s).__name__ == cls_name
    assert s.sde is sde
    assert s.score_fn is _zero_score


def test_make_sampler_unknown_name_lists_options():
    # 4.3: nombre desconocido -> ValueError enumerando las opciones válidas.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    with pytest.raises(ValueError) as excinfo:
        make_sampler("does_not_exist", sde, _zero_score)
    msg = str(excinfo.value)
    for opt in ("euler", "heun", "pc", "pf_ode"):
        assert opt in msg


def test_make_sampler_discards_inapplicable_kwargs():
    # 4.4: kwargs que no aplican al sampler elegido se descartan sin fallar (caller genérico).
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    # snr/n_corrector son exclusivos de PC; euler debe ignorarlos sin error.
    s = make_sampler("euler", sde, _zero_score, snr=0.5, n_corrector=3)
    assert type(s).__name__ == "EulerMaruyama"
    assert not hasattr(s, "n_corrector")
    assert not hasattr(s, "snr")


def test_make_sampler_applies_pc_kwargs():
    # 4.4: para PC, los kwargs propios SÍ se aplican (no se descartan).
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler("pc", sde, _zero_score, snr=0.5, n_corrector=3)
    assert type(s).__name__ == "PredictorCorrector"
    assert s.n_corrector == 3
    assert s.snr == pytest.approx(0.5)


def test_make_sampler_passes_common_kwargs():
    # 4.1/8.4: kwargs comunes (n_steps, t_eps) llegan al constructor base.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler("euler", sde, _zero_score, n_steps=33, t_eps=5e-3)
    assert s.n_steps == 33
    assert s.t_eps == pytest.approx(5e-3)


# --------------------------------------- task 3.2: generate_from_checkpoint


def _make_checkpoint(tmp_path, sde_name="vp", data_dim=2):
    """Arma un checkpoint válido SIN entrenar: red sin entrenar -> save_checkpoint.

    Reusa el contrato real (R5-c) de :mod:`diffusion.training` (``TrainResult`` con ``data_dim``
    +  ``save_checkpoint`` con ``model_spec``), de modo que ``load_checkpoint`` devuelva el mismo
    ``(state_dict, meta)`` que produciría una corrida real y ``generate_from_checkpoint``
    reconstruya la red vía la receta ``meta["model"]``. Devuelve la ruta del ``.pt`` guardado.
    """
    ScoreMLP = pytest.importorskip("diffusion.models").ScoreMLP
    from diffusion.training import TrainConfig, TrainResult, save_checkpoint

    net = ScoreMLP(data_dim=data_dim)
    result = TrainResult(
        net=net, history=[1.0, 0.5], config=TrainConfig(), sde_name=sde_name, data_dim=data_dim
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(result, path, model_spec={"name": "mlp", "kwargs": {"data_dim": data_dim}})
    return path


def test_generate_from_checkpoint_returns_samples(tmp_path):
    # 6.1: a partir de un checkpoint reconstruye SDE+red y genera (N, data_dim) float32.
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path, sde_name="vp", data_dim=2)
    x0 = generate_from_checkpoint(ckpt, "pf_ode", n_samples=8, n_steps=5, seed=0)
    assert x0.shape == (8, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_generate_from_checkpoint_writes_npz(tmp_path):
    # 6.2: con `out` persiste un .npz con la clave `samples`.
    pytest.importorskip("numpy")
    import numpy as np
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path)
    out = tmp_path / "gen" / "out.npz"  # subdir inexistente: debe crearse
    x0 = generate_from_checkpoint(
        ckpt, "pf_ode", n_samples=8, n_steps=5, seed=0, out=out
    )
    assert out.exists()
    with np.load(out) as data:
        assert "samples" in data
        assert data["samples"].shape == (8, 2)
        assert np.allclose(data["samples"], x0.cpu().numpy())


def test_generate_from_checkpoint_saves_trajectory(tmp_path):
    # 6.3: con save_trajectory el .npz también guarda la trayectoria.
    pytest.importorskip("numpy")
    import numpy as np
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path)
    out = tmp_path / "out.npz"
    n_steps = 5
    generate_from_checkpoint(
        ckpt, "pf_ode", n_samples=8, n_steps=n_steps, seed=0, out=out,
        save_trajectory=True,
    )
    with np.load(out) as data:
        assert "samples" in data
        assert "trajectory" in data
        # (n_steps + 1, N, data_dim): incluye el estado inicial x_T.
        assert data["trajectory"].shape == (n_steps + 1, 8, 2)


def test_generate_from_checkpoint_no_trajectory_key_when_disabled(tmp_path):
    # 6.3: sin save_trajectory el .npz NO contiene la clave `trajectory`.
    pytest.importorskip("numpy")
    import numpy as np
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path)
    out = tmp_path / "out.npz"
    generate_from_checkpoint(ckpt, "pf_ode", n_samples=8, n_steps=5, seed=0, out=out)
    with np.load(out) as data:
        assert "trajectory" not in data


def test_generate_from_checkpoint_reproducible_with_seed(tmp_path):
    # 6.1/6.2: misma seed -> resultado reproducible (incluso para un sampler estocástico).
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path)
    a = generate_from_checkpoint(ckpt, "euler", n_samples=8, n_steps=5, seed=42)
    b = generate_from_checkpoint(ckpt, "euler", n_samples=8, n_steps=5, seed=42)
    assert torch.equal(a, b)


def test_generate_from_checkpoint_reconstructs_sde_from_meta(tmp_path):
    # 6.1: la SDE se reconstruye desde meta["sde_name"] (no se pasa por argumento).
    # Un checkpoint con sde_name="ve" debe generar con la SDE VE sin pedir el nombre.
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path, sde_name="ve", data_dim=2)
    x0 = generate_from_checkpoint(ckpt, "pf_ode", n_samples=8, n_steps=5, seed=0)
    assert x0.shape == (8, 2)
    assert torch.all(torch.isfinite(x0))


def test_generate_from_checkpoint_missing_path_raises(tmp_path):
    # 6.4: ruta inexistente -> error claro que menciona la ruta.
    from diffusion.samplers import generate_from_checkpoint

    missing = tmp_path / "no_existe.pt"
    with pytest.raises((FileNotFoundError, ValueError)) as excinfo:
        generate_from_checkpoint(missing, "pf_ode", n_samples=4, n_steps=3)
    assert "no_existe.pt" in str(excinfo.value)


def test_generate_from_checkpoint_invalid_meta_raises(tmp_path):
    # 6.4: checkpoint sin las claves de meta esperadas -> error claro.
    torch_mod = pytest.importorskip("torch")
    from diffusion.samplers import generate_from_checkpoint

    bad = tmp_path / "bad.pt"
    torch_mod.save({"not": "a checkpoint"}, bad)
    with pytest.raises((KeyError, ValueError)):
        generate_from_checkpoint(bad, "pf_ode", n_samples=4, n_steps=3)


# --------------------------------------- task 4.4: huecos de generación checkpoint-driven


def _make_checkpoint_blob_missing_meta_key(tmp_path, missing_key, data_dim=2):
    """Arma a mano un blob que SOBREVIVE ``load_checkpoint`` pero le falta ``missing_key``.

    Aísla la guarda propia de ``generate.py`` (la lectura de ``meta["sde_name"]``/
    ``meta["data_dim"]``): con el contrato R5-c ``load_checkpoint`` sólo devuelve
    ``(state_dict, meta)`` sin tocar la metadata, así que el blob siempre "sobrevive" la carga
    y el error aparece recién cuando ``generate_from_checkpoint`` lee la clave faltante (p. ej.
    ``meta["sde_name"]``). Esto discrimina del test ``_invalid_meta_raises`` (que ni siquiera
    tiene la clave ``meta`` y dispara el ``KeyError("meta")`` en ``load_checkpoint``).

    El ``state_dict`` proviene de una ``ScoreMLP`` real (sin entrenar) coherente con la receta
    ``meta["model"]`` (``{name, kwargs}``), de modo que la reconstrucción posterior no falle por
    claves de pesos.
    """
    ScoreMLP = pytest.importorskip("diffusion.models").ScoreMLP

    net = ScoreMLP(data_dim=data_dim)
    meta = {
        "sde_name": "vp",
        "data_dim": data_dim,
        "model": {"name": "mlp", "kwargs": {"data_dim": data_dim}},
        "history": [1.0, 0.5],
    }
    del meta[missing_key]  # quita SOLO la clave bajo prueba; el resto queda válido
    blob = {"model_state": net.state_dict(), "meta": meta}
    path = tmp_path / "incomplete_meta.pt"
    torch.save(blob, path)
    return path


def test_generate_from_checkpoint_meta_missing_sde_name_raises_clear_error(tmp_path):
    # 6.4: aísla la guarda PROPIA de generate.py. El checkpoint tiene un `meta` válido
    # salvo por `sde_name`: sobrevive load_checkpoint y falla recién al reconstruir la SDE.
    # El error debe identificar claramente la clave faltante (no un KeyError("meta") opaco).
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint_blob_missing_meta_key(tmp_path, "sde_name")
    with pytest.raises(KeyError) as excinfo:
        generate_from_checkpoint(ckpt, "pf_ode", n_samples=4, n_steps=3)
    # Mensaje claro y atribuible a la(s) clave(s) de meta esperada(s) por generate.py.
    msg = str(excinfo.value)
    assert "sde_name" in msg
    assert "data_dim" in msg


def test_generate_from_checkpoint_end_to_end_npz_roundtrip_seeded(tmp_path):
    # 6.1+6.2+6.4 consolidado y discriminante con un sampler ESTOCÁSTICO (euler):
    # checkpoint -> generate -> .npz, el contenido releído con np.load coincide con el
    # tensor devuelto, y dos corridas con la misma seed reproducen byte a byte. Cierra el
    # round-trip completo del seam ejercitando el camino estocástico (no sólo pf_ode det.).
    pytest.importorskip("numpy")
    import numpy as np
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path, sde_name="vp", data_dim=2)
    out = tmp_path / "roundtrip" / "euler.npz"  # subdir inexistente: debe crearse
    x0 = generate_from_checkpoint(
        ckpt, "euler", n_samples=8, n_steps=5, seed=7, out=out
    )
    assert out.exists()
    with np.load(out) as data:
        assert set(data.files) == {"samples"}  # sin trajectory cuando no se pide
        assert data["samples"].shape == (8, 2)
        assert data["samples"].dtype == np.float32
        # El .npz persiste EXACTAMENTE el tensor devuelto.
        assert np.array_equal(data["samples"], x0.cpu().numpy())
    # Reproducibilidad por seed del camino estocástico vía el checkpoint.
    x0_again = generate_from_checkpoint(ckpt, "euler", n_samples=8, n_steps=5, seed=7)
    assert torch.equal(x0, x0_again)


# ------------------------------------------------- task 3.3: smoke entrypoint


def test_main_smoke_runs_all_samplers():
    # 2.1: el smoke entrypoint recorre el registry, corre cada sampler sobre una ScoreMLP
    # sin entrenar y reporta salidas finitas. main() devuelve un resumen assertable
    # {name: is_finite} con los cuatro samplers, todos finitos.
    pytest.importorskip("diffusion.models")
    from diffusion.samplers.__main__ import main
    from diffusion.samplers import available_samplers

    summary = main()
    assert set(summary) == set(available_samplers())
    assert len(summary) == 4
    assert all(summary.values())


# ------------------------------------------------------- task 3.4: CLI sample.py


def _load_sample_cli():
    """Importa ``scripts/sample.py`` por ruta (vive fuera del paquete ``diffusion``)."""
    import importlib.util
    import pathlib

    script = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sample.py"
    )
    spec = importlib.util.spec_from_file_location("_sample_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_sample_writes_npz(tmp_path):
    # 6.1/6.2: la CLI genera un .npz con `samples` (N, data_dim) desde un checkpoint.
    pytest.importorskip("diffusion.models")
    pytest.importorskip("numpy")
    import numpy as np

    cli = _load_sample_cli()
    ckpt = _make_checkpoint(tmp_path, sde_name="vp", data_dim=2)
    out = tmp_path / "gen" / "out.npz"  # subdir inexistente: debe crearse
    rc = cli.main(
        [
            str(ckpt),
            "--sampler", "pf_ode",
            "--n-samples", "8",
            "--n-steps", "5",
            "--seed", "0",
            "--out", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    with np.load(out) as data:
        assert "samples" in data
        assert data["samples"].shape == (8, 2)
        assert "trajectory" not in data


def test_cli_sample_trajectory_flag(tmp_path):
    # 6.3: con --trajectory el .npz también guarda la trayectoria.
    pytest.importorskip("diffusion.models")
    pytest.importorskip("numpy")
    import numpy as np

    cli = _load_sample_cli()
    ckpt = _make_checkpoint(tmp_path)
    out = tmp_path / "out.npz"
    n_steps = 5
    rc = cli.main(
        [
            str(ckpt),
            "--sampler", "pf_ode",
            "--n-samples", "8",
            "--n-steps", str(n_steps),
            "--seed", "0",
            "--out", str(out),
            "--trajectory",
        ]
    )
    assert rc == 0
    with np.load(out) as data:
        assert "samples" in data
        assert "trajectory" in data
        assert data["trajectory"].shape == (n_steps + 1, 8, 2)


def test_cli_sample_parser_has_sampler_choices():
    # 4.2/6.1: el parser ofrece --sampler con las opciones del registry.
    from diffusion.samplers import available_samplers

    cli = _load_sample_cli()
    parser = cli.build_parser()
    # parse mínimo válido: checkpoint + sampler del registry.
    args = parser.parse_args(["ckpt.pt", "--sampler", available_samplers()[0]])
    assert args.sampler in available_samplers()
    assert args.n_steps == 500  # default de generate_from_checkpoint


# ================================================================================
# Task 4.1: contrato + factory consolidados sobre la matriz 4 samplers × 3 SDEs.
#
# Las tasks 1.x–3.1 ya cubrieron cada ítem con tests focalizados (construyendo las
# clases concretas directamente). Acá se consolida el producto cruzado COMPLETO por
# el camino del registry (``make_sampler``), que es la interfaz real del estudio de
# ablación, y se cierran los huecos que sólo se ejercitaban sobre el doble de test
# ``_PFODESampler`` (trayectoria y red-intacta) o sólo a nivel atributo (n_steps).
# ================================================================================

# Los cuatro samplers del Eje 2 × las tres SDEs escalares (2.1, 7.2).
_ALL_SAMPLERS = ["euler", "pf_ode", "heun", "pc"]
_SCALAR_SDES = ["vp", "ve", "sub_vp"]
_SAMPLER_X_SDE = [(s, d) for s in _ALL_SAMPLERS for d in _SCALAR_SDES]


@pytest.mark.parametrize("sampler_name,sde_name", _SAMPLER_X_SDE)
def test_contract_sample_shape_dtype_finite_via_factory(sampler_name, sde_name):
    # 1.1, 1.4, 8.3, 2.1, 4.1, 7.2: para CADA sampler × CADA SDE escalar, construido
    # por el registry (make_sampler), sample(N) devuelve (N, data_dim) float32 finito.
    from diffusion.samplers import make_sampler

    sde = make_sde(sde_name)
    # _linear_score (apunta al origen) es no trivial y mantiene a PC numéricamente sano.
    s = make_sampler(sampler_name, sde, _linear_score, n_steps=15)
    n = 24
    x0 = s.sample(n, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


@pytest.mark.parametrize("sampler_name", _ALL_SAMPLERS)
def test_contract_return_trajectory_shape_via_factory(sampler_name):
    # 1.5, 8.4: para cada sampler real (vía factory), return_trajectory devuelve
    # (n_steps+1, N, data_dim) float32, coherente con el n_steps configurado, y el
    # último estado de la trayectoria coincide con x_0.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    n_steps = 12
    s = make_sampler(sampler_name, sde, _linear_score, n_steps=n_steps)
    n = 8
    x0, traj = s.sample(
        n, generator=torch.Generator().manual_seed(0), return_trajectory=True
    )
    assert x0.shape == (n, sde.data_dim)
    assert traj.shape == (n_steps + 1, n, sde.data_dim)
    assert traj.dtype == torch.float32
    assert torch.equal(traj[-1], x0)


@pytest.mark.parametrize("sampler_name", _ALL_SAMPLERS)
def test_contract_n_steps_configurable_changes_trajectory_length(sampler_name):
    # 8.4: n_steps es configurable y se refleja de punta a punta — la longitud de la
    # trayectoria sigue a n_steps (n_steps+1 capas), no es un valor fijo cableado.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    n = 8
    s_short = make_sampler(sampler_name, sde, _linear_score, n_steps=7)
    s_long = make_sampler(sampler_name, sde, _linear_score, n_steps=21)
    assert s_short.n_steps == 7
    assert s_long.n_steps == 21
    _, traj_short = s_short.sample(
        n, generator=torch.Generator().manual_seed(0), return_trajectory=True
    )
    _, traj_long = s_long.sample(
        n, generator=torch.Generator().manual_seed(0), return_trajectory=True
    )
    assert traj_short.shape[0] == 7 + 1
    assert traj_long.shape[0] == 21 + 1


@pytest.mark.parametrize("sampler_name", _ALL_SAMPLERS)
def test_contract_grid_starts_at_T_ends_at_t_eps_via_factory(sampler_name):
    # 1.2, 8.2, 8.4: la grilla de cada sampler (vía factory) arranca en T, termina en
    # t_eps (>0) y tiene n_steps+1 puntos, decreciente.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    n_steps = 10
    t_eps = 2e-3
    s = make_sampler(sampler_name, sde, _linear_score, n_steps=n_steps, t_eps=t_eps)
    grid = s._time_grid()
    assert grid.shape == (n_steps + 1,)
    assert grid[0].item() == pytest.approx(sde.T)
    assert grid[-1].item() == pytest.approx(t_eps)
    assert torch.all(grid > 0.0)
    assert torch.all(grid[:-1] > grid[1:])


@pytest.mark.parametrize("sampler_name", _ALL_SAMPLERS)
def test_contract_net_params_unchanged_after_sample_via_factory(sampler_name):
    # 3.3, 3.2: con una ScoreMLP real inyectada como score_fn, ningún sampler (vía
    # factory) altera los parámetros de la red durante sample().
    ScoreMLP = pytest.importorskip("diffusion.models").ScoreMLP
    from diffusion.samplers import make_sampler

    sde = make_sde("vp", data_dim=2)
    net = ScoreMLP(data_dim=2)
    net.eval()
    before = [p.detach().clone() for p in net.parameters()]

    def score_fn(x, t):
        return net(x, t)

    s = make_sampler(sampler_name, sde, score_fn, n_steps=10)
    s.sample(16, generator=torch.Generator().manual_seed(0))
    after = list(net.parameters())
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert torch.equal(b, a)


# ================================================================================
# Task 4.2: matriz consolidada de determinismo / reproducibilidad por la factory.
#
# Las tasks 2.x ya cubrieron cada propiedad con tests focalizados construyendo las
# clases concretas directamente (test_pf_ode_deterministic_same_init,
# test_pf_ode_ignores_generator, test_heun_deterministic_same_init,
# test_heun_ignores_generator, test_euler_reproducible_same_seed,
# test_euler_differs_with_different_seed, test_pc_reproducible_same_seed,
# test_pc_differs_with_different_seed). Acá se consolida la CLASE de propiedad por
# el camino real del estudio (``make_sampler``), afirmándola de modo uniforme:
#  - determinísticos (pf_ode, heun): mismo init -> idénticos; e idénticos aun con
#    semillas distintas (prueba que ignoran la aleatoriedad del generator).
#  - estocásticos (euler, pc): mismo generator sembrado -> idénticos; semillas
#    distintas -> distintos.
# ================================================================================

_DETERMINISTIC_SAMPLERS = ["pf_ode", "heun"]
_STOCHASTIC_SAMPLERS = ["euler", "pc"]


@pytest.mark.parametrize("sampler_name", _DETERMINISTIC_SAMPLERS)
def test_deterministic_same_init_identical_via_factory(sampler_name):
    # 5.1, 2.3, 4.1: sampler determinístico (vía factory) dos veces con el MISMO init
    # -> resultados idénticos (torch.equal). Aísla el integrador del muestreo del prior.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler(sampler_name, sde, _linear_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init)
    b = s.sample(16, init=init)
    assert torch.equal(a, b)


@pytest.mark.parametrize("sampler_name", _DETERMINISTIC_SAMPLERS)
def test_deterministic_ignores_generator_seed_via_factory(sampler_name):
    # 5.1, 2.3, 4.1: con el MISMO init, dos semillas DISTINTAS del generator dan el
    # mismo resultado -> el sampler determinístico ignora la aleatoriedad inyectada.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler(sampler_name, sde, _linear_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, init=init, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)


@pytest.mark.parametrize("sampler_name", _STOCHASTIC_SAMPLERS)
def test_stochastic_same_seed_reproducible_via_factory(sampler_name):
    # 5.2, 2.2, 4.1: sampler estocástico (vía factory) dos veces con generadores del
    # MISMO seed -> resultados idénticos (un generator FRESCO por llamada reproduce el
    # sorteo del prior y de cada inyección de ruido).
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler(sampler_name, sde, _linear_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(123))
    b = s.sample(16, generator=torch.Generator().manual_seed(123))
    assert torch.equal(a, b)


@pytest.mark.parametrize("sampler_name", _STOCHASTIC_SAMPLERS)
def test_stochastic_different_seed_differs_via_factory(sampler_name):
    # 5.3, 2.2, 4.1: sampler estocástico (vía factory) con semillas DISTINTAS -> las
    # muestras difieren (la estocasticidad del sampler es real, no un no-op).
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler(sampler_name, sde, _linear_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


# ================================================================================
# Task 4.3: correctitud matemática con score analítico (test CLAVE) + seam e2e.
#
# Es el guardián de que la matemática del proceso reverso esté bien implementada,
# INDEPENDIENTE del entrenamiento de la red (Req 7). Idea (design.md "Correctness
# Test", §research.md): para un target gaussiano isotrópico p_data = N(μ, σ0² I), bajo
# VP/VE/sub-VP la marginal p_t es gaussiana y su score analítico es cerrado:
#
#   marginal_prob(x0, t) -> (mean, std) del kernel p_t(x_t|x_0) = N(α_t·x0, std_t² I)
#   => marginal sobre p_data:  p_t(x) = N(α_t·μ, (α_t²·σ0² + std_t²) I)
#   => score:  s(x, t) = -(x - α_t·μ) / (α_t²·σ0² + std_t²)        (isótropo)
#
# α_t se extrae llamando marginal_prob con x0 = ones (mean == α_t broadcast); std_t es
# el std devuelto. Inyectando ese ScoreFn, CADA sampler debe RECUPERAR N(μ, σ0² I):
# media empírica ≈ μ y std empírico por dim ≈ σ0 dentro de tolerancia Monte Carlo.
#
# Calibración (medida empíricamente, semilla fija): N=3000, n_steps=300 corre en ~6 s
# total en CPU. Errores observados de media ≤ 0.02 (VP/sub-VP, todos los samplers; VE
# euler/pc) y ≤ 0.163 (VE pf_ode/heun: el flujo determinístico no borra del todo el
# desfasaje del prior de VE, N(0, σ_max²), respecto de la marginal real N(μ, σ0²+σ_max²)
# en t=T — estable en 0.135–0.163 sobre 5 semillas). Errores de std ≤ 0.025. Tolerancias
# elegidas (mean abs ≤ 0.20, std abs ≤ 0.10) quedan holgadas sobre lo observado pero
# DISCRIMINANTES: con score de signo invertido el error de media va de 16 a >400 000, y
# con score nulo el std explota a 140+ (VP/sub-VP) o queda en 5.0 (VE) ≠ 0.5 — el test
# falla rotundamente ante cualquier integración incorrecta del reverso.
# ================================================================================

_MU = [1.5, -1.0]
_SIGMA0 = 0.5
# Cobertura 4×3 (Req 7.2): los cuatro samplers del Eje 2 × las tres SDEs escalares.
_CORRECTNESS_CELLS = [(s, d) for s in _ALL_SAMPLERS for d in _SCALAR_SDES]


def _analytic_marginal_score(sde, mu, sigma0):
    """``ScoreFn`` del score analítico de la marginal gaussiana p_t sobre N(μ, σ0² I).

    Captura ``sde``, ``mu`` (tensor (d,)) y ``sigma0``. Extrae α_t y std_t de
    ``sde.marginal_prob`` y devuelve ``s(x,t) = -(x - α_t·μ) / (α_t²·σ0² + std_t²)``.
    Es isótropo: el denominador es escalar por muestra, shape (B,1), broadcast sobre d.
    """
    d = mu.shape[0]
    ones = torch.ones(1, d)

    def score(x, t):
        mean_alpha, std_t = sde.marginal_prob(ones, t)  # mean_alpha == α_t (broadcast)
        alpha_t = mean_alpha[:, :1]                      # (B,1) o (1,1)
        var_t = alpha_t ** 2 * sigma0 ** 2 + std_t ** 2  # (B,1) varianza de la marginal
        return -(x - alpha_t * mu) / var_t

    return score


@pytest.mark.parametrize("sampler_name,sde_name", _CORRECTNESS_CELLS)
def test_recovers_gaussian_with_analytic_score(sampler_name, sde_name):
    # 7.1, 7.2: con el score analítico de N(μ, σ0² I) inyectado, cada sampler (×SDE
    # escalar) recupera la media y el desvío del target dentro de tolerancia Monte Carlo.
    from diffusion.samplers import make_sampler

    mu = torch.tensor(_MU)
    sde = make_sde(sde_name)
    score_fn = _analytic_marginal_score(sde, mu, _SIGMA0)
    s = make_sampler(sampler_name, sde, score_fn, n_steps=300)
    x0 = s.sample(3000, generator=torch.Generator().manual_seed(0))

    assert torch.all(torch.isfinite(x0))
    emp_mean = x0.mean(dim=0)
    emp_std = x0.std(dim=0)
    # Media empírica ≈ μ (tolerancia abs por dim; calibrada > peor caso 0.163 de VE det.).
    assert torch.allclose(emp_mean, mu, atol=0.20), (
        f"{sampler_name}/{sde_name}: mean {emp_mean.tolist()} != {_MU} (atol=0.20)"
    )
    # Desvío empírico por dim ≈ σ0 (tolerancia abs; calibrada > peor caso ~0.025).
    target_std = torch.full_like(emp_std, _SIGMA0)
    assert torch.allclose(emp_std, target_std, atol=0.10), (
        f"{sampler_name}/{sde_name}: std {emp_std.tolist()} != {_SIGMA0} (atol=0.10)"
    )


# ------------------------------------------------------------- seam end-to-end (7.3)


def test_seam_make_sde_scoremlp_make_sampler_shapes():
    # 7.3: la cadena make_sde + ScoreMLP (red real) + make_sampler encaja en shapes de
    # punta a punta: sample(N) -> (N, data_dim) float32 finito en la dimensión por defecto.
    ScoreMLP = pytest.importorskip("diffusion.models").ScoreMLP
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")  # data_dim=2 por defecto
    net = ScoreMLP(data_dim=2)
    net.eval()

    def score_fn(x, t):
        return net(x, t)

    s = make_sampler("euler", sde, score_fn, n_steps=15)
    n = 16
    x0 = s.sample(n, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (n, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_seam_dimension_agnostic_non_default_dim():
    # 7.3: la cadena es agnóstica de la dimensión — con data_dim=3 (no el default 2) en
    # SDE y red, make_sampler genera salida (N, 3). Confirma que data_dim fluye coherente.
    ScoreMLP = pytest.importorskip("diffusion.models").ScoreMLP
    from diffusion.samplers import make_sampler

    sde = make_sde("vp", data_dim=3)
    assert sde.data_dim == 3
    net = ScoreMLP(data_dim=3)
    net.eval()

    def score_fn(x, t):
        return net(x, t)

    s = make_sampler("pf_ode", sde, score_fn, n_steps=15)
    n = 16
    x0 = s.sample(n, init=torch.randn(n, 3))
    assert x0.shape == (n, 3)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


# ================================================================================
# Task 2 (nd-shapes): prior N-D en los samplers.
#
# El muestreo del prior arma la forma a partir de la forma de evento de la SDE
# (``sde.data_shape``), de modo que la integración reversa corre sobre cualquier rango.
# El driver, los step() de los 4 samplers y la normalización temporal del sampler NO
# cambian (la SDE re-expande ``t`` contra el estado). Se ejercita sobre una forma
# tipo-imagen chica ``(3, 8, 8)`` (rápida en CPU) por los cuatro samplers vía la factory.
# Requisitos: 2.1, 2.2, 2.3, 2.4, 5.2.
# ================================================================================

_IMAGE_SHAPE = (3, 8, 8)


@pytest.mark.parametrize("sampler_name", _ALL_SAMPLERS)
def test_sample_nd_shape_dtype_finite_via_factory(sampler_name):
    # 2.1, 2.2, 2.4: para una SDE con forma de evento (3, 8, 8), cada sampler (vía
    # factory) muestrea el prior N-D e integra el reverso devolviendo (n, 3, 8, 8)
    # float32 finito. Un score nulo mantiene la geometría y no introduce NaN/Inf.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp", data_dim=_IMAGE_SHAPE)
    s = make_sampler(sampler_name, sde, _zero_score, n_steps=4)
    n = 4
    x0 = s.sample(n, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (n, *_IMAGE_SHAPE)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


@pytest.mark.parametrize("sampler_name", _ALL_SAMPLERS)
def test_sample_nd_return_trajectory_shape_via_factory(sampler_name):
    # 2.3: con captura de trayectoria sobre una forma de evento (3, 8, 8), cada sampler
    # devuelve (n_steps+1, n, 3, 8, 8) float32; el último estado coincide con x_0.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp", data_dim=_IMAGE_SHAPE)
    n_steps = 4
    s = make_sampler(sampler_name, sde, _zero_score, n_steps=n_steps)
    n = 4
    x0, traj = s.sample(
        n, generator=torch.Generator().manual_seed(0), return_trajectory=True
    )
    assert x0.shape == (n, *_IMAGE_SHAPE)
    assert traj.shape == (n_steps + 1, n, *_IMAGE_SHAPE)
    assert traj.dtype == torch.float32
    assert torch.equal(traj[-1], x0)


@pytest.mark.parametrize("sampler_name", _ALL_SAMPLERS)
def test_sample_nd_finite_across_scalar_sdes(sampler_name):
    # 2.2: finitud N-D sobre las TRES SDEs escalares (no solo VP) para la forma imagen.
    from diffusion.samplers import make_sampler

    for sde_name in _SCALAR_SDES:
        sde = make_sde(sde_name, data_dim=_IMAGE_SHAPE)
        s = make_sampler(sampler_name, sde, _zero_score, n_steps=4)
        x0 = s.sample(4, generator=torch.Generator().manual_seed(0))
        assert x0.shape == (4, *_IMAGE_SHAPE), sde_name
        assert torch.all(torch.isfinite(x0)), sde_name


@pytest.mark.parametrize("sampler_name", _ALL_SAMPLERS)
def test_sample_2d_still_flat_no_regression_via_factory(sampler_name):
    # 5.2: el caso 2D no cambia — la forma de evento (2,) sigue devolviendo (n, 2).
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")  # data_dim=2 por defecto -> data_shape=(2,)
    s = make_sampler(sampler_name, sde, _linear_score, n_steps=6)
    n = 8
    x0 = s.sample(n, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (n, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


# ================================================================================
# Task 4.2 (nd-shapes): generación checkpoint-driven con forma de imagen (round-trip).
#
# La matriz N-D de los samplers (Req 2.1–2.4) ya quedó cubierta por los tests de la
# Task 2 más arriba (test_sample_nd_shape_dtype_finite_via_factory,
# test_sample_nd_return_trajectory_shape_via_factory, test_sample_nd_finite_across_
# scalar_sdes, test_sample_2d_still_flat_no_regression_via_factory), parametrizados
# sobre la forma imagen-chica (3, 8, 8) por los cuatro samplers vía la factory. NO se
# duplican acá.
#
# Lo GENUINAMENTE nuevo de 4.2 es el *seam* de generación checkpoint-driven para
# imágenes (Req 4.1, 4.2, 4.3): construir un checkpoint (red SIN entrenar) con forma de
# evento tipo-imagen, generar vía generate_from_checkpoint y verificar la forma
# (n, *E) y el .npz de salida; y que una meta que no permita reconstruir la forma del
# dato falle con un error claro. La red image-capable es una ScoreUNet MINÚSCULA (8×8,
# pocos canales) para correr en CPU en fracciones de segundo; generate la reconstruye
# desde meta["model"] vía make_model, igual que el camino 2D reconstruye una ScoreMLP.
# ================================================================================

# Config chica de ScoreUNet coherente con _IMAGE_SHAPE=(3, 8, 8): image_size=8 (soporta
# el único Downsample de channel_mults=(1,2): 8 % 2 == 0), canales chicos, groups=8
# divide a los anchos 8 y 16, embed_dim par. Se reusa IDÉNTICA al construir el checkpoint
# y al reconstruir la red (make_model filtra por firma del constructor), para que el
# state_dict cargue sin discrepancias de arquitectura.
_UNET_KWARGS = {
    "in_channels": 3,
    "image_size": 8,
    "base_channels": 8,
    "channel_mults": (1, 2),
    "num_res_blocks": 1,
    "embed_dim": 8,
    "time_embed_dim": 16,
    "attn_resolutions": (),
    "groups": 8,
}


def _make_image_checkpoint(tmp_path, sde_name="vp"):
    """Checkpoint válido SIN entrenar con forma de evento imagen (3, 8, 8) -> ScoreUNet chica.

    Espejo de :func:`_make_checkpoint` pero para el camino de imágenes: ``TrainResult`` con
    ``data_dim=(3, 8, 8)`` + ``save_checkpoint`` con una ``model_spec`` de receta ``unet``, de
    modo que ``generate_from_checkpoint`` reconstruya la SDE con su forma de evento (vía
    ``make_sde(sde_name, data_dim=(3,8,8))``) y la red vía ``make_model("unet", **kwargs)``.
    No corre ningún loop de entrenamiento (la red arranca con pesos aleatorios). Devuelve la
    ruta del ``.pt`` guardado.
    """
    models = pytest.importorskip("diffusion.models")
    from diffusion.training import TrainConfig, TrainResult, save_checkpoint

    net = models.ScoreUNet(**_UNET_KWARGS)
    result = TrainResult(
        net=net,
        history=[1.0, 0.5],
        config=TrainConfig(),
        sde_name=sde_name,
        data_dim=_IMAGE_SHAPE,
    )
    path = tmp_path / "ckpt_image.pt"
    save_checkpoint(
        result, path, model_spec={"name": "unet", "kwargs": dict(_UNET_KWARGS)}
    )
    return path


def test_generate_from_checkpoint_image_shape_returns_samples(tmp_path):
    # 4.1, 4.2: desde un checkpoint cuya forma de evento es (3, 8, 8), generate reconstruye la
    # SDE con esa forma y la ScoreUNet vía make_model, y devuelve muestras (n, 3, 8, 8) float32
    # finito — no la forma plana 2D del camino MLP.
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_image_checkpoint(tmp_path, sde_name="vp")
    x0 = generate_from_checkpoint(ckpt, "pf_ode", n_samples=2, n_steps=3, seed=0)
    assert x0.shape == (2, *_IMAGE_SHAPE)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_generate_from_checkpoint_image_writes_npz_roundtrip(tmp_path):
    # 4.1, 4.2: round-trip end-to-end del seam de imágenes — checkpoint imagen -> generate con
    # `out` -> el .npz releído con np.load contiene `samples` de shape (n, 3, 8, 8) float32,
    # EXACTAMENTE el tensor devuelto. Cierra 4.2 (generación de imágenes desde checkpoint y
    # el archivo de salida). El subdir inexistente debe crearse.
    pytest.importorskip("numpy")
    import numpy as np
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_image_checkpoint(tmp_path, sde_name="vp")
    out = tmp_path / "gen_img" / "img.npz"  # subdir inexistente: debe crearse
    x0 = generate_from_checkpoint(
        ckpt, "pf_ode", n_samples=2, n_steps=3, seed=0, out=out
    )
    assert out.exists()
    assert x0.shape == (2, *_IMAGE_SHAPE)
    with np.load(out) as data:
        assert "samples" in data
        assert data["samples"].shape == (2, *_IMAGE_SHAPE)
        assert data["samples"].dtype == np.float32
        # El .npz persiste EXACTAMENTE el tensor devuelto (forma de imagen incluida).
        assert np.array_equal(data["samples"], x0.cpu().numpy())


def _make_image_checkpoint_blob_missing_meta_key(tmp_path, missing_key):
    """Blob imagen que SOBREVIVE ``load_checkpoint`` pero le falta ``missing_key`` de la meta.

    Espejo de :func:`_make_checkpoint_blob_missing_meta_key` para el camino de imágenes: el
    ``state_dict`` proviene de una ScoreUNet chica coherente con la receta ``unet`` y la meta
    trae ``sde_name``/``data_dim``/``model`` válidos salvo la clave bajo prueba. Aísla la guarda
    PROPIA de generate.py sobre la forma del dato: sin ``data_dim`` no hay con qué reconstruir la
    forma de evento de la SDE, y el error debe ser claro (no un ``KeyError("meta")`` opaco de
    load_checkpoint, ni un error críptico de make_sde).
    """
    models = pytest.importorskip("diffusion.models")

    net = models.ScoreUNet(**_UNET_KWARGS)
    meta = {
        "sde_name": "vp",
        "data_dim": _IMAGE_SHAPE,
        "model": {"name": "unet", "kwargs": dict(_UNET_KWARGS)},
        "history": [1.0, 0.5],
    }
    del meta[missing_key]  # quita SOLO la clave bajo prueba; el resto queda válido
    blob = {"model_state": net.state_dict(), "meta": meta}
    path = tmp_path / "incomplete_meta_image.pt"
    torch.save(blob, path)
    return path


def test_generate_from_checkpoint_image_meta_missing_data_dim_raises_clear_error(tmp_path):
    # 4.3: un checkpoint de imagen cuya meta NO trae la forma del dato (`data_dim`) no permite
    # reconstruir la forma de evento -> error claro que identifica las claves esperadas por
    # generate.py ('sde_name', 'data_dim'), en vez de un error opaco de make_sde. Discrimina de
    # _invalid_meta_raises (que ni siquiera tiene la clave `meta`).
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_image_checkpoint_blob_missing_meta_key(tmp_path, "data_dim")
    with pytest.raises(KeyError) as excinfo:
        generate_from_checkpoint(ckpt, "pf_ode", n_samples=2, n_steps=3)
    msg = str(excinfo.value)
    assert "sde_name" in msg
    assert "data_dim" in msg


# --------------------------------------------------------------- device del sampleo


def test_sample_device_param_cpu_equivale_a_default():
    """El `device` nuevo es inerte en CPU: `device=None` y `device="cpu"` dan lo mismo (mismo seed)."""
    from diffusion.samplers import make_sampler

    sde = make_sde("vp", data_dim=2)

    def run(dev):
        g = torch.Generator().manual_seed(0)
        s = make_sampler("euler", sde, _zero_score, n_steps=10)
        return s.sample(B, generator=g, device=dev)

    assert torch.equal(run(None), run("cpu"))


def test_sample_pasa_t_en_el_device_de_x():
    """La grilla temporal se mueve al device de x: el score recibe `t` y `x` en el MISMO device.

    En CPU la invariante es trivial, pero guarda contra una regresión que cree `t` en un device
    fijo (el bug que hacía imposible samplear en GPU: `x` en cuda, `t` en cpu).
    """
    from diffusion.samplers import make_sampler

    sde = make_sde("vp", data_dim=2)
    vistos: list[tuple] = []

    def spy(x, t):
        vistos.append((x.device, t.device))
        return torch.zeros_like(x)

    make_sampler("euler", sde, spy, n_steps=5).sample(B, generator=torch.Generator().manual_seed(0))

    assert vistos and all(xd == td for xd, td in vistos)


# --------------------------------------------------------------- denoise final (Tweedie)


def test_sample_denoise_true_is_default_and_matches_tweedie():
    """`denoise=True` (default) aplica el paso final de Tweedie ``E[x_0|x_{t_eps}]``;
    `denoise=False` devuelve el estado crudo en ``t_eps``. Verifica la fórmula exacta."""
    from diffusion.samplers import make_sampler

    sde = make_sde("vp", data_dim=2)
    n = 64
    init = torch.randn(n, 2, generator=torch.Generator().manual_seed(0))
    s = make_sampler("pf_ode", sde, _linear_score, n_steps=50)  # determinístico dado init

    raw = s.sample(n, init=init, denoise=False)
    default = s.sample(n, init=init)  # denoise=True por default
    den = s.sample(n, init=init, denoise=True)

    assert raw.shape == den.shape == (n, 2)
    assert torch.all(torch.isfinite(den))
    assert torch.equal(default, den)  # el default ES denoise=True
    assert not torch.equal(raw, den)  # el denoise cambió la salida (quitó el último ruido)

    # Fórmula exacta de Tweedie contra el estado crudo en t_eps: (x + σ_t² s) / α_t.
    t_eps_b = torch.full((n, 1), s.t_eps)
    alpha, std = sde.marginal_prob(torch.ones(n, 2), t_eps_b)
    expected = (raw + std ** 2 * _linear_score(raw, t_eps_b)) / alpha
    assert torch.allclose(den, expected, atol=1e-6)


def test_sample_denoise_keeps_trajectory_last_equals_x0():
    """Con `return_trajectory`, la última capa sigue siendo el ``x_0`` devuelto (post-denoise)."""
    from diffusion.samplers import make_sampler

    sde = make_sde("vp", data_dim=2)
    n, n_steps = 8, 20
    s = make_sampler("euler", sde, _zero_score, n_steps=n_steps)
    gen = torch.Generator().manual_seed(0)
    x0, traj = s.sample(n, generator=gen, return_trajectory=True)

    assert traj.shape == (n_steps + 1, n, 2)  # el denoise no agrega una capa extra
    assert torch.equal(traj[-1], x0)           # invariante preservado bajo denoise


def test_sample_denoise_ve_uses_unit_alpha():
    """Para VE (``α_t = 1``) el denoise es ``x + σ_t² s`` (sin dividir)."""
    from diffusion.samplers import make_sampler

    sde = make_sde("ve", data_dim=2)
    n = 32
    init = torch.randn(n, 2, generator=torch.Generator().manual_seed(0))
    s = make_sampler("pf_ode", sde, _linear_score, n_steps=50)

    raw = s.sample(n, init=init, denoise=False)
    den = s.sample(n, init=init, denoise=True)

    t_eps_b = torch.full((n, 1), s.t_eps)
    _, std = sde.marginal_prob(torch.ones(n, 2), t_eps_b)
    expected = raw + std ** 2 * _linear_score(raw, t_eps_b)  # α_t = 1 en VE
    assert torch.allclose(den, expected, atol=1e-6)


# --------------------------------------------------- grilla temporal configurable (time_grid)
#
# La distribución de los tiempos de integración es una elección numérica del Eje 2 (no
# reentrena). El default `uniform` debe quedar byte-idéntico a la fórmula previa, y `logsnr`
# se valida contra la forma cerrada de VP: λ(t) = -log(expm1(B(t))) con
# B(t) = β_min t + ½(β_max - β_min) t², invertida analíticamente.

from diffusion.samplers import available_time_grids, make_time_grid


def _vp_logsnr_grid_closed_form(n_steps, t_min, t_max, b_min=0.1, b_max=20.0):
    """Grilla λ-uniforme de VP por forma cerrada (referencia independiente del código)."""
    import numpy as np

    b_int = lambda t: b_min * t + 0.5 * (b_max - b_min) * t ** 2
    lam = lambda t: -np.log(np.expm1(b_int(t)))
    t_of = lambda l: (
        -b_min + np.sqrt(b_min ** 2 + 2 * (b_max - b_min) * np.logaddexp(0.0, -l))
    ) / (b_max - b_min)
    lams = np.linspace(lam(t_max), lam(t_min), n_steps + 1)
    return torch.from_numpy(t_of(lams)).float()


def test_available_time_grids_lists_registered_names():
    assert available_time_grids() == ["logsnr", "uniform"]


@pytest.mark.parametrize("n_steps", [1, 7, 500])
def test_uniform_grid_is_byte_identical_to_linspace(n_steps):
    """Retrocompatibilidad: el default reproduce exactamente la grilla previa al cambio."""
    sde = make_sde("vp")
    s = _NoOpSampler(sde, _zero_score, n_steps=n_steps, t_eps=1e-3)
    previous = torch.linspace(sde.T, 1e-3, n_steps + 1, dtype=torch.float32)
    assert torch.equal(s._time_grid(), previous)


def test_uniform_is_the_default_time_grid():
    sde = make_sde("vp")
    default = _NoOpSampler(sde, _zero_score, n_steps=16)
    explicit = _NoOpSampler(sde, _zero_score, n_steps=16, time_grid="uniform")
    assert torch.equal(default._time_grid(), explicit._time_grid())


@pytest.mark.parametrize("n_steps", [10, 100])
def test_logsnr_grid_matches_vp_closed_form(n_steps):
    """`logsnr` (bisección sobre marginal_prob) == la inversa analítica de λ en VP."""
    sde = make_sde("vp")
    s = _NoOpSampler(sde, _zero_score, n_steps=n_steps, t_eps=1e-3, time_grid="logsnr")
    expected = _vp_logsnr_grid_closed_form(n_steps, 1e-3, sde.T)
    assert torch.allclose(s._time_grid(), expected, atol=1e-4)


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_logsnr_grid_contract_per_sde(sde_name):
    """Extremos exactos, largo y monotonía estricta para las 3 SDEs del Eje 1."""
    sde = make_sde(sde_name)
    n_steps, t_eps = 32, 1e-3
    grid = _NoOpSampler(
        sde, _zero_score, n_steps=n_steps, t_eps=t_eps, time_grid="logsnr"
    )._time_grid()
    assert grid.shape == (n_steps + 1,)
    assert grid.dtype == torch.float32
    assert float(grid[0]) == pytest.approx(sde.T)
    assert float(grid[-1]) == pytest.approx(t_eps)
    assert bool((grid[1:] < grid[:-1]).all())


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_logsnr_grid_is_equispaced_in_log_snr(sde_name):
    """El invariante que define la grilla: λ queda equiespaciado."""
    sde = make_sde(sde_name)
    tg = make_time_grid("logsnr", sde, 1e-3)
    lam = tg._log_snr(tg.grid(50))
    deltas = lam[1:] - lam[:-1]
    # Coeficiente de variación ~0: todos los saltos de λ son iguales (límite float32).
    assert float(deltas.std() / deltas.mean().abs()) < 1e-2


def test_logsnr_equals_uniform_for_ve():
    """En VE, σ_t geométrico ⟹ λ lineal en t ⟹ λ-uniforme coincide con t-uniforme."""
    sde = make_sde("ve")
    n_steps = 40
    logsnr = _NoOpSampler(
        sde, _zero_score, n_steps=n_steps, t_eps=1e-3, time_grid="logsnr"
    )._time_grid()
    uniform = _NoOpSampler(
        sde, _zero_score, n_steps=n_steps, t_eps=1e-3, time_grid="uniform"
    )._time_grid()
    assert torch.allclose(logsnr, uniform, atol=1e-4)


@pytest.mark.parametrize("sde_name", ["vp", "sub_vp"])
def test_logsnr_differs_from_uniform_for_vp_family(sde_name):
    """En VP/sub-VP sí cambia la grilla (y concentra pasos en t chico)."""
    sde = make_sde(sde_name)
    logsnr = _NoOpSampler(
        sde, _zero_score, n_steps=40, t_eps=1e-3, time_grid="logsnr"
    )._time_grid()
    uniform = _NoOpSampler(
        sde, _zero_score, n_steps=40, t_eps=1e-3, time_grid="uniform"
    )._time_grid()
    assert not torch.allclose(logsnr, uniform, atol=1e-3)
    # Más de la mitad de los pasos por debajo del punto medio de la grilla uniforme.
    assert int((logsnr < 0.5).sum()) > int((uniform < 0.5).sum())


def test_callable_time_grid_is_accepted_and_used():
    """El escape hatch: un callable (n_steps, t_min, t_max) propio gobierna la grilla."""

    def quadratic(n_steps, t_min, t_max):
        u = torch.linspace(0.0, 1.0, n_steps + 1)
        return t_max + (t_min - t_max) * u ** 2

    sde = make_sde("vp")
    s = _NoOpSampler(sde, _zero_score, n_steps=12, t_eps=1e-3, time_grid=quadratic)
    grid = s._time_grid()
    assert torch.allclose(grid, quadratic(12, 1e-3, sde.T), atol=1e-6)
    assert not torch.allclose(grid, torch.linspace(sde.T, 1e-3, 13), atol=1e-3)


def test_callable_time_grid_accepts_numpy_output():
    np = pytest.importorskip("numpy")

    def np_grid(n_steps, t_min, t_max):
        return np.linspace(t_max, t_min, n_steps + 1)

    sde = make_sde("vp")
    grid = _NoOpSampler(
        sde, _zero_score, n_steps=6, t_eps=1e-3, time_grid=np_grid
    )._time_grid()
    assert grid.dtype == torch.float32
    assert grid.shape == (7,)


def test_unknown_time_grid_name_raises_listing_options():
    sde = make_sde("vp")
    with pytest.raises(ValueError, match="Grilla temporal desconocida"):
        _NoOpSampler(sde, _zero_score, time_grid="karras")


def test_time_grid_rejects_non_str_non_callable():
    sde = make_sde("vp")
    with pytest.raises(ValueError, match="nombre de una grilla registrada o un callable"):
        _NoOpSampler(sde, _zero_score, time_grid=123)


@pytest.mark.parametrize(
    "bad_fn, match",
    [
        (lambda n, a, b: torch.linspace(b, a, n + 5), r"n_steps \+ 1"),
        (lambda n, a, b: torch.linspace(a, b, n + 1), "decreciente"),
        (lambda n, a, b: torch.full((n + 1,), float("nan")), "no finitos"),
        (lambda n, a, b: torch.linspace(0.5, a, n + 1), "de T="),
    ],
)
def test_callable_time_grid_validates_output(bad_fn, match):
    """Una grilla mal formada falla con mensaje claro, no rompe la integración."""
    sde = make_sde("vp")
    s = _NoOpSampler(sde, _zero_score, n_steps=5, t_eps=1e-3, time_grid=bad_fn)
    with pytest.raises(ValueError, match=match):
        s._time_grid()


@pytest.mark.parametrize("sampler_name", ["euler", "pf_ode", "heun", "pc"])
def test_all_samplers_accept_time_grid_via_factory(sampler_name):
    """make_sampler propaga time_grid a los 4 samplers (incl. `pc`, que define su __init__)."""
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler(
        sampler_name, sde, _linear_score, n_steps=8, t_eps=1e-3, time_grid="logsnr"
    )
    assert s.time_grid.name == "logsnr"
    expected = _vp_logsnr_grid_closed_form(8, 1e-3, sde.T)
    assert torch.allclose(s._time_grid(), expected, atol=1e-4)
    x0 = s.sample(4, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (4, 2)
    assert bool(torch.isfinite(x0).all())


def test_time_grid_changes_samples_but_not_shape():
    """Cambiar la grilla cambia el resultado numérico sin tocar el contrato de salida."""
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    init = sde.prior_sampling((8, 2), generator=torch.Generator().manual_seed(0))
    a = make_sampler("pf_ode", sde, _linear_score, n_steps=10).sample(8, init=init)
    b = make_sampler(
        "pf_ode", sde, _linear_score, n_steps=10, time_grid="logsnr"
    ).sample(8, init=init)
    assert a.shape == b.shape == (8, 2)
    assert not torch.equal(a, b)


def test_logsnr_grid_works_on_image_event_shape():
    """La grilla es independiente de la geometría del dato: anda en (B, C, H, W)."""
    from diffusion.samplers import make_sampler

    sde = make_sde("vp", data_dim=(3, 8, 8))
    s = make_sampler("heun", sde, _linear_score, n_steps=6, time_grid="logsnr")
    x0 = s.sample(2, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (2, 3, 8, 8)
    assert bool(torch.isfinite(x0).all())
