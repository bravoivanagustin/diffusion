"""Tests de los procesos forward (`diffusion.sde`)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.sde import (
    ForwardSDE,
    SubVPSDE,
    VESDE,
    VPSDE,
    available_sdes,
    make_sde,
)

SCALAR = ["vp", "ve", "sub_vp"]
B = 16


def _x0_t(n=B, dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, dim, generator=g)
    t = torch.rand(n, generator=g)
    return x0, t


# --------------------------------------------------------------- registry / factory


def test_available_sdes():
    assert set(available_sdes()) == {"vp", "ve", "sub_vp"}


def test_factory_returns_right_type():
    assert isinstance(make_sde("vp"), VPSDE)
    assert isinstance(make_sde("ve"), VESDE)
    assert isinstance(make_sde("sub_vp"), SubVPSDE)


def test_unknown_sde_raises():
    with pytest.raises(ValueError):
        make_sde("no_existe")


def test_factory_filters_kwargs():
    # beta_min no aplica a VE y se descarta; sigma_max sí se aplica.
    sde = make_sde("ve", beta_min=0.1, sigma_max=3.0)
    assert isinstance(sde, VESDE)
    assert sde.sigma_max == 3.0


# ---------------------------------------------------------------- shapes / dtype


@pytest.mark.parametrize("name", SCALAR)
def test_marginal_prob_shapes_dtype(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    mean, std = sde.marginal_prob(x0, t)
    assert mean.shape == (B, 2)
    assert std.shape == (B, 1)
    assert mean.dtype == torch.float32 and std.dtype == torch.float32
    assert torch.all(torch.isfinite(mean)) and torch.all(torch.isfinite(std))


@pytest.mark.parametrize("name", SCALAR)
def test_perturb_shapes_dtype(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    x_t, eps = sde.perturb(x0, t)
    assert x_t.shape == (B, 2) and eps.shape == (B, 2)
    assert x_t.dtype == torch.float32 and eps.dtype == torch.float32


@pytest.mark.parametrize("name", SCALAR)
def test_sde_drift_diffusion_shapes(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    drift, diffusion = sde.sde(x0, t)
    assert drift.shape == (B, 2) and diffusion.shape == (B, 1)
    if name == "ve":
        assert torch.all(drift == 0.0)  # VE no tiene drift


@pytest.mark.parametrize("name", SCALAR)
def test_accepts_t_as_B_and_B1(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    m1, s1 = sde.marginal_prob(x0, t)
    m2, s2 = sde.marginal_prob(x0, t.reshape(B, 1))
    assert torch.equal(m1, m2) and torch.equal(s1, s2)


# ----------------------------------------------------------------- determinismo


@pytest.mark.parametrize("name", SCALAR)
def test_perturb_deterministic_with_generator(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    a, _ = sde.perturb(x0, t, generator=torch.Generator().manual_seed(7))
    b, _ = sde.perturb(x0, t, generator=torch.Generator().manual_seed(7))
    c, _ = sde.perturb(x0, t, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ------------------------------------------------------- límites / forma cerrada


def test_vp_small_t():
    sde = VPSDE()
    x0 = torch.randn(B, 2)
    t = torch.full((B,), 1e-4)
    mean, std = sde.marginal_prob(x0, t)
    assert torch.allclose(mean, x0, atol=1e-2)
    assert torch.all(std < 1e-2)


def test_vp_t_equals_T():
    sde = VPSDE()
    x0 = torch.randn(B, 2)
    t = torch.full((B,), 1.0)
    mean, std = sde.marginal_prob(x0, t)
    assert torch.all(mean.abs() < 0.1)          # mean -> 0
    assert torch.allclose(std, torch.ones_like(std), atol=1e-3)  # std -> 1


def test_ve_no_drift_and_sigma():
    sde = VESDE(sigma_min=0.01, sigma_max=5.0)
    x0 = torch.randn(B, 2)
    t = torch.rand(B)
    mean, std = sde.marginal_prob(x0, t)
    assert torch.equal(mean, x0)                # sin drift: mean == x0 exacto
    expected = 0.01 * (5.0 / 0.01) ** t.reshape(B, 1)
    assert torch.allclose(std, expected)
    # std(T) ~ sigma_max
    _, std_T = sde.marginal_prob(x0, torch.ones(B))
    assert torch.allclose(std_T, torch.full_like(std_T, 5.0))


def test_subvp_std_below_vp_same_mean():
    vp, sub = VPSDE(), SubVPSDE()
    x0 = torch.randn(B, 2)
    t = torch.rand(B) * 0.9 + 0.05  # 0 < t <= T
    m_vp, s_vp = vp.marginal_prob(x0, t)
    m_sub, s_sub = sub.marginal_prob(x0, t)
    assert torch.allclose(m_vp, m_sub)          # mismo alpha_t -> misma media
    assert torch.all(s_sub < s_vp)              # varianza estrictamente por debajo


@pytest.mark.parametrize("name", SCALAR)
def test_variance_ode_consistency(name):
    # Chequeo de cálculo: dSigma/dt debe coincidir con 2 f_coef Sigma + g^2,
    # donde el drift es f_coef * x (lineal) y g = diffusion. Diferencias finitas.
    sde = make_sde(name)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
    x = torch.ones(t.shape[0], 2)
    drift, diffusion = sde.sde(x, t)
    f_coef = drift[:, :1]                        # f_coef * 1 = drift  (x == 1)
    g2 = diffusion ** 2

    def var(tt):
        return sde.marginal_prob(x, tt)[1] ** 2

    h = 1e-3
    dvar = (var(t + h) - var(t - h)) / (2 * h)
    rhs = 2 * f_coef * var(t) + g2
    assert torch.allclose(dvar, rhs, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------- score target


@pytest.mark.parametrize("name", SCALAR)
def test_score_target(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    x_t, eps = sde.perturb(x0, t, generator=torch.Generator().manual_seed(3))
    score_real, weight = sde.score_target(x0, t, eps)
    _, std = sde.marginal_prob(x0, t)
    assert torch.allclose(score_real, -eps / std, atol=1e-5)   # -eps/sigma_t
    assert torch.allclose(weight, std ** 2, atol=1e-5)          # lambda(t) = sigma_t^2
    nz = eps.abs() > 1e-3
    assert torch.equal(torch.sign(score_real[nz]), -torch.sign(eps[nz]))


# -------------------------------------------------------------- prior_sampling


@pytest.mark.parametrize("name", SCALAR)
def test_prior_sampling_shape(name):
    sde = make_sde(name)
    z = sde.prior_sampling((10, 2), generator=torch.Generator().manual_seed(0))
    assert z.shape == (10, 2) and z.dtype == torch.float32


def test_prior_variance():
    g = torch.Generator().manual_seed(0)
    n = 40000
    for name in ("vp", "sub_vp"):
        z = make_sde(name).prior_sampling((n, 2), generator=g)
        assert abs(z.var().item() - 1.0) < 0.1
    ve = make_sde("ve", sigma_max=5.0)
    z = ve.prior_sampling((n, 2), generator=g)
    assert abs(z.var().item() - 25.0) < 25.0 * 0.1  # ~ sigma_max^2


# ------------------------------------------------------------ seam sde x mlp


@pytest.mark.parametrize("name", SCALAR)
def test_seam_with_scoremlp(name):
    from diffusion.models import ScoreMLP

    sde = make_sde(name)
    net = ScoreMLP(data_dim=2)
    x0, t = _x0_t()
    x_t, eps = sde.perturb(x0, t, generator=torch.Generator().manual_seed(1))
    pred = net(x_t, t)
    target, _ = sde.score_target(x0, t, eps)
    assert pred.shape == target.shape == (B, 2)
    assert torch.all(torch.isfinite(pred))


# ============================================================ dimensión arbitraria


def test_data_dim_must_be_positive():
    with pytest.raises(ValueError):
        make_sde("vp", data_dim=0)


def test_factory_passes_data_dim():
    assert make_sde("vp", data_dim=7).data_dim == 7


@pytest.mark.parametrize("name", SCALAR)
@pytest.mark.parametrize("dim", [1, 3, 7])
def test_scalar_arbitrary_dim(name, dim):
    sde = make_sde(name, data_dim=dim)
    assert sde.data_dim == dim
    x0, t = torch.randn(B, dim), torch.rand(B)
    x_t, eps = sde.perturb(x0, t, generator=torch.Generator().manual_seed(0))
    assert x_t.shape == (B, dim) and eps.shape == (B, dim)
    mean, std = sde.marginal_prob(x0, t)
    assert mean.shape == (B, dim) and std.shape == (B, 1)
    score, weight = sde.score_target(x0, t, eps)
    assert score.shape == (B, dim) and weight.shape == (B, 1)
    drift, diffusion = sde.sde(x0, t)
    assert drift.shape == (B, dim) and diffusion.shape == (B, 1)
    assert sde.prior_sampling((5, dim)).shape == (5, dim)


# ============================================ event shapes N-D (nd-shapes, task 1)

IMG = (3, 8, 8)  # forma tipo-imagen chica (rápida en CPU)


def test_expand_t_rank_aware():
    # (B,) o (B,1) -> (B, 1, ..., 1) con ref.ndim-1 unos.
    t = torch.rand(B)
    x2 = torch.randn(B, 2)
    ex2 = ForwardSDE._expand_t(t, x2)
    assert ex2.shape == (B, 1)
    assert torch.equal(ex2, t.reshape(-1, 1))  # byte-idéntico al comportamiento 2D previo
    ximg = torch.randn(B, *IMG)
    assert ForwardSDE._expand_t(t, ximg).shape == (B, 1, 1, 1)
    # t ya como (B,1) da lo mismo que (B,)
    assert torch.equal(
        ForwardSDE._expand_t(t.reshape(B, 1), ximg), ForwardSDE._expand_t(t, ximg)
    )


@pytest.mark.parametrize("name", SCALAR)
def test_construct_with_int_and_tuple_exposes_data_shape(name):
    sde_int = make_sde(name, data_dim=2)
    assert sde_int.data_shape == (2,)
    assert sde_int.data_dim == 2  # crudo (backward-compat: meta + MLP 2D)
    sde_flat = make_sde(name, data_dim=5)
    assert sde_flat.data_shape == (5,) and sde_flat.data_dim == 5
    sde_img = make_sde(name, data_dim=IMG)
    assert sde_img.data_shape == IMG
    assert sde_img.data_dim == IMG  # tupla cruda conservada


@pytest.mark.parametrize("bad", [0, -1, (), (3, 0, 8), (0,), (3, -1, 8)])
def test_invalid_shape_raises(bad):
    with pytest.raises(ValueError):
        make_sde("vp", data_dim=bad)


# --------------------------- matriz consolidada VP/VE/sub-VP × event shape (task 4.1)
#
# Cross-product de la familia escalar sobre una forma 2D `(2,)` y una forma tipo-imagen
# chica `(3, 8, 8)`: las cuatro operaciones (`perturb`/`score_target`/`sde`/
# `marginal_prob`) devuelven/broadcastean sobre `(B, *E)` en float32 y finito. Consolida
# los chequeos de shape/dtype/finitud que antes vivían en tests N-D separados por forma.
# `B=16` es distinto de las dims espaciales `(3, 8, 8)`, así que un coeficiente colapsado
# `(B, 1)` NO broadcastearía contra `(B, 3, 8, 8)` (16 vs 8) -> el chequeo es
# discriminante frente a una regresión de coeficiente rank-2, no cosmético.

EVENT_SHAPES = [(2,), IMG]


def _x0_t_event(event, n=B, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, *event, generator=g)
    t = torch.rand(n, generator=g)
    return x0, t


@pytest.mark.parametrize("name", SCALAR)
@pytest.mark.parametrize("event", EVENT_SHAPES)
def test_scalar_family_matrix_shapes_dtype_finite(name, event):
    # Construcción con int para el dato plano `(2,)` y con tupla para la imagen.
    data_dim = event[0] if len(event) == 1 else event
    sde = make_sde(name, data_dim=data_dim)
    assert sde.data_shape == event
    x0, t = _x0_t_event(event)
    expected = (B, *event)

    # marginal_prob: media `(B, *E)` + std por-muestra que broadcastea contra el estado.
    mean, std = sde.marginal_prob(x0, t)
    assert mean.shape == expected and std.shape[0] == B
    assert (mean + std).shape == expected
    assert mean.dtype == torch.float32 and std.dtype == torch.float32
    assert torch.all(torch.isfinite(mean)) and torch.all(torch.isfinite(std))

    # perturb: `(x_t, eps)` ambos `(B, *E)` float32 y finitos.
    x_t, eps = sde.perturb(x0, t, generator=torch.Generator().manual_seed(0))
    assert x_t.shape == expected and eps.shape == expected
    assert x_t.dtype == torch.float32 and eps.dtype == torch.float32
    assert torch.all(torch.isfinite(x_t)) and torch.all(torch.isfinite(eps))

    # score_target: score `(B, *E)` + peso por-muestra que broadcastea (rank-matched).
    score, weight = sde.score_target(x0, t, eps)
    assert score.shape == expected and weight.shape[0] == B
    assert (score * weight).shape == expected
    assert score.dtype == torch.float32
    assert torch.all(torch.isfinite(score)) and torch.all(torch.isfinite(weight))

    # sde: drift `(B, *E)` + diffusion por-muestra que broadcastea.
    drift, diffusion = sde.sde(x0, t)
    assert drift.shape == expected and diffusion.shape[0] == B
    assert (drift + diffusion).shape == expected
    assert drift.dtype == torch.float32 and diffusion.dtype == torch.float32
    assert torch.all(torch.isfinite(drift)) and torch.all(torch.isfinite(diffusion))


@pytest.mark.parametrize("name", SCALAR)
def test_nd_t_as_B_and_B1_equal(name):
    sde = make_sde(name, data_dim=IMG)
    x0 = torch.randn(B, *IMG)
    t = torch.rand(B)
    m1, s1 = sde.marginal_prob(x0, t)
    m2, s2 = sde.marginal_prob(x0, t.reshape(B, 1))
    assert torch.equal(m1, m2) and torch.equal(s1, s2)
    d1, g1 = sde.sde(x0, t)
    d2, g2 = sde.sde(x0, t.reshape(B, 1))
    assert torch.equal(d1, d2) and torch.equal(g1, g2)


@pytest.mark.parametrize("name", SCALAR)
def test_2d_invariance_after_generalization(name):
    # La generalización N-D no cambia números ni shapes en 2D (mismo seed).
    sde = make_sde(name)
    x0, t = _x0_t()
    mean, std = sde.marginal_prob(x0, t)
    assert mean.shape == (B, 2) and std.shape == (B, 1)
    x_t, eps = sde.perturb(x0, t, generator=torch.Generator().manual_seed(42))
    assert torch.allclose(x_t, mean + std * eps)  # reconstrucción escalar-gaussiana
    drift, diffusion = sde.sde(x0, t)
    assert drift.shape == (B, 2) and diffusion.shape == (B, 1)
