"""Tests de la feature `training-resume`.

Task 1 — **Persistencia del estado de resume (fundación)**: define el estado de resume
(:class:`ResumeState`), su envoltorio (:class:`TrainSnapshot`) y el I/O del *sidecar*
(:func:`save_resume_state` / :func:`load_resume_state`), separado del checkpoint de pesos.

Torch es dependencia dura del módulo (igual que en `test_training.py`), así que se hace
`importorskip` al tope. Se usan redes chicas para correr en CPU en segundos.
"""

from __future__ import annotations

import contextlib
import copy
import io
import pathlib
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")

from diffusion.data_generation import infinite_bare, make_distribution
from diffusion.models import ScoreMLP
from diffusion.sde import make_sde
from diffusion.training import (
    ResumePlan,
    ResumeState,
    TrainConfig,
    TrainResult,
    TrainSnapshot,
    discover_snapshots,
    load_checkpoint,
    load_resume,
    load_resume_state,
    prune_snapshots,
    resolve_resume,
    resume_sidecar_path,
    save_checkpoint,
    save_resume_state,
    train,
    validate_compatible,
)

SIDECAR_KEYS = {"optimizer_state", "step", "torch_rng_state", "generator_state"}


def _small_net(sde) -> ScoreMLP:
    return ScoreMLP(data_dim=sde.data_dim, hidden_dim=64, num_blocks=2)


def _data(dist, n=256, batch_size=64, *, shuffle=True):
    """Fuente infinita de tensores crudos que consume ``train`` (loader finito envuelto)."""
    return infinite_bare(dist.dataloader(n, batch_size, shuffle=shuffle))


def _net_with_optimizer_state():
    """``ScoreMLP`` + ``Adam`` que ya dio un paso (el optimizador tiene estado de momentos).

    Un round-trip trivial (optimizador recién creado) tendría ``state == {}`` y no probaría que
    los momentos de Adam sobreviven la serialización. Se fuerza un paso para poblarlo.
    """
    torch.manual_seed(0)
    net = ScoreMLP(data_dim=2, hidden_dim=16, num_blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(8, 2)
    t = torch.rand(8)
    loss = net(x, t).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return net, opt


def _resume_state(opt, *, start_step=5, history=None) -> ResumeState:
    return ResumeState(
        optimizer_state=opt.state_dict(),
        start_step=start_step,
        torch_rng_state=torch.get_rng_state(),
        generator_state=torch.Generator().manual_seed(123).get_state(),
        history=list(history if history is not None else [1.0, 0.5, 0.25]),
    )


# ------------------------------------------------------------ round-trip (1.1)


def test_save_load_resume_state_roundtrip(tmp_path):
    """Ida-y-vuelta del sidecar: preserva optimizador, paso y ambos estados de azar (1.1).

    El ``optimizer_state`` vuelve a cargar en un ``Adam`` fresco (mismos params) y los momentos
    coinciden; el ``step`` coincide; los tensores de RNG son ``torch.equal`` a los originales.
    """
    net, opt = _net_with_optimizer_state()
    torch_rng = torch.get_rng_state()
    gen_state = torch.Generator().manual_seed(123).get_state()

    resume = ResumeState(
        optimizer_state=opt.state_dict(),
        start_step=5,
        torch_rng_state=torch_rng,
        generator_state=gen_state,
        history=[1.0, 0.5, 0.25],
    )

    # Path con directorios intermedios inexistentes: save debe crearlos.
    path = tmp_path / "sub" / "vp_gaussian_step00005.resume.pt"
    out = save_resume_state(path, resume)
    assert out == path
    assert path.exists()

    loaded = load_resume_state(path)

    assert loaded["step"] == 5
    assert torch.equal(loaded["torch_rng_state"], torch_rng)
    assert torch.equal(loaded["generator_state"], gen_state)

    # El optimizer_state carga de vuelta en un Adam fresco y los momentos de Adam coinciden.
    fresh = torch.optim.Adam(net.parameters(), lr=1e-3)
    fresh.load_state_dict(loaded["optimizer_state"])
    orig, new = opt.state_dict()["state"], fresh.state_dict()["state"]
    assert orig.keys() == new.keys()
    assert orig  # el optimizador original tenía estado (no vacío) => la prueba es real
    for k in orig:
        assert torch.equal(orig[k]["exp_avg"], new[k]["exp_avg"])
        assert torch.equal(orig[k]["exp_avg_sq"], new[k]["exp_avg_sq"])


# ------------------------------------------------ history no se duplica (1.3)


def test_sidecar_no_incluye_history(tmp_path):
    """El sidecar persiste solo {optimizer_state, step, RNGs} — NO el ``history`` (1.3).

    El ``history`` ya vive en el ``meta`` del checkpoint de pesos; duplicarlo sería redundante.
    """
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=2, history=[1.0, 2.0])

    path = tmp_path / "ckpt.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert "history" not in loaded
    assert set(loaded) == SIDECAR_KEYS


# ---------------------------------------------------------- fail-fast (1.4)


@pytest.mark.parametrize(
    "field", ["optimizer_state", "start_step", "torch_rng_state", "generator_state"]
)
def test_save_resume_state_falla_si_incompleto(tmp_path, field):
    """Estado incompleto (cualquier campo requerido en ``None``) => ``ValueError`` (1.4).

    No se escribe un sidecar parcial: el archivo NO debe quedar en disco tras el fallo.
    """
    _, opt = _net_with_optimizer_state()
    kwargs = dict(
        optimizer_state=opt.state_dict(),
        start_step=1,
        torch_rng_state=torch.get_rng_state(),
        generator_state=torch.Generator().manual_seed(0).get_state(),
        history=[1.0],
    )
    kwargs[field] = None
    resume = ResumeState(**kwargs)

    path = tmp_path / "parcial.resume.pt"
    with pytest.raises(ValueError, match=field):
        save_resume_state(path, resume)
    assert not path.exists()  # no se persiste un artefacto parcial


# --------------------------------------- checkpoint de pesos intacto (1.2)


def test_sidecar_no_altera_el_checkpoint_de_pesos(tmp_path):
    """Guardar el sidecar no toca el checkpoint de pesos (mismo archivo, misma meta) (1.2)."""
    sde = make_sde("vp")
    net = ScoreMLP(data_dim=2, hidden_dim=16, num_blocks=1)
    result = TrainResult(net=net, history=[1.0, 2.0], sde_name=sde.name, data_dim=sde.data_dim)
    model_spec = {"name": "mlp", "kwargs": {"data_dim": 2, "hidden_dim": 16, "num_blocks": 1}}

    weights = tmp_path / "vp_gaussian_step00005.pt"
    save_checkpoint(result, weights, model_spec=model_spec)
    weights_bytes = weights.read_bytes()  # foto byte-a-byte antes del sidecar

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    resume = _resume_state(opt, start_step=5, history=[1.0, 2.0])
    sidecar = tmp_path / "vp_gaussian_step00005.resume.pt"
    save_resume_state(sidecar, resume)

    # El checkpoint de pesos quedó intacto byte-a-byte y sigue cargando con la misma meta.
    assert weights.read_bytes() == weights_bytes
    _, meta = load_checkpoint(weights)
    assert set(meta) == {"sde_name", "data_dim", "history", "model"}
    assert meta["sde_name"] == "vp"
    assert meta["history"] == pytest.approx([1.0, 2.0])


# ----------------------------------------- envoltorio TrainSnapshot (1.1)


def test_trainsnapshot_envuelve_result_y_resume():
    """``TrainSnapshot`` agrupa el ``TrainResult`` (pesos+history) y el ``ResumeState`` (sidecar).

    Es el envoltorio con que el estado de resume viaja junto a los pesos en cada checkpoint.
    """
    sde = make_sde("vp")
    net = ScoreMLP(data_dim=2, hidden_dim=8, num_blocks=1)
    result = TrainResult(net=net, history=[1.0], sde_name=sde.name, data_dim=sde.data_dim)
    opt = torch.optim.Adam(net.parameters())
    resume = _resume_state(opt, start_step=1, history=[1.0])

    snap = TrainSnapshot(result=result, resume=resume)

    assert snap.result is result
    assert snap.resume is resume


# =============================================================================
# Task 2.1 — Reanudación del loop de entrenamiento (`train(resume=...)`)
# =============================================================================


# --------------------------------------------- corre solo los restantes (2.2, 2.3)


def test_resume_corre_solo_los_pasos_restantes():
    """Reanudar desde el paso N con ``num_steps`` total corre solo los pasos restantes (2.2)
    y continúa el ``history`` previo hasta cubrir toda la corrida (2.3).

    Se corre una corrida fresca hasta 4 pasos con snapshot en el paso 2 y se **congela** ese
    snapshot (``deepcopy``): así los pesos y el estado del optimizador quedan fijos en el paso 2
    (las corridas siguientes del loop mutarían los tensores in-place). Reanudando desde ese
    snapshot con ``num_steps=4`` el loop itera ``range(2, 4)`` → 2 pasos nuevos, y el ``history``
    final mide 4 arrancando por los 2 previos.
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)

    frozen: dict[str, TrainSnapshot] = {}

    def capture(tag, snap):
        # deepcopy: congela pesos + optimizer_state en el paso del snapshot (el loop sigue
        # mutando esos tensores in-place después).
        frozen[tag] = copy.deepcopy(snap)

    net = _small_net(sde)
    train(
        sde,
        net,
        _data(dist),
        TrainConfig(num_steps=4, checkpoint_every=2, seed=0),
        on_checkpoint=capture,
    )

    snap = frozen["step00002"]
    assert snap.resume.start_step == 2
    assert len(snap.resume.history) == 2  # history del paso 2

    result = train(
        sde,
        snap.result.net,  # pesos congelados del paso 2
        _data(dist),
        TrainConfig(num_steps=4, seed=0),  # num_steps = TOTAL a alcanzar
        resume=snap.resume,
    )

    assert len(result.history) == 4  # 2 previos + 2 nuevos == num_steps total (2.3)
    assert len(result.history) - len(snap.resume.history) == 2  # corrió exactamente 2 pasos (2.2)
    assert result.history[:2] == snap.resume.history  # continuó el history previo, no lo reinició


# ----------------------------------------------------- no-op si ya completo (2.4)


@pytest.mark.parametrize("start_step", [4, 5, 10])
def test_resume_no_op_si_ya_completo(start_step):
    """Si el paso inicial ya alcanzó/superó ``num_steps``, no se corre ningún paso (2.4).

    El resultado devuelve el ``history`` previo sin cambios (ni un append más).
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    prior_history = [1.0, 0.5, 0.25, 0.2]

    resume = ResumeState(
        optimizer_state=opt.state_dict(),
        start_step=start_step,
        torch_rng_state=torch.get_rng_state(),
        generator_state=torch.Generator().manual_seed(0).get_state(),
        history=list(prior_history),
    )

    result = train(
        sde, net, _data(dist), TrainConfig(num_steps=4, seed=0), resume=resume
    )

    assert result.history == prior_history  # sin pasos nuevos: history intacto
    assert len(result.history) == len(prior_history)


# --------------------------------------- contrato del callback: TrainSnapshot (1.1)


def test_on_checkpoint_recibe_trainsnapshot_con_resume_state():
    """Con ``checkpoint_every>0`` el callback recibe un :class:`TrainSnapshot` cuyo ``.result`` es
    un :class:`TrainResult` y cuyo ``.resume`` es un :class:`ResumeState` con el optimizador
    poblado, ambos estados de azar y el paso actual (1.1).
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    calls: list[tuple[str, TrainSnapshot]] = []

    net = _small_net(sde)
    train(
        sde,
        net,
        _data(dist),
        TrainConfig(num_steps=9, checkpoint_every=3, seed=0),
        on_checkpoint=lambda tag, snap: calls.append((tag, snap)),
    )

    assert calls  # se emitieron snapshots
    for _tag, snap in calls:
        assert isinstance(snap, TrainSnapshot)
        assert isinstance(snap.result, TrainResult)
        assert isinstance(snap.resume, ResumeState)
        # optimizador poblado (Adam ya dio pasos → momentos presentes).
        assert snap.resume.optimizer_state["state"]
        # ambos estados de azar presentes como tensores.
        assert isinstance(snap.resume.torch_rng_state, torch.Tensor)
        assert isinstance(snap.resume.generator_state, torch.Tensor)

    # Los snapshots periódicos llevan como paso los pasos ya completados (= N del tag).
    periodic = {tag: snap for tag, snap in calls if tag.startswith("step")}
    assert periodic["step00003"].resume.start_step == 3
    assert periodic["step00006"].resume.start_step == 6
    assert len(periodic["step00003"].resume.history) == 3  # history hasta el paso 3


# ------------------------------------------------------------- gate: sin snapshots (1.5)


def test_sin_checkpoint_every_no_emite_snapshots():
    """Con ``checkpoint_every=0`` el callback NUNCA se invoca: no hay puntos de reanudación (1.5)."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    calls: list[str] = []

    train(
        sde,
        _small_net(sde),
        _data(dist),
        TrainConfig(num_steps=6, seed=0),  # checkpoint_every=0 por defecto
        on_checkpoint=lambda tag, snap: calls.append(tag),
    )

    assert calls == []


# ------------------------------------------------------------- regresión resume=None


def test_resume_none_entrena_normal():
    """Sin ``resume`` (default) el loop entrena desde cero como siempre: ``len(history)==num_steps``."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)

    result = train(sde, net, _data(dist), TrainConfig(num_steps=5, seed=0))

    assert len(result.history) == 5
    assert result.net is net
    assert result.sde_name == "vp"


# =============================================================================
# Task 2.2 — Equivalencia de la reanudación (gate de fidelidad, 2.6 / 2.1)
# =============================================================================
#
# Se remueve el confundidor del ORDEN DE DATOS con una fuente de **orden fijo**: un iterador
# infinito que yield-ea el MISMO batch en cada paso. Como ``train`` reconstruye ``iter(data)``
# por llamada y una corrida reanudada itera ``range(start_step, num_steps)`` sobre un iterador
# fresco, una fuente constante garantiza dato idéntico en cada paso tanto para la corrida
# ininterrumpida como para la reanudada. (Una fuente barajada/posicional divergiría — esa
# divergencia es la R2.6 relajada y aceptada; el test la controla a propósito para aislar y
# probar exactamente la restauración de optimizador + azar + paso.)

_N = 6  # paso del checkpoint intermedio
_TOTAL = 2 * _N  # total de la corrida (num_steps): el snapshot en _N cae ANTES del último paso


def _const_source(batch):
    """Fuente infinita de ORDEN FIJO: yield-ea el MISMO batch en cada paso (resume-invariante)."""
    while True:
        yield batch


def _fixed_batch(n=64, dim=2, seed=1234):
    """Batch fijo, con un ``Generator`` propio para NO tocar el RNG global de torch.

    El batch se crea antes de ``train``; usar un generador aparte evita perturbar el RNG global
    que la corrida ininterrumpida siembra (``config.seed``) y que la reanudada restaura.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=gen)


def _equiv_net() -> ScoreMLP:
    """Red chica y determinística, misma arquitectura para A / B / contraste."""
    return ScoreMLP(data_dim=2, hidden_dim=32, num_blocks=1)


def _weights_allclose(a, b) -> bool:
    """``True`` si TODO el ``state_dict`` de ``a`` y ``b`` es ``allclose`` (tolerancia tight, default)."""
    sda, sdb = a.state_dict(), b.state_dict()
    assert sda.keys() == sdb.keys()
    return all(torch.allclose(sda[k], sdb[k]) for k in sda)


def _run_uninterrupted():
    """Corrida A: entrena ininterrumpido hasta ``_TOTAL`` sobre la fuente de orden fijo.

    Devuelve ``(sde, batch, result_a, net_a, snap_N)`` donde ``snap_N`` es el
    :class:`TrainSnapshot` **congelado** (``deepcopy``) del paso ``_N``: el loop sigue mutando los
    pesos y el estado del optimizador in-place después del snapshot, así que hay que congelarlo
    para reanudar desde ese punto exacto.
    """
    sde = make_sde("vp")
    batch = _fixed_batch()
    frozen: dict[str, TrainSnapshot] = {}

    def capture(tag, snap):
        frozen[tag] = copy.deepcopy(snap)

    net_a = _equiv_net()
    result_a = train(
        sde,
        net_a,
        _const_source(batch),
        TrainConfig(num_steps=_TOTAL, checkpoint_every=_N, seed=0),
        on_checkpoint=capture,
    )
    return sde, batch, result_a, net_a, frozen[f"step{_N:05d}"]


def test_resume_equivalente_a_corrida_ininterrumpida():
    """Gate de fidelidad (2.6, 2.1): con orden fijo, reanudar equivale a no interrumpir.

    - A: corrida entera hasta ``_TOTAL`` con snapshot en el paso ``_N``.
    - B: red **fresca** con los pesos del paso ``_N`` cargados, reanudada con ``resume=snap.resume``
      hasta ``_TOTAL`` sobre la misma fuente de orden fijo.

    Removido el confundidor del orden de datos, restaurar optimizador + azar + paso hace que B
    reproduzca A: pesos ``allclose`` (tolerancia tight) e ``history`` idéntico paso a paso.
    """
    sde, batch, result_a, net_a, snap = _run_uninterrupted()

    # El snapshot es del paso _N: history de largo _N y start_step == _N; A cubrió _TOTAL.
    assert snap.resume.start_step == _N
    assert len(snap.resume.history) == _N
    assert len(result_a.history) == _TOTAL

    net_b = _equiv_net()
    net_b.load_state_dict(snap.result.net.state_dict())  # pesos congelados del paso _N
    result_b = train(
        sde,
        net_b,
        _const_source(batch),  # misma fuente de orden fijo
        TrainConfig(num_steps=_TOTAL, seed=0),  # num_steps = TOTAL a alcanzar
        resume=snap.resume,
    )

    # Equivalencia de la curva completa (2.3) y de los pesos finales (2.1, 2.6).
    assert result_b.history == result_a.history
    assert _weights_allclose(net_a, net_b)


def test_resume_sin_restaurar_optimizador_difiere():
    """Contraste (2.1): reanudar SIN restaurar el optimizador rompe la equivalencia.

    Idéntico a la reanudación fiel salvo que el ``optimizer_state`` se reemplaza por el de un
    ``Adam`` fresco (estado vacío = warm restart sin momentos). Con todo lo demás igual —pesos,
    azar, paso y datos—, los pesos finales YA NO son ``allclose`` a A: demuestra que la
    restauración del optimizador (los momentos de Adam) es lo que hace fiel a la reanudación.
    """
    sde, batch, result_a, net_a, snap = _run_uninterrupted()

    cfg = TrainConfig(num_steps=_TOTAL, seed=0)
    net_c = _equiv_net()
    net_c.load_state_dict(snap.result.net.state_dict())  # mismos pesos del paso _N que B

    # Optimizador SIN restaurar: estado vacío de un Adam fresco (mismo lr que la corrida, para que
    # la ÚNICA diferencia con la reanudación fiel sea la ausencia de momentos).
    fresh_opt = torch.optim.Adam(net_c.parameters(), lr=cfg.lr)
    no_opt_resume = ResumeState(
        optimizer_state=fresh_opt.state_dict(),  # estado vacío => warm restart, sin momentos
        start_step=snap.resume.start_step,
        torch_rng_state=snap.resume.torch_rng_state,
        generator_state=snap.resume.generator_state,
        history=list(snap.resume.history),
    )

    result_c = train(sde, net_c, _const_source(batch), cfg, resume=no_opt_resume)

    # Sin restaurar el optimizador los pesos finales difieren de A (la restauración importa)...
    assert not _weights_allclose(net_a, net_c)
    # ...y la curva diverge, aunque conserve el prefijo previo (el 1.er loss es pre-update, igual).
    assert result_c.history != result_a.history
    assert result_c.history[:_N] == result_a.history[:_N]


# =============================================================================
# Task 3.1 — Resolver de resume: descubrimiento + decisión skip/fresh/resume
# =============================================================================
#
# Lógica **pura** de rutas/decisión (sin torch, sin entrenar): se simulan los checkpoints con
# archivos ``.pt`` vacíos (``.touch()``) sobre ``tmp_path``. La convención de nombres es la del
# CLI: el checkpoint final es ``X.pt`` y los snapshots hermanos ``X_stepNNNNN.pt`` / ``X_best.pt``
# (más los sidecars ``X_stepNNNNN.resume.pt`` de la feature).

_STEM = "vp_gaussian"


def _final(tmp_path):
    """Ruta del checkpoint final (``X.pt``); el caller decide si existe (``.touch()``) o no."""
    return tmp_path / f"{_STEM}.pt"


def _touch(tmp_path, name):
    p = tmp_path / name
    p.touch()
    return p


# ----------------------------------------------- discover_snapshots (3.3)


def test_discover_snapshots_ordena_y_excluye(tmp_path):
    """Descubre los ``X_stepNNNNN.pt`` hermanos, ordenados ASC por paso (3.3), excluyendo el
    checkpoint final, el ``X_best.pt`` y los sidecars ``.resume.pt``; y sin colar snapshots de
    OTRA corrida (distinto stem) en el mismo directorio.
    """
    final = _final(tmp_path)
    final.touch()  # el final mismo NO es un snapshot
    s2 = _touch(tmp_path, f"{_STEM}_step00002.pt")
    s10 = _touch(tmp_path, f"{_STEM}_step00010.pt")
    _touch(tmp_path, f"{_STEM}_best.pt")  # best: excluido
    _touch(tmp_path, f"{_STEM}_step00010.resume.pt")  # sidecar: excluido
    _touch(tmp_path, "ve_gaussian_step00003.pt")  # otra corrida: excluida (distinto stem)

    snaps = discover_snapshots(final)

    assert [step for step, _ in snaps] == [2, 10]  # ascendente, sin best/sidecar/otra corrida
    assert snaps[0] == (2, s2)
    assert snaps[1] == (10, s10)


def test_discover_snapshots_vacio_sin_snapshots(tmp_path):
    """Sin snapshots hermanos (aunque el directorio o el final existan) → lista vacía."""
    final = _final(tmp_path)
    final.touch()
    assert discover_snapshots(final) == []


def test_discover_snapshots_directorio_inexistente(tmp_path):
    """Si el directorio del final no existe, no hay dónde buscar → lista vacía (sin excepción)."""
    final = tmp_path / "no_existe" / f"{_STEM}.pt"
    assert discover_snapshots(final) == []


# --------------------------------------------------------- skip (3.1)


def test_resolve_skip_si_final_existe(tmp_path):
    """Final presente y sin ``force`` → acción ``skip`` (corrida ya completa) (3.1)."""
    final = _final(tmp_path)
    final.touch()

    plan = resolve_resume(final)

    assert isinstance(plan, ResumePlan)
    assert plan.action == "skip"
    assert plan.weights_path is None
    assert plan.step is None


# --------------------------------------------------------- force (3.2)


def test_resolve_force_reanuda_desde_el_mas_nuevo(tmp_path):
    """Con ``force`` el chequeo del final se saltea: existiendo snapshots, reanuda desde el más
    nuevo en lugar de saltear (3.2)."""
    final = _final(tmp_path)
    final.touch()
    _touch(tmp_path, f"{_STEM}_step00005.pt")
    newest = _touch(tmp_path, f"{_STEM}_step00020.pt")

    plan = resolve_resume(final, force=True)

    assert plan.action == "resume"
    assert plan.step == 20
    assert plan.weights_path == newest


def test_resolve_force_sin_snapshots_es_fresh(tmp_path):
    """Con ``force`` pero sin snapshots (aunque el final exista) → ``fresh`` (reentrena de cero)."""
    final = _final(tmp_path)
    final.touch()

    plan = resolve_resume(final, force=True)

    assert plan.action == "fresh"


# ------------------------------------------------ auto-resume más nuevo (3.3)


def test_resolve_auto_resume_desde_el_mas_nuevo(tmp_path):
    """Final ausente + snapshots presentes → ``resume`` desde el de mayor paso (3.3)."""
    final = _final(tmp_path)  # NO existe
    _touch(tmp_path, f"{_STEM}_step00003.pt")
    _touch(tmp_path, f"{_STEM}_step00007.pt")
    newest = _touch(tmp_path, f"{_STEM}_step00030.pt")

    plan = resolve_resume(final)

    assert plan.action == "resume"
    assert plan.step == 30
    assert plan.weights_path == newest


# --------------------------------------------------------- fresh (3.4)


def test_resolve_fresh_si_no_hay_nada(tmp_path):
    """Final ausente y sin snapshots → ``fresh`` (desde cero) (3.4)."""
    plan = resolve_resume(_final(tmp_path))
    assert plan.action == "fresh"


def test_resolve_fresh_si_final_none():
    """``final_checkpoint=None`` → no hay dónde saltear ni buscar → ``fresh`` (3.4)."""
    plan = resolve_resume(None)
    assert plan.action == "fresh"


# ------------------------------------------------ --resume-from (3.5)


def test_resolve_resume_from_por_paso(tmp_path):
    """``--resume-from`` por número de paso elige ese snapshot puntual, aun con el final presente
    (el pedido explícito manda sobre el skip automático) (3.5)."""
    final = _final(tmp_path)
    final.touch()
    chosen = _touch(tmp_path, f"{_STEM}_step00005.pt")
    _touch(tmp_path, f"{_STEM}_step00010.pt")

    plan = resolve_resume(final, resume_from="5")

    assert plan.action == "resume"
    assert plan.step == 5
    assert plan.weights_path == chosen


def test_resolve_resume_from_por_ruta(tmp_path):
    """``--resume-from`` por ruta usa ese checkpoint y parsea su paso del nombre (3.5)."""
    final = _final(tmp_path)
    chosen = _touch(tmp_path, f"{_STEM}_step00010.pt")

    plan = resolve_resume(final, resume_from=str(chosen))

    assert plan.action == "resume"
    assert plan.weights_path == chosen
    assert plan.step == 10


# ------------------------------------- --resume-from inexistente (3.7)


def test_resolve_resume_from_paso_inexistente_lista_disponibles(tmp_path):
    """``--resume-from`` con un paso inexistente → ``ValueError`` que lista los pasos disponibles
    (3.7)."""
    final = _final(tmp_path)
    _touch(tmp_path, f"{_STEM}_step00002.pt")
    _touch(tmp_path, f"{_STEM}_step00010.pt")

    with pytest.raises(ValueError) as exc:
        resolve_resume(final, resume_from="99")

    msg = str(exc.value)
    assert "2" in msg and "10" in msg  # lista los pasos disponibles


def test_resolve_resume_from_ruta_inexistente_lista_disponibles(tmp_path):
    """``--resume-from`` con una ruta inexistente → ``ValueError`` accionable (lista disponibles).
    """
    final = _final(tmp_path)
    _touch(tmp_path, f"{_STEM}_step00002.pt")

    with pytest.raises(ValueError, match="2"):
        resolve_resume(final, resume_from=str(tmp_path / f"{_STEM}_step99999.pt"))


# ------------------------------------- convención del sidecar


def test_resume_sidecar_path_convencion(tmp_path):
    """``X_stepNNNNN.pt`` → ``X_stepNNNNN.resume.pt`` (sidecar hermano) (Data Models)."""
    weights = tmp_path / f"{_STEM}_step00300.pt"
    assert resume_sidecar_path(weights) == tmp_path / f"{_STEM}_step00300.resume.pt"


# =============================================================================
# Task 3.2 — Carga y validación del punto de reanudación (load_resume /
#            validate_compatible)
# =============================================================================
#
# Carga los pesos + el sidecar del checkpoint elegido y arma el ResumeState listo para reanudar,
# tomando el ``history`` del ``meta`` del checkpoint de PESOS (no del sidecar, que no lo persiste)
# (1.3). Exige el sidecar (falta → error claro, 3.6) y valida compatibilidad EXACTA —SDE,
# ``data_dim`` y receta de red— contra el ``meta`` del checkpoint (2.5).

# Receta de red consistente con ``_net_with_optimizer_state`` (ScoreMLP 2D chico).
_MODEL_SPEC = {"name": "mlp", "kwargs": {"data_dim": 2, "hidden_dim": 16, "num_blocks": 1}}


def _build_checkpoint_and_sidecar(
    tmp_path,
    *,
    sde_name="vp",
    data_dim=2,
    model_spec=None,
    history=None,
    start_step=5,
    with_sidecar=True,
):
    """Arma un checkpoint de pesos real + (opcional) su sidecar hermano con los helpers commiteados.

    Devuelve ``(weights, sidecar, saved)`` donde ``saved`` es el :class:`ResumeState` que se
    persistió (o ``None`` si ``with_sidecar=False``), para comparar contra lo que ``load_resume``
    reconstruye.
    """
    if history is None:
        history = [1.0, 0.5, 0.25, 0.2, 0.1]
    net, opt = _net_with_optimizer_state()
    result = TrainResult(
        net=net, history=list(history), sde_name=sde_name, data_dim=data_dim
    )
    weights = tmp_path / f"{sde_name}_gaussian_step{start_step:05d}.pt"
    save_checkpoint(result, weights, model_spec=model_spec)
    sidecar = resume_sidecar_path(weights)
    saved = None
    if with_sidecar:
        saved = _resume_state(opt, start_step=start_step, history=history)
        save_resume_state(sidecar, saved)
    return weights, sidecar, saved


def _meta(*, sde_name="vp", data_dim=2, model=_MODEL_SPEC):
    """``meta`` sintético al estilo :func:`save_checkpoint` (la clave ``model`` es opcional)."""
    meta = {"sde_name": sde_name, "data_dim": data_dim, "history": [1.0]}
    if model is not None:
        meta["model"] = model
    return meta


# ------------------------------------------------- load_resume happy path (1.3)


def test_load_resume_happy_path(tmp_path):
    """Cargar un punto válido entrega ``(state_dict, meta, ResumeState)`` listo para reanudar.

    El ``history`` del :class:`ResumeState` viene del ``meta`` del checkpoint de PESOS (1.3) —el
    sidecar no lo persiste—; el paso, el optimizador y el azar vienen del sidecar.
    """
    history = [1.0, 0.5, 0.25, 0.2, 0.1]
    weights, sidecar, saved = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, history=history, start_step=5
    )
    expected = {"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}

    state_dict, meta, resume = load_resume(weights, expected=expected)

    assert isinstance(resume, ResumeState)
    # (1.3) history del meta de los PESOS, no del sidecar.
    assert resume.history == history
    assert resume.history == meta["history"]
    # paso y azar reconstruidos desde el sidecar.
    assert resume.start_step == 5
    assert torch.equal(resume.torch_rng_state, saved.torch_rng_state)
    assert torch.equal(resume.generator_state, saved.generator_state)
    # el state_dict es el de la red guardada (claves de un ScoreMLP; no vacío).
    ref_net = ScoreMLP(data_dim=2, hidden_dim=16, num_blocks=1)
    assert state_dict.keys() == ref_net.state_dict().keys()
    # el meta devuelto es el del checkpoint de pesos.
    assert meta["sde_name"] == "vp"
    assert meta["data_dim"] == 2
    # el optimizer_state (del sidecar) carga en un Adam fresco y trae momentos poblados.
    fresh = torch.optim.Adam(ref_net.parameters(), lr=1e-3)
    fresh.load_state_dict(resume.optimizer_state)
    assert fresh.state_dict()["state"]  # Adam ya tenía estado => round-trip real


def test_load_resume_history_del_meta_no_del_sidecar(tmp_path):
    """El ``history`` reconstruido es exactamente el del ``meta`` del checkpoint de pesos (1.3).

    Se guarda un ``history`` en el checkpoint de pesos y NADA de history en el sidecar (por diseño);
    ``load_resume`` debe rellenar el ``ResumeState`` desde el ``meta``.
    """
    history = [3.0, 2.0, 1.0]
    weights, sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, history=history, start_step=3
    )
    # confirmamos que el sidecar NO persistió history (contrato del sidecar).
    assert "history" not in load_resume_state(sidecar)

    _, meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )
    assert resume.history == history == meta["history"]


# ------------------------------------------------ sidecar faltante (3.6)


def test_load_resume_sidecar_faltante_error_claro(tmp_path):
    """Pesos presentes pero SIN sidecar → error claro que NOMBRA el artefacto faltante (3.6)."""
    weights, sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, with_sidecar=False
    )
    assert not sidecar.exists()
    expected = {"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}

    with pytest.raises((FileNotFoundError, ValueError)) as exc:
        load_resume(weights, expected=expected)

    assert sidecar.name in str(exc.value)  # el mensaje identifica el sidecar faltante


# ------------------------------------------------ validate_compatible (2.5)


def test_validate_compatible_match_no_levanta():
    """``meta`` idéntico a la corrida (SDE + data_dim + receta) → no levanta, devuelve ``None``."""
    assert (
        validate_compatible(_meta(), sde_name="vp", model_spec=_MODEL_SPEC, data_dim=2)
        is None
    )


def test_validate_compatible_sde_difiere():
    """SDE distinta entre ``meta`` y la corrida → ``ValueError`` (2.5)."""
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(sde_name="vp"), sde_name="ve", model_spec=_MODEL_SPEC, data_dim=2
        )


def test_validate_compatible_data_dim_int_difiere():
    """``data_dim`` int distinto (2 vs 4) → ``ValueError`` (2.5)."""
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(data_dim=2), sde_name="vp", model_spec=_MODEL_SPEC, data_dim=4
        )


def test_validate_compatible_data_dim_int_vs_tupla_difiere():
    """``data_dim`` int vs tupla (2 vs (1, 28, 28)) → ``ValueError`` (2.5).

    La comparación es por igualdad, así que un entero y una forma de evento nunca matchean.
    """
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(data_dim=2), sde_name="vp", model_spec=_MODEL_SPEC, data_dim=(1, 28, 28)
        )


def test_validate_compatible_data_dim_tupla_match():
    """``data_dim`` tupla igual (forma de evento de imágenes) → no levanta."""
    assert (
        validate_compatible(
            _meta(data_dim=(1, 28, 28)),
            sde_name="vp",
            model_spec=_MODEL_SPEC,
            data_dim=(1, 28, 28),
        )
        is None
    )


def test_validate_compatible_model_spec_difiere():
    """Receta de red distinta (kwargs distintos) → ``ValueError`` (2.5)."""
    otro = {"name": "mlp", "kwargs": {"data_dim": 2, "hidden_dim": 128, "num_blocks": 4}}
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(model=_MODEL_SPEC), sde_name="vp", model_spec=otro, data_dim=2
        )


def test_validate_compatible_ambos_model_none_ok():
    """``model_spec=None`` y ``meta`` sin clave ``model`` → match (``None == None``) (2.5)."""
    meta = _meta(model=None)  # sin clave 'model'
    assert "model" not in meta
    assert (
        validate_compatible(meta, sde_name="vp", model_spec=None, data_dim=2) is None
    )


def test_validate_compatible_receta_presente_vs_ausente_difiere():
    """Uno con receta y el otro sin → mismatch en ambos sentidos (2.5)."""
    # meta CON receta, corrida SIN.
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(model=_MODEL_SPEC), sde_name="vp", model_spec=None, data_dim=2
        )
    # meta SIN receta, corrida CON.
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(model=None), sde_name="vp", model_spec=_MODEL_SPEC, data_dim=2
        )


# ------------------------------ load_resume integra la validación (2.5)


def test_load_resume_incompatible_levanta(tmp_path):
    """Un ``expected`` incompatible con el ``meta`` (SDE distinta) → ``ValueError`` (2.5)."""
    weights, _sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, sde_name="vp", model_spec=_MODEL_SPEC, start_step=5
    )
    expected = {"sde_name": "ve", "model_spec": _MODEL_SPEC, "data_dim": 2}

    with pytest.raises(ValueError):
        load_resume(weights, expected=expected)


def test_load_resume_valida_antes_de_exigir_el_sidecar(tmp_path):
    """La compatibilidad se chequea ANTES de exigir el sidecar (2.5 precede a 3.6).

    Con un ``expected`` incompatible y sin sidecar, ``load_resume`` falla por incompatibilidad
    —no por el sidecar faltante— (el mensaje no nombra el sidecar).
    """
    weights, sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, sde_name="vp", model_spec=_MODEL_SPEC, with_sidecar=False
    )
    expected = {"sde_name": "ve", "model_spec": _MODEL_SPEC, "data_dim": 2}

    with pytest.raises(ValueError) as exc:
        load_resume(weights, expected=expected)

    assert sidecar.name not in str(exc.value)  # falló por compat, no por el sidecar


# =============================================================================
# Task 4 — Orquestación del CLI de entrenamiento (integración)
# =============================================================================
#
# Tests de nivel CLI de ``scripts/train.py`` (que no es un módulo del paquete): se carga su
# ``main`` con ``importlib.util.spec_from_file_location`` (el script auto-inyecta ``src`` en
# ``sys.path``) y se lo invoca con ``main(argv=[...])``. Se usan configs YAML mínimos en
# ``tmp_path`` (``num_steps`` chico, ``checkpoint_every`` > 0, ``out.checkpoint`` bajo
# ``tmp_path``) para correr en segundos de CPU y ejercitar el wiring skip/resume/fresh, el
# callback corregido (pesos + sidecar) y el reporte de la acción.

import importlib.util

yaml = pytest.importorskip("yaml")

_TRAIN_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train.py"


def _load_train_main():
    """Carga ``main`` de ``scripts/train.py`` como función invocable (el script no es un módulo)."""
    spec = importlib.util.spec_from_file_location("train_cli_under_test", _TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _write_cli_config(
    tmp_path,
    *,
    num_steps=6,
    checkpoint_every=2,
    with_checkpoint=True,
    with_loss_curve=False,
):
    """Escribe un ``.yaml`` mínimo (ScoreMLP chico) y devuelve ``(config_path, ckpt_path)``.

    Rutas de salida absolutas bajo ``tmp_path`` (posix, que pathlib maneja en Windows) para no
    depender del cwd. ``ckpt_path`` es ``None`` si ``with_checkpoint=False``.
    """
    run_dir = tmp_path / "run"
    ckpt = run_dir / "vp_gaussian.pt"
    out: dict = {}
    if with_checkpoint:
        out["checkpoint"] = ckpt.as_posix()
    if with_loss_curve:
        out["loss_curve"] = (run_dir / "vp_gaussian_loss.png").as_posix()
    cfg = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {
            "num_steps": num_steps,
            "lr": 0.002,
            "seed": 0,
            "checkpoint_every": checkpoint_every,
        },
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1},
        "out": out,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return str(config_path), (ckpt if with_checkpoint else None)


# ---------------------------------------- corrida fresca: final + snapshot + sidecar (1.1)


def test_cli_fresh_crea_final_snapshot_y_sidecar(tmp_path):
    """Corrida fresca con checkpointing intermedio: crea el final ``.pt``, al menos un snapshot
    ``…_stepNNNNN.pt`` y su sidecar ``…_stepNNNNN.resume.pt`` (prueba la corrección del callback,
    1.1).
    """
    cfg, ckpt = _write_cli_config(tmp_path, num_steps=6, checkpoint_every=2)
    main = _load_train_main()

    rc = main(["--config", cfg])

    assert rc == 0
    assert ckpt.exists()  # checkpoint final (3.9)
    snaps = discover_snapshots(ckpt)  # excluye final/best/sidecars
    assert snaps, "debería haber al menos un snapshot intermedio"
    step, snap_path = snaps[-1]
    sidecar = resume_sidecar_path(snap_path)
    assert sidecar.exists(), "el callback debe persistir el sidecar de resume junto a los pesos"
    # el sidecar es cargable y trae el estado de resume (paso + azar + optimizador).
    sc = load_resume_state(sidecar)
    assert sc["step"] == step
    assert set(sc) == SIDECAR_KEYS


# ---------------------------------------------------- skip si el final existe (3.1) + force (3.2)


def test_cli_skip_si_final_existe_y_force_reentrena(tmp_path, capsys):
    """Con el final presente la corrida se saltea sin sobrescribir (3.1); ``--force`` reentrena
    (reanudando desde el snapshot más nuevo) y sí reescribe el final (3.2).
    """
    cfg, ckpt = _write_cli_config(tmp_path, num_steps=6, checkpoint_every=2)
    main = _load_train_main()
    assert main(["--config", cfg]) == 0
    assert ckpt.exists()

    # Sentinela robusto (independiente de la resolución de mtime): si el skip NO reentrena, estos
    # bytes sobreviven; si el force reentrena, se sobrescriben con un checkpoint real.
    sentinel = b"SENTINEL-NO-TOCAR"
    ckpt.write_bytes(sentinel)
    capsys.readouterr()  # limpiar

    # (3.1) re-correr con el final presente => skip, salida 0, sin tocar el archivo.
    rc = main(["--config", cfg])
    assert rc == 0
    assert ckpt.read_bytes() == sentinel  # NO se sobrescribió
    out_skip = capsys.readouterr().out
    assert "--force" in out_skip and ("complet" in out_skip.lower())

    # (3.2) --force => reentrena (reanuda desde el más nuevo) y reescribe el final.
    rc = main(["--config", cfg, "--force"])
    assert rc == 0
    assert ckpt.read_bytes() != sentinel  # se sobrescribió con un checkpoint real
    out_force = capsys.readouterr().out
    assert "ya completa" not in out_force.lower()  # NO se salteó
    assert "reanud" in out_force.lower()  # force reanuda desde el snapshot más nuevo (3.2/3.3)
    # sigue siendo un checkpoint válido que cubre toda la corrida.
    _, meta = load_checkpoint(ckpt)
    assert len(meta["history"]) == 6


# ---------------------------------------------- auto-resume desde el más nuevo (3.3, 3.8, 3.9)


def test_cli_auto_resume_recrea_final(tmp_path, capsys):
    """Corrida interrumpida simulada (snapshots + sidecars, sin final): re-correr auto-reanuda
    desde el snapshot más nuevo, reporta la acción (3.8) y recrea el final cubriendo toda la
    corrida (3.3, 3.9).
    """
    cfg, ckpt = _write_cli_config(
        tmp_path, num_steps=6, checkpoint_every=2, with_loss_curve=True
    )
    main = _load_train_main()
    assert main(["--config", cfg]) == 0
    snaps = discover_snapshots(ckpt)
    assert snaps
    newest_step = snaps[-1][0]

    # Simular la interrupción: se borra el final pero quedan los snapshots + sidecars.
    ckpt.unlink()
    loss_curve = ckpt.parent / "vp_gaussian_loss.png"
    if loss_curve.exists():
        loss_curve.unlink()
    assert not ckpt.exists()
    capsys.readouterr()  # limpiar

    rc = main(["--config", cfg])

    assert rc == 0
    out = capsys.readouterr().out
    assert "reanud" in out.lower()  # reporta la acción de resume (3.8)
    assert f"paso {newest_step}" in out  # y el origen (paso del snapshot elegido)
    assert ckpt.exists()  # final recreado (3.9)
    _, meta = load_checkpoint(ckpt)
    assert len(meta["history"]) == 6  # la curva/historia cubren toda la corrida (3.9)
    assert loss_curve.exists()  # y la curva de pérdida se reescribe sobre toda la corrida (3.9)


# ------------------------------------------------ advertencia sin puntos de reanudación (1.5)


def test_cli_advierte_sin_puntos_de_reanudacion(tmp_path, capsys):
    """Con ``checkpoint_every=0`` el CLI advierte que la corrida NO dejará puntos de reanudación
    (1.5).
    """
    cfg, _ = _write_cli_config(tmp_path, num_steps=4, checkpoint_every=0)
    main = _load_train_main()

    rc = main(["--config", cfg])

    assert rc == 0
    out = capsys.readouterr().out
    assert "checkpoint_every" in out
    assert "reanud" in out.lower()


# ------------------------------------------------ --resume-from por paso puntual (3.5)


def test_cli_resume_from_step_selecciona_snapshot(tmp_path, capsys):
    """``--resume-from N`` reanuda desde ese snapshot puntual (no el más nuevo) y completa (3.5)."""
    cfg, ckpt = _write_cli_config(tmp_path, num_steps=6, checkpoint_every=2)
    main = _load_train_main()
    assert main(["--config", cfg]) == 0
    ckpt.unlink()  # sin final, para que la acción sea claramente un resume
    snaps = discover_snapshots(ckpt)
    assert len(snaps) >= 2
    chosen_step = snaps[0][0]  # el MÁS VIEJO, distinto del auto (más nuevo)
    capsys.readouterr()

    rc = main(["--config", cfg, "--resume-from", str(chosen_step)])

    assert rc == 0
    out = capsys.readouterr().out
    assert f"paso {chosen_step}" in out  # reanudó desde el snapshot elegido, no el más nuevo
    assert ckpt.exists()
    _, meta = load_checkpoint(ckpt)
    assert len(meta["history"]) == 6


# ------------------------------------------------ --resume-from inexistente => error limpio (3.7)


def test_cli_resume_from_inexistente_error_exit_2(tmp_path, capsys):
    """``--resume-from`` con un paso inexistente => salida 2 (error limpio); lista disponibles."""
    cfg, ckpt = _write_cli_config(tmp_path, num_steps=6, checkpoint_every=2)
    main = _load_train_main()
    assert main(["--config", cfg]) == 0
    ckpt.unlink()
    capsys.readouterr()

    rc = main(["--config", cfg, "--resume-from", "99999"])

    assert rc == 2  # ValueError mapeado a exit code 2 (patrón del script)


# =============================================================================
# ema-weights task 3 — sidecar extendido (crudos + sombra) y carga del punto
# =============================================================================
#
# Con EMA activo el checkpoint de pesos publica la SOMBRA (R2.1/R2.2), así que los pesos con los
# que hay que **continuar entrenando** —los crudos del momento— tienen que viajar en el sidecar y
# ``load_resume`` tiene que preferirlos (R3.1/R3.2). Las claves nuevas son **opcionales**: los
# campos requeridos del sidecar no cambian, así que un sidecar anterior a esta spec (corrida sin
# EMA, cuyo checkpoint ya es crudo) se sigue cargando exactamente como hoy (R3.3).

#: Claves del sidecar de una corrida CON EMA: las de siempre más las dos nuevas.
_SIDECAR_KEYS_EMA = SIDECAR_KEYS | {"raw_model_state", "ema_state"}


def _crudos_y_sombra(net):
    """Dos fotos distinguibles del ``state_dict`` de ``net``: los "crudos" y una "sombra".

    La sombra se desplaza (``+1``) para que ninguna comparación tensor a tensor pueda pasar por
    casualidad (un ``mul`` dejaría iguales los tensores nulos, p. ej. los bias recién inicializados).
    """
    crudos = {k: v.detach().clone() for k, v in net.state_dict().items()}
    sombra = {k: v.detach().clone().add_(1.0) for k, v in net.state_dict().items()}
    return crudos, sombra


# ------------------------------------------------- el sidecar lleva crudos + sombra (3.1)


def test_sidecar_con_ema_persiste_crudos_y_sombra(tmp_path):
    """Con EMA el sidecar incluye además los pesos crudos del momento y la sombra (3.1).

    Round-trip por disco: las dos claves nuevas vuelven tensor a tensor (``torch.equal``) sin
    alterar los campos requeridos (el ``step`` y el azar siguen ahí, el ``history`` sigue afuera).
    """
    net, opt = _net_with_optimizer_state()
    crudos, sombra = _crudos_y_sombra(net)
    resume = _resume_state(opt, start_step=5)
    resume.raw_model_state = crudos
    resume.ema_state = sombra

    path = tmp_path / "vp_gaussian_step00005.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert set(loaded) == _SIDECAR_KEYS_EMA
    assert loaded["step"] == 5
    assert "history" not in loaded  # el contrato del sidecar no cambia (1.3)
    for key in crudos:
        assert torch.equal(loaded["raw_model_state"][key], crudos[key]), key
        assert torch.equal(loaded["ema_state"][key], sombra[key]), key


def test_sidecar_sin_ema_no_agrega_claves(tmp_path):
    """Sin EMA el sidecar es **el de hoy**: ni ``raw_model_state`` ni ``ema_state`` (3.3).

    Las claves nuevas se persisten **solo si están**, así que el contenido del sidecar de una
    corrida sin EMA queda idéntico al de antes de esta feature — es lo que hace que la
    retrocompatibilidad sea por construcción y no por tolerancia del lector.
    """
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=2)
    assert resume.raw_model_state is None  # default del dataclass: sin EMA
    assert resume.ema_state is None

    path = tmp_path / "sin_ema.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert set(loaded) == SIDECAR_KEYS


# --------------------------------- load_resume prefiere los crudos del sidecar (3.1, 3.2)


def _checkpoint_ema_con_sidecar(tmp_path, *, start_step=4, decay=0.6):
    """Punto de reanudación de una corrida **con EMA**, como lo deja el CLI.

    El checkpoint de pesos publica la **sombra** (política de :func:`save_checkpoint`) y el sidecar
    hermano lleva los **crudos** del momento más la sombra.

    Returns:
        ``(weights, crudos, sombra, history)``.
    """
    net, opt = _net_with_optimizer_state()
    crudos, sombra = _crudos_y_sombra(net)
    history = [1.0, 0.5, 0.25, 0.2]
    result = TrainResult(
        net=net,
        history=list(history),
        config=TrainConfig(ema_decay=decay),
        sde_name="vp",
        data_dim=2,
        ema_state=sombra,
    )
    weights = tmp_path / f"vp_gaussian_step{start_step:05d}.pt"
    save_checkpoint(result, weights, model_spec=_MODEL_SPEC)

    resume = _resume_state(opt, start_step=start_step, history=history)
    resume.raw_model_state = crudos
    resume.ema_state = sombra
    save_resume_state(resume_sidecar_path(weights), resume)
    return weights, crudos, sombra, history


def test_load_resume_con_ema_devuelve_los_crudos_del_sidecar(tmp_path):
    """El checkpoint publica EMA ⇒ ``load_resume`` devuelve los **crudos** del sidecar (3.1, 3.2).

    Es el punto que el enfoque A obliga a extender: devolver el ``state_dict`` del checkpoint de
    pesos cargaría la **sombra** como punto de partida del entrenamiento (pesos promediados
    disfrazados de crudos, corriendo el optimizador sobre un estado que nunca existió). El
    ``meta``/``history`` siguen viniendo del checkpoint de pesos, y el :class:`ResumeState` se arma
    con la sombra para restaurarla en el loop.
    """
    weights, crudos, sombra, history = _checkpoint_ema_con_sidecar(tmp_path, start_step=4)
    expected = {"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}

    state_dict, meta, resume = load_resume(weights, expected=expected)

    # Lo publicado en el checkpoint es la sombra; lo que devuelve load_resume son los crudos.
    publicado, _ = load_checkpoint(weights)
    for key in crudos:
        assert torch.equal(publicado[key], sombra[key]), f"el checkpoint no publicó EMA en {key}"
        assert torch.equal(state_dict[key], crudos[key]), f"no devolvió los crudos en {key}"
    assert any(not torch.equal(state_dict[k], publicado[k]) for k in crudos)  # crudos ≠ EMA

    # El ResumeState lleva la sombra (la restaura ``train``); los crudos NO se duplican en él —
    # ya se devuelven como ``state_dict`` para que el caller los cargue en la red.
    assert resume.ema_state is not None
    for key in sombra:
        assert torch.equal(resume.ema_state[key], sombra[key]), key
    assert resume.raw_model_state is None

    # El resto del contrato de load_resume, intacto: paso y azar del sidecar, history del meta.
    assert resume.start_step == 4
    assert resume.history == history == meta["history"]
    assert meta["ema"] == {"decay": pytest.approx(0.6)}


def test_load_resume_sidecar_viejo_sin_claves_de_ema_como_hoy(tmp_path):
    """Sidecar anterior a esta spec (sin claves de EMA) ⇒ carga **exactamente como hoy** (3.3).

    El sidecar de una corrida sin EMA no tiene ``raw_model_state``, y su checkpoint de pesos ya
    publica los crudos: ``load_resume`` devuelve ese ``state_dict`` como siempre y el
    :class:`ResumeState` queda sin sombra (lo que después habilita el camino sin EMA del loop).
    """
    weights, sidecar, saved = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, start_step=5
    )
    assert set(load_resume_state(sidecar)) == SIDECAR_KEYS  # sidecar "viejo": sin claves nuevas

    publicado, _ = load_checkpoint(weights)
    state_dict, meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )

    assert state_dict.keys() == publicado.keys()
    for key in publicado:
        assert torch.equal(state_dict[key], publicado[key]), key  # los pesos del checkpoint
    assert resume.ema_state is None
    assert resume.raw_model_state is None
    assert "ema" not in meta
    assert resume.start_step == 5
    assert torch.equal(resume.generator_state, saved.generator_state)


# =============================================================================
# gpu-training-efficiency task 3 — persistencia/restauración del escalador AMP
# =============================================================================
#
# Con AMP activo el loop mantiene un ``GradScaler`` cuyo estado (el *scale factor* y su tracker de
# crecimiento) debe viajar en el sidecar para que la reanudación sea fiel (R2.1/R2.2). La clave es
# **opcional** —calca ``ema_state``—: sin AMP el sidecar no gana ninguna clave y los sidecars
# previos siguen válidos (R2.4). Los dos config↔sidecar incoherentes se rechazan (R2.5). En CPU el
# escalador va deshabilitado (``state_dict() == {}``), así que "presencia" se decide por ``is None``,
# no por dict vacío.

#: Claves del sidecar de una corrida CON AMP: las de siempre más la nueva.
_SIDECAR_KEYS_AMP = SIDECAR_KEYS | {"scaler_state"}


def _scaler_state_no_vacio() -> dict:
    """``state_dict()`` de un ``GradScaler`` HABILITADO (no vacío), para un round-trip real.

    En CPU el loop usa un escalador **deshabilitado** (``state_dict() == {}``); acá se fuerza uno
    habilitado para que el round-trip pruebe que las claves del escalador (scale + tracker)
    sobreviven la serialización, igual que ``_net_with_optimizer_state`` fuerza estado en Adam.
    """
    return torch.amp.GradScaler("cpu", enabled=True).state_dict()


def _data_prohibida_amp():
    """Fuente que revienta si se la consume: prueba que los guards fallan ANTES de tocar datos."""
    raise AssertionError("no se debe consumir datos: el guard de AMP falla fail-fast antes")
    yield  # pragma: no cover  (nunca se alcanza; hace de esto un generador)


# ------------------------------------------------- round-trip del sidecar (2.1)


def test_sidecar_con_amp_persiste_scaler_state(tmp_path):
    """Con AMP el sidecar incluye ``scaler_state`` y vuelve por disco intacto (2.1).

    Las claves requeridas no cambian; la clave nueva se suma y round-trippea (el *scale factor* y
    su tracker se preservan). El ``history`` sigue afuera (1.3).
    """
    _, opt = _net_with_optimizer_state()
    scaler_state = _scaler_state_no_vacio()
    resume = _resume_state(opt, start_step=5)
    resume.scaler_state = scaler_state

    path = tmp_path / "vp_gaussian_step00005.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert set(loaded) == _SIDECAR_KEYS_AMP
    assert "history" not in loaded  # el contrato del sidecar no cambia (1.3)
    assert loaded["scaler_state"] == scaler_state  # scale + tracker preservados


def test_sidecar_sin_amp_no_agrega_scaler_state(tmp_path):
    """Sin AMP el sidecar es **el de hoy**: ni ``scaler_state`` (2.4).

    La clave nueva se persiste **solo si está**, así que el contenido del sidecar de una corrida
    sin AMP queda idéntico al de antes de esta feature — retrocompatibilidad por construcción.
    """
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=2)
    assert resume.scaler_state is None  # default del dataclass: sin AMP

    path = tmp_path / "sin_amp.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert set(loaded) == SIDECAR_KEYS
    assert "scaler_state" not in loaded


# ----------------------------------- load_resume expone el scaler_state (2.2)


def test_load_resume_devuelve_scaler_state_del_sidecar(tmp_path):
    """El sidecar trae ``scaler_state`` ⇒ ``load_resume`` lo devuelve en el ``ResumeState`` (2.2)."""
    net, opt = _net_with_optimizer_state()
    history = [1.0, 0.5, 0.25, 0.2, 0.1]
    result = TrainResult(net=net, history=list(history), sde_name="vp", data_dim=2)
    weights = tmp_path / "vp_gaussian_step00005.pt"
    save_checkpoint(result, weights, model_spec=_MODEL_SPEC)

    scaler_state = _scaler_state_no_vacio()
    saved = _resume_state(opt, start_step=5, history=history)
    saved.scaler_state = scaler_state
    save_resume_state(resume_sidecar_path(weights), saved)

    _sd, _meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )

    assert resume.scaler_state == scaler_state


def test_load_resume_sidecar_sin_scaler_state_es_none(tmp_path):
    """Sidecar sin la clave (corrida sin AMP / anterior a la feature) ⇒ ``scaler_state is None`` (2.4)."""
    weights, sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, start_step=5
    )
    assert "scaler_state" not in load_resume_state(sidecar)  # sidecar "viejo"

    _sd, _meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )
    assert resume.scaler_state is None


# ------------------------------------------------- guards cruzados (2.5)


def test_train_con_amp_rechaza_reanudar_sin_scaler_en_el_sidecar():
    """AMP pedido + sidecar sin escalador ⇒ ``ValueError`` fail-fast antes de entrenar (2.5).

    No se reanuda con un escalador inventado: el punto de reanudación no lo guardó (¿la corrida
    original corrió sin AMP?). El guard va con el resto del fail-fast (la fuente prohibida convierte
    cualquier consumo de datos en un ``AssertionError`` distinto del error esperado).
    """
    sde = make_sde("vp")
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=5)  # scaler_state None por defecto
    cfg = TrainConfig(num_steps=8, seed=0, amp=True)

    with pytest.raises(ValueError) as exc:
        train(sde, _small_net(sde), _data_prohibida_amp(), cfg, resume=resume)

    assert "escalador" in str(exc.value).lower() or "amp" in str(exc.value).lower()


def test_train_sin_amp_rechaza_reanudar_con_scaler_en_el_sidecar():
    """Escalador en el sidecar + AMP no pedido ⇒ ``ValueError`` (2.5, simetría del guard).

    La corrida original usaba AMP; continuar sin AMP descartaría el escalador en silencio. Se falla
    con el mismo criterio fail-fast.
    """
    sde = make_sde("vp")
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=5)
    resume.scaler_state = _scaler_state_no_vacio()
    cfg = TrainConfig(num_steps=8, seed=0)  # sin AMP

    with pytest.raises(ValueError) as exc:
        train(sde, _small_net(sde), _data_prohibida_amp(), cfg, resume=resume)

    assert "escalador" in str(exc.value).lower() or "amp" in str(exc.value).lower()


# ------------------------------------------------- fidelidad en CPU (2.3)


def test_resume_con_amp_equivalente_a_corrida_ininterrumpida():
    """Gate de fidelidad (2.3): con AMP activo y orden fijo, reanudar equivale a no interrumpir.

    Mismo montaje que ``test_resume_equivalente_a_corrida_ininterrumpida`` pero con ``amp=True``: el
    snapshot del paso ``_N`` lleva el ``scaler_state`` (en CPU va deshabilitado ⇒ ``{}``, pero
    **presente**, no ``None``), y la corrida B —red fresca con los pesos del paso ``_N``, reanudada
    con AMP— reproduce A: ``history`` idéntico y pesos ``allclose``.
    """
    sde = make_sde("vp")
    batch = _fixed_batch()
    frozen: dict[str, TrainSnapshot] = {}

    def capture(tag, snap):
        frozen[tag] = copy.deepcopy(snap)

    net_a = _equiv_net()
    result_a = train(
        sde,
        net_a,
        _const_source(batch),
        TrainConfig(num_steps=_TOTAL, checkpoint_every=_N, seed=0, amp=True),
        on_checkpoint=capture,
    )

    snap = frozen[f"step{_N:05d}"]
    assert snap.resume.start_step == _N
    assert snap.resume.scaler_state is not None  # el escalador viajó en el sidecar (2.1)

    net_b = _equiv_net()
    net_b.load_state_dict(snap.result.net.state_dict())  # pesos congelados del paso _N
    result_b = train(
        sde,
        net_b,
        _const_source(batch),  # misma fuente de orden fijo
        TrainConfig(num_steps=_TOTAL, seed=0, amp=True),
        resume=snap.resume,
    )

    assert result_b.history == result_a.history
    assert _weights_allclose(net_a, net_b)


# --------------------------------------------- retención rolling (prune_snapshots)


def _touch_path(p: pathlib.Path) -> None:
    """Crea un archivo vacío (los tests de prune son puro filesystem, no necesitan pesos reales).

    Nombre propio (`_touch_path`) para NO pisar el `_touch(dir, name)` que ya usa el resto del
    archivo: éste recibe una ruta completa ya armada.
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _stub_snapshots(base, steps, *, sidecars=True, final=True, raw=False, best=False):
    """Arma el layout de checkpoints en disco (final + snapshots …_stepNNNNN.pt + sidecars)."""
    if final:
        _touch_path(base)  # X.pt
    if raw:
        _touch_path(base.with_stem(base.stem + "_raw"))  # X_raw.pt (contraparte cruda del final)
    if best:
        _touch_path(base.with_stem(base.stem + "_best"))  # X_best.pt legado
    for s in steps:
        snap = base.with_stem(f"{base.stem}_step{s:05d}")
        _touch_path(snap)
        if sidecars:
            _touch_path(resume_sidecar_path(snap))


def _steps_en_disco(base):
    return {step for step, _ in discover_snapshots(base)}


def test_prune_snapshots_conserva_los_n_mas_nuevos(tmp_path):
    """Conserva los `keep_last` snapshots de mayor paso y borra los viejos + sus sidecars."""
    base = tmp_path / "run.pt"
    _stub_snapshots(base, [100, 200, 300, 400, 500])

    borrados = prune_snapshots(base, keep_last=2)

    # 3 snapshots viejos (100/200/300) × (.pt + .resume.pt) = 6 archivos borrados.
    assert len(borrados) == 6
    assert _steps_en_disco(base) == {400, 500}
    for s in (100, 200, 300):
        snap = base.with_stem(f"{base.stem}_step{s:05d}")
        assert not snap.exists()
        assert not resume_sidecar_path(snap).exists()
    for s in (400, 500):
        assert resume_sidecar_path(base.with_stem(f"{base.stem}_step{s:05d}")).exists()


def test_prune_snapshots_noop_si_hay_pocos(tmp_path):
    """Con `<= keep_last` snapshots no borra nada."""
    base = tmp_path / "run.pt"
    _stub_snapshots(base, [100, 200])
    assert prune_snapshots(base, keep_last=5) == []
    assert _steps_en_disco(base) == {100, 200}


def test_prune_snapshots_nunca_toca_final_ni_raw_ni_best(tmp_path):
    """El checkpoint final, su `_raw` y los `_best` legados nunca se borran."""
    base = tmp_path / "run.pt"
    _stub_snapshots(base, [100, 200, 300], raw=True, best=True)

    prune_snapshots(base, keep_last=1)

    assert base.exists()  # final intacto
    assert base.with_stem(base.stem + "_raw").exists()
    assert base.with_stem(base.stem + "_best").exists()
    assert _steps_en_disco(base) == {300}  # solo se conserva el más nuevo


def test_prune_snapshots_sidecar_ausente_no_falla(tmp_path):
    """Un snapshot sin sidecar (corrida sin resume) se borra igual, sin error."""
    base = tmp_path / "run.pt"
    _stub_snapshots(base, [100, 200, 300], sidecars=False)

    borrados = prune_snapshots(base, keep_last=1)

    assert len(borrados) == 2  # solo los .pt viejos (100, 200), sin sidecars
    assert _steps_en_disco(base) == {300}


def test_prune_snapshots_keep_last_invalido(tmp_path):
    """`keep_last < 1` es error: conservar cero snapshots dejaría la corrida sin resume."""
    base = tmp_path / "run.pt"
    _stub_snapshots(base, [100])
    with pytest.raises(ValueError):
        prune_snapshots(base, keep_last=0)


# ---------------------------------------- log de entrenamiento (.jsonl) vía CLI


def test_cli_escribe_train_log_jsonl(tmp_path):
    """Con `out.train_log`, el CLI escribe un .jsonl con start + steps + end (con timestamp)."""
    import json

    run_dir = tmp_path / "run"
    log_path = run_dir / "train_log.jsonl"
    cfg = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 10, "lr": 0.002, "seed": 0, "log_every": 5},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1},
        "out": {"checkpoint": (run_dir / "m.pt").as_posix(), "train_log": log_path.as_posix()},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    assert _load_train_main()(["--config", str(config_path), "--quiet"]) == 0

    lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event"] == "start"
    assert lines[0]["sde"] == "vp" and lines[0]["num_steps"] == 10
    assert lines[-1]["event"] == "end"
    assert lines[-1]["step"] == 10 and "loss_final" in lines[-1] and "elapsed_s" in lines[-1]
    steps = [r for r in lines if r["event"] == "step"]
    assert steps and all({"step", "loss", "elapsed_s", "t"} <= set(r) for r in steps)


# =============================================================================
# validation-loss task 5 — persistencia de la serie de validación y reanudación
# =============================================================================
#
# La serie dispersa de validación viaja en el ``meta`` del checkpoint de PESOS, igual que el
# ``history`` y por el mismo motivo: no se duplica en el sidecar (que sigue con sus cuatro claves
# requeridas). ``load_resume`` la lee de forma **tolerante** (``meta.get``), así que un checkpoint
# anterior a esta feature se reanuda exactamente como siempre (5.5). No hay guard cruzado
# config↔sidecar: la serie es una *observación*, no estado necesario para continuar la optimización.
#
# El criterio 6.4 (mismo examen después del corte) se cubre acá con un test y **sin código**: el
# examen fijo se re-siembra con una constante del módulo en cada evaluación, así que no hay ningún
# estado que persistir — reconstruirlo tras el corte da el mismo número por construcción.

from diffusion.training import (  # noqa: E402  (sección append-only, task 5)
    FixedValExam,
    make_time_sampler,
)

#: Receta de red consistente con ``_equiv_net`` (el ``ScoreMLP`` de las corridas de esta suite).
_RECETA_EQUIV = {"name": "mlp", "kwargs": {"data_dim": 2, "hidden_dim": 32, "num_blocks": 1}}

#: Serie sintética de un punto, con las cuatro claves de ``ValPoint``.
_PUNTOS_VAL: list[dict] = [{"step": 5, "raw": 1.5, "ema": None, "train": 1.2}]


def _fuente_examen_resume(n=6, batch=4, dim=2, seed=99):
    """Fuente RE-ITERABLE de batches en memoria, con una cola parcial (como el examen real).

    Recorrerla dos veces entrega la misma secuencia de tensores, que es lo que el examen fijo
    necesita, sin tocar el disco ni depender de torchvision.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=gen)
    return [x[i:i + batch] for i in range(0, n, batch)]


def _pasos(serie) -> list[int]:
    """Los pasos de una serie dispersa de validación (el índice de la serie)."""
    return [punto["step"] for punto in serie]


# ------------------------------------ el sidecar sigue sin llevar la serie (5.5)


def test_el_sidecar_no_persiste_la_serie_de_validacion(tmp_path):
    """La serie viaja en el ``meta`` del checkpoint, NO en el sidecar (5.2, 5.5).

    Calca la decisión del ``history`` (1.3): el sidecar guarda solo lo que el checkpoint de pesos no
    tiene, y la serie ya está en su ``meta``. La tupla de campos requeridos del sidecar la excluye, así
    que un ``ResumeState`` con serie produce un sidecar con **exactamente** las claves de siempre.
    """
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=5)
    resume.val_history = [dict(punto) for punto in _PUNTOS_VAL]

    path = tmp_path / "con_serie.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert set(loaded) == SIDECAR_KEYS
    assert "val_history" not in loaded
    assert "history" not in loaded  # el contrato del sidecar no cambia (1.3)


# ------------------------ load_resume lee la serie del meta de los pesos (6.3)


def test_load_resume_devuelve_la_serie_del_meta_del_checkpoint(tmp_path):
    """El ``meta`` trae la serie ⇒ ``load_resume`` la pone en el ``ResumeState`` (6.3).

    Misma procedencia que el ``history``: el checkpoint de **pesos**, no el sidecar. Se asevera
    también que el sidecar del mismo punto no la trae, para que el test no pueda pasar por la ruta
    equivocada.
    """
    net, opt = _net_with_optimizer_state()
    history = [1.0, 0.5, 0.25, 0.2, 0.1]
    result = TrainResult(net=net, history=list(history), sde_name="vp", data_dim=2)
    result.val_history = [dict(punto) for punto in _PUNTOS_VAL]
    weights = tmp_path / "vp_gaussian_step00005.pt"
    save_checkpoint(result, weights, model_spec=_MODEL_SPEC)
    sidecar = resume_sidecar_path(weights)
    save_resume_state(sidecar, _resume_state(opt, start_step=5, history=history))

    assert "val_history" not in load_resume_state(sidecar)  # no llega por el sidecar

    _sd, meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )

    assert meta["val_history"] == _PUNTOS_VAL
    assert resume.val_history == meta["val_history"]
    assert resume.history == history  # el resto del contrato de load_resume, intacto


def test_load_resume_checkpoint_sin_la_clave_deja_la_serie_en_none(tmp_path):
    """Checkpoint anterior a la feature (sin la clave) ⇒ ``val_history is None``, sin error (5.5).

    La lectura es **tolerante** (``meta.get``): reanudar un checkpoint viejo no puede fallar por una
    clave que en su momento no existía, y ``None`` es exactamente "este punto de reanudación no trae
    serie" — el loop arranca la serie vacía y sigue.
    """
    weights, sidecar, _saved = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, start_step=5
    )
    _sd0, meta0 = load_checkpoint(weights)
    assert "val_history" not in meta0  # checkpoint "viejo": la clave no está

    _sd, _meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )
    assert resume.val_history is None
    assert resume.start_step == 5  # se reanuda igual que siempre
    assert "val_history" not in load_resume_state(sidecar)


# --------------- el snapshot del paso N y la continuación de la serie (6.3)


def test_reanudar_desde_el_snapshot_del_paso_n_continua_la_serie_sin_duplicar(tmp_path):
    """End-to-end (6.3): el snapshot de N trae los puntos hasta N y reanudar continúa la serie.

    Corrida A de 4 pasos con cadencia 2 mide en 2 y 4, y persiste el snapshot del paso 2 (el 4 es el
    último paso, que el disparador de snapshots excluye). Tres cosas se verifican sobre ese artefacto:

    - su ``meta`` trae **exactamente** el punto medido hasta 2 (la evaluación va antes del snapshot);
    - reanudar desde ahí hasta 6 pasos deja la serie en ``[2, 4, 6]``: el punto previo se conserva
      **idéntico** y no se vuelve a medir (el paso 2 ya está completado, así que la próxima evaluación
      cae en 4);
    - el sidecar del punto no lleva la serie: llegó por el ``meta``.
    """
    sde = make_sde("vp")
    batch = _fixed_batch()
    base = tmp_path / "vp_val.pt"

    def persistir(tag, snap):
        weights = base.with_stem(f"{base.stem}_{tag}")
        save_checkpoint(snap.result, weights, model_spec=_RECETA_EQUIV)
        save_resume_state(resume_sidecar_path(weights), snap.resume)

    net_a = _equiv_net()
    res_a = train(
        sde,
        net_a,
        _const_source(batch),
        TrainConfig(num_steps=4, checkpoint_every=2, seed=0),
        on_checkpoint=persistir,
        val_batches=_fuente_examen_resume(),
        train_exam_batches=_fuente_examen_resume(seed=7),
    )
    assert _pasos(res_a.val_history) == [2, 4]

    weights = base.with_stem(f"{base.stem}_step00002")
    state_dict, meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _RECETA_EQUIV, "data_dim": 2}
    )

    # (a) el meta del snapshot del paso 2 trae exactamente los puntos medidos hasta 2.
    assert meta["val_history"] == res_a.val_history[:1]
    assert _pasos(meta["val_history"]) == [2]
    assert resume.val_history == meta["val_history"]
    assert "val_history" not in load_resume_state(resume_sidecar_path(weights))

    # (b) reanudar continúa la serie: ni pierde el punto previo ni duplica el del paso 2.
    net_b = _equiv_net()
    net_b.load_state_dict(state_dict)
    res_b = train(
        sde,
        net_b,
        _const_source(batch),
        TrainConfig(num_steps=6, checkpoint_every=2, seed=0),
        resume=resume,
        val_batches=_fuente_examen_resume(),
        train_exam_batches=_fuente_examen_resume(seed=7),
    )

    assert _pasos(res_b.val_history) == [2, 4, 6]
    assert res_b.val_history[0] == res_a.val_history[0]  # el punto previo, intacto
    assert len(res_b.history) == 6


# ------------------------------- el examen se rearma igual tras el corte (6.4)


def test_el_examen_se_reconstruye_igual_despues_del_corte():
    """El examen fijo se rearma **idéntico** tras un corte: no hay estado que persistir (6.4).

    Es la contracara de la persistencia de la serie: los puntos medidos antes y después del corte son
    comparables porque el examen se reconstruye igual, y eso sale gratis de la re-siembra —cada
    evaluación crea un generator nuevo sembrado con ``VAL_EXAM_SEED``—. El test simula el corte
    tirando examen, fuente y muestreador y rearmándolos de cero, como haría un proceso nuevo al
    reanudar, y **mueve el RNG global en el medio** para que la igualdad no pueda venir de que el
    estado del proceso se mantuvo intacto.
    """
    sde = make_sde("vp")
    net = _equiv_net()

    def armar_examen() -> FixedValExam:
        # Todo nuevo: la fuente, el muestreador de tiempos y el examen. Nada viaja del corte.
        return FixedValExam(
            sde,
            _fuente_examen_resume(),
            time_sampler=make_time_sampler("uniform", sde.T, 1e-3),
            device="cpu",
        )

    antes = armar_examen().evaluate(net)
    torch.manual_seed(31337)  # el corte cambia el estado del proceso; el examen no depende de él
    torch.randn(5)
    despues = armar_examen().evaluate(net)

    assert despues == antes  # igualdad EXACTA de floats, no aproximada
    # El examen tampoco depende de la RED por accidente: con otros pesos el número cambia (si no,
    # la igualdad de arriba sería trivial y no probaría nada).
    otra = _equiv_net()
    assert armar_examen().evaluate(otra) != antes


# =============================================================================
# validation-loss task 7.1 — wiring del CLI y el evento de validación en el log
# =============================================================================
#
# Tests de nivel CLI (mismo patrón que la sección de la task 4: ``scripts/train.py`` se carga por
# ruta con ``importlib``). Acá el config es de IMÁGENES —la validación solo aplica a esa fuente—
# con dos carpetas HERMANAS sintetizadas con PIL en ``tmp_path`` y una U-Net mínima, así la corrida
# entera son unos pocos pasos de CPU.
#
# Lo que se asevera es el contrato del ``.jsonl`` y de la consola:
#
# - el registro de validación llega por el ``on_log`` YA EXISTENTE y se distingue por su campo
#   ``event``, que sobreescribe el ``"step"`` genérico del envoltorio del CLI porque el registro se
#   expande **al final** (``**rec``). Ese orden es un invariante de la feature: invertirlo escribiría
#   todos los puntos de validación como si fueran pasos de entrenamiento;
# - las ausencias viajan como clave presente con ``null`` (nunca omitidas): un consumidor del
#   ``.jsonl`` distingue "esta corrida no tenía EMA" de "este formato no tiene el campo";
# - sin ``data.val_root`` el archivo y la consola salen exactamente como antes de la feature.

#: Claves de un registro de validación en el ``.jsonl`` (las del loop + las que agrega el CLI).
_CLAVES_VAL_JSONL = {
    "t", "event", "elapsed_s", "step", "val_raw", "val_ema", "train_fijo", "device",
}

#: Claves de un registro de paso de entrenamiento (contrato previo a esta feature, sin tocar).
_CLAVES_STEP_JSONL = {"t", "event", "elapsed_s", "step", "loss"}


def _load_train_module():
    """Carga ``scripts/train.py`` como módulo (no solo su ``main``).

    Igual que :func:`_load_train_main`, pero devuelve el módulo: los tests que espían la escritura
    compatible con la barra de progreso necesitan el ``tqdm`` que el script importó.
    """
    spec = importlib.util.spec_from_file_location("train_cli_under_test", _TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _carpetas_imagenes_cli(tmp_path, *, n_train=8, n_val=6, size=8):
    """Sintetiza dos carpetas **hermanas** de PNGs: ``imgs_train/`` y ``imgs_val/``.

    Hermanas y no anidadas a propósito: el descubrimiento de imágenes es **recursivo**, así que un
    ``val/`` dentro del ``root`` de entrenamiento contaminaría el set de train (advertencia
    operativa de la feature).

    ``n_val=6`` con ``batch_size=4`` deja una **cola parcial** en los dos exámenes (el de train se
    dimensiona con la cantidad de imágenes del de validación), que es la forma real del examen.
    """
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("torchvision")  # la fuente de imágenes (transforms + loader)
    train_dir, val_dir = tmp_path / "imgs_train", tmp_path / "imgs_val"
    train_dir.mkdir()
    val_dir.mkdir()
    for i in range(n_train):
        Image.new("RGB", (size, size), color=(i * 20, 40, 200 - i * 10)).save(
            train_dir / f"t{i}.png"
        )
    for i in range(n_val):
        Image.new("RGB", (size, size), color=(10, i * 30, 60)).save(val_dir / f"v{i}.png")
    return train_dir, val_dir


def _write_cli_config_val(
    tmp_path,
    *,
    val_dir=None,
    num_steps=3,
    checkpoint_every=2,
    ema_decay=None,
    size=8,
    with_train_log=True,
    with_loss_curve=False,
):
    """Escribe el ``.yaml`` de una corrida de imágenes mínima; devuelve ``(cfg, ckpt, log)``.

    ``val_dir=None`` reproduce el camino sin validación (la clave ``data.val_root`` simplemente no
    se declara: es opt-in). La U-Net es la mínima de los smokes de ``test_config_image``: 2 niveles
    (reduction 2, divide a ``size=8``), 1 res-block, sin atención, ``groups=4``.

    ``with_train_log=False`` omite ``out.train_log``: el ``.jsonl`` también es opt-in y la
    validación tiene que informar igual por consola sin él.

    ``with_loss_curve=True`` agrega ``out.loss_curve`` (el PNG queda en
    ``<tmp_path>/run/vp_imgs_loss.png``, hermano del checkpoint), para los tests del gráfico.
    """
    train_dir = tmp_path / "imgs_train"
    run_dir = tmp_path / "run"
    ckpt = run_dir / "vp_imgs.pt"
    log_path = run_dir / "train_log.jsonl"
    data: dict = {
        "kind": "images",
        "root": str(train_dir),
        "image_size": size,
        "batch_size": 4,
        "augment": False,   # determinístico y más rápido
        "crop": True,
        "shuffle": False,
        "seed": 0,
    }
    if val_dir is not None:
        data["val_root"] = str(val_dir)
    train_raw: dict = {
        "num_steps": num_steps,
        "lr": 1e-3,
        "seed": 0,
        "device": "cpu",
        "checkpoint_every": checkpoint_every,
        "log_every": 1,
    }
    if ema_decay is not None:
        train_raw["ema_decay"] = ema_decay
    out: dict = {"checkpoint": ckpt.as_posix()}
    if with_train_log:
        out["train_log"] = log_path.as_posix()
    if with_loss_curve:
        out["loss_curve"] = (run_dir / "vp_imgs_loss.png").as_posix()
    cfg = {
        "sde": {"name": "vp"},
        "data": data,
        "train": train_raw,
        "model": {
            "name": "unet",
            "image_size": size,
            "base_channels": 8,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "embed_dim": 16,
            "time_embed_dim": 32,
            "groups": 4,
        },
        "out": out,
    }
    config_path = tmp_path / "config_val.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return str(config_path), ckpt, log_path


def _leer_jsonl(path) -> list[dict]:
    import json

    return [json.loads(linea) for linea in path.read_text(encoding="utf-8").splitlines()]


# ------------------------ el registro de validación en el .jsonl (5.1, 5.7)


def test_cli_con_validacion_escribe_el_evento_val_en_el_jsonl(tmp_path):
    """Con ``data.val_root`` el ``.jsonl`` trae una línea por evaluación, distinguible (5.1, 5.7).

    Cubre el wiring completo de la task: que el CLI reenvíe **las dos** fuentes de la corrida
    ensamblada al loop (la validación mide ⇒ ``val_batches`` llegó; ``train_fijo`` no es nulo ⇒
    ``train_exam_batches`` también), y que el registro se distinga de los de paso por su ``event``
    propio, con el paso, los tres valores, el timestamp, el tiempo transcurrido y el **device**
    (el examen fijo es específico del device).

    Los pasos medidos son 2 y 3 con ``num_steps=3`` / ``checkpoint_every=2``: la cadencia y el
    **último paso**, que es el disparador propio de la validación.
    """
    _, val_dir = _carpetas_imagenes_cli(tmp_path)
    cfg, ckpt, log_path = _write_cli_config_val(
        tmp_path, val_dir=val_dir, num_steps=3, checkpoint_every=2, ema_decay=0.6
    )

    assert _load_train_main()(["--config", cfg, "--quiet"]) == 0

    lineas = _leer_jsonl(log_path)
    vals = [r for r in lineas if r.get("event") == "val"]
    assert [r["step"] for r in vals] == [2, 3]
    for r in vals:
        assert set(r) == _CLAVES_VAL_JSONL
        assert isinstance(r["val_raw"], float) and r["val_raw"] > 0
        assert isinstance(r["val_ema"], float)      # EMA activo ⇒ segundo valor real (4.1)
        assert isinstance(r["train_fijo"], float)   # el examen fijo de train llegó al loop (3.8)
        assert r["device"] == "cpu"
        assert isinstance(r["elapsed_s"], float) and r["t"]

    # Los registros de PASO no cambian: mismo event genérico y mismas claves que antes de la
    # feature, sin ninguna de las de validación (el envoltorio no las filtra ni las agrega).
    steps = [r for r in lineas if r.get("event") == "step"]
    assert steps and all(set(r) == _CLAVES_STEP_JSONL for r in steps)

    # El log y la serie del checkpoint son la MISMA medición (no dos caminos distintos).
    _, meta = load_checkpoint(ckpt)
    assert [p["step"] for p in meta["val_history"]] == [2, 3]
    assert [p["raw"] for p in meta["val_history"]] == [r["val_raw"] for r in vals]


def test_cli_evento_start_marca_que_la_corrida_tiene_validacion(tmp_path):
    """El evento ``start`` dice si la corrida mide validación, para poder interpretar el archivo."""
    _, val_dir = _carpetas_imagenes_cli(tmp_path)
    cfg, _, log_path = _write_cli_config_val(tmp_path, val_dir=val_dir, num_steps=2)

    assert _load_train_main()(["--config", cfg, "--quiet"]) == 0

    start = _leer_jsonl(log_path)[0]
    assert start["event"] == "start"
    assert start["val"] is True


def test_cli_sin_ema_registra_la_ausencia_como_null(tmp_path):
    """Sin EMA el valor **falta explícitamente**: clave presente con ``null``, no omitida (4.2).

    La distinción es del consumidor del ``.jsonl``: una clave ausente significaría "este formato no
    tiene el campo" y una presente en ``null``, "esta corrida no mantenía sombra EMA". Se asevera
    sobre el texto serializado, no solo sobre el dict parseado, para que un filtro de ``None`` en el
    camino de escritura no pueda pasar desapercibido.
    """
    _, val_dir = _carpetas_imagenes_cli(tmp_path)
    cfg, _, log_path = _write_cli_config_val(
        tmp_path, val_dir=val_dir, num_steps=2, checkpoint_every=2  # sin ema_decay
    )

    assert _load_train_main()(["--config", cfg, "--quiet"]) == 0

    texto = log_path.read_text(encoding="utf-8")
    vals = [r for r in _leer_jsonl(log_path) if r.get("event") == "val"]
    assert vals
    for r in vals:
        assert "val_ema" in r and r["val_ema"] is None
        assert isinstance(r["train_fijo"], float)
    assert '"val_ema": null' in texto


# ------------------------------- resumen de consola y barra de progreso (5.4, 5.6)


def test_cli_informa_por_consola_el_ultimo_punto_medido(tmp_path, capsys):
    """Al terminar, la consola informa el **último** valor medido: crudo, EMA y examen de train (5.4).

    Sin ``--quiet``, así que la corrida lleva barra de progreso y los avisos por evaluación pasan
    por el escritor compatible con ella. Los valores del resumen se comparan contra el último punto
    de la serie del checkpoint, no contra una constante: el resumen tiene que informar **ese** punto
    y no el primero.
    """
    _, val_dir = _carpetas_imagenes_cli(tmp_path)
    cfg, ckpt, _ = _write_cli_config_val(
        tmp_path, val_dir=val_dir, num_steps=3, checkpoint_every=2, ema_decay=0.6
    )

    assert _load_train_main()(["--config", cfg]) == 0

    out = capsys.readouterr().out
    _, meta = load_checkpoint(ckpt)
    ultimo = meta["val_history"][-1]
    assert ultimo["step"] == 3
    # El resumen final (línea con la flecha, como el resto de los artefactos) trae el último punto.
    resumen = [l for l in out.splitlines() if "Validación" in l and "->" in l]
    assert len(resumen) == 1
    assert f"{ultimo['raw']:.6f}" in resumen[0]
    assert f"{ultimo['ema']:.6f}" in resumen[0]
    assert f"{ultimo['train']:.6f}" in resumen[0]
    assert f"{ultimo['step']}" in resumen[0]
    # …y hubo además un aviso por evaluación (dos evaluaciones: la cadencia y el último paso).
    avisos = [l for l in out.splitlines() if "Validación" in l and "->" not in l]
    assert len(avisos) == 2
    assert f"{meta['val_history'][0]['raw']:.6f}" in avisos[0]


def test_cli_los_avisos_de_validacion_van_por_el_escritor_de_la_barra(tmp_path, monkeypatch):
    """Los mensajes emitidos **durante** una evaluación usan ``tqdm.write``, no ``print`` (5.6).

    La evaluación cae dentro del paso, con la barra de progreso activa: un ``print`` la partiría en
    dos. El test espía el ``tqdm.write`` que importó el script —si el aviso saliera por ``print``, el
    espía no vería nada— y verifica que los dos avisos pasaron por ahí.
    """
    _, val_dir = _carpetas_imagenes_cli(tmp_path)
    cfg, _, _ = _write_cli_config_val(
        tmp_path, val_dir=val_dir, num_steps=3, checkpoint_every=2, ema_decay=0.6
    )
    module = _load_train_module()
    escritos: list[str] = []
    monkeypatch.setattr(module.tqdm, "write", lambda msg, *a, **k: escritos.append(str(msg)))

    assert module.main(["--config", cfg]) == 0

    avisos = [msg for msg in escritos if "Validación" in msg]
    assert len(avisos) == 2  # una por evaluación (cadencia 2 + último paso 3)
    assert all("paso" in msg and "val=" in msg for msg in avisos)


def test_cli_con_validacion_sin_train_log_informa_igual(tmp_path, capsys):
    """El ``.jsonl`` también es opt-in: sin ``out.train_log`` la validación informa por consola igual.

    Es el camino en que el sumidero de estados existe **solo** por la validación (no hay archivo
    donde escribir): si el envoltorio intentara serializar sin archivo abierto, la corrida
    reventaría en la primera evaluación en lugar de terminar en 0.
    """
    _, val_dir = _carpetas_imagenes_cli(tmp_path)
    cfg, ckpt, log_path = _write_cli_config_val(
        tmp_path, val_dir=val_dir, num_steps=2, checkpoint_every=2, with_train_log=False
    )

    assert _load_train_main()(["--config", cfg]) == 0

    assert not log_path.exists()  # no se declaró: no se crea nada
    out = capsys.readouterr().out
    assert "Validación ->" in out
    _, meta = load_checkpoint(ckpt)
    assert [p["step"] for p in meta["val_history"]] == [2]


# --------------------------------- sin data.val_root: todo igual que antes (5.5)


def test_cli_sin_val_root_no_escribe_ninguna_linea_de_validacion(tmp_path, capsys):
    """Sin la clave, el ``.jsonl``, la consola y el checkpoint salen como antes de la feature (5.5).

    Misma config de imágenes, misma cantidad de pasos: la única diferencia es que no se declara
    ``data.val_root``. No hay registro de validación, el evento ``start`` lo dice, la consola no
    menciona nada y el ``meta`` del checkpoint no gana la clave de la serie.
    """
    _carpetas_imagenes_cli(tmp_path)
    cfg, ckpt, log_path = _write_cli_config_val(
        tmp_path, val_dir=None, num_steps=3, checkpoint_every=2
    )

    assert _load_train_main()(["--config", cfg]) == 0

    lineas = _leer_jsonl(log_path)
    assert not [r for r in lineas if r.get("event") == "val"]
    assert lineas[0]["event"] == "start" and lineas[0]["val"] is False
    steps = [r for r in lineas if r.get("event") == "step"]
    assert steps and all(set(r) == _CLAVES_STEP_JSONL for r in steps)
    assert "Validación" not in capsys.readouterr().out
    _, meta = load_checkpoint(ckpt)
    assert "val_history" not in meta


# =============================================================================
# validation-loss task 7.2 — las curvas de validación en el gráfico de pérdida
# =============================================================================
#
# El gráfico se asevera por el **contenido de la figura** (las series dibujadas, sus etiquetas, sus
# coordenadas y la escala del eje Y), no por píxeles: un PNG no se puede interrogar y comparar bytes
# ataría la suite a la versión de matplotlib. Para poder inspeccionar la figura que arma
# ``save_loss_curve`` se neutraliza el ``plt.close`` del script (ver :func:`_capturar_figura`).
#
# Los dos invariantes que se protegen acá:
#
# - la serie dispersa se dibuja en los **pasos medidos**, no en índices consecutivos: la curva densa
#   está indexada por posición y la dispersa por paso, así que mezclar las dos convenciones
#   comprimiría los puntos de validación contra el origen del eje;
# - la leyenda distingue la curva **densa per-step** del **examen fijo de entrenamiento**: son dos
#   mediciones distintas del mismo conjunto (una con ``t`` re-sorteado por paso, la otra con el
#   examen congelado) y confundirlas invierte la lectura del gap. El par comparable es el examen
#   fijo de train contra la validación con pesos vivos.


def _val_history(pasos, raws, emas=None, trains=None):
    """Arma una serie dispersa de validación (lista de ``ValPoint``) para los tests del gráfico.

    ``emas``/``trains`` en ``None`` producen la ausencia **por serie completa** (una corrida sin
    sombra EMA, o sin fuente de examen de train), que es la forma en que la ausencia aparece de
    verdad; una lista con ``None`` intercalados cubre la ausencia parcial.
    """
    n = len(pasos)
    emas = [None] * n if emas is None else emas
    trains = [None] * n if trains is None else trains
    return [
        {"step": s, "raw": r, "ema": e, "train": tr}
        for s, r, e, tr in zip(pasos, raws, emas, trains)
    ]


def _espiar_figuras(monkeypatch) -> list:
    """Espía el ``plt.close`` y devuelve la lista donde se apilan las figuras cerradas.

    Se intercepta el ``plt.close(fig)`` para quedarse con la referencia a la figura y se lo deja
    correr igual (el código bajo test tiene que seguir cerrándola: si no, una corrida larga
    acumularía figuras). Cerrar una figura la saca del registro de pyplot pero **no** destruye el
    objeto ni sus artistas, así que sigue siendo interrogable: series, etiquetas, leyenda y escala
    del eje.

    El ``import matplotlib.pyplot`` del script es diferido, pero devuelve el mismo objeto módulo que
    este, así que el parche llega.
    """
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    capturadas: list = []
    close_real = plt.close

    def _close_espia(fig=None, *a, **k):
        capturadas.append(fig)
        return close_real(fig, *a, **k)

    monkeypatch.setattr(plt, "close", _close_espia)
    return capturadas


def _capturar_figura(module, monkeypatch, path, history, title="titulo", **kwargs):
    """Ejecuta ``save_loss_curve`` y devuelve la figura que armó, para inspeccionarla."""
    capturadas = _espiar_figuras(monkeypatch)
    module.save_loss_curve(path, history, title, **kwargs)
    assert len(capturadas) == 1, "save_loss_curve debe cerrar exactamente una figura"
    assert capturadas[0] is not None, "debe cerrar la figura que creó, no todas"
    return capturadas[0]


def _etiquetas(ax):
    """Etiquetas de las series dibujadas, en orden de dibujo."""
    return [linea.get_label() for linea in ax.get_lines()]


def _etiquetas_de_leyenda(ax):
    leyenda = ax.get_legend()
    return None if leyenda is None else [t.get_text() for t in leyenda.get_texts()]


# ---------------------------------- sin serie de validación: el gráfico de hoy (5.5)


def test_grafico_sin_validacion_dibuja_una_sola_serie_sin_leyenda(
    tmp_path, monkeypatch
):
    """Sin la serie dispersa, la figura es la de hoy: una serie, sin leyenda, y el PNG se escribe (5.5).

    El parámetro nuevo es opcional y el llamador viejo (``save_loss_curve(path, history, title)``)
    tiene que seguir andando sin cambios.
    """
    module = _load_train_module()
    png = tmp_path / "curva.png"

    fig = _capturar_figura(module, monkeypatch, png, [1.0, 0.8, 0.6, 0.5])

    ax = fig.axes[0]
    assert len(ax.get_lines()) == 1
    assert _etiquetas_de_leyenda(ax) is None  # hoy no hay leyenda y sin validación no aparece
    assert ax.get_yscale() == "log"           # historia positiva ⇒ escala log, como hoy
    assert png.exists()


def test_grafico_con_serie_vacia_es_identico_a_sin_serie(tmp_path, monkeypatch):
    """Una serie de validación **vacía** (corrida sin validación) no agrega curvas ni leyenda (5.5).

    Es el valor real que llega desde el resultado de la corrida cuando no hay ``data.val_root``:
    ``val_history`` es una lista vacía, no ``None``.
    """
    module = _load_train_module()

    fig = _capturar_figura(
        module, monkeypatch, tmp_path / "curva.png", [1.0, 0.8], val_history=[]
    )

    ax = fig.axes[0]
    assert len(ax.get_lines()) == 1
    assert _etiquetas_de_leyenda(ax) is None


# ------------------------------- las cuatro series y la leyenda (5.3, 5.7)


def test_grafico_con_validacion_dibuja_cuatro_series_con_leyenda(
    tmp_path, monkeypatch
):
    """Con las tres mediciones, la figura tiene **cuatro** series y leyenda con las cuatro (5.3, 5.7)."""
    module = _load_train_module()
    val = _val_history([4, 8, 10], [0.9, 0.7, 0.6], [0.85, 0.65, 0.55], [0.5, 0.4, 0.35])

    fig = _capturar_figura(
        module, monkeypatch, tmp_path / "curva.png", [1.0] * 10, val_history=val
    )

    ax = fig.axes[0]
    assert len(ax.get_lines()) == 4
    etiquetas = _etiquetas(ax)
    assert len(set(etiquetas)) == 4, f"etiquetas repetidas: {etiquetas}"
    assert _etiquetas_de_leyenda(ax) == etiquetas  # la leyenda lista las cuatro, en orden de dibujo


def test_grafico_la_leyenda_distingue_la_curva_densa_del_examen_fijo_de_train(
    tmp_path, monkeypatch
):
    """La leyenda no puede confundir la curva densa per-step con el examen fijo de train (5.3).

    Son dos mediciones distintas del **mismo** conjunto: la densa re-sortea ``t`` en cada paso, el
    examen los tiene congelados. Si la leyenda las nombra igual (o solo una de las dos dice "train"),
    el lector compara la curva equivocada contra la de validación y el gap se lee invertido.
    """
    module = _load_train_module()
    val = _val_history([2, 4], [0.9, 0.7], [0.85, 0.65], [0.5, 0.4])

    fig = _capturar_figura(
        module, monkeypatch, tmp_path / "curva.png", [1.0, 0.9, 0.8, 0.7], val_history=val
    )

    ax = fig.axes[0]
    etiquetas = _etiquetas(ax)
    densa = etiquetas[0]  # la primera serie dibujada es siempre la historia per-step
    de_train = [e for e in etiquetas if "train" in e.lower()]
    assert len(de_train) == 2, f"esperaba dos series de train (densa + examen), hay {de_train}"
    assert densa in de_train and densa != de_train[1]
    # Cada una se nombra por lo que la distingue: la densa por ser per-step, la otra por ser el examen.
    assert "per-step" in densa.lower()
    examen = [e for e in de_train if e != densa][0]
    assert "examen" in examen.lower()
    # Y las dos de validación se distinguen entre sí por los pesos (vivos vs EMA).
    de_val = [e for e in etiquetas if e not in de_train]
    assert len(de_val) == 2 and any("ema" in e.lower() for e in de_val)


# ------------------- la serie dispersa va en los pasos medidos, con marcadores (5.3)


def test_grafico_dibuja_la_serie_dispersa_en_los_pasos_medidos(
    tmp_path, monkeypatch
):
    """Las tres curvas dispersas se dibujan en los **pasos medidos**, no en índices consecutivos.

    Es el error más fácil de cometer: la curva densa está indexada por posición (``1..len``) y la
    dispersa por paso. Con cadencia 4 en una corrida de 10 pasos, usar índices pondría los puntos en
    1, 2 y 3 — todos apilados contra el origen — en lugar de 4, 8 y 10.
    """
    module = _load_train_module()
    pasos = [4, 8, 10]
    val = _val_history(pasos, [0.9, 0.7, 0.6], [0.85, 0.65, 0.55], [0.5, 0.4, 0.35])

    fig = _capturar_figura(
        module, monkeypatch, tmp_path / "curva.png", [1.0] * 10, val_history=val
    )

    ax = fig.axes[0]
    densa, *dispersas = ax.get_lines()
    assert list(densa.get_xdata()) == list(range(1, 11))  # la densa no cambió de convención
    assert len(dispersas) == 3
    for linea in dispersas:
        assert list(linea.get_xdata()) == pasos
    # Los valores llegan a la serie que les corresponde (no cruzados entre sí).
    por_etiqueta = {l.get_label(): list(l.get_ydata()) for l in dispersas}
    assert sorted(por_etiqueta.values()) == sorted(
        [[0.9, 0.7, 0.6], [0.85, 0.65, 0.55], [0.5, 0.4, 0.35]]
    )


def test_grafico_la_serie_dispersa_lleva_marcadores(tmp_path, monkeypatch):
    """Las curvas dispersas llevan línea **y** marcadores (la densa sigue sin marcador).

    Con una cadencia grande la serie puede tener dos o tres puntos: sin marcadores, una línea de dos
    puntos se lee como una tendencia continua en lugar de como dos mediciones.
    """
    module = _load_train_module()
    val = _val_history([5, 10], [0.9, 0.7], [0.85, 0.65], [0.5, 0.4])

    fig = _capturar_figura(
        module, monkeypatch, tmp_path / "curva.png", [1.0] * 10, val_history=val
    )

    densa, *dispersas = fig.axes[0].get_lines()
    assert densa.get_marker() in ("", "None", None)  # la densa no cambia
    for linea in dispersas:
        assert linea.get_marker() not in ("", "None", None)
        assert linea.get_linestyle() not in ("", "None", None)  # línea además del marcador


# ----------------------------------- series ausentes: no se dibujan y no revientan


def test_grafico_saltea_las_series_enteramente_ausentes(tmp_path, monkeypatch):
    """Una curva cuyos valores son todos ``None`` **no se dibuja**: ni crashea ni aplana en cero.

    Es el caso real de una corrida sin sombra EMA (``ema`` en ``None`` en todos los puntos) y sin
    fuente de examen de train. Dibujar ``None`` como 0 pondría una línea plana falsa en el piso del
    gráfico y, con escala log, tumbaría el eje entero.
    """
    module = _load_train_module()
    hist = [1.0] * 6

    # Sin EMA: quedan la densa, val con pesos vivos y el examen de train.
    solo_sin_ema = _val_history([3, 6], [0.9, 0.7], None, [0.5, 0.4])
    fig = _capturar_figura(module, monkeypatch, tmp_path / "a.png", hist, val_history=solo_sin_ema)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 3
    assert not any("ema" in e.lower() for e in _etiquetas(ax))

    # Sin EMA ni examen de train: solo la densa y la validación con pesos vivos.
    solo_raw = _val_history([3, 6], [0.9, 0.7])
    fig = _capturar_figura(module, monkeypatch, tmp_path / "b.png", hist, val_history=solo_raw)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 2
    assert _etiquetas_de_leyenda(ax) == _etiquetas(ax)  # con validación siempre hay leyenda


def test_grafico_omite_solo_los_puntos_ausentes_de_una_serie_parcial(
    tmp_path, monkeypatch
):
    """Con ausencias **parciales** se dibujan los puntos presentes, en sus pasos, sin rellenar."""
    module = _load_train_module()
    val = _val_history([2, 4, 6], [0.9, 0.8, 0.7], [None, 0.75, 0.65])

    fig = _capturar_figura(
        module, monkeypatch, tmp_path / "curva.png", [1.0] * 6, val_history=val
    )

    lineas = {l.get_label(): l for l in fig.axes[0].get_lines()}
    ema = [l for etiqueta, l in lineas.items() if "ema" in etiqueta.lower()]
    assert len(ema) == 1
    assert list(ema[0].get_xdata()) == [4, 6]
    assert list(ema[0].get_ydata()) == [0.75, 0.65]


# ------------------------------- la escala log mira TODAS las series graficadas


def test_grafico_escala_log_solo_si_todas_las_series_son_positivas(
    tmp_path, monkeypatch
):
    """La escala logarítmica se sujeta a **todas** las series, no solo a la de entrenamiento.

    Con la condición vieja (que solo mira la historia), una serie de validación con un valor <= 0
    entraría en un eje logarítmico y ese punto quedaría fuera del gráfico sin ningún aviso.
    """
    module = _load_train_module()
    hist = [1.0, 0.9, 0.8, 0.7]

    # Todas positivas ⇒ log.
    todas_positivas = _val_history([2, 4], [0.9, 0.7], [0.85, 0.65], [0.5, 0.4])
    fig = _capturar_figura(module, monkeypatch, tmp_path / "a.png", hist, val_history=todas_positivas)
    assert fig.axes[0].get_yscale() == "log"

    # Un cero en la validación con pesos vivos ⇒ escala lineal.
    con_cero = _val_history([2, 4], [0.9, 0.0], [0.85, 0.65], [0.5, 0.4])
    fig = _capturar_figura(module, monkeypatch, tmp_path / "b.png", hist, val_history=con_cero)
    assert fig.axes[0].get_yscale() == "linear"

    # Un negativo en el examen fijo de train (la serie que se agrega última) ⇒ lineal también.
    con_negativo = _val_history([2, 4], [0.9, 0.7], [0.85, 0.65], [0.5, -0.1])
    fig = _capturar_figura(module, monkeypatch, tmp_path / "c.png", hist, val_history=con_negativo)
    assert fig.axes[0].get_yscale() == "linear"

    # Y la condición vieja sigue valiendo: un cero en la historia ⇒ lineal aunque la val sea positiva.
    fig = _capturar_figura(
        module, monkeypatch, tmp_path / "d.png", [1.0, 0.0], val_history=todas_positivas
    )
    assert fig.axes[0].get_yscale() == "linear"


# --------------------------------------- el PNG por el camino del CLI (5.3, 5.5)


def test_cli_con_validacion_escribe_el_png_con_las_curvas(tmp_path):
    """El CLI enchufa la serie de la corrida al gráfico: el PNG sale y trae las cuatro series (5.3).

    Corrida real de punta a punta (imágenes sintéticas, 3 pasos, EMA activo) para que el gráfico se
    arme con la serie que **midió el loop** y no con una construida a mano; la figura se vuelve a
    generar con la serie del checkpoint para poder aseverar su contenido.
    """
    _, val_dir = _carpetas_imagenes_cli(tmp_path)
    cfg, ckpt, _ = _write_cli_config_val(
        tmp_path,
        val_dir=val_dir,
        num_steps=3,
        checkpoint_every=2,
        ema_decay=0.6,
        with_loss_curve=True,
    )

    assert _load_train_main()(["--config", cfg, "--quiet"]) == 0

    png = ckpt.parent / "vp_imgs_loss.png"
    assert png.exists() and png.stat().st_size > 0
    _, meta = load_checkpoint(ckpt)
    assert [p["step"] for p in meta["val_history"]] == [2, 3]
    assert all(p["ema"] is not None and p["train"] is not None for p in meta["val_history"])


def test_cli_sin_validacion_escribe_el_png_igual(tmp_path):
    """Sin ``data.val_root`` el gráfico se sigue generando, con una sola serie (5.5)."""
    _carpetas_imagenes_cli(tmp_path)
    cfg, ckpt, _ = _write_cli_config_val(
        tmp_path, val_dir=None, num_steps=3, checkpoint_every=2, with_loss_curve=True
    )

    assert _load_train_main()(["--config", cfg, "--quiet"]) == 0

    png = ckpt.parent / "vp_imgs_loss.png"
    assert png.exists() and png.stat().st_size > 0


# =============================================================================
# validation-loss task 8.1 — E2E del CLI: el par con y sin validación
# =============================================================================
#
# Las tasks 7.1 y 7.2 aseveran **cada superficie por separado** (una corrida para el ``.jsonl``, otra
# para la consola, otra para el PNG, y la figura del gráfico con series construidas a mano). Acá se
# cierra el **par**: dos corridas reales del CLI —una con ``data.val_root`` y otra sin la clave, con
# el mismo YAML, las mismas imágenes y los mismos pasos— y sobre cada una los **cuatro observables
# juntos**: código de salida, ``.jsonl``, PNG/figura y consola.
#
# Lo que agrega sobre 7.1/7.2:
#
# - la figura que se interroga es la que dibujó **el CLI durante la corrida**, no una que el test
#   armó llamando a ``save_loss_curve`` con una serie sintética;
# - las tres superficies informan **la misma medición**: el valor del ``.jsonl``, el que el resumen
#   de consola imprime y el que la curva del PNG dibuja salen del mismo ``ValPoint``. Un cableado que
#   graficara otra serie —o que informara el primer punto en vez del último— pasa los tests por
#   superficie y falla acá;
# - el **contraste** está en un solo lugar (:func:`test_e2e_el_par_difiere_solo_en_la_validacion`):
#   lo que no es de validación sale igual en las dos corridas y lo que sí, difiere. Así ninguna de
#   las dos mitades puede pasar por la razón equivocada.
#
# Cada corrida del CLI cuesta unos segundos de CPU, así que el par se corre **una vez** por módulo
# (fixture :func:`par_e2e`) y los tres tests leen los observables cacheados.


class _ObservablesCLI(NamedTuple):
    """Los cuatro observables de una corrida del CLI, para poder aseverarlos juntos."""

    codigo: int                 # código de salida de ``main``
    lineas: list[dict]          # el ``.jsonl`` entero, parseado
    vals: list[dict]            # solo los registros de validación (``event == "val"``)
    figura: object              # la figura de matplotlib que dibujó el CLI
    png: pathlib.Path           # la ruta del PNG que escribió
    consola: str                # todo lo que salió por stdout (resumen + avisos por evaluación)


def _correr_cli_e2e(tmp_path, monkeypatch, *, con_validacion: bool) -> _ObservablesCLI:
    """Corre ``scripts/train.py`` de punta a punta y devuelve sus observables.

    Una corrida real con **todas** las salidas declaradas (``.jsonl`` + PNG + checkpoint) y **sin**
    ``--quiet`` —barra de progreso y consola activas—, que es la forma en que el autor la corre a
    mano. ``con_validacion=False`` escribe el mismo YAML sin la clave ``data.val_root``: es la única
    diferencia entre las dos mitades del par (las dos carpetas de imágenes se sintetizan igual, así
    que la carpeta de validación existe también en la corrida que no la declara).

    La consola se captura con ``redirect_stdout`` en lugar del fixture ``capsys`` para que el helper
    sirva también desde un fixture de módulo (``capsys`` es de función). ``print`` y ``tqdm.write``
    resuelven ``sys.stdout`` en la llamada, así que los dos caen adentro.
    """
    _, val_dir = _carpetas_imagenes_cli(tmp_path)
    cfg, ckpt, log_path = _write_cli_config_val(
        tmp_path,
        val_dir=val_dir if con_validacion else None,
        num_steps=3,
        checkpoint_every=2,
        ema_decay=0.6,
        with_loss_curve=True,
    )
    module = _load_train_module()
    figuras = _espiar_figuras(monkeypatch)
    consola = io.StringIO()
    with contextlib.redirect_stdout(consola):
        codigo = module.main(["--config", cfg])

    assert len(figuras) == 1, "el CLI dibuja y cierra exactamente una curva de pérdida"
    lineas = _leer_jsonl(log_path)
    return _ObservablesCLI(
        codigo=codigo,
        lineas=lineas,
        vals=[r for r in lineas if r.get("event") == "val"],
        figura=figuras[0],
        png=ckpt.parent / "vp_imgs_loss.png",
        consola=consola.getvalue(),
    )


@pytest.fixture(scope="module")
def par_e2e(tmp_path_factory):
    """Las dos corridas del par (con y sin ``data.val_root``), cacheadas para los tres tests.

    Los tests solo **leen** los observables, así que compartirlos entre los tres es seguro y evita
    pagar cuatro corridas del CLI. ``pytest.MonkeyPatch.context()`` reemplaza al fixture
    ``monkeypatch``, que es de función y no se puede pedir desde un fixture de módulo.
    """
    with pytest.MonkeyPatch.context() as mp:
        con = _correr_cli_e2e(
            tmp_path_factory.mktemp("e2e_con_val"), mp, con_validacion=True
        )
    with pytest.MonkeyPatch.context() as mp:
        sin = _correr_cli_e2e(
            tmp_path_factory.mktemp("e2e_sin_val"), mp, con_validacion=False
        )
    return con, sin


# ------------------------------------- la mitad CON validación (5.1, 5.3, 5.4)


def test_e2e_cli_con_validacion_log_grafico_y_consola(par_e2e):
    """Con ``data.val_root``: código 0, líneas ``val`` en el ``.jsonl``, PNG con las cuatro curvas y
    el resumen en consola — los cuatro observables de **una misma** corrida (5.1, 5.3, 5.4).

    El cierre que no cubren 7.1 ni 7.2: las tres superficies informan **la misma medición**. La
    curva de validación con pesos vivos es la primera serie dispersa (orden de dibujo del CLI) y sus
    puntos son exactamente los pasos y los valores que quedaron en el ``.jsonl``; el resumen de
    consola trae el **último** de ellos.
    """
    con, _ = par_e2e
    assert con.codigo == 0

    # (1) Log: una línea de validación por evaluación — la cadencia (2) y el último paso (3).
    assert [r["step"] for r in con.vals] == [2, 3]

    # (2) PNG: el archivo existe y la figura que dibujó el CLI trae las cuatro series con leyenda.
    assert con.png.exists() and con.png.stat().st_size > 0
    ax = con.figura.axes[0]
    densa, *dispersas = ax.get_lines()
    assert len(dispersas) == 3
    assert _etiquetas_de_leyenda(ax) == _etiquetas(ax)

    # (3) Consola: el resumen final informa el ÚLTIMO punto medido, con el mismo formato del aviso.
    ultimo = con.vals[-1]
    assert (
        f"Validación -> paso {ultimo['step']}: val={ultimo['val_raw']:.6f}" in con.consola
    )

    # (4) La misma medición en las tres superficies: la curva del PNG contra los registros del log.
    assert list(densa.get_xdata()) == [1, 2, 3]  # la densa sigue indexada por posición
    assert list(dispersas[0].get_xdata()) == [r["step"] for r in con.vals]
    assert list(dispersas[0].get_ydata()) == [r["val_raw"] for r in con.vals]


# --------------------------------------------- la mitad SIN validación (5.5)


def test_e2e_cli_sin_validacion_log_grafico_y_consola(par_e2e):
    """Sin la clave: código 0, ni una línea ``val``, PNG de una sola serie y consola sin mención de
    validación — los mismos cuatro observables de la mitad de arriba (5.5).

    Mismo YAML, mismas imágenes sintéticas (la carpeta de validación existe igual) y los mismos
    pasos: la única diferencia es que ``data.val_root`` no se declara.
    """
    _, sin = par_e2e
    assert sin.codigo == 0
    assert sin.vals == []
    assert sin.png.exists() and sin.png.stat().st_size > 0
    ax = sin.figura.axes[0]
    assert len(ax.get_lines()) == 1
    assert _etiquetas_de_leyenda(ax) is None
    assert "Validación" not in sin.consola


# ------------------------------------------- el contraste, en un solo lugar (5.5, 6.5)


def test_e2e_el_par_difiere_solo_en_la_validacion(par_e2e):
    """Los dos observables lado a lado: lo que no es de validación sale igual; lo que sí, difiere.

    Es lo que impide que alguna de las dos mitades pase por la razón equivocada —una corrida que
    fallara temprano no escribiría registros de validación y satisfaría por accidente la mitad "sin
    validación"—, y la evidencia de que la clave es realmente **opt-in**: nada más cambia.
    """
    con, sin = par_e2e

    # Igual en las dos: cierran bien, registran los mismos pasos de entrenamiento y grafican.
    assert (con.codigo, sin.codigo) == (0, 0)
    pasos_con = [r["step"] for r in con.lineas if r.get("event") == "step"]
    pasos_sin = [r["step"] for r in sin.lineas if r.get("event") == "step"]
    assert pasos_con == pasos_sin == [1, 2, 3]
    assert con.png.exists() and sin.png.exists()

    # Distinto SOLO en la validación: registros, curvas, leyenda y resumen de consola.
    assert len(con.vals) == 2 and sin.vals == []
    assert len(con.figura.axes[0].get_lines()) == 4
    assert len(sin.figura.axes[0].get_lines()) == 1
    assert _etiquetas_de_leyenda(con.figura.axes[0]) is not None
    assert _etiquetas_de_leyenda(sin.figura.axes[0]) is None
    assert "Validación ->" in con.consola and "Validación" not in sin.consola
