"""Tests del módulo de entrenamiento (denoising score matching, `diffusion.training`).

Torch es dependencia dura del módulo, así que se hace `importorskip` al tope. Las corridas de
entrenamiento usan redes y datasets chicos para correr en CPU en segundos.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from diffusion.data_generation import infinite_bare, make_distribution
from diffusion.models import ScoreMLP, ScoreModel, make_model
from diffusion.sde import make_sde
from diffusion.training import (
    RunSpec,
    TrainConfig,
    TrainResult,
    TrainSnapshot,
    build_run,
    dsm_loss,
    load_checkpoint,
    load_config,
    sample_timesteps,
    save_checkpoint,
    train,
)

SDE_NAMES = ["vp", "ve", "sub_vp"]


def _small_net(sde) -> ScoreMLP:
    return ScoreMLP(data_dim=sde.data_dim, hidden_dim=64, num_blocks=2)


class _DummyScoreNet(torch.nn.Module):
    """Red de score mínima N-D: devuelve el estado escalado por un parámetro entrenable.

    Respeta el contrato ``(x, t) -> score`` con salida de la **misma shape** que ``x`` (funciona
    con cualquier rango, incluido ``(B, C, H, W)``) y expone un parámetro para que la pérdida
    tenga camino de gradiente. Sustituye a la U-Net para ejercitar ``dsm_loss`` sobre un batch
    tipo-imagen sin depender de la arquitectura convolucional.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.scale * x


def _data(dist, n=256, batch_size=64, *, shuffle=True):
    """Fuente infinita de tensores crudos que consume ``train`` (loader finito envuelto)."""
    return infinite_bare(dist.dataloader(n, batch_size, shuffle=shuffle))


def _tiny_config(**overrides) -> TrainConfig:
    base = dict(num_steps=4, seed=0)
    base.update(overrides)
    return TrainConfig(**base)


# ------------------------------------------------------------------ dsm_loss


@pytest.mark.parametrize("name", SDE_NAMES)
def test_dsm_loss_escalar_finito_con_gradiente(name):
    """La pérdida es un escalar finito y diferenciable, con gradientes finitos en la red.

    Parametrizado por las 3 SDEs.
    """
    sde = make_sde(name)
    net = _small_net(sde)
    x0 = torch.randn(32, 2)
    t = torch.rand(32) * sde.T

    loss = dsm_loss(net, sde, x0, t)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.all(torch.isfinite(g)) for g in grads)


@pytest.mark.parametrize("name", SDE_NAMES)
def test_dsm_loss_nd_safe_sobre_batch_tipo_imagen(name):
    """La pérdida DSM broadcastea el peso por muestra sobre un batch tipo-imagen (5.3).

    Con una SDE de forma de evento ``(3, 8, 8)`` el peso ``λ(t) = std²`` sale ``(B, 1, 1, 1)``
    (rank-matched vía ``_expand_t``) y broadcastea contra el error ``(B, 3, 8, 8)`` sin error de
    shape: la pérdida es un escalar finito y diferenciable, con gradiente en la red. Confirma
    que ``dsm_loss`` es **N-D-safe sin cambios** en ``losses.py`` (el peso ya queda rank-matched;
    un peso plano ``(B, 1)`` reventaría el broadcasting contra ``(B, 3, 8, 8)``).

    Se usa una red dummy que devuelve la shape del estado (no la U-Net) para aislar el
    broadcasting de la pérdida de la arquitectura. Parametrizado por las 3 SDEs.
    """
    event_shape = (3, 8, 8)
    sde = make_sde(name, data_dim=event_shape)
    net = _DummyScoreNet()
    x0 = torch.randn(4, *event_shape)
    t = torch.rand(4) * sde.T

    loss = dsm_loss(net, sde, x0, t)

    assert loss.ndim == 0  # escalar 0-dim: .mean() colapsó todas las dims de evento
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert grads  # la red dummy expone parámetros (camino de gradiente real)
    assert all(g is not None and torch.all(torch.isfinite(g)) for g in grads)


def test_dsm_loss_reproducible_con_generator():
    sde = make_sde("vp")
    net = _small_net(sde)
    x0 = torch.randn(16, 2)
    t = torch.rand(16)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    assert torch.equal(
        dsm_loss(net, sde, x0, t, generator=g1),
        dsm_loss(net, sde, x0, t, generator=g2),
    )


# ------------------------------------------------------------ sample_timesteps


def test_sample_timesteps_rango_shape_y_reproducibilidad():
    g1 = torch.Generator().manual_seed(0)
    t = sample_timesteps(1000, T=1.0, t_eps=1e-3, generator=g1)
    assert t.shape == (1000,)
    assert float(t.min()) >= 1e-3 - 1e-9
    assert float(t.max()) <= 1.0 + 1e-9

    g2 = torch.Generator().manual_seed(0)
    t2 = sample_timesteps(1000, T=1.0, t_eps=1e-3, generator=g2)
    assert torch.equal(t, t2)


def test_sample_timesteps_respeta_horizonte_distinto():
    g = torch.Generator().manual_seed(1)
    t = sample_timesteps(500, T=3.0, t_eps=0.5, generator=g)
    assert float(t.min()) >= 0.5 - 1e-9
    assert float(t.max()) <= 3.0 + 1e-9


# ------------------------------------------------------------------- train


@pytest.mark.parametrize("name", SDE_NAMES)
def test_train_usa_la_red_recibida_y_registra_data_dim(name):
    """train() usa la red que recibe (no construye ninguna) y registra el data_dim de la SDE."""
    sde = make_sde(name)
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)
    result = train(sde, net, _data(dist), _tiny_config(num_steps=4))

    assert isinstance(result, TrainResult)
    assert result.net is net  # usa la instancia recibida, no una nueva
    assert result.data_dim == sde.data_dim  # = sde.data_dim (fuente del checkpoint)
    assert result.sde_name == name
    assert len(result.history) == 4  # una entrada por paso (num_steps=4)
    assert all(math.isfinite(v) for v in result.history)


def test_history_es_per_step_e_independiente_de_log_every():
    """history guarda la pérdida de CADA paso: len == num_steps y NO depende de log_every.

    Regresión del bug en que ``log_every`` gobernaba la cadencia de registro (y el CLI la forzaba
    a num_steps//10 → ~10 puntos). Ahora es la serie completa; ``log_every`` solo afecta el print.
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    for log_every in (0, 2, 5):
        net = _small_net(sde)
        result = train(
            sde, net, _data(dist), TrainConfig(num_steps=7, log_every=log_every, seed=0)
        )
        assert len(result.history) == 7  # una por paso, sin importar log_every
        assert all(math.isfinite(v) for v in result.history)


def test_train_baja_la_perdida():
    """Smoke de aprendizaje: tras muchos pasos la pérdida final es menor que la inicial.

    Se compara la **tendencia** con medias de bloque (primer vs último quinto), no valores paso a
    paso: la pérdida per-step es ruidosa por el t aleatorio de cada paso.
    """
    sde = make_sde("vp")
    dist = make_distribution("mixture", 2, n_components=8, seed=0)
    torch.manual_seed(0)
    net = ScoreMLP(data_dim=sde.data_dim, hidden_dim=64, num_blocks=2)
    data = _data(dist, n=512, batch_size=128)
    result = train(sde, net, data, TrainConfig(num_steps=240, seed=0))

    assert len(result.history) == 240  # serie per-step completa
    assert all(math.isfinite(v) for v in result.history)
    # Tendencia sobre medias de bloque (la pérdida per-step es ruidosa por el t aleatorio).
    hist = result.history
    k = max(1, len(hist) // 5)
    assert sum(hist[-k:]) / k < sum(hist[:k]) / k


def test_train_reproducible_con_misma_seed():
    def run():
        torch.manual_seed(0)  # fija los pesos iniciales de la red (idénticos entre corridas)
        sde = make_sde("vp")
        net = _small_net(sde)
        dist = make_distribution("gaussian", 2, seed=1)
        data = _data(dist, n=256, batch_size=64)
        return train(sde, net, data, TrainConfig(num_steps=20, seed=7)).history

    assert run() == pytest.approx(run())


def test_train_con_grad_clip_corre():
    sde = make_sde("ve")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)
    result = train(sde, net, _data(dist), _tiny_config(num_steps=4, grad_clip=1.0))
    assert len(result.history) == 4
    assert all(math.isfinite(v) for v in result.history)


def test_trainconfig_acotado_al_loop():
    """TrainConfig lleva num_steps + campos del loop y NO acepta los campos removidos (3.1/3.2)."""
    cfg = TrainConfig(
        num_steps=10, lr=1e-3, t_eps=1e-3, grad_clip=1.0, seed=0, device="cpu", log_every=2
    )
    assert cfg.num_steps == 10
    for removed in (
        "epochs", "batch_size", "n_samples", "embed_dim", "hidden_dim", "num_blocks",
        "activation",
    ):
        assert not hasattr(cfg, removed)
    with pytest.raises(TypeError):
        TrainConfig(epochs=5)  # campo removido: ya no es aceptado


# -------------------------------------------------------------- checkpoints


def test_checkpoint_roundtrip(tmp_path):
    """Round-trip model-agnóstico (R5-c): ``save_checkpoint`` guarda ``state_dict`` + ``meta``
    con receta genérica ``model={name, kwargs}`` (sin campos de arquitectura hardcodeados);
    ``load_checkpoint`` devuelve ``(state_dict, meta)`` **sin** reconstruir; y el caller
    rearma la red con ``make_model`` + ``load_state_dict`` obteniendo la misma salida.

    Usa una arquitectura NO por defecto (``hidden_dim=64``, ``num_blocks=2``) a propósito: la
    reconstrucción debe respetar la receta, no caer en los defaults del constructor.
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)  # hidden_dim=64, num_blocks=2 (no son los defaults del ScoreMLP)
    result = train(sde, net, _data(dist), _tiny_config(num_steps=2))

    model_spec = {
        "name": "mlp",
        "kwargs": {"data_dim": sde.data_dim, "hidden_dim": 64, "num_blocks": 2},
    }
    path = tmp_path / "ckpt.pt"
    save_checkpoint(result, path, model_spec=model_spec)

    state_dict, meta = load_checkpoint(path)

    # load_checkpoint devuelve el state_dict crudo (no una red) + la metadata (5.2).
    assert isinstance(state_dict, dict)
    assert "output_proj.weight" in state_dict  # es el state_dict, no un objeto red
    # meta model-agnóstica: sde_name / data_dim / history / receta model (5.1).
    assert meta["sde_name"] == "vp"
    assert meta["data_dim"] == sde.data_dim == 2
    assert meta["history"] == pytest.approx(result.history)
    assert meta["model"] == model_spec
    # Sin hiperparámetros de arquitectura hardcodeados fuera de la receta genérica.
    assert set(meta) == {"sde_name", "data_dim", "history", "model"}

    # El caller reconstruye la red con make_model (receta {name, kwargs}) y le carga los pesos.
    recipe = meta["model"]
    net2 = make_model(recipe["name"], **recipe["kwargs"])
    net2.load_state_dict(state_dict)

    x = torch.randn(8, 2)
    t = torch.rand(8)
    result.net.eval()
    net2.eval()
    with torch.no_grad():
        assert torch.allclose(result.net(x, t), net2(x, t))


def test_save_checkpoint_sin_model_spec_omite_la_receta(tmp_path):
    """Sin ``model_spec`` el checkpoint es válido pero no lleva la clave ``model`` (5.1)."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    result = train(sde, _small_net(sde), _data(dist), _tiny_config(num_steps=2))

    path = tmp_path / "ckpt_sin_receta.pt"
    save_checkpoint(result, path)  # sin model_spec
    state_dict, meta = load_checkpoint(path)

    assert isinstance(state_dict, dict)
    assert "model" not in meta
    assert set(meta) == {"sde_name", "data_dim", "history"}


def test_checkpoint_roundtrip_conserva_forma_de_imagen(tmp_path):
    """La forma de evento de una SDE tipo-imagen viaja por la metadata del checkpoint (4.1) y
    ``make_sde`` la reconstruye desde ahí (4.2).

    El checkpoint transporta ``data_dim`` como el valor **crudo** de la SDE: un entero para el
    dato plano 2D o una **tupla** (forma de evento) para imágenes. Se arma un ``TrainResult`` a
    mano (sin correr entrenamiento): la identidad de la red es irrelevante acá —solo se verifica
    el round-trip de la forma en la meta y la reconstrucción de la SDE—. La generación
    end-to-end a ``(n, *E)`` se cubre en los tests de samplers (task 4.2).
    """
    event_shape = (3, 8, 8)
    sde = make_sde("vp", data_dim=event_shape)
    assert sde.data_dim == event_shape  # la SDE conserva el valor crudo (tupla)

    # Red mínima sin entrenar: solo actúa de portador del state_dict; no se reconstruye acá.
    net = ScoreMLP(data_dim=2, hidden_dim=8, num_blocks=1)
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=sde.data_dim)

    path = tmp_path / "ckpt_imagen.pt"
    save_checkpoint(result, path)
    _, meta = load_checkpoint(path)

    # 4.1: la forma de evento (tupla) sobrevive torch.save/torch.load sin perder el tipo.
    assert meta["data_dim"] == event_shape
    assert isinstance(meta["data_dim"], tuple)

    # 4.2: make_sde reconstruye la SDE con esa forma (data_shape normalizada).
    sde2 = make_sde(meta["sde_name"], data_dim=meta["data_dim"])
    assert sde2.data_shape == event_shape


# ------------------------------------------------ checkpointing intermedio


def test_trainconfig_checkpoint_every_default_cero():
    """El switch de checkpointing intermedio arranca apagado (0 = solo el checkpoint final)."""
    assert TrainConfig().checkpoint_every == 0


def test_train_sin_checkpoint_every_no_llama_callback():
    """Gate por defecto: con ``checkpoint_every=0`` el callback NO se invoca (sin regresión)."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    calls: list[str] = []
    result = train(
        sde,
        _small_net(sde),
        _data(dist),
        _tiny_config(num_steps=6),  # checkpoint_every=0 por defecto
        on_checkpoint=lambda tag, snap: calls.append(tag),
    )
    assert calls == []  # ni snapshots periódicos ni "best"
    assert result.history


def test_train_checkpoint_every_emite_solo_periodicos():
    """Con ``checkpoint_every=N`` el loop emite **solo** snapshots periódicos (ningún "best").

    Los tags periódicos son ``step{N:05d}`` en los múltiplos de ``checkpoint_every``, **excluido
    el último paso** (lo cubre el checkpoint final del caller). Con ``num_steps=9`` y
    ``checkpoint_every=3`` se esperan los pasos 3 y 6 (el 9 es el último y se omite).

    NB (excepción documentada, R6.2 / tarea 2.3 de la spec ``ema-weights``): este test **existía**
    y exigía el tag ``"best"``; se reescribió para verificar su **ausencia** cuando el mecanismo se
    retiró (R2.6, decisión del autor 27/07/2026: la selección por pérdida cruda per-step es ruidosa
    —``t`` aleatorio— y correlaciona mal con calidad de muestras, que es justo la razón de ser del
    EMA). Es la única excepción a la convención de no modificar tests existentes.
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    calls: list[tuple[str, TrainSnapshot]] = []
    net = _small_net(sde)
    result = train(
        sde,
        net,
        _data(dist),
        _tiny_config(num_steps=9, checkpoint_every=3),
        on_checkpoint=lambda tag, snap: calls.append((tag, snap)),
    )

    tags = [tag for tag, _ in calls]
    assert tags == ["step00003", "step00006"]  # múltiplos de 3, sin el último (9)
    assert "best" not in tags  # el mecanismo "best" ya no existe (R2.6)
    # Cada snapshot es un TrainSnapshot cuyo TrainResult apunta a la MISMA red que se entrena.
    assert all(
        isinstance(snap, TrainSnapshot) and snap.result.net is net for _, snap in calls
    )
    assert result.net is net


def test_train_checkpoints_intermedios_persisten_y_cargan(tmp_path):
    """El callback estilo-CLI escribe snapshots hermanos (…_stepNNNNN.pt) cargables.

    Reproduce el wiring de ``scripts/train.py``: deriva rutas del checkpoint base con
    ``Path.with_stem`` y persiste con ``save_checkpoint``. Verifica que los archivos existen, que
    el último paso NO genera snapshot periódico (lo cubre el final) y que ``load_checkpoint``
    recupera la metadata esperada.
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    base = tmp_path / "vp_gaussian.pt"
    model_spec = {
        "name": "mlp",
        "kwargs": {"data_dim": sde.data_dim, "hidden_dim": 64, "num_blocks": 2},
    }

    def on_checkpoint(tag, snap):
        # Nuevo contrato: snap es un TrainSnapshot; save_checkpoint consume su .result.
        save_checkpoint(snap.result, base.with_stem(f"{base.stem}_{tag}"), model_spec=model_spec)

    train(
        sde,
        _small_net(sde),
        _data(dist),
        _tiny_config(num_steps=4, checkpoint_every=2),
        on_checkpoint=on_checkpoint,
    )

    step_ckpt = tmp_path / "vp_gaussian_step00002.pt"
    best_ckpt = tmp_path / "vp_gaussian_best.pt"
    assert step_ckpt.exists()  # paso 2 (el 4 es el último: lo omite el periódico)
    assert not best_ckpt.exists()  # el tag "best" se retiró (R2.6): no hay artefacto que escribir
    assert not (tmp_path / "vp_gaussian_step00004.pt").exists()  # último paso excluido

    _, meta = load_checkpoint(step_ckpt)
    assert meta["sde_name"] == "vp"
    assert meta["data_dim"] == 2
    assert meta["model"] == model_spec


# ------------------------------------------------------------------- config


def test_build_run_desde_dict():
    """build_run arma (sde, model, data, config): la red por defecto es un MLP dimensionado
    desde la SDE y la data es un iterador infinito de tensores crudos con el batch_size pedido."""
    raw = {
        "sde": {"name": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "data": {
            "shape": "mixture", "dim": 2, "n_samples": 512, "batch_size": 128,
            "n_components": 8, "seed": 0,
        },
        "train": {"num_steps": 3, "lr": 1e-3, "seed": 0},
        "out": {"checkpoint": "models/x.pt", "loss_curve": "models/x.png"},
    }
    spec = build_run(raw)

    assert isinstance(spec, RunSpec)
    assert spec.sde.name == "vp"
    assert spec.config.num_steps == 3
    # Sin bloque 'model:' -> default MLP dimensionado desde el data_dim de la SDE.
    assert isinstance(spec.model, ScoreModel)
    assert isinstance(spec.model, ScoreMLP)
    assert spec.model.data_dim == spec.sde.data_dim
    # 'data' es un iterador infinito que yield-ea tensores crudos (B, data_dim).
    batch = next(iter(spec.data))
    assert batch.shape == (128, 2)  # batch_size del bloque 'data'
    assert spec.checkpoint.name == "x.pt"
    assert spec.loss_curve.name == "x.png"


def test_build_run_con_bloque_model_sobreescribe_el_default():
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1},
    }
    spec = build_run(raw)
    assert isinstance(spec.model, ScoreMLP)
    assert spec.model.hidden_dim == 32
    assert spec.model.num_blocks == 1
    assert spec.model.data_dim == spec.sde.data_dim  # el data_dim lo sigue aportando la SDE


def test_build_run_inyecta_data_dim_entero_en_el_modelo():
    """Gate de config (4.1): con una SDE de dato plano (``data_dim`` entero) el default MLP
    recibe ``data_dim`` inyectado desde la SDE (path 2D, sin regresión)."""
    raw = {
        "sde": {"name": "vp"},  # data_dim entero (2 por defecto)
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
    }
    spec = build_run(raw)

    assert isinstance(spec.sde.data_dim, int)
    assert spec.model_spec["kwargs"]["data_dim"] == spec.sde.data_dim == 2
    assert spec.model.data_dim == 2


def test_build_run_no_inyecta_forma_tupla_en_el_modelo():
    """Gate de config (4.1): con una SDE de forma de evento multidimensional (tupla) la forma
    NO se inyecta como hiperparámetro del modelo (la U-Net trae su propia config).

    Sin el gate, ``setdefault('data_dim', (3,8,8))`` inyectaría la tupla en la receta del MLP
    y ``ScoreMLP(data_dim=(3,8,8))`` reventaría (``int()`` sobre una tupla). El gate deja la
    forma fuera de la config del modelo; acá se usa el default MLP solo para ejercitar la rama
    del gate (en un run real de imágenes el bloque ``model:`` nombra la U-Net)."""
    raw = {
        "sde": {"name": "vp", "data_dim": (3, 8, 8)},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
    }
    spec = build_run(raw)

    assert spec.sde.data_dim == (3, 8, 8)
    assert not isinstance(spec.sde.data_dim, int)
    assert "data_dim" not in spec.model_spec["kwargs"]


def test_build_run_acepta_checkpoint_every():
    """El validador estricto de ``train:`` acepta el nuevo campo ``checkpoint_every`` y lo pasa
    al ``TrainConfig`` (el camino config-driven activa el checkpointing intermedio por YAML)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1, "checkpoint_every": 2},
    }
    spec = build_run(raw)
    assert spec.config.checkpoint_every == 2


def test_build_run_falla_sin_claves_obligatorias():
    with pytest.raises(ValueError):
        build_run({"data": {"shape": "gaussian"}})  # falta sde.name
    with pytest.raises(ValueError):
        build_run({"sde": {"name": "vp"}})  # falta data.shape


def test_build_run_rechaza_clave_desconocida():
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1, "lr_typo": 0.1},  # clave desconocida para TrainConfig
    }
    with pytest.raises(ValueError):
        build_run(raw)


def test_load_config_yaml_y_build_run(tmp_path):
    pytest.importorskip("yaml")
    text = (
        "sde:\n"
        "  name: vp\n"
        "data:\n"
        "  shape: gaussian\n"
        "  dim: 2\n"
        "  n_samples: 256\n"
        "  batch_size: 64\n"
        "train:\n"
        "  num_steps: 2\n"
        "  lr: 0.001\n"
    )
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")

    spec = build_run(load_config(path))
    assert spec.sde.name == "vp"
    assert isinstance(spec.model, ScoreMLP)
    assert spec.config.num_steps == 2
    batch = next(iter(spec.data))
    assert batch.shape == (64, 2)  # batch_size del bloque 'data'


def test_build_run_2d_entrena_end_to_end_tras_el_gate():
    """El camino config-driven 2D corre end-to-end tras el gate de configuración (4, 5.1).

    Compone ``build_run`` (que aplica el gate ``setdefault('data_dim', ...)`` solo para el
    entero 2D) con ``train`` sobre el ``RunSpec`` resultante: la red default (MLP dimensionado
    desde la SDE) entrena sin regresión y produce un ``history`` finito con ``data_dim`` entero.
    Ningún otro test corre ``train`` sobre la salida de ``build_run``; esto cierra el path
    config→entrenamiento del toy 2D."""
    raw = {
        "sde": {"name": "vp"},  # data_dim entero (2) -> el gate SÍ inyecta al MLP
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 2, "seed": 0},
    }
    spec = build_run(raw)

    result = train(spec.sde, spec.model, spec.data, spec.config)

    assert isinstance(result, TrainResult)
    assert result.net is spec.model  # entrena la red que armó build_run (default MLP)
    assert result.data_dim == spec.sde.data_dim == 2  # dato plano 2D, sin regresión
    assert result.history
    assert all(math.isfinite(v) for v in result.history)


# ---------------------------- time-embedding-scale: round-trip de la receta (task 3.1)


def test_roundtrip_receta_con_time_scale(tmp_path):
    """Round-trip de la escala temporal: config → build_run → checkpoint → make_model (3.1–3.3).

    Una corrida config-driven que declara ``time_scale`` en el bloque ``model:`` construye la
    red con ese valor (3.1), lo persiste dentro de la receta ``model_spec`` del checkpoint
    (3.2) y la reconstrucción desde la receta —sin el config original— produce una red con la
    **misma escala** usada al entrenar, a la que el ``state_dict`` guardado carga sin error y
    con salida idéntica (3.3). La aserción sobre el valor reconstruido fallaría si la receta
    dropeara el kwarg (no es solo un no-exception).
    """
    raw = {
        "sde": {"name": "vp"},
        "data": {
            "shape": "mixture", "dim": 2, "n_samples": 128, "batch_size": 64,
            "n_components": 2, "seed": 0,
        },
        "train": {"num_steps": 2, "seed": 0},
        "model": {
            "name": "mlp", "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1,
            "time_scale": 1000.0,
        },
    }
    spec = build_run(raw)

    # 3.1: la corrida config-driven construye la red con la escala declarada...
    assert spec.model.time_scale == pytest.approx(1000.0)
    # ...y la receta del checkpoint la transporta (reconstruible sin el YAML original).
    assert spec.model_spec["kwargs"]["time_scale"] == pytest.approx(1000.0)

    result = train(spec.sde, spec.model, spec.data, spec.config)
    path = tmp_path / "ckpt_time_scale.pt"
    save_checkpoint(result, path, model_spec=spec.model_spec)

    # 3.2: la escala sobrevive el round-trip por disco dentro de la receta.
    state_dict, meta = load_checkpoint(path)
    recipe = meta["model"]
    assert recipe["kwargs"]["time_scale"] == pytest.approx(1000.0)

    # 3.3: la reconstrucción desde la receta produce una red con la MISMA escala entrenada.
    net2 = make_model(recipe["name"], **recipe["kwargs"])
    assert net2.time_scale == pytest.approx(1000.0)
    net2.load_state_dict(state_dict)  # shapes intactas: la escala no agrega parámetros

    # La red reconstruida es funcionalmente la entrenada (misma salida sobre los mismos insumos).
    x = torch.randn(8, 2)
    t = torch.rand(8)
    result.net.eval()
    net2.eval()
    with torch.no_grad():
        assert torch.allclose(result.net(x, t), net2(x, t))


def test_receta_vieja_sin_time_scale_reconstruye_con_default(tmp_path):
    """Una receta anterior al cambio (sin la clave ``time_scale``) reconstruye con el default (3.4).

    Se arma un checkpoint cuya receta NO trae ``time_scale`` (formato pre-cambio) y se verifica
    que ``make_model`` reconstruye sin error una red con la escala default retrocompatible
    (``time_scale == 1.0``) y que el ``state_dict`` guardado carga sobre ella (shapes idénticas:
    la escala no es un parámetro entrenable).
    """
    kwargs = {"data_dim": 2, "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1}
    net = ScoreMLP(**kwargs)  # red pre-cambio: construida sin el kwarg
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=2)
    path = tmp_path / "ckpt_receta_vieja.pt"
    save_checkpoint(result, path, model_spec={"name": "mlp", "kwargs": dict(kwargs)})

    state_dict, meta = load_checkpoint(path)
    assert "time_scale" not in meta["model"]["kwargs"]  # receta vieja, sin la clave

    net2 = make_model(meta["model"]["name"], **meta["model"]["kwargs"])  # sin error (3.4)
    assert net2.time_scale == pytest.approx(1.0)  # default retrocompatible
    net2.load_state_dict(state_dict)  # los pesos viejos cargan sin conflicto de shapes

    x = torch.randn(8, 2)
    t = torch.rand(8)
    net.eval()
    net2.eval()
    with torch.no_grad():
        assert torch.allclose(net(x, t), net2(x, t))


def test_generate_from_checkpoint_reconstruye_con_time_scale(tmp_path):
    """La generación checkpoint-driven reconstruye la red con la escala de la receta (3.3).

    End-to-end en el seam real: un checkpoint cuya receta declara ``time_scale=1000.0`` pasa por
    ``generate_from_checkpoint`` (que reconstruye vía ``make_model(recipe)``) y produce muestras
    finitas con la shape esperada, sin error de reconstrucción ni de carga de pesos.
    """
    from diffusion.samplers import generate_from_checkpoint

    kwargs = {
        "data_dim": 2, "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1,
        "time_scale": 1000.0,
    }
    net = ScoreMLP(**kwargs)
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=2)
    path = tmp_path / "ckpt_scale_gen.pt"
    save_checkpoint(result, path, model_spec={"name": "mlp", "kwargs": dict(kwargs)})

    x0 = generate_from_checkpoint(path, "pf_ode", n_samples=8, n_steps=5, seed=0)

    assert x0.shape == (8, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


# ------------------- small-t-training-signal: muestreo de tiempos (task 1.1)

from diffusion.training import (  # noqa: E402
    TimeSampler,
    available_time_samplers,
    make_time_sampler,
)


def test_uniform_time_sampler_identidad_por_seed():
    """El default ``uniform`` reproduce EXACTAMENTE ``sample_timesteps`` con el mismo generator (1.2).

    Es el invariante de retrocompatibilidad: con la misma semilla, la variante uniforme del
    submódulo nuevo produce la misma secuencia de tiempos que la fórmula actual del loop
    (``torch.equal``, no ``allclose``: debe ser la misma llamada a ``torch.rand`` seguida de la
    misma transformación afín). Los pesos de la uniforme son ``None`` (likelihood ratio trivial).
    """
    sampler = make_time_sampler("uniform", T=1.0, t_eps=1e-4)
    assert isinstance(sampler, TimeSampler)

    g1 = torch.Generator().manual_seed(42)
    t, weights = sampler.sample(1000, generator=g1)

    g2 = torch.Generator().manual_seed(42)
    t_ref = sample_timesteps(1000, T=1.0, t_eps=1e-4, generator=g2)

    assert torch.equal(t, t_ref)  # identidad bit a bit, no aproximada
    assert weights is None  # la uniforme no corrige: no hay ratio que aplicar


def test_log_uniform_shape_dtype_y_rango():
    """La log-uniforme cumple el contrato de salida: ``(n,)`` float32 en ``[t_eps, T]``."""
    sampler = make_time_sampler("log_uniform", T=1.0, t_eps=1e-4)
    g = torch.Generator().manual_seed(0)
    t, weights = sampler.sample(4096, generator=g)

    assert t.shape == (4096,)
    assert t.dtype == torch.float32
    assert float(t.min()) >= 1e-4 - 1e-9
    assert float(t.max()) <= 1.0 + 1e-9
    assert weights is not None
    assert weights.shape == (4096,)
    assert weights.dtype == torch.float32


def test_log_uniform_concentra_masa_en_t_chico():
    """La distribución recomendada pone ≥ 30% de las muestras en ``[t_eps, 0.01]`` (1.5).

    Con ``t = t_eps·(T/t_eps)^u`` y ``u ~ U(0,1)``, la masa teórica de ``[t_eps, 0.01]`` es
    ``ln(0.01/t_eps)/ln(T/t_eps) = ln(100)/ln(10000) = 0.50`` para ``t_eps=1e-4, T=1``.

    Tolerancia estadística: la fracción empírica con ``n = 200_000`` tiene
    ``SE = sqrt(0.5·0.5/200_000) ≈ 0.0011``, así que el umbral 0.30 queda a ~180·SE del valor
    teórico 0.50 — el test no puede fallar por azar con la seed fija (ni con casi ninguna otra).
    """
    sampler = make_time_sampler("log_uniform", T=1.0, t_eps=1e-4)
    g = torch.Generator().manual_seed(0)
    t, _ = sampler.sample(200_000, generator=g)

    masa = float((t <= 0.01).float().mean())
    assert masa >= 0.30
    # Chequeo de coherencia con la fórmula (0.50 teórico), no parte del gate del requisito.
    assert masa == pytest.approx(0.50, abs=0.01)


def test_log_uniform_pesos_formula_y_media_uno():
    """Los pesos son el likelihood ratio contra la uniforme: fórmula puntual y ``E_q[w] = 1``.

    Fórmula: ``w(t) = p_unif(t)/q(t) = t·ln(T/t_eps)/(T − t_eps)`` — positiva en todo el rango.

    Tolerancia estadística de la media: con ``t`` log-uniforme en ``[1e-4, 1]``,
    ``Var[w] = ln(r)·(T²−t_eps²)/(2(T−t_eps)²) − 1 ≈ ln(10⁴)/2 − 1 ≈ 3.6`` (``sd ≈ 1.9``), así que
    con ``n = 200_000`` la media empírica tiene ``SE ≈ 1.9/√200_000 ≈ 0.0043``; la tolerancia
    ``abs=0.02`` es ≈ 4.7·SE — holgada para la seed fija sin ocultar un ratio mal calculado.
    """
    T, t_eps = 1.0, 1e-4
    sampler = make_time_sampler("log_uniform", T=T, t_eps=t_eps)
    g = torch.Generator().manual_seed(0)
    t, weights = sampler.sample(200_000, generator=g)

    assert torch.all(weights > 0)
    esperado = t * math.log(T / t_eps) / (T - t_eps)
    assert torch.allclose(weights, esperado)
    assert float(weights.mean()) == pytest.approx(1.0, abs=0.02)


def test_time_sampler_factory_y_validacion():
    """Factory fail-fast (1.1, 1.6): registry ordenado, nombre desconocido con opciones listadas,
    y ``t_eps`` fuera de ``(0, T)`` rechazado en construcción."""
    assert available_time_samplers() == ["log_uniform", "uniform"]

    with pytest.raises(ValueError, match="log_uniform.*uniform"):
        make_time_sampler("desconocido", T=1.0, t_eps=1e-4)

    for bad_eps in (0.0, -1e-4, 1.0, 2.0):  # fuera de (0, T) con T=1.0
        with pytest.raises(ValueError):
            make_time_sampler("uniform", T=1.0, t_eps=bad_eps)
        with pytest.raises(ValueError):
            make_time_sampler("log_uniform", T=1.0, t_eps=bad_eps)


@pytest.mark.parametrize("name", ["uniform", "log_uniform"])
def test_time_sampler_reproducible_con_generator(name):
    """Mismo generator → secuencia idéntica; seeds distintas → secuencias distintas."""
    sampler = make_time_sampler(name, T=1.0, t_eps=1e-4)

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    t1, w1 = sampler.sample(256, generator=g1)
    t2, w2 = sampler.sample(256, generator=g2)
    assert torch.equal(t1, t2)
    if w1 is not None:
        assert torch.equal(w1, w2)

    g3 = torch.Generator().manual_seed(8)
    t3, _ = sampler.sample(256, generator=g3)
    assert not torch.equal(t1, t3)


# ------------- small-t-training-signal: pesos en dsm_loss (task 1.2)


def _per_sample_dsm(net, sde, x0, t, *, generator):
    """Réplica por-muestra de la fórmula de ``dsm_loss`` con los mismos componentes públicos.

    Devuelve ``(B,)`` con ``mean_evento( λ(t) · (s_θ − s_real)² )`` por dato: su media de batch
    coincide con el escalar de ``dsm_loss`` (misma seed de ``generator``), lo que se verifica
    explícitamente en el test de equivalencia antes de usarla para estadística.
    """
    x_t, eps = sde.perturb(x0, t, generator=generator)
    score_real, weight = sde.score_target(x0, t, eps)
    with torch.no_grad():
        score_pred = net(x_t, t)
    err = weight * (score_pred - score_real).pow(2)
    return err.reshape(err.shape[0], -1).mean(dim=1)


def test_dsm_loss_retrocompatible_sin_pesos():
    """Sin ``sample_weights`` (omitido o ``None``) el valor es idéntico al actual (1.3, 5.1).

    Se compara bit a bit (``torch.equal``) contra la réplica manual de la fórmula vigente
    (``perturb`` → ``score_target`` → media pesada por λ) con la misma seed, y entre la llamada
    sin kwarg y la llamada con ``sample_weights=None`` explícito: ambas deben recorrer el mismo
    camino de código.
    """
    sde = make_sde("vp")
    net = _small_net(sde)
    x0 = torch.randn(32, 2, generator=torch.Generator().manual_seed(1))
    t = sample_timesteps(32, T=sde.T, t_eps=1e-3, generator=torch.Generator().manual_seed(2))

    # Réplica manual de la fórmula actual (el "valor previo") con la misma seed del kernel.
    g_ref = torch.Generator().manual_seed(3)
    x_t, eps = sde.perturb(x0, t, generator=g_ref)
    score_real, weight = sde.score_target(x0, t, eps)
    esperado = (weight * (net(x_t, t) - score_real).pow(2)).mean()

    g1 = torch.Generator().manual_seed(3)
    sin_kwarg = dsm_loss(net, sde, x0, t, generator=g1)
    g2 = torch.Generator().manual_seed(3)
    con_none = dsm_loss(net, sde, x0, t, generator=g2, sample_weights=None)

    assert torch.equal(sin_kwarg, esperado)
    assert torch.equal(con_none, sin_kwarg)


@pytest.mark.parametrize(
    ("data_shape", "make_net"),
    [
        ((32, 2), None),  # dato 2D: red MLP real
        ((8, 1, 4, 4), _DummyScoreNet),  # dato tipo imagen: dummy N-D-safe
    ],
    ids=["2d", "imagen"],
)
def test_dsm_loss_aplica_pesos_por_muestra(data_shape, make_net):
    """Los pesos por muestra escalan la pérdida y broadcastean sobre las dims de evento (1.3).

    - ``w = 1`` → idéntico a sin pesos; ``w = 2`` → exactamente 2×.
    - Shapes ``(B,)`` y ``(B, 1)`` producen el mismo resultado.
    - Funciona igual con dato 2D ``(B, 2)`` y con dato imagen ``(B, 1, 4, 4)`` (el peso se
      expande a ``(B, 1, …, 1)`` como hace ``λ(t)``).
    """
    B = data_shape[0]
    event_shape = data_shape[1:]
    sde = make_sde("vp", data_dim=event_shape[0] if len(event_shape) == 1 else event_shape)
    net = _small_net(sde) if make_net is None else make_net()
    x0 = torch.randn(*data_shape, generator=torch.Generator().manual_seed(5))
    t = sample_timesteps(B, T=sde.T, t_eps=1e-3, generator=torch.Generator().manual_seed(6))

    def loss_con(w):
        g = torch.Generator().manual_seed(7)
        return dsm_loss(net, sde, x0, t, generator=g, sample_weights=w)

    base = loss_con(None)
    assert torch.allclose(loss_con(torch.ones(B)), base)
    assert torch.allclose(loss_con(2.0 * torch.ones(B)), 2.0 * base)
    # (B,) y (B, 1) son shapes equivalentes para el mismo vector de pesos.
    w = torch.rand(B, generator=torch.Generator().manual_seed(8)) + 0.5
    assert torch.allclose(loss_con(w), loss_con(w.reshape(B, 1)))


def test_dsm_loss_equivalencia_en_esperanza_log_uniforme():
    """La log-uniforme corregida estima la MISMA pérdida esperada que la uniforme (1.4, 5.1).

    Es el test clave del importance sampling: con un modelo fijo chico (``ScoreMLP`` sin
    entrenar, seed fija) y la SDE VP, la pérdida esperada ``E[λ(t)·‖s_θ − s_real‖²]`` se estima
    por Monte Carlo de dos formas — (a) ``t`` uniforme sin pesos y (b) ``t`` log-uniforme con su
    likelihood ratio ``w(t)`` — y ambas deben coincidir dentro del error estadístico.

    Tolerancia (derivación): cada estimador es una media de ``n`` términos i.i.d. por-muestra
    (``ℓ_i`` para (a); ``w_i·ℓ_i`` para (b)), así que su error estándar es
    ``SE = sd_empírico/√n``. La diferencia de las dos medias tiene, si fueran independientes,
    ``SE_comb = sqrt(SE_a² + SE_b²)``; acá comparten el mismo ``x0`` (números aleatorios comunes,
    correlación ≥ 0), lo que solo puede REDUCIR la varianza de la diferencia, así que
    ``SE_comb`` es un techo conservador. Se exige ``|media_a − media_b| ≤ 3·SE_comb``: bajo la
    hipótesis nula (mismo objetivo en esperanza, R1.3) un desvío de 3·SE tiene probabilidad
    < 0.3%, y con seeds fijas el test es determinístico — no puede volverse flaky. Con
    ``n = 200_000`` la tolerancia resultante es ~1% de la pérdida, así que un ratio mal
    calculado (p. ej. sin el factor ``ln(T/t_eps)``, que sesga la media en ~7× — 1/ln(1000);
    incluso la falta total de corrección, sesgo ~4% de la media, cae fuera de la tolerancia).

    Las estadísticas por-muestra se calculan con ``_per_sample_dsm`` (réplica de la fórmula con
    los mismos componentes públicos ``perturb``/``score_target``), cuya coherencia con
    ``dsm_loss`` — sin y con pesos — se asserta antes de usarla.
    """
    torch.manual_seed(0)  # inicialización determinística de la red (modelo fijo)
    net = ScoreMLP(data_dim=2, embed_dim=16, hidden_dim=16, num_blocks=1)
    net.eval()
    sde = make_sde("vp")
    T, t_eps, n = sde.T, 1e-3, 200_000

    x0 = torch.randn(n, 2, generator=torch.Generator().manual_seed(10))

    # (a) t uniforme, sin pesos.
    t_unif, w_unif = make_time_sampler("uniform", T=T, t_eps=t_eps).sample(
        n, generator=torch.Generator().manual_seed(11)
    )
    assert w_unif is None
    por_muestra_a = _per_sample_dsm(net, sde, x0, t_unif, generator=torch.Generator().manual_seed(12))

    # (b) t log-uniforme, corregido por el likelihood ratio.
    t_log, w = make_time_sampler("log_uniform", T=T, t_eps=t_eps).sample(
        n, generator=torch.Generator().manual_seed(13)
    )
    por_muestra_b = w * _per_sample_dsm(net, sde, x0, t_log, generator=torch.Generator().manual_seed(14))

    # Coherencia de la réplica: su media de batch ES el escalar de dsm_loss (misma seed),
    # sin pesos y con pesos — así el test mide de verdad la pérdida de producción.
    assert torch.allclose(
        dsm_loss(net, sde, x0, t_unif, generator=torch.Generator().manual_seed(12)),
        por_muestra_a.mean(),
    )
    assert torch.allclose(
        dsm_loss(net, sde, x0, t_log, generator=torch.Generator().manual_seed(14), sample_weights=w),
        por_muestra_b.mean(),
    )

    # Estadística en float64 para que la comparación no dependa de la suma en float32.
    a = por_muestra_a.double()
    b = por_muestra_b.double()
    media_a, media_b = a.mean().item(), b.mean().item()
    se_a = (a.std() / math.sqrt(n)).item()
    se_b = (b.std() / math.sqrt(n)).item()
    se_comb = math.sqrt(se_a**2 + se_b**2)

    assert abs(media_a - media_b) <= 3.0 * se_comb, (
        f"media_unif={media_a:.6f}, media_log_unif_corregida={media_b:.6f}, "
        f"|dif|={abs(media_a - media_b):.6f} > 3·SE_comb={3.0 * se_comb:.6f}"
    )
    # La tolerancia debe ser chica frente a la pérdida (el test no pasa por manga ancha).
    assert 3.0 * se_comb < 0.05 * media_a


# ------------- small-t-training-signal: config y loop (task 1.3)


def test_trainconfig_time_sampling_default_uniforme():
    """``TrainConfig()`` sin el campo nuevo declara ``time_sampling="uniform"`` (1.2, 5.1).

    El default retrocompatible: quien no configura nada obtiene el muestreo uniforme actual.
    """
    cfg = TrainConfig()
    assert cfg.time_sampling == "uniform"
    # El campo es configurable como cualquier otro del dataclass.
    assert TrainConfig(time_sampling="log_uniform").time_sampling == "log_uniform"


def test_train_default_bit_identico_al_loop_previo():
    """Con el default (``time_sampling="uniform"``) ``train()`` es BIT-IDÉNTICO al loop previo (1.2).

    Prueba de equivalencia determinística elegida: una **réplica manual del código previo del
    loop** (``sample_timesteps`` + ``dsm_loss`` sin pesos, mismo orden de operaciones y mismo
    consumo del RNG) contra ``train()`` con la config default. Ambas corridas parten de los
    mismos pesos iniciales (``torch.manual_seed`` antes de construir la red) y la misma seed del
    loop, así que si el cambio alterara el stream del generator, la secuencia de ``t`` o la
    fórmula de la pérdida, la igualdad EXACTA (no aproximada) de historia y pesos finales
    fallaría. Nota: este test también pasa sobre el código anterior al campo — es el guard de
    regresión del invariante R1.2, no un test de feature.
    """

    def _fresh():
        torch.manual_seed(123)  # pesos iniciales idénticos entre ambas corridas
        sde = make_sde("vp")
        net = _small_net(sde)
        dist = make_distribution("gaussian", 2, seed=0)
        data = _data(dist, n=128, batch_size=32, shuffle=False)
        return sde, net, data, TrainConfig(num_steps=3, seed=0)

    # (a) train() con la config default (sin declarar time_sampling).
    sde, net_a, data, cfg = _fresh()
    hist_a = train(sde, net_a, data, cfg).history

    # (b) réplica manual del loop previo al cambio: misma inicialización del azar, un batch por
    # paso, t por sample_timesteps y dsm_loss SIN pesos (el código de trainer.py pre-1.3).
    sde, net_b, data, cfg = _fresh()
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    net_b = net_b.to(device)
    net_b.train()
    data_iter = iter(data)
    optimizer = torch.optim.Adam(net_b.parameters(), lr=cfg.lr)
    hist_b = []
    for _ in range(cfg.num_steps):
        x0 = next(data_iter).to(device)
        t = sample_timesteps(x0.shape[0], sde.T, cfg.t_eps, generator=generator, device=device)
        loss = dsm_loss(net_b, sde, x0, t, generator=generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        hist_b.append(loss.item())

    assert hist_a == hist_b  # igualdad exacta de floats por paso, no aproximada
    sd_a, sd_b = net_a.state_dict(), net_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"peso final distinto en {key}"


def test_train_rechaza_time_sampling_desconocido_antes_de_entrenar():
    """Una distribución desconocida revienta con ``ValueError`` ANTES de cualquier paso (1.6).

    El loop construye el sampler una única vez, fail-fast: si el nombre no está registrado, el
    error sale antes de consumir data — la fuente centinela convierte cualquier consumo prematuro
    en un ``AssertionError`` distinto del ``ValueError`` esperado.
    """

    class _DataProhibida:
        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("el loop consumió data antes de validar time_sampling")

    sde = make_sde("vp")
    net = _small_net(sde)
    cfg = TrainConfig(num_steps=2, seed=0, time_sampling="desconocida")
    with pytest.raises(ValueError, match="desconocida"):
        train(sde, net, _DataProhibida(), cfg)


def test_train_con_log_uniform_entrena_y_usa_la_distribucion():
    """Con ``time_sampling="log_uniform"`` el loop entrena con pérdidas finitas y la distribución
    declarada realmente gobierna el muestreo: misma seed, historia distinta a la del uniforme."""

    def _run(time_sampling):
        torch.manual_seed(123)
        sde = make_sde("vp")
        net = _small_net(sde)
        dist = make_distribution("gaussian", 2, seed=0)
        cfg = TrainConfig(num_steps=3, seed=0, time_sampling=time_sampling)
        return train(sde, net, _data(dist, n=128, batch_size=32, shuffle=False), cfg).history

    hist_log = _run("log_uniform")
    assert len(hist_log) == 3
    assert all(math.isfinite(v) for v in hist_log)
    assert hist_log != _run("uniform")  # el campo cambia de verdad el muestreo de t


def test_build_run_acepta_time_sampling_y_sigue_estricto(tmp_path):
    """El bloque ``train:`` del YAML acepta la clave nueva y la validación estricta sigue
    rechazando claves inválidas (1.7); la corrida config-driven con la distribución recomendada
    construye el ``TrainConfig`` con ella y entrena end-to-end (observable de la tarea)."""
    pytest.importorskip("yaml")
    text = (
        "sde:\n"
        "  name: vp\n"
        "data:\n"
        "  shape: gaussian\n"
        "  dim: 2\n"
        "  n_samples: 128\n"
        "  batch_size: 64\n"
        "  seed: 0\n"
        "train:\n"
        "  num_steps: 2\n"
        "  seed: 0\n"
        "  time_sampling: log_uniform\n"
    )
    path = tmp_path / "run_log_uniform.yaml"
    path.write_text(text, encoding="utf-8")

    spec = build_run(load_config(path))
    assert spec.config.time_sampling == "log_uniform"

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert len(result.history) == 2
    assert all(math.isfinite(v) for v in result.history)

    # La validación estricta de claves del bloque train: sigue funcionando con el campo nuevo
    # registrado (una clave inválida se rechaza nombrándola).
    bad = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1, "time_sampling_typo": "log_uniform"},
    }
    with pytest.raises(ValueError, match="time_sampling_typo"):
        build_run(bad)


# ------------- small-t-training-signal: activación config-driven de la parametrización (task 2.2)

from diffusion.models import EpsilonScoreWrapper  # noqa: E402  (sección append-only, task 2.2)


def test_build_run_con_score_parametrization_envuelve_y_persiste_la_clave():
    """Con ``score_parametrization: epsilon`` en el bloque ``model:``, ``build_run`` envuelve la
    red con :class:`EpsilonScoreWrapper` y registra la clave en la receta del checkpoint (2.4, 2.5).

    La receta transporta la clave como hermana de ``name``/``kwargs`` y los ``kwargs`` quedan
    PELADOS (sin la clave): el camino de reconstrucción es ``make_model(name, **kwargs)`` y
    después el wrap — si la clave se filtrara a los kwargs, la factory la rechazaría o la
    ignoraría en silencio.
    """
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 1, "seed": 0},
        "model": {
            "name": "mlp", "hidden_dim": 32, "num_blocks": 1,
            "score_parametrization": "epsilon",
        },
    }
    spec = build_run(raw)

    # La red del RunSpec ES el wrapper, con la red interna construida desde el bloque model:.
    assert isinstance(spec.model, EpsilonScoreWrapper)
    assert isinstance(spec.model.inner, ScoreMLP)
    assert spec.model.inner.hidden_dim == 32
    assert spec.model.inner.num_blocks == 1
    assert spec.model.inner.data_dim == spec.sde.data_dim  # el data_dim lo sigue aportando la SDE

    # La receta persiste la clave al lado de name/kwargs; los kwargs quedan pelados.
    assert spec.model_spec["score_parametrization"] == "epsilon"
    assert spec.model_spec["name"] == "mlp"
    assert "score_parametrization" not in spec.model_spec["kwargs"]


def test_build_run_sin_la_clave_pipeline_identico_al_actual():
    """Sin ``score_parametrization`` el pipeline es idéntico al actual: red pelada (sin wrapper)
    y receta con la forma vieja, sin la clave nueva (retrocompatibilidad, 2.4)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1},
    }
    spec = build_run(raw)

    assert not isinstance(spec.model, EpsilonScoreWrapper)
    assert isinstance(spec.model, ScoreMLP)
    assert "score_parametrization" not in spec.model_spec
    assert set(spec.model_spec) == {"name", "kwargs"}  # forma vieja de la receta, intacta


def test_build_run_rechaza_score_parametrization_desconocida():
    """Un valor desconocido de ``score_parametrization`` revienta con ``ValueError`` explícito
    que menciona el valor recibido (fail-fast antes de entrenar)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
        "model": {"name": "mlp", "score_parametrization": "otra"},
    }
    with pytest.raises(ValueError, match="otra"):
        build_run(raw)


def test_build_run_envuelve_con_la_sigma_de_la_sde_de_la_corrida():
    """El wrap usa la σ de la SDE de la corrida: el score del RunSpec es exactamente
    ``-inner(x, t) / clamp(σ_t, 1e-5)`` con la σ de ``spec.sde.marginal_prob`` (2.1 vía 2.4)."""
    raw = {
        "sde": {"name": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "data": {"shape": "gaussian", "dim": 2, "seed": 0},
        "train": {"num_steps": 1, "seed": 0},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1,
                  "score_parametrization": "epsilon"},
    }
    spec = build_run(raw)

    torch.manual_seed(0)
    x = torch.randn(16, 2)
    t = torch.rand(16).clamp_min(1e-3)
    spec.model.eval()
    with torch.no_grad():
        score = spec.model(x, t)
        std = spec.sde.marginal_prob(x, t)[1]
        esperado = -spec.model.inner(x, t) / std.clamp_min(1e-5)
    assert torch.allclose(score, esperado, rtol=0.0, atol=0.0)


def test_roundtrip_checkpoint_con_parametrizacion_epsilon(tmp_path):
    """Smoke del seam config→train→checkpoint con la parametrización activa (2.5, 5.1).

    ``train()`` consume el RunSpec envuelto (2 pasos, historia finita); el checkpoint guardado
    con ``model_spec`` transporta la clave en la meta, y la reconstrucción por la receta —
    ``make_model(name, **kwargs)`` + ``load_state_dict`` — funciona sobre la RED PELADA (el
    state_dict persistido no tiene prefijo de wrapper)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 2, "seed": 0},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1,
                  "score_parametrization": "epsilon"},
    }
    spec = build_run(raw)

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert result.net is spec.model  # entrena la red envuelta que armó build_run
    assert len(result.history) == 2
    assert all(math.isfinite(v) for v in result.history)

    path = tmp_path / "ckpt_epsilon.pt"
    save_checkpoint(result, path, model_spec=spec.model_spec)

    state_dict, meta = load_checkpoint(path)
    recipe = meta["model"]
    assert recipe["score_parametrization"] == "epsilon"
    assert "score_parametrization" not in recipe["kwargs"]

    # Reconstrucción por la receta: red pelada + state_dict sin prefijo de wrapper.
    net2 = make_model(recipe["name"], **recipe["kwargs"])
    net2.load_state_dict(state_dict)
    x = torch.randn(4, 2)
    t = torch.rand(4)
    spec.model.eval()
    net2.eval()
    with torch.no_grad():
        # La red reconstruida ES la interna entrenada (misma salida en unidades de ε).
        assert torch.allclose(net2(x, t), spec.model.inner(x, t))


# ------------- small-t-training-signal: reconstrucción al generar (task 2.3)


def test_generate_from_checkpoint_seam_e2e_con_ambos_ejes(tmp_path):
    """Seam e2e config→train→checkpoint→generate con ambos ejes activos (2.3, 2.5).

    Una corrida corta con ``time_sampling: log_uniform`` (eje 1) y ``score_parametrization:
    epsilon`` (eje 2) produce un checkpoint cuya generación checkpoint-driven devuelve muestras
    finitas con la shape esperada. Además se verifica que ``generate_from_checkpoint``
    reconstruyó la red ENVUELTA con la MISMA σ: como la función no expone la red, se compara
    por equivalencia — la reconstrucción a mano (``make_model`` + ``load_state_dict`` +
    :class:`EpsilonScoreWrapper` con la σ de la SDE default) integrada con el mismo sampler y
    la misma semilla produce ``torch.equal`` (mismo prior y mismo ruido por seed). Si generate
    dejara la red pelada (sin dividir por σ), el score consumido sería otro y las trayectorias
    divergirían. El contrato de los samplers no cambia: el wrapper entra como ``score_fn``
    igual que una red pelada (2.3).
    """
    from diffusion.samplers import generate_from_checkpoint, make_sampler

    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 2, "seed": 0, "time_sampling": "log_uniform"},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1,
                  "score_parametrization": "epsilon"},
    }
    spec = build_run(raw)
    result = train(spec.sde, spec.model, spec.data, spec.config)
    path = tmp_path / "ckpt_seam_ambos_ejes.pt"
    save_checkpoint(result, path, model_spec=spec.model_spec)

    x0 = generate_from_checkpoint(path, "euler", n_samples=8, n_steps=5, seed=0)

    assert x0.shape == (8, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))

    # Reconstrucción a mano del MISMO pipeline: red pelada por la receta + wrap con la σ de la
    # SDE default reconstruida desde la meta (mismo criterio que el lado de entrenamiento).
    state_dict, meta = load_checkpoint(path)
    recipe = meta["model"]
    assert recipe["score_parametrization"] == "epsilon"  # la clave viajó en el checkpoint
    net2 = make_model(recipe["name"], **recipe["kwargs"])
    net2.load_state_dict(state_dict)
    sde = make_sde(meta["sde_name"], data_dim=meta["data_dim"])
    wrapper = EpsilonScoreWrapper(net2, lambda x, t: sde.marginal_prob(x, t)[1])
    wrapper.eval()
    sampler = make_sampler("euler", sde, wrapper, n_steps=5)
    generator = torch.Generator()
    generator.manual_seed(0)
    esperado = sampler.sample(8, generator=generator)

    # Igualdad exacta: mismo prior, mismo ruido, mismo score ⇒ misma trayectoria bit a bit.
    assert torch.equal(x0, esperado)


def test_generate_from_checkpoint_receta_vieja_reconstruye_pelada(tmp_path):
    """Una receta anterior al cambio (sin ``score_parametrization``) genera SIN wrap (2.6).

    Checkpoint con la forma vieja de la receta (``{name, kwargs}``): la generación corre sin
    error y equivale bit a bit a integrar con la red PELADA reconstruida a mano — el
    comportamiento previo queda intacto (retrocompatibilidad sin warning ni wrap).
    """
    from diffusion.samplers import generate_from_checkpoint, make_sampler

    kwargs = {"data_dim": 2, "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1}
    net = ScoreMLP(**kwargs)
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=2)
    path = tmp_path / "ckpt_receta_vieja_gen.pt"
    save_checkpoint(result, path, model_spec={"name": "mlp", "kwargs": dict(kwargs)})

    x0 = generate_from_checkpoint(path, "euler", n_samples=8, n_steps=5, seed=0)

    assert x0.shape == (8, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))

    # Equivalencia contra la reconstrucción pelada: sin la clave no hay wrap.
    state_dict, meta = load_checkpoint(path)
    assert "score_parametrization" not in meta["model"]  # receta vieja, sin la clave
    net2 = make_model(meta["model"]["name"], **meta["model"]["kwargs"])
    net2.load_state_dict(state_dict)
    net2.eval()
    sde = make_sde(meta["sde_name"], data_dim=meta["data_dim"])
    sampler = make_sampler("euler", sde, net2, n_steps=5)
    generator = torch.Generator()
    generator.manual_seed(0)
    esperado = sampler.sample(8, generator=generator)

    assert torch.equal(x0, esperado)


def test_generate_from_checkpoint_rechaza_parametrizacion_desconocida(tmp_path):
    """Un valor desconocido de ``score_parametrization`` en la receta revienta con
    ``ValueError`` que menciona el valor recibido — mismo criterio de validación que
    ``build_run`` en el lado de entrenamiento (los dos call sites validan igual)."""
    from diffusion.samplers import generate_from_checkpoint

    kwargs = {"data_dim": 2, "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1}
    net = ScoreMLP(**kwargs)
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=2)
    path = tmp_path / "ckpt_parametrizacion_invalida.pt"
    save_checkpoint(
        result, path,
        model_spec={"name": "mlp", "kwargs": dict(kwargs), "score_parametrization": "otra"},
    )

    with pytest.raises(ValueError, match="otra"):
        generate_from_checkpoint(path, "euler", n_samples=4, n_steps=3, seed=0)


# ------------- ema-weights: la sombra EMA como pieza aislada (task 1)

from diffusion.training import EmaShadow  # noqa: E402  (sección append-only, task 1)


class _EmaToyModule(torch.nn.Module):
    """Módulo mínimo para ejercitar la matemática de la sombra sin optimizador.

    Un único parámetro entrenable (``w``, que el test pisa a mano paso a paso para simular la
    trayectoria de Adam) y un buffer constante (``denom``, que imita el del embedding
    sinusoidal): así se puede verificar de una vez la fórmula del EMA sobre los parámetros y la
    política de buffers (copiados del módulo vivo al publicar).
    """

    def __init__(self, w0: torch.Tensor) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(w0.clone())
        self.register_buffer("denom", torch.tensor([2.0, 4.0]))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.w.sum() * x


def _replica_ema(theta0: torch.Tensor, thetas: list[torch.Tensor], decay: float) -> torch.Tensor:
    """Réplica cerrada e independiente de la sombra (misma convención de contador).

    ``ema_0 = θ_0`` (los pesos con los que se construyó la sombra) y, para cada paso completado
    ``s = 1, 2, …`` (1-indexado), ``ema_s = d_s·ema_{s−1} + (1−d_s)·θ_s`` con la rampa de warmup
    ``d_s = min(d, (1+s)/(10+s))`` — la convención fijada en `research.md` ("Ponderación y warmup").
    """
    ema = theta0.clone()
    for s, theta in enumerate(thetas, start=1):
        d = min(decay, (1.0 + s) / (10.0 + s))
        ema = d * ema + (1.0 - d) * theta
    return ema


def _corrida_sintetica(net: _EmaToyModule, shadow: EmaShadow, thetas: list[torch.Tensor]) -> None:
    """Simula la corrida: pisa el parámetro con cada ``θ_s`` y actualiza la sombra (1-indexado)."""
    for s, theta in enumerate(thetas, start=1):
        with torch.no_grad():
            net.w.copy_(theta)
        shadow.update(s)


def _thetas(n: int, seed: int, dim: int = 3) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(dim, generator=g) for _ in range(n)]


def test_ema_shadow_matematica_replica_cerrada_en_warmup():
    """La sombra iguala la réplica cerrada mientras la rampa de warmup manda (1.3).

    Con ``d = 0.999`` y 6 pasos, ``(1+s)/(10+s) ≤ 7/16 = 0.4375 < d``: **todos** los pasos usan
    el peso de la rampa, nunca el decay configurado. La comparación es contra una réplica
    aritmética independiente (no contra la propia implementación) con tolerancia ajustada de
    forma cerrada en float32.
    """
    theta0 = torch.tensor([0.5, -1.0, 2.0])
    net = _EmaToyModule(theta0)
    shadow = EmaShadow(net, decay=0.999)
    thetas = _thetas(6, seed=11)

    _corrida_sintetica(net, shadow, thetas)

    esperado = _replica_ema(theta0, thetas, 0.999)
    assert torch.allclose(shadow.state_dict()["w"], esperado, rtol=0, atol=1e-6)
    # Sentinela: la sombra promedia de verdad (ni la última foto ni la inicial).
    assert not torch.allclose(esperado, thetas[-1], atol=1e-3)
    assert not torch.allclose(esperado, theta0, atol=1e-3)


def test_ema_shadow_matematica_replica_cerrada_cruza_el_warmup():
    """La sombra iguala la réplica cerrada cruzando warmup → régimen estacionario (1.3).

    Con ``d = 0.5`` la rampa alcanza el decay configurado en ``(1+s)/(10+s) = 0.5 ⇒ s = 8``: en
    12 pasos los primeros 7 usan la rampa (peso < 0.5) y del 8º en adelante manda ``d`` fijo. El
    decay chico es deliberado (distingue los dos regímenes en pocos pasos). Además se verifica
    que la rampa **está aplicada**: una réplica con ``d`` constante (sin warmup) da otro valor.
    """
    theta0 = torch.tensor([1.0, 0.0, -0.5])
    net = _EmaToyModule(theta0)
    shadow = EmaShadow(net, decay=0.5)
    thetas = _thetas(12, seed=5)

    # Los dos regímenes quedan efectivamente ejercitados por esta secuencia.
    assert (1.0 + 7) / (10.0 + 7) < 0.5  # paso 7: manda la rampa
    assert (1.0 + 8) / (10.0 + 8) == 0.5  # paso 8: la rampa toca el decay configurado

    _corrida_sintetica(net, shadow, thetas)

    esperado = _replica_ema(theta0, thetas, 0.5)
    assert torch.allclose(shadow.state_dict()["w"], esperado, rtol=0, atol=1e-6)

    sin_warmup = theta0.clone()
    for theta in thetas:
        sin_warmup = 0.5 * sin_warmup + 0.5 * theta
    assert not torch.allclose(esperado, sin_warmup, atol=1e-4)


@pytest.mark.parametrize(
    "bad", [0.0, 1.0, -0.5, 1.5, float("inf"), float("-inf"), float("nan")]
)
def test_ema_shadow_rechaza_decay_invalido(bad):
    """Decay no finito o fuera del intervalo abierto (0, 1) ⇒ ``ValueError`` con el valor (1.6).

    Fail-fast en construcción, antes de tocar la trayectoria de entrenamiento; el mensaje debe
    incluir el valor recibido (mismo criterio que la validación de ``t_eps`` del time sampler).
    """
    net = _EmaToyModule(torch.zeros(3))
    with pytest.raises(ValueError) as exc:
        EmaShadow(net, decay=bad)
    assert str(bad) in str(exc.value)


@pytest.mark.parametrize("ok", [1e-8, 0.5, 0.999, 1 - 1e-8])
def test_ema_shadow_acepta_decay_valido(ok):
    """Cualquier decay finito en el intervalo abierto (0, 1) se acepta (1.6, borde permitido)."""
    net = _EmaToyModule(torch.zeros(3))
    assert EmaShadow(net, decay=ok).decay == pytest.approx(ok)


@pytest.mark.parametrize("bad_step", [0, -1, -10])
def test_ema_shadow_rechaza_step_invalido(bad_step):
    """``update`` exige el contador 1-indexado de pasos completados (precondición ``step >= 1``)."""
    net = _EmaToyModule(torch.zeros(3))
    shadow = EmaShadow(net, decay=0.9)
    with pytest.raises(ValueError) as exc:
        shadow.update(bad_step)
    assert str(bad_step) in str(exc.value)


def _modulos_para_agnosticismo() -> dict[str, torch.nn.Module]:
    """Tres módulos de forma distinta para probar la agnosticismo de la sombra (1.5)."""
    sde = make_sde("vp")
    return {
        "mlp": ScoreMLP(data_dim=2, embed_dim=16, hidden_dim=32, num_blocks=1),
        "dummy": _DummyScoreNet(),
        "wrapper": EpsilonScoreWrapper(
            ScoreMLP(data_dim=2, embed_dim=16, hidden_dim=32, num_blocks=1),
            lambda x, t: sde.marginal_prob(x, t)[1],
        ),
    }


@pytest.mark.parametrize("clave", ["mlp", "dummy", "wrapper"])
def test_ema_shadow_claves_iguales_a_las_del_modulo(clave):
    """Las claves de ``state_dict()`` == las del ``state_dict`` del módulo, sea cual sea la red (1.5).

    Invariante de compatibilidad: lo que publica la sombra se carga en la red reconstruida con
    ``load_state_dict(strict=True)``. Se cubren tres formas de módulo sin ramificar por tipo:
    el ``ScoreMLP`` (parámetros + buffer del embedding), una red mínima de un solo parámetro y
    el :class:`EpsilonScoreWrapper` — cuyo ``state_dict`` delega al interno, así que las claves
    de la sombra son **de red pelada** (sin prefijo ``_net.``), lo que hace componer EMA con la
    parametrización ε sin traducción de claves.
    """
    net = _modulos_para_agnosticismo()[clave]
    shadow = EmaShadow(net, decay=0.9)
    shadow.update(1)

    foto = shadow.state_dict()
    assert set(foto) == set(net.state_dict())
    if clave == "wrapper":
        assert not any(k.startswith("_net.") for k in foto)  # claves de red pelada
    net.load_state_dict(foto)  # strict=True: encaja sin faltantes ni sobrantes


def test_ema_shadow_publica_parametros_ema_y_buffers_del_modulo_vivo():
    """``state_dict()`` = parámetros EMA + buffers copiados del módulo vivo (decisión de research)."""
    theta0 = torch.tensor([0.5, -1.0, 2.0])
    net = _EmaToyModule(theta0)
    shadow = EmaShadow(net, decay=0.9)
    _corrida_sintetica(net, shadow, _thetas(4, seed=3))

    with torch.no_grad():  # el buffer del vivo cambia ⇒ la foto lo refleja tal cual
        net.denom.copy_(torch.tensor([7.0, 9.0]))

    foto = shadow.state_dict()
    assert torch.equal(foto["denom"], net.denom)  # buffer: copia del vivo, no promediado
    assert not torch.equal(foto["w"], net.w.detach())  # parámetro: EMA, ≠ pesos crudos


def test_ema_shadow_state_dict_devuelve_clones():
    """La foto es un clon estable: ni la sombra ni el llamador se pisan entre sí.

    Postcondición del diseño (fotos estables para snapshots/checkpoints): actualizar la sombra
    después de tomar la foto no cambia la foto, y mutar la foto no cambia la sombra.
    """
    net = _EmaToyModule(torch.zeros(3))
    shadow = EmaShadow(net, decay=0.9)
    _corrida_sintetica(net, shadow, _thetas(3, seed=1))

    foto = shadow.state_dict()
    congelada = {k: v.clone() for k, v in foto.items()}

    _corrida_sintetica(net, shadow, _thetas(3, seed=2))  # la sombra sigue moviéndose
    assert all(torch.equal(foto[k], congelada[k]) for k in foto)
    assert not torch.equal(shadow.state_dict()["w"], foto["w"])

    antes = shadow.state_dict()["w"].clone()
    foto["w"].add_(100.0)  # mutar la foto no toca la sombra
    assert torch.equal(shadow.state_dict()["w"], antes)


def test_ema_shadow_es_pasiva_y_no_consume_rng():
    """La sombra observa: no escribe en el módulo ni consume RNG (1.4, 1.5).

    Invariante de pasividad a nivel de la pieza: tras construir, actualizar y publicar, los
    parámetros y buffers del módulo quedan **exactamente** como los dejó el "optimizador", y el
    estado del RNG global de torch no se movió (la actualización es aritmética determinística).
    """
    net = _EmaToyModule(torch.tensor([0.5, -1.0, 2.0]))
    thetas = _thetas(4, seed=8)

    torch.manual_seed(123)
    rng_antes = torch.get_rng_state()
    shadow = EmaShadow(net, decay=0.9)
    _corrida_sintetica(net, shadow, thetas)
    shadow.state_dict()

    assert torch.equal(net.w.detach(), thetas[-1])  # el módulo quedó donde lo dejó el paso
    assert torch.equal(net.denom, torch.tensor([2.0, 4.0]))
    assert torch.equal(torch.get_rng_state(), rng_antes)  # sin RNG


def test_ema_shadow_determinista_ante_la_misma_secuencia():
    """Misma secuencia de pesos ⇒ misma sombra, bit a bit (1.5: determinismo por seed)."""
    theta0 = torch.tensor([0.25, 0.5, -0.75])
    thetas = _thetas(7, seed=4)

    fotos = []
    for _ in range(2):
        net = _EmaToyModule(theta0)
        shadow = EmaShadow(net, decay=0.99)
        _corrida_sintetica(net, shadow, thetas)
        fotos.append(shadow.state_dict()["w"])
    assert torch.equal(fotos[0], fotos[1])


def test_ema_shadow_load_state_reanuda_la_sombra():
    """``load_state`` restaura la sombra: interrumpir y reanudar == corrida seguida (1.3, R3).

    Sombra A corre 9 pasos. Sombra B arranca de cero, corre 4, se le carga la foto de A al paso
    4 (como haría el resume desde el sidecar) y sigue los 5 pasos restantes con el mismo
    contador global: el resultado es idéntico bit a bit al de A.
    """
    theta0 = torch.tensor([1.5, -0.5, 0.25])
    thetas = _thetas(9, seed=2)

    net_a = _EmaToyModule(theta0)
    shadow_a = EmaShadow(net_a, decay=0.95)
    fotos_a = []
    for s, theta in enumerate(thetas, start=1):
        with torch.no_grad():
            net_a.w.copy_(theta)
        shadow_a.update(s)
        fotos_a.append(shadow_a.state_dict())

    net_b = _EmaToyModule(theta0)
    shadow_b = EmaShadow(net_b, decay=0.95)
    for s, theta in enumerate(thetas[:4], start=1):
        with torch.no_grad():
            net_b.w.copy_(theta)
        shadow_b.update(s)
    shadow_b.load_state(fotos_a[3])  # restauración desde la foto del paso 4
    for s, theta in enumerate(thetas[4:], start=5):
        with torch.no_grad():
            net_b.w.copy_(theta)
        shadow_b.update(s)

    assert torch.equal(shadow_b.state_dict()["w"], shadow_a.state_dict()["w"])


def test_ema_shadow_load_state_rechaza_foto_incompleta():
    """Una foto sin las claves de la sombra se rechaza con ``ValueError`` que las nombra.

    Nunca continuar con una sombra a medias en silencio (mismo criterio fail-fast del resto del
    módulo).
    """
    net = _EmaToyModule(torch.zeros(3))
    shadow = EmaShadow(net, decay=0.9)
    with pytest.raises(ValueError) as exc:
        shadow.load_state({"denom": torch.tensor([2.0, 4.0])})
    assert "w" in str(exc.value)


# ------------- ema-weights: orquestación de la sombra en el loop (task 2.1)


class _EmaModuloDeStateDictOpaco(torch.nn.Module):
    """Módulo cuyo ``state_dict`` **no reenvía** ``keep_vars`` al de ``nn.Module``.

    Caso patológico realista (un wrapper que reescribe ``state_dict`` a mano): los tensores que
    ve la sombra son copias detachadas, así que la identificación por identidad no encuentra
    ningún parámetro entrenable y la sombra quedaría **vacía** — promediando nada en silencio.
    Sirve para verificar el guard fail-fast de :class:`EmaShadow`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(3))

    def state_dict(self, *args, **kwargs):  # noqa: D102
        kwargs.pop("keep_vars", None)  # el bug que el guard tiene que detectar
        return super().state_dict(*args, **kwargs)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.w.sum() * x


class _DataProhibidaEma:
    """Fuente centinela: cualquier consumo de datos es un fallo del orden fail-fast del loop."""

    def __iter__(self):
        return self

    def __next__(self):
        raise AssertionError("el loop consumió data antes de validar ema_decay")


def _fresh_ema(**overrides):
    """Corrida chica reproducible: mismos pesos iniciales y misma fuente en cada invocación."""
    torch.manual_seed(321)  # pesos iniciales idénticos entre corridas comparadas
    sde = make_sde("vp")
    net = _small_net(sde)
    dist = make_distribution("gaussian", 2, seed=0)
    data = _data(dist, n=128, batch_size=32, shuffle=False)
    base = dict(num_steps=4, seed=0)
    base.update(overrides)
    return sde, net, data, TrainConfig(**base)


def _generador(cfg: TrainConfig) -> torch.Generator:
    """El ``generator`` del loop, explícito, para poder comparar el azar consumido."""
    g = torch.Generator(device=torch.device(cfg.device))
    g.manual_seed(cfg.seed)
    return g


def _replica_loop_crudo(sde, net, data, cfg, generator):
    """Réplica manual del loop **sin sombra**, bit a bit equivalente al camino default.

    Mismo orden de operaciones y mismo consumo del RNG que :func:`train` con
    ``time_sampling="uniform"`` (el invariante que ya sostiene el test del default de
    ``time_sampling``). El caller siembra el azar global antes de llamar.

    Returns:
        ``(history, thetas)``: la pérdida por paso y, en ``thetas[s-1]``, el clon del
        ``state_dict`` tras el paso completado ``s`` (1-indexado) — la trayectoria cruda de Adam
        que la sombra tiene que promediar.
    """
    device = torch.device(cfg.device)
    net = net.to(device)
    net.train()
    data_iter = iter(data)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    history: list[float] = []
    thetas: list[dict] = []
    for _ in range(cfg.num_steps):
        x0 = next(data_iter).to(device)
        t = sample_timesteps(x0.shape[0], sde.T, cfg.t_eps, generator=generator, device=device)
        loss = dsm_loss(net, sde, x0, t, generator=generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        thetas.append({k: v.detach().clone() for k, v in net.state_dict().items()})
    return history, thetas


def test_trainconfig_ema_decay_default_sin_ema():
    """``TrainConfig()`` sin el campo nuevo declara ``ema_decay=None`` — sin EMA (1.1).

    El default retrocompatible: quien no configura nada entrena exactamente como hasta hoy.
    """
    cfg = TrainConfig()
    assert cfg.ema_decay is None
    # El campo es configurable como cualquier otro del dataclass.
    assert TrainConfig(ema_decay=0.999).ema_decay == pytest.approx(0.999)


def test_train_default_sin_ema_bit_identico_al_loop_previo():
    """Con el default (``ema_decay=None``) ``train()`` es BIT-IDÉNTICO al loop previo (1.2).

    Mismo patrón de prueba que el default de ``time_sampling``: una réplica manual del loop (sin
    sombra, mismo orden de operaciones) contra ``train()`` con la config default. Se comparan las
    tres cosas que el criterio 1.2 nombra —historia (igualdad EXACTA de floats), pesos finales
    (``torch.equal``) y el azar consumido (RNG global de torch + estado del ``generator``)—: si la
    rama nueva tocara el stream o el orden, alguna fallaría. Además ``ema_state`` queda en
    ``None``: sin configurar nada, el resultado no gana contenido.
    """
    sde, net_a, data, cfg = _fresh_ema()
    g_a = _generador(cfg)
    res_a = train(sde, net_a, data, cfg, generator=g_a)
    rng_a = torch.get_rng_state()

    sde, net_b, data, cfg = _fresh_ema()
    g_b = _generador(cfg)
    torch.manual_seed(cfg.seed)  # lo que hace train() al arrancar desde cero
    hist_b, _ = _replica_loop_crudo(sde, net_b, data, cfg, g_b)
    rng_b = torch.get_rng_state()

    assert res_a.history == hist_b  # igualdad exacta de floats por paso, no aproximada
    assert res_a.ema_state is None
    sd_a, sd_b = net_a.state_dict(), net_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"peso final distinto en {key}"
    assert torch.equal(rng_a, rng_b)  # misma secuencia del RNG global
    assert torch.equal(g_a.get_state(), g_b.get_state())  # mismo stream del generator


def test_train_con_ema_promedia_la_trayectoria_cruda_de_adam():
    """Con EMA activo, ``result.ema_state`` == réplica cerrada sobre la trayectoria cruda (1.3).

    Verifica la **orquestación** (no la aritmética de la pieza, ya cubierta): el loop actualiza
    la sombra una vez por paso, después del ``optimizer.step()``, con el contador 1-indexado de
    pasos completados. La réplica manual del loop registra los ``θ_s`` de cada paso y
    :func:`_replica_ema` los promedia desde ``θ_0`` con ``d_s = min(d, (1+s)/(10+s))``. Un
    ``update`` de más/de menos, o un contador corrido, rompe la igualdad. Los buffers no se
    promedian (se copian del módulo vivo), así que se contrastan aparte.
    """
    decay = 0.6  # chico a propósito: la rampa y el techo se distinguen en pocos pasos
    sde, net_a, data, cfg = _fresh_ema(ema_decay=decay)
    res_a = train(sde, net_a, data, cfg, generator=_generador(cfg))

    sde, net_b, data, cfg = _fresh_ema(ema_decay=decay)
    theta0 = {k: v.detach().clone() for k, v in net_b.state_dict().items()}
    entrenables = set(dict(net_b.named_parameters()))
    g_b = _generador(cfg)
    torch.manual_seed(cfg.seed)
    _, thetas = _replica_loop_crudo(sde, net_b, data, cfg, g_b)

    assert res_a.ema_state is not None
    assert set(res_a.ema_state) == set(net_a.state_dict())  # foto publicable tal cual
    for key in entrenables:
        esperado = _replica_ema(theta0[key], [th[key] for th in thetas], decay)
        assert torch.allclose(res_a.ema_state[key], esperado, rtol=0, atol=1e-6), key
    for key in set(theta0) - entrenables:  # buffers: copia del vivo, no promedio
        assert torch.equal(res_a.ema_state[key], net_a.state_dict()[key]), key

    # Sentinela del contador: un ``update`` de más (o el contador corrido) daría otro valor, así
    # que la igualdad de arriba no es trivialmente satisfacible.
    clave = next(iter(entrenables))
    de_mas = _replica_ema(
        theta0[clave], [th[clave] for th in thetas] + [thetas[-1][clave]], decay
    )
    assert not torch.allclose(res_a.ema_state[clave], de_mas, rtol=0, atol=1e-6)


def test_train_con_ema_no_altera_la_trayectoria_de_optimizacion():
    """Pasividad: EMA on/off con la misma semilla ⇒ historia y pesos CRUDOS idénticos (1.4).

    La sombra es un observador: no escribe en la red, no toca el optimizador y no consume RNG.
    Dos corridas con la misma semilla —una con EMA y otra sin— deben terminar con la misma
    historia, los mismos pesos crudos (``torch.equal``) y el ``generator`` en el mismo estado. El
    cierre del test verifica que el EMA igual hizo algo: la foto difiere de los pesos crudos.
    """

    def _run(ema_decay):
        sde, net, data, cfg = _fresh_ema(ema_decay=ema_decay)
        g = _generador(cfg)
        res = train(sde, net, data, cfg, generator=g)
        crudos = {k: v.detach().clone() for k, v in net.state_dict().items()}
        return res, crudos, g.get_state()

    res_off, crudos_off, g_off = _run(None)
    res_on, crudos_on, g_on = _run(0.9)

    assert res_on.history == res_off.history  # igualdad exacta, no aproximada
    assert crudos_on.keys() == crudos_off.keys()
    for key in crudos_off:
        assert torch.equal(crudos_on[key], crudos_off[key]), f"peso crudo distinto en {key}"
    assert torch.equal(g_on, g_off)  # la sombra no consume RNG

    assert res_off.ema_state is None
    assert res_on.ema_state is not None
    assert any(not torch.equal(res_on.ema_state[k], crudos_on[k]) for k in crudos_on)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5, float("inf"), float("nan")])
def test_train_rechaza_ema_decay_invalido_antes_de_consumir_data(bad):
    """Un ``ema_decay`` inválido revienta con ``ValueError`` ANTES de tocar los datos (1.6).

    El loop construye la sombra una única vez, fail-fast, con el mismo criterio que el time
    sampler: la fuente centinela convierte cualquier consumo prematuro en un ``AssertionError``
    distinto del ``ValueError`` esperado, y el mensaje debe nombrar el valor recibido.
    """
    sde = make_sde("vp")
    net = _small_net(sde)
    cfg = TrainConfig(num_steps=2, seed=0, ema_decay=bad)
    with pytest.raises(ValueError) as exc:
        train(sde, net, _DataProhibidaEma(), cfg)
    assert str(bad) in str(exc.value)


def test_train_con_ema_snapshots_llevan_fotos_clonadas_e_independientes():
    """La foto clonada viaja en el resultado y en cada snapshot intermedio (observable de 2.1).

    ``_snapshot`` fotografía vía ``_result``, así que los checkpoints intermedios llevan la
    sombra **de su momento**: dos snapshots sucesivos difieren y son objetos independientes
    (mutar uno no toca al otro ni al resultado final). Es lo que hace segura la publicación.
    """
    sde, net, data, cfg = _fresh_ema(num_steps=6, checkpoint_every=2, ema_decay=0.9)
    fotos: list[tuple[str, dict]] = []

    def _cb(tag, snapshot):
        if tag.startswith("step"):
            fotos.append((tag, snapshot.result.ema_state))

    res = train(sde, net, data, cfg, on_checkpoint=_cb, generator=_generador(cfg))

    assert [tag for tag, _ in fotos] == ["step00002", "step00004"]
    (_, f2), (_, f4) = fotos
    assert all(estado is not None for _, estado in fotos)
    # La sombra avanzó entre snapshots y sigue avanzando hasta el final de la corrida.
    assert any(not torch.equal(f2[k], f4[k]) for k in f2)
    assert any(not torch.equal(f4[k], res.ema_state[k]) for k in f4)

    # Independencia: mutar la foto del paso 2 no altera la del 4 ni la final (clones profundos).
    congelada_f4 = {k: v.clone() for k, v in f4.items()}
    congelada_fin = {k: v.clone() for k, v in res.ema_state.items()}
    for v in f2.values():
        v.add_(100.0)
    assert all(torch.equal(f4[k], congelada_f4[k]) for k in f4)
    assert all(torch.equal(res.ema_state[k], congelada_fin[k]) for k in res.ema_state)


def test_build_run_acepta_ema_decay_y_sigue_estricto():
    """El bloque ``train:`` acepta ``ema_decay`` y la validación estricta sigue viva (4.1).

    ``TrainConfig`` se valida por **introspección de sus fields**, así que el campo nuevo queda
    disponible desde la config sin tocar ``build_run``; una corrida config-driven corta entrena
    con él (observable de la tarea) y una clave inválida se sigue rechazando nombrándola. Se
    ejercita con un ``dict`` —el mismo que produce ``load_config``— para no depender de PyYAML
    (el camino de archivo va en el test hermano).
    """
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 2, "seed": 0, "ema_decay": 0.999},
    }
    spec = build_run(raw)
    assert spec.config.ema_decay == pytest.approx(0.999)

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert len(result.history) == 2
    assert all(math.isfinite(v) for v in result.history)
    assert result.ema_state is not None
    assert set(result.ema_state) == set(spec.model.state_dict())

    bad = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1, "ema_decay_typo": 0.999},
    }
    with pytest.raises(ValueError, match="ema_decay_typo"):
        build_run(bad)


def test_build_run_desde_yaml_con_ema_decay(tmp_path):
    """El mismo camino pero desde un archivo YAML real: ``train.ema_decay`` llega al loop (4.1).

    Complementa al test hermano cubriendo el front-end de archivo (``load_config``); se skipea si
    PyYAML no está instalado, como el resto de los tests del camino YAML de la suite.
    """
    pytest.importorskip("yaml")
    text = (
        "sde:\n"
        "  name: vp\n"
        "data:\n"
        "  shape: gaussian\n"
        "  dim: 2\n"
        "  n_samples: 128\n"
        "  batch_size: 64\n"
        "  seed: 0\n"
        "train:\n"
        "  num_steps: 2\n"
        "  seed: 0\n"
        "  ema_decay: 0.999\n"
    )
    path = tmp_path / "run_ema.yaml"
    path.write_text(text, encoding="utf-8")

    spec = build_run(load_config(path))
    assert spec.config.ema_decay == pytest.approx(0.999)

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert len(result.history) == 2
    assert result.ema_state is not None


def test_ema_shadow_rechaza_modulo_sin_tensores_rastreables():
    """Un módulo cuyo ``state_dict`` no reenvía ``keep_vars`` se rechaza en construcción.

    Guard fail-fast: sin él la sombra quedaría vacía y **promediaría nada en silencio**,
    publicando pesos crudos disfrazados de EMA. El mensaje nombra el problema (``keep_vars``) en
    lugar de dejar pasar una corrida con EMA inerte.
    """
    with pytest.raises(ValueError) as exc:
        EmaShadow(_EmaModuloDeStateDictOpaco(), decay=0.9)
    assert "keep_vars" in str(exc.value)


# ------------- ema-weights: publicación en checkpoints + hermano de crudos (task 2.2)

from diffusion.training import discover_snapshots  # noqa: E402  (sección append-only, task 2.2)

#: Receta de la red chica de la suite (``_small_net``): lo que el caller pasa como ``model_spec``.
_RECETA_CHICA = {"name": "mlp", "kwargs": {"data_dim": 2, "hidden_dim": 64, "num_blocks": 2}}


def _corrida_publicable(**overrides):
    """Corrida corta reproducible + su receta, lista para publicar en un checkpoint.

    Reusa los helpers de la sección anterior (``_fresh_ema``/``_generador``: misma red chica,
    mismos pesos iniciales y mismo azar en cada invocación).

    Returns:
        ``(result, model_spec, crudos)``: el :class:`TrainResult`, la receta genérica de la red y
        un clon del ``state_dict`` **crudo** final (para contrastarlo contra lo publicado).
    """
    sde, net, data, cfg = _fresh_ema(**overrides)
    result = train(sde, net, data, cfg, generator=_generador(cfg))
    crudos = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return result, {"name": "mlp", "kwargs": dict(_RECETA_CHICA["kwargs"])}, crudos


def test_save_checkpoint_con_ema_publica_la_sombra_y_marca_el_decay(tmp_path):
    """Con EMA activo el checkpoint publica la sombra como su ``state_dict`` (2.1) + marca (2.3).

    Un solo punto de publicación: ``save_checkpoint`` toma ``result.ema_state`` en lugar de los
    pesos vivos. Se verifica lo tres cosas que pide el criterio: (a) el ``model_state`` es la foto
    EMA **tensor a tensor** (``torch.equal``) y difiere de los crudos —si publicara los crudos, la
    corrida con EMA no aportaría nada—; (b) el **formato no cambia** (mismas claves de blob y de
    meta que hoy, más la marca) y la receta sigue reconstruyendo la red; (c) la meta registra la
    trazabilidad ``ema.decay``.
    """
    decay = 0.6
    result, model_spec, crudos = _corrida_publicable(num_steps=6, ema_decay=decay)

    path = tmp_path / "vp_pub.pt"
    save_checkpoint(result, path, model_spec=model_spec)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert set(blob) == {"model_state", "meta"}  # formato intacto (2.1)

    state_dict, meta = load_checkpoint(path)
    assert state_dict.keys() == crudos.keys()
    for key in result.ema_state:
        assert torch.equal(state_dict[key], result.ema_state[key]), f"no publicó la sombra en {key}"
    assert any(not torch.equal(state_dict[k], crudos[k]) for k in crudos)  # ≠ pesos crudos

    # Marca de trazabilidad (2.3): quién publicó y con qué decay.
    assert meta["ema"] == {"decay": pytest.approx(decay)}
    # El resto de la meta es la de siempre (2.1: receta incluida, sin campos nuevos de más).
    assert set(meta) == {"sde_name", "data_dim", "history", "model", "ema"}
    assert meta["sde_name"] == "vp"
    assert meta["data_dim"] == 2
    assert meta["history"] == pytest.approx(result.history)
    assert meta["model"] == model_spec

    # La receta sigue reconstruyendo la red y los pesos EMA cargan con strict=True.
    net2 = make_model(meta["model"]["name"], **meta["model"]["kwargs"])
    net2.load_state_dict(state_dict)


def test_save_checkpoint_sin_ema_contenido_identico_al_actual(tmp_path):
    """Sin EMA el checkpoint es **idéntico al actual** en contenido, sin la marca nueva (2.5).

    Retrocompatibilidad: ``ema_decay=None`` ⇒ ``model_state`` == los pesos vivos ``torch.equal``
    tensor a tensor y meta con exactamente las claves de antes (la ausencia de ``ema`` es lo que
    un consumidor lee como "pesos crudos", 2.3).
    """
    result, model_spec, crudos = _corrida_publicable(num_steps=6)  # ema_decay=None (default)
    assert result.ema_state is None

    path = tmp_path / "vp_sin_ema.pt"
    save_checkpoint(result, path, model_spec=model_spec)

    state_dict, meta = load_checkpoint(path)
    assert state_dict.keys() == crudos.keys()
    for key in crudos:
        assert torch.equal(state_dict[key], crudos[key]), f"peso publicado distinto en {key}"
    assert "ema" not in meta
    assert set(meta) == {"sde_name", "data_dim", "history", "model"}


def test_checkpoints_intermedios_publican_la_sombra_de_su_momento(tmp_path):
    """Los checkpoints intermedios periódicos publican también la sombra de su momento (2.2).

    Reproduce el wiring del CLI (``on_checkpoint`` → ``save_checkpoint`` sobre rutas hermanas):
    como los intermedios pasan por el **mismo** punto de publicación, ganan el EMA sin cambio
    propio. Dos snapshots sucesivos **difieren** (la foto es un clonado profundo del momento, no
    una referencia viva que terminaría igual al final de la corrida) y ambos llevan la marca.
    """
    decay = 0.6
    sde, net, data, cfg = _fresh_ema(num_steps=6, checkpoint_every=2, ema_decay=decay)
    base = tmp_path / "vp_inter.pt"
    fotos: dict[str, dict] = {}

    def on_checkpoint(tag, snapshot):
        save_checkpoint(
            snapshot.result, base.with_stem(f"{base.stem}_{tag}"), model_spec=dict(_RECETA_CHICA)
        )
        if tag.startswith("step"):
            fotos[tag] = {k: v.clone() for k, v in snapshot.result.ema_state.items()}

    train(sde, net, data, cfg, on_checkpoint=on_checkpoint, generator=_generador(cfg))

    assert sorted(fotos) == ["step00002", "step00004"]
    publicados = {}
    for tag in ("step00002", "step00004"):
        state_dict, meta = load_checkpoint(tmp_path / f"vp_inter_{tag}.pt")
        assert meta["ema"] == {"decay": pytest.approx(decay)}  # marca en los intermedios (2.3)
        for key in fotos[tag]:
            assert torch.equal(state_dict[key], fotos[tag][key]), f"{tag}: no publicó su sombra"
        publicados[tag] = state_dict

    # La sombra avanzó entre snapshots: los dos artefactos no son el mismo estado.
    assert any(
        not torch.equal(publicados["step00002"][k], publicados["step00004"][k])
        for k in publicados["step00002"]
    )


def test_save_checkpoint_hermano_de_crudos_publica_los_pesos_crudos(tmp_path):
    """``raw_sibling=True`` escribe además ``{stem}_raw.pt`` con los pesos CRUDOS finales (2.7).

    El hermano es un checkpoint **estándar** (mismo formato, receta incluida) para habilitar la
    comparativa crudo-vs-EMA de la misma corrida: su ``state_dict`` son los crudos
    (``torch.equal``) y difiere del EMA del principal; su meta **no** lleva la marca ``ema`` (la
    ausencia = pesos crudos, 2.3) y sí un puntero al checkpoint del que es contraparte. Se verifica
    que todo el tooling existente lo consume sin cambios (``load_checkpoint`` + ``make_model`` +
    ``generate_from_checkpoint``) y que el descubrimiento de snapshots **no** lo levanta.
    """
    from diffusion.samplers import generate_from_checkpoint

    result, model_spec, crudos = _corrida_publicable(num_steps=6, ema_decay=0.6)

    path = tmp_path / "vp_final.pt"
    save_checkpoint(result, path, model_spec=model_spec, raw_sibling=True)

    raw_path = tmp_path / "vp_final_raw.pt"
    assert raw_path.exists()

    raw_state, raw_meta = load_checkpoint(raw_path)
    ema_state, ema_meta = load_checkpoint(path)
    for key in crudos:
        assert torch.equal(raw_state[key], crudos[key]), f"el hermano no lleva los crudos en {key}"
    assert any(not torch.equal(raw_state[k], ema_state[k]) for k in crudos)  # crudos ≠ EMA

    # Marcado como contraparte cruda y sin la marca de EMA (no se puede confundir con el oficial).
    assert "ema" not in raw_meta
    assert raw_meta["raw_of"] == path.name
    assert ema_meta["ema"] == {"decay": pytest.approx(0.6)}
    # Formato estándar con receta: el resto de la meta es la del principal.
    assert set(raw_meta) == {"sde_name", "data_dim", "history", "model", "raw_of"}
    assert raw_meta["model"] == model_spec
    assert raw_meta["history"] == pytest.approx(result.history)

    # Reconstruible por la receta y consumible por la generación checkpoint-driven.
    net2 = make_model(raw_meta["model"]["name"], **raw_meta["model"]["kwargs"])
    net2.load_state_dict(raw_state)
    x_raw = generate_from_checkpoint(raw_path, "euler", n_samples=8, n_steps=5, seed=0)
    x_ema = generate_from_checkpoint(path, "euler", n_samples=8, n_steps=5, seed=0)
    for x in (x_raw, x_ema):
        assert x.shape == (8, 2)
        assert x.dtype == torch.float32
        assert torch.all(torch.isfinite(x))
    # Pesos distintos ⇒ muestras distintas con la misma semilla (la comparativa tiene sentido).
    assert not torch.equal(x_raw, x_ema)

    # El hermano no es un snapshot periódico: el descubrimiento no lo levanta (2.7).
    assert discover_snapshots(path) == []


def test_hermano_de_crudos_no_se_escribe_sin_ema(tmp_path):
    """Sin EMA activo, ``raw_sibling=True`` **no** escribe el hermano (sería un duplicado).

    El hermano existe para separar crudos de EMA; sin EMA el checkpoint principal YA publica los
    crudos, así que emitirlo sería escribir dos veces lo mismo. El parámetro queda por eso seguro
    de activar incondicionalmente en el guardado final del CLI (2.5: sin EMA, nada cambia).
    """
    result, model_spec, crudos = _corrida_publicable(num_steps=4)  # sin EMA
    path = tmp_path / "vp_sin_ema_final.pt"
    save_checkpoint(result, path, model_spec=model_spec, raw_sibling=True)

    assert not (tmp_path / "vp_sin_ema_final_raw.pt").exists()
    state_dict, meta = load_checkpoint(path)
    assert "ema" not in meta and "raw_of" not in meta
    for key in crudos:
        assert torch.equal(state_dict[key], crudos[key])


def test_discover_snapshots_ignora_el_hermano_de_crudos(tmp_path):
    """Mecánico: ``{stem}_raw.pt`` no matchea el patrón ``_stepNNNNN`` del descubrimiento (2.7).

    Se tocan los tres artefactos posibles de una corrida (final, hermano de crudos y un snapshot
    periódico con su sidecar) sin escribir contenido: la política de descubrimiento mira solo
    nombres. Solo el periódico debe salir en la lista — el hermano nunca puede elegirse como punto
    de reanudación.
    """
    base = tmp_path / "vp_mixture.pt"
    for nombre in (
        "vp_mixture.pt",
        "vp_mixture_raw.pt",
        "vp_mixture_step00002.pt",
        "vp_mixture_step00002.resume.pt",
    ):
        (tmp_path / nombre).touch()

    assert discover_snapshots(base) == [(2, tmp_path / "vp_mixture_step00002.pt")]


def test_publicacion_ema_compone_con_la_parametrizacion_epsilon(tmp_path):
    """El EMA publicado compone con la parametrización ε de la receta (2.4).

    Corrida corta config-driven con la red **envuelta** (``EpsilonScoreWrapper``): su
    ``state_dict`` delega al interno, así que la sombra —y por lo tanto el checkpoint publicado—
    queda en claves de **red pelada**. Consecuencia verificada acá: el checkpoint EMA se
    reconstruye por ``make_model`` + wrap por receta y ``generate_from_checkpoint`` produce
    muestras finitas sin ningún cambio en su contrato; el hermano de crudos también.
    """
    from diffusion.samplers import generate_from_checkpoint

    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 4, "seed": 0, "ema_decay": 0.6},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1,
                  "score_parametrization": "epsilon"},
    }
    spec = build_run(raw)
    assert isinstance(spec.model, EpsilonScoreWrapper)
    result = train(spec.sde, spec.model, spec.data, spec.config)

    # Claves de red pelada (el wrapper delega): sin prefijo ``_net.``, como las de make_model.
    assert result.ema_state is not None
    assert all(not k.startswith("_net.") for k in result.ema_state)
    assert set(result.ema_state) == set(make_model("mlp", data_dim=2, hidden_dim=32,
                                                  num_blocks=1).state_dict())

    path = tmp_path / "ckpt_epsilon_ema.pt"
    save_checkpoint(result, path, model_spec=spec.model_spec, raw_sibling=True)
    raw_path = tmp_path / "ckpt_epsilon_ema_raw.pt"

    state_dict, meta = load_checkpoint(path)
    assert meta["model"]["score_parametrization"] == "epsilon"  # la receta viaja igual que antes
    assert meta["ema"] == {"decay": pytest.approx(0.6)}
    for key in result.ema_state:
        assert torch.equal(state_dict[key], result.ema_state[key])

    # Reconstrucción por la receta: red pelada + wrap con la σ de la SDE (lo que hace generate).
    sde = make_sde(meta["sde_name"], data_dim=meta["data_dim"])
    net2 = make_model(meta["model"]["name"], **meta["model"]["kwargs"])
    net2.load_state_dict(state_dict)
    wrapper = EpsilonScoreWrapper(net2, lambda x, t: sde.marginal_prob(x, t)[1])
    wrapper.eval()

    x_ema = generate_from_checkpoint(path, "euler", n_samples=8, n_steps=5, seed=0)
    x_raw = generate_from_checkpoint(raw_path, "euler", n_samples=8, n_steps=5, seed=0)
    for x in (x_ema, x_raw):
        assert x.shape == (8, 2)
        assert torch.all(torch.isfinite(x))
    assert not torch.equal(x_ema, x_raw)  # el hermano lleva otros pesos (crudos)


# ------------- ema-weights: reanudación con EMA (task 3)
#
# Con EMA activo los checkpoints publican la sombra, así que el estado de una corrida vive
# **partido**: el checkpoint lleva el EMA y el sidecar de resume los crudos + la sombra. Esta
# sección cubre el lado del LOOP de ese contrato —los dos guards de coherencia (3.4), la
# restauración de la sombra y el round-trip interrumpida ≡ ininterrumpida (3.1, 3.2)—; el lado del
# sidecar/``load_resume`` vive en ``test_resume.py``, donde están sus tests.

from diffusion.training import (  # noqa: E402  (sección append-only, task 3)
    ResumeState,
    load_resume,
    resume_sidecar_path,
    save_resume_state,
)

_EMA_RESUME_N = 4  # paso del snapshot intermedio (el punto de interrupción)
_EMA_RESUME_TOTAL = 8  # total de la corrida (el snapshot cae antes del último paso)
_EMA_RESUME_DECAY = 0.6  # chico a propósito: la sombra se separa de los crudos en pocos pasos


def _fuente_constante(batch):
    """Fuente infinita de ORDEN FIJO (el mismo batch en cada paso): resume-invariante.

    ``train`` reconstruye ``iter(data)`` en cada llamada, así que una fuente posicional le daría a
    la corrida reanudada los batches del arranque y no los del paso ``N``. Con un batch constante
    el dato de cada paso es el mismo en las dos corridas y la comparación aísla exactamente lo que
    esta tarea prueba (crudos + sombra + optimizador + azar restaurados). Es el mismo recurso que
    usan los tests de equivalencia de ``training-resume``.
    """
    while True:
        yield batch


def _batch_constante(n=32, dim=2, seed=4321):
    """Batch fijo con ``Generator`` propio: no toca el RNG global que las corridas siembran."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=gen)


def _net_resume_ema(sde) -> ScoreMLP:
    """Red chica con pesos iniciales idénticos en cada invocación (misma semilla)."""
    torch.manual_seed(321)
    return _small_net(sde)


def _resume_dummy(**overrides) -> ResumeState:
    """:class:`ResumeState` mínimo para ejercitar los guards (el loop falla antes de usarlo)."""
    base = dict(
        optimizer_state={},
        start_step=1,
        torch_rng_state=torch.get_rng_state(),
        generator_state=torch.Generator().manual_seed(0).get_state(),
        history=[1.0],
    )
    base.update(overrides)
    return ResumeState(**base)


def test_train_con_ema_rechaza_reanudar_sin_sombra_en_el_sidecar():
    """EMA pedido + sidecar sin sombra ⇒ ``ValueError`` explícito antes de entrenar (3.4).

    Nunca una sombra inventada en silencio: reconstruirla de cero desde los pesos del paso ``N``
    perdería el promedio de toda la primera mitad de la corrida y el checkpoint final publicaría un
    EMA que no corresponde a la corrida. El guard va con el resto del fail-fast (la fuente centinela
    convierte cualquier consumo de datos en un ``AssertionError`` distinto del error esperado) y el
    mensaje nombra el sidecar y el decay configurado.
    """
    sde = make_sde("vp")
    cfg = TrainConfig(num_steps=2, seed=0, ema_decay=0.9)

    with pytest.raises(ValueError) as exc:
        train(sde, _small_net(sde), _DataProhibidaEma(), cfg, resume=_resume_dummy())

    mensaje = str(exc.value)
    assert "sidecar" in mensaje
    assert "ema_decay" in mensaje


def test_train_sin_ema_rechaza_reanudar_con_sombra_en_el_sidecar():
    """Sombra en el sidecar + EMA no pedido ⇒ ``ValueError`` explícito (3.4, simetría del guard).

    La corrida original mantenía una sombra; continuar sin EMA la descartaría y el checkpoint final
    publicaría los pesos crudos donde antes iba el promedio — un artefacto inconsistente con los
    intermedios de la misma corrida. Se falla, con el mismo criterio fail-fast.
    """
    sde = make_sde("vp")
    net = _small_net(sde)
    sombra = {k: v.detach().clone() for k, v in net.state_dict().items()}
    cfg = TrainConfig(num_steps=2, seed=0)  # sin EMA

    with pytest.raises(ValueError) as exc:
        train(sde, net, _DataProhibidaEma(), cfg, resume=_resume_dummy(ema_state=sombra))

    mensaje = str(exc.value)
    assert "sidecar" in mensaje
    assert "ema_decay" in mensaje


def test_snapshot_con_ema_lleva_crudos_y_sombra_clonados():
    """Cada snapshot intermedio lleva en su ``ResumeState`` los crudos y la sombra del momento (3.1).

    Ambos **clonados**: el loop sigue mutando los pesos in-place y avanzando la sombra después del
    snapshot, así que una referencia viva haría que el sidecar del paso 2 persistiera el estado del
    final de la corrida (y la reanudación desde ese punto sería una ficción).
    """
    sde, net, data, cfg = _fresh_ema(num_steps=6, checkpoint_every=2, ema_decay=0.9)
    estados: dict[str, ResumeState] = {}

    def _cb(tag, snapshot):
        estados[tag] = snapshot.resume

    train(sde, net, data, cfg, on_checkpoint=_cb, generator=_generador(cfg))

    assert sorted(estados) == ["step00002", "step00004"]
    r2, r4 = estados["step00002"], estados["step00004"]
    finales = net.state_dict()
    for estado in (r2, r4):
        assert estado.raw_model_state is not None
        assert estado.ema_state is not None
        assert set(estado.raw_model_state) == set(finales)
        assert set(estado.ema_state) == set(finales)

    # Fotos del momento, no referencias vivas: paso 2 ≠ paso 4 ≠ final de la corrida.
    assert any(not torch.equal(r2.raw_model_state[k], r4.raw_model_state[k]) for k in finales)
    assert any(not torch.equal(r2.raw_model_state[k], finales[k]) for k in finales)
    assert any(not torch.equal(r2.ema_state[k], r4.ema_state[k]) for k in finales)
    # La sombra es un promedio, no una copia de los crudos de su mismo momento.
    assert any(not torch.equal(r4.ema_state[k], r4.raw_model_state[k]) for k in finales)


def test_resume_con_ema_equivalente_a_corrida_ininterrumpida(tmp_path):
    """Round-trip (3.2): interrumpida y reanudada ≡ ininterrumpida, por comparación **exacta**.

    Se recorre el camino real del CLI de punta a punta: la corrida A persiste en el paso
    ``_EMA_RESUME_N`` el checkpoint de pesos (que publica la **sombra**) y su sidecar (crudos +
    sombra); la corrida B lo levanta con :func:`load_resume` —de donde salen los **crudos** para la
    red— y completa los pasos restantes. Al terminar deben coincidir las tres cosas que nombra el
    criterio, con ``torch.equal``: los pesos **crudos**, la **sombra** y el **checkpoint publicado**
    (más el ``history``). Sin la extensión del sidecar, B arrancaría de los pesos EMA del
    checkpoint intermedio y nada de eso cerraría.
    """
    sde = make_sde("vp")
    batch = _batch_constante()
    receta = {"name": "mlp", "kwargs": dict(_RECETA_CHICA["kwargs"])}
    expected = {"sde_name": "vp", "model_spec": receta, "data_dim": 2}
    base = tmp_path / "vp_ema.pt"

    def on_checkpoint(tag, snapshot):
        # Wiring del CLI: pesos (con la política de publicación EMA) + sidecar de resume.
        tagged = base.with_stem(f"{base.stem}_{tag}")
        save_checkpoint(snapshot.result, tagged, model_spec=receta)
        save_resume_state(resume_sidecar_path(tagged), snapshot.resume)

    # --- A: corrida ininterrumpida (deja el punto de reanudación del paso _EMA_RESUME_N).
    net_a = _net_resume_ema(sde)
    res_a = train(
        sde,
        net_a,
        _fuente_constante(batch),
        TrainConfig(
            num_steps=_EMA_RESUME_TOTAL,
            seed=0,
            ema_decay=_EMA_RESUME_DECAY,
            checkpoint_every=_EMA_RESUME_N,
        ),
        on_checkpoint=on_checkpoint,
    )
    crudos_a = {k: v.detach().clone() for k, v in net_a.state_dict().items()}
    ckpt_a = tmp_path / "final_a.pt"
    save_checkpoint(res_a, ckpt_a, model_spec=receta)

    # --- B: reanudada desde el snapshot por el camino real (load_resume).
    snap = base.with_stem(f"{base.stem}_step{_EMA_RESUME_N:05d}")
    state_dict, _meta, resume = load_resume(snap, expected=expected)
    assert resume.start_step == _EMA_RESUME_N
    assert resume.ema_state is not None  # la sombra viajó en el sidecar (3.1)
    # El checkpoint intermedio publica la SOMBRA: lo que se carga en la red son los CRUDOS.
    publicado, _ = load_checkpoint(snap)
    assert any(not torch.equal(state_dict[k], publicado[k]) for k in state_dict)

    net_b = _net_resume_ema(sde)
    net_b.load_state_dict(state_dict)
    res_b = train(
        sde,
        net_b,
        _fuente_constante(batch),
        TrainConfig(num_steps=_EMA_RESUME_TOTAL, seed=0, ema_decay=_EMA_RESUME_DECAY),
        resume=resume,
    )
    ckpt_b = tmp_path / "final_b.pt"
    save_checkpoint(res_b, ckpt_b, model_spec=receta)

    # (1) misma curva y mismos pesos CRUDOS.
    assert res_b.history == res_a.history  # igualdad exacta de floats, paso a paso
    crudos_b = net_b.state_dict()
    for key in crudos_a:
        assert torch.equal(crudos_b[key], crudos_a[key]), f"peso crudo distinto en {key}"

    # (2) misma SOMBRA: restaurada del sidecar, no reconstruida desde los pesos cargados.
    assert res_b.ema_state is not None
    for key in res_a.ema_state:
        assert torch.equal(res_b.ema_state[key], res_a.ema_state[key]), f"sombra distinta en {key}"

    # (3) mismo checkpoint publicado (contenido: state_dict tensor a tensor + meta).
    sd_a, meta_a = load_checkpoint(ckpt_a)
    sd_b, meta_b = load_checkpoint(ckpt_b)
    assert meta_a == meta_b
    for key in sd_a:
        assert torch.equal(sd_b[key], sd_a[key]), f"peso publicado distinto en {key}"

    # La sombra NO es los crudos: la igualdad de (2) y (3) no es trivialmente satisfacible.
    assert any(not torch.equal(res_a.ema_state[k], crudos_a[k]) for k in crudos_a)


def test_resume_sin_ema_no_cambia_con_sidecar_viejo(tmp_path):
    """Sidecar sin claves de EMA + config sin EMA ⇒ reanudación **exactamente como hoy** (3.3).

    El camino retrocompatible completo: la corrida A no configura EMA (su checkpoint intermedio
    publica los crudos y su sidecar no gana ninguna clave), B reanuda desde ahí y termina idéntica
    a la ininterrumpida. Ningún guard se dispara y ``ema_state`` queda en ``None`` de punta a punta.
    """
    sde = make_sde("vp")
    batch = _batch_constante()
    receta = {"name": "mlp", "kwargs": dict(_RECETA_CHICA["kwargs"])}
    base = tmp_path / "vp_sin_ema.pt"

    def on_checkpoint(tag, snapshot):
        tagged = base.with_stem(f"{base.stem}_{tag}")
        save_checkpoint(snapshot.result, tagged, model_spec=receta)
        save_resume_state(resume_sidecar_path(tagged), snapshot.resume)

    net_a = _net_resume_ema(sde)
    res_a = train(
        sde,
        net_a,
        _fuente_constante(batch),
        TrainConfig(num_steps=_EMA_RESUME_TOTAL, seed=0, checkpoint_every=_EMA_RESUME_N),
        on_checkpoint=on_checkpoint,
    )
    crudos_a = {k: v.detach().clone() for k, v in net_a.state_dict().items()}
    assert res_a.ema_state is None

    snap = base.with_stem(f"{base.stem}_step{_EMA_RESUME_N:05d}")
    state_dict, meta, resume = load_resume(
        snap, expected={"sde_name": "vp", "model_spec": receta, "data_dim": 2}
    )
    assert "ema" not in meta  # el intermedio publicó crudos, como siempre
    assert resume.ema_state is None
    publicado, _ = load_checkpoint(snap)
    for key in publicado:
        assert torch.equal(state_dict[key], publicado[key]), key  # los pesos del checkpoint

    net_b = _net_resume_ema(sde)
    net_b.load_state_dict(state_dict)
    res_b = train(
        sde,
        net_b,
        _fuente_constante(batch),
        TrainConfig(num_steps=_EMA_RESUME_TOTAL, seed=0),
        resume=resume,
    )

    assert res_b.ema_state is None
    assert res_b.history == res_a.history
    crudos_b = net_b.state_dict()
    for key in crudos_a:
        assert torch.equal(crudos_b[key], crudos_a[key]), f"peso crudo distinto en {key}"


# --------------------------------------------- gate de la API torch.amp (AMP)
#
# Feature `gpu-training-efficiency`, tarea 1 (Foundation / gate): antes de cablear la precisión
# mixta en el loop (tarea 2), el resume (tarea 3) o la config, estos tests fijan de primera mano el
# contrato de `torch.amp` del que dependen AmpLoop y AmpResume en torch 2.12. Si una versión futura
# rompe alguna de estas propiedades, fallan acá —el punto más barato— y no en medio del loop.
#
# Contrato afirmado (todo en CPU, sin GPU):
#   1. `autocast(enabled=False)` es no-op bit a bit  -> sostiene R1.4 (default off = sin regresión).
#   2. `GradScaler(device, enabled=False)` es passthrough (scale/unscale_/step/update no alteran nada)
#      y su estado hace round-trip -> sostiene R2.1/R2.2 y el sentinel de los guards (R2.5).
# Verificado en el entorno objetivo (torch 2.12.0+cpu): la API unificada `torch.amp.*` está
# disponible, así que NO se necesita el fallback documentado en design.md (§Implementation Order &
# API Fallback). Si estos tests fallan en otro entorno, aplicar ese fallback.


def test_amp_autocast_desactivado_es_noop():
    """`autocast(device_type="cpu", enabled=False)` no altera el forward (R1.4).

    Es la base de "AMP off = sin regresión": envolver el forward/pérdida en un autocast
    desactivado debe dar exactamente el mismo resultado que no envolverlo.
    """
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 4)
    x = torch.randn(3, 4)

    y_plain = net(x)
    with torch.autocast(device_type="cpu", enabled=False):
        y_off = net(x)

    assert torch.equal(y_plain, y_off)


def test_amp_autocast_habilitado_corre_en_cpu():
    """`autocast(device_type="cpu", enabled=True)` corre y computa en bfloat16 dentro del contexto.

    Confirma que el camino "AMP on" es ejercitable en CPU (sin GPU): el diseño testea la rama
    activada con la forma CPU del autocast (R5.2).
    """
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 4)
    x = torch.randn(3, 4)

    with torch.autocast(device_type="cpu", enabled=True):
        y_on = net(x)

    assert y_on.dtype is torch.bfloat16


def test_amp_gradscaler_desactivado_es_passthrough():
    """Un `GradScaler(device, enabled=False)` reproduce el paso plano del optimizador bit a bit.

    Con el escalador desactivado, `scale(loss).backward()` + `unscale_` + `step` + `update` deben
    dejar los pesos idénticos a `loss.backward()` + `optimizer.step()`. Es lo que permite construir
    el escalador solo cuando AMP está activo y confiar en que la rama con escalador (deshabilitado en
    CPU) no perturba la trayectoria (R1.2, R2.3).
    """

    def _un_paso(*, con_escalador: bool) -> dict:
        torch.manual_seed(0)
        net = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        x = torch.randn(3, 4)  # mismas extracciones de RNG en ambos caminos

        loss = net(x).sum()
        opt.zero_grad()
        if con_escalador:
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        return {k: v.detach().clone() for k, v in net.state_dict().items()}

    plano = _un_paso(con_escalador=False)
    escalado = _un_paso(con_escalador=True)

    for key in plano:
        assert torch.equal(plano[key], escalado[key]), f"peso distinto en {key}"


def test_amp_gradscaler_estado_hace_round_trip():
    """El estado del escalador se guarda y se restaura sin error (R2.1/R2.2).

    Un escalador desactivado (el caso CPU) tiene `state_dict()` vacío; el round-trip debe funcionar
    igual. Además fija el sentinel de los guards: el estado es un `dict` (posiblemente vacío), NUNCA
    `None` cuando el escalador existe —por eso los guards distinguen AMP on/off por `is None`, no por
    dict vacío (R2.4/R2.5)—.
    """
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    estado = scaler.state_dict()
    assert isinstance(estado, dict)  # dict (vacío en CPU), no None

    otro = torch.amp.GradScaler("cpu", enabled=False)
    otro.load_state_dict(estado)  # no debe levantar


# --------------------------------------------- AmpLoop: precisión mixta en el loop (tarea 2)
#
# Feature `gpu-training-efficiency`, tarea 2 (AmpLoop): el campo opt-in `TrainConfig.amp` y su
# semántica en el núcleo de `train()` —autocast sobre forward/pérdida, escalador de gradiente
# condicional, `unscale_` ANTES del recorte, sombra EMA en float32 tras el paso real, fail-fast ante
# un `amp` mal formado—. Todo verificado en CPU (autocast bf16 + escalador passthrough): la rama on
# se ejercita sin GPU y la rama off queda byte-idéntica al loop previo (R1.1–R1.5, R5.1–R5.3).


def test_trainconfig_amp_default_off():
    """`TrainConfig()` sin el campo nuevo declara `amp=False` — sin precisión mixta (R1.4, R5.1)."""
    cfg = TrainConfig()
    assert cfg.amp is False
    # El campo es configurable como cualquier otro del dataclass.
    assert TrainConfig(amp=True).amp is True


def test_train_amp_off_bit_identico_al_loop_previo():
    """Con `amp=False` (default) `train()` es BIT-IDÉNTICO al loop previo (R1.4/R5.1).

    Mismo patrón que el guard del default de EMA/time_sampling: una réplica manual del loop crudo
    (sin escalador ni autocast) contra `train()` con la config default. Se comparan historia
    (igualdad EXACTA de floats), pesos finales (`torch.equal`) y el azar consumido (RNG global +
    estado del generator): si construir el escalador o envolver en autocast tocara el camino off,
    alguna fallaría. El escalador se construye SOLO con `amp=True`, así que off no gana rama alguna.
    """
    sde, net_a, data, cfg = _fresh_ema()  # amp=False, ema off (defaults)
    g_a = _generador(cfg)
    res_a = train(sde, net_a, data, cfg, generator=g_a)
    rng_a = torch.get_rng_state()

    sde, net_b, data, cfg = _fresh_ema()
    g_b = _generador(cfg)
    torch.manual_seed(cfg.seed)  # lo que hace train() al arrancar desde cero
    hist_b, _ = _replica_loop_crudo(sde, net_b, data, cfg, g_b)
    rng_b = torch.get_rng_state()

    assert res_a.history == hist_b  # igualdad exacta de floats por paso, no aproximada
    sd_a, sd_b = net_a.state_dict(), net_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"peso final distinto en {key}"
    assert torch.equal(rng_a, rng_b)  # misma secuencia del RNG global
    assert torch.equal(g_a.get_state(), g_b.get_state())  # mismo stream del generator


def test_train_amp_off_reproducible_por_seed():
    """Dos corridas `amp=False` con la misma semilla dan la MISMA trayectoria (R1.4)."""

    def run():
        torch.manual_seed(0)  # pesos iniciales idénticos entre corridas
        sde = make_sde("vp")
        net = _small_net(sde)
        dist = make_distribution("gaussian", 2, seed=1)
        data = _data(dist, n=256, batch_size=64)
        return train(sde, net, data, TrainConfig(num_steps=20, seed=7, amp=False)).history

    assert run() == pytest.approx(run())


def test_train_amp_on_corre_en_cpu_y_baja_la_perdida():
    """Con `amp=True` el loop corre en CPU (autocast bf16 + escalador passthrough) y aprende (R1.1/R5.2).

    Se compara la **tendencia** con medias de bloque (primer vs último quinto): la pérdida per-step
    es ruidosa por el t aleatorio. El punto es que la rama AMP es ejercitable sin GPU y no rompe.
    """
    sde = make_sde("vp")
    dist = make_distribution("mixture", 2, n_components=8, seed=0)
    torch.manual_seed(0)
    net = ScoreMLP(data_dim=sde.data_dim, hidden_dim=64, num_blocks=2)
    data = _data(dist, n=512, batch_size=128)
    result = train(sde, net, data, TrainConfig(num_steps=240, seed=0, amp=True))

    assert len(result.history) == 240
    assert all(math.isfinite(v) for v in result.history)
    hist = result.history
    k = max(1, len(hist) // 5)
    assert sum(hist[-k:]) / k < sum(hist[:k]) / k


def test_train_amp_on_con_grad_clip_desescala_antes_del_clip(monkeypatch):
    """Con AMP y `grad_clip`, el `unscale_` ocurre ANTES del recorte y este antes del `step` (R1.2).

    El recorte debe operar sobre la norma REAL del gradiente, así que hay que des-escalar primero.
    Se instrumenta el escalador (subclase spy sobre el real) y `clip_grad_norm_` para registrar el
    orden de llamadas del núcleo de optimización; se exige `unscale_` < `clip` < `step`.
    """
    orden: list[str] = []
    _RealScaler = torch.amp.GradScaler

    class _SpyScaler(_RealScaler):
        def unscale_(self, optimizer):  # noqa: D102
            orden.append("unscale_")
            return super().unscale_(optimizer)

        def step(self, optimizer, *args, **kwargs):  # noqa: D102
            orden.append("step")
            return super().step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.amp, "GradScaler", _SpyScaler)

    _real_clip = torch.nn.utils.clip_grad_norm_

    def _spy_clip(*args, **kwargs):
        orden.append("clip")
        return _real_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy_clip)

    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)
    train(sde, net, _data(dist), _tiny_config(num_steps=2, grad_clip=1.0, amp=True))

    assert {"unscale_", "clip", "step"} <= set(orden)
    assert orden.index("unscale_") < orden.index("clip") < orden.index("step")


@pytest.mark.parametrize("bad", ["true", 1, 0, None, 1.0])
def test_train_rechaza_amp_malformado_antes_de_entrenar(bad):
    """Un `amp` no booleano revienta con `ValueError` ANTES de consumir data (R1.5/R5.3).

    Mismo criterio fail-fast que `time_sampling`/`ema_decay`: la fuente centinela convierte
    cualquier consumo prematuro en un `AssertionError` distinto del `ValueError` esperado. Un `int`
    (incluidos `0`/`1`) o un `float` no son `bool` (`isinstance(1, bool)` es `False`): se rechazan.
    """
    sde = make_sde("vp")
    net = _small_net(sde)
    cfg = TrainConfig(num_steps=2, seed=0, amp=bad)
    with pytest.raises(ValueError):
        train(sde, net, _DataProhibidaEma(), cfg)


def test_train_amp_on_con_ema_publica_pesos_float32(tmp_path):
    """Con AMP + EMA la sombra y los pesos publicados quedan en float32 (R1.3).

    autocast no cambia el dtype de los parámetros (viven en float32); la sombra los promedia en
    float32 y el checkpoint publica float32 —mismo formato y contrato de consumo que sin AMP—.
    """
    decay = 0.9
    sde, net, data, cfg = _fresh_ema(ema_decay=decay, amp=True)
    result = train(sde, net, data, cfg, generator=_generador(cfg))

    assert result.ema_state is not None
    assert all(v.dtype == torch.float32 for v in result.ema_state.values())
    # Los pesos crudos de la red también siguen en float32 (autocast no cambia sus dtypes).
    assert all(v.dtype == torch.float32 for v in net.state_dict().values())

    path = tmp_path / "ckpt_amp_ema.pt"
    save_checkpoint(result, path)
    state_dict, _ = load_checkpoint(path)
    assert all(v.dtype == torch.float32 for v in state_dict.values())


# ------------------------- NonBlocking · CudnnSetup: setup de device (tarea 5)
#
# Feature `gpu-training-efficiency`, tarea 5: transferencia de batch no bloqueante (R3.3) + autotune
# de kernels convolucionales en GPU (R4.1–4.3). Es setup/plumbing puro: en CPU no hay cambio
# observable (el `non_blocking=True` es transparente; `cudnn.benchmark` no se toca), así que la rama
# CUDA se ejercita SIN GPU con un `torch.device("cuda")` —un descriptor, no requiere hardware— sobre
# el helper de gate, y la rama CPU se verifica corriendo `train()` real. El estado global
# `torch.backends.cudnn.benchmark` se salva/restaura alrededor de cada test para no filtrarlo.

from diffusion.training import trainer as _trainer_mod  # noqa: E402
from diffusion.training import VAL_EXAM_SEED  # noqa: E402  (semilla compartida por los exámenes)


@pytest.fixture
def _cudnn_benchmark_restaurado():
    """Salva y restaura `torch.backends.cudnn.benchmark` para no filtrar estado global entre tests."""
    previo = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previo


def test_enable_cudnn_autotune_prende_benchmark_en_cuda(_cudnn_benchmark_restaurado):
    """En un device CUDA el helper prende `cudnn.benchmark` (R4.1).

    `torch.device("cuda")` es solo un descriptor (no requiere GPU), así que la rama gateada se
    ejercita sin hardware. Se parte de `benchmark=False` para probar que el helper lo lleva a True.
    """
    torch.backends.cudnn.benchmark = False
    _trainer_mod._enable_cudnn_autotune(torch.device("cuda"))
    assert torch.backends.cudnn.benchmark is True


def test_enable_cudnn_autotune_no_toca_benchmark_en_cpu(_cudnn_benchmark_restaurado):
    """En CPU el helper NO toca `cudnn.benchmark` (R4.2): queda con el valor que tenía."""
    for previo in (True, False):
        torch.backends.cudnn.benchmark = previo
        _trainer_mod._enable_cudnn_autotune(torch.device("cpu"))
        assert torch.backends.cudnn.benchmark is previo


def test_train_en_cpu_no_altera_cudnn_benchmark(_cudnn_benchmark_restaurado):
    """Una corrida real de `train()` en CPU deja `cudnn.benchmark` intacto (R4.2/R4.3).

    Guard de no-regresión del camino CPU: el setup de device no debe tocar el estado global de
    autotune cuando el device no es CUDA, y el resultado sigue siendo válido (R4.3).
    """
    torch.backends.cudnn.benchmark = False
    sde = make_sde("vp")
    net = _small_net(sde)
    dist = make_distribution("gaussian", 2, seed=0)
    result = train(sde, net, _data(dist), _tiny_config(num_steps=2))
    assert torch.backends.cudnn.benchmark is False
    assert len(result.history) == 2  # y el resultado sigue siendo válido (R4.3)


class _BatchGrabadora:
    """Envuelve un batch para registrar los kwargs con que el loop llama a `.to(...)` (R3.3).

    Solo se le pide `.to(...)` (es lo único que el loop invoca sobre el batch crudo). Devuelve un
    tensor REAL, así que el resto del paso entrena normal y el resultado no se altera.
    """

    def __init__(self, tensor, registro):
        self._tensor = tensor
        self._registro = registro

    def to(self, *args, **kwargs):
        self._registro.append(kwargs)
        return self._tensor.to(*args, **kwargs)


class _FuenteGrabadora:
    """Fuente infinita que envuelve cada batch en `_BatchGrabadora` para espiar la transferencia."""

    def __init__(self, base, registro):
        self._base = iter(base)
        self._registro = registro

    def __iter__(self):
        return self

    def __next__(self):
        return _BatchGrabadora(next(self._base), self._registro)


def test_train_transfiere_batch_no_bloqueante():
    """El loop transfiere cada batch con `non_blocking=True` (R3.3).

    Se espía el `.to(...)` del batch crudo: el loop debe pasar `non_blocking=True` (incondicional —
    inofensivo en CPU). El envoltorio devuelve un tensor real, así que la corrida entrena normal y el
    resultado no se altera (los tests de reproducibilidad existentes son el guard de no-regresión).
    """
    registro: list[dict] = []
    sde = make_sde("vp")
    net = _small_net(sde)
    dist = make_distribution("gaussian", 2, seed=0)
    fuente = _FuenteGrabadora(_data(dist), registro)

    result = train(sde, net, fuente, _tiny_config(num_steps=3))

    assert len(registro) == 3  # un `.to` por paso
    assert all(kw.get("non_blocking") is True for kw in registro)
    assert len(result.history) == 3  # entrena normal: el resultado no se altera


# --------------------------------------------- barra de progreso (display-only)


def test_train_progress_no_altera_resultado():
    """La barra de progreso (`progress=True`) es display-only: no cambia el resultado.

    Con la misma semilla, `progress=True` y `progress=False` producen el MISMO `history`
    (la barra solo envuelve la iteración y escribe a stderr; no toca el RNG ni el cómputo).
    """

    def run(progress):
        torch.manual_seed(0)  # pesos iniciales idénticos entre corridas
        sde = make_sde("vp")
        net = _small_net(sde)
        dist = make_distribution("gaussian", 2, seed=1)
        data = _data(dist, n=256, batch_size=64)
        return train(
            sde, net, data, TrainConfig(num_steps=12, seed=7), progress=progress
        ).history

    assert run(progress=True) == pytest.approx(run(progress=False))


# --------------------------------------------- logging estructurado (on_log)


def test_train_on_log_emite_estados_por_cadencia():
    """`on_log` recibe {step, loss} en la cadencia de `log_every`, sin cambiar el resultado."""
    recs: list[dict] = []

    def run(collect):
        torch.manual_seed(0)
        sde = make_sde("vp")
        net = _small_net(sde)
        dist = make_distribution("gaussian", 2, seed=1)
        data = _data(dist, n=256, batch_size=64)
        cb = recs.append if collect else None
        return train(
            sde, net, data, TrainConfig(num_steps=20, seed=7, log_every=5), on_log=cb
        ).history

    hist_with = run(collect=True)
    assert [r["step"] for r in recs] == [5, 10, 15, 20]  # cadencia log_every=5
    assert all(isinstance(r["loss"], float) for r in recs)

    recs.clear()
    hist_without = run(collect=False)
    assert hist_with == pytest.approx(hist_without)  # display-only: no altera el history


def test_train_on_log_funciona_con_log_every_cero():
    """Con `log_every=0` (p. ej. --quiet), `on_log` igual emite en la cadencia num_steps//20."""
    recs: list[dict] = []
    torch.manual_seed(0)
    sde = make_sde("vp")
    net = _small_net(sde)
    dist = make_distribution("gaussian", 2, seed=1)
    data = _data(dist, n=256, batch_size=64)
    train(sde, net, data, TrainConfig(num_steps=40, seed=0, log_every=0), on_log=recs.append)
    assert len(recs) == 20  # num_steps//20 = 2 -> pasos 2,4,...,40
    assert recs[-1]["step"] == 40


# ------------- validation-loss: entrada opt-in, exámenes y fail-fast (task 4.1)
#
# Feature `validation-loss`, tarea 4.1: el SEAM de la validación en el loop —los dos parámetros
# keyword-only, la construcción de los dos :class:`FixedValExam` en el bloque de fail-fast y el
# rechazo de la cadencia deshabilitada (3.7)—, más los campos de la serie en ``TrainResult`` y
# ``ResumeState`` y su continuación al reanudar. La CADENCIA y las tres mediciones son de la tarea
# 4.2: acá los exámenes se construyen y todavía nadie los llama, así que ningún test de esta sección
# asevera que la serie se llene durante la corrida.
#
# Se reusan los helpers de la sección de EMA (``_fresh_ema`` / ``_generador``: corrida chica con
# pesos iniciales y fuente de datos idénticos en cada invocación) y de la de reanudación
# (``_fuente_constante`` / ``_batch_constante``).


class _DataProhibidaVal:
    """Fuente centinela: cualquier consumo de datos delata un fail-fast tardío de la validación."""

    def __iter__(self):
        return self

    def __next__(self):
        raise AssertionError("el loop consumió data antes de validar la cadencia de validación")


class _FuenteValVigilada:
    """Fuente re-iterable de batches que **cuenta** sus recorridos.

    Sirve para aseverar el otro lado del fail-fast: cuando la cadencia es inválida, el loop no solo
    no consume el primer batch de entrenamiento — tampoco toca la fuente del examen.
    """

    def __init__(self, batches: list) -> None:
        self._batches = batches
        self.recorridos = 0

    def __iter__(self):
        self.recorridos += 1
        return iter(self._batches)


def _fuente_examen(n=6, batch=4, dim=2, seed=99):
    """Fuente RE-ITERABLE de batches en memoria (una lista), con una cola parcial.

    Es lo que el examen fijo necesita (recorrerla dos veces entrega la misma secuencia) sin tocar el
    disco ni depender de torchvision, igual que en la suite del evaluador.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=gen)
    return [x[i:i + batch] for i in range(0, n, batch)]


def _espiar_examenes(monkeypatch) -> list[dict]:
    """Reemplaza ``FixedValExam`` en el loop por un doble que solo registra cómo lo construyeron.

    Permite verificar la construcción —cuántos exámenes, con qué fuente, con qué muestreador de
    tiempos y en qué device— sin evaluar nada (en esta tarea el loop todavía no los llama).
    """
    llamadas: list[dict] = []

    def _doble(sde, batches, **kwargs):
        llamadas.append({"sde": sde, "batches": batches, **kwargs})
        return object()  # centinela: nadie lo usa todavía

    monkeypatch.setattr(_trainer_mod, "FixedValExam", _doble)
    return llamadas


def _espiar_time_sampler(monkeypatch) -> list:
    """Registra el muestreador de tiempos que construye el loop, sin cambiar su comportamiento."""
    construidos: list = []
    real = _trainer_mod.make_time_sampler

    def _spy(*args, **kwargs):
        muestreador = real(*args, **kwargs)
        construidos.append(muestreador)
        return muestreador

    monkeypatch.setattr(_trainer_mod, "make_time_sampler", _spy)
    return construidos


def _resume_val(net, **overrides) -> ResumeState:
    """:class:`ResumeState` **cargable** (optimizador real) para ejercitar la reanudación.

    A diferencia de ``_resume_dummy`` —pensado para guards que fallan antes de tocarlo— este pasa
    el ``optimizer.load_state_dict`` del loop, así que la corrida reanudada llega a correr pasos.
    """
    base = dict(
        optimizer_state=torch.optim.Adam(net.parameters(), lr=1e-3).state_dict(),
        start_step=2,
        torch_rng_state=torch.get_rng_state(),
        generator_state=torch.Generator().manual_seed(0).get_state(),
        history=[1.0, 2.0],
    )
    base.update(overrides)
    return ResumeState(**base)


def test_trainresult_y_resumestate_declaran_la_serie_como_campo_opcional():
    """La serie de validación es un campo con default inerte en los dos dataclasses (5.2).

    ``TrainResult.val_history`` arranca en lista **vacía** (y por ``default_factory``, así que dos
    resultados no comparten la misma lista) y ``ResumeState.val_history`` en ``None`` — el
    equivalente de "este punto de reanudación no trae serie", que es el caso de todos los
    checkpoints anteriores a esta feature.
    """
    sde = make_sde("vp")
    net = _small_net(sde)

    res = TrainResult(net=net)
    assert res.val_history == []
    assert TrainResult(net=net).val_history is not res.val_history  # default_factory, no compartida

    assert _resume_val(net).val_history is None


def test_train_sin_val_batches_no_construye_nada_ni_cambia_la_corrida(monkeypatch):
    """Sin fuente de validación: cero exámenes construidos y corrida idéntica (6.1).

    El parámetro con default ``None`` **es** la palanca: una llamada que ni lo menciona y una que lo
    pasa explícitamente en ``None`` tienen que producir la misma historia (igualdad EXACTA de
    floats), los mismos pesos finales (``torch.equal``) y el mismo estado del ``generator``. El
    espía sobre ``FixedValExam`` prueba lo estructural —no se construye **nada**, así que no hay
    ninguna rama nueva activa— y la segunda corrida trae además ``train_exam_batches``: sin fuente
    de validación esa fuente se **ignora** en silencio (no hay serie donde publicar su valor), no se
    rechaza.
    """
    llamadas = _espiar_examenes(monkeypatch)

    sde, net_a, data, cfg = _fresh_ema()
    g_a = _generador(cfg)
    res_a = train(sde, net_a, data, cfg, generator=g_a)  # parámetros AUSENTES de la llamada

    sde, net_b, data, cfg = _fresh_ema()
    g_b = _generador(cfg)
    res_b = train(
        sde,
        net_b,
        data,
        cfg,
        generator=g_b,
        val_batches=None,
        train_exam_batches=_fuente_examen(),
    )

    assert llamadas == []  # ni un examen construido: cero ramas activas
    assert res_a.history == res_b.history  # igualdad exacta de floats por paso
    sd_a, sd_b = net_a.state_dict(), net_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"peso final distinto en {key}"
    assert torch.equal(g_a.get_state(), g_b.get_state())  # mismo stream del generator
    assert res_a.val_history == [] and res_b.val_history == []


@pytest.mark.parametrize("cadencia", [0, -1])
def test_train_rechaza_validacion_con_cadencia_deshabilitada(cadencia):
    """Validación pedida + cadencia **efectiva** <= 0 ⇒ ``ValueError`` antes del primer batch (3.7).

    Sin ``val_every`` declarado la cadencia efectiva es la de los checkpoints, así que pedir
    validación con esa cadencia apagada es una corrida que nunca mediría nada: se falla temprano en
    lugar de entrenar en silencio sin validación. El chequeo vive en el loop —no en el config
    layer— porque es el único punto que ve el valor **efectivo** (el CLI lo sobreescribe después de
    armar la corrida). El mensaje nombra además ``val_every``: con esa clave la corrida **sí** puede
    medir con los snapshots apagados, así que el error tiene que ofrecer la salida y no solo exigir
    ``checkpoint_every > 0``. La fuente centinela convierte cualquier consumo prematuro de datos en
    un ``AssertionError`` distinto del error esperado, y la fuente del examen tampoco se recorre.
    """
    sde = make_sde("vp")
    net = _small_net(sde)
    cfg = TrainConfig(num_steps=2, seed=0, checkpoint_every=cadencia)
    fuente_val = _FuenteValVigilada(_fuente_examen())

    with pytest.raises(ValueError) as exc:
        train(sde, net, _DataProhibidaVal(), cfg, val_batches=fuente_val)

    mensaje = str(exc.value)
    assert "checkpoint_every" in mensaje
    assert "val_every" in mensaje  # la salida sin snapshots, ofrecida en el propio error (3.10)
    assert str(cadencia) in mensaje  # nombra el valor recibido
    assert fuente_val.recorridos == 0


def test_train_construye_los_examenes_antes_del_primer_batch_con_su_time_sampler(monkeypatch):
    """Los dos exámenes se construyen en el fail-fast, con el muestreador del loop (3.4, 3.8).

    Dos cosas estructurales. Primero, la construcción ocurre **antes** de pedir el primer batch: la
    fuente centinela revienta en el primer ``next``, y los exámenes ya están construidos cuando eso
    pasa. Segundo, los dos reciben **el mismo objeto** ``TimeSampler`` que usa el loop (no uno
    propio): así "val y train usan el mismo criterio de muestreo de ``t``" no depende de que alguien
    lo recuerde. Se verifica además el orden (validación primero, examen de train después) y que
    cada uno reciba su propia fuente y el device de la corrida.
    """
    llamadas = _espiar_examenes(monkeypatch)
    muestreadores = _espiar_time_sampler(monkeypatch)

    sde = make_sde("vp")
    net = _small_net(sde)
    fuente_val, fuente_train = _fuente_examen(), _fuente_examen(seed=7)
    cfg = TrainConfig(num_steps=2, seed=0, checkpoint_every=1)

    with pytest.raises(AssertionError, match="consumió data"):
        train(
            sde,
            net,
            _DataProhibidaVal(),
            cfg,
            val_batches=fuente_val,
            train_exam_batches=fuente_train,
        )

    assert len(llamadas) == 2
    assert len(muestreadores) == 1  # un único muestreador en la corrida
    assert llamadas[0]["batches"] is fuente_val
    assert llamadas[1]["batches"] is fuente_train
    for llamada in llamadas:
        assert llamada["sde"] is sde
        assert llamada["time_sampler"] is muestreadores[0]  # el del loop, no uno propio
        assert torch.device(llamada["device"]) == torch.device(cfg.device)
        # Los dos exámenes comparten la MISMA semilla (3.8): es el mecanismo por el que ambos
        # sortean la misma secuencia de t y de ruido, y por lo tanto lo único que hace
        # interpretable el gap train↔val. Que uno de los dos la override —aunque sea con un
        # valor "razonable"— convierte la distancia entre las curvas en un artefacto de dos
        # sorteos distintos, con la forma de un gap de generalización pero sin significado.
        assert llamada.get("seed", VAL_EXAM_SEED) == VAL_EXAM_SEED


def test_train_sin_fuente_de_examen_de_train_construye_solo_el_de_validacion(monkeypatch):
    """``train_exam_batches`` es opcional: sin él se construye un solo examen (el de validación)."""
    llamadas = _espiar_examenes(monkeypatch)
    sde = make_sde("vp")
    net = _small_net(sde)
    fuente_val = _fuente_examen()
    cfg = TrainConfig(num_steps=2, seed=0, checkpoint_every=1)

    with pytest.raises(AssertionError, match="consumió data"):
        train(sde, net, _DataProhibidaVal(), cfg, val_batches=fuente_val)

    assert len(llamadas) == 1
    assert llamadas[0]["batches"] is fuente_val


def test_train_con_validacion_completa_la_corrida():
    """Con validación pedida y cadencia válida la corrida entrena de punta a punta.

    Smoke del seam con los exámenes **reales** (sin dobles): construirlos no rompe el loop y el
    ``history`` sigue siendo la serie per-step completa. La serie de validación se llena en la
    tarea 4.2, así que acá solo se pide que exista como lista.
    """
    sde, net, data, cfg = _fresh_ema(num_steps=4, checkpoint_every=2)
    res = train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        val_batches=_fuente_examen(),
        train_exam_batches=_fuente_examen(seed=7),
    )
    assert len(res.history) == 4
    assert isinstance(res.val_history, list)


def test_train_reanuda_continuando_la_serie_de_validacion():
    """Al reanudar, la serie del punto de reanudación se **continúa**, no se pierde (6.3).

    Los puntos ya medidos entran en la serie del resultado tal cual, y la lista del
    :class:`ResumeState` no se muta (se copia, igual que el ``history``): un caller que reusara el
    mismo ``ResumeState`` para otra corrida no encontraría puntos ajenos.
    """
    puntos = [{"step": 2, "raw": 1.5, "ema": None, "train": 1.2}]
    sde = make_sde("vp")
    net = _small_net(sde)
    cfg = TrainConfig(num_steps=3, seed=0, checkpoint_every=2)

    res = train(
        sde,
        net,
        _fuente_constante(_batch_constante()),
        cfg,
        resume=_resume_val(net, val_history=puntos),
        val_batches=_fuente_examen(),
    )

    assert res.val_history[:1] == puntos  # los previos, intactos y al frente de la serie
    assert res.val_history is not puntos  # copia, no la lista del resume
    assert len(puntos) == 1  # la lista del resume no ganó puntos


def test_train_reanuda_sin_guard_cruzado_entre_config_y_sidecar():
    """La serie no tiene guard cruzado config↔sidecar, a diferencia del EMA y del escalador.

    Esos dos son estado **necesario** para continuar la optimización y perderlos corrompería la
    corrida, así que las combinaciones cruzadas se rechazan. La serie de validación es una
    **observación**: que falte solo hace que arranque vacía, y que sobre no hace daño. Las dos
    combinaciones cruzadas corren sin levantar nada.
    """
    puntos = [{"step": 2, "raw": 1.5, "ema": None, "train": None}]
    cfg = TrainConfig(num_steps=3, seed=0, checkpoint_every=2)

    # (a) Serie en el punto de reanudación, corrida SIN fuente de validación: se conserva tal cual
    # (sin fuente no hay nada que medir, así que la serie queda con los puntos previos y nada más).
    sde = make_sde("vp")
    net = _small_net(sde)
    res_sin_fuente = train(
        sde,
        net,
        _fuente_constante(_batch_constante()),
        cfg,
        resume=_resume_val(net, val_history=puntos),
    )
    assert res_sin_fuente.val_history == puntos

    # (b) Fuente de validación, punto de reanudación SIN serie (sidecar/checkpoint viejo): arranca
    # vacía y la corrida sigue normalmente.
    sde = make_sde("vp")
    net = _small_net(sde)
    res_sin_serie = train(
        sde,
        net,
        _fuente_constante(_batch_constante()),
        cfg,
        resume=_resume_val(net),
        val_batches=_fuente_examen(),
    )
    assert isinstance(res_sin_serie.val_history, list)
    assert len(res_sin_serie.history) == 3  # continuó la corrida hasta el total


# ------------- validation-loss: cadencia y las tres mediciones (task 4.2)
#
# Feature `validation-loss`, tarea 4.2: el DISPARADOR de la evaluación y las hasta tres mediciones
# que produce cada una. Lo que la tarea 4.1 dejó construido pero sin llamar, acá se consume: la
# serie se llena y el punto se emite por ``on_log`` como variante discriminada por ``event``.
#
# El riesgo específico de esta tarea es conflacionar el disparador de la evaluación con el de los
# snapshots: se **parecen** (los dos miran ``checkpoint_every``) pero el de snapshots excluye el
# último paso y exige un ``on_checkpoint`` inyectado, mientras que el de la evaluación incluye el
# último paso y no depende de ningún callback (3.1). Hay un test dedicado a los dos a la vez.
#
# Se reusan los helpers de las secciones previas (``_fresh_ema`` / ``_generador`` / ``_fuente_examen``).


def _pasos_de_val(res) -> list[int]:
    """Los pasos en que la corrida midió validación (el índice de la serie dispersa)."""
    return [punto["step"] for punto in res.val_history]


def test_train_evalua_en_la_cadencia_y_en_el_ultimo_paso_sin_callback_de_checkpoint():
    """Disparador: múltiplos de ``checkpoint_every`` **más** el último paso, sin callback (3.1).

    10 pasos con cadencia 4 dan puntos en 4, 8 y **10**: los dos primeros por la cadencia y el
    tercero porque es el último paso de la corrida (que la cadencia no alcanza, 10 no es múltiplo de
    4). La llamada **no** inyecta ``on_checkpoint``: la validación es una medición, no un artefacto
    de checkpointing, así que no puede depender de que el caller quiera persistir snapshots.
    """
    sde, net, data, cfg = _fresh_ema(num_steps=10, checkpoint_every=4)
    res = train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        val_batches=_fuente_examen(),
        train_exam_batches=_fuente_examen(seed=7),
    )

    assert _pasos_de_val(res) == [4, 8, 10]
    assert len(res.history) == 10  # la serie densa sigue siendo una entrada por paso


def test_los_disparadores_de_evaluacion_y_de_snapshot_son_independientes():
    """El disparador de la evaluación **no** es el de los snapshots, en las dos direcciones (3.1).

    Con la misma corrida (10 pasos, cadencia 4) y un ``on_checkpoint`` inyectado:

    - los snapshots caen en 4 y 8 y **excluyen** el 10 (el último paso lo cubre el checkpoint final
      del caller);
    - los puntos de validación caen en 4, 8 y **también** en 10.

    Conflacionar los dos disparadores rompe exactamente uno de estos dos asserts, así que el test
    falla si alguien reusa la condición del snapshot para evaluar (perdería el punto final) o la de
    la evaluación para el snapshot (escribiría un snapshot de más).
    """
    sde, net, data, cfg = _fresh_ema(num_steps=10, checkpoint_every=4)
    tags: list[str] = []

    res = train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        on_checkpoint=lambda tag, _snapshot: tags.append(tag),
        val_batches=_fuente_examen(),
        train_exam_batches=_fuente_examen(seed=7),
    )

    assert tags == ["step00004", "step00008"]  # el snapshot excluye el último paso
    assert _pasos_de_val(res) == [4, 8, 10]  # la evaluación lo incluye


def test_train_con_ema_mide_los_tres_valores_por_evaluacion():
    """Con EMA activo cada punto trae los tres valores, distintos entre sí (4.1, 4.3, 3.8).

    ``raw`` ≠ ``ema`` porque los pesos difieren (la sombra es un promedio de la trayectoria, no la
    foto del último paso) y ``train`` ≠ ``raw`` porque son **otras** imágenes — el mismo examen, otro
    conjunto: esa distancia vertical es justamente el gap que la feature existe para mirar.
    """
    sde, net, data, cfg = _fresh_ema(num_steps=4, checkpoint_every=2, ema_decay=0.6)
    res = train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        val_batches=_fuente_examen(),
        train_exam_batches=_fuente_examen(seed=7),
    )

    assert _pasos_de_val(res) == [2, 4]
    for punto in res.val_history:
        assert set(punto) == {"step", "raw", "ema", "train"}
        for clave in ("raw", "ema", "train"):
            assert isinstance(punto[clave], float)
            assert math.isfinite(punto[clave])
        assert punto["raw"] != punto["ema"]  # pesos distintos, mismo examen
        assert punto["raw"] != punto["train"]  # mismo procedimiento, otras imágenes


def test_train_sin_ema_deja_el_valor_de_ema_ausente():
    """Sin sombra EMA el valor de EMA está **ausente**, nunca inventado (4.2).

    ``None`` es la representación de la ausencia (lo que anota ``ValPoint``): rellenarla con el valor
    de los pesos crudos dibujaría una segunda curva idéntica a la primera y sugeriría que el
    promediado no aporta nada, que es una conclusión y no un dato faltante.
    """
    sde, net, data, cfg = _fresh_ema(num_steps=4, checkpoint_every=2)  # ema_decay=None
    registros: list[dict] = []
    res = train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        on_log=registros.append,
        val_batches=_fuente_examen(),
        train_exam_batches=_fuente_examen(seed=7),
    )

    assert _pasos_de_val(res) == [2, 4]
    assert all(punto["ema"] is None for punto in res.val_history)
    assert all(isinstance(punto["raw"], float) for punto in res.val_history)
    assert all(isinstance(punto["train"], float) for punto in res.val_history)

    # La ausencia viaja como clave presente con ``None``, no como clave omitida: el consumidor del
    # .jsonl (task 7.1) distingue "no había EMA en esta corrida" de "el campo no existe en este
    # formato", y solo la primera lectura es la correcta. Un registro que filtrara los ``None``
    # borraría esa diferencia sin que nada más se rompa.
    de_val = [r for r in registros if r.get("event") == "val"]
    assert len(de_val) == 2
    for registro in de_val:
        assert set(registro) == {"event", "step", "val_raw", "val_ema", "train_fijo", "device"}
        assert registro["val_ema"] is None


def test_train_sin_fuente_de_examen_de_train_deja_ese_valor_ausente():
    """Sin ``train_exam_batches`` el examen de entrenamiento está ausente, no en cero (3.8)."""
    sde, net, data, cfg = _fresh_ema(num_steps=4, checkpoint_every=2)
    res = train(sde, net, data, cfg, generator=_generador(cfg), val_batches=_fuente_examen())

    assert _pasos_de_val(res) == [2, 4]
    assert all(punto["train"] is None for punto in res.val_history)
    assert all(isinstance(punto["raw"], float) for punto in res.val_history)


def test_el_examen_de_train_se_mide_solo_con_los_pesos_crudos():
    """El examen de entrenamiento se mide **una** vez, con los pesos vivos (3.9).

    La lectura raw↔EMA vive en la curva de validación; una cuarta medición (el examen de train bajo
    la sombra) duplicaría el costo de la evaluación sin agregar información. El test cuenta los
    recorridos de cada fuente: con EMA activo la de validación se recorre **dos** veces por
    evaluación (crudos + sombra) y la del examen de train **una** sola.
    """
    fuente_val = _FuenteValVigilada(_fuente_examen())
    fuente_train = _FuenteValVigilada(_fuente_examen(seed=7))
    sde, net, data, cfg = _fresh_ema(num_steps=4, checkpoint_every=2, ema_decay=0.6)

    train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        val_batches=fuente_val,
        train_exam_batches=fuente_train,
    )

    assert fuente_val.recorridos == 4  # 2 evaluaciones × (crudos + sombra)
    assert fuente_train.recorridos == 2  # 2 evaluaciones × solo crudos


def test_train_emite_el_registro_de_validacion_por_on_log():
    """El punto se emite por ``on_log`` como variante discriminada por ``event`` (5.1, 5.2).

    Un solo callback y dos variantes de registro: el de paso de entrenamiento **no** trae la clave
    ``event`` (asimetría deliberada: el escritor del CLI expande el registro al final, así que un
    ``event`` propio sobreescribe su default) y el de validación la trae en ``"val"`` con los tres
    valores y el ``device``. El ``device`` viaja porque el examen fijo es específico del device: sin
    ese dato, una serie medida en dos máquinas es indistinguible de un salto real del modelo.
    """
    registros: list[dict] = []
    sde, net, data, cfg = _fresh_ema(num_steps=4, checkpoint_every=2, ema_decay=0.6)
    res = train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        on_log=registros.append,
        val_batches=_fuente_examen(),
        train_exam_batches=_fuente_examen(seed=7),
    )

    de_val = [r for r in registros if r.get("event") == "val"]
    de_paso = [r for r in registros if "event" not in r]
    assert de_paso  # los registros de paso siguen existiendo, y sin clave ``event``
    assert all(set(r) == {"step", "loss"} for r in de_paso)  # variante existente, intacta

    assert [r["step"] for r in de_val] == [2, 4]
    for registro, punto in zip(de_val, res.val_history, strict=True):
        assert set(registro) == {"event", "step", "val_raw", "val_ema", "train_fijo", "device"}
        assert registro["step"] == punto["step"]
        assert registro["val_raw"] == punto["raw"]
        assert registro["val_ema"] == punto["ema"]
        assert registro["train_fijo"] == punto["train"]
        assert registro["device"] == str(torch.device(cfg.device))


def test_train_sin_on_log_mide_igual_y_no_revienta():
    """Sin ``on_log`` la serie se llena igual: la emisión es opcional, la medición no."""
    sde, net, data, cfg = _fresh_ema(num_steps=4, checkpoint_every=2)
    res = train(sde, net, data, cfg, generator=_generador(cfg), val_batches=_fuente_examen())
    assert _pasos_de_val(res) == [2, 4]


def test_el_snapshot_del_paso_n_ya_contiene_el_punto_medido_en_n():
    """La evaluación va **antes** del snapshot, así el snapshot de N ya trae el punto de N (6.3).

    Es lo que hace que reanudar desde el checkpoint del paso N restaure **todos** los puntos ya
    medidos: si la evaluación corriera después de emitir el snapshot, el punto de N quedaría fuera de
    su metadata y se perdería al reanudar. Se verifica en las dos fotos del snapshot (el
    ``TrainResult`` y el ``ResumeState``), y que cada una sea una **copia** (la serie sigue creciendo
    después).
    """
    sde, net, data, cfg = _fresh_ema(num_steps=6, checkpoint_every=2)
    fotos: dict[str, TrainSnapshot] = {}

    res = train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        on_checkpoint=lambda tag, snapshot: fotos.__setitem__(tag, snapshot),
        val_batches=_fuente_examen(),
        train_exam_batches=_fuente_examen(seed=7),
    )

    assert sorted(fotos) == ["step00002", "step00004"]
    assert _pasos_de_val(fotos["step00002"].result) == [2]
    assert [p["step"] for p in fotos["step00002"].resume.val_history] == [2]
    assert _pasos_de_val(fotos["step00004"].result) == [2, 4]
    assert _pasos_de_val(res) == [2, 4, 6]  # la corrida completa, con el punto del último paso
    # Copias, no la lista viva del loop: la foto del paso 2 no ganó los puntos posteriores.
    assert fotos["step00002"].result.val_history is not res.val_history


def test_una_evaluacion_con_ema_no_perturba_la_optimizacion():
    """Evaluar entre pasos —con swap de pesos incluido— no mueve la trayectoria de Adam (6.1, 6.2).

    Es el chequeo más fuerte disponible de que la restauración del swap EMA fue **in-place**: el
    optimizador guarda referencias a los tensores-parámetro desde ``Adam(net.parameters())``, así que
    una restauración que los **rebindeara** dejaría a Adam actualizando tensores huérfanos y los
    pasos posteriores a la primera evaluación divergirían. La cadencia 2 sobre 6 pasos garantiza que
    haya pasos después de una evaluación (3, 4, 5, 6).

    La regresión completa de no intervención —incluido el estado del ``generator`` y del RNG
    global— es de la tarea 4.3; acá se compara lo que esta tarea puede romper: la serie de pérdida y
    los pesos finales.
    """
    def corrida(con_validacion: bool):
        sde, net, data, cfg = _fresh_ema(num_steps=6, checkpoint_every=2, ema_decay=0.6)
        g = _generador(cfg)
        extra = (
            dict(val_batches=_fuente_examen(), train_exam_batches=_fuente_examen(seed=7))
            if con_validacion
            else {}
        )
        res = train(sde, net, data, cfg, generator=g, **extra)
        return res, net, g

    res_val, net_val, g_val = corrida(True)
    res_sin, net_sin, g_sin = corrida(False)

    assert _pasos_de_val(res_val) == [2, 4, 6]  # hubo evaluaciones, y con pasos después
    assert res_val.history == res_sin.history  # igualdad EXACTA de floats, paso por paso
    sd_val, sd_sin = net_val.state_dict(), net_sin.state_dict()
    for clave in sd_sin:
        assert torch.equal(sd_val[clave], sd_sin[clave]), f"peso final distinto en {clave}"
    for clave in res_sin.ema_state:
        assert torch.equal(res_val.ema_state[clave], res_sin.ema_state[clave])
    assert torch.equal(g_val.get_state(), g_sin.get_state())  # mismo stream del generator


# ------------------------- validation-loss: regresión de no intervención (task 4.3)


def _fingerprint(res, net, generator) -> dict:
    """Todo lo que una corrida produce y que la validación **no** debe alterar (6.1, 6.2).

    Incluye el estado del RNG **global** de torch además del ``generator`` del loop: son dos
    streams distintos y la validación podría mover cualquiera de los dos (una fuente de datos que
    se itere sin generator propio mueve el global; reusar el generator del loop mueve el otro).
    """
    return {
        "history": list(res.history),  # floats exactos, no aproximados
        "pesos": {k: v.detach().clone() for k, v in net.state_dict().items()},
        "ema": None if res.ema_state is None else {k: v.clone() for k, v in res.ema_state.items()},
        "generator": generator.get_state(),
        "rng_global": torch.random.get_rng_state(),
    }


def _assert_mismo_fingerprint(con, sin) -> None:
    """Compara dos fingerprints exigiendo igualdad exacta, nombrando qué se movió."""
    assert con["history"] == sin["history"], "la serie de pérdida de entrenamiento cambió"
    for clave in sin["pesos"]:
        assert torch.equal(con["pesos"][clave], sin["pesos"][clave]), f"peso distinto en {clave}"
    if sin["ema"] is not None:
        for clave in sin["ema"]:
            assert torch.equal(con["ema"][clave], sin["ema"][clave]), f"sombra distinta en {clave}"
    assert torch.equal(con["generator"], sin["generator"]), "se movió el generator del loop"
    assert torch.equal(con["rng_global"], sin["rng_global"]), "se movió el RNG global de torch"


@pytest.mark.parametrize("ema_decay", [None, 0.6], ids=["sin_ema", "con_ema"])
def test_activar_la_validacion_no_cambia_nada_de_la_corrida(ema_decay):
    """Medir no cambia lo medido: con y sin validación, misma seed ⇒ corrida idéntica (6.1, 6.2).

    Es el invariante que protege **todas** las celdas del estudio ya corridas: si activar la
    validación moviera la trayectoria, las corridas con y sin validación dejarían de ser
    comparables y el eje del ablation study se rompería sin que nada falle a la vista.

    El mecanismo por el que se rompería es el azar: ``dsm_loss`` y el muestreo de ``t`` **consumen**
    el ``generator`` que reciben, así que evaluar con el generator del loop correría su stream y
    cambiaría todos los pasos siguientes; y cualquier sorteo sin ``generator=`` explícito movería el
    RNG global, que además se persiste en el sidecar de resume. Por eso el fingerprint compara los
    dos streams y no solo los pesos.

    Se corre con y sin sombra EMA: con EMA hay además un swap de pesos por evaluación, que es la
    otra vía por la que la trayectoria podría contaminarse.
    """
    def corrida(con_validacion: bool):
        torch.manual_seed(777)  # mismo punto de partida del RNG global en ambas corridas
        sde, net, data, cfg = _fresh_ema(num_steps=6, checkpoint_every=2, ema_decay=ema_decay)
        g = _generador(cfg)
        extra = (
            dict(val_batches=_fuente_examen(), train_exam_batches=_fuente_examen(seed=7))
            if con_validacion
            else {}
        )
        res = train(sde, net, data, cfg, generator=g, **extra)
        return res, _fingerprint(res, res.net, g)

    res_con, fp_con = corrida(True)
    _res_sin, fp_sin = corrida(False)

    assert _pasos_de_val(res_con) == [2, 4, 6]  # la validación efectivamente corrió
    _assert_mismo_fingerprint(fp_con, fp_sin)


# ------------------- validation-loss: la serie en el meta del checkpoint (task 5)
#
# Feature `validation-loss`, tarea 5, lado de la **escritura**: la serie dispersa viaja en el ``meta``
# del checkpoint de PESOS —la misma ruta que el ``history``, no el sidecar de resume— y **solo si no
# está vacía**, con el mismo patrón de clave opcional que ya usa ``ema`` acá (y ``ema_state`` /
# ``scaler_state`` en el sidecar). Sin validación el ``meta`` conserva exactamente las claves de
# antes de esta feature (5.5): es lo que hace que los checkpoints ya escritos sigan siendo del mismo
# formato y que las celdas del estudio ya corridas no cambien de artefacto.
#
# El lado de la **lectura** (``load_resume``), la reanudación end-to-end y la estabilidad del examen
# tras el corte viven en ``test_resume.py``, la suite dueña del resolver.

#: Serie sintética de dos puntos con las cuatro claves de ``ValPoint``, incluida la ausencia de EMA.
_SERIE_VAL: list[dict] = [
    {"step": 2, "raw": 1.5, "ema": None, "train": 1.2},
    {"step": 4, "raw": 1.1, "ema": 1.0, "train": 0.9},
]

#: Claves del ``meta`` de una corrida sin validación (con receta): la expectativa PREVIA a la feature.
_META_SIN_VAL = {"sde_name", "data_dim", "history", "model"}


def test_save_checkpoint_con_serie_persiste_val_history_en_el_meta(tmp_path):
    """Con serie no vacía el ``meta`` gana ``val_history`` y la serie vuelve intacta del disco (5.2).

    La serie viaja por la **misma ruta que el ``history``** (el ``meta`` del checkpoint de pesos), así
    que un snapshot es autosuficiente para reconstruir las dos curvas. Se verifica además que cada
    punto vuelva del disco como un ``dict`` **pelado**: ``ValPoint`` es un ``TypedDict`` justamente
    para que leer el checkpoint no dependa de que la clase exista o sea importable, y convertirlo a un
    dataclass o a una tupla en el camino al disco rompería esa propiedad sin que nada más falle.
    """
    result, model_spec, _crudos = _corrida_publicable(num_steps=4)
    result.val_history = [dict(punto) for punto in _SERIE_VAL]

    path = tmp_path / "vp_con_val.pt"
    save_checkpoint(result, path, model_spec=model_spec)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert set(blob) == {"model_state", "meta"}  # el formato del blob no cambia

    _state_dict, meta = load_checkpoint(path)
    assert set(meta) == _META_SIN_VAL | {"val_history"}  # la única clave nueva
    assert meta["val_history"] == _SERIE_VAL
    for punto in meta["val_history"]:
        # ``type(...) is dict``, no ``isinstance``: un dataclass o un ``UserDict`` pasarían el
        # isinstance de un Mapping y sí atarían la lectura del archivo a la clase.
        assert type(punto) is dict
        assert set(punto) == {"step", "raw", "ema", "train"}


def test_save_checkpoint_sin_validacion_deja_el_meta_como_antes_de_la_feature(tmp_path):
    """Serie vacía ⇒ el ``meta`` NO gana ninguna clave: idéntico al de antes de la feature (5.5).

    La "palanca" de esta tarea es que la serie no esté vacía, así que una corrida sin validación
    —todas las anteriores a la feature, y todas las que no declaren ``data.val_root``— tiene que
    producir el ``meta`` de siempre. Se compara el **conjunto de claves** contra la expectativa previa
    (la misma que asevera ``test_checkpoint_roundtrip``), no solo la ausencia de la clave nueva: así
    el test también falla si alguien escribiera la clave con un valor "vacío" en lugar de omitirla.
    """
    result, model_spec, _crudos = _corrida_publicable(num_steps=4)
    assert result.val_history == []  # corrida sin fuente de validación

    path = tmp_path / "vp_sin_val.pt"
    save_checkpoint(result, path, model_spec=model_spec)

    _state_dict, meta = load_checkpoint(path)
    assert set(meta) == _META_SIN_VAL
    assert "val_history" not in meta


def test_el_hermano_de_crudos_lleva_la_serie_de_la_misma_corrida(tmp_path):
    """El hermano ``_raw`` también lleva la serie: es la **misma** corrida (5.2).

    ``raw_sibling`` copia el ``meta`` del principal salvo la marca ``ema`` (que reemplaza por
    ``raw_of``), así que la serie viaja con él por construcción. Es lo que corresponde —los dos
    checkpoints son dos publicaciones de pesos del mismo entrenamiento, y la serie describe el
    entrenamiento, no el juego de pesos—, pero conviene que sea **deliberado** y no un accidente del
    filtrado: si mañana el hermano se armara desde cero en lugar de copiar el ``meta``, este test lo
    delata.
    """
    result, model_spec, _crudos = _corrida_publicable(num_steps=6, ema_decay=0.6)
    result.val_history = [dict(punto) for punto in _SERIE_VAL]

    path = tmp_path / "vp_final_con_val.pt"
    save_checkpoint(result, path, model_spec=model_spec, raw_sibling=True)

    _ema_state, ema_meta = load_checkpoint(path)
    raw_path = tmp_path / "vp_final_con_val_raw.pt"
    _raw_state, raw_meta = load_checkpoint(raw_path)

    assert ema_meta["val_history"] == _SERIE_VAL
    assert raw_meta["val_history"] == _SERIE_VAL  # misma corrida ⇒ misma serie
    # El resto del contrato del hermano no cambia: sin la marca ``ema``, con el puntero al principal.
    assert set(raw_meta) == _META_SIN_VAL | {"raw_of", "val_history"}
    assert "ema" not in raw_meta and raw_meta["raw_of"] == path.name


def test_el_hermano_de_crudos_sin_validacion_no_gana_la_clave(tmp_path):
    """Sin serie, el ``meta`` del hermano ``_raw`` también queda como antes de la feature (5.5)."""
    result, model_spec, _crudos = _corrida_publicable(num_steps=6, ema_decay=0.6)
    assert result.val_history == []

    path = tmp_path / "vp_final_sin_val.pt"
    save_checkpoint(result, path, model_spec=model_spec, raw_sibling=True)

    _raw_state, raw_meta = load_checkpoint(tmp_path / "vp_final_sin_val_raw.pt")
    assert set(raw_meta) == _META_SIN_VAL | {"raw_of"}
    assert "val_history" not in raw_meta


# ------------- validation-loss: cadencia propia de la validación (val_every)
#
# Corrección de diseño **posterior** a la feature (3.10 y 3.11 nuevos; 3.1 y 3.7 reformulados): la
# cadencia de la evaluación dejó de estar atada a ``checkpoint_every``. Atarlas mezclaba dos
# concerns — subir la resolución de la curva obligaba a escribir más snapshots (708 MB cada uno en
# la celda de gatos), así que una palanca de **observabilidad** cargaba con un costo de disco que no
# le corresponde; con ``checkpoint_every: 15000`` sobre ``num_steps: 60000`` la curva quedaba en 4
# puntos. ``val_every`` fija la cadencia por su cuenta y con el default ``None`` cae en
# ``checkpoint_every``, así que la corrida de antes no cambia en nada.
#
# ``log_every`` fue **evaluado y descartado** como cadencia alternativa: está documentado como "solo
# consola" y el CLI lo pone en 0 con ``--quiet``, así que una corrida larga en background dejaría de
# medir sola. Un flag de verbosidad no puede decidir si se mide.
#
# Se reusan los helpers de las secciones previas (``_fresh_ema`` / ``_generador`` /
# ``_fuente_examen`` / ``_pasos_de_val`` / ``_fingerprint`` / ``_assert_mismo_fingerprint``).


def test_trainconfig_declara_val_every_con_default_none():
    """El campo nuevo existe y su default es ``None`` = "usar ``checkpoint_every``" (3.10).

    El default es la palanca apagada: mientras nadie declare el campo, la cadencia efectiva es la
    de siempre y ninguna corrida ya hecha cambia de comportamiento.
    """
    assert TrainConfig().val_every is None
    assert TrainConfig(val_every=250).val_every == 250


def test_sin_val_every_la_cadencia_efectiva_es_checkpoint_every():
    """Flag-off: sin declarar ``val_every`` se evalúa en la cadencia de los snapshots (3.1).

    10 pasos con ``checkpoint_every=4`` siguen dando puntos en 4, 8 y **10** (el último paso), que
    es exactamente lo que medía la feature antes de esta corrección. Además, una llamada que ni
    menciona el campo y una que lo pasa explícitamente en ``None`` producen la **misma** corrida:
    misma serie densa, mismos pesos, misma sombra, mismo azar consumido y la misma serie dispersa.
    """
    def corrida(**overrides):
        torch.manual_seed(777)  # mismo punto de partida del RNG global en ambas corridas
        sde, net, data, cfg = _fresh_ema(
            num_steps=10, checkpoint_every=4, ema_decay=0.6, **overrides
        )
        g = _generador(cfg)
        res = train(
            sde,
            net,
            data,
            cfg,
            generator=g,
            val_batches=_fuente_examen(),
            train_exam_batches=_fuente_examen(seed=7),
        )
        return res, _fingerprint(res, res.net, g)

    res_ausente, fp_ausente = corrida()  # el campo NI SE MENCIONA
    res_explicito, fp_explicito = corrida(val_every=None)  # declarado, pero apagado

    assert _pasos_de_val(res_ausente) == [4, 8, 10]  # la cadencia de siempre
    assert res_explicito.val_history == res_ausente.val_history  # misma serie dispersa, exacta
    _assert_mismo_fingerprint(fp_explicito, fp_ausente)


def test_val_every_desacopla_la_curva_de_los_snapshots():
    """``val_every=2`` con ``checkpoint_every=5``: dos cadencias genuinamente independientes (3.10).

    En **una sola** corrida de 10 pasos: la curva gana un punto cada 2 pasos (más el último) sin que
    se escriba un snapshot de más — el ``on_checkpoint`` recibe únicamente el del paso 5 (el 10 lo
    excluye el disparador de snapshots, que deja el último paso al checkpoint final del caller).
    Ese es el punto entero de la corrección: la resolución de la curva ya no se paga en disco.
    """
    sde, net, data, cfg = _fresh_ema(num_steps=10, checkpoint_every=5, val_every=2)
    tags: list[str] = []

    res = train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        on_checkpoint=lambda tag, _snapshot: tags.append(tag),
        val_batches=_fuente_examen(),
        train_exam_batches=_fuente_examen(seed=7),
    )

    assert _pasos_de_val(res) == [2, 4, 6, 8, 10]  # la cadencia de validación, propia
    assert tags == ["step00005"]  # la de snapshots, intacta y sin puntos de más


def test_val_every_mide_con_los_snapshots_apagados():
    """``checkpoint_every=0`` + ``val_every`` = medir sin escribir un solo snapshot (3.10, 3.7).

    Antes de la corrección esta combinación era un ``ValueError``: la cadencia de la validación
    *era* la de los snapshots, así que apagar los snapshots apagaba la medición. Ahora es una
    corrida **legítima** —observabilidad sin costo de disco— y tiene que correr sin levantar nada,
    con el ``on_checkpoint`` inyectado y sin usarse.
    """
    sde, net, data, cfg = _fresh_ema(num_steps=4, checkpoint_every=0, val_every=2)
    tags: list[str] = []

    res = train(
        sde,
        net,
        data,
        cfg,
        generator=_generador(cfg),
        on_checkpoint=lambda tag, _snapshot: tags.append(tag),
        val_batches=_fuente_examen(),
        train_exam_batches=_fuente_examen(seed=7),
    )

    assert _pasos_de_val(res) == [2, 4]
    assert tags == []  # ni un snapshot escrito
    assert len(res.history) == 4


@pytest.mark.parametrize("checkpoint_every", [0, 5], ids=["sin_snapshots", "con_snapshots"])
@pytest.mark.parametrize("val_every", [0, -1])
def test_train_rechaza_val_every_menor_que_uno(val_every, checkpoint_every):
    """``val_every < 1`` ⇒ ``ValueError`` nombrando el valor, antes del primer batch (3.11).

    Un ``val_every`` declarado en 0 o negativo no es "usar el default" —eso lo dice ``None``— sino
    una cadencia imposible: se rechaza explícitamente en lugar de degradar en silencio a la de los
    snapshots. El error es el **del campo declarado**, con independencia de que ``checkpoint_every``
    sea válido: la parametrización cruzada lo fija (con ``checkpoint_every=5`` la corrida podría
    "funcionar" cayendo al default, y eso es justamente lo que no debe pasar).

    La fuente centinela convierte cualquier consumo prematuro de datos en un ``AssertionError``
    distinto del error esperado, y la fuente del examen tampoco se recorre.
    """
    sde = make_sde("vp")
    net = _small_net(sde)
    cfg = TrainConfig(
        num_steps=2, seed=0, checkpoint_every=checkpoint_every, val_every=val_every
    )
    fuente_val = _FuenteValVigilada(_fuente_examen())

    with pytest.raises(ValueError) as exc:
        train(sde, net, _DataProhibidaVal(), cfg, val_batches=fuente_val)

    mensaje = str(exc.value)
    assert "val_every" in mensaje
    assert str(val_every) in mensaje  # nombra el valor recibido
    assert fuente_val.recorridos == 0  # ni siquiera se tocó la fuente de validación


@pytest.mark.parametrize("val_every", [0, -1])
def test_train_rechaza_val_every_menor_que_uno_aunque_no_haya_validacion(val_every):
    """El valor DECLARADO se valida siempre, haya o no fuente de validación (3.11).

    El criterio 3.11 dice "if ``val_every`` **se declara** con un valor menor que 1", sin
    condicionarlo a que haya validación, y es el criterio de la casa para un knob mal formado:
    ``keep_last_checkpoints`` se rechaza aunque no haya checkpoints y ``amp`` aunque el device sea
    CPU. Un ``val_every: 0`` en un config sin ``data.val_root`` es un malentendido —el autor creyó
    estar apagando la validación, o le falta la clave de datos—, y tolerarlo esconde el error hasta
    que alguien agregue la fuente y la corrida reviente recién ahí.

    No confundir con :func:`test_val_every_sin_fuente_de_validacion_es_inerte`: un valor **válido**
    sin fuente sí es inerte (no hay nada que medir); lo que se rechaza es el valor imposible.
    """
    sde = make_sde("vp")
    net = _small_net(sde)
    cfg = TrainConfig(num_steps=2, seed=0, checkpoint_every=5, val_every=val_every)

    with pytest.raises(ValueError) as exc:
        train(sde, net, _DataProhibidaVal(), cfg)  # SIN val_batches

    assert "val_every" in str(exc.value)
    assert str(val_every) in str(exc.value)


def test_val_every_sin_fuente_de_validacion_es_inerte():
    """Declarar ``val_every`` sin validación configurada no cambia **nada** de la corrida (6.1).

    El campo describe *cada cuánto* medir, no *si* medir: sin ``val_batches`` no hay nada que medir
    y la corrida tiene que ser idéntica —pesos, serie densa, sombra y los dos streams de azar— a la
    de una config que ni lo menciona. Vale también para un valor válido pero absurdo para la
    corrida (``val_every=3`` sobre 6 pasos): nada que hacer, nada que cambiar.
    """
    def corrida(**overrides):
        torch.manual_seed(777)
        sde, net, data, cfg = _fresh_ema(
            num_steps=6, checkpoint_every=2, ema_decay=0.6, **overrides
        )
        g = _generador(cfg)
        res = train(sde, net, data, cfg, generator=g)  # sin val_batches
        return res, _fingerprint(res, res.net, g)

    res_con, fp_con = corrida(val_every=3)
    _res_sin, fp_sin = corrida()

    assert res_con.val_history == []  # sin fuente no se mide, con campo o sin él
    _assert_mismo_fingerprint(fp_con, fp_sin)


def test_build_run_acepta_val_every_y_sigue_estricto():
    """El bloque ``train:`` acepta ``val_every`` sin tocar el config layer, y sigue estricto.

    ``build_run`` valida las claves de ``train:`` por **introspección de los fields** de
    ``TrainConfig``, así que el campo nuevo queda disponible desde la config por construcción — sin
    una lista de claves paralela que mantener. Se ejercita con un ``dict`` (el mismo que produce
    ``load_config``) para no depender de PyYAML; el camino de archivo va en el test hermano. Una
    clave inválida se sigue rechazando nombrándola.
    """
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 2, "seed": 0, "checkpoint_every": 4, "val_every": 1},
    }
    spec = build_run(raw)
    assert spec.config.val_every == 1
    assert spec.config.checkpoint_every == 4  # las dos cadencias conviven, independientes

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert len(result.history) == 2

    bad = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1, "val_every_typo": 5},
    }
    with pytest.raises(ValueError, match="val_every_typo"):
        build_run(bad)


def test_build_run_desde_yaml_con_val_every(tmp_path):
    """El mismo camino desde un archivo YAML real: ``train.val_every`` llega al ``TrainConfig``.

    Complementa al test hermano cubriendo el front-end de archivo (``load_config``); se skipea si
    PyYAML no está instalado, como el resto de los tests del camino YAML de la suite. El YAML deja
    ``checkpoint_every`` en 0 a propósito: es la combinación que la corrección habilita (medir sin
    escribir snapshots) y prueba que llega entera desde el archivo.
    """
    pytest.importorskip("yaml")
    text = (
        "sde:\n"
        "  name: vp\n"
        "data:\n"
        "  shape: gaussian\n"
        "  dim: 2\n"
        "  n_samples: 128\n"
        "  batch_size: 64\n"
        "  seed: 0\n"
        "train:\n"
        "  num_steps: 2\n"
        "  seed: 0\n"
        "  checkpoint_every: 0\n"
        "  val_every: 2\n"
    )
    path = tmp_path / "run_val_every.yaml"
    path.write_text(text, encoding="utf-8")

    spec = build_run(load_config(path))
    assert spec.config.val_every == 2
    assert spec.config.checkpoint_every == 0

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert len(result.history) == 2
