"""Tests de las redes de score (`diffusion.models`): piezas compartidas + ScoreMLP + ScoreUNet."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.models import (
    REGISTRY,
    ResidualBlock,
    ScoreMLP,
    ScoreModel,
    ScoreUNet,
    SinusoidalEmbedding,
    available_models,
    make_model,
)


# --------------------------------------------------------- SinusoidalEmbedding


def test_embedding_odd_dim_raises():
    with pytest.raises(ValueError):
        SinusoidalEmbedding(embed_dim=127)


@pytest.mark.parametrize("embed_dim", [2, 8, 128])
def test_embedding_output_shape(embed_dim):
    emb = SinusoidalEmbedding(embed_dim)
    out = emb(torch.rand(16))
    assert out.shape == (16, embed_dim)


def test_embedding_accepts_both_input_shapes():
    emb = SinusoidalEmbedding(32)
    t = torch.rand(16)
    a = emb(t)
    b = emb(t.reshape(16, 1))
    assert a.shape == b.shape == (16, 32)
    assert torch.equal(a, b)


def test_embedding_is_deterministic():
    emb = SinusoidalEmbedding(64)
    t = torch.rand(10)
    assert torch.equal(emb(t), emb(t))


def test_embedding_values_bounded():
    emb = SinusoidalEmbedding(128)
    out = emb(torch.tensor([0.0, 0.5, 1.0, 7.0, 999.0]))
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


@pytest.mark.parametrize(
    "t",
    [
        torch.linspace(0.0, 1.0, 8),       # rango [0, 1]
        torch.linspace(0.0, 1000.0, 8),    # rango [0, T]
        torch.arange(8).float(),           # pasos enteros
    ],
)
def test_embedding_any_scale_finite(t):
    out = SinusoidalEmbedding(64)(t)
    assert out.shape == (8, 64)
    assert torch.all(torch.isfinite(out))


def test_embedding_interleaves_sin_cos():
    # Para t -> 0, el primer denominador es 1: sin(0)=0 (índice par),
    # cos(0)=1 (índice impar). Verifica el orden 2i=sin, 2i+1=cos.
    emb = SinusoidalEmbedding(8)
    out = emb(torch.zeros(1))[0]
    assert torch.allclose(out[0::2], torch.zeros(4))   # senos
    assert torch.allclose(out[1::2], torch.ones(4))    # cosenos


def test_embedding_denom_is_buffer_not_param():
    emb = SinusoidalEmbedding(16)
    assert "denom" in dict(emb.named_buffers())
    assert list(emb.parameters()) == []


# ------------------------------------------------------------- ResidualBlock


def test_residual_block_preserves_shape():
    block = ResidualBlock(hidden_dim=32)
    x = torch.randn(8, 32)
    assert block(x).shape == (8, 32)


def test_residual_block_bad_activation_raises():
    with pytest.raises(ValueError):
        ResidualBlock(hidden_dim=16, activation="no_existe")


# ----------------------------------------------------------------- ScoreMLP


@pytest.mark.parametrize("data_dim", [2, 4])
def test_scoremlp_output_shape(data_dim):
    net = ScoreMLP(data_dim=data_dim)
    x = torch.randn(16, data_dim)
    out = net(x, torch.rand(16))
    assert out.shape == (16, data_dim)


def test_scoremlp_accepts_both_t_shapes():
    net = ScoreMLP().eval()
    x = torch.randn(16, 2)
    t = torch.rand(16)
    with torch.no_grad():
        a = net(x, t)              # t de shape (B,)
        b = net(x, t.reshape(16, 1))  # t de shape (B, 1)
    assert a.shape == b.shape == (16, 2)
    assert torch.equal(a, b)       # ambas formas dan el mismo resultado


def test_scoremlp_is_deterministic():
    # La red es la variable de control: mismo (x, t) -> misma salida.
    net = ScoreMLP().eval()
    x, t = torch.randn(16, 2), torch.rand(16)
    with torch.no_grad():
        assert torch.equal(net(x, t), net(x, t))


def test_scoremlp_no_stochastic_layers():
    # Sin dropout ni batchnorm: la red debe ser enteramente determinística.
    net = ScoreMLP()
    for module in net.modules():
        assert not isinstance(module, torch.nn.Dropout)
        assert not isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        )


def test_scoremlp_has_trainable_params():
    net = ScoreMLP()
    n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert n > 0


def test_scoremlp_num_blocks_configurable():
    assert len(ScoreMLP(num_blocks=2).blocks) == 2
    assert len(ScoreMLP(num_blocks=6).blocks) == 6


def test_scoremlp_gradients_flow():
    net = ScoreMLP()
    x, t = torch.randn(8, 2), torch.rand(8)
    net(x, t).pow(2).sum().backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.all(torch.isfinite(g)) for g in grads)


# ----------------------------------------------------------------- ScoreUNet
# Config tiny del diseño (Testing Strategy): una instancia por resolución de
# trabajo (image_size 32 o 64) con los mismos anchos reducidos, para mantener el
# tiempo de la suite en el orden del resto del repo. El nivel 16×16 ejercita la
# atención. Estos tests fijan el CONTRATO de la red (shape, tiempo, Protocol);
# determinismo, configuración y errores se cubren en su propia sección.


def _tiny_unet(in_channels: int = 3, image_size: int = 32) -> ScoreUNet:
    """Construye una ScoreUNet con la config tiny del diseño para la resolución dada."""
    return ScoreUNet(
        in_channels=in_channels,
        image_size=image_size,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        embed_dim=8,
        time_embed_dim=16,
        groups=4,
        attn_resolutions=(16,),
    )


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("image_size", [32, 64])
def test_scoreunet_output_shape(in_channels, image_size):
    # Contrato 1.1 / 2.1 / 2.2: (B, C, H, W) -> (B, C, H, W) en float32 para los
    # canales candidatos (grises y RGB) y las resoluciones de referencia.
    net = _tiny_unet(in_channels=in_channels, image_size=image_size)
    x = torch.randn(2, in_channels, image_size, image_size)
    out = net(x, torch.rand(2))
    assert out.shape == (2, in_channels, image_size, image_size)
    assert out.dtype == torch.float32


def test_scoreunet_accepts_both_t_shapes():
    # Contrato 1.2: t como (B,) o (B, 1) -> el mismo resultado (lo normaliza el
    # embedding reusado).
    net = _tiny_unet().eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    with torch.no_grad():
        a = net(x, t)                 # t de shape (B,)
        b = net(x, t.reshape(2, 1))   # t de shape (B, 1)
    assert a.shape == b.shape == (2, 3, 32, 32)
    assert torch.equal(a, b)          # ambas formas dan el mismo resultado


@pytest.mark.parametrize(
    "t",
    [
        torch.linspace(0.0, 1.0, 2),       # rango [0, 1]
        torch.linspace(0.0, 1000.0, 2),    # rango [0, T]
        torch.arange(2).float(),           # pasos enteros
    ],
)
def test_scoreunet_any_t_scale_finite(t):
    # Contrato 1.3: salidas finitas para las escalas de tiempo usadas por las SDEs.
    net = _tiny_unet().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = net(x, t)
    assert out.shape == (2, 3, 32, 32)
    assert torch.all(torch.isfinite(out))


def test_scoreunet_time_conditioning_effective():
    # Contrato 1.4: mismo x, dos tiempos distintos -> salidas distintas (el
    # condicionamiento temporal es efectivo).
    net = _tiny_unet().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        a = net(x, torch.zeros(2))
        b = net(x, torch.ones(2))
    assert not torch.allclose(a, b)


def test_scoreunet_output_unbounded_both_signs():
    # Contrato 1.5: salida no acotada -> valores positivos y negativos (ninguna
    # activación final la restringe).
    net = _tiny_unet().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = net(x, torch.rand(2))
    assert (out > 0).any() and (out < 0).any()


def test_scoreunet_satisfies_scoremodel_protocol():
    # Contrato 1.6: satisface el Protocol ScoreModel estructuralmente (sin herencia).
    net = _tiny_unet()
    assert isinstance(net, ScoreModel)


# Determinismo (Req 3): la red es la variable de control -> enteramente determinística.


def test_scoreunet_is_deterministic():
    # Contrato 3.1: mismo (x, t) evaluado dos veces en eval -> salida bitwise idéntica
    # (mismo grafo, mismas rutas de cómputo).
    net = _tiny_unet().eval()
    x, t = torch.randn(2, 3, 32, 32), torch.rand(2)
    with torch.no_grad():
        assert torch.equal(net(x, t), net(x, t))


def test_scoreunet_no_stochastic_layers():
    # Contrato 3.2: sin dropout ni batchnorm (la normalización es GroupNorm,
    # independiente del batch); la red debe ser enteramente determinística.
    net = _tiny_unet()
    for module in net.modules():
        assert not isinstance(module, torch.nn.Dropout)
        assert not isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        )


def test_scoreunet_batch_independence():
    # Contrato 3.3: una misma muestra evaluada sola o dentro de un batch da salidas
    # numéricamente equivalentes (la normalización no depende del resto del batch).
    # No es bitwise en CPU float32 (~6e-07 por paralelización de convs); se usa
    # allclose con margen (atol=1e-5) y seed fijo para que el test sea determinístico
    # (sin seed, el ~6e-07 rozaba 1e-6 y el test era flaky). Un acople real del batch
    # produce diferencias mucho mayores (~1e-2), así que 1e-5 sigue detectándolo.
    torch.manual_seed(0)
    net = _tiny_unet().eval()
    x_single = torch.randn(1, 3, 32, 32)
    t_single = torch.rand(1)
    # La primera fila del batch es exactamente la muestra evaluada sola.
    x_batch = torch.cat([x_single, torch.randn(3, 3, 32, 32)], dim=0)
    t_batch = torch.cat([t_single, torch.rand(3)], dim=0)
    with torch.no_grad():
        out_single = net(x_single, t_single)
        out_batch = net(x_batch, t_batch)
    assert torch.allclose(out_single, out_batch[0:1], atol=1e-5)


def test_scoreunet_gradients_flow():
    # Contrato 3.4: backward sobre una salida -> gradientes finitos en todos los
    # parámetros entrenables (ninguno queda desconectado del grafo).
    net = _tiny_unet()
    x, t = torch.randn(2, 3, 32, 32), torch.rand(2)
    net(x, t).pow(2).sum().backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.all(torch.isfinite(g)) for g in grads)


# Configuración y errores (Req 4 + 2.3): defaults como arquitectura de referencia,
# conteo de parámetros reproducible y validaciones fail-fast con ValueError.


def test_scoreunet_param_count_reproducible():
    # Contrato 4.1 / 4.2: dos instancias con los MISMOS argumentos tienen exactamente
    # el mismo conteo de parámetros entrenables (la arquitectura es reproducible, sin
    # números mágicos escondidos que varíen entre construcciones).
    a = _tiny_unet()
    b = _tiny_unet()
    n_a = sum(p.numel() for p in a.parameters())
    n_b = sum(p.numel() for p in b.parameters())
    assert n_a == n_b
    assert n_a > 0


def test_scoreunet_unknown_activation_raises():
    # Contrato 4.3: activación con nombre desconocido -> ValueError (mismo registry de
    # activaciones que el resto del módulo, vía _make_activation). _tiny_unet no expone
    # activation, así que se construye ScoreUNet directo con anchos tiny + config válida.
    with pytest.raises(ValueError):
        ScoreUNet(
            in_channels=3,
            image_size=32,
            base_channels=8,
            channel_mults=(1, 2),
            num_res_blocks=1,
            embed_dim=8,
            time_embed_dim=16,
            groups=4,
            attn_resolutions=(16,),
            activation="no_existe",
        )


def test_scoreunet_incompatible_groups_raises():
    # Contrato 2.3 / 4.3: groups debe dividir a todos los anchos de canal; groups=3
    # contra el nivel base de 8 (8 % 3 != 0) -> ValueError en construcción. image_size
    # y embed_dim se dejan válidos para aislar la infracción de grupos.
    with pytest.raises(ValueError):
        ScoreUNet(
            in_channels=3,
            image_size=32,
            base_channels=8,
            channel_mults=(1, 2),
            num_res_blocks=1,
            embed_dim=8,
            time_embed_dim=16,
            groups=3,
            attn_resolutions=(16,),
        )


def test_scoreunet_indivisible_image_size_raises():
    # Contrato 2.3: la resolución de trabajo debe ser divisible por el factor total de
    # reducción 2**(len(channel_mults)-1); con channel_mults=(1, 2) el factor es 2 y un
    # image_size impar (15) no es divisible -> ValueError en construcción.
    with pytest.raises(ValueError):
        ScoreUNet(
            in_channels=3,
            image_size=15,
            base_channels=8,
            channel_mults=(1, 2),
            num_res_blocks=1,
            embed_dim=8,
            time_embed_dim=16,
            groups=4,
            attn_resolutions=(16,),
        )


def test_scoreunet_wrong_input_size_raises():
    # Contrato 2.3: la arquitectura queda fijada por image_size en construcción; un
    # forward con H/W distintos de image_size (red de 32, entrada de 16) -> ValueError.
    net = _tiny_unet(image_size=32)
    x = torch.randn(2, 3, 16, 16)
    with pytest.raises(ValueError):
        net(x, torch.rand(2))


def test_scoreunet_reference_defaults_forward():
    # Contrato 4.1 / 5.4: ÚNICO test que instancia los defaults (la arquitectura de
    # referencia del estudio) y corre un forward completo bajo pytest. Excepción
    # deliberada a la config tiny (research.md, ~100-200 ms): batch 1 y sin parametrizar
    # para no multiplicar el costo. Cubre el camino de 4 niveles con mult 4.
    net = ScoreUNet()
    x = torch.randn(1, 3, 64, 64)
    out = net(x, torch.rand(1))
    assert out.shape == (1, 3, 64, 64)


# ------------------------------------------------ make_model / registry (Req 5.3, 6.1)
# Factory por nombre additivo, espejo de make_sde / make_distribution: construye la red
# desde una receta (name, kwargs) para el config-driven y la reconstrucción de checkpoints.


def test_make_model_mlp_returns_scoremlp():
    # make_model("mlp", ...) devuelve un ScoreMLP usable (nn.Module con forward válido).
    net = make_model("mlp", data_dim=2)
    assert isinstance(net, ScoreMLP)
    assert isinstance(net, torch.nn.Module)
    out = net(torch.randn(4, 2), torch.rand(4))
    assert out.shape == (4, 2)


def test_make_model_unet_returns_scoreunet():
    # make_model("unet", ...) devuelve un ScoreUNet usable; se pasan los anchos tiny para
    # mantener el costo del test en el orden del resto de la suite.
    net = make_model(
        "unet",
        in_channels=3,
        image_size=32,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        embed_dim=8,
        time_embed_dim=16,
        groups=4,
        attn_resolutions=(16,),
    )
    assert isinstance(net, ScoreUNet)
    assert isinstance(net, torch.nn.Module)
    out = net(torch.randn(2, 3, 32, 32), torch.rand(2))
    assert out.shape == (2, 3, 32, 32)


def test_make_model_satisfies_scoremodel_protocol():
    # Lo que construye el registry satisface el Protocol ScoreModel (contrato (x, t) -> score).
    assert isinstance(make_model("mlp", data_dim=2), ScoreModel)


def test_available_models_expected_set():
    # available_models() == conjunto esperado, y REGISTRY mapea a las clases correctas.
    assert set(available_models()) == {"mlp", "unet"}
    assert REGISTRY["mlp"] is ScoreMLP
    assert REGISTRY["unet"] is ScoreUNet


def test_make_model_unknown_name_raises():
    # Nombre desconocido -> ValueError que nombra las opciones válidas (patrón del repo).
    with pytest.raises(ValueError):
        make_model("no_existe")


def test_make_model_filters_unknown_kwargs():
    # Espejo de make_sde / make_distribution: los kwargs que no aplican a la red se
    # descartan (se filtran por la firma del constructor), así un caller genérico puede
    # pasar siempre el mismo conjunto de parámetros sin que falle la construcción.
    net = make_model("mlp", data_dim=2, no_aplica_a_mlp=123)
    assert isinstance(net, ScoreMLP)
    assert net.data_dim == 2


# ------------------------------- SinusoidalEmbedding: escala temporal (Req 1, 6.1)
# Spec time-embedding-scale: escala configurable aplicada al tiempo antes de la
# codificación, retrocompatible por default (scale=1.0), con validación fail-fast.


def _closed_form_embedding(t, embed_dim):
    """Fórmula cerrada vigente: sin/cos(t / 10000^{2i/d}) con sin/cos intercalados."""
    i = torch.arange(embed_dim // 2, dtype=torch.float32)
    denom = torch.pow(10000.0, (2.0 * i) / embed_dim)
    t = t.reshape(-1)
    args = t[:, None] / denom[None, :]
    emb = torch.stack((torch.sin(args), torch.cos(args)), dim=-1)
    return emb.reshape(t.shape[0], embed_dim)


def test_embedding_default_matches_closed_form():
    # Req 1.2: sin escala explícita, la salida es idéntica bit a bit a la
    # implementación vigente (misma fórmula cerrada, mismas operaciones).
    t = torch.tensor([0.0, 1e-4, 1e-2, 0.5, 1.0, 999.0])
    assert torch.equal(SinusoidalEmbedding(64)(t), _closed_form_embedding(t, 64))


def test_embedding_scale_one_identical_to_default():
    # Req 1.2: scale=1.0 explícito == construcción sin el kwarg, bit a bit.
    t = torch.rand(16)
    a = SinusoidalEmbedding(64, scale=1.0)(t)
    b = SinusoidalEmbedding(64)(t)
    assert torch.equal(a, b)


def test_embedding_scale_applied_before_encoding():
    # Req 1.1 / postcondición del diseño: forward(t) con escala s equivale a la
    # codificación vigente evaluada en t * s (bit a bit: es la misma operación).
    t = torch.rand(16)
    scaled = SinusoidalEmbedding(64, scale=1000.0)(t)
    reference = SinusoidalEmbedding(64)(t * 1000.0)
    assert torch.equal(scaled, reference)


def test_embedding_scale_resolves_small_t():
    # Req 1.3: con la escala recomendada (1000), la distancia euclídea entre los
    # embeddings de t=1e-4 y t=1e-2 es al menos 50x la obtenida con el default
    # (la codificación pasa a distinguir tiempos chicos).
    t_lo = torch.tensor([1e-4])
    t_hi = torch.tensor([1e-2])

    def dist(scale: float) -> float:
        emb = SinusoidalEmbedding(64, scale=scale)
        return torch.linalg.vector_norm(emb(t_lo) - emb(t_hi)).item()

    assert dist(1000.0) >= 50.0 * dist(1.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_embedding_invalid_scale_raises(scale):
    # Req 1.4: escala no positiva o no finita -> ValueError en construcción cuyo
    # mensaje incluye el valor recibido.
    with pytest.raises(ValueError, match=str(scale)):
        SinusoidalEmbedding(64, scale=scale)


def test_embedding_scaled_keeps_shape_dtype_contract():
    # Req 1.5: con escala activa, t como (B,) o (B, 1) -> mismo resultado bit a bit,
    # salida (B, embed_dim) float32 y finita (el contrato vigente no cambia).
    emb = SinusoidalEmbedding(32, scale=1000.0)
    t = torch.rand(16)
    a = emb(t)
    b = emb(t.reshape(16, 1))
    assert a.shape == b.shape == (16, 32)
    assert torch.equal(a, b)
    assert a.dtype == torch.float32
    assert torch.all(torch.isfinite(a))


def test_embedding_scale_no_new_params_or_buffers():
    # Invariante del diseño: la escala no agrega parámetros entrenables ni buffers
    # (solo el buffer denom existente) y queda expuesta como atributo de introspección.
    emb = SinusoidalEmbedding(16, scale=1000.0)
    assert list(emb.parameters()) == []
    assert list(dict(emb.named_buffers())) == ["denom"]
    assert emb.scale == 1000.0


# --------------------------------- ScoreMLP: passthrough de time_scale (Req 2, 6.1)
# Spec time-embedding-scale: la red 2D acepta `time_scale` en construcción, lo pasa
# al embedding compartido (passthrough puro, sin re-validar ni re-aplicar la escala)
# y lo expone como atributo de introspección. Default retrocompatible (1.0).


def test_scoremlp_default_forward_identical_without_kwarg():
    # Req 2.5: sin el kwarg y con time_scale=1.0 explícito, el forward es idéntico
    # bit a bit (misma seed de init -> mismos pesos -> misma salida).
    x, t = torch.randn(16, 2), torch.rand(16)
    torch.manual_seed(0)
    net_default = ScoreMLP().eval()
    torch.manual_seed(0)
    net_explicit = ScoreMLP(time_scale=1.0).eval()
    with torch.no_grad():
        assert torch.equal(net_default(x, t), net_explicit(x, t))


def test_scoremlp_time_scale_passthrough_to_embedding():
    # Req 2.1: el kwarg llega al embedding compartido y queda expuesto para
    # introspección (self.time_scale coincide con la escala del embedding).
    net = ScoreMLP(time_scale=1000.0)
    assert net.time_scale == 1000.0
    assert net.time_embed.scale == 1000.0
    # Default: sin el kwarg, la escala es 1.0 (retrocompatible).
    assert ScoreMLP().time_scale == 1.0


def test_scoremlp_time_scale_changes_forward():
    # Req 2.1: la escala se aplica de verdad — con los mismos pesos (misma seed),
    # una escala distinta de 1.0 produce un forward distinto para t no trivial.
    x, t = torch.randn(16, 2), torch.rand(16)
    torch.manual_seed(0)
    net_default = ScoreMLP().eval()
    torch.manual_seed(0)
    net_scaled = ScoreMLP(time_scale=1000.0).eval()
    with torch.no_grad():
        assert not torch.allclose(net_default(x, t), net_scaled(x, t))


@pytest.mark.parametrize("time_scale", [0.0, float("nan")])
def test_scoremlp_invalid_time_scale_raises(time_scale):
    # Req 2.1 / 1.4: la validación fail-fast del embedding se propaga por la red
    # (ValueError en construcción con el valor recibido; la red no la re-implementa).
    with pytest.raises(ValueError, match=str(time_scale)):
        ScoreMLP(time_scale=time_scale)


def test_scoremlp_scaled_is_deterministic():
    # Req 2.4: con la escala activa la red sigue determinística — dos forwards con
    # los mismos insumos son bit a bit idénticos.
    net = ScoreMLP(time_scale=1000.0).eval()
    x, t = torch.randn(16, 2), torch.rand(16)
    with torch.no_grad():
        assert torch.equal(net(x, t), net(x, t))


def test_scoremlp_time_scale_no_new_params_or_shapes():
    # Req 2.4: el kwarg no agrega parámetros entrenables ni cambia el state_dict —
    # mismo conteo y mismo set de claves con las mismas shapes con/sin kwarg.
    net_default = ScoreMLP()
    net_scaled = ScoreMLP(time_scale=1000.0)
    n_default = sum(p.numel() for p in net_default.parameters() if p.requires_grad)
    n_scaled = sum(p.numel() for p in net_scaled.parameters() if p.requires_grad)
    assert n_default == n_scaled
    sd_default = net_default.state_dict()
    sd_scaled = net_scaled.state_dict()
    assert set(sd_default) == set(sd_scaled)
    assert all(sd_default[k].shape == sd_scaled[k].shape for k in sd_default)


# ---------------------- TimeMLP + ScoreUNet: passthrough de time_scale (Req 2, 6.1)
# Spec time-embedding-scale: la red de imágenes acepta `time_scale` en construcción,
# lo transporta al embedding compartido vía TimeMLP (passthrough puro, sin re-validar
# ni re-aplicar la escala) y ambas clases lo exponen como atributo de introspección.
# Default retrocompatible (1.0). Config tiny (misma que _tiny_unet) para CPU rápida.


def _tiny_unet_kwargs() -> dict:
    """Kwargs de la config tiny de _tiny_unet, para construir pares con/sin time_scale."""
    return dict(
        in_channels=3,
        image_size=32,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        embed_dim=8,
        time_embed_dim=16,
        groups=4,
        attn_resolutions=(16,),
    )


def test_scoreunet_default_forward_identical_without_kwarg():
    # Req 2.5: sin el kwarg y con time_scale=1.0 explícito, el forward es idéntico
    # bit a bit (misma seed de init -> mismos pesos -> misma salida).
    x, t = torch.randn(2, 3, 32, 32), torch.rand(2)
    torch.manual_seed(0)
    net_default = ScoreUNet(**_tiny_unet_kwargs()).eval()
    torch.manual_seed(0)
    net_explicit = ScoreUNet(**_tiny_unet_kwargs(), time_scale=1.0).eval()
    with torch.no_grad():
        assert torch.equal(net_default(x, t), net_explicit(x, t))


def test_scoreunet_time_scale_passthrough_to_embedding():
    # Req 2.2: el kwarg atraviesa ScoreUNet -> TimeMLP -> embedding compartido y
    # queda expuesto para introspección en cada eslabón de la cadena.
    net = ScoreUNet(**_tiny_unet_kwargs(), time_scale=1000.0)
    assert net.time_scale == 1000.0
    assert net.time_mlp.time_scale == 1000.0
    assert net.time_mlp.embed.scale == 1000.0
    # Default: sin el kwarg, la escala es 1.0 (retrocompatible).
    assert ScoreUNet(**_tiny_unet_kwargs()).time_scale == 1.0


def test_timemlp_time_scale_passthrough_direct():
    # Req 2.2: TimeMLP (la proyección temporal) también acepta el kwarg, lo pasa al
    # embedding y lo expone; default retrocompatible 1.0.
    from diffusion.models.unet import TimeMLP

    proj = TimeMLP(embed_dim=8, time_embed_dim=16, time_scale=1000.0)
    assert proj.time_scale == 1000.0
    assert proj.embed.scale == 1000.0
    assert TimeMLP(embed_dim=8, time_embed_dim=16).time_scale == 1.0


def test_scoreunet_time_scale_changes_forward():
    # Req 2.2: la escala se aplica de verdad — con los mismos pesos (misma seed),
    # una escala distinta de 1.0 produce un forward distinto para t no trivial.
    x, t = torch.randn(2, 3, 32, 32), torch.rand(2)
    torch.manual_seed(0)
    net_default = ScoreUNet(**_tiny_unet_kwargs()).eval()
    torch.manual_seed(0)
    net_scaled = ScoreUNet(**_tiny_unet_kwargs(), time_scale=1000.0).eval()
    with torch.no_grad():
        assert not torch.allclose(net_default(x, t), net_scaled(x, t))


@pytest.mark.parametrize("time_scale", [0.0, float("nan")])
def test_scoreunet_invalid_time_scale_raises(time_scale):
    # Req 2.2 / 1.4: la validación fail-fast del embedding se propaga por la red
    # (ValueError en construcción con el valor recibido; la red no la re-implementa).
    with pytest.raises(ValueError, match=str(time_scale)):
        ScoreUNet(**_tiny_unet_kwargs(), time_scale=time_scale)


def test_scoreunet_scaled_is_deterministic():
    # Req 2.4: con la escala activa la red sigue determinística — dos forwards con
    # los mismos insumos son bit a bit idénticos.
    net = ScoreUNet(**_tiny_unet_kwargs(), time_scale=1000.0).eval()
    x, t = torch.randn(2, 3, 32, 32), torch.rand(2)
    with torch.no_grad():
        assert torch.equal(net(x, t), net(x, t))


def test_scoreunet_time_scale_no_new_params_or_shapes():
    # Req 2.4: el kwarg no agrega parámetros entrenables ni cambia el state_dict —
    # mismo conteo y mismo set de claves con las mismas shapes con/sin kwarg.
    net_default = ScoreUNet(**_tiny_unet_kwargs())
    net_scaled = ScoreUNet(**_tiny_unet_kwargs(), time_scale=1000.0)
    n_default = sum(p.numel() for p in net_default.parameters() if p.requires_grad)
    n_scaled = sum(p.numel() for p in net_scaled.parameters() if p.requires_grad)
    assert n_default == n_scaled
    sd_default = net_default.state_dict()
    sd_scaled = net_scaled.state_dict()
    assert set(sd_default) == set(sd_scaled)
    assert all(sd_default[k].shape == sd_scaled[k].shape for k in sd_default)


# ------------------- Factory + paridad entre redes: time_scale (Req 2.3, 2.4, 6.1)
# Spec time-embedding-scale, tarea 2.3: la factory por nombre acepta el kwarg nuevo
# sin cambios en su mecanismo de filtrado por firma (kwarg ausente -> default), y el
# embedding crudo de ambas redes coincide para el mismo t, embed_dim y time_scale.


def test_make_model_mlp_time_scale_passthrough():
    # Req 2.3: make_model("mlp", time_scale=...) construye con el valor (el kwarg
    # atraviesa el filtrado por firma existente hasta el embedding compartido).
    net = make_model("mlp", data_dim=2, time_scale=1000.0)
    assert isinstance(net, ScoreMLP)
    assert net.time_scale == 1000.0
    assert net.time_embed.scale == 1000.0


def test_make_model_mlp_time_scale_absent_defaults():
    # Req 2.3: sin el kwarg, la factory produce el default retrocompatible (1.0).
    net = make_model("mlp", data_dim=2)
    assert net.time_scale == 1.0
    assert net.time_embed.scale == 1.0


def test_make_model_unet_time_scale_passthrough():
    # Req 2.3: make_model("unet", time_scale=...) construye con el valor; anchos
    # tiny (misma config que _tiny_unet_kwargs) para mantener el costo de la suite.
    net = make_model("unet", **_tiny_unet_kwargs(), time_scale=1000.0)
    assert isinstance(net, ScoreUNet)
    assert net.time_scale == 1000.0
    assert net.time_mlp.embed.scale == 1000.0


def test_make_model_unet_time_scale_absent_defaults():
    # Req 2.3: sin el kwarg, la factory produce el default retrocompatible (1.0).
    net = make_model("unet", **_tiny_unet_kwargs())
    assert net.time_scale == 1.0
    assert net.time_mlp.embed.scale == 1.0


def test_make_model_filtering_intact_with_time_scale():
    # Req 2.3: el mecanismo de filtrado por firma sigue intacto tras el kwarg nuevo —
    # un kwarg que no aplica a la red se descarta en silencio (comportamiento
    # documentado en make_model y ya cubierto para mlp sin time_scale), mientras
    # time_scale sí llega al constructor. Se verifica en ambas redes.
    mlp = make_model("mlp", data_dim=2, time_scale=1000.0, no_aplica=123)
    assert isinstance(mlp, ScoreMLP)
    assert mlp.time_scale == 1000.0
    unet = make_model("unet", **_tiny_unet_kwargs(), time_scale=1000.0, no_aplica=123)
    assert isinstance(unet, ScoreUNet)
    assert unet.time_scale == 1000.0


@pytest.mark.parametrize("time_scale", [1.0, 1000.0])
def test_embedding_parity_mlp_unet(time_scale):
    # Req 2.4 / invariante del diseño: para el mismo t, la misma dimensión de
    # embedding y la misma escala, el embedding crudo de MLP y U-Net coincide bit a
    # bit (ambas redes transportan la escala a la MISMA capa compartida, sin
    # re-aplicarla ni re-implementarla). Se cubre el default (1.0) y la escala
    # recomendada (1000).
    embed_dim = 16
    mlp = ScoreMLP(embed_dim=embed_dim, time_scale=time_scale)
    unet_kwargs = _tiny_unet_kwargs() | {"embed_dim": embed_dim}
    unet = ScoreUNet(**unet_kwargs, time_scale=time_scale)
    t = torch.tensor([0.0, 1e-4, 1e-3, 1e-2, 0.5, 1.0])
    with torch.no_grad():
        emb_mlp = mlp.time_embed(t)
        emb_unet = unet.time_mlp.embed(t)
    assert emb_mlp.shape == emb_unet.shape == (6, embed_dim)
    assert torch.equal(emb_mlp, emb_unet)


def test_embedding_parity_not_vacuous_across_scales():
    # Sanidad del test de paridad: la igualdad anterior no es vacía — con escalas
    # DISTINTAS los embeddings crudos difieren para t no trivial (es decir, la
    # paridad depende de que time_scale llegue de verdad a ambos embeddings).
    embed_dim = 16
    mlp = ScoreMLP(embed_dim=embed_dim, time_scale=1000.0)
    unet = ScoreUNet(**(_tiny_unet_kwargs() | {"embed_dim": embed_dim}))
    t = torch.tensor([1e-3, 1e-2, 0.5])
    with torch.no_grad():
        assert not torch.allclose(mlp.time_embed(t), unet.time_mlp.embed(t))


# --------------------- EpsilonScoreWrapper: parametrización ε (Req 2.1, 2.2, 2.7, 5.1)
# Spec small-t-training-signal: la red interna predice una cantidad acotada (unidades de
# ε) y el score consumido es -inner(x,t)/clamp(σ(x,t), 1e-5). La σ llega como callable
# opaco (models no importa sde; estos TESTS sí pueden, para construir el callable real) y
# el state_dict es transparente: el checkpoint sigue siendo de red pelada, reconstruible
# con make_model.

from diffusion.models import EpsilonScoreWrapper  # noqa: E402


def _tiny_mlp(seed: int = 0) -> ScoreMLP:
    """MLP chico con seed fija, para tests CPU-rápidos y reproducibles del wrapper."""
    torch.manual_seed(seed)
    return ScoreMLP(data_dim=2, embed_dim=16, hidden_dim=32, num_blocks=1)


def _vp_std_callable():
    """La σ real de una VP-SDE como callable opaco (patrón del diseño: get_score_fn)."""
    from diffusion.sde import make_sde

    sde = make_sde("vp", data_dim=2)
    return sde, (lambda x, t: sde.marginal_prob(x, t)[1])


def test_wrapper_parity_synthetic_sigma():
    # Req 2.1: paridad numérica contra la fórmula con una σ sintética constante.
    net = _tiny_mlp().eval()
    sigma = lambda x, t: torch.full((x.shape[0], 1), 0.5)  # noqa: E731
    wrapper = EpsilonScoreWrapper(net, sigma).eval()
    x, t = torch.randn(16, 2), torch.rand(16)
    with torch.no_grad():
        expected = -net(x, t) / torch.clamp(sigma(x, t), min=1e-5)
        assert torch.allclose(wrapper(x, t), expected)
        assert wrapper(x, t).shape == x.shape


def test_wrapper_parity_vp_sigma():
    # Req 2.1: paridad numérica con la σ real de make_sde("vp") (el callable que
    # construirán los call sites: lambda x, t: sde.marginal_prob(x, t)[1]).
    sde, sigma = _vp_std_callable()
    net = _tiny_mlp().eval()
    wrapper = EpsilonScoreWrapper(net, sigma).eval()
    x = torch.randn(16, 2)
    t = torch.tensor([1e-3, 1e-2, 0.1, 0.5] * 4)
    with torch.no_grad():
        std = sde.marginal_prob(x, t)[1]
        expected = -net(x, t) / torch.clamp(std, min=1e-5)
        assert torch.allclose(wrapper(x, t), expected)


def test_wrapper_clamp_floor_active():
    # Req 2.1 / numerics.md: con σ < 1e-5 el piso se activa — la división usa 1e-5 y
    # la salida queda finita (nunca división por ~0).
    net = _tiny_mlp().eval()
    sigma = lambda x, t: torch.full((x.shape[0], 1), 1e-8)  # noqa: E731
    wrapper = EpsilonScoreWrapper(net, sigma).eval()
    x, t = torch.randn(8, 2), torch.rand(8)
    with torch.no_grad():
        expected = -net(x, t) / 1e-5
        assert torch.allclose(wrapper(x, t), expected)
        assert torch.all(torch.isfinite(wrapper(x, t)))


def test_wrapper_state_dict_transparent():
    # Prerrequisito de 2.5: claves y tensores del wrapper == los del interno, SIN
    # prefijo (el checkpoint sigue siendo de red pelada).
    net = _tiny_mlp()
    wrapper = EpsilonScoreWrapper(net, lambda x, t: torch.ones(x.shape[0], 1))
    sd_wrapper = wrapper.state_dict()
    sd_inner = net.state_dict()
    assert set(sd_wrapper) == set(sd_inner)
    assert all(torch.equal(sd_wrapper[k], sd_inner[k]) for k in sd_inner)


def test_wrapper_trained_state_dict_reloads_into_bare_net():
    # Prerrequisito de 2.5: entrenar el wrapper 2 pasos (optimizador sobre
    # wrapper.parameters()) y reconstruir por factory una red pelada desde su
    # state_dict — debe cargar sin error y reproducir el forward del interno.
    from diffusion.training import dsm_loss

    sde, sigma = _vp_std_callable()
    wrapper = EpsilonScoreWrapper(_tiny_mlp(), sigma)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(wrapper.parameters(), lr=1e-3)
    for _ in range(2):
        x0 = torch.randn(32, 2, generator=gen)
        t = 1e-3 + (1.0 - 1e-3) * torch.rand(32, generator=gen)
        loss = dsm_loss(wrapper, sde, x0, t, generator=gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    bare = make_model("mlp", data_dim=2, embed_dim=16, hidden_dim=32, num_blocks=1)
    bare.load_state_dict(wrapper.state_dict())
    bare.eval()
    wrapper.eval()
    x, t = torch.randn(16, 2), torch.rand(16)
    with torch.no_grad():
        assert torch.equal(bare(x, t), wrapper.inner(x, t))


def test_wrapper_load_state_dict_accepts_bare_net():
    # Prerrequisito de 2.5/2.6: load_state_dict acepta el state_dict de una red
    # pelada (mismas claves sin prefijo) y el interno queda con esos pesos.
    donor = _tiny_mlp(seed=1)
    wrapper = EpsilonScoreWrapper(_tiny_mlp(seed=0), lambda x, t: torch.ones(x.shape[0], 1))
    wrapper.load_state_dict(donor.state_dict())
    wrapper.eval()
    donor.eval()
    x, t = torch.randn(16, 2), torch.rand(16)
    with torch.no_grad():
        assert torch.equal(wrapper.inner(x, t), donor(x, t))


def test_wrapper_deterministic():
    # Req 2.7: sin fuentes de aleatoriedad — dos forwards idénticos bit a bit.
    _, sigma = _vp_std_callable()
    wrapper = EpsilonScoreWrapper(_tiny_mlp(), sigma).eval()
    x, t = torch.randn(16, 2), torch.rand(16)
    with torch.no_grad():
        assert torch.equal(wrapper(x, t), wrapper(x, t))


def test_wrapper_no_new_parameters():
    # Req 2.7: el wrapper no agrega parámetros entrenables — mismo conteo y misma
    # cantidad de tensores de parámetros que el interno.
    net = _tiny_mlp()
    wrapper = EpsilonScoreWrapper(net, lambda x, t: torch.ones(x.shape[0], 1))
    assert len(list(wrapper.parameters())) == len(list(net.parameters()))
    n_wrapper = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    n_inner = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert n_wrapper == n_inner


def test_wrapper_exposes_inner_and_parametrization():
    # Contrato del diseño: expone el interno (.inner) y el nombre de la
    # parametrización ("epsilon"); satisface el Protocol ScoreModel; y las
    # delegaciones de nn.Module no rompen (smoke de .to/.train/.eval).
    net = _tiny_mlp()
    wrapper = EpsilonScoreWrapper(net, lambda x, t: torch.ones(x.shape[0], 1))
    assert wrapper.inner is net
    assert wrapper.parametrization == "epsilon"
    assert isinstance(wrapper, ScoreModel)
    wrapper.to("cpu")
    wrapper.train()
    assert net.training
    wrapper.eval()
    assert not net.training
    out = wrapper(torch.randn(4, 2), torch.rand(4))
    assert out.shape == (4, 2)


def test_wrapper_inner_stays_in_epsilon_scale_on_delta_data():
    # Req 2.2 (test funcional): con dato tipo delta (2 puntos fijos) y la pérdida DSM
    # real, el pesado vigente λ=σ² deja la regresión interna en unidades de ε — tras
    # ~80 pasos, en t=1e-3 la magnitud típica del interno sigue O(1) (< 10) mientras
    # el score del wrapper es grande (≥ 10× la del interno; σ_VP(1e-3) ≈ 0.01).
    from diffusion.training import dsm_loss

    sde, sigma = _vp_std_callable()
    net = _tiny_mlp()
    wrapper = EpsilonScoreWrapper(net, sigma)
    x0_points = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])  # dato delta: 2 puntos fijos
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(wrapper.parameters(), lr=3e-3)
    wrapper.train()
    for _ in range(80):
        idx = torch.randint(0, 2, (64,), generator=gen)
        x0 = x0_points[idx]
        t = 1e-3 + (1.0 - 1e-3) * torch.rand(64, generator=gen)
        loss = dsm_loss(wrapper, sde, x0, t, generator=gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    wrapper.eval()
    idx = torch.randint(0, 2, (256,), generator=gen)
    x0 = x0_points[idx]
    t_small = torch.full((256,), 1e-3)
    x_t, _ = sde.perturb(x0, t_small, generator=gen)
    with torch.no_grad():
        inner_mag = net(x_t, t_small).abs().mean().item()
        wrapper_mag = wrapper(x_t, t_small).abs().mean().item()
    assert inner_mag < 10.0                     # el interno queda acotado (unidades de ε)
    assert wrapper_mag >= 10.0 * inner_mag      # el score sale de la división por σ
    assert wrapper_mag > 10.0                   # y es grande en términos absolutos
