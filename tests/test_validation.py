"""Tests del examen fijo de validación (`diffusion.training.validation`).

Cubre las dos piezas del submódulo: :class:`FixedValExam` (la métrica congelada) y
:func:`evaluate_with_weights` (la evaluación con un juego de pesos ajeno, que deja la red
exactamente como estaba).

Torch es dependencia dura del módulo, así que se hace `importorskip` al tope. Todos los tests
usan **listas de tensores en memoria** como fuente (sin tocar el disco ni el loader de imágenes):
el evaluador solo pide una fuente re-iterable de batches, así que una lista alcanza para fijar
su contrato.

Varios tests usan una red artificial cuya pérdida es una **constante exacta**, independiente del
``t`` y del ruido sorteados (ver :class:`_LossConstante`). Es lo que permite aislar el estimador
—el promedio ponderado por imágenes— del azar del examen: con la re-siembra, el ``t`` y el ruido
que le toca a cada imagen dependen de su posición en el stream del generator, así que dos
batcheos distintos del mismo conjunto sortean valores distintos; lo que el criterio 3.2 fija es
que el **promedio** sea por imagen y no por batch.
"""

from __future__ import annotations

import math
from types import MappingProxyType

import pytest

torch = pytest.importorskip("torch")

from diffusion.models import EpsilonScoreWrapper, ScoreMLP
from diffusion.sde import make_sde
from diffusion.training import (
    VAL_EXAM_SEED,
    EmaShadow,
    FixedValExam,
    UniformTimeSampler,
    ValPoint,
    dsm_loss,
    evaluate_with_weights,
    make_time_sampler,
)


def _sampler(sde, name: str = "uniform"):
    """El muestreador de tiempos que el loop inyectaría (ya construido)."""
    return make_time_sampler(name, sde.T, 1e-3)


def _net(sde) -> ScoreMLP:
    """Red de score chica y determinística (la misma clase que entrena la Fase 1)."""
    torch.manual_seed(0)
    return ScoreMLP(data_dim=sde.data_dim, hidden_dim=32, num_blocks=2)


def _batches(sizes, *, dim: int = 2, seed: int = 3) -> list[torch.Tensor]:
    """Fuente re-iterable: una lista de batches con datos fijos (no sorteados por evaluación)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, dim, generator=g) for b in sizes]


def _ceros(sizes, *, dim: int = 2) -> list[torch.Tensor]:
    """Fuente de ``x_0 = 0``: la que hace exacta la pérdida de :class:`_LossConstante`."""
    return [torch.zeros(b, dim) for b in sizes]


def _foto(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copia **desacoplada** del ``state_dict`` de la red (clones, no los tensores vivos)."""
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _pesos_desplazados(net: torch.nn.Module, delta: float = 0.5) -> dict[str, torch.Tensor]:
    """Un juego de pesos **ajeno**: los de ``net`` corridos por ``delta`` (mismas claves)."""
    return {k: v + delta for k, v in _foto(net).items()}


def _mismos_tensores(uno: dict[str, torch.Tensor], otro: dict[str, torch.Tensor]) -> bool:
    """``True`` si los dos ``state_dict`` tienen las mismas claves y son iguales tensor a tensor."""
    return set(uno) == set(otro) and all(torch.equal(uno[k], otro[k]) for k in uno)


class _LossConstante(torch.nn.Module):
    """Red artificial con pérdida DSM **constante**, independiente de ``t`` y del ruido.

    Con la VE-SDE (``mean = x_0``) y ``x_0 = 0`` se tiene ``x_t = σ_t·ε`` y el target del score
    es ``-ε/σ_t``. Devolviendo ``-x/σ_t² + c/σ_t`` la diferencia contra el target queda
    ``c/σ_t``, y la pérdida pesada por ``λ(t) = σ_t²`` vale exactamente ``c²`` para toda muestra.

    ``constantes`` da un ``c`` por **llamada** (es decir, por batch): permite construir batches
    con pérdidas distintas y conocidas, y así distinguir el promedio ponderado por imágenes del
    promedio de las pérdidas de cada batch.
    """

    def __init__(self, sde, constantes: list[float]) -> None:
        super().__init__()
        self._sde = sde
        self._constantes = constantes
        self.llamadas = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c = self._constantes[min(self.llamadas, len(self._constantes) - 1)]
        self.llamadas += 1
        _, std = self._sde.marginal_prob(x, t)
        return -x / std**2 + c / std


class _SpyTimeSampler(UniformTimeSampler):
    """Muestreador uniforme que registra con qué ``n`` y con qué generator lo llamaron."""

    def __init__(self, T: float, t_eps: float) -> None:
        super().__init__(T, t_eps)
        self.llamadas: list[tuple[int, torch.Generator | None]] = []

    def sample(self, n, *, generator=None, device=None):
        self.llamadas.append((n, generator))
        return super().sample(n, generator=generator, device=device)


class _RedQueRevienta(torch.nn.Module):
    """Red que falla en el forward: sirve para verificar la restauración del modo en el ``finally``."""

    def __init__(self) -> None:
        super().__init__()
        self.escala = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        raise RuntimeError("forward roto a propósito")


class _EspiaDeDtype(torch.nn.Module):
    """Envoltorio que registra el ``dtype`` con el que sale el forward de la red envuelta.

    Sirve para observar la precisión efectiva de la evaluación: envuelve una red real (con capas
    ``Linear``, que son justo las que ``autocast`` degradaría a bfloat16) y deja el valor intacto,
    así el mismo test puede comparar el número y el ``dtype``.
    """

    def __init__(self, interna: torch.nn.Module) -> None:
        super().__init__()
        self.interna = interna
        self.dtypes: list[torch.dtype] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        salida = self.interna(x, t)
        self.dtypes.append(salida.dtype)
        return salida


# ------------------------------------------------------------------ examen fijo (3.3)


def test_dos_evaluaciones_de_una_red_inalterada_dan_el_mismo_float():
    sde = make_sde("ve")
    exam = FixedValExam(sde, _batches([4, 4, 2]), time_sampler=_sampler(sde))
    net = _net(sde)
    primero = exam.evaluate(net)
    segundo = exam.evaluate(net)
    assert isinstance(primero, float)
    assert math.isfinite(primero)
    assert primero == segundo


def test_dos_instancias_del_examen_dan_el_mismo_valor():
    # El examen no guarda estado: se reconstruye igual (base del criterio 6.4).
    sde = make_sde("ve")
    net = _net(sde)
    uno = FixedValExam(sde, _batches([3, 3]), time_sampler=_sampler(sde)).evaluate(net)
    otro = FixedValExam(sde, _batches([3, 3]), time_sampler=_sampler(sde)).evaluate(net)
    assert uno == otro


def test_cada_evaluacion_usa_un_generator_nuevo_y_local():
    sde = make_sde("ve")
    spy = _SpyTimeSampler(sde.T, 1e-3)
    exam = FixedValExam(sde, _batches([3, 2]), time_sampler=spy)
    net = _net(sde)
    exam.evaluate(net)
    exam.evaluate(net)
    assert [n for n, _ in spy.llamadas] == [3, 2, 3, 2]  # un sample por batch, con su tamaño
    gens = [g for _, g in spy.llamadas]
    assert all(g is not None for g in gens)
    assert gens[0] is gens[1]  # un único generator por evaluación
    assert gens[2] is gens[3]
    assert gens[0] is not gens[2]  # y uno NUEVO en la siguiente


def test_el_valor_cambia_cuando_la_red_cambia():
    sde = make_sde("ve")
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde))
    net = _net(sde)
    antes = exam.evaluate(net)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.5)
    assert exam.evaluate(net) != antes


def test_la_seed_del_examen_es_configurable_y_cambia_el_examen():
    sde = make_sde("ve")
    net = _net(sde)
    fuente = _batches([4])
    por_defecto = FixedValExam(sde, fuente, time_sampler=_sampler(sde)).evaluate(net)
    otra = FixedValExam(
        sde, fuente, time_sampler=_sampler(sde), seed=VAL_EXAM_SEED + 1
    ).evaluate(net)
    assert por_defecto != otra


# --------------------------------------------------- promedio ponderado por imágenes (3.2)


def test_promedia_ponderando_por_la_cantidad_de_imagenes():
    # Batches de 3 y 1 imagen con pérdidas exactas 1 y 9: ponderado por imágenes da 3.0;
    # promediar las pérdidas de los batches daría 5.0.
    sde = make_sde("ve")
    net = _LossConstante(sde, [1.0, 3.0])
    exam = FixedValExam(sde, _ceros([3, 1]), time_sampler=_sampler(sde))
    valor = exam.evaluate(net)
    assert valor == pytest.approx(3.0, rel=1e-4)
    assert valor != pytest.approx(5.0, rel=1e-2)


def test_el_batcheo_no_cambia_el_valor():
    sde = make_sde("ve")
    entero = FixedValExam(sde, _ceros([10]), time_sampler=_sampler(sde)).evaluate(
        _LossConstante(sde, [2.0])
    )
    partido = FixedValExam(sde, _ceros([4, 4, 2]), time_sampler=_sampler(sde)).evaluate(
        _LossConstante(sde, [2.0])
    )
    assert entero == pytest.approx(4.0, rel=1e-4)
    assert partido == pytest.approx(entero, rel=1e-4)


def test_recorre_todas_las_imagenes_incluida_la_cola_parcial():
    sde = make_sde("ve")
    net = _LossConstante(sde, [1.0])
    exam = FixedValExam(sde, _ceros([4, 4, 2]), time_sampler=_sampler(sde))
    exam.evaluate(net)
    assert net.llamadas == 3  # los tres batches, sin descartar el parcial


# ------------------------------------------- mismo criterio que el entrenamiento (3.4)


def test_coincide_con_dsm_loss_bajo_el_mismo_muestreo():
    sde = make_sde("ve")
    net = _net(sde)
    x0 = _batches([5])[0]
    sampler = _sampler(sde)
    g = torch.Generator().manual_seed(VAL_EXAM_SEED)
    t, pesos = sampler.sample(5, generator=g, device=torch.device("cpu"))
    with torch.no_grad():
        net.eval()
        esperado = dsm_loss(net, sde, x0, t, generator=g, sample_weights=pesos).item()
        net.train()
    valor = FixedValExam(sde, [x0], time_sampler=_sampler(sde)).evaluate(net)
    assert valor == pytest.approx(esperado, rel=1e-6)


def test_propaga_los_pesos_de_importance_sampling_a_dsm_loss():
    # Mismo criterio que el test anterior, pero con un muestreador cuyos pesos NO son ``None``:
    # con ``log_uniform`` la pérdida comparable es la **ponderada** por el likelihood ratio, así
    # que el examen tiene que pasarle esos pesos a ``dsm_loss``. Sin la ponderación el número
    # queda a otra escala y deja de ser comparable con la curva de entrenamiento (criterio 3.4).
    sde = make_sde("ve")
    net = _net(sde)
    x0 = _batches([5])[0]
    sampler = _sampler(sde, "log_uniform")

    g = torch.Generator().manual_seed(VAL_EXAM_SEED)
    t, pesos = sampler.sample(5, generator=g, device=torch.device("cpu"))
    assert pesos is not None  # si fueran ``None`` el test no distinguiría nada
    with torch.no_grad():
        net.eval()
        con_pesos = dsm_loss(net, sde, x0, t, generator=g, sample_weights=pesos).item()
        net.train()

    # El mismo examen sin la ponderación: se rearma la secuencia del generator desde cero para
    # que el ruido sea idéntico y la única diferencia sea el argumento ``sample_weights``.
    g_sin = torch.Generator().manual_seed(VAL_EXAM_SEED)
    t_sin, _ = sampler.sample(5, generator=g_sin, device=torch.device("cpu"))
    with torch.no_grad():
        net.eval()
        sin_pesos = dsm_loss(net, sde, x0, t_sin, generator=g_sin).item()
        net.train()
    assert sin_pesos != pytest.approx(con_pesos, rel=1e-3)

    valor = FixedValExam(sde, [x0], time_sampler=_sampler(sde, "log_uniform")).evaluate(net)
    assert valor == pytest.approx(con_pesos, rel=1e-6)


def test_usa_el_muestreador_de_tiempos_inyectado():
    # Cambiar el esquema de muestreo cambia el examen: la clase no construye uno propio.
    sde = make_sde("ve")
    net = _net(sde)
    fuente = _batches([6])
    uniforme = FixedValExam(sde, fuente, time_sampler=_sampler(sde)).evaluate(net)
    log_uniforme = FixedValExam(
        sde, fuente, time_sampler=_sampler(sde, "log_uniform")
    ).evaluate(net)
    assert uniforme != log_uniforme


@pytest.mark.parametrize("nombre", ["vp", "ve", "sub_vp"])
def test_anda_con_las_tres_sdes(nombre):
    sde = make_sde(nombre)
    exam = FixedValExam(sde, _batches([4, 3]), time_sampler=_sampler(sde))
    assert math.isfinite(exam.evaluate(_net(sde)))


def test_anda_con_forma_de_evento_de_imagen():
    sde = make_sde("ve", data_dim=(1, 4, 4))
    batches = [torch.zeros(3, 1, 4, 4), torch.zeros(1, 1, 4, 4)]
    net = _LossConstante(sde, [1.0, 3.0])
    valor = FixedValExam(sde, batches, time_sampler=_sampler(sde)).evaluate(net)
    assert valor == pytest.approx(3.0, rel=1e-4)


# ------------------------------------------------- precisión fija en fp32 (3.4 / diseño)


def test_fuerza_fp32_dentro_de_una_region_de_autocast():
    # La evaluación fija fp32 con ``autocast(enabled=False)`` aunque se la llame desde una región
    # de precisión mixta: el número debe ser comparable entre celdas entrenadas con y sin AMP.
    sde = make_sde("ve")
    fuente = _batches([4, 2])
    limpio = FixedValExam(sde, fuente, time_sampler=_sampler(sde)).evaluate(_net(sde))

    espia = _EspiaDeDtype(_net(sde))  # misma seed que ``limpio``: idénticos pesos
    exam = FixedValExam(sde, fuente, time_sampler=_sampler(sde))
    control = torch.nn.Linear(2, 2)  # testigo de que la región está realmente activa
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        bajo_autocast = exam.evaluate(espia)
        dtype_del_control = control(fuente[0]).dtype

    # Sin el testigo el test podría pasar por vacuidad (si autocast no estuviera activo acá).
    assert dtype_del_control is torch.bfloat16
    assert espia.dtypes == [torch.float32, torch.float32]  # un forward por batch, en fp32
    assert bajo_autocast == limpio  # bit a bit: la región no cambia el valor


# -------------------------------------------------------------- sin efectos (3.5, 6.2)


def test_no_modifica_los_pesos_ni_el_modo_de_la_red():
    sde = make_sde("ve")
    net = _net(sde)
    net.train()
    antes = {k: v.detach().clone() for k, v in net.state_dict().items()}
    FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde)).evaluate(net)
    assert net.training is True
    despues = net.state_dict()
    assert set(despues) == set(antes)
    for k, v in antes.items():
        assert torch.equal(despues[k], v)


def test_respeta_una_red_que_ya_venia_en_modo_evaluacion():
    sde = make_sde("ve")
    net = _net(sde)
    net.eval()
    FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde)).evaluate(net)
    assert net.training is False


@pytest.mark.parametrize("modo_entrenamiento", [True, False])
def test_restaura_el_modo_aunque_la_evaluacion_reviente(modo_entrenamiento):
    sde = make_sde("ve")
    net = _RedQueRevienta()
    net.train(modo_entrenamiento)
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde))
    with pytest.raises(RuntimeError, match="forward roto"):
        exam.evaluate(net)
    assert net.training is modo_entrenamiento


def test_la_red_corre_en_modo_evaluacion_y_sin_gradientes():
    sde = make_sde("ve")
    observado: list[tuple[bool, bool]] = []

    class _Espia(torch.nn.Module):
        def forward(self, x, t):  # noqa: ARG002
            observado.append((self.training, torch.is_grad_enabled()))
            return torch.zeros_like(x)

    net = _Espia()
    net.train()
    FixedValExam(sde, _batches([4, 2]), time_sampler=_sampler(sde)).evaluate(net)
    assert observado == [(False, False), (False, False)]


def test_no_deja_gradientes_acumulados():
    sde = make_sde("ve")
    net = _net(sde)
    FixedValExam(sde, _batches([4, 2]), time_sampler=_sampler(sde)).evaluate(net)
    assert all(p.grad is None for p in net.parameters())


def test_no_mueve_el_rng_global_de_torch():
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([4, 2]), time_sampler=_sampler(sde))
    estado = torch.get_rng_state()
    exam.evaluate(net)
    assert torch.equal(torch.get_rng_state(), estado)


def test_no_mueve_el_generator_del_loop():
    sde = make_sde("ve")
    net = _net(sde)
    del_loop = torch.Generator().manual_seed(7)
    exam = FixedValExam(sde, _batches([4, 2]), time_sampler=_sampler(sde))
    estado = del_loop.get_state()
    exam.evaluate(net)
    assert torch.equal(del_loop.get_state(), estado)


# ---------------------------------------------------------------------- device (3.6)


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_acepta_el_device_como_texto_o_como_objeto(device):
    sde = make_sde("ve")
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde), device=device)
    assert math.isfinite(exam.evaluate(_net(sde)))


def test_mueve_los_batches_al_device_del_examen():
    sde = make_sde("ve")
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde), device="cpu")

    vistos: list[torch.device] = []

    class _Espia(torch.nn.Module):
        def forward(self, x, t):  # noqa: ARG002
            vistos.append(x.device)
            return torch.zeros_like(x)

    exam.evaluate(_Espia())
    assert vistos == [torch.device("cpu")]


# ---------------------------------------------------------------------- errores


def test_fuente_sin_imagenes_levanta_valueerror():
    sde = make_sde("ve")
    exam = FixedValExam(sde, [], time_sampler=_sampler(sde))
    with pytest.raises(ValueError, match="ninguna imagen"):
        exam.evaluate(_net(sde))


def test_fuente_de_batches_vacios_levanta_valueerror():
    sde = make_sde("ve")
    exam = FixedValExam(sde, [torch.zeros(0, 2)], time_sampler=_sampler(sde))
    with pytest.raises(ValueError, match="ninguna imagen"):
        exam.evaluate(_net(sde))


# ------------------------------------------ evaluación con pesos ajenos (4.1, 4.4)


def test_devuelve_la_perdida_bajo_los_pesos_ajenos():
    # El valor tiene que ser el del examen medido **con los pesos recibidos**: se compara contra
    # una red aparte cargada con esos mismos pesos (idéntico, no aproximado: el examen es
    # determinístico) y contra el valor de los pesos vivos (distinto, si no la función no swapeó).
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([4, 2]), time_sampler=_sampler(sde))
    con_vivos = exam.evaluate(net)
    otros = _pesos_desplazados(net)

    referencia = _net(sde)
    referencia.load_state_dict(otros)
    esperado = exam.evaluate(referencia)

    valor = evaluate_with_weights(exam, net, otros)
    assert valor == esperado
    assert valor != con_vivos


def test_deja_el_state_dict_identico_tensor_por_tensor():
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde))
    antes = _foto(net)
    evaluate_with_weights(exam, net, _pesos_desplazados(net))
    assert _mismos_tensores(_foto(net), antes)


def test_una_segunda_llamada_vuelve_a_medir_los_pesos_vivos():
    # Corolario de la restauración: después del swap, evaluar la red da lo mismo que antes. Es el
    # test que un "respaldo" sin clonar rompe (la copia quedaría sobreescrita al cargar los otros
    # pesos y la red se quedaría con los ajenos para siempre).
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([4, 2]), time_sampler=_sampler(sde))
    antes = exam.evaluate(net)
    evaluate_with_weights(exam, net, _pesos_desplazados(net))
    assert exam.evaluate(net) == antes


def test_dos_swaps_con_los_mismos_pesos_dan_el_mismo_valor():
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([3, 2]), time_sampler=_sampler(sde))
    otros = _pesos_desplazados(net)
    assert evaluate_with_weights(exam, net, otros) == evaluate_with_weights(exam, net, otros)


@pytest.mark.parametrize("modo_entrenamiento", [True, False])
def test_restaura_los_pesos_aunque_la_evaluacion_reviente(modo_entrenamiento):
    # La restauración va en un ``finally``: una excepción del forward no puede dejar la red con
    # los pesos ajenos, o el loop seguiría entrenando desde la sombra EMA.
    sde = make_sde("ve")
    net = _RedQueRevienta()
    net.train(modo_entrenamiento)
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde))
    antes = _foto(net)
    with pytest.raises(RuntimeError, match="forward roto"):
        evaluate_with_weights(exam, net, _pesos_desplazados(net))
    assert _mismos_tensores(_foto(net), antes)
    assert net.training is modo_entrenamiento


def test_recibe_los_pesos_como_mapping_generico():
    # La firma pide un ``Mapping``, no un ``dict``: la función no conoce la clase del EMA ni
    # depende de que los pesos vengan en un contenedor mutable.
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde))
    otros = MappingProxyType(_pesos_desplazados(net))
    assert not isinstance(otros, dict)
    assert math.isfinite(evaluate_with_weights(exam, net, otros))


def test_no_muta_los_pesos_recibidos():
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde))
    otros = _pesos_desplazados(net)
    copia = {k: v.detach().clone() for k, v in otros.items()}
    evaluate_with_weights(exam, net, otros)
    assert _mismos_tensores(otros, copia)


def test_no_mueve_el_rng_global_ni_deja_gradientes():
    # No consume azar propio: delega enteramente en ``evaluate`` (que se re-siembra local).
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([4, 2]), time_sampler=_sampler(sde))
    otros = _pesos_desplazados(net)
    estado = torch.get_rng_state()
    evaluate_with_weights(exam, net, otros)
    assert torch.equal(torch.get_rng_state(), estado)
    assert all(p.grad is None for p in net.parameters())


def test_anda_con_una_red_envuelta_en_la_parametrizacion_epsilon():
    # La red de producción de la celda de gatos va envuelta en ``EpsilonScoreWrapper``, cuyo
    # ``state_dict``/``load_state_dict`` delegan al interno: las claves son las de la red pelada
    # (que son justo las que publica la sombra EMA), así que el swap funciona igual.
    sde = make_sde("ve")
    interna = _net(sde)
    net = EpsilonScoreWrapper(interna, lambda x, t: sde.marginal_prob(x, t)[1])
    exam = FixedValExam(sde, _batches([4, 2]), time_sampler=_sampler(sde))
    otros = _pesos_desplazados(net)
    assert set(otros) == set(interna.state_dict())  # claves de red pelada, sin prefijo ``_net.``

    con_vivos = exam.evaluate(net)
    antes = _foto(net)
    valor = evaluate_with_weights(exam, net, otros)
    assert math.isfinite(valor)
    assert valor != con_vivos
    assert _mismos_tensores(_foto(net), antes)
    assert _mismos_tensores(_foto(interna), antes)  # el interno también quedó como estaba


def test_anda_con_la_foto_de_la_sombra_ema():
    # El caller de producción (4.1): los pesos ajenos son ``EmaShadow.state_dict()``, cuya
    # precondición de claves está garantizada por el propio EMA.
    sde = make_sde("ve")
    net = _net(sde)
    sombra = EmaShadow(net, decay=0.9)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.5)
    sombra.update(1)  # la sombra queda entre los pesos iniciales y los actuales

    exam = FixedValExam(sde, _batches([4, 2]), time_sampler=_sampler(sde))
    con_vivos = exam.evaluate(net)
    antes = _foto(net)
    con_ema = evaluate_with_weights(exam, net, sombra.state_dict())
    assert con_ema != con_vivos
    assert _mismos_tensores(_foto(net), antes)


# ---------------------------------------------------------------------- errores del swap


def test_claves_faltantes_levantan_runtimeerror_y_restauran_los_pesos():
    # ``load_state_dict`` copia las claves que sí matchean **antes** de reportar las que faltan,
    # así que la carga tiene que estar dentro del bloque protegido: si no, el fail-fast dejaría
    # la red con una mezcla de pesos vivos y ajenos.
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde))
    incompletos = _pesos_desplazados(net)
    faltante = sorted(incompletos)[0]
    del incompletos[faltante]
    antes = _foto(net)
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluate_with_weights(exam, net, incompletos)
    assert _mismos_tensores(_foto(net), antes)


def test_claves_desconocidas_levantan_runtimeerror():
    sde = make_sde("ve")
    net = _net(sde)
    exam = FixedValExam(sde, _batches([4]), time_sampler=_sampler(sde))
    intrusos = _pesos_desplazados(net)
    intrusos["no_existe"] = torch.zeros(())
    antes = _foto(net)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        evaluate_with_weights(exam, net, intrusos)
    assert _mismos_tensores(_foto(net), antes)


# ------------------------------------------------------------- forma del punto (5.2)


def test_valpoint_es_un_dict_con_las_cuatro_claves():
    punto: ValPoint = {"step": 4, "raw": 1.5, "ema": None, "train": 1.2}
    assert isinstance(punto, dict)
    assert set(punto) == {"step", "raw", "ema", "train"}


def test_la_semilla_del_examen_es_una_constante_congelada():
    assert VAL_EXAM_SEED == 12345
