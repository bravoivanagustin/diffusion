"""Capa de configuración: de un archivo YAML a una corrida lista para :func:`train`.

Cada **celda del estudio de ablación** (una combinación SDE × dataset × hiperparámetros) se
describe en un ``.yaml`` versionable. Esta capa es un front-end fino y aislado del núcleo: el
loop y la pérdida no saben nada de archivos. :func:`load_config` lee el YAML a un ``dict`` y
:func:`build_run` lo ensambla en un :class:`RunSpec` —reusando los factories
``make_sde``/``make_distribution``, que ya filtran kwargs por firma—.

Estructura esperada del YAML::

    sde:                  # -> make_sde(name, **resto)
      name: vp
      beta_min: 0.1
    data:                 # -> make_distribution(shape, dim, **resto); n_samples/batch_size
      shape: mixture      #    describen la fuente de datos (no van al TrainConfig)
      dim: 2
      n_samples: 4000
      batch_size: 256
      n_components: 8
      val_root: data/cats_val  # opcional, SOLO kind: images: carpeta held-out sobre la que se
                          #   mide la pérdida de validación. Sin la clave no se mide nada
                          #   (opt-in); con una fuente de puntos se rechaza. Declararla deriva
                          #   ADEMÁS el examen fijo de entrenamiento (desde `root`, con tantas
                          #   imágenes como tenga `val_root`): no hay clave que lo controle.
    train:                # -> campos de TrainConfig (solo el loop de optimización)
      num_steps: 300
      lr: 0.002
      checkpoint_every: 50  # opcional: 0 = solo final; N>0 = snapshots …_stepNNNNN.pt
    model:                # opcional: receta de la red -> make_model(name, **resto)
      name: mlp           #   si falta, se usa {name: mlp} dimensionado desde el dato/SDE
      hidden_dim: 256
      score_parametrization: epsilon  # opcional: envuelve la red con EpsilonScoreWrapper
                          #   (score = -red/σ_t de la SDE de la corrida); sin la clave, red
                          #   pelada como siempre. Único valor válido: "epsilon".
    out:                  # rutas de salida (relativas al cwd)
      checkpoint: models/vp_mixture.pt
      loss_curve: models/vp_mixture_loss.png
      train_log: models/vp_mixture_log.jsonl  # opcional: estados del entrenamiento con timestamp
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields

from ..data_generation import (
    count_images,
    finite_batches,
    infinite_bare,
    infinite_batches,
    make_distribution,
)
from ..models import EpsilonScoreWrapper, ScoreModel, make_model
from ..sde import ForwardSDE, make_sde
from .trainer import TrainConfig

# Defaults de la fuente de datos cuando el bloque ``data:`` no los especifica (valores de la
# corrida por épocas previa, para no cambiar el comportamiento de los configs existentes).
_DEFAULT_N_SAMPLES = 4000
_DEFAULT_BATCH_SIZE = 256

# Valores válidos del discriminador ``data.kind`` (fuente de datos). ``points`` es el default
# retrocompatible; ``images`` arma la fuente de imágenes desde ``data:`` (feature config-image-training).
_VALID_KINDS = ("points", "images")


@dataclass
class RunSpec:
    """Una corrida lista: SDE + red + fuente de datos + hiperparámetros + rutas de salida.

    Lleva la red (``model``) y el iterador infinito de datos (``data``) ya construidos, listos
    para pasarle a :func:`~diffusion.training.train` (en vez de una ``distribution`` finita).
    Además transporta ``model_spec`` —la receta ``{name, kwargs}`` con la que se construyó la
    red— para que ``scripts/train.py`` la pase a :func:`~diffusion.training.save_checkpoint` y
    el checkpoint quede reconstruible sin el config original.

    ``val_data`` y ``train_exam_data`` son las dos fuentes de examen de la feature
    ``validation-loss``: la de validación y la del examen fijo de **entrenamiento** (su
    referencia). Son no nulas **exactamente** cuando el config declara ``data.val_root`` —las dos
    juntas, nunca una sin la otra— y el CLI las reenvía al loop.
    """

    sde: ForwardSDE
    model: ScoreModel
    data: Iterator
    config: TrainConfig
    model_spec: dict | None = None  # receta {name, kwargs} para el checkpoint model-agnóstico
    checkpoint: pathlib.Path | None = None
    loss_curve: pathlib.Path | None = None
    train_log: pathlib.Path | None = None  # .jsonl de estados del entrenamiento (opcional, con timestamps)
    val_data: Iterable | None = None  # examen de validación (None = corrida sin validación)
    train_exam_data: Iterable | None = None  # examen fijo de train (None junto con val_data)


@dataclass
class DataSources:
    """Las fuentes que salen del bloque ``data:`` y la forma de evento derivada.

    Valor de retorno de :func:`build_data_source`. Es un dataclass y no una tupla porque con
    más de una fuente las posiciones se leen mal (y sumar una fuente futura no vuelve a romper
    la firma). ``train`` y ``event_shape`` son el contrato de siempre; ``val`` y ``train_exam``
    son las dos fuentes de examen de la feature ``validation-loss``, no nulas **exactamente**
    cuando el config declara ``data.val_root`` (las dos juntas, nunca una sin la otra).

    Attributes:
        train: Fuente **infinita** de tensores crudos que consume el loop de entrenamiento.
        event_shape: Forma de evento ``(C, H, W)`` para imágenes; ``None`` para puntos (dato
            plano: la dimensión la lleva la SDE).
        val: Fuente **finita y re-iterable** del examen de validación, o ``None`` si no se
            declaró ``data.val_root``.
        train_exam: Fuente finita del examen fijo de **entrenamiento** (la referencia que hace
            legible la curva de validación), o ``None`` si no se declaró ``data.val_root``. Sale
            de ``data.root`` —la misma carpeta que la fuente del loop— pero recortada a tantas
            imágenes como tenga el set de validación, para que los dos exámenes tengan el mismo
            tamaño y, por lo tanto, la misma varianza de estimador.
    """

    train: Iterator
    event_shape: tuple[int, ...] | None
    val: Iterable | None = None
    train_exam: Iterable | None = None


def load_config(path: str | pathlib.Path) -> dict:
    """Lee un archivo YAML de configuración a un ``dict``.

    Args:
        path: Ruta del ``.yaml``.

    Returns:
        El contenido parseado como ``dict``.

    Raises:
        ModuleNotFoundError: Si ``pyyaml`` no está instalado.
    """
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
        raise ModuleNotFoundError(
            "Leer configs YAML necesita PyYAML y no está instalado. "
            "Instalalo con `uv add pyyaml` (o `pip install pyyaml`)."
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_data_source(data_raw: dict) -> DataSources:
    """Dispatcha la fuente de datos por ``data_raw['kind']`` ('points' default | 'images').

    Es el **único dueño** del parseo del bloque ``data:``: :func:`build_run` obtiene sus fuentes
    a través de este resolver, sin duplicar el parseo. Para puntos arma la distribución de
    juguete y la envuelve en un iterador infinito de tensores crudos; la forma de evento es
    ``None`` (dato plano, la dimensión la lleva la SDE). Para imágenes arma la fuente infinita
    con :func:`~diffusion.data_generation.infinite_batches`, deriva la forma de evento
    ``(3, image_size, image_size)`` (canales fijos en 3) y la valida **peekeando** el primer
    batch del iterador real, que devuelve tal cual (ya posicionado tras ese batch).

    Con ``val_root`` (feature ``validation-loss``) arma **además** la fuente del examen de
    validación con :func:`~diffusion.data_generation.finite_batches`: finita, re-iterable, en
    orden y orientación canónicos, y con el **mismo** ``batch_size``, ``image_size`` y ``crop``
    que la de entrenamiento, para que ambas pérdidas se computen sobre la misma forma de evento.
    Su forma también se valida con un peek, inocuo acá porque la fuente es re-iterable (a
    diferencia de la infinita, donde el peek obliga a devolver el iterador ya avanzado).

    Con ``val_root`` deriva **además** el examen fijo de **entrenamiento**, desde ``root`` y con
    los mismos parámetros, acotado con ``max_images`` a la cantidad de imágenes de ``val_root``
    (contada con :func:`~diffusion.data_generation.count_images`, sin introspeccionar el loader).
    Que los dos exámenes tengan **exactamente** la misma cantidad de imágenes es lo que les da la
    misma varianza de estimador y hace interpretable el gap train↔val. **No hay ninguna clave
    para pedirlo ni para apagarlo**: el examen de train no es una preferencia del usuario sino la
    referencia que hace legible la curva de validación, así que se construye siempre que haya
    ``val_root``. Si el set de entrenamiento tuviera menos imágenes que el de validación, el tope
    simplemente no recorta (los exámenes dejan de ser del mismo tamaño y no es un error).

    Args:
        data_raw: El bloque ``data:`` del config (se copia; no se muta el ``dict`` del caller).
            ``kind`` ausente ⇒ ``points``. Para puntos: ``shape``/``name`` (obligatorio),
            ``dim`` (default 2), ``n_samples``/``batch_size`` (defaults de la corrida previa),
            ``shuffle`` (default True), + params de la forma (p. ej. ``n_components``). Para
            imágenes: ``root`` (obligatorio), ``image_size`` (default 64),
            ``batch_size`` (default de la corrida previa), ``augment``/``crop``/``shuffle``
            (default True), ``seed`` (default None), ``num_workers`` (default 0: carga en el
            proceso principal), ``pin_memory`` (default False: sin memoria fijada),
            ``val_root`` (default None: sin validación). Otras claves se rechazan
            (``infinite_batches`` no filtra por firma).

    Returns:
        Un :class:`DataSources` con la fuente infinita de entrenamiento, la forma de evento
        (``(C, H, W)`` para imágenes o ``None`` para puntos) y las dos fuentes de examen —la de
        validación y la del examen fijo de entrenamiento, no nulas **juntas** y exactamente
        cuando se declaró ``val_root``—.

    Raises:
        ValueError: Si ``kind`` no está reconocido (lista los válidos); si se declara
            ``val_root`` con una fuente que no es de imágenes; —camino de puntos— si falta
            ``data.shape``/``data.name``; —camino de imágenes— si falta ``data.root``, si quedan
            claves desconocidas, si la carpeta no existe/está vacía/tiene menos imágenes que
            ``batch_size`` (propagado desde ``infinite_batches``), si la forma emitida no
            coincide con la derivada, o si ``val_root`` no existe/está vacía (propagado desde
            ``finite_batches`` y ``count_images``, que **no** exigen un mínimo de ``batch_size``:
            un set más chico que el batch es un batch parcial válido) o emite otra forma de
            evento.
    """
    data_raw = dict(data_raw or {})
    kind = data_raw.pop("kind", "points")
    # La clave de validación se popea ACÁ, antes del dispatch, porque las dos ramas la necesitan
    # fuera del dict por motivos opuestos: la de imágenes rechaza claves desconocidas (dejarla
    # sin popear haría fallar toda config que la declare) y la de puntos pasa el resto a
    # make_distribution, que filtra por firma (la clave se ignoraría en silencio). Ausente ⇒
    # comportamiento idéntico al de antes de la feature.
    val_root = data_raw.pop("val_root", None)

    if kind == "points":
        if val_root is not None:
            raise ValueError(
                "config: 'data.val_root' solo aplica a kind: images; la pérdida de validación "
                "se mide sobre una fuente de imágenes y esta corrida es de puntos "
                f"(kind={kind!r}). Saca la clave o pasá la corrida a kind: images."
            )
        # Camino de puntos (movido verbatim desde build_run): n_samples/batch_size son params de
        # la fuente (no del TrainConfig); el resto de las claves van a make_distribution, que
        # filtra por firma.
        shape = data_raw.pop("shape", None) or data_raw.pop("name", None)
        if shape is None:
            raise ValueError(
                "config: falta 'data.shape' (p. ej. gaussian / mixture / two_moons)."
            )
        dim = data_raw.pop("dim", 2)
        n_samples = data_raw.pop("n_samples", None) or _DEFAULT_N_SAMPLES
        batch_size = data_raw.pop("batch_size", None) or _DEFAULT_BATCH_SIZE
        shuffle = data_raw.pop("shuffle", True)
        distribution = make_distribution(shape, dim, **data_raw)
        data = infinite_bare(distribution.dataloader(n_samples, batch_size, shuffle=shuffle))
        # Dato plano: sin forma de evento (la dimensión la lleva la SDE) y sin validación.
        return DataSources(train=data, event_shape=None)

    if kind == "images":
        # Camino de imágenes: se mapean SOLO los params conocidos de infinite_batches (que NO
        # filtra kwargs por firma), así que las claves sobrantes se rechazan explícitamente. La
        # forma de evento se deriva de image_size (canales fijos en 3) y se valida peekeando el
        # primer batch del iterador REAL, que se devuelve tal cual (ya posicionado tras el batch 1;
        # inocuo en una fuente infinita reshuffled y evita un segundo escaneo de la carpeta).
        root = data_raw.pop("root", None)
        if root is None:
            raise ValueError("config: falta 'data.root' para kind: images.")
        image_size = data_raw.pop("image_size", 64)
        batch_size = data_raw.pop("batch_size", None) or _DEFAULT_BATCH_SIZE
        augment = data_raw.pop("augment", True)
        crop = data_raw.pop("crop", True)
        shuffle = data_raw.pop("shuffle", True)
        seed = data_raw.pop("seed", None)
        # Knobs de carga eficiente en GPU (feature gpu-training-efficiency, R3.1/R3.2): se popean
        # ANTES del rechazo de unknowns (así se suman a las claves conocidas) y se pasan tal cual a
        # infinite_batches, que ya los acepta. Sus defaults (0 / False) reproducen el comportamiento
        # actual —carga en el proceso principal, sin memoria fijada— sin cambio observable.
        num_workers = data_raw.pop("num_workers", 0)
        pin_memory = data_raw.pop("pin_memory", False)
        if data_raw:
            raise ValueError(
                f"config: claves desconocidas en data: para kind: images: "
                f"{sorted(data_raw)}."
            )
        data = infinite_batches(
            root,
            batch_size,
            image_size=image_size,
            augment=augment,
            crop=crop,
            shuffle=shuffle,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        event_shape = (3, image_size, image_size)
        # Peek de validación sobre el iterador real (infinite_batches fail-fast-ea root
        # faltante/vacío/pocas imágenes al construirse, así que ese ValueError ya propagó arriba).
        first = next(data)
        obtained = tuple(first.shape[1:])
        if obtained != event_shape:
            raise ValueError(
                f"config: la fuente de imágenes emite forma {obtained}, se esperaba "
                f"{event_shape} (derivada de image_size={image_size}, canales=3)."
            )
        # Examen de validación (opt-in): misma forma de evento que el entrenamiento (mismos
        # batch_size / image_size / crop) pero recorrido de medición, no de entrenamiento — la
        # fuente finita fuerza orden y orientación canónicos (sin barajado ni espejado) y entrega
        # la cola como batch parcial, sin mínimo de batch_size. num_workers/pin_memory se fijan
        # en los valores in-process: una evaluación no amortiza el arranque de los workers.
        val = None
        train_exam = None
        if val_root is not None:
            val = finite_batches(
                val_root,
                batch_size,
                image_size=image_size,
                crop=crop,
                num_workers=0,
                pin_memory=False,
            )
            # Peek de validación de la forma, con el mismo criterio que la fuente de
            # entrenamiento. Acá es inocuo: la fuente es re-iterable, así que consumir el primer
            # batch no le saca ningún dato al examen (arranca de cero en cada recorrido).
            obtained_val = tuple(next(iter(val)).shape[1:])
            if obtained_val != event_shape:
                raise ValueError(
                    f"config: la fuente de validación ('data.val_root': {val_root}) emite forma "
                    f"{obtained_val}, se esperaba {event_shape} (la misma que la de "
                    "entrenamiento, para que ambas pérdidas sean comparables)."
                )
            # Examen fijo de ENTRENAMIENTO: la referencia que hace legible la curva de validación
            # (sin ella, un valor de validación solo no dice si el modelo generaliza o si el
            # examen es más difícil). Sale de la misma carpeta que la fuente del loop (``root``)
            # pero con el recorrido de medición —orden y orientación canónicos, sin espejado—, y
            # acotado a la MISMA cantidad de imágenes que el set de validación: igual tamaño ⇒
            # igual varianza de estimador ⇒ el gap train↔val es atribuible al modelo. El conteo
            # sale de ``count_images`` justamente para no introspeccionar el ``.dataset`` del
            # loader de validación (rompería el contrato de ``finite_batches``, declarado como un
            # simple iterable de tensores). Se construye SIEMPRE que haya validación: no hay
            # clave que lo pida ni que lo apague (no es una preferencia del usuario). Si ``root``
            # tuviera menos imágenes que ``val_root``, el tope no recorta y listo.
            train_exam = finite_batches(
                root,
                batch_size,
                image_size=image_size,
                crop=crop,
                max_images=count_images(val_root),
                num_workers=0,
                pin_memory=False,
            )
        return DataSources(
            train=data, event_shape=event_shape, val=val, train_exam=train_exam
        )

    raise ValueError(
        f"config: data.kind desconocido: {kind!r}. Válidos: {', '.join(_VALID_KINDS)}."
    )


def build_run(raw: dict) -> RunSpec:
    """Ensambla un :class:`RunSpec` desde un ``dict`` de configuración.

    Args:
        raw: Config parseado (de :func:`load_config` o construido a mano).

    Returns:
        Un :class:`RunSpec` con la SDE, la red, la fuente de datos infinita, el
        :class:`TrainConfig`, las rutas y —si el config declara ``data.val_root``— las dos
        fuentes de examen: la de validación y la del examen fijo de entrenamiento.

    Raises:
        ValueError: Si faltan claves obligatorias (``sde.name``, ``data.shape``), si una corrida
            de imágenes declara ``data_dim`` en ``sde:`` (la forma se deriva de ``data:``; única
            fuente de verdad), si el bloque ``train:`` trae claves desconocidas para
            :class:`TrainConfig`, si el ``name`` del bloque ``model:`` no está registrado en
            ``make_model``, o si ``model.score_parametrization`` trae un valor distinto de
            ``"epsilon"``. Propaga además los errores del bloque ``data:`` (ver
            :func:`build_data_source`, incluida la validación de ``val_root``).
    """
    raw = dict(raw or {})

    # --- datos: el resolver es el único dueño del parseo de 'data:' (dispatch por kind). Para
    # puntos devuelve event_shape=None (comportamiento idéntico al de antes); para imágenes la
    # forma de evento (C, H, W) alimenta la SDE de abajo. Se arma ANTES que la SDE para que la
    # forma derivada pueda configurarla (única fuente de verdad en 'data:'). ---
    sources = build_data_source(raw.get("data") or {})
    data, event_shape = sources.train, sources.event_shape

    # --- SDE: para imágenes se construye con la forma de evento como data_dim (derivada de
    # 'data:', no declarada en 'sde:'); para puntos, exactamente como antes. ---
    sde_raw = dict(raw.get("sde") or {})
    if "name" not in sde_raw:
        raise ValueError("config: falta 'sde.name' (p. ej. vp / ve / sub_vp).")
    if event_shape is not None:
        # Única fuente de verdad: la forma vive en 'data:' y se deriva; declararla también en
        # 'sde:' abriría la puerta a que se desincronicen, así que se rechaza (R3.3).
        if "data_dim" in sde_raw:
            raise ValueError(
                "config: no declares 'data_dim' en sde: para imágenes; se deriva de data: "
                f"(forma de evento {event_shape})."
            )
        sde = make_sde(**sde_raw, data_dim=event_shape)
    else:
        sde = make_sde(**sde_raw)

    # --- hiperparámetros del loop -> TrainConfig (validación estricta contra sus campos) ---
    train_raw = dict(raw.get("train") or {})
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(train_raw) - valid
    if unknown:
        raise ValueError(
            f"config: claves desconocidas en train: {sorted(unknown)}. "
            f"Válidas: {sorted(valid)}."
        )
    config = TrainConfig(**train_raw)

    # --- red: bloque 'model:' opcional. El default depende de la fuente: 'unet' para imágenes
    # (forma de evento presente), 'mlp' para puntos (dimensionado desde el dato/SDE). Las claves
    # del bloque van a make_model, que filtra por firma (no se validan acá). ---
    model_raw = dict(raw.get("model") or {})
    default_model = "unet" if event_shape is not None else "mlp"
    model_name = model_raw.pop("name", default_model)
    # La clave de parametrización se saca ANTES de make_model: no es un hiperparámetro de la
    # red (no debe llegar al constructor ni a los kwargs de la receta) sino una decisión de
    # cómo se consume su salida. Sin la clave -> pipeline idéntico al actual (red pelada).
    parametrization = model_raw.pop("score_parametrization", None)
    if parametrization is not None and parametrization != "epsilon":
        raise ValueError(
            f"config: score_parametrization desconocida: {parametrization!r}. "
            "Válidas: 'epsilon' (u omitir la clave para la red pelada)."
        )
    # Gate: inyectamos ``data_dim`` en la receta del modelo solo cuando es un entero (path MLP
    # 2D: dimensiona el default desde la SDE). Para una forma de evento multidimensional (tupla,
    # imágenes) NO se inyecta: la red de imágenes es la U-Net, que trae su propia config y no
    # toma ``data_dim`` —además ``ScoreMLP`` haría ``int(tupla)`` y reventaría—.
    if isinstance(sde.data_dim, int):
        model_raw.setdefault("data_dim", sde.data_dim)
    model = make_model(model_name, **model_raw)
    # Receta genérica {name, kwargs} para el checkpoint model-agnóstico: la misma con la que se
    # construyó la red, así generate.py la reconstruye con make_model sin el config original.
    model_spec = {"name": model_name, "kwargs": dict(model_raw)}
    if parametrization == "epsilon":
        # Wrap con la σ de la SDE de la corrida: el RunSpec entrena/consume score = -red/σ_t
        # (misma fórmula que la reconstrucción al generar). La clave viaja como hermana de
        # name/kwargs en la receta —los kwargs quedan pelados: el camino de reconstrucción
        # sigue siendo make_model(name, **kwargs) y recién después el wrap—.
        model = EpsilonScoreWrapper(model, lambda x, t: sde.marginal_prob(x, t)[1])
        model_spec["score_parametrization"] = "epsilon"

    # --- salidas ---
    out_raw = dict(raw.get("out") or {})
    checkpoint = out_raw.get("checkpoint")
    loss_curve = out_raw.get("loss_curve")
    train_log = out_raw.get("train_log")
    return RunSpec(
        sde=sde,
        model=model,
        data=data,
        config=config,
        model_spec=model_spec,
        checkpoint=pathlib.Path(checkpoint) if checkpoint else None,
        loss_curve=pathlib.Path(loss_curve) if loss_curve else None,
        train_log=pathlib.Path(train_log) if train_log else None,
        # Se copian tal cual desde el resolver, sin más lógica: son None exactamente cuando el
        # config no declara 'data.val_root' (validación opt-in), y no nulas las DOS cuando sí lo
        # declara — nunca una sin la otra. El CLI las reenvía al loop.
        val_data=sources.val,
        train_exam_data=sources.train_exam,
    )
