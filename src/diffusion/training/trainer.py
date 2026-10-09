"""Loop de entrenamiento por denoising score matching y persistencia de la red.

Reúne el muestreo de pares de entrenamiento (:mod:`losses`), una red de score cualquiera
(cualquier :class:`diffusion.models.ScoreModel`: el ``ScoreMLP`` de Fase 1, la ``ScoreUNet``
de Fase 2, …) y un proceso forward (:class:`diffusion.sde.ForwardSDE`) en un :func:`train`
que devuelve la red entrenada y la historia de pérdida.

:func:`train` es **agnóstico a la red y al origen de datos**: recibe la ``model`` ya
construida y un iterador **infinito** de tensores crudos, y corre un loop **por pasos**
(``config.num_steps``). No construye la red ni ramifica por su tipo — esa responsabilidad vive
en el caller (``make_model`` / el config-driven). La **regla del Eje 1** sigue vigente: cambiar
de SDE = una red nueva y un entrenamiento desde cero (los samplers del Eje 2 reusan la misma
red sin reentrenar).

:func:`save_checkpoint` / :func:`load_checkpoint` son **model-agnósticos** (R5-c): guardan el
``state_dict`` de la red junto con una metadata mínima (nombre de la SDE, ``data_dim`` y una
**receta genérica** ``model={name, kwargs}`` opcional) y devuelven ``(state_dict, meta)`` **sin
reconstruir** la red. Es el caller quien reconstruye la red (vía ``make_model`` o una instancia
explícita) y carga el ``state_dict`` — así el mismo checkpoint sirve al ``ScoreMLP`` y a la
``ScoreUNet`` sin que ``training`` importe ninguna red concreta.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch

from ..models import ScoreModel
from ..sde import ForwardSDE
from .ema import EmaShadow
from .losses import dsm_loss
from .time_sampling import make_time_sampler
from .validation import FixedValExam, ValPoint, evaluate_with_weights


def _enable_cudnn_autotune(device: torch.device) -> None:
    """Autotune de kernels convolucionales en GPU (R4.1–4.3).

    Con un device **CUDA** prende ``torch.backends.cudnn.benchmark`` para que cuDNN elija los kernels
    convolucionales más rápidos para los shapes fijos de la corrida (la arquitectura es la variable de
    control del estudio: los shapes no cambian entre pasos). Es **auto-on en CUDA**, sin flag de
    config. En **CPU** no toca **nada** (R4.2): el estado global de autotune queda como estaba, así que
    el camino CPU —el de la suite de tests— no tiene efecto observable. Solo cambia la **elección de
    kernels**, no la arquitectura ni el objetivo (R4.3).
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


@dataclass
class TrainConfig:
    """Hiperparámetros del **loop** de entrenamiento (solo optimización y corrida).

    Ya no carga hiperparámetros de red (viven en el constructor de la red / ``make_model``) ni
    de tamaño de dataset (viven en la fuente de datos). Pensado para construirse desde un
    archivo de config (ver :mod:`diffusion.training.config`), pero es un dataclass plano y se
    arma igual de fácil a mano en los tests.
    """

    num_steps: int = 1000  # pasos de optimización (≈ epochs × n_samples/batch_size viejos)
    lr: float = 2e-3
    t_eps: float = 1e-3
    grad_clip: float | None = None
    seed: int | None = 0
    device: str = "cpu"
    log_every: int = 0  # 0 = silencioso; N = imprime (media móvil) cada N pasos. Solo consola.
    # Checkpointing intermedio: 0 = solo el checkpoint final (comportamiento por defecto, sin
    # regresión); N>0 = además, un snapshot periódico cada N pasos. El *cómo/dónde* persistir lo
    # decide el callback ``on_checkpoint`` de :func:`train` (el loop no toca el filesystem); sin
    # ese callback este campo no hace nada.
    checkpoint_every: int = 0
    # Retención rolling de snapshots intermedios: ``None`` = conservarlos **todos** (default, sin
    # regresión); ``N >= 1`` = conservar solo los ``N`` snapshots ``…_stepNNNNN`` más nuevos (con su
    # sidecar), borrando los más viejos a medida que se generan. El **borrado** —como el guardado— lo
    # hace el callback ``on_checkpoint`` del caller (``train`` no toca el filesystem): este campo es el
    # valor que el CLI lee para aplicar :func:`~diffusion.training.prune_snapshots` tras cada snapshot.
    # El checkpoint final nunca cuenta ni se borra. ``train`` no lo usa; un ``N < 1`` lo rechaza el CLI.
    keep_last_checkpoints: int | None = None
    # Cadencia de la **pérdida de validación**, independiente de la de los snapshots (3.10): ``None``
    # = usar ``checkpoint_every`` (default, comportamiento bit a bit idéntico al previo al campo);
    # ``N >= 1`` = evaluar cada N pasos sin importar cada cuánto se persisten snapshots. Existe
    # porque atar la medición a ``checkpoint_every`` mezclaba dos concerns: subir la resolución de la
    # curva obligaba a escribir más snapshots (708 MB cada uno en la celda de gatos), así que una
    # palanca de observabilidad cargaba con un costo de disco que no le corresponde. La consecuencia
    # es que las dos cadencias se eligen por separado —60 puntos de curva sin un snapshot de más, e
    # incluso medir con ``checkpoint_every=0``—; conviene mantener ``checkpoint_every`` **múltiplo**
    # de ``val_every`` para que cada snapshot guardado tenga un punto medido en su paso exacto. Un
    # ``N < 1`` declarado lo rechaza :func:`train` fail-fast (3.11); sin fuente de validación el campo
    # es inerte.
    val_every: int | None = None
    # Distribución de muestreo de t del loop (ver :mod:`diffusion.training.time_sampling`):
    # "uniform" = el default retrocompatible (misma secuencia por seed que antes del campo);
    # "log_uniform" = la recomendada para concentrar señal de entrenamiento en t chico (la
    # pérdida se corrige por likelihood ratio — mismo objetivo en esperanza, menos varianza).
    time_sampling: str = "uniform"
    # Sombra EMA de los pesos (ver :mod:`diffusion.training.ema`): ``None`` = **sin EMA**, el
    # default retrocompatible bit a bit (R1.1/R1.2 — cero ramas nuevas activas). Un decay finito en
    # el intervalo abierto (0, 1) —0.999 es el recomendado del estudio— activa la media móvil
    # exponencial con rampa de warmup que después publican los checkpoints; es lo que samplean las
    # implementaciones de referencia (Song, DDPM, EDM) en lugar de la foto del último paso de Adam.
    ema_decay: float | None = None
    # Precisión mixta (AMP, ver el núcleo de :func:`train`): ``False`` = **desactivada**, el default
    # retrocompatible bit a bit (R1.4 — cero ramas nuevas activas, camino de optimización idéntico al
    # actual). Con ``True`` el forward y la pérdida se computan bajo ``torch.autocast`` y —en GPU— el
    # gradiente se escala con un ``torch.amp.GradScaler`` para el paso del optimizador; en CPU el
    # autocast usa bfloat16 y el escalador queda en passthrough (el camino se ejercita igual en los
    # tests, sin GPU). Es opt-in y **booleano estricto**: un valor no ``bool`` revienta fail-fast en
    # :func:`train` antes de consumir el primer batch (R1.5).
    amp: bool = False


@dataclass
class TrainResult:
    """Resultado de :func:`train`: la red entrenada y la traza de la corrida."""

    net: ScoreModel  # cualquier red de score (ScoreMLP, ScoreUNet, …), no atada a una clase
    history: list[float] = field(default_factory=list)  # pérdida por paso (serie completa: una por paso)
    config: TrainConfig = field(default_factory=TrainConfig)
    sde_name: str = ""
    # = sde.data_dim (valor crudo): un entero para dato plano 2D o una tupla (forma de evento,
    # imágenes). Lo copia save_checkpoint a la meta y lo consume generate.py para reconstruir la SDE.
    data_dim: int | tuple[int, ...] = 0
    # Foto CLONADA de la sombra EMA (``EmaShadow.state_dict()``: mismas claves que el state_dict de
    # la red, con los parámetros promediados) cuando ``config.ema_decay`` está configurado; ``None``
    # sin EMA. Es lo que publican los checkpoints en lugar de los pesos crudos.
    ema_state: dict | None = None
    # Serie DISPERSA de validación (:class:`~diffusion.training.validation.ValPoint` por evaluación,
    # indexada por el paso en que se midió), al lado de la serie DENSA per-step de ``history``. Queda
    # **vacía** —el default— cuando la corrida no recibe fuente de validación, que es el caso de
    # todas las corridas anteriores a esta feature: el campo existe pero no agrega contenido (6.1).
    # Al reanudar continúa la serie del punto de reanudación en lugar de arrancar de cero (6.3).
    val_history: list[ValPoint] = field(default_factory=list)


@dataclass
class ResumeState:
    """Estado en memoria necesario para **reanudar** una corrida desde un checkpoint intermedio.

    Agrupa lo que :func:`train` no puede recuperar de los pesos: el estado del optimizador (los
    momentos de Adam), el número de paso ya alcanzado y el **azar** del loop —el RNG global de
    torch y el estado del ``generator`` del ruido del kernel / muestreo de ``t``—. El ``history``
    viaja acá en memoria (para poder continuar la curva al reanudar), pero **no** se persiste en el
    sidecar: ya vive en el ``meta`` del checkpoint de pesos (:func:`save_checkpoint`) y duplicarlo
    sería redundante (ver :func:`save_resume_state`); al cargar se rellena desde ese ``meta``.

    Con **EMA activo** el estado de la corrida queda *partido* entre los dos artefactos: el
    checkpoint de pesos publica la **sombra** (ver :func:`save_checkpoint`), así que los pesos
    **crudos** —lo que hace falta para seguir optimizando— y la sombra misma viajan acá, en el
    sidecar (R3.1). Son campos **opcionales**: sin EMA quedan en ``None`` y el sidecar no gana
    ninguna clave, así que los sidecars anteriores a la feature siguen siendo válidos (R3.3).

    Attributes:
        optimizer_state: ``optimizer.state_dict()`` (estado de Adam).
        start_step: Pasos ya completados (= ``N`` del nombre ``…_stepNNNNN``).
        torch_rng_state: ``torch.get_rng_state()`` (RNG global de torch).
        generator_state: ``generator.get_state()`` (RNG del ruido del kernel / muestreo de ``t``).
        history: Pérdida per-step hasta ``start_step`` (se rellena desde el ``meta`` al cargar).
        ema_state: Foto **clonada** de la sombra EMA del momento (``EmaShadow.state_dict()``) o
            ``None`` sin EMA. Al reanudar se restaura con ``EmaShadow.load_state`` en lugar de
            reconstruirse desde los pesos (una sombra nueva perdería el promedio acumulado).
        raw_model_state: Pesos **crudos** del momento (``net.state_dict()`` clonado) o ``None`` sin
            EMA. Es lo que :func:`~diffusion.training.load_resume` devuelve para cargar en la red
            cuando el checkpoint de pesos publica EMA; sin esta clave (sidecar viejo = corrida sin
            EMA) el checkpoint ya trae los crudos y se usa ese.
        scaler_state: Estado del escalador de gradiente AMP (``GradScaler.state_dict()``) o ``None``
            sin AMP. Campo **opcional** (calca ``ema_state``): con AMP el loop restaura este estado
            en su ``GradScaler`` para que la reanudación sea fiel; sin AMP queda en ``None`` y el
            sidecar no gana ninguna clave (R2.4). En CPU el escalador va deshabilitado, así que su
            ``state_dict()`` es ``{}`` (un dict **vacío**, no ``None``): "presencia" se decide por
            ``is None``, no por dict vacío.
        val_history: Serie de validación medida hasta ``start_step``
            (:class:`~diffusion.training.validation.ValPoint` por evaluación) o ``None`` si el punto
            de reanudación no la trae. Viaja acá en memoria como el ``history`` —y por el mismo
            motivo: **no** se persiste en el sidecar, vive en el ``meta`` del checkpoint de pesos y
            se rellena desde ahí al cargar—. Campo **opcional**, así que un checkpoint anterior a
            esta feature se reanuda igual que siempre. A diferencia de ``ema_state`` /
            ``scaler_state`` **no** tiene guard cruzado contra la config: la serie es una
            *observación*, no estado necesario para continuar la optimización, así que que falte
            solo hace que arranque vacía en lugar de corromper la corrida.
    """

    optimizer_state: dict
    start_step: int
    torch_rng_state: torch.Tensor
    generator_state: torch.Tensor
    history: list[float]
    ema_state: dict | None = None
    raw_model_state: dict | None = None
    scaler_state: dict | None = None
    val_history: list[ValPoint] | None = None


@dataclass
class TrainSnapshot:
    """Envoltorio del estado que viaja junto a los pesos en cada checkpoint intermedio.

    Agrupa el :class:`TrainResult` (pesos + history, lo que consume :func:`save_checkpoint`) y el
    :class:`ResumeState` (lo que consume :func:`save_resume_state` para el sidecar), de modo que el
    caller pueda persistir ambos artefactos —el checkpoint de pesos y el sidecar de resume— desde
    un único snapshot del loop.

    Attributes:
        result: Pesos + history del punto de checkpoint (para el checkpoint de pesos).
        resume: Estado para reanudar (para el sidecar ``…_resume.pt``).
    """

    result: TrainResult
    resume: ResumeState


def train(
    sde: ForwardSDE,
    model: ScoreModel,
    data: Iterator[torch.Tensor],
    config: TrainConfig,
    *,
    generator: torch.Generator | None = None,
    on_checkpoint: Callable[[str, TrainSnapshot], None] | None = None,
    on_log: Callable[[dict], None] | None = None,
    resume: ResumeState | None = None,
    progress: bool = False,
    val_batches: Iterable[torch.Tensor] | None = None,
    train_exam_batches: Iterable[torch.Tensor] | None = None,
) -> TrainResult:
    """Entrena la red ``model`` para aproximar el score de ``sde`` por DSM.

    Loop **por pasos** y agnóstico a la red: usa la ``model`` recibida (no construye ninguna ni
    ramifica por su tipo) y consume ``data`` con ``next()`` — un batch por paso.

    Es **reanudable**: sin ``resume`` entrena desde cero (paso 0, optimizador nuevo, azar
    sembrado con ``config.seed``); con un :class:`ResumeState` continúa una corrida previa —
    restaura el optimizador y el azar, arranca en el paso guardado y sigue el ``history``— hasta
    completar ``config.num_steps`` (interpretado como el **total** a alcanzar, no como pasos
    adicionales).

    Args:
        sde: Proceso forward (define el kernel y el target del score). Su ``data_dim`` queda
            registrado en el resultado (lo usa el checkpoint).
        model: Red de score ya construida (cualquier :class:`ScoreModel`). Se mueve al
            dispositivo de forma idempotente y se pone en modo entrenamiento. Al reanudar, el
            caller ya le cargó los pesos del checkpoint elegido.
        data: Iterador/iterable **infinito** que yield-ea tensores crudos ``(B, ...)`` (p. ej.
            ``infinite_bare(distribution.dataloader(...))``). Se le pide un batch por paso.
        config: Hiperparámetros del loop de entrenamiento. ``num_steps`` es el **total** de la
            corrida (al reanudar se corren solo los pasos restantes).
        generator: Generador opcional para el ruido del kernel / muestreo de ``t``. Sin
            ``resume`` se crea (si es ``None``) sembrado con ``config.seed``; con ``resume`` se
            crea si hace falta el objeto pero su estado se **restaura** desde el ``ResumeState``
            (no se re-siembra).
        on_checkpoint: Callback **opcional** de checkpointing intermedio. Se invoca con
            ``(tag, snapshot)`` donde ``tag`` es siempre ``"step{N:05d}"`` (snapshot **periódico**
            cada ``config.checkpoint_every`` pasos: es la única cadencia que emite el loop) y
            ``snapshot`` es un :class:`TrainSnapshot` — el :class:`TrainResult` con la red en
            ese punto **más** el :class:`ResumeState` (optimizador + paso + azar + history) para
            poder reanudar. El loop decide **cuándo** llamar; el callback decide **cómo/dónde**
            persistir — ``train`` no toca el filesystem. Sin este callback (o con
            ``checkpoint_every=0``) solo se entrena; el checkpoint final lo guarda el caller con
            :func:`save_checkpoint`.

            Hubo además un tag ``"best"`` (mínimo de una media de ventana de la pérdida)
            **retirado** el 27/07/2026 por decisión del autor (R2.6): seleccionar un checkpoint por
            la pérdida cruda per-step es ruidoso —el ``t`` de cada paso es aleatorio— y correlaciona
            mal con la calidad de las muestras, que es exactamente el problema que resuelve la
            sombra EMA; el mecanismo nunca se usó para ninguna decisión del estudio. Los
            ``X_best.pt`` que quedaron en disco siguen tolerados (excluidos) por
            :func:`diffusion.training.discover_snapshots`.
        resume: Estado de resume **opcional**. Si es ``None`` (default) el loop entrena desde
            cero (sin regresión). Si se provee, restaura el optimizador (``load_state_dict``) y el
            azar (``torch.set_rng_state`` + ``generator.set_state``) **sin re-sembrar**, arranca
            en ``resume.start_step``, continúa ``resume.history`` e itera
            ``range(start_step, num_steps)``. Si ``start_step >= num_steps`` no corre ningún paso
            y devuelve el resultado ya completo. Si trae ``ema_state`` (corrida con EMA) la sombra
            se restaura desde ahí; ver los guards de coherencia más abajo.
        on_log: Callback **opcional** de logging estructurado, con **dos variantes** de registro
            discriminadas por la clave opcional ``event``:

            - **Paso de entrenamiento** — ``{"step", "loss"}`` (paso 1-indexado completado y media
              móvil de la pérdida), **sin** clave ``event``: su ausencia *significa* "paso de
              entrenamiento". Cadencia propia —``config.log_every`` si es ``>0``, si no
              ``num_steps//20``— **independiente** del print/barra, así emite estados aun con
              ``log_every=0`` (p. ej. corridas `--quiet` en background).
            - **Validación** (solo con ``val_batches``) — ``{"event": "val", "step", "val_raw",
              "val_ema", "train_fijo", "device"}``, en la cadencia de la evaluación (la **efectiva**:
              ``config.val_every`` si se declara, si no ``config.checkpoint_every``; más el último
              paso). ``val_ema`` / ``train_fijo`` son ``None`` cuando no hay sombra EMA / fuente de
              examen de train. El ``device`` va en el
              registro porque el examen fijo es **específico del device**: sin ese dato, una serie
              medida en dos máquinas es indistinguible de una serie con un salto real del modelo.
              Se emite **después** de agregar el punto a ``val_history``.

            La asimetría (una variante con ``event`` y otra sin) es deliberada y retrocompatible: un
            consumidor que ya escribía los registros de paso sigue funcionando sin cambios, y —como
            expande el registro al final— el ``event`` del de validación sobreescribe su default. El
            caller decide qué hacer con cada registro (p. ej. escribirlo a un ``.jsonl`` con
            timestamp); ``train`` no toca el filesystem ni el reloj.
        progress: Si es ``True``, muestra una barra de progreso (``tqdm``) con porcentaje, ETA e
            it/s mientras entrena; la pérdida (media móvil de ``log_every``) va como postfix. Es
            **display-only** (default ``False``): no cambia el resultado ni el ``history``, escribe a
            stderr, y al reanudar la barra va de ``start_step`` a ``num_steps`` (así el % y el ETA
            son correctos). El import de ``tqdm`` es diferido: solo se carga si ``progress=True``.
        val_batches: Fuente **re-iterable** y **opcional** de batches de **validación** (imágenes
            held-out; típicamente lo que devuelve ``finite_batches``). Es la palanca **opt-in** de
            la pérdida de validación: con el default (``None``) no se construye nada —cero ramas
            nuevas activas— y la corrida es idéntica a la de antes de esta feature para la misma
            semilla (6.1). Con una fuente, el loop arma un
            :class:`~diffusion.training.validation.FixedValExam` en el bloque de fail-fast —antes de
            consumir el primer batch— y **exige una cadencia efectiva > 0**: ``config.val_every`` si
            se declara, si no ``config.checkpoint_every`` (3.1/3.10). Con las dos apagadas la corrida
            nunca mediría nada y se falla temprano (3.7); un ``val_every`` declarado por debajo de 1
            también (3.11). La validación **observa y no interviene**: corre sin gradientes, saca su
            azar de un generator propio y no altera los pesos, el optimizador ni la sombra EMA.
        train_exam_batches: Fuente **re-iterable** y **opcional** del **examen fijo de
            entrenamiento** (imágenes del set de train, la misma cantidad que las de validación),
            medido con el mismo procedimiento para que la distancia vertical entre las dos curvas
            sea generalización y no una diferencia de estimador (3.8). Se usa **solo** si hay
            ``val_batches``: sin fuente de validación no hay serie donde publicar su valor, así que
            se **ignora** en silencio (no se rechaza). Su examen comparte la semilla y el
            muestreador de ``t`` con el de validación, de modo que ambos sortean la misma secuencia.

    Mantiene además, si ``config.ema_decay`` está configurado, una **sombra EMA** de los pesos
    (:class:`~diffusion.training.ema.EmaShadow`): se construye antes de consumir datos (fail-fast
    ante un decay inválido), se actualiza después de cada paso del optimizador y viaja clonada en
    ``TrainResult.ema_state``. Es un observador pasivo — con la misma semilla, la trayectoria de
    optimización (pesos crudos e historia) es idéntica a la de una corrida sin EMA.

    Al **reanudar** una corrida con EMA la sombra se **restaura** desde ``resume.ema_state`` (no se
    reconstruye desde los pesos cargados, que valen θ_N y no el promedio acumulado), y los
    snapshots persisten en el sidecar los pesos **crudos** del momento junto a la sombra (R3.1) —
    los pesos del checkpoint son los EMA. Las dos combinaciones incoherentes entre la config y el
    sidecar (EMA pedido sin sombra guardada; sombra guardada sin EMA pedido) se rechazan con
    ``ValueError`` antes de entrenar: nunca se continúa con una sombra inventada ni descartada en
    silencio (R3.4).

    Con una fuente en ``val_batches`` mide además la **pérdida de validación** por examen fijo
    (:mod:`diffusion.training.validation`): los exámenes se construyen en el fail-fast, con el mismo
    ``TimeSampler`` que usa el loop —así el criterio de muestreo de ``t`` es el mismo por
    construcción (3.4)— y la serie medida viaja en ``TrainResult.val_history``. Sin la fuente no se
    construye nada y la corrida es la de siempre (6.1).

    La evaluación corre cuando el paso completado es múltiplo de la **cadencia efectiva** —
    ``config.val_every`` si se declara, si no ``config.checkpoint_every`` (3.10)— **o** es el último
    de la corrida (3.1). Es un disparador **independiente** del de los snapshots, que excluye el
    último paso y exige ``on_checkpoint``: la validación mide aunque la corrida no persista nada, y
    con ``val_every`` la resolución de la curva se elige sin escribir un snapshot de más (incluso con
    ``checkpoint_every=0``). Conviene, eso sí, mantener ``checkpoint_every`` **múltiplo** de
    ``val_every``, así cada snapshot guardado tiene un punto medido en su paso exacto; no es un
    requisito del código (con cadencias no alineadas el ``meta`` del checkpoint del paso *N*
    simplemente lleva los puntos acumulados hasta *N*).
    Cada evaluación produce hasta **tres** valores, en este orden: validación con los pesos vivos,
    validación con la sombra EMA si está activa (4.1; ``None`` si no, nunca un valor inventado, 4.2)
    y el examen fijo de entrenamiento con los pesos vivos (3.8, ``None`` sin su fuente). Va **después**
    del ``ema.update`` y del registro de la pérdida del paso, y **antes** del snapshot periódico, así
    que la foto del checkpoint del paso *N* ya contiene el punto medido en *N* (6.3). Cada punto se
    agrega a la serie y se emite por ``on_log`` como la variante ``event="val"`` (ver ``on_log``).

    Returns:
        :class:`TrainResult` con la red entrenada, la historia de pérdida (**serie per-step
        completa**: una entrada por paso, ``len(history) == num_steps`` cuando
        ``start_step < num_steps``; el ``history`` previo intacto en el caso no-op), el ``config``
        usado, el nombre de la SDE, su ``data_dim``, la foto de la sombra EMA en ``ema_state``
        (``None`` si el EMA no está activo) y la serie **dispersa** de validación en ``val_history``
        —un :class:`~diffusion.training.validation.ValPoint` por evaluación, indexado por el paso en
        que se midió (lista **vacía** sin fuente de validación; continúa la del ``resume`` si la
        trae)—.

    Con ``config.amp=True`` activa **precisión mixta** (opt-in, R1.1): el forward y la pérdida se
    computan bajo ``torch.autocast`` (bfloat16 en CPU) y —en GPU— el gradiente se escala con un
    ``torch.amp.GradScaler`` para el paso del optimizador, des-escalándolo **antes** del recorte
    (``grad_clip``) para que este opere sobre la norma real (R1.2). La sombra EMA se actualiza
    **después** del paso real (igual que sin AMP) y los pesos/sombra viven en float32, así que el
    checkpoint publica lo mismo y su formato no cambia (R1.3). Con el default (``False``) no se
    construye escalador y el núcleo de optimización queda **byte-idéntico** al previo (R1.4).

    Raises:
        ValueError: Si ``config.ema_decay`` es inválido (R1.6), si ``config.amp`` no es ``bool``
            (R1.5), si ``config.time_sampling`` no existe, si se pide validación (``val_batches``)
            con ``config.val_every`` declarado por debajo de 1 (3.11) o con la cadencia **efectiva**
            en 0 o menos —sin ``val_every`` y con ``config.checkpoint_every <= 0`` (3.7)—, o si al
            reanudar la config y el sidecar no coinciden en el uso del EMA (R3.4). Todos antes de
            consumir el primer batch.
    """
    device = torch.device(config.device)
    # Autotune de kernels convolucionales: auto-on en GPU (aprovecha los shapes fijos de la corrida),
    # sin efecto en CPU (R4.1–4.3). Setup de device, antes de mover la red o consumir data.
    _enable_cudnn_autotune(device)
    if resume is None:
        # Corrida desde cero: siembra el azar con config.seed (comportamiento histórico).
        if config.seed is not None:
            torch.manual_seed(config.seed)
        if generator is None:
            generator = torch.Generator(device=device)
            if config.seed is not None:
                generator.manual_seed(config.seed)
    elif generator is None:
        # Reanudación: NO se re-siembra; el estado del generator se restaura más abajo desde el
        # ResumeState. Igual hay que crear el objeto si el caller no lo pasó.
        generator = torch.Generator(device=device)

    # Muestreo de t configurable (R1): el sampler se construye UNA sola vez, fail-fast — un
    # nombre desconocido (o un t_eps fuera de (0, T)) revienta acá, antes de mover la red o de
    # consumir data. No toca el RNG: el default "uniform" reproduce la fórmula previa del loop
    # (misma llamada a torch.rand por paso), así que el stream del generator no cambia.
    time_sampler = make_time_sampler(config.time_sampling, sde.T, config.t_eps)

    net = model.to(device)  # idempotente: no falla si el caller ya la movió
    net.train()

    # Sombra EMA (opt-in, R1.1): se construye acá —después de mover la red al device (la sombra
    # clona sus tensores, tienen que estar ya en el device final) y ANTES de consumir datos—, así un
    # decay inválido revienta fail-fast con el mismo criterio que el time sampler (R1.6). Con el
    # default (``None``) no se construye nada: cero ramas nuevas activas, corrida bit a bit idéntica
    # a la previa (R1.2). La sombra es un observador **pasivo**: lee los pesos después del paso del
    # optimizador, no escribe en la red y no consume RNG (R1.4).
    ema = EmaShadow(net, config.ema_decay) if config.ema_decay is not None else None

    # Precisión mixta (AMP, opt-in, R1.1): fail-fast ante un ``amp`` mal formado —antes de mover más
    # estado o de consumir data— con el mismo criterio que el time sampler / el decay del EMA (R1.5).
    # ``bool`` estricto: un ``int`` (incluidos 0/1) o un ``float`` NO son booleanos (``isinstance(1,
    # bool)`` es ``False``), así que se rechazan en vez de degradar en silencio.
    if not isinstance(config.amp, bool):
        raise ValueError(
            f"config.amp debe ser bool (True/False); recibido {config.amp!r} "
            f"(tipo {type(config.amp).__name__})."
        )
    # Escalador de gradiente CONDICIONAL (calca EMA): se construye SOLO con AMP activo. En CPU el
    # escalado no aplica (autocast usa bfloat16), así que queda ``enabled=False`` (passthrough) pero
    # EXISTE para uniformar el camino; en GPU se habilita. Con ``amp=False`` el escalador es ``None`` y
    # el núcleo de optimización queda byte-idéntico al actual (R1.4).
    scaler = (
        torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
        if config.amp
        else None
    )

    # Validación por examen fijo (opt-in, 6.1): con ``val_batches=None`` —el default— no se construye
    # NADA, así que la corrida no gana ninguna rama activa y es idéntica a la de antes de esta
    # feature para la misma semilla. Con fuente, los exámenes se arman acá, junto al resto del
    # fail-fast: ANTES de consumir el primer batch, mismo criterio que el time sampler / el decay del
    # EMA / el ``amp``.
    #
    # La cadencia se rechaza en ESTE punto y no en el config layer (3.7/3.11) porque el loop es el
    # único que ve el valor **efectivo**: el CLI sobreescribe ``checkpoint_every`` *después* de armar
    # la corrida, así que un chequeo en la config se saltearía (o dispararía sobre un valor que la
    # corrida ya no usa).
    #
    # CADENCIA EFECTIVA (3.10): ``val_every`` si se declara, ``checkpoint_every`` si no. Las dos son
    # independientes a propósito — la resolución de la curva es observabilidad y no puede costar
    # snapshots de 708 MB—, así que con ``val_every`` la corrida mide incluso con los snapshots
    # apagados (``checkpoint_every=0``, que deja de ser un error). Se computa acá arriba, fuera del
    # ``if``, para que el disparador del loop la lea ya resuelta; sin fuente de validación puede
    # valer 0 y nunca se usa (el disparador la gatea con ``val_exam is not None``).
    val_cadence = config.val_every if config.val_every is not None else config.checkpoint_every
    # El valor DECLARADO se valida SIEMPRE (3.11), haya o no fuente de validación: es el criterio de
    # la casa para un knob mal formado —``keep_last_checkpoints`` se rechaza igual aunque no haya
    # checkpoints, y ``amp`` aunque el device sea CPU—. Un ``val_every: 0`` en un config sin
    # ``val_root`` es un malentendido del autor (creyó estar apagando la validación, o le falta la
    # clave de datos), y avisarlo cuesta nada; tolerarlo en silencio esconde el error hasta que
    # alguien agregue la fuente y la corrida reviente recién ahí.
    if config.val_every is not None and config.val_every < 1:
        raise ValueError(
            f"config.val_every={config.val_every} no es una cadencia válida: la validación se "
            "evalúa cada N pasos, así que N tiene que ser >= 1. Declarar 0 o un negativo NO es "
            "'usar el default' —eso lo dice val_every: null, que cae en checkpoint_every—: es "
            "pedir una cadencia imposible, y se rechaza en vez de degradar en silencio."
        )
    val_exam: FixedValExam | None = None
    train_exam: FixedValExam | None = None
    if val_batches is not None:
        # La cadencia EFECTIVA (3.7) sí depende de que haya fuente: sin validación, un
        # ``checkpoint_every=0`` es perfectamente válido y no hay nada que medir.
        if val_cadence <= 0:
            raise ValueError(
                "La corrida pide pérdida de validación pero se queda sin cadencia de evaluación: no "
                f"declara val_every y tiene los checkpoints deshabilitados "
                f"(config.checkpoint_every={config.checkpoint_every}), así que entrenaría completa "
                "sin medir ni un punto. Declará val_every > 0 —fija la cadencia de la validación con "
                "independencia de los snapshots, así que podés medir sin escribir ninguno— o poné "
                "checkpoint_every > 0; si no querés medir, quitá la fuente de validación."
            )
        # Los dos exámenes reciben el ``time_sampler`` recién construido —EL MISMO objeto que usa el
        # loop— y la semilla por defecto del examen: así "val y train usan el mismo criterio de
        # muestreo de t" (3.4) es estructural en lugar de depender de que alguien lo recuerde, y
        # ambos exámenes sortean la misma secuencia de t y de ruido (3.8), de modo que la diferencia
        # entre las dos curvas no pueda venir del examen.
        val_exam = FixedValExam(sde, val_batches, time_sampler=time_sampler, device=device)
        if train_exam_batches is not None:
            train_exam = FixedValExam(
                sde, train_exam_batches, time_sampler=time_sampler, device=device
            )
    # NB: ``train_exam_batches`` sin ``val_batches`` se IGNORA y no se rechaza — sin fuente de
    # validación no hay serie donde publicar su valor, así que no hay nada que hacer con esa fuente.

    # Coherencia config ↔ sidecar al reanudar (R3.4). Con EMA el estado de la corrida vive PARTIDO
    # (el checkpoint publica la sombra; los crudos y la sombra van en el sidecar), así que las dos
    # combinaciones cruzadas arrancarían de un estado inconsistente y hay que fallar en las dos —
    # acá, junto al resto del fail-fast: antes de restaurar el optimizador y antes de consumir data.
    if resume is not None:
        if ema is not None and resume.ema_state is None:
            raise ValueError(
                f"La corrida configuró ema_decay={config.ema_decay} pero el punto de reanudación "
                "no trae la sombra EMA: el sidecar de resume del checkpoint elegido no la "
                "persistió (¿la corrida original entrenó sin EMA?). No se reanuda con una sombra "
                "inventada —perdería el promedio de todos los pasos ya corridos y el checkpoint "
                "final publicaría un EMA que no corresponde a la corrida—: entrená desde cero o "
                "reanudá con la misma configuración que la corrida original."
            )
        if ema is None and resume.ema_state is not None:
            raise ValueError(
                "El punto de reanudación trae una sombra EMA en su sidecar de resume (la corrida "
                "original la mantenía) pero esta corrida no configuró ema_decay: continuar "
                "descartaría la sombra en silencio y el checkpoint final publicaría los pesos "
                "crudos donde los intermedios de la misma corrida publicaron el promedio. "
                "Declará el ema_decay de la corrida original o reanudá desde otro punto."
            )
        if ema is not None:
            # Restaurada, no reconstruida: la sombra recién construida arriba vale θ_N (los pesos
            # cargados), no el promedio acumulado hasta el paso N.
            ema.load_state(resume.ema_state)

        # Coherencia del ESCALADOR AMP (R2.5), calca los guards de EMA. El escalador existe (no
        # ``None``) exactamente cuando ``config.amp`` está activo; "presencia" del estado guardado se
        # decide por ``is None`` (en CPU el escalador va deshabilitado y su ``state_dict()`` es ``{}``,
        # que cuenta como presente). Las dos combinaciones cruzadas arrancarían de un estado de
        # precisión inconsistente, así que se fallan las dos antes de restaurar el optimizador.
        if scaler is not None and resume.scaler_state is None:
            raise ValueError(
                "La corrida configuró amp=True pero el punto de reanudación no trae el estado del "
                "escalador de gradiente: el sidecar de resume del checkpoint elegido no lo "
                "persistió (¿la corrida original entrenó sin AMP?). No se reanuda con un escalador "
                "inventado —el factor de escala continuaría desde un valor que no corresponde a la "
                "corrida—: entrená desde cero o reanudá con la misma configuración de precisión que "
                "la corrida original."
            )
        if scaler is None and resume.scaler_state is not None:
            raise ValueError(
                "El punto de reanudación trae el estado del escalador de gradiente en su sidecar de "
                "resume (la corrida original usaba amp=True) pero esta corrida no configuró amp: "
                "continuar descartaría el escalador en silencio y la reanudación no sería fiel a la "
                "corrida original. Declará amp=True como en la corrida original o reanudá desde otro "
                "punto."
            )
        if scaler is not None:
            # Restaurado, no reconstruido: el escalador recién construido arriba arranca con el
            # factor de escala inicial, no con el que la corrida ya había ajustado hasta el paso N.
            scaler.load_state_dict(resume.scaler_state)

    data_iter = iter(data)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    # Estado inicial del loop: desde cero o restaurado de un ResumeState.
    if resume is None:
        start_step = 0
        history: list[float] = []
        val_history: list[ValPoint] = []
    else:
        # Restaurar optimizador y azar antes de continuar (2.1). El load_state_dict actúa además
        # de guard de compatibilidad: levanta si las shapes del optimizador no corresponden (2.5).
        optimizer.load_state_dict(resume.optimizer_state)
        torch.set_rng_state(resume.torch_rng_state)
        generator.set_state(resume.generator_state)
        start_step = resume.start_step
        history = list(resume.history)  # continuar la curva previa (2.3)
        # Serie de validación: se **continúa** la del punto de reanudación (6.3) y se COPIA, igual
        # que el history, para no mutar la lista del ``ResumeState`` recibido. Un punto de
        # reanudación sin serie (``None``: un checkpoint anterior a esta feature, o una corrida
        # original sin validación) simplemente arranca vacío — no hay guard cruzado contra la
        # config, a diferencia del EMA y del escalador: la serie es una observación, no estado
        # necesario para continuar la optimización.
        val_history = list(resume.val_history or [])

    # ``history`` guarda la pérdida de CADA paso (serie completa, la fuente de verdad).
    # ``log_every`` gobierna solo el print de consola, desacoplado.

    # Checkpointing intermedio (opt-in): activo solo si el caller inyecta un callback y
    # ``checkpoint_every > 0``. El loop decide *cuándo* (una única cadencia periódica); el callback
    # decide *cómo/dónde* persistir — ``train`` no toca el filesystem.
    do_checkpoints = on_checkpoint is not None and config.checkpoint_every > 0

    def _result() -> TrainResult:
        # Foto del TrainResult actual (pesos vía la red viva + copia del history + foto de la
        # sombra si el EMA está activo). ``EmaShadow.state_dict()`` ya devuelve CLONES, así que la
        # foto no se mueve cuando la sombra sigue avanzando: es lo que hace seguro publicarla en un
        # checkpoint intermedio sin congelar el loop.
        return TrainResult(
            net=net,
            history=list(history),
            config=config,
            sde_name=sde.name,
            data_dim=sde.data_dim,
            ema_state=ema.state_dict() if ema is not None else None,
            # Copia de la serie dispersa, con el mismo criterio que el ``history``: la foto de un
            # checkpoint intermedio no puede seguir creciendo cuando la corrida agrega puntos.
            val_history=list(val_history),
        )

    def _snapshot(completed_steps: int) -> TrainSnapshot:
        # Foto completa para que el callback persista pesos y sidecar: el TrainResult (pesos +
        # history) más el ResumeState del momento (optimizador + paso + azar + history). El azar
        # se lee DESPUÉS del paso, así que reanudar desde acá continúa el mismo stream (2.6).
        #
        # Con EMA activo el sidecar lleva además los pesos CRUDOS y la sombra del momento (R3.1):
        # el checkpoint de pesos publica la sombra, así que sin los crudos acá una reanudación
        # continuaría desde los pesos promediados. Los crudos se CLONAN explícitamente (el loop
        # sigue mutando los tensores de la red in-place; una referencia viva haría que el sidecar
        # del paso N terminara con los pesos del final de la corrida) y la sombra ya viene clonada
        # de ``EmaShadow.state_dict()``.
        return TrainSnapshot(
            result=_result(),
            resume=ResumeState(
                optimizer_state=optimizer.state_dict(),
                start_step=completed_steps,  # pasos ya completados (= N del tag step{N:05d})
                torch_rng_state=torch.get_rng_state(),
                generator_state=generator.get_state(),
                history=list(history),
                ema_state=ema.state_dict() if ema is not None else None,
                raw_model_state=(
                    {k: v.detach().clone() for k, v in net.state_dict().items()}
                    if ema is not None
                    else None
                ),
                # Estado del escalador AMP (R2.1): presente exactamente cuando el escalador existe
                # (AMP activo). En CPU es ``{}`` (deshabilitado) pero se persiste igual —presencia por
                # ``is None``—; sin AMP queda ``None`` y el sidecar no gana ninguna clave (R2.4).
                scaler_state=scaler.state_dict() if scaler is not None else None,
                # Serie de validación del momento, copiada (mismo criterio que el ``history``). Viaja
                # acá por simetría con el estado en memoria del punto de reanudación; el sidecar
                # NO la persiste —vive en el ``meta`` del checkpoint de pesos, como el ``history``—.
                val_history=list(val_history),
            ),
        )

    # ``num_steps`` es el TOTAL a alcanzar: se corren solo los pasos restantes (2.2). Si el paso
    # inicial ya lo alcanzó/superó, el rango es vacío y no se ejecuta ningún paso (no-op, 2.4).
    step_iter = range(start_step, config.num_steps)
    pbar = None
    if progress:
        # Barra de progreso opt-in (display-only). Import diferido: ``tqdm`` no se arrastra si no se
        # pide. ``initial``/``total`` hacen que el % y el ETA sean correctos al reanudar (la barra va
        # de ``start_step`` a ``num_steps``). Escribe a stderr; no altera el resultado ni el history.
        from tqdm.auto import tqdm

        pbar = tqdm(
            step_iter,
            initial=start_step,
            total=config.num_steps,
            unit="paso",
            desc=sde.name,
            dynamic_ncols=True,
        )
        step_iter = pbar

    # Cadencia del logging estructurado (on_log): sigue log_every si está seteado, si no cae en
    # num_steps//20 — INDEPENDIENTE del print/barra, para emitir estados también con log_every=0.
    log_cadence = config.log_every if config.log_every > 0 else max(1, config.num_steps // 20)

    for step in step_iter:
        # Transferencia no bloqueante (R3.3): `non_blocking=True` es INCONDICIONAL — inofensivo sin
        # memoria fijada y en CPU (torch lo ignora si no aplica); su beneficio real aparece con
        # `pin_memory=True` + CUDA. `train()` no conoce el `pin_memory` del loader, así que no se gatea.
        x0 = next(data_iter).to(device, non_blocking=True)
        # Tiempos y pesos por paso: la uniforme devuelve weights=None (camino idéntico al
        # previo); una variante no uniforme devuelve el likelihood ratio y la pérdida lo aplica.
        t, sample_weights = time_sampler.sample(
            x0.shape[0], generator=generator, device=device
        )
        # Forward + pérdida bajo autocast (R1.1): con ``amp=False`` es ``enabled=False``, no-op bit a
        # bit (R1.4); con ``amp=True`` computa en la precisión reducida del device (bfloat16 en CPU).
        # El ``backward`` queda FUERA del autocast (contrato de torch.amp).
        with torch.autocast(device_type=device.type, enabled=config.amp):
            loss = dsm_loss(
                net, sde, x0, t, generator=generator, sample_weights=sample_weights
            )

        optimizer.zero_grad()
        if scaler is None:
            # Camino sin AMP: exactamente el núcleo previo (R1.4) — byte-idéntico.
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            optimizer.step()
        else:
            # Camino con AMP (R1.1): backward sobre la pérdida escalada. El recorte debe operar sobre
            # la norma REAL del gradiente, así que se DES-ESCALA antes del clip (R1.2); recién después
            # el escalador da el paso y actualiza su factor.
            scaler.scale(loss).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        # La sombra se actualiza DESPUÉS del paso del optimizador, con el contador de pasos
        # completados que fija la convención de :class:`EmaShadow` (1-indexado): el ``step`` del
        # rango 0-indexado del loop equivale al paso completado ``step + 1`` (R1.3). Una sola
        # llamada por paso: la rampa de warmup depende del contador, así que un update de más o de
        # menos cambiaría la ponderación.
        if ema is not None:
            ema.update(step + 1)

        history.append(loss.item())  # serie completa: la pérdida de cada paso
        is_last = step == config.num_steps - 1

        # Evaluación de validación por examen fijo (3.1). Ubicada ACÁ a propósito: después del
        # ``ema.update`` (así la sombra que se mide es la del paso ya completado) y del
        # ``history.append``, y ANTES del snapshot periódico — de ese modo la foto del checkpoint del
        # paso N ya contiene el punto medido en N, y reanudar desde ahí restaura todos los puntos ya
        # medidos en lugar de perder el último (6.3).
        #
        # El DISPARADOR NO ES EL DE LOS SNAPSHOTS, aunque se le parezca: corre sobre la **cadencia
        # efectiva** (``val_every`` si se declaró, ``checkpoint_every`` si no — 3.10), incluye el
        # último paso (``is_last``, que el snapshot excluye porque el checkpoint final lo guarda el
        # caller) y no exige ``on_checkpoint`` (``do_checkpoints``), porque la validación es una
        # medición de la corrida y no un artefacto de checkpointing — una corrida sin persistencia
        # igual mide. Con ``val_every`` declarado los dos ni siquiera comparten la cadencia. La
        # ``val_cadence`` está garantizada > 0 acá por el fail-fast de más arriba (3.7/3.11).
        # Conflacionarlos es el error más probable de esta feature: reusar la condición del snapshot
        # perdería el punto final; reusar esta para el snapshot escribiría snapshots de más.
        if val_exam is not None and ((step + 1) % val_cadence == 0 or is_last):
            # Hasta TRES mediciones, en este orden. La primera con los pesos vivos.
            val_raw = val_exam.evaluate(net)
            # La segunda con la sombra EMA, y solo si está activa (4.1): sin sombra el valor queda
            # explícitamente AUSENTE (``None``, 4.2) en lugar de repetir el de los crudos —una
            # segunda curva idéntica sugeriría que el promediado no aporta nada, que es una
            # conclusión y no un dato faltante—. El swap clona los crudos, carga la sombra, mide el
            # MISMO examen y restaura in-place en un ``finally``: el optimizador conserva sus
            # referencias a los tensores-parámetro y los pasos siguientes no se desvían.
            val_ema = (
                evaluate_with_weights(val_exam, net, ema.state_dict())
                if ema is not None
                else None
            )
            # La tercera es el examen fijo de ENTRENAMIENTO, medido **solo con los pesos vivos**
            # (3.9): la lectura raw↔EMA se hace sobre la curva de validación, así que una cuarta
            # medición duplicaría el costo de la evaluación sin agregar información.
            train_fijo = train_exam.evaluate(net) if train_exam is not None else None

            punto: ValPoint = {
                "step": step + 1,  # paso completado, 1-indexado (misma convención que el EMA)
                "raw": val_raw,
                "ema": val_ema,
                "train": train_fijo,
            }
            val_history.append(punto)
            # Emisión por el callback YA EXISTENTE, como segunda variante de registro discriminada
            # por ``event`` (5.1): no se agrega un segundo callback porque no daría capacidad nueva
            # —el CLI expande el registro al final, así que un ``event`` propio sobreescribe su
            # default— y el loop sigue sin tocar el filesystem ni el reloj (el timestamp lo pone el
            # caller). El ``device`` viaja en el registro porque el examen fijo es específico del
            # device: sin ese dato, una serie medida en dos máquinas es indistinguible de una serie
            # con un salto real del modelo.
            if on_log is not None:
                on_log(
                    {
                        "event": "val",
                        "step": step + 1,
                        "val_raw": val_raw,
                        "val_ema": val_ema,
                        "train_fijo": train_fijo,
                        "device": str(device),
                    }
                )

        # Snapshot periódico: cadencia propia, chequeada cada paso para que ``checkpoint_every``
        # no tenga que ser múltiplo de nada. El último paso lo cubre el checkpoint final del
        # caller, así que se excluye acá.
        if do_checkpoints and not is_last and (step + 1) % config.checkpoint_every == 0:
            on_checkpoint(f"step{step + 1:05d}", _snapshot(step + 1))

        # Logging estructurado (p. ej. a un .jsonl): cadencia propia, independiente del print/barra,
        # así emite estados aun con log_every=0 (corridas --quiet en background). El caller le pone el
        # timestamp y decide dónde escribirlo — train no toca el filesystem ni el reloj.
        if on_log is not None and ((step + 1) % log_cadence == 0 or is_last):
            recent_log = history[-log_cadence:]
            on_log({"step": step + 1, "loss": sum(recent_log) / len(recent_log)})

        # Progreso en consola (solo display, desacoplado del history): media móvil de los últimos
        # log_every pasos. Con la barra activa va como postfix (para no romper la línea de la barra);
        # sin barra, un print como siempre.
        if config.log_every > 0 and ((step + 1) % config.log_every == 0 or is_last):
            recent = history[-config.log_every:]
            avg = sum(recent) / len(recent)
            if pbar is not None:
                pbar.set_postfix(perdida=f"{avg:.6f}")
            else:
                print(
                    f"[{sde.name}] paso {step + 1}/{config.num_steps}  "
                    f"pérdida(móvil)={avg:.6f}"
                )

    if pbar is not None:
        pbar.close()

    return TrainResult(
        net=net,
        history=history,
        config=config,
        sde_name=sde.name,
        data_dim=sde.data_dim,
        # Foto final de la sombra (clonada), lo que el checkpoint publica cuando el EMA está activo.
        ema_state=ema.state_dict() if ema is not None else None,
        # Serie dispersa completa de la corrida: vacía sin fuente de validación (6.1), y con los
        # puntos previos al frente cuando se reanudó una corrida que ya había medido (6.3).
        val_history=val_history,
    )


# ----------------------------------------------------------------- persistencia


def save_checkpoint(
    result: TrainResult,
    path: str | pathlib.Path,
    *,
    model_spec: dict | None = None,
    raw_sibling: bool = False,
) -> pathlib.Path:
    """Guarda la red entrenada y su metadata **model-agnóstica** en ``path`` (``.pt`` de torch).

    El checkpoint es una receta portable: guarda el ``state_dict`` de la red y una ``meta``
    sin hiperparámetros de arquitectura hardcodeados. La receta de red (``model``) es
    **opcional** y la aporta el caller: sin ella el checkpoint sigue siendo válido, pero al
    generar habrá que pasar una red explícita (ver :func:`load_checkpoint` y
    :func:`diffusion.samplers.generate_from_checkpoint`).

    **Punto único de publicación del EMA**: si ``result.ema_state`` está presente (la corrida
    configuró ``TrainConfig.ema_decay``), lo que se guarda como ``model_state`` es la **sombra
    EMA** —no los pesos vivos del último paso de Adam— y la meta gana la marca de trazabilidad
    ``ema = {"decay": …}``. El formato del blob no cambia (``{model_state, meta}``), así que la
    generación, el wrapper ε y las mediciones consumen los pesos promediados **sin modificarse**.
    Todos los checkpoints (el final y los intermedios del callback ``on_checkpoint``) pasan por
    acá, así que la política vale para los dos. Sin EMA el contenido es **idéntico al actual**
    (mismos pesos, misma meta, sin la clave ``ema``).

    Convención de lectura de la marca (la que interpretan los consumidores):

    - meta **sin** ``ema`` ⇒ los pesos publicados son **crudos** (retrocompatibilidad: es lo que
      son todos los checkpoints anteriores a esta feature).
    - meta **con** ``ema = {"decay": d}`` ⇒ los pesos publicados son la sombra EMA con ese decay.
    - meta con ``raw_of = "<archivo>.pt"`` (y **sin** ``ema``) ⇒ es el *hermano de crudos* de
      ``raw_sibling`` (ver abajo): pesos crudos, contraparte del checkpoint EMA que nombra.
    - meta con ``val_history = [ValPoint, …]`` ⇒ la corrida midió **validación** y ahí viaja su serie
      dispersa (un punto por evaluación, ver :class:`~diffusion.training.validation.ValPoint`). La
      clave se escribe **solo si la serie no está vacía**, con el mismo patrón que ``ema``: sin
      validación el contenido queda bit a bit igual al de antes de esa feature, así que meta **sin**
      ``val_history`` ⇒ corrida sin validación (es lo que son todos los checkpoints previos). Viaja
      por la misma ruta que ``history`` —el ``meta`` de los pesos, no el sidecar de resume— y de ahí
      la recupera :func:`~diffusion.training.load_resume` para continuar la serie al reanudar. Cada
      punto es un ``dict`` pelado a propósito: leer el checkpoint no depende de que la clase
      ``ValPoint`` exista ni sea importable.

    Args:
        result: Resultado de :func:`train`.
        path: Ruta de salida (se crean los directorios intermedios).
        model_spec: Receta genérica de la red ``{"name": str, "kwargs": dict}`` para poder
            reconstruirla vía :func:`diffusion.models.make_model`. Si es ``None`` no se guarda
            la clave ``model`` (la red se pasa aparte al generar).
        raw_sibling: Si es ``True`` **y** la corrida publica EMA, escribe además un checkpoint
            hermano ``{stem}_raw.pt`` con los **pesos crudos finales** (``result.net``): un
            checkpoint estándar (mismo formato, misma receta) cuya meta no lleva la marca ``ema``
            y sí ``raw_of`` apuntando al principal — así queda inequívoco que sus pesos son crudos
            y de qué corrida son contraparte. Habilita la comparativa crudo-vs-EMA de la **misma**
            corrida. Lo activan los guardados **finales** (los intermedios no lo necesitan: sus
            crudos ya viajan en el sidecar de resume). Sin EMA activo no escribe nada: el
            principal ya publica los crudos y el hermano sería un duplicado exacto. El sufijo
            ``_raw`` **no** matchea el patrón ``_stepNNNNN`` de
            :func:`diffusion.training.discover_snapshots`, así que nunca se elige como punto de
            reanudación.

    Returns:
        La ruta donde se guardó el checkpoint **principal** (como :class:`pathlib.Path`); la del
        hermano de crudos, cuando se escribe, es ``out.with_stem(out.stem + "_raw")``.
    """
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "sde_name": result.sde_name,
        # = sde.data_dim (int o tupla; lo registró train); torch.save serializa la tupla sin
        # problema y generate.py la reusa para reconstruir la SDE con su forma de evento.
        "data_dim": result.data_dim,
        "history": list(result.history),
    }
    if model_spec is not None:
        meta["model"] = model_spec  # receta genérica {name, kwargs}, independiente de la clase

    # Serie dispersa de validación: clave OPCIONAL, presente solo si la corrida midió algo (5.2).
    # El ``if`` es lo que sostiene la retrocompatibilidad por construcción —igual que con ``ema`` más
    # abajo y que con las claves opcionales del sidecar—: con la serie vacía (toda corrida sin
    # ``val_batches``, es decir todas las anteriores a esta feature) el ``meta`` no gana NINGUNA clave
    # y queda bit a bit igual al de antes (5.5). Se copia la lista, con el mismo criterio que el
    # ``history``: la meta que se serializa no debe aliasar la serie viva del loop.
    if result.val_history:
        meta["val_history"] = list(result.val_history)

    # Publicación: la sombra EMA cuando está (la foto ya viene clonada de EmaShadow.state_dict) o
    # los pesos vivos como siempre. Con ema_state=None no se agrega NINGUNA clave a la meta: el
    # contenido queda bit a bit igual al de antes de esta feature.
    publica_ema = result.ema_state is not None
    if publica_ema:
        meta["ema"] = {"decay": result.config.ema_decay}
    blob = {
        "model_state": result.ema_state if publica_ema else result.net.state_dict(),
        "meta": meta,
    }
    torch.save(blob, out)

    if raw_sibling and publica_ema:
        # Hermano de crudos: checkpoint estándar con los pesos vivos y la MISMA meta, salvo que la
        # marca de EMA se reemplaza por el puntero al principal (sin ``ema`` = pesos crudos).
        # El filtro excluye SOLO ``ema``, así que ``val_history`` viaja al hermano — y es lo que
        # corresponde, deliberadamente: los dos checkpoints son dos publicaciones de pesos del MISMO
        # entrenamiento, y la serie describe el entrenamiento, no el juego de pesos publicado.
        raw_meta = {clave: valor for clave, valor in meta.items() if clave != "ema"}
        raw_meta["raw_of"] = out.name
        torch.save(
            {"model_state": result.net.state_dict(), "meta": raw_meta},
            out.with_stem(f"{out.stem}_raw"),
        )
    return out


def load_checkpoint(
    path: str | pathlib.Path, *, map_location: torch.device | str = "cpu"
) -> tuple[dict, dict]:
    """Carga un checkpoint de :func:`save_checkpoint` como ``(state_dict, meta)``.

    **No reconstruye** ninguna red concreta (R5-c): devuelve el ``state_dict`` crudo y la
    metadata, y es el caller quien arma la red (vía :func:`diffusion.models.make_model` con la
    receta ``meta["model"]`` o una instancia propia) y le carga el ``state_dict``. Así
    ``training`` no depende de ninguna clase de red.

    Args:
        path: Ruta del ``.pt`` guardado.
        map_location: Dispositivo donde cargar los pesos (default ``"cpu"``).

    Returns:
        ``(state_dict, meta)``: el ``state_dict`` de la red y el ``dict`` de metadata guardado.

    Raises:
        KeyError: Si el archivo no tiene la forma de un checkpoint de :func:`save_checkpoint`
            (faltan ``"model_state"`` o ``"meta"``).
    """
    # weights_only=False: es nuestro propio checkpoint (incluye un dict de metadata).
    blob = torch.load(path, map_location=map_location, weights_only=False)
    return blob["model_state"], blob["meta"]


# --------------------------------------------------- sidecar de resume (C1)

# Campos obligatorios de un ``ResumeState`` a persistir: sin cualquiera de ellos la reanudación
# es imposible, así que se falla antes de escribir (1.4). El ``history`` queda fuera a propósito
# (vive en el ``meta`` del checkpoint de pesos; no se duplica, 1.3).
_REQUIRED_RESUME_FIELDS = ("optimizer_state", "start_step", "torch_rng_state", "generator_state")


def save_resume_state(
    path: str | pathlib.Path, resume: ResumeState
) -> pathlib.Path:
    """Persiste el **sidecar de resume** de un checkpoint (``torch.save``), sin el ``history``.

    El sidecar es un archivo aparte del checkpoint de pesos: guarda solo lo que ese no tiene —el
    estado del optimizador, el paso alcanzado y el azar (RNG global de torch + estado del
    ``generator``)—. El ``history`` se **omite** a propósito: ya está en el ``meta`` del checkpoint
    de pesos (:func:`save_checkpoint`) y se recupera de ahí al reanudar (1.3). El checkpoint de
    pesos no se toca ni cambia de formato/tamaño (1.2).

    Con **EMA activo** el sidecar lleva además dos claves **opcionales** (R3.1): ``raw_model_state``
    (los pesos crudos del momento — el checkpoint de pesos publica la sombra, así que sin ellos no
    se podría continuar optimizando) y ``ema_state`` (la sombra, para restaurarla en el loop). Se
    escriben **solo si están**: los campos requeridos no cambian, así que el sidecar de una corrida
    sin EMA queda idéntico en contenido al de antes de la feature y los sidecars viejos siguen
    siendo válidos (R3.3).

    Con **AMP activo** el sidecar lleva además la clave **opcional** ``scaler_state`` (el estado del
    ``GradScaler``, para restaurarlo en el loop y que la reanudación sea fiel — R2.1). Igual que las
    de EMA, se escribe **solo si está**: sin AMP el sidecar no gana ninguna clave y los sidecars
    previos siguen siendo válidos (R2.4).

    Args:
        path: Ruta de salida del sidecar (se crean los directorios intermedios). Convención de
            nombre: ``X_stepNNNNN.resume.pt`` hermano de ``X_stepNNNNN.pt``.
        resume: Estado a persistir. Debe tener el optimizador, el paso y ambos estados de RNG
            presentes (no ``None``); ``raw_model_state`` / ``ema_state`` son opcionales (van solo
            si la corrida tiene EMA).

    Returns:
        La ruta donde se guardó (como :class:`pathlib.Path`).

    Raises:
        ValueError: Si falta alguno de ``optimizer_state`` / ``start_step`` / ``torch_rng_state``
            / ``generator_state`` (estado incompleto). No se escribe un sidecar parcial (1.4).
    """
    missing = [name for name in _REQUIRED_RESUME_FIELDS if getattr(resume, name) is None]
    if missing:
        raise ValueError(
            "Estado de resume incompleto: falta(n) "
            f"{', '.join(missing)}. No se persiste un sidecar parcial "
            "(se requieren optimizer_state, start_step, torch_rng_state y generator_state)."
        )

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "optimizer_state": resume.optimizer_state,
        "step": resume.start_step,  # el sidecar usa la clave 'step' (= start_step)
        "torch_rng_state": resume.torch_rng_state,
        "generator_state": resume.generator_state,
        # NB: 'history' NO se persiste acá (vive en el meta del checkpoint de pesos; 1.3).
    }
    # Claves opcionales de EMA: presentes solo en corridas con sombra (R3.1). El ``if`` es lo que
    # sostiene la retrocompatibilidad por construcción — sin EMA el blob no gana ninguna clave.
    if resume.raw_model_state is not None:
        blob["raw_model_state"] = resume.raw_model_state
    if resume.ema_state is not None:
        blob["ema_state"] = resume.ema_state
    # Estado del escalador AMP: presente solo en corridas con AMP (R2.1). Mismo patrón que EMA — el
    # ``if`` sostiene la retrocompatibilidad por construcción: sin AMP el blob no gana la clave (R2.4).
    if resume.scaler_state is not None:
        blob["scaler_state"] = resume.scaler_state
    torch.save(blob, out)
    return out


def load_resume_state(
    path: str | pathlib.Path, *, map_location: torch.device | str = "cpu"
) -> dict:
    """Carga un sidecar de :func:`save_resume_state` como ``dict``.

    Args:
        path: Ruta del sidecar ``…_resume.pt``.
        map_location: Dispositivo donde cargar los tensores (default ``"cpu"``).

    Returns:
        ``{optimizer_state, step, torch_rng_state, generator_state}`` (sin ``history``: se toma del
        ``meta`` del checkpoint de pesos al reanudar), más ``raw_model_state`` y ``ema_state`` **si
        el sidecar los trae** (corrida con EMA, R3.1) y ``scaler_state`` **si lo trae** (corrida con
        AMP, R2.1). Un sidecar anterior a esas features no tiene esas claves y se lee igual que
        siempre (R3.3/R2.4), así que el consumidor las pide con ``get``.
    """
    # weights_only=False: es nuestro propio archivo (incluye el state_dict del optimizador).
    return torch.load(path, map_location=map_location, weights_only=False)
