"""CLI para entrenar la red de score por denoising score matching desde un config YAML.

Cada corrida (una celda del estudio de ablación) se describe en un ``.yaml`` con secciones
``sde`` / ``data`` / ``train`` / ``model`` (opcional) / ``out``. Ver ``config/toy/vp_mixture.yaml``.

Ejemplos (correr desde ``diffusion-models/``)::

    python scripts/train.py --config config/toy/vp_mixture.yaml
    python scripts/train.py --config config/toy/vp_mixture.yaml --num-steps 50 --device cpu
    python scripts/train.py --config config/toy/vp_mixture.yaml --checkpoint-every 50

Guarda los pesos entrenados (``.pt`` con ``state_dict`` + metadata) y una curva de pérdida
(``.png``) en las rutas de la sección ``out`` del config (relativas al cwd). Con
``train.checkpoint_every > 0`` (o ``--checkpoint-every``) guarda además, junto al checkpoint
final, un snapshot periódico ``…_stepNNNNN.pt`` cada N pasos (con su sidecar de resume). El
``…_best.pt`` que se guardaba antes **ya no se emite** (retirado el 27/07/2026, R2.6 de
``ema-weights``: elegir un checkpoint por la pérdida cruda per-step es ruidoso y correlaciona mal
con la calidad de las muestras — el problema que resuelve el EMA).

Con ``train.ema_decay`` configurado los checkpoints publican la **sombra EMA** de los pesos (ver
:func:`diffusion.training.save_checkpoint`) y el guardado final escribe además el **hermano de
crudos** ``…_raw.pt`` con los pesos del último paso de Adam, para poder comparar crudo vs EMA
dentro de la misma corrida.

Con ``data.val_root`` (solo ``kind: images``) la corrida mide además la **pérdida de validación**
por examen fijo cada ``train.val_every`` pasos —o, si esa clave no se declara, cada
``train.checkpoint_every``— y en el paso final. ``val_every`` es la palanca para elegir la
resolución de la curva **sin** escribir más snapshots (subirla vía ``checkpoint_every`` cuesta un
archivo de pesos por punto), y sirve incluso con los snapshots apagados. El CLI reenvía al loop las
dos fuentes que armó el config layer (el examen de validación y el examen fijo de entrenamiento que
lo hace legible), escribe un registro propio por evaluación en el ``.jsonl`` —``event: "val"``, con
los tres valores y el ``device``—, suma las tres curvas al gráfico de pérdida (con leyenda, ver
:func:`save_loss_curve`) e informa el último punto medido al cerrar. Sin la clave no se mide nada y
la salida —log, gráfico y consola— es exactamente la de antes de la feature.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time

# Permitir ejecutar el script sin instalar el paquete (agrega ./src al path).
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tqdm import tqdm

from diffusion.training import (
    build_run,
    load_config,
    load_resume,
    prune_snapshots,
    resolve_resume,
    resume_sidecar_path,
    save_checkpoint,
    save_resume_state,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrena la red de score (DSM) a partir de un config YAML."
    )
    p.add_argument("--config", required=True, help="Ruta del .yaml de la corrida.")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Override de la cantidad de pasos de entrenamiento del config.")
    p.add_argument("--device", type=str, default=None,
                   help="Override del dispositivo (p. ej. cpu / cuda).")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="Override de cada cuántos pasos guardar un snapshot intermedio "
                        "(0 = solo el checkpoint final; requiere 'out.checkpoint').")
    p.add_argument("--keep-last", type=int, default=None, metavar="N",
                   help="Override de la retención: conservar solo los N snapshots intermedios más "
                        "nuevos (con su sidecar), borrando los más viejos a medida que se generan. "
                        "El checkpoint final nunca se borra. Sin este flag ni 'train.keep_last_"
                        "checkpoints', se conservan todos.")
    p.add_argument("--force", action="store_true",
                   help="Reentrenar aunque el checkpoint final ya exista (saltea el skip). "
                        "Si hay snapshots intermedios, reanuda desde el más nuevo.")
    p.add_argument("--resume-from", type=str, default=None, metavar="PATH_O_STEP",
                   help="Reanudar desde un checkpoint puntual: la ruta de un snapshot "
                        "'…_stepNNNNN.pt' o su número de paso (en vez del más nuevo automático).")
    p.add_argument("--quiet", action="store_true",
                   help="No imprimir el progreso por paso.")
    return p


def format_val_values(raw: float, ema: float | None, train: float | None) -> str:
    """Formatea los valores de **una** evaluación de validación para consola.

    Un solo formateador para los dos mensajes que los muestran (el aviso por evaluación y el
    resumen final), así los dos leen igual y no se desincronizan. Los valores ausentes se
    **omiten** en lugar de imprimirse como ``None``: en la consola la ausencia se lee sola (sin
    EMA no hay curva de EMA), a diferencia del ``.jsonl``, donde la clave viaja igual con ``null``
    porque su consumidor necesita distinguir "no había EMA" de "el campo no existe".

    Args:
        raw: Pérdida de validación con los pesos vivos.
        ema: Ídem con la sombra EMA, o ``None`` si la corrida no mantiene sombra.
        train: Examen fijo de entrenamiento (la referencia del gap), o ``None``.

    Returns:
        Una línea del tipo ``"val=0.123456  val(EMA)=0.120000  train fijo=0.100000"``.
    """
    partes = [f"val={raw:.6f}"]
    if ema is not None:
        partes.append(f"val(EMA)={ema:.6f}")
    if train is not None:
        partes.append(f"train fijo={train:.6f}")
    return "  ".join(partes)


#: Etiqueta de la curva **densa** de entrenamiento (una pérdida por paso, con ``t`` re-sorteado en
#: cada uno). Nombrarla "per-step" es load-bearing: la corrida puede dibujar además el examen fijo
#: de entrenamiento, que mide el MISMO conjunto con ``t`` y ruido congelados. Son dos estimadores
#: distintos y confundirlos invierte la lectura del gap — el par comparable es el examen fijo de
#: train contra la validación con pesos vivos, nunca la curva densa contra la de validación.
_ETIQUETA_TRAIN_DENSA = "train per-step"

#: Las tres series **dispersas** de la validación: (clave del ``ValPoint``, etiqueta de la leyenda).
#: El orden es el de dibujo y el de la leyenda: las dos de validación juntas (así la comparación
#: raw↔EMA se lee de un vistazo) y el examen fijo de train al final, que es la referencia contra la
#: cual se mide el gap de generalización.
_SERIES_VAL: tuple[tuple[str, str], ...] = (
    ("raw", "val (pesos vivos)"),
    ("ema", "val (EMA)"),
    ("train", "train (examen fijo)"),
)


def save_loss_curve(
    path: str | pathlib.Path,
    history: list[float],
    title: str,
    *,
    val_history: list[dict] | None = None,
) -> None:
    """Dibuja la curva de pérdida de la corrida y la guarda como PNG.

    Args:
        path: Ruta del ``.png`` (se crean los directorios padre que falten).
        history: Serie **densa** de la pérdida de entrenamiento, un valor por paso.
        title: Título de la figura (típicamente ``"<sde> · <modelo>"``).
        val_history: Serie **dispersa** de validación (los ``ValPoint`` de la corrida, indexados por
            paso), o ``None``/vacía si la corrida no midió validación. Con la serie presente se
            dibujan hasta tres curvas más —validación con pesos vivos, validación con la sombra EMA
            y el examen fijo de entrenamiento— con línea **y** marcadores, porque la serie es
            dispersa y con una cadencia grande puede tener dos o tres puntos: sin marcadores, una
            línea de dos puntos se lee como una tendencia continua. Una curva cuyos valores son
            todos ``None`` (sin EMA, o sin fuente de examen de train) simplemente no se dibuja.
            Sin la serie el gráfico sale exactamente como antes de la feature ``validation-loss``:
            una sola curva y sin leyenda.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    # La densa va indexada por POSICIÓN (paso 1..N); las dispersas, por el paso que trae cada punto.
    # Mezclar las dos convenciones apilaría los puntos de validación contra el origen del eje.
    ax.plot(
        range(1, len(history) + 1), history,
        linewidth=0.7, alpha=0.8, label=_ETIQUETA_TRAIN_DENSA,
    )
    graficadas = [history]
    if val_history:
        for clave, etiqueta in _SERIES_VAL:
            presentes = [
                (p["step"], p[clave]) for p in val_history if p.get(clave) is not None
            ]
            if not presentes:
                continue  # serie ausente (p. ej. sin sombra EMA): no se dibuja
            pasos, valores = zip(*presentes)
            ax.plot(pasos, valores, marker="o", markersize=3.5, linewidth=1.2, label=etiqueta)
            graficadas.append(list(valores))
        # La leyenda aparece solo si se dibujó algo más que la curva densa: es lo que hace
        # distinguible la densa del examen fijo de train (y lo que deja el gráfico sin validación
        # igual que antes, sin leyenda).
        if len(graficadas) > 1:
            ax.legend(fontsize="small")
    # La escala log tiene que mirar TODAS las series dibujadas: un valor <= 0 en cualquiera de ellas
    # quedaría fuera de un eje logarítmico, sin ningún aviso.
    if all(min(serie) > 0 for serie in graficadas):
        ax.set_yscale("log")
    ax.set_xlabel("paso")
    ax.set_ylabel("pérdida DSM")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    # En Windows (py<3.15) la consola no usa UTF-8 por defecto; forzarlo evita
    # mojibake en los acentos de los mensajes.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    try:
        spec = build_run(load_config(args.config))
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Overrides de la línea de comandos.
    if args.num_steps is not None:
        spec.config.num_steps = args.num_steps
    if args.device is not None:
        spec.config.device = args.device
    if args.checkpoint_every is not None:
        spec.config.checkpoint_every = args.checkpoint_every
    if args.keep_last is not None:
        spec.config.keep_last_checkpoints = args.keep_last
    if args.quiet:
        spec.config.log_every = 0
    elif spec.config.log_every == 0:
        spec.config.log_every = max(1, spec.config.num_steps // 10)

    # Validación fail-fast del knob de retención (train() no lo usa; lo aplica el callback de abajo).
    keep_last = spec.config.keep_last_checkpoints
    if keep_last is not None and keep_last < 1:
        print(
            f"error: keep_last_checkpoints debe ser >= 1 (recibido {keep_last}): conservar cero "
            "snapshots dejaría la corrida sin punto de reanudación (usá un N>=1 o quitá el knob).",
            file=sys.stderr,
        )
        return 2

    # --- Resolución de resume (skip / resume / fresh) a partir del .yaml y los flags ---
    # resolve_resume solo mira el filesystem (no entrena ni escribe); --resume-from inexistente
    # levanta ValueError que se mapea a exit 2 (patrón del script).
    try:
        plan = resolve_resume(
            spec.checkpoint, force=args.force, resume_from=args.resume_from
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if plan.action == "skip":
        # El checkpoint final ya existe: corrida completa. No se sobrescribe nada (3.1).
        print(
            f"Corrida ya completa: el checkpoint final '{spec.checkpoint}' ya existe; "
            "nada que entrenar (usá --force para reentrenar)."
        )
        return 0

    # --- Callback de checkpointing intermedio + advertencias sobre puntos de reanudación ---
    # El callback deriva la ruta hermana (…_stepNNNNN.pt, el único tag que emite el loop) del
    # checkpoint final y persiste AMBOS artefactos: los pesos (save_checkpoint) y el sidecar de resume
    # (save_resume_state), así una interrupción deja un punto reanudable (1.1). train() sigue sin
    # tocar el filesystem: decide *cuándo*; esto decide *dónde/cómo*.
    on_checkpoint = None
    if spec.config.checkpoint_every > 0:
        if spec.checkpoint is not None:
            base = spec.checkpoint

            def on_checkpoint(tag, snapshot):
                tagged = base.with_stem(f"{base.stem}_{tag}")
                save_checkpoint(snapshot.result, tagged, model_spec=spec.model_spec)
                save_resume_state(resume_sidecar_path(tagged), snapshot.resume)
                # tqdm.write en vez de print: escribe por encima de la barra de progreso sin
                # romperla (y se comporta como print cuando la barra no está activa).
                tqdm.write(f"Checkpoint ({tag}) -> {tagged}  (+ sidecar de resume)")
                # Retención rolling: tras guardar el snapshot nuevo, borra los más viejos y deja los
                # keep_last más nuevos (cada uno con su sidecar). El checkpoint final nunca se toca.
                if keep_last is not None:
                    borrados = prune_snapshots(base, keep_last)
                    if borrados:
                        tqdm.write(
                            f"  retención: {keep_last} snapshots conservados; "
                            f"{len(borrados)} archivos viejos borrados"
                        )
        else:
            print(
                "nota: 'train.checkpoint_every' > 0 pero falta 'out.checkpoint'; "
                "no se guardarán snapshots intermedios."
            )
    else:
        # Sin snapshots periódicos no hay puntos de reanudación: la resumabilidad requiere
        # checkpoint_every>0 (1.5).
        print(
            "advertencia: 'train.checkpoint_every' = 0: esta corrida no dejará puntos de "
            "reanudación (sin snapshots intermedios ni sidecars de resume). Para poder reanudar "
            "una corrida larga interrumpida, seteá 'train.checkpoint_every' > 0 con 'out.checkpoint'."
        )

    # --- Aplicar el plan: reanudar (cargar pesos + estado) o empezar de cero; reportar (3.8) ---
    resume = None
    if plan.action == "resume":
        expected = {
            "sde_name": spec.sde.name,
            "model_spec": spec.model_spec,
            "data_dim": spec.sde.data_dim,
        }
        try:
            state_dict, _meta, resume = load_resume(plan.weights_path, expected=expected)
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        spec.model.load_state_dict(state_dict)
        print(
            f"Acción: reanudando desde {plan.weights_path} (paso {resume.start_step}) "
            f"hasta num_steps={spec.config.num_steps}."
        )
    else:  # fresh
        print("Acción: entrenando desde cero.")

    print(
        f"Entrenando sde={spec.sde.name} (data_dim={spec.sde.data_dim}) "
        f"con {type(spec.model).__name__}: pasos={spec.config.num_steps} "
        f"device={spec.config.device}"
    )
    # --- Log de entrenamiento (.jsonl) opcional: start + estados periódicos + end, con timestamp ---
    # train() no toca el filesystem ni el reloj: emite estados por on_log y acá se les pone el
    # timestamp y se escriben (modo append: un resume continúa el mismo log).
    #
    # El mismo callback atiende las DOS variantes de registro que emite el loop (ver el contrato de
    # eventos de train): los estados de paso y —con validación configurada— el punto medido en cada
    # evaluación. La validación es opt-in por 'data.val_root': el config layer entrega las dos
    # fuentes de examen juntas o ninguna, así que una sola bandera describe la corrida.
    tiene_val = spec.val_data is not None
    log_fh = None
    t0 = time.time()

    def _log(rec):
        rec = {"t": datetime.datetime.now().isoformat(timespec="seconds"), **rec}
        log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_fh.flush()  # cada línea persiste ya: un corte deja el log parcial usable

    if spec.train_log is not None:
        spec.train_log.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(spec.train_log, "a", encoding="utf-8")
        _log({
            "event": "start",
            "sde": spec.sde.name,
            "device": spec.config.device,
            "num_steps": spec.config.num_steps,
            "amp": spec.config.amp,
            "model": type(spec.model).__name__,
            "data_dim": spec.sde.data_dim,
            "resume_from_step": resume.start_step if resume is not None else 0,
            # Si la corrida mide validación: lo que le permite a un lector posterior del archivo
            # saber si la ausencia de registros 'val' significa "no se midió" o "se cortó antes".
            "val": tiene_val,
        })

    def _on_log(rec):
        """Sumidero de los estados del loop: consola (validación) + ``.jsonl`` (todo)."""
        if rec.get("event") == "val":
            # La evaluación cae DENTRO del paso, con la barra de progreso activa: el aviso va por
            # tqdm.write (misma convención que los avisos de checkpoint), que escribe por encima de
            # la barra sin partirla — un print la duplicaría.
            tqdm.write(
                f"Validación (paso {rec['step']}): "
                + format_val_values(rec["val_raw"], rec["val_ema"], rec["train_fijo"])
            )
        if log_fh is not None:
            # INVARIANTE: ``**rec`` va ÚLTIMO. De eso depende que el registro de validación —que
            # trae su propio 'event'— sobreescriba el "step" genérico y quede distinguible en el
            # archivo; invertir el orden escribiría todos los puntos de validación como si fueran
            # pasos de entrenamiento. No reordenar sin revisar el contrato de eventos del loop.
            _log({"event": "step", "elapsed_s": round(time.time() - t0, 3), **rec})

    # Sin log ni validación no hay nada que hacer con los estados: el callback queda en None y el
    # loop no paga ni la llamada (comportamiento previo a la feature, sin ramas nuevas).
    on_log = _on_log if (log_fh is not None or tiene_val) else None

    # Barra de progreso (%, ETA, it/s) salvo --quiet, que apaga tanto la barra como el print.
    result = train(
        spec.sde, spec.model, spec.data, spec.config,
        on_checkpoint=on_checkpoint, on_log=on_log, resume=resume, progress=not args.quiet,
        # Las dos fuentes de examen que armó el config layer (None sin 'data.val_root'): la de
        # validación y la del examen fijo de entrenamiento que hace legible su curva.
        val_batches=spec.val_data, train_exam_batches=spec.train_exam_data,
    )
    hist = result.history
    k = max(1, len(hist) // 20)  # media de extremos: la pérdida per-step es ruidosa
    ini, fin = sum(hist[:k]) / k, sum(hist[-k:]) / k
    print(
        f"Listo. pérdida inicial≈{ini:.6f} -> final≈{fin:.6f}  "
        f"(medias de {k} pasos; {len(hist)} pasos guardados)"
    )
    if result.val_history:
        # Último punto de la serie dispersa: la foto de validación más reciente de la corrida (el
        # disparador incluye el paso final, así que es el del último paso). La serie completa vive en
        # el .jsonl y en el meta del checkpoint; acá se informa solo el cierre.
        ultimo = result.val_history[-1]
        print(
            f"Validación -> paso {ultimo['step']}: "
            + format_val_values(ultimo["raw"], ultimo["ema"], ultimo["train"])
        )

    if log_fh is not None:
        _log({
            "event": "end",
            "step": len(hist),
            "loss_final": round(fin, 6),
            "elapsed_s": round(time.time() - t0, 3),
        })
        log_fh.close()
        print(f"Log        -> {spec.train_log}")

    if spec.checkpoint:
        # raw_sibling=True: guardado FINAL, así que si la corrida tiene EMA activo se escribe
        # además el hermano de crudos ``…_raw.pt`` (para la comparativa crudo-vs-EMA de la misma
        # corrida). Sin EMA no escribe nada extra: el principal ya publica los crudos.
        save_checkpoint(
            result, spec.checkpoint, model_spec=spec.model_spec, raw_sibling=True
        )
        print(f"Checkpoint -> {spec.checkpoint}")
        if result.ema_state is not None:
            raw_path = spec.checkpoint.with_stem(f"{spec.checkpoint.stem}_raw")
            print(f"Crudos     -> {raw_path}  (contraparte cruda del checkpoint EMA)")
    if spec.loss_curve:
        # La serie de validación viaja como argumento opcional: vacía (corrida sin 'data.val_root')
        # el gráfico sale exactamente como antes de la feature.
        save_loss_curve(
            spec.loss_curve, result.history, f"{spec.sde.name} · {type(spec.model).__name__}",
            val_history=result.val_history,
        )
        print(f"Curva      -> {spec.loss_curve}")
    if not spec.checkpoint and not spec.loss_curve:
        print("(sin 'out.checkpoint' ni 'out.loss_curve': no se guardó nada)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
