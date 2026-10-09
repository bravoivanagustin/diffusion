"""Resolución de reanudación (*resume*) de una corrida de entrenamiento.

Helper **puro y testeable** (sin torch, sin entrenar): decide —a partir de la ruta del
checkpoint final del ``.yaml``— si una corrida ya está **completa** (``skip``), si debe
**reanudar** desde un snapshot intermedio (``resume``) o si empieza **de cero** (``fresh``), y
descubre los snapshots intermedios que dejó una corrida previa.

Convención de nombres (la del CLI ``scripts/train.py``): el checkpoint final es ``X.pt`` y sus
snapshots hermanos son ``X_stepNNNNN.pt`` (periódicos); cada snapshot periódico lleva además un
*sidecar* de resume ``X_stepNNNNN.resume.pt`` (el estado del optimizador + paso + azar; ver
:mod:`diffusion.training.trainer`). Puede haber también ``X_best.pt`` **legados** en disco: el
loop dejó de emitir ese tag el 27/07/2026 (R2.6 de ``ema-weights``), pero los archivos que
quedaron de corridas previas se siguen tolerando (se excluyen del descubrimiento).

La **política de decisión** (``discover_snapshots`` / ``resolve_resume``) solo mira el filesystem por
**existencia/glob** (no lo muta) y no importa torch. La **carga y validación** del punto elegido
(``load_resume`` / ``validate_compatible``) completan el resolver: ``load_resume`` arma el estado de
resume desde el checkpoint de pesos + su *sidecar* (usa el ``trainer`` vía **import diferido**, para no
arrastrar torch al importar el módulo), y ``validate_compatible`` es una comparación **pura** de
metadata (SDE, ``data_dim`` y receta de red) que no toca torch.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # solo para anotaciones; en runtime load_resume hace import diferido del trainer
    import torch

    from .trainer import ResumeState

# Sufijo de un snapshot periódico: ``…_stepNNNNN.pt``. El grupo captura el entero del paso. Se
# ancla al final (``$``) para NO matchear los sidecars ``…_stepNNNNN.resume.pt`` (tienen
# ``.resume.pt`` tras los dígitos, no ``.pt``). Cambiar esta convención → revalidar
# ``discover_snapshots`` (ver Revalidation Triggers del diseño).
_STEP_RE = re.compile(r"_step(\d+)\.pt$")

# Sufijo del sidecar de resume hermano de un checkpoint de pesos ``X.pt`` → ``X.resume.pt``.
_RESUME_SUFFIX = ".resume.pt"


@dataclass
class ResumePlan:
    """Decisión de reanudación (DTO puro, sin persistencia).

    Attributes:
        action: ``"skip"`` (corrida ya completa), ``"fresh"`` (desde cero) o ``"resume"``
            (continuar desde ``weights_path``).
        weights_path: Checkpoint de pesos desde el que reanudar (solo en ``"resume"``).
        step: Paso ya completado del checkpoint elegido (solo en ``"resume"``; ``None`` si no se
            pudo parsear del nombre).
    """

    action: str
    weights_path: pathlib.Path | None = None
    step: int | None = None


def resume_sidecar_path(weights_path: pathlib.Path) -> pathlib.Path:
    """Devuelve la ruta del *sidecar* de resume hermano de un checkpoint de pesos.

    Convención: ``X_stepNNNNN.pt`` → ``X_stepNNNNN.resume.pt`` (mismo directorio y stem, con el
    sufijo ``.pt`` reemplazado por ``.resume.pt``).

    Args:
        weights_path: Ruta del checkpoint de pesos (``…_stepNNNNN.pt``).

    Returns:
        La ruta del sidecar ``…_stepNNNNN.resume.pt``.
    """
    return pathlib.Path(weights_path).with_suffix(_RESUME_SUFFIX)


def discover_snapshots(
    final_checkpoint: pathlib.Path,
) -> list[tuple[int, pathlib.Path]]:
    """Descubre los snapshots periódicos hermanos de un checkpoint final, ordenados por paso.

    Busca en el directorio del checkpoint final los archivos ``{stem}_stepNNNNN.pt`` (donde
    ``stem`` es el nombre del final sin extensión) y parsea el entero del paso. **Excluye** el
    checkpoint final mismo, los sidecars ``{stem}_stepNNNNN.resume.pt`` y cualquier
    ``{stem}_best.pt`` **legado** (tolerancia a artefactos de corridas previas: el loop ya no emite
    ese tag, R2.6); y solo considera snapshots del **mismo** ``stem`` (no los de otras corridas del
    directorio).

    Args:
        final_checkpoint: Ruta del checkpoint final ``X.pt`` de la corrida (puede no existir; se
            usa solo para derivar ``stem`` y el directorio donde buscar).

    Returns:
        Lista de ``(step, path)`` ordenada **ascendente** por ``step``. Vacía si el directorio no
        existe o no hay snapshots.
    """
    final = pathlib.Path(final_checkpoint)
    parent = final.parent
    if not parent.is_dir():
        return []  # no hay dónde buscar

    # Ancla al stem del final: ``{stem}_step(\d+)\.pt`` (fullmatch) → no cuela snapshots de otras
    # corridas del mismo directorio, ni el final, ni el sidecar, ni un ``_best.pt`` legado.
    pattern = re.compile(re.escape(final.stem) + r"_step(\d+)\.pt$")
    snaps: list[tuple[int, pathlib.Path]] = []
    for entry in parent.iterdir():
        m = pattern.fullmatch(entry.name)
        if m is not None:
            snaps.append((int(m.group(1)), entry))
    snaps.sort(key=lambda pair: pair[0])
    return snaps


def prune_snapshots(
    final_checkpoint: pathlib.Path, keep_last: int
) -> list[pathlib.Path]:
    """Borra los snapshots intermedios más viejos, conservando los ``keep_last`` más nuevos.

    Política de **retención rolling** del checkpointing intermedio: descubre los snapshots
    ``X_stepNNNNN.pt`` hermanos del checkpoint final (vía :func:`discover_snapshots`, que ya excluye
    el final, su hermano ``_raw`` y los ``_best`` legados), conserva los ``keep_last`` de **mayor
    paso** y borra el resto **junto con su sidecar** de resume (``X_stepNNNNN.resume.pt``: los dos
    van siempre juntos). Nunca toca el checkpoint final ni su ``_raw``, y —con ``keep_last >= 1``—
    siempre conserva al menos el snapshot más nuevo, así que el auto-resume sigue teniendo desde
    dónde reanudar.

    Es **puro filesystem** (no importa torch): solo lista y borra. El borrado es idempotente —un
    sidecar ausente (corrida sin resume) no es un error, simplemente no se borra.

    Args:
        final_checkpoint: Ruta del checkpoint final ``X.pt`` (se usa solo para derivar ``stem`` y el
            directorio, igual que :func:`discover_snapshots`; puede no existir todavía).
        keep_last: Cantidad de snapshots intermedios más nuevos a conservar. Debe ser ``>= 1``.

    Returns:
        Lista de rutas efectivamente borradas (los ``.pt`` y sus ``.resume.pt`` que existían), de
        más viejo a más nuevo. Vacía si no había nada que borrar (había ``<= keep_last`` snapshots).

    Raises:
        ValueError: Si ``keep_last < 1`` (conservar cero snapshots dejaría la corrida sin punto de
            reanudación).
    """
    if keep_last < 1:
        raise ValueError(
            f"keep_last debe ser >= 1 (recibido {keep_last}): conservar cero snapshots dejaría "
            "la corrida sin punto de reanudación."
        )
    snaps = discover_snapshots(final_checkpoint)  # (step, path) ascendente por paso
    if len(snaps) <= keep_last:
        return []  # nada que borrar: hay a lo sumo keep_last snapshots
    # La cola (``snaps[-keep_last:]``) son los más nuevos que se conservan; el resto se borra.
    borrados: list[pathlib.Path] = []
    for _step, weights_path in snaps[:-keep_last]:
        for archivo in (weights_path, resume_sidecar_path(weights_path)):
            if archivo.exists():
                archivo.unlink()
                borrados.append(archivo)
    return borrados


def resolve_resume(
    final_checkpoint: pathlib.Path | None,
    *,
    force: bool = False,
    resume_from: str | None = None,
) -> ResumePlan:
    """Decide la acción de resume (``skip`` / ``fresh`` / ``resume``) a partir del config.

    Sigue la resolución del CLI:

    1. Si se pasa ``resume_from`` (ruta o número de paso) se resuelve a un snapshot puntual y se
       reanuda desde ahí (el pedido explícito manda sobre el skip automático) (3.5); si no se
       puede resolver → ``ValueError`` que lista los snapshots disponibles (3.7).
    2. Si no, y el ``final_checkpoint`` existe y no hay ``force`` → ``skip`` (corrida completa)
       (3.1).
    3. Si no (final ausente **o** ``force``): se descubren los snapshots; si hay alguno → ``resume``
       desde el más nuevo (mayor paso) (3.2 con ``force`` / 3.3); si no hay ninguno → ``fresh``
       (3.4).

    ``final_checkpoint=None`` (el ``.yaml`` no define ``out.checkpoint``) no tiene dónde saltear ni
    buscar → ``fresh``.

    Este helper **no muta** el filesystem: solo consulta existencia y lista snapshots.

    Args:
        final_checkpoint: Ruta del checkpoint final del ``.yaml`` (``spec.checkpoint``) o ``None``.
        force: Si es ``True`` se saltea el chequeo del final (se reentrena/reanuda igual) (3.2).
        resume_from: Selector explícito del checkpoint a reanudar: una **ruta** a un snapshot
            existente o un **número de paso** que matchee un snapshot descubierto (3.5).

    Returns:
        El :class:`ResumePlan` con la acción resuelta.

    Raises:
        ValueError: Si ``resume_from`` no resuelve a ningún snapshot (ruta o paso inexistente); el
            mensaje lista los snapshots disponibles (3.7).
    """
    final = pathlib.Path(final_checkpoint) if final_checkpoint is not None else None

    # 1) Selección explícita: manda sobre el skip automático.
    if resume_from is not None:
        weights_path, step = _resolve_resume_from(final, resume_from)
        return ResumePlan("resume", weights_path=weights_path, step=step)

    # 2) Final ya presente (sin force) → corrida completa.
    if final is not None and final.exists() and not force:
        return ResumePlan("skip")

    # 3) Final ausente o force: reanudar desde el más nuevo si hay snapshots; si no, desde cero.
    snaps = discover_snapshots(final) if final is not None else []
    if snaps:
        step, weights_path = snaps[-1]  # más nuevo = mayor paso (lista ASC)
        return ResumePlan("resume", weights_path=weights_path, step=step)
    return ResumePlan("fresh")


def _resolve_resume_from(
    final: pathlib.Path | None, resume_from: str
) -> tuple[pathlib.Path, int | None]:
    """Resuelve ``resume_from`` (ruta o paso) a ``(weights_path, step)``.

    Se intenta primero como **ruta** existente; si no, como **número de paso** contra los
    snapshots descubiertos. Si ninguno resuelve → ``ValueError`` listando los disponibles (3.7).
    """
    snaps = discover_snapshots(final) if final is not None else []

    # (a) ¿es una ruta a un snapshot existente?
    candidate = pathlib.Path(resume_from)
    if candidate.exists():
        return candidate, _parse_step(candidate.name)

    # (b) ¿es un número de paso que matchea un snapshot descubierto?
    if resume_from.isdigit():
        wanted = int(resume_from)
        for step, path in snaps:
            if step == wanted:
                return path, step

    # (c) no resuelve → error accionable que lista los disponibles.
    raise ValueError(_unresolved_message(resume_from, snaps))


def _parse_step(name: str) -> int | None:
    """Extrae el entero del paso de un nombre ``…_stepNNNNN.pt`` (``None`` si no matchea)."""
    m = _STEP_RE.search(name)
    return int(m.group(1)) if m is not None else None


def _unresolved_message(
    resume_from: str, snaps: list[tuple[int, pathlib.Path]]
) -> str:
    """Arma el mensaje de error de ``--resume-from`` inexistente, listando los disponibles (3.7)."""
    if snaps:
        steps = ", ".join(str(step) for step, _ in snaps)
        disponibles = f"Snapshots disponibles (por paso): [{steps}]."
    else:
        disponibles = "No hay snapshots disponibles para reanudar."
    return (
        f"No se pudo resolver --resume-from={resume_from!r} "
        f"(no es una ruta existente ni un paso descubierto). {disponibles}"
    )


# ------------------------------------------- carga y validación del punto (C3)


def validate_compatible(
    meta: dict,
    *,
    sde_name: str,
    model_spec: dict | None,
    data_dim,
) -> None:
    """Valida que el ``meta`` de un checkpoint sea **compatible** con la corrida actual (2.5).

    Reanudar solo tiene sentido si el checkpoint elegido corresponde a la MISMA corrida: la misma
    SDE (Eje 1), la misma forma del dato (``data_dim``) y la misma receta de red. Cualquier
    diferencia significaría cargar el optimizador/azar sobre un estado que no corresponde, así que
    se falla antes de reanudar. La comparación es por **igualdad exacta** (no hay tolerancia):

    - ``meta["sde_name"] == sde_name`` — misma SDE.
    - ``meta["data_dim"] == data_dim`` — misma dimensión/forma de evento (un ``int`` para el toy 2D
      o una tupla para imágenes; se comparan por igualdad, así que un ``int`` y una tupla nunca
      matchean).
    - ``meta.get("model") == model_spec`` — misma receta de red ``{name, kwargs}``. Ambos pueden ser
      ``None`` (checkpoint sin receta + corrida sin receta = match); si uno está y el otro no →
      mismatch.

    Args:
        meta: Metadata del checkpoint de pesos (de :func:`~diffusion.training.load_checkpoint`).
        sde_name: Nombre de la SDE de la corrida actual.
        model_spec: Receta de red ``{name, kwargs}`` de la corrida actual (o ``None`` si el
            checkpoint no llevaba receta).
        data_dim: ``data_dim`` de la corrida actual (``int`` o tupla; se compara por igualdad).

    Returns:
        ``None`` si TODO coincide (no levanta).

    Raises:
        ValueError: Si difiere la SDE, el ``data_dim`` o la receta de red; el mensaje detalla qué
            no coincide (checkpoint vs corrida) (2.5).
    """
    mismatches: list[str] = []
    if meta.get("sde_name") != sde_name:
        mismatches.append(
            f"SDE (checkpoint={meta.get('sde_name')!r} vs corrida={sde_name!r})"
        )
    if meta.get("data_dim") != data_dim:
        mismatches.append(
            f"data_dim (checkpoint={meta.get('data_dim')!r} vs corrida={data_dim!r})"
        )
    if meta.get("model") != model_spec:
        mismatches.append(
            f"receta de red (checkpoint={meta.get('model')!r} vs corrida={model_spec!r})"
        )
    if mismatches:
        raise ValueError(
            "El checkpoint elegido es incompatible con la corrida actual y no se puede reanudar "
            "desde un estado que no corresponde. Difiere: " + "; ".join(mismatches) + "."
        )


def load_resume(
    weights_path: pathlib.Path,
    *,
    expected: dict,
    map_location: torch.device | str = "cpu",
) -> tuple[dict, dict, ResumeState]:
    """Carga un punto de reanudación (pesos + sidecar) y arma el :class:`ResumeState` a reanudar.

    Reúne los dos artefactos de un snapshot —el checkpoint de **pesos** (``X_stepNNNNN.pt``) y su
    **sidecar** de resume (``X_stepNNNNN.resume.pt``)— en un estado listo para
    ``model.load_state_dict(state_dict)`` + ``train(resume=...)``. El ``history`` se toma del ``meta``
    del checkpoint de PESOS (el sidecar no lo persiste, 1.3).

    Flujo (falla temprano):

    1. Carga ``(state_dict, meta)`` del checkpoint de pesos.
    2. Valida compatibilidad **antes** de tocar el sidecar (SDE / ``data_dim`` / receta de red vía
       :func:`validate_compatible`, 2.5) — un config incompatible falla acá.
    3. Exige el sidecar hermano (:func:`resume_sidecar_path`); si falta → error claro que lo nombra
       (3.6). Sin él no hay optimizador/paso/azar para reanudar.
    4. Carga el sidecar y arma el :class:`ResumeState` (optimizador + paso + azar del sidecar;
       ``history`` del ``meta``, 1.3; la sombra EMA del sidecar si está, R3.1; el estado del
       escalador AMP del sidecar si está, R2.1; la **serie de validación** del ``meta`` si está, 6.3).

    **Corridas con EMA**: el checkpoint de pesos publica la **sombra** (ver
    :func:`~diffusion.training.save_checkpoint`), así que los pesos con los que hay que *continuar
    optimizando* no son los suyos. Si el sidecar trae ``raw_model_state`` (los crudos del momento),
    es **eso** lo que se devuelve como ``state_dict`` a cargar en la red — el checkpoint queda solo
    como fuente de ``meta``/``history``— y la sombra viaja en ``ResumeState.ema_state`` para que el
    loop la restaure (R3.1, R3.2). Si no lo trae (sidecar anterior a la feature = corrida sin EMA,
    cuyo checkpoint ya publica los crudos) el comportamiento es **exactamente el de antes** (R3.3).

    Args:
        weights_path: Ruta del checkpoint de pesos elegido (``…_stepNNNNN.pt``).
        expected: Metadata esperada de la corrida actual para validar compatibilidad: un ``dict`` con
            ``{"sde_name", "model_spec", "data_dim"}`` (se pasa como ``**expected`` a
            :func:`validate_compatible`).
        map_location: Dispositivo donde cargar los tensores de pesos y del sidecar (default
            ``"cpu"``).

    Returns:
        ``(state_dict, meta, resume)``: el ``state_dict`` de la red (a cargar en el modelo: los
        **crudos** del sidecar si están, si no los del checkpoint), el ``meta`` del checkpoint de
        pesos y el :class:`ResumeState` listo para ``train(resume=...)``. Procedencia de cada campo
        del :class:`ResumeState`: del **sidecar** el optimizador, el paso, los dos estados de azar y
        —si las trae— la sombra EMA y el estado del escalador AMP; del ``meta`` del checkpoint de
        **pesos** el ``history`` (1.3) y la ``val_history`` (6.3), las dos series de la corrida.

    Raises:
        ValueError: Si el checkpoint es incompatible con la corrida (SDE / ``data_dim`` / receta;
            2.5).
        FileNotFoundError: Si falta el sidecar de resume del checkpoint elegido; el mensaje nombra el
            artefacto faltante (3.6).
    """
    # Import diferido: mantiene el módulo (discover_snapshots/resolve_resume) libre de torch al
    # importar; la carga sí necesita el trainer (dirección de dependencia permitida: resume→trainer).
    from .trainer import ResumeState, load_checkpoint, load_resume_state

    weights_path = pathlib.Path(weights_path)

    # 1) pesos + meta del checkpoint elegido.
    state_dict, meta = load_checkpoint(weights_path, map_location=map_location)

    # 2) compatibilidad EXACTA contra el meta, ANTES de exigir el sidecar (2.5).
    validate_compatible(meta, **expected)

    # 3) el sidecar es OBLIGATORIO: sin él no se puede reanudar (3.6).
    sidecar = resume_sidecar_path(weights_path)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Falta el sidecar de resume esperado: {sidecar}. Sin el estado del "
            f"optimizador/paso/azar no se puede reanudar desde {weights_path.name} "
            "(¿la corrida original corrió con checkpoint_every>0?)."
        )

    # 4) estado del sidecar (optimizador + paso + azar; sin history) → ResumeState.
    sc = load_resume_state(sidecar, map_location=map_location)

    # Corrida con EMA: los pesos a CARGAR son los crudos del sidecar, no los del checkpoint (que
    # publica la sombra) — si no, el optimizador continuaría desde un promedio que ninguna
    # trayectoria de Adam visitó (R3.1/R3.2). Se pide con ``get``: un sidecar viejo no tiene la
    # clave y entonces el ``state_dict`` del checkpoint queda como siempre (R3.3).
    crudos = sc.get("raw_model_state")
    if crudos is not None:
        state_dict = crudos

    resume = ResumeState(
        optimizer_state=sc["optimizer_state"],
        start_step=sc["step"],
        torch_rng_state=sc["torch_rng_state"],
        generator_state=sc["generator_state"],
        history=meta["history"],  # history del checkpoint de PESOS, no del sidecar (1.3)
        # La sombra la restaura ``train`` con ``EmaShadow.load_state``; los crudos NO se duplican
        # acá: ya se devuelven como ``state_dict`` para que el caller los cargue en la red.
        ema_state=sc.get("ema_state"),
        # El escalador AMP lo restaura ``train`` con ``GradScaler.load_state_dict``. Se pide con
        # ``get``: un sidecar sin AMP (o anterior a la feature) no tiene la clave → ``None`` (R2.4).
        scaler_state=sc.get("scaler_state"),
        # Serie de validación del ``meta`` de los PESOS, no del sidecar (misma ruta que el
        # ``history``, 6.3): ``save_checkpoint`` la escribe ahí y solo si no está vacía. La lectura es
        # **tolerante** a propósito (``get``, no ``[...]``): un checkpoint de una corrida sin
        # validación —o anterior a esta feature— no tiene la clave y se reanuda exactamente como
        # siempre, con la serie en ``None`` y el loop arrancándola vacía (5.5). No hay guard cruzado
        # contra la config, a diferencia de ``ema_state``/``scaler_state``: la serie es una
        # observación, no estado necesario para continuar la optimización.
        val_history=meta.get("val_history"),
    )
    return state_dict, meta, resume
