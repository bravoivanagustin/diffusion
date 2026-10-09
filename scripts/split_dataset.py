"""Herramienta de split determinístico de un dataset de imágenes en ``train/`` y ``val/``.

Parte una carpeta de imágenes en dos conjuntos disjuntos a partir de un porcentaje y una seed,
de modo que el conjunto held-out quede fijo, reproducible y documentado por esa seed.

Se corre **una sola vez, a mano, antes de entrenar** (ejemplo desde ``diffusion-models/``)::

    python scripts/split_dataset.py --src data/cats --out data/cats_split --val-frac 0.1 --seed 0
    python scripts/split_dataset.py --src data/cats --out data/cats_split --move --overwrite

Deja ``<out>/train/<ruta relativa>`` y ``<out>/val/<ruta relativa>``: cada archivo se replica
preservando su ruta relativa al origen, así un dataset con subcarpetas no sufre colisiones de
nombre. Por defecto **copia** (el origen queda intacto) y solo mueve con ``--move``.

El **reparto** (:func:`split_paths`) está separado de la E/S a propósito: es una función pura de
la stdlib —sin torch, sin disco— que depende solo de ``(paths, val_frac, seed)``. El
descubrimiento de las imágenes ordena las rutas antes de llamarla, así que la partición nunca
depende del orden en que el filesystem las entregue.

El descubrimiento usa la **misma** definición de "imagen" que el loader de entrenamiento
(:data:`diffusion.data_generation.images.IMAGE_EXTENSIONS`, importada en vez de duplicada): es un
requisito de corrección, no una comodidad. Si el split repartiera un conjunto distinto del que el
loader lee, un archivo podría contar como validación y no ser leído nunca (o al revés). Como
contrapartida, los archivos con extensiones que el loader ignora quedan fuera del reparto — es
deliberado: el split refleja exactamente lo que el entrenamiento va a leer.

Todas las validaciones corren **antes** de tocar el primer archivo, así un error nunca deja un
destino a medias. ``--overwrite`` habilita escribir sobre un destino que ya tiene contenido y, para
que el reparto resultante sea una partición de verdad, **borra** ``<out>/train`` y ``<out>/val``
antes de replicar: re-partir con otra seed es el motivo habitual para re-correr la herramienta, y
sin ese borrado las imágenes que cambian de lado quedarían en las dos carpetas a la vez —
entrenadas y contadas como held-out al mismo tiempo—. El borrado ocurre **después** de todas las
validaciones y **justo antes** de la primera réplica, de modo que un error igual no toca el destino.

Como ese borrado es destructivo, tiene una precondición propia: origen y destino no pueden ser la
misma carpeta ni estar uno dentro del otro (se comparan las rutas **resueltas**, así ``.``/``..`` no
la burlan). Sin ella, ``--src data/cats --out data/cats`` borraría el ``data/cats/train`` original,
y un destino anidado en el origen haría que el descubrimiento recursivo barra las copias de la
corrida anterior y reparta más imágenes de las que hay.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import shutil
import sys

# Permitir ejecutar el script sin instalar el paquete (agrega ./src al path).
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diffusion.data_generation.images import IMAGE_EXTENSIONS

#: Nombres de las dos carpetas que se escriben bajo el destino.
TRAIN_DIRNAME = "train"
VAL_DIRNAME = "val"


def _validar_frac(val_frac: float) -> None:
    """Verifica que el porcentaje de validación caiga en el intervalo abierto (0, 1).

    Args:
        val_frac: Fracción del total que va a ``val/``.

    Raises:
        ValueError: Si ``val_frac`` no está estrictamente entre 0 y 1, informando el valor
            recibido.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(
            f"val_frac debe caer en el intervalo abierto (0, 1); se recibió {val_frac!r}."
        )


def _split_indices(n_total: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Reparte los índices ``0..n_total-1`` en (train, val) permutándolos con un RNG sembrado.

    A ``val`` le toca la cantidad que resulta de redondear ``val_frac`` sobre el total, y a
    ``train`` el resto. La permutación usa una instancia propia de :class:`random.Random`, no el
    RNG global: el resultado depende solo de ``(n_total, val_frac, seed)`` y evaluar el split no
    consume azar de nadie más.

    Args:
        n_total: Cantidad de elementos a repartir.
        val_frac: Fracción del total que va a ``val`` (ya validada).
        seed: Semilla de la permutación.

    Returns:
        Las dos listas de índices, cada una en orden creciente.

    Raises:
        ValueError: Si el reparto dejaría vacía cualquiera de las dos partes, informando el total
            de imágenes disponibles y el porcentaje pedido.
    """
    n_val = round(n_total * val_frac)
    if n_val == 0 or n_val >= n_total:
        raise ValueError(
            f"el reparto de {n_total} imágenes con val_frac={val_frac!r} dejaría vacía una de las "
            f"dos partes (val={n_val}, train={n_total - n_val}); se necesitan más imágenes o otro "
            f"porcentaje."
        )

    orden = list(range(n_total))
    random.Random(seed).shuffle(orden)
    return sorted(orden[n_val:]), sorted(orden[:n_val])


def split_paths(
    paths: list[pathlib.Path], val_frac: float, seed: int
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Reparte una lista **ordenada** de rutas en (train, val) de forma determinística.

    Función pura: no toca el disco ni muta la lista recibida. ``(train, val)`` es una partición de
    ``paths`` —la unión reconstruye la entrada y la intersección es vacía— y ambas partes quedan
    no vacías. Dos llamadas con la misma seed, el mismo porcentaje y el mismo conjunto de archivos
    devuelven exactamente el mismo reparto.

    Args:
        paths: Rutas a repartir, no vacía y **ya ordenada** (el orden fija el determinismo).
        val_frac: Fracción del total que va a validación, en el intervalo abierto (0, 1).
        seed: Semilla de la permutación.

    Returns:
        La tupla ``(train, val)``, cada lista en el mismo orden relativo que ``paths``, con
        ``len(val) == round(len(paths) * val_frac)``.

    Raises:
        ValueError: Si ``val_frac`` está fuera de (0, 1), o si el reparto dejaría vacía a
            ``train`` o a ``val``.
    """
    _validar_frac(val_frac)
    idx_train, idx_val = _split_indices(len(paths), val_frac, seed)
    return [paths[i] for i in idx_train], [paths[i] for i in idx_val]


def _descubrir(root: pathlib.Path) -> list[pathlib.Path]:
    """Descubre las imágenes bajo ``root`` recursivamente y en orden ordenado.

    Usa :data:`IMAGE_EXTENSIONS` (la constante del loader de entrenamiento) para decidir qué es
    una imagen, comparando la extensión en minúsculas: ``.JPG`` cuenta igual que ``.jpg``. El
    ``sorted`` es la precondición de :func:`split_paths`: sin él la partición dependería del orden
    en que el filesystem entrega los archivos.

    Args:
        root: Carpeta de origen, que se recorre recursivamente.

    Returns:
        Lista ordenada de rutas a los archivos de imagen encontrados.

    Raises:
        ValueError: Si ``root`` no existe (o no es un directorio), o si no contiene ninguna
            imagen; en los dos casos el mensaje informa la ruta.
    """
    if not root.is_dir():
        raise ValueError(f"la carpeta de origen no existe o no es un directorio: '{root}'.")
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        exts = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(
            f"no se encontraron imágenes en '{root}' (extensiones buscadas: {exts})."
        )
    return paths


def _validar_rutas(src: pathlib.Path, out: pathlib.Path) -> None:
    """Verifica que origen y destino no se solapen: ni la misma carpeta ni uno dentro del otro.

    Es la **precondición del borrado** de ``--overwrite``: con ``--src data/cats --out data/cats``,
    limpiar ``<out>/train`` borraría imágenes originales, y con el destino anidado en el origen el
    descubrimiento recursivo barrería las copias de la corrida anterior (repartiendo más imágenes
    de las que hay). Se comparan las rutas **resueltas**, así ``.``/``..`` y las diferencias de
    mayúsculas no la burlan.

    Args:
        src: Carpeta de origen.
        out: Carpeta de destino.

    Raises:
        ValueError: Si las dos rutas apuntan a la misma carpeta o si una está contenida en la otra,
            informando ambas rutas.
    """
    src_abs, out_abs = src.resolve(), out.resolve()
    if src_abs == out_abs:
        raise ValueError(
            f"el origen y el destino son la misma carpeta ('{src_abs}'); el split necesita un "
            f"destino aparte para poder limpiarlo sin borrar el origen."
        )
    if out_abs.is_relative_to(src_abs) or src_abs.is_relative_to(out_abs):
        raise ValueError(
            f"el origen '{src_abs}' y el destino '{out_abs}' están uno dentro del otro; el split "
            f"necesita carpetas separadas para no barrer sus propias copias ni borrar el origen."
        )


def _validar_destino(out: pathlib.Path, overwrite: bool) -> None:
    """Verifica que ``<out>/train`` y ``<out>/val`` no tengan contenido, salvo ``--overwrite``.

    Corre **antes** de replicar el primer archivo: si el destino está ocupado, el split aborta sin
    haber escrito nada. Una carpeta existente pero vacía no es un conflicto.

    Args:
        out: Carpeta de destino, la que va a contener ``train/`` y ``val/``.
        overwrite: Si es ``True`` no se valida nada (el usuario pidió sobreescribir).

    Raises:
        ValueError: Si alguna de las dos carpetas ya contiene archivos, informando la ruta en
            conflicto.
    """
    if overwrite:
        return
    for sub in (out / TRAIN_DIRNAME, out / VAL_DIRNAME):
        if sub.exists() and any(sub.rglob("*")):
            raise ValueError(
                f"el destino '{sub}' ya contiene archivos; borralo o pasá --overwrite para "
                f"reemplazarlo."
            )


def _limpiar_destino(out: pathlib.Path) -> None:
    """Borra ``<out>/train`` y ``<out>/val`` completas, si existen (el camino de ``--overwrite``).

    Es lo que hace que el reparto escrito sea una **partición**: sin este borrado, un segundo split
    con otra seed dejaría las imágenes que cambiaron de lado en las dos carpetas a la vez, y los
    conteos informados dejarían de describir lo que hay en disco. Se llama **después** de todas las
    validaciones (incluida :func:`_validar_rutas`, su precondición) y **justo antes** de la primera
    réplica.

    Args:
        out: Carpeta de destino, la que contiene ``train/`` y ``val/``.

    Raises:
        OSError: Si el borrado falla (permisos, archivo en uso); lo maneja :func:`main`.
    """
    for sub in (out / TRAIN_DIRNAME, out / VAL_DIRNAME):
        if sub.exists():
            shutil.rmtree(sub)


def _replicar(
    paths: list[pathlib.Path], src: pathlib.Path, dest: pathlib.Path, move: bool
) -> None:
    """Replica ``paths`` bajo ``dest`` preservando su ruta relativa a ``src``.

    Crea las carpetas intermedias que hagan falta, de modo que un dataset con subcarpetas se
    reproduzca con la misma estructura y dos archivos homónimos en subcarpetas distintas no
    colisionen.

    Args:
        paths: Rutas a replicar (todas dentro de ``src``).
        src: Raíz del origen, contra la que se calcula la ruta relativa.
        dest: Carpeta destino de esta partición (``<out>/train`` o ``<out>/val``).
        move: Si es ``True`` mueve los archivos; si no, los copia con metadata (``copy2``).

    Raises:
        OSError: Si la escritura falla (permisos, disco lleno, cross-device); lo maneja
            :func:`main`. Es el único camino que puede dejar el destino a medio escribir.
    """
    for p in paths:
        destino = dest / p.relative_to(src)
        destino.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(p), str(destino))
        else:
            shutil.copy2(p, destino)


def build_parser() -> argparse.ArgumentParser:
    """Arma el parser del CLI del split."""
    p = argparse.ArgumentParser(
        description=(
            "Parte una carpeta de imágenes en train/ y val/ de forma determinística "
            "(porcentaje + seed)."
        )
    )
    p.add_argument("--src", required=True,
                   help="Carpeta de origen (se recorre recursivamente).")
    p.add_argument("--out", required=True,
                   help="Carpeta de destino: se escriben '<out>/train' y '<out>/val'.")
    p.add_argument("--val-frac", dest="val_frac", type=float, default=0.1,
                   help="Fracción del total que va a val/, en el intervalo abierto (0, 1).")
    p.add_argument("--seed", type=int, default=0,
                   help="Semilla del reparto: la que documenta y reproduce el split.")
    p.add_argument("--move", action="store_true",
                   help="Mover los archivos en vez de copiarlos (vacía el origen).")
    p.add_argument("--overwrite", action="store_true",
                   help="Reemplazar un destino que ya tiene contenido: borra '<out>/train' y "
                        "'<out>/val' antes de repartir, así el resultado no mezcla el split "
                        "anterior con el nuevo.")
    return p


def main(argv=None) -> int:
    """Punto de entrada del CLI: descubre, reparte y replica.

    Args:
        argv: Argumentos de la línea de comandos (``None`` usa ``sys.argv``).

    Returns:
        ``0`` si el split se completó; ``2`` si alguna validación falló (con el mensaje en
        ``stderr`` y sin haber escrito nada) o si la E/S falló a mitad de camino (avisando que el
        destino puede haber quedado incompleto), el mismo mapeo que ``scripts/train.py``.
    """
    # En Windows (py<3.15) la consola no usa UTF-8 por defecto; forzarlo evita
    # mojibake en los acentos de los mensajes.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    src, out = pathlib.Path(args.src), pathlib.Path(args.out)

    # Todas las validaciones ANTES de tocar el primer archivo: un error no deja un destino a
    # medias (ni borrado a medias). Cubre el porcentaje (1.9), el origen (1.11), el solapamiento
    # de origen y destino (precondición del borrado de --overwrite), el destino ocupado (1.8) y el
    # reparto que dejaría una partición vacía (1.10).
    try:
        _validar_frac(args.val_frac)
        paths = _descubrir(src)
        _validar_rutas(src, out)
        _validar_destino(out, args.overwrite)
        train, val = split_paths(paths, args.val_frac, args.seed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Recién acá se toca el disco. El borrado de --overwrite va inmediatamente antes de la primera
    # réplica para que el destino describa solo este reparto (1.1) y los conteos que se informan
    # después coincidan con lo que quedó en disco (1.7).
    try:
        if args.overwrite:
            _limpiar_destino(out)
        _replicar(train, src, out / TRAIN_DIRNAME, args.move)
        _replicar(val, src, out / VAL_DIRNAME, args.move)
    except OSError as exc:
        print(
            f"error: falló la E/S del split: {exc}. El destino '{out}' puede haber quedado "
            f"incompleto; revisalo antes de reintentar.",
            file=sys.stderr,
        )
        return 2

    modo = "movidas" if args.move else "copiadas"
    print(
        f"Split de '{src}': {len(paths)} imágenes {modo} "
        f"(val_frac={args.val_frac}, seed={args.seed})"
    )
    print(f"train -> {out / TRAIN_DIRNAME} ({len(train)} imágenes)")
    print(f"val   -> {out / VAL_DIRNAME} ({len(val)} imágenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
