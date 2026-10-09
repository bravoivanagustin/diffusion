"""
Deduplica imagenes pixel a pixel y las renombra secuencialmente (00001.jpg, 00002.jpg, ...).

Uso:
    python limpiar_imagenes.py "C:\\Users\\bravo\\Downloads\\cat-image"
    python limpiar_imagenes.py "C:\\Users\\bravo\\Downloads\\cat-image" --dry-run

Requiere: pip install pillow
"""

import argparse
import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pixel_hash(path: Path) -> str:
    """Hash del contenido de pixeles (normalizado a RGB, no de los bytes del archivo)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        return hashlib.sha1(im.tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Dedup pixel a pixel + renombrado secuencial.")
    ap.add_argument("folder", help="Carpeta con las imagenes")
    ap.add_argument("--dry-run", action="store_true", help="No borra ni renombra, solo muestra que haria")
    ap.add_argument("--ext", default=".jpg", help="Extension final (default: .jpg)")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        raise SystemExit(f"No existe la carpeta: {folder}")

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"{len(files)} imagenes encontradas en {folder}")
    if not files:
        return

    # --- 1. Dedup pixel a pixel ---
    seen: dict[str, Path] = {}
    keep: list[Path] = []
    dupes: list[Path] = []
    corrupt: list[Path] = []

    for i, p in enumerate(files, start=1):
        try:
            h = pixel_hash(p)
        except (UnidentifiedImageError, OSError) as e:
            corrupt.append(p)
            print(f"  [corrupta/ilegible] {p.name}: {e}")
            continue
        if h in seen:
            dupes.append(p)
        else:
            seen[h] = p
            keep.append(p)
        if i % 2000 == 0:
            print(f"  ...procesadas {i}/{len(files)}")

    print(f"\n{len(keep)} unicas | {len(dupes)} repetidas | {len(corrupt)} corruptas")

    if not args.dry_run:
        for p in dupes:
            p.unlink()
        print(f"Borradas {len(dupes)} repetidas.")

    # --- 2. Renombrado secuencial (dos fases para evitar colisiones) ---
    keep = sorted(keep)  # orden estable
    width = len(str(len(keep)))

    if args.dry_run:
        print(f"[dry-run] renombraria {len(keep)} archivos -> "
              f"{1:0{width}d}{args.ext} .. {len(keep):0{width}d}{args.ext}")
        return

    # Fase A: a nombres temporales unicos
    tmp: list[Path] = []
    for i, p in enumerate(keep):
        t = p.with_name(f"__tmp_{i}{args.ext}")
        p.rename(t)
        tmp.append(t)

    # Fase B: a nombres finales secuenciales
    for i, t in enumerate(tmp, start=1):
        t.rename(t.with_name(f"{i:0{width}d}{args.ext}"))

    print(f"Renombradas {len(keep)} -> {1:0{width}d}{args.ext} .. {len(keep):0{width}d}{args.ext}")


if __name__ == "__main__":
    main()