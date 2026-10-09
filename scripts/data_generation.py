"""CLI para generar datasets de puntos de juguete y un preview opcional.

Ejemplos::

    python scripts/data_generation.py --shape two_moons --dim 2 --n-samples 2000 \\
        --seed 0 --out data/two_moons.npz --preview data/two_moons.png
    python scripts/data_generation.py --shape gaussian --dim 5 --n-samples 1000 \\
        --out data/gauss5.npz --preview data/gauss5.png

El dataset se guarda como ``.npz`` (clave ``X`` float32 + ``meta`` en JSON, y
``color``/``mean``/``std`` cuando corresponden). El preview es un scatter
(PCA->2D si ``dim > 2``).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Permitir ejecutar el script sin instalar el paquete (agrega ./src al path).
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from diffusion.data_generation import available_shapes, make_distribution


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera un dataset de puntos de juguete (+ preview opcional)."
    )
    p.add_argument("--shape", required=True, choices=available_shapes(),
                   help="Tipo de distribución.")
    p.add_argument("--dim", type=int, default=2, help="Dimensión de los puntos.")
    p.add_argument("--n-samples", dest="n_samples", type=int, default=2000,
                   help="Cantidad de puntos a generar.")
    p.add_argument("--seed", type=int, default=None,
                   help="Semilla para reproducibilidad.")
    p.add_argument("--noise", type=float, default=None,
                   help="Ruido (donde aplica: moons / spiral / swiss_roll).")
    p.add_argument("--n-components", dest="n_components", type=int, default=None,
                   help="Cantidad de componentes (solo 'mixture').")
    p.add_argument("--standardize", action="store_true",
                   help="Estandarizar a media 0 / std 1 por columna.")
    p.add_argument("--out", type=str, default=None, help="Ruta .npz de salida.")
    p.add_argument("--preview", type=str, default=None,
                   help="Ruta .png para el scatter de preview.")
    return p


def save_npz(path: str, dist, x: np.ndarray) -> None:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "shape": dist.name,
        "dim": dist.dim,
        "n_samples": int(len(x)),
        "seed": dist.seed,
        "standardize": dist.standardize,
        "noise": dist.noise,
    }
    extras = {}
    if dist.color_ is not None:
        extras["color"] = np.asarray(dist.color_)
    if dist.mean_ is not None:
        extras["mean"] = dist.mean_
        extras["std"] = dist.std_
    np.savez(out, X=x.astype(np.float32), meta=json.dumps(meta), **extras)


def save_preview(path: str, dist, x: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if x.shape[1] == 1:
        pts = np.column_stack([x[:, 0], np.zeros(len(x))])
        xlabel, ylabel = "x1", ""
    elif x.shape[1] == 2:
        pts = x
        xlabel, ylabel = "x1", "x2"
    else:
        from sklearn.decomposition import PCA

        pts = PCA(n_components=2).fit_transform(x)
        xlabel, ylabel = "PCA-1", "PCA-2"

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter_kwargs = {"s": 6, "alpha": 0.7}
    if dist.color_ is not None:
        scatter_kwargs.update(c=dist.color_, cmap="viridis")
    sc = ax.scatter(pts[:, 0], pts[:, 1], **scatter_kwargs)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dist.name} (dim={dist.dim}, n={len(x)})")
    if dist.color_ is not None:
        fig.colorbar(sc, ax=ax, shrink=0.8)
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
    kwargs = {"standardize": args.standardize, "seed": args.seed}
    if args.noise is not None:
        kwargs["noise"] = args.noise
    if args.n_components is not None:
        kwargs["n_components"] = args.n_components

    try:
        dist = make_distribution(args.shape, args.dim, **kwargs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    x = dist.sample(args.n_samples)
    print(
        f"Generado {dist.name}: dim={dist.dim} n={len(x)} dtype={x.dtype} "
        f"rango=[{x.min():.3f}, {x.max():.3f}]"
    )
    if args.out:
        save_npz(args.out, dist, x)
        print(f"Dataset  -> {args.out}")
    if args.preview:
        save_preview(args.preview, dist, x)
        print(f"Preview  -> {args.preview}")
    if not args.out and not args.preview:
        print("(sin --out ni --preview: no se guardó nada)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
