"""CLI para generar muestras ``x_0`` desde un checkpoint entrenado (Eje 2).

Envuelve :func:`diffusion.samplers.generate_from_checkpoint`: carga la red y su metadata,
reconstruye la SDE del Eje 1, arma el sampler del Eje 2 elegido por ``--sampler`` e integra el
proceso reverso. Cambiar de sampler NO reentrena la red.

Ejemplos (correr desde ``diffusion-models/``)::

    python scripts/sample.py checkpoints/vp_mixture.pt --sampler pf_ode \\
        --n-samples 2000 --n-steps 500 --seed 0 --out data/vp_mixture_pf_ode.npz
    python scripts/sample.py checkpoints/vp_mixture.pt --sampler pc \\
        --n-samples 1000 --snr 0.1 --n-corrector 1 --out data/pc.npz --trajectory

Guarda un ``.npz`` con la clave ``samples`` (y ``trajectory`` si se pasa ``--trajectory``).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Permitir ejecutar el script sin instalar el paquete (agrega ./src al path).
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diffusion.samplers import (
    available_samplers,
    available_time_grids,
    generate_from_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Genera muestras x_0 desde un checkpoint entrenado (proceso reverso, Eje 2)."
    )
    p.add_argument("checkpoint", help="Ruta del checkpoint .pt (state_dict + metadata).")
    p.add_argument("--sampler", required=True, choices=available_samplers(),
                   help="Sampler del proceso reverso a usar.")
    p.add_argument("--n-samples", dest="n_samples", type=int, default=2000,
                   help="Cantidad de muestras a generar.")
    p.add_argument("--n-steps", dest="n_steps", type=int, default=500,
                   help="Cantidad de pasos de integración del sampler.")
    p.add_argument("--t-eps", dest="t_eps", type=float, default=None,
                   help="Tiempo terminal de la integración (0 < t_eps < T); default del "
                        "sampler: 1e-3. Conviene no bajarlo del t_eps de entrenamiento.")
    p.add_argument("--time-grid", dest="time_grid", type=str, default=None,
                   choices=available_time_grids(),
                   help="Distribución de los tiempos de integración: 'uniform' (default, "
                        "espaciado constante en t) o 'logsnr' (constante en log-SNR).")
    p.add_argument("--seed", type=int, default=None,
                   help="Semilla para reproducibilidad (prior y pasos estocásticos).")
    p.add_argument("--out", type=str, default=None,
                   help="Ruta .npz de salida (clave 'samples', y 'trajectory' con --trajectory).")
    p.add_argument("--trajectory", action="store_true",
                   help="Capturar y guardar la trayectoria de integración.")
    p.add_argument("--map-location", dest="map_location", type=str, default="cpu",
                   help="Dispositivo donde cargar los pesos del checkpoint (p. ej. cpu / cuda).")
    # Parámetros exclusivos de predictor-corrector; los demás samplers los descartan.
    p.add_argument("--snr", type=float, default=None,
                   help="Signal-to-noise ratio del corrector de Langevin (solo sampler 'pc').")
    p.add_argument("--n-corrector", dest="n_corrector", type=int, default=None,
                   help="Correcciones de Langevin por nivel de ruido (solo sampler 'pc').")
    return p


def main(argv=None) -> int:
    # En Windows (py<3.15) la consola no usa UTF-8 por defecto; forzarlo evita
    # mojibake en los acentos de los mensajes.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    # Solo reenviar los kwargs del sampler que el usuario haya provisto; el factory descarta
    # los que no apliquen al sampler elegido (criterio 4.4).
    sampler_kwargs = {}
    if args.t_eps is not None:
        sampler_kwargs["t_eps"] = args.t_eps
    if args.time_grid is not None:
        sampler_kwargs["time_grid"] = args.time_grid
    if args.snr is not None:
        sampler_kwargs["snr"] = args.snr
    if args.n_corrector is not None:
        sampler_kwargs["n_corrector"] = args.n_corrector

    try:
        x0 = generate_from_checkpoint(
            args.checkpoint,
            args.sampler,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            seed=args.seed,
            out=args.out,
            save_trajectory=args.trajectory,
            map_location=args.map_location,
            **sampler_kwargs,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Generado sampler={args.sampler}: muestras={tuple(x0.shape)} dtype={x0.dtype} "
        f"n_steps={args.n_steps} seed={args.seed}"
    )
    if args.out:
        print(f"Salida -> {args.out}" + ("  (+ trayectoria)" if args.trajectory else ""))
    else:
        print("(sin --out: no se guardó nada)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
