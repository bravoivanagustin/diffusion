"""CLI: GIF de la trayectoria reversa ruido -> gato desde un checkpoint (Eje 2).

Integra el proceso reverso con el sampler elegido, captura la trayectoria completa y guarda
``--n-frames`` estados **equiespaciados en pasos** desde ``x_T`` (ruido puro, paso 0) hasta la
muestra final (paso ``n_steps``, con el denoising de Tweedie del driver). Con varias muestras
cada frame es una grilla, todas avanzando en el mismo paso.

Ejemplos (correr desde la raíz del repo)::

    uv run python scripts/trajectory_gif.py models/phase_2/cats_vp.pt --sampler pc \\
        --n-steps 1000 --seed 42 --device cuda --out-dir outputs/gif_pc
    uv run python scripts/trajectory_gif.py models/phase_2/cats_vp.pt --sampler heun \\
        --n-steps 250 --time-grid logsnr --n-samples 4 --device cuda --out-dir outputs/gif_heun

Salida en ``--out-dir``:

- ``frames/frame_XX_stepYYYY.png``: un PNG por frame.
- ``trajectory.gif``: la animación (se queda ``--hold`` segundos en el gato final).
- ``contact_sheet.png``: los frames en una sola imagen, para un informe o póster.
- ``frames.npz``: ``frames`` (estados crudos en [-1, 1]), ``steps`` y ``times`` de cada frame.

Ojo con la grilla ``uniform``: en VP la imagen recién aparece en el último tramo de ``t``, así
que muchos frames equiespaciados en pasos son casi puro ruido. ``--time-grid logsnr`` reparte
los pasos parejo en señal-ruido y el gato emerge más gradualmente a lo largo del GIF.

La trayectoria completa se guarda en el device del sampleo antes de submuestrear:
``(n_steps + 1) x n_samples x 3 x H x W`` floats (~50 MB por muestra a 1000 pasos y 64x64).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Permitir ejecutar el script sin instalar el paquete (agrega ./src al path).
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

from diffusion.samplers import (
    available_samplers,
    available_time_grids,
    load_score_model,
    make_sampler,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GIF de la trayectoria reversa x_T -> x_0 desde un checkpoint entrenado."
    )
    p.add_argument("checkpoint", help="Ruta del checkpoint .pt (state_dict + metadata).")
    p.add_argument("--out-dir", dest="out_dir", required=True,
                   help="Carpeta de salida (frames, GIF, contact sheet y .npz).")
    p.add_argument("--sampler", default="pc", choices=available_samplers(),
                   help="Sampler del proceso reverso (default: pc).")
    p.add_argument("--n-steps", dest="n_steps", type=int, default=1000,
                   help="Pasos de integración del sampler (default: 1000).")
    p.add_argument("--n-frames", dest="n_frames", type=int, default=50,
                   help="Frames equiespaciados a guardar, extremos incluidos (default: 50).")
    p.add_argument("--n-samples", dest="n_samples", type=int, default=1,
                   help="Muestras en paralelo; con más de una cada frame es una grilla.")
    p.add_argument("--nrow", type=int, default=None,
                   help="Muestras por fila de la grilla (default: ~cuadrada).")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla del prior y de los pasos estocásticos (default: 42).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device del sampleo (default: cuda si hay).")
    p.add_argument("--t-eps", dest="t_eps", type=float, default=None,
                   help="Tiempo terminal de la integración; default del sampler: 1e-3.")
    p.add_argument("--time-grid", dest="time_grid", default=None,
                   choices=available_time_grids(),
                   help="Grilla temporal: 'uniform' (default) o 'logsnr'.")
    p.add_argument("--snr", type=float, default=None,
                   help="SNR del corrector de Langevin (solo 'pc').")
    p.add_argument("--n-corrector", dest="n_corrector", type=int, default=None,
                   help="Correcciones de Langevin por paso (solo 'pc').")
    p.add_argument("--no-denoise", dest="denoise", action="store_false",
                   help="No aplicar el denoising de Tweedie en el último frame.")
    p.add_argument("--scale", type=int, default=4,
                   help="Factor de ampliación (vecino más cercano) de cada imagen (default: 4).")
    p.add_argument("--no-label", dest="label", action="store_false",
                   help="No escribir paso y t debajo de cada frame.")
    p.add_argument("--duration", type=float, default=0.08,
                   help="Segundos por frame del GIF (default: 0.08).")
    p.add_argument("--hold", type=float, default=2.0,
                   help="Segundos que el GIF se queda en el frame final (default: 2.0).")
    return p


def frame_indices(n_steps: int, n_frames: int) -> np.ndarray:
    """``n_frames`` pasos equiespaciados en ``[0, n_steps]``, extremos incluidos y sin repetir."""
    if not 2 <= n_frames <= n_steps + 1:
        raise ValueError(
            f"n_frames debe estar entre 2 y n_steps + 1 = {n_steps + 1}; recibí {n_frames}"
        )
    return np.round(np.linspace(0, n_steps, n_frames)).astype(int)


def to_image(x: torch.Tensor, nrow: int, scale: int) -> Image.Image:
    """Lote ``(B, 3, H, W)`` en [-1, 1] -> imagen PIL (grilla si ``B > 1``), ampliada."""
    x = (x.float() * 0.5 + 0.5).clamp(0.0, 1.0)
    grid = make_grid(x, nrow=nrow, padding=2 if x.shape[0] > 1 else 0, pad_value=1.0)
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    img = Image.fromarray(arr)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def add_label(img: Image.Image, text: str) -> Image.Image:
    """Agrega una franja blanca debajo de ``img`` con ``text`` centrado."""
    try:
        font = ImageFont.load_default(size=max(12, img.width // 20))
    except TypeError:  # Pillow < 10.1 no acepta size
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    band = (bottom - top) + 10
    out = Image.new("RGB", (img.width, img.height + band), "white")
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text(
        ((img.width - (right - left)) // 2, img.height + 5 - top), text, fill="black", font=font
    )
    return out


def contact_sheet(frames: list[Image.Image], cols: int = 10) -> Image.Image:
    """Pega los frames en una grilla de ``cols`` columnas."""
    w, h = frames[0].size
    rows = -(-len(frames) // cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for k, f in enumerate(frames):
        sheet.paste(f, ((k % cols) * w, (k // cols) * h))
    return sheet


def main(argv=None) -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    sampler_kwargs = {}
    for key in ("t_eps", "time_grid", "snr", "n_corrector"):
        if getattr(args, key) is not None:
            sampler_kwargs[key] = getattr(args, key)

    try:
        idx = frame_indices(args.n_steps, args.n_frames)
        net, sde, meta = load_score_model(args.checkpoint, device=args.device)
        sampler = make_sampler(args.sampler, sde, net, n_steps=args.n_steps, **sampler_kwargs)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    print(f"checkpoint={args.checkpoint} sde={meta['sde_name']} sampler={sampler!r} "
          f"device={args.device} seed={args.seed}")

    _, trajectory = sampler.sample(
        args.n_samples, generator=generator, device=args.device,
        return_trajectory=True, denoise=args.denoise,
    )
    states = trajectory[torch.as_tensor(idx, device=trajectory.device)].cpu()
    del trajectory
    # Tiempo de cada estado guardado: el paso k de la trayectoria vive en grid[k].
    times = sampler._time_grid()[idx].numpy()

    if not torch.isfinite(states[-1]).all():
        print("aviso: la muestra final tiene NaN/inf (el sampler divergió)", file=sys.stderr)

    out_dir = pathlib.Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    nrow = args.nrow or int(np.ceil(np.sqrt(args.n_samples)))
    images = []
    for k, (step, t) in enumerate(zip(idx, times)):
        img = to_image(states[k], nrow, args.scale)
        if args.label:
            final = args.denoise and step == args.n_steps
            img = add_label(
                img, f"paso {step}/{args.n_steps}   t = {t:.3f}" + (" + Tweedie" if final else "")
            )
        img.save(frames_dir / f"frame_{k:02d}_step{step:04d}.png")
        images.append(img)

    ms = max(20, int(round(args.duration * 1000)))
    durations = [ms] * (len(images) - 1) + [max(ms, int(round(args.hold * 1000)))]
    gif_path = out_dir / "trajectory.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False)

    contact_sheet(images).save(out_dir / "contact_sheet.png")
    np.savez(out_dir / "frames.npz", frames=states.numpy(), steps=idx, times=times)

    print(f"{len(images)} frames (pasos {idx[0]}..{idx[-1]}) -> {out_dir}")
    print(f"GIF -> {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
