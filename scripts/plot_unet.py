"""Diagrama de la arquitectura de la ``ScoreUNet`` de una corrida, derivado del modelo real.

No dibuja un esquema genérico de U-Net: **construye la red** desde el bloque ``model:`` de un
config y la instrumenta con hooks para leer las resoluciones y los canales que realmente
atraviesa un forward, más el conteo de parámetros por etapa. Si la config cambia, el diagrama
cambia con ella.

Uso (desde ``diffusion-models/``)::

    uv run python scripts/plot_unet.py --config config/cats/cats_vp.yaml --out figs/unet_cats.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # permite correr el script sin instalar el paquete
    sys.path.insert(0, str(_SRC))

# Paleta categórica validada (checks de banda de luminosidad, croma, separación CVD sobre TODOS
# los pares y piso de visión normal). El tipo de bloque va SIEMPRE con su etiqueta de texto: el
# color acompaña, nunca identifica solo.
_AZUL = "#2a78d6"      # atención
_NARANJA = "#eb6834"   # downsample
_AQUA = "#1baf7a"      # upsample
_VIOLETA = "#4a3aa7"   # entrada / salida
_TINTA = "#1a1a1a"
_TINTA_2 = "#5a5a5a"
_NEUTRO = "#e8e6e1"    # ConvResBlock: es el material por defecto, va recesivo
_BORDE = "#b8b4ac"


def _introspeccionar(config_path: pathlib.Path) -> dict:
    """Construye la red del config y devuelve su estructura medida (no inferida)."""
    import yaml
    import torch
    from diffusion.models import make_model
    from diffusion.models.unet import AttentionBlock, ConvResBlock, Downsample, Upsample

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    crudo = dict(cfg["model"])
    crudo.pop("score_parametrization", None)
    nombre = crudo.pop("name", "unet")
    net = make_model(nombre, **crudo).eval()

    def params(mod) -> int:
        return sum(p.numel() for p in mod.parameters())

    # Resoluciones reales: se leen de un forward, no se calculan a mano.
    resoluciones: dict[str, int] = {}

    def hook(clave):
        def f(_mod, _inp, out):
            resoluciones[clave] = out.shape[-1]
        return f

    handles = [net.conv_in.register_forward_hook(hook("conv_in"))]
    for i, etapa in enumerate(net.down_blocks):
        handles.append(etapa[-1].register_forward_hook(hook(f"down{i}")))
    for i, etapa in enumerate(net.up_blocks):
        handles.append(etapa[-1].register_forward_hook(hook(f"up{i}")))
    handles.append(net.mid_block2.register_forward_hook(hook("mid")))
    lado = net.image_size
    with torch.no_grad():
        net(torch.zeros(1, net.in_channels, lado, lado), torch.tensor([0.5]))
    for h in handles:
        h.remove()

    # Agrupación por nivel de resolución: es como se lee una U-Net, y evita 28 cajitas.
    niveles: list[dict] = []
    for nivel, mult in enumerate(net.channel_mults):
        res = lado // (2 ** nivel)
        niveles.append({
            "nivel": nivel,
            "resolucion": res,
            "canales": net.base_channels * mult,
            "atencion": res in set(net.attn_resolutions),
            "enc_bloques": net.num_res_blocks,
            "dec_bloques": net.num_res_blocks + 1,
            "enc_params": 0,
            "dec_params": 0,
            "baja": nivel != len(net.channel_mults) - 1,
        })

    # Parámetros por nivel, sumando las etapas que le corresponden (el orden de down_blocks es
    # nivel a nivel; el de up_blocks es el espejo).
    idx = 0
    for lv in niveles:
        for _ in range(lv["enc_bloques"]):
            lv["enc_params"] += params(net.down_blocks[idx]); idx += 1
        if lv["baja"]:
            lv["enc_params"] += params(net.down_blocks[idx]); idx += 1
    idx = 0
    for lv in reversed(niveles):
        for _ in range(lv["dec_bloques"]):
            lv["dec_params"] += params(net.up_blocks[idx]); idx += 1

    return {
        "niveles": niveles,
        "image_size": lado,
        "in_channels": net.in_channels,
        "base_channels": net.base_channels,
        "groups": net.groups,
        "embed_dim": net.time_mlp.embed_dim,
        "time_embed_dim": net.time_mlp.time_embed_dim,
        "time_scale": net.time_scale,
        "total": params(net),
        "p_time": params(net.time_mlp),
        "p_io": params(net.conv_in) + params(net.out_norm) + params(net.conv_out),
        "p_mid": params(net.mid_block1) + params(net.mid_attn) + params(net.mid_block2),
        "res_mid": resoluciones["mid"],
        "config": config_path.name,
        "parametrizacion": cfg["model"].get("score_parametrization"),
    }


def _caja(ax, x, y, w, h, texto, color, borde=_BORDE, tinta=_TINTA, tam=8.0, negrita=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.055",
        facecolor=color, edgecolor=borde, linewidth=0.9, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=tam,
            color=tinta, zorder=4, linespacing=1.35,
            fontweight="bold" if negrita else "normal")


def dibujar(info: dict, salida: pathlib.Path) -> None:
    """Dibuja la U y la guarda en ``salida`` (y en el .pdf hermano, vectorial para el informe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    niveles = info["niveles"]
    n = len(niveles)
    fig, ax = plt.subplots(figsize=(11.5, 1.45 * n + 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(-2.05, 1.45 * n + 2.05)
    ax.axis("off")

    x_enc, x_dec, ancho = 2.15, 7.5, 2.35
    alto = 0.82
    y0 = 1.45 * (n - 1)

    # --- Entrada y salida (fuera de la U, arriba) ---
    y_io = y0 + 1.85
    _caja(ax, x_enc, y_io, ancho, alto,
          f"$x_t$   ({info['in_channels']}, {info['image_size']}, {info['image_size']})\n"
          f"conv 3×3 → {info['base_channels']} canales",
          "#f0eefb", _VIOLETA, tam=8.5)
    _caja(ax, x_dec, y_io, ancho, alto,
          f"GroupNorm + SiLU + conv\n"
          f"$s_\\theta(x_t,t)$   ({info['in_channels']}, {info['image_size']}, {info['image_size']})",
          "#f0eefb", _VIOLETA, tam=8.5)

    # --- Embedding temporal: alimenta TODOS los ConvResBlock ---
    _caja(ax, 0.05, y_io, 1.75, alto,
          f"$t$ → sinusoidal\n→ MLP ({info['time_embed_dim']})",
          "#f7f6f3", _TINTA_2, tam=8)
    ax.annotate("", xy=(0.92, y_io - 0.12), xytext=(0.92, -1.05),
                arrowprops=dict(arrowstyle="-", color=_TINTA_2, lw=1.0, ls=(0, (4, 3))), zorder=1)
    ax.text(0.92, -1.18, "el vector temporal entra\nen cada bloque residual",
            ha="center", va="top", fontsize=7.2, color=_TINTA_2, style="italic")

    for lv in niveles:
        y = y0 - 1.45 * lv["nivel"]
        res, ch = lv["resolucion"], lv["canales"]

        # Banda de resolución
        ax.text(0.05, y + alto / 2, f"{res}×{res}", ha="left", va="center",
                fontsize=9.5, color=_TINTA, fontweight="bold")
        ax.text(0.05, y + alto / 2 - 0.28, f"{ch} canales", ha="left", va="center",
                fontsize=7.5, color=_TINTA_2)
        ax.plot([0, 12], [y + alto + 0.31] * 2, color="#eeece7", lw=0.8, zorder=0)

        attn = "  +  atención" if lv["atencion"] else ""
        _caja(ax, x_enc, y, ancho, alto,
              f"{lv['enc_bloques']} ×  ResBlock{attn}\n{ch} canales",
              _AZUL if lv["atencion"] else _NEUTRO,
              tinta="white" if lv["atencion"] else _TINTA)
        ax.text(x_enc - 0.12, y + alto / 2, f"{lv['enc_params'] / 1e6:.2f} M",
                ha="right", va="center", fontsize=7.2, color=_TINTA_2)

        _caja(ax, x_dec, y, ancho, alto,
              f"{lv['dec_bloques']} ×  ResBlock{attn}\n{ch} canales",
              _AZUL if lv["atencion"] else _NEUTRO,
              tinta="white" if lv["atencion"] else _TINTA)
        ax.text(x_dec + ancho + 0.12, y + alto / 2, f"{lv['dec_params'] / 1e6:.2f} M",
                ha="left", va="center", fontsize=7.2, color=_TINTA_2)

        # Skip connection: el aporte de la U. Va con etiqueta, no solo con el color.
        ax.add_patch(FancyArrowPatch(
            (x_enc + ancho, y + alto / 2), (x_dec, y + alto / 2),
            arrowstyle="-|>", mutation_scale=11, color="#8a8680", lw=1.2,
            linestyle=(0, (5, 3)), zorder=2, shrinkA=3, shrinkB=3))
        ax.text((x_enc + ancho + x_dec) / 2, y + alto / 2 + 0.15,
                "skip  (concatena)", ha="center", va="bottom",
                fontsize=7.2, color="#8a8680", style="italic")

        if lv["baja"]:
            y_sig = y - 1.45
            _caja(ax, x_enc + 0.62, y - 0.60, 1.1, 0.36, "↓ 2", _NARANJA, _NARANJA,
                  tinta="white", tam=7.5, negrita=True)
            _caja(ax, x_dec + 0.62, y - 0.60, 1.1, 0.36, "↑ 2", _AQUA, _AQUA,
                  tinta="white", tam=7.5, negrita=True)
            for xx, col in ((x_enc + 1.17, _NARANJA), (x_dec + 1.17, _AQUA)):
                ax.annotate("", xy=(xx, y_sig + alto + 0.02), xytext=(xx, y - 0.02),
                            arrowprops=dict(arrowstyle="-", color=col, lw=1.1), zorder=1)

    # --- Bottleneck ---
    y_mid = y0 - 1.45 * (n - 1) - 1.28
    _caja(ax, x_enc, y_mid, x_dec + ancho - x_enc, 0.62,
          f"Bottleneck  ·  ResBlock → atención → ResBlock  ·  "
          f"{niveles[-1]['canales']} canales @ {info['res_mid']}×{info['res_mid']}"
          f"   ({info['p_mid'] / 1e6:.2f} M)",
          _AZUL, _AZUL, tinta="white", tam=8.5)
    ax.annotate("", xy=(x_enc + ancho / 2, y_mid + 0.66), xytext=(x_enc + ancho / 2, y0 - 1.45 * (n - 1) - 0.02),
                arrowprops=dict(arrowstyle="-", color=_TINTA_2, lw=1.1), zorder=1)
    ax.annotate("", xy=(x_dec + ancho / 2, y0 - 1.45 * (n - 1) - 0.02), xytext=(x_dec + ancho / 2, y_mid + 0.66),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=11, color=_TINTA_2, lw=1.1), zorder=1)

    # Flechas de entrada y salida
    ax.annotate("", xy=(x_enc + ancho / 2, y0 + alto + 0.02), xytext=(x_enc + ancho / 2, y_io - 0.02),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=11, color=_TINTA_2, lw=1.1), zorder=1)
    ax.annotate("", xy=(x_dec + ancho / 2, y_io - 0.02), xytext=(x_dec + ancho / 2, y0 + alto + 0.02),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=11, color=_TINTA_2, lw=1.1), zorder=1)

    ax.text(x_enc + ancho / 2, y_io + alto + 0.42, "ENCODER", ha="center", fontsize=9,
            color=_TINTA_2, fontweight="bold")
    ax.text(x_dec + ancho / 2, y_io + alto + 0.42, "DECODER", ha="center", fontsize=9,
            color=_TINTA_2, fontweight="bold")

    # --- Título y ficha ---
    par = f",  parametrización $\\varepsilon$" if info["parametrizacion"] == "epsilon" else ""
    fig.suptitle(
        f"ScoreUNet — {info['config']}   ·   {info['total'] / 1e6:.1f} M parámetros{par}",
        fontsize=12.5, y=0.985, color=_TINTA, fontweight="bold")
    ax.text(6, y_io + alto + 1.02,
            f"$s_\\theta(x_t, t) \\approx \\nabla_x \\log p_t(x)$   "
            f"·   GroupNorm({info['groups']})   ·   sin dropout ni batchnorm: la red es determinística",
            ha="center", fontsize=8.5, color=_TINTA_2)

    ax.text(6, y_io + alto + 0.72,
            f"embedding temporal {info['p_time'] / 1e6:.2f} M   ·   entrada + salida "
            f"{info['p_io'] / 1e3:.1f} k   ·   escala de tiempo {info['time_scale']:.0f}"
            f"   ·   los números al costado de cada nivel son sus parámetros",
            ha="center", va="center", fontsize=7.5, color=_TINTA_2)

    # Leyenda: cada tipo lleva su etiqueta, el color solo acompaña
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=_NEUTRO, edgecolor=_BORDE, label="bloque residual (conv 3×3 ×2 + tiempo)"),
        Patch(facecolor=_AZUL, edgecolor=_AZUL, label="con auto-atención"),
        Patch(facecolor=_NARANJA, edgecolor=_NARANJA, label="reducción ×2 (conv stride 2)"),
        Patch(facecolor=_AQUA, edgecolor=_AQUA, label="ampliación ×2 (nearest + conv)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False, fontsize=8)

    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(salida.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Diagrama de la ScoreUNet de un config.")
    p.add_argument("--config", required=True, help="YAML de la corrida (usa su bloque model:)")
    p.add_argument("--out", required=True, help="PNG de salida (se escribe también el .pdf)")
    args = p.parse_args(argv)

    info = _introspeccionar(pathlib.Path(args.config))
    salida = pathlib.Path(args.out)
    dibujar(info, salida)
    print(f"Diagrama -> {salida}  (+ {salida.with_suffix('.pdf').name})")
    print(f"  {info['total']:,} parámetros · {len(info['niveles'])} niveles · "
          f"atención en {[l['resolucion'] for l in info['niveles'] if l['atencion']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
