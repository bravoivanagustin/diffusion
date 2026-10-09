# Diffusion Models

TP final de Cálculo Estocástico: modelos de difusión basados en SDEs, desde datos de juguete 2D hasta imágenes de gatos con una U-Net.

## Estructura

| Carpeta | Contenido |
|---|---|
| `src/diffusion/` | Paquete principal: `sde`, `models`, `samplers`, `training`, `data_generation`, `analytic` |
| `scripts/` | CLIs: entrenar, samplear, generar datos, partir el dataset, graficar la U-Net |
| `config/` | Configs YAML de cada corrida: `toy/` (mezcla 2D), `cats/` (imágenes), `test/` (smoke tests del pipeline) |
| `notebooks/` | Recorrido numerado `00`–`09`; `audit/` tiene los chequeos de validación |
| `tests/` | Tests con pytest |

Los datos (`data/`) y los checkpoints (`*.pt`) no se versionan.

## Uso

Todo se corre desde la raíz del repo, con [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group analysis                                       # instalar dependencias
uv run python scripts/train.py --config config/toy/vp_mixture.yaml   # entrenar
uv run python scripts/sample.py models/phase_1/vp_mixture.pt --sampler pf_ode --out data/muestras.npz
uv run pytest                                                  # tests
```

Las rutas de `data:` y `out:` en las configs son relativas a la raíz del repo.
