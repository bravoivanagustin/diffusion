"""Smoke test manual del módulo: corre los 4 samplers sobre una red sin entrenar.

Correr (desde ``diffusion-models/src/``)::

    python -m diffusion.samplers

Recorre el registry (``available_samplers``) y, para cada sampler, lo instancia sobre una
``ScoreMLP`` **sin entrenar** y una VP-SDE, genera un batch chico e imprime el shape, la
media y un chequeo de finitud de la salida. No valida la calidad de las muestras (la red no
está entrenada): solo que la cadena ``make_sampler -> sample`` corre sin error y produce
salidas finitas sobre las SDEs escalares. Se usa ``-m`` porque el módulo usa imports
relativos y no es ejecutable como script suelto.
"""

from __future__ import annotations

import torch

from diffusion.models import ScoreMLP
from diffusion.samplers import available_samplers, make_sampler
from diffusion.sde import make_sde


def main() -> dict[str, bool]:
    """Corre cada sampler del registry sobre una red sin entrenar y reporta finitud.

    Construye una VP-SDE escalar y una ``ScoreMLP(data_dim=2)`` sin entrenar, la consume como
    función pura ``(x, t) -> score`` y, para cada sampler disponible, genera un batch chico en
    CPU con pocos pasos. Imprime nombre, shape, media y chequeo de finitud por sampler.

    Returns:
        Un resumen ``{nombre_sampler: salida_finita}`` con los cuatro samplers del registry,
        pensado para que el smoke sea assertable sin parsear stdout.
    """
    sde = make_sde("vp")
    net = ScoreMLP(data_dim=2)  # sin entrenar: solo ejercitamos la cadena, no la calidad
    net.eval()

    def score_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return net(x, t)

    n_samples = 64
    summary: dict[str, bool] = {}
    for name in available_samplers():
        sampler = make_sampler(name, sde, score_fn, n_steps=20)
        x0 = sampler.sample(n_samples, generator=torch.Generator().manual_seed(0))
        is_finite = bool(torch.all(torch.isfinite(x0)))
        summary[name] = is_finite
        print(
            f"{name:7s} -> {tuple(x0.shape)}  |mean|~{x0.abs().mean():.3f}  "
            f"finito={is_finite}"
        )
    return summary


if __name__ == "__main__":
    main()
