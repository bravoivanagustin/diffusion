"""Smoke test manual del módulo: entrena pocos pasos y reporta la pérdida.

Correr (desde ``diffusion-models/src/``)::

    python -m diffusion.training

Para cada SDE construye una red chica y entrena unos pocos pasos sobre una mezcla de gaussianas
de juguete en CPU, e imprime la pérdida inicial y final (debería bajar). Se usa ``-m`` porque el
módulo usa imports relativos y no es ejecutable como script suelto.
"""

from __future__ import annotations

from diffusion.data_generation import infinite_bare, make_distribution
from diffusion.models import ScoreMLP
from diffusion.sde import available_sdes, make_sde
from diffusion.training import TrainConfig, train


def main() -> None:
    dist = make_distribution("mixture", dim=2, n_components=8, seed=0)
    # num_steps ≈ epochs × (n_samples / batch_size) = 40 × (512 / 128) = 160 (corrida vieja).
    config = TrainConfig(num_steps=160, lr=2e-3, seed=0)
    for name in available_sdes():
        sde = make_sde(name)
        net = ScoreMLP(data_dim=sde.data_dim, hidden_dim=64, num_blocks=2)
        data = infinite_bare(dist.dataloader(512, 128, shuffle=True))
        result = train(sde, net, data, config)
        hist = result.history
        k = max(1, len(hist) // 10)  # medias de extremos: la pérdida per-step es ruidosa
        ini, fin = sum(hist[:k]) / k, sum(hist[-k:]) / k
        print(
            f"{name:7s} data_dim={sde.data_dim}  "
            f"pérdida inicial~{ini:.4f} -> final~{fin:.4f}"
        )


if __name__ == "__main__":
    main()
