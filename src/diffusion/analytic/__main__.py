"""Smoke test manual del módulo: score contra autograd y masa unitaria, en CPU.

Correr (desde ``diffusion-models/src/``)::

    python -m diffusion.analytic

Para cada SDE escalar del Eje 1 (VP, VE y sub-VP) construye el oráculo sobre una mixtura
exacta chica —pesos desbalanceados y una componente rotada— y verifica las dos afirmaciones
que sostienen todo el módulo:

1. que el score en **forma cerrada** coincide con el gradiente de ``log_prob`` calculado por
   ``torch.autograd``, que es una segunda cuenta independiente sobre la misma log-densidad; y
2. que la densidad **integra a uno** sobre el plano por cuadratura, que es el autochequeo de
   normalización.

Todo corre en CPU y en doble precisión, así que el desacuerdo esperable es del orden del
epsilon de máquina y la masa da uno con error de cuadratura. Se usa ``-m`` porque el módulo
usa imports relativos y no es ejecutable como script suelto.
"""

from __future__ import annotations

import math

import torch

from diffusion.analytic import MixtureOracle
from diffusion.data_generation import ExactGaussianMixture
from diffusion.sde import make_sde

_SDES = ("vp", "ve", "sub_vp")
_TIEMPOS = (0.05, 0.5, 1.0)
_N_PUNTOS = 256


def _mixtura() -> ExactGaussianMixture:
    """Mixtura exacta chica: pesos desbalanceados y una componente rotada y estirada."""
    angulo = math.pi / 3.0
    rotacion = torch.tensor(
        [[math.cos(angulo), -math.sin(angulo)], [math.sin(angulo), math.cos(angulo)]],
        dtype=torch.float64,
    )
    escalas = torch.diag(torch.tensor([1.0, 0.09], dtype=torch.float64))
    rotada = rotacion @ escalas @ rotacion.T
    return ExactGaussianMixture(
        2,
        weights=[0.3, 0.7],
        means=[[-1.5, 0.5], [2.0, -1.0]],
        covariances=[rotada.tolist(), [[0.25, 0.0], [0.0, 0.04]]],
        seed=0,
    )


def _discrepancia_contra_autograd(
    oraculo: MixtureOracle, x: torch.Tensor, instante: float
) -> float:
    """Máxima diferencia absoluta entre el score cerrado y el gradiente por autograd.

    Args:
        oraculo: Oráculo ya construido sobre una mixtura y una SDE.
        x: Puntos del plano, de forma ``(B, 2)``, en doble precisión.
        instante: Tiempo escalar en el que se evalúan las dos cuentas.

    Returns:
        El desacuerdo máximo sobre las ``2B`` componentes, en valor absoluto.
    """
    tiempos = torch.full((x.shape[0],), instante, dtype=x.dtype)
    puntos = x.clone().requires_grad_(True)
    (gradiente,) = torch.autograd.grad(oraculo.log_prob(puntos, tiempos).sum(), puntos)
    cerrado = oraculo.score(x, tiempos)
    return float((cerrado - gradiente).abs().max())


def main() -> dict[str, tuple[float, float]]:
    """Corre el doble chequeo del oráculo sobre las tres SDEs y reporta el resultado.

    Imprime, por SDE y por tiempo, el desacuerdo entre el score en forma cerrada y el
    gradiente por autograd, y la masa integrada de la densidad. Los tiempos barren el rango
    útil, incluido uno chico donde ``σ_t`` es del orden de ``1e-2`` y el score es grande.

    Returns:
        Un resumen ``{nombre_sde: (discrepancia_maxima, masa_mas_alejada_de_uno)}`` con el
        peor caso de cada SDE sobre todos los tiempos, pensado para que el smoke sea
        assertable sin parsear stdout.
    """
    mixtura = _mixtura()
    x = torch.randn(
        _N_PUNTOS, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )

    resumen: dict[str, tuple[float, float]] = {}
    for nombre in _SDES:
        oraculo = MixtureOracle(mixtura, make_sde(nombre))
        peor_discrepancia = 0.0
        peor_masa = 1.0
        for instante in _TIEMPOS:
            discrepancia = _discrepancia_contra_autograd(oraculo, x, instante)
            masa = oraculo.total_mass(instante)
            peor_discrepancia = max(peor_discrepancia, discrepancia)
            if abs(masa - 1.0) > abs(peor_masa - 1.0):
                peor_masa = masa
            # Solo ASCII en stdout: la consola de Windows usa cp1252 y no encodea
            # los simbolos matematicos que si aparecen en los docstrings.
            print(
                f"{nombre:7s} t={instante:<5.3g} score vs autograd: "
                f"max|dif|={discrepancia:.3e}   masa=int p_t={masa:.12f}"
            )
        resumen[nombre] = (peor_discrepancia, peor_masa)
    return resumen


if __name__ == "__main__":
    main()
