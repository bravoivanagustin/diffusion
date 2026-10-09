"""Verdad analítica del laboratorio 2D: densidad, score y sesgo exactos.

Este módulo es el **patrón de referencia** contra el que se mide el resto del estudio.
Sobre una mixtura de gaussianas 2D de parámetros conocidos en forma cerrada calcula
``p_t(x)``, ``log p_t(x)`` y ``∇_x log p_t(x)`` **exactos** para las tres SDEs escalares
del Eje 1 (VP, VE y sub-VP), más el sesgo de inicialización de cada una. Con eso el error
de estimación del score se puede *apagar* y separar del error de discretización y del de
truncación.

Acoplamiento con el pipeline:

- **Consume** el contrato marginal ya publicado por :mod:`diffusion.sde`
  (``marginal_prob`` da ``α_t`` y ``σ_t``, y ``T`` el horizonte) y los parámetros
  verdaderos de la mixtura exacta de :mod:`diffusion.data_generation`. No re-deriva los
  schedules ni ramifica por variante: un único camino de código cubre las tres SDEs.
- **No importa** :mod:`diffusion.samplers`: la relación es por contrato de callable
  ``(x, t) -> score``, el mismo que los cuatro samplers ya aceptan de una red entrenada,
  así que samplear con score exacto no requiere tocarlos. La dirección obligatoria de
  dependencias es ``data_generation → sde → analytic``.

``torch`` es dependencia dura del módulo (como en :mod:`diffusion.sde`), no diferida.

Uso típico::

    from diffusion.analytic import MixtureOracle, auto_grid, integrate

    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    score = oraculo(x, t)                 # el mismo contrato que consumen los samplers
    masa = oraculo.total_mass(0.5)        # autochequeo por cuadratura: debe dar 1

Los cinco nombres se resuelven **de forma diferida** (PEP 562): el ``__init__`` no importa
los submódulos al cargarse, sino la primera vez que se pide el atributo. No es cosmético.
Importar cualquier submódulo ejecuta este ``__init__``, así que un reexport directo del
oráculo acoplaría ``diffusion.analytic.quadrature`` a :mod:`diffusion.sde` y a
:mod:`diffusion.data_generation` por la puerta de atrás, y la cuadratura dejaría de ser la
utilidad numérica autónoma que dice ser (solo ``torch`` y la biblioteca estándar). Con la
resolución diferida la ruta corta ``from diffusion.analytic import MixtureOracle`` sigue
funcionando y la garantía de aislamiento se mantiene: hay un test que la fija.

Nota: el oráculo opera **solo** sobre mixturas de parámetros exactos. Los modelos
entrenados sobre datos con estandarización empírica no quedan descritos por él, porque en
ese caso los parámetros efectivos dependen del sorteo de la muestra.

Correr ``python -m diffusion.analytic`` (desde ``diffusion-models/src/``) ejercita el módulo
de punta a punta: contrasta el score en forma cerrada contra el gradiente de la log-densidad
por autograd e integra la densidad para verificar que la masa es uno.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # solo para editores y type checkers: en runtime nada de esto se ejecuta
    from .mixture_oracle import BiasReport, MixtureOracle
    from .quadrature import QuadratureGrid, auto_grid, integrate

__all__ = [
    "BiasReport",
    "MixtureOracle",
    "QuadratureGrid",
    "auto_grid",
    "integrate",
]

# Nombre público -> submódulo que lo define. Es el mapa que consulta __getattr__ para
# resolver el import recién cuando alguien pide el nombre.
_EXPORTS: dict[str, str] = {
    "BiasReport": "mixture_oracle",
    "MixtureOracle": "mixture_oracle",
    "QuadratureGrid": "quadrature",
    "auto_grid": "quadrature",
    "integrate": "quadrature",
}


def __getattr__(name: str) -> Any:
    """Resuelve un nombre público importando su submódulo la primera vez que se pide.

    Args:
        name: Atributo pedido sobre el paquete.

    Returns:
        El objeto exportado, ya cacheado en el namespace del paquete para que las consultas
        siguientes no vuelvan a pasar por acá.

    Raises:
        AttributeError: Si el nombre no es parte de la API pública, con el mismo mensaje que
            daría un módulo sin resolución diferida.
    """
    submodulo = _EXPORTS.get(name)
    if submodulo is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    valor = getattr(import_module(f".{submodulo}", __name__), name)
    globals()[name] = valor
    return valor


def __dir__() -> list[str]:
    """Lista la API pública además de lo ya resuelto, para autocompletado e introspección."""
    return sorted(set(globals()) | set(__all__))
