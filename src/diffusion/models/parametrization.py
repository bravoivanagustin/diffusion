"""Parametrización ε del score: el wrapper que divide la predicción interna por σ_t.

Spec ``small-t-training-signal`` (eje 2). A ``t`` chico la magnitud del score real crece como
``1/σ_t`` (con dato tipo delta la red debería emitir valores ~100–300, y no llega). La
parametrización estándar de referencia (Song et al., ICLR 2021 — ``get_score_fn`` para VP
continuo; equivalente a la predicción de ε de DDPM) resuelve eso por álgebra: la red interna
predice una cantidad **acotada** (unidades de ε) y el score consumido se computa como

    score(x, t) = -inner(x, t) / clamp(σ_t(x, t), 1e-5)

Así la magnitud ``1/σ_t`` sale de la matemática y no de la capacidad de la red. Con el pesado
vigente de la pérdida (``λ(t) = σ²``, ver :meth:`diffusion.sde.base.ForwardSDE.score_target`),
la regresión interna queda automáticamente en unidades de ε — no hace falta tocar la pérdida.

Dos decisiones de frontera del diseño:

- **La σ llega como callable opaco** ``(x, t) -> std`` broadcastable ``(B, 1, …, 1)``: este
  módulo **no** importa ``diffusion.sde`` (la dirección de dependencias queda intacta). El
  callable lo construyen los call sites desde la SDE que ya poseen, p. ej.
  ``lambda x, t: sde.marginal_prob(x, t)[1]``.
- **``state_dict``/``load_state_dict`` delegan al interno, sin prefijo**: el checkpoint sigue
  siendo de red pelada, reconstruible con :func:`diffusion.models.make_model` desde la receta
  ``model_spec`` existente. Sin esta delegación transparente, las claves saldrían con el
  prefijo del submódulo (``_net.``) y el round-trip por checkpoint (entrenar envuelto →
  reconstruir por factory) se rompería.

Uso típico::

    from diffusion.models import EpsilonScoreWrapper

    net = ScoreMLP(data_dim=2)
    wrapper = EpsilonScoreWrapper(net, lambda x, t: sde.marginal_prob(x, t)[1])
    score = wrapper(x, t)               # == -net(x, t) / clamp(σ_t, 1e-5)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .base import ScoreModel

#: Piso numérico para σ antes de dividir — la misma constante ``1e-5`` que usa
#: ``ForwardSDE._std_eps`` (ver ``.kiro/steering/numerics.md``): evita la división por
#: cero cuando ``t → 0`` y ``σ_t → 0``.
_SIGMA_FLOOR: float = 1e-5


class EpsilonScoreWrapper(nn.Module):
    """Score = ``-inner(x, t) / clamp(σ(x, t), 1e-5)`` con el contrato ``(x, t) -> score``.

    Envuelve una red que predice en unidades de ε (acotada) y expone el mismo contrato
    :class:`~diffusion.models.base.ScoreModel` que una red pelada, así la pérdida y los
    cuatro samplers la consumen sin cambios. El interno se registra como submódulo, de
    modo que ``.to()``, ``.train()``, ``.eval()`` y ``.parameters()`` delegan naturalmente.

    El wrapper es **enteramente determinístico** y **sin parámetros propios**: no agrega
    capas, tensores entrenables ni fuentes de aleatoriedad — solo la división por σ.

    ``state_dict()`` y ``load_state_dict()`` están **sobreescritos para delegar al
    interno**: devuelven/aceptan las claves de la red pelada, sin el prefijo ``_net.`` que
    agregaría la delegación default de ``nn.Module``. Es deliberado y es lo que hace
    posible el round-trip del checkpoint: lo que se persiste al entrenar envuelto es un
    checkpoint de red pelada, que ``make_model(receta)`` + ``load_state_dict`` reconstruye
    igual que siempre (y un checkpoint viejo carga en un wrapper nuevo sin traducción).

    Attributes:
        parametrization: Nombre de la parametrización, siempre ``"epsilon"`` (los call
            sites lo persisten en la receta del checkpoint para reconstruir el wrap).
    """

    def __init__(
        self,
        net: ScoreModel,
        marginal_std: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """Inicializa el wrapper.

        Args:
            net: Red interna que satisface :class:`ScoreModel`; con la parametrización
                activa predice en unidades de ε (acotada), no el score.
            marginal_std: Callable opaco ``(x, t) -> std`` con ``std`` positivo y
                broadcastable ``(B, 1, …, 1)`` contra ``x`` (típicamente
                ``lambda x, t: sde.marginal_prob(x, t)[1]``). Este módulo no conoce la
                SDE: solo evalúa el callable.
        """
        super().__init__()
        self._net = net  # submódulo: .to/.train/.eval/.parameters delegan solos
        self._marginal_std = marginal_std
        self.parametrization = "epsilon"

    @property
    def inner(self) -> ScoreModel:
        """La red interna envuelta (la que predice en unidades de ε)."""
        return self._net

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computa el score dividiendo la predicción interna por la σ clampeada.

        Args:
            x: Dato ruidoso de shape ``(B, *E)``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Score de la misma shape que ``x``:
            ``-inner(x, t) / clamp(marginal_std(x, t), 1e-5)``.
        """
        std = self._marginal_std(x, t)
        return -self._net(x, t) / std.clamp_min(_SIGMA_FLOOR)

    # ------------------------------------------------- delegación transparente
    # El checkpoint debe seguir siendo de red pelada (round-trip por make_model):
    # sin estos overrides, nn.Module serializaría las claves con prefijo "_net.".

    def state_dict(self, *args, **kwargs):
        """State dict del interno, con sus claves peladas (sin prefijo ``_net.``)."""
        return self._net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Carga en el interno un state dict de red pelada (claves sin prefijo)."""
        return self._net.load_state_dict(state_dict, strict=strict, assign=assign)
