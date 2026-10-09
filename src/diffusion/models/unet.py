"""Red de score (U-Net) para imágenes — Fase 2.

Aproxima el score :math:`s_\\theta(x, t) \\approx \\nabla_x \\log p_t(x)` sobre tensores
imagen ``(B, C, H, W)`` con una U-Net condicionada en el tiempo. Igual que la
:class:`~diffusion.models.mlp.ScoreMLP` de Fase 1, es la **variable de control** del estudio
de ablación: su arquitectura e hiperparámetros quedan fijos en las 12 celdas SDE × sampler,
y la red es **enteramente determinística** — GroupNorm, sin dropout, batchnorm ni ninguna
capa estocástica. Toda la estocasticidad vive *fuera* de esta clase, en el marco de SDEs
(proceso forward, muestreo de pares de entrenamiento, sampler reverso).

Este archivo reúne los bloques privados de la U-Net (no se re-exportan del paquete: la
proyección temporal, el bloque residual convolucional, la atención y el cambio de
resolución) y la clase pública ``ScoreUNet``. El embedding de tiempo
:class:`~diffusion.models.layers.SinusoidalEmbedding` y el registry de activaciones son
compartidos entre redes y viven en :mod:`diffusion.models.layers`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SinusoidalEmbedding, _make_activation


class TimeMLP(nn.Module):
    """Proyección del tiempo al vector de condicionamiento de la U-Net.

    Embebe ``t`` con el :class:`~diffusion.models.layers.SinusoidalEmbedding`
    compartido y lo proyecta con un MLP de dos capas:
    ``SinusoidalEmbedding -> Linear -> activación -> Linear``. La salida
    ``(B, time_embed_dim)`` se computa **una sola vez** por forward y la
    comparten todos los bloques de la red (cada bloque la re-proyecta a sus
    canales).
    """

    def __init__(
        self,
        embed_dim: int,
        time_embed_dim: int,
        activation: str = "silu",
        time_scale: float = 1.0,
    ) -> None:
        """Inicializa la proyección temporal.

        Args:
            embed_dim: Dimensión del embedding sinusoidal de entrada (debe ser
                par; lo valida el embedding reusado).
            time_embed_dim: Dimensión del vector de condicionamiento de salida.
            activation: Nombre de la activación entre las dos lineales.
            time_scale: Factor de escala temporal; se pasa tal cual al
                :class:`~diffusion.models.layers.SinusoidalEmbedding` (que lo
                valida y lo aplica). Default ``1.0`` (comportamiento previo).

        Raises:
            ValueError: Si ``time_scale`` no es finito y positivo (la validación
                la hace el embedding al construirse).
        """
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.time_embed_dim = int(time_embed_dim)

        self.embed = SinusoidalEmbedding(embed_dim, scale=time_scale)
        #: Escala temporal (introspección): el valor ya validado por el embedding.
        self.time_scale = self.embed.scale
        self.lin1 = nn.Linear(embed_dim, time_embed_dim)
        self.act = _make_activation(activation)
        self.lin2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Proyecta el tiempo al vector de condicionamiento.

        Args:
            t: Tensor de tiempos de shape ``(B,)`` o ``(B, 1)`` (lo normaliza el
                embedding reusado), en cualquier escala usada por las SDEs del
                repo (``[0, 1]``, ``[0, T]`` o pasos enteros).

        Returns:
            Tensor de shape ``(B, time_embed_dim)``.
        """
        return self.lin2(self.act(self.lin1(self.embed(t))))


class ConvResBlock(nn.Module):
    """Bloque residual convolucional con inyección aditiva del tiempo.

    Análogo convolucional del :class:`~diffusion.models.mlp.ResidualBlock` lineal
    del MLP (comparten la idea, no el código — la pieza compartida vive en
    :mod:`~diffusion.models.layers`, no acá). El flujo es::

        GroupNorm -> activación -> Conv 3x3           # primera etapa
        + proyección temporal  (broadcast espacial)   # condicionamiento
        GroupNorm -> activación -> Conv 3x3           # segunda etapa
        + skip                                        # conexión residual

    El vector temporal ``t_emb`` (el ``(B, time_embed_dim)`` que produce
    :class:`TimeMLP`, compartido por todos los bloques del forward) se re-proyecta
    a los ``out_channels`` de este bloque con una lineal y se suma **tras la
    primera convolución**, expandido a ``(B, out_channels, 1, 1)`` para
    broadcastear sobre las dimensiones espaciales ``H`` y ``W``. Ese
    condicionamiento es lo que hace que dos tiempos distintos den salidas
    distintas.

    El skip es identidad cuando ``in_channels == out_channels``; si difieren, una
    convolución ``1x1`` proyecta la entrada a ``out_channels`` para poder sumarla.
    La normalización es :class:`~torch.nn.GroupNorm` (determinística e
    independiente del batch): **no** hay dropout ni batchnorm, en línea con la red
    como variable de control.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        groups: int = 8,
        activation: str = "silu",
    ) -> None:
        """Inicializa el bloque.

        Args:
            in_channels: Canales del tensor de entrada.
            out_channels: Canales del tensor de salida (y de la proyección
                temporal).
            time_embed_dim: Dimensión del vector temporal ``t_emb`` de entrada
                (la salida de :class:`TimeMLP`).
            groups: Grupos de las capas :class:`~torch.nn.GroupNorm`; debe dividir
                tanto a ``in_channels`` como a ``out_channels``.
            activation: Nombre de la activación (compartida con el resto del
                módulo vía :func:`~diffusion.models.layers._make_activation`).
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.time_embed_dim = int(time_embed_dim)

        # Primera etapa: normaliza la entrada, activa y convoluciona a out_channels.
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = _make_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Proyección del tiempo a los canales de salida (se suma tras conv1).
        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        # Segunda etapa: normaliza, activa y convoluciona manteniendo out_channels.
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = _make_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip: identidad si los canales coinciden, conv 1x1 si hay que reajustar.
        if in_channels == out_channels:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Aplica el bloque con inyección de tiempo y conexión residual.

        Args:
            x: Tensor de shape ``(B, in_channels, H, W)``.
            t_emb: Vector temporal de shape ``(B, time_embed_dim)`` (salida de
                :class:`TimeMLP`), compartido por todos los bloques del forward.

        Returns:
            Tensor de shape ``(B, out_channels, H, W)`` (misma resolución
            espacial que la entrada).
        """
        h = self.conv1(self.act1(self.norm1(x)))       # (B, out_channels, H, W)
        # Broadcast del tiempo sobre H y W: (B, out_channels) -> (B, out_channels, 1, 1).
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))        # (B, out_channels, H, W)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Auto-atención espacial *single-head* con conexión residual.

    Deja que cada posición espacial atienda a todas las demás del mapa de
    características, capturando dependencias de largo alcance que la convolución
    (con su campo receptivo local) no alcanza. En la U-Net se coloca en las
    resoluciones bajas (16×16 y el bottleneck), donde el número de tokens
    ``H·W`` es manejable. El flujo es::

        GroupNorm -> proyección QKV (conv 1x1) -> scaled_dot_product_attention
                  -> proyección de salida (conv 1x1) -> + skip

    La proyección QKV es una única convolución ``1x1`` que produce ``3·C``
    canales, partidos luego en las tres matrices ``Q``, ``K`` y ``V``; cada mapa
    ``(B, C, H, W)`` se aplana a ``(B, H·W, C)`` para tratar las ``H·W``
    posiciones como una secuencia de tokens de dimensión ``C`` (una sola
    cabeza). :func:`torch.nn.functional.scaled_dot_product_attention` calcula la
    atención escalada por ``1/sqrt(C)`` y es determinística en CPU (float32), en
    línea con la red como variable de control. La convolución ``1x1`` de salida
    reproyecta el resultado y se suma a la entrada (skip identidad), por lo que el
    bloque **preserva la shape** ``(B, C, H, W)``. La normalización es
    :class:`~torch.nn.GroupNorm`: no hay dropout ni batchnorm.
    """

    def __init__(self, channels: int, groups: int = 8) -> None:
        """Inicializa el bloque de atención.

        Args:
            channels: Canales del tensor de entrada y de salida (se conservan).
            groups: Grupos de la capa :class:`~torch.nn.GroupNorm`; debe dividir
                a ``channels``.
        """
        super().__init__()
        self.channels = int(channels)

        self.norm = nn.GroupNorm(groups, channels)
        # Proyección conjunta a Q, K, V (3 x channels), luego se parte en 3.
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica la auto-atención espacial con conexión residual.

        Args:
            x: Tensor de shape ``(B, channels, H, W)``.

        Returns:
            Tensor de shape ``(B, channels, H, W)`` (misma shape que la entrada).
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        # (B, 3C, H, W) -> tres mapas (B, C, H, W).
        q, k, v = self.to_qkv(h).chunk(3, dim=1)
        # Aplanar a tokens: (B, C, H, W) -> (B, H*W, C).
        q = q.reshape(B, C, H * W).transpose(1, 2)
        k = k.reshape(B, C, H * W).transpose(1, 2)
        v = v.reshape(B, C, H * W).transpose(1, 2)
        # Atención single-head sobre los H*W tokens (escala 1/sqrt(C) interna).
        attn = F.scaled_dot_product_attention(q, k, v)   # (B, H*W, C)
        # De vuelta al mapa espacial: (B, H*W, C) -> (B, C, H, W).
        attn = attn.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj_out(attn)


class Downsample(nn.Module):
    """Reducción espacial ×2 por convolución ``3x3`` con stride 2.

    Divide ``H`` y ``W`` por 2 aprendiendo el submuestreo (a diferencia de un
    pooling fijo), manteniendo el número de canales. Con ``kernel_size=3``,
    ``stride=2`` y ``padding=1``, una resolución par ``H`` pasa a ``H/2``.
    """

    def __init__(self, channels: int) -> None:
        """Inicializa la reducción.

        Args:
            channels: Canales del tensor de entrada y de salida (se conservan).
        """
        super().__init__()
        self.channels = int(channels)
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce la resolución espacial a la mitad.

        Args:
            x: Tensor de shape ``(B, channels, H, W)`` con ``H`` y ``W`` pares.

        Returns:
            Tensor de shape ``(B, channels, H // 2, W // 2)``.
        """
        return self.conv(x)


class Upsample(nn.Module):
    """Ampliación espacial ×2 por interpolación *nearest* + convolución ``3x3``.

    Duplica ``H`` y ``W`` con interpolación al vecino más cercano y luego suaviza
    el resultado con una convolución ``3x3`` (``padding=1``, preserva la
    resolución) que además aprende a atenuar los artefactos de bloque. Separar el
    reescalado de la convolución evita el *checkerboard* típico de la
    convolución transpuesta. El número de canales se conserva.
    """

    def __init__(self, channels: int) -> None:
        """Inicializa la ampliación.

        Args:
            channels: Canales del tensor de entrada y de salida (se conservan).
        """
        super().__init__()
        self.channels = int(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Amplía la resolución espacial al doble.

        Args:
            x: Tensor de shape ``(B, channels, H, W)``.

        Returns:
            Tensor de shape ``(B, channels, H * 2, W * 2)``.
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ScoreUNet(nn.Module):
    """Red de score convolucional (U-Net estilo DDPM) para imágenes — Fase 2.

    Aproxima el score :math:`s_\\theta(x, t) \\approx \\nabla_x \\log p_t(x)` sobre
    tensores imagen ``(B, C, H, W)`` con una U-Net condicionada en el tiempo:
    encoder → bottleneck → decoder con *skip connections* concatenadas por canales.
    Igual que la :class:`~diffusion.models.mlp.ScoreMLP` de Fase 1, es la **variable
    de control** del estudio de ablación: una vez fijada por los defaults del
    constructor (la arquitectura de referencia), queda idéntica en las 12 celdas
    SDE × sampler, y la red es **enteramente determinística** — GroupNorm, sin
    dropout, batchnorm ni ninguna capa estocástica. La salida final no lleva
    activación: el score es no acotado (puede ser positivo o negativo).

    El armado sigue el esquema estándar de las U-Nets de difusión (Ho et al. 2020;
    Song et al. 2021):

    - **Entrada**: convolución ``3x3`` de ``in_channels`` a ``base_channels``.
    - **Encoder**: un nivel por entrada de ``channel_mults``; el nivel ``i`` trabaja
      con ``base_channels * channel_mults[i]`` canales y ``num_res_blocks``
      :class:`ConvResBlock`, con :class:`AttentionBlock` en los niveles cuya
      resolución (``image_size / 2**i``) pertenece a ``attn_resolutions``. Entre
      niveles hay un :class:`Downsample` (``len(channel_mults) - 1`` en total). Cada
      activación del encoder se guarda como *skip*.
    - **Bottleneck**: ``ConvResBlock → AttentionBlock → ConvResBlock`` (la atención
      va **siempre** acá).
    - **Decoder**: espejo del encoder; antes de cada :class:`ConvResBlock` se
      **concatena por canales** el *skip* correspondiente del encoder, y se
      :class:`Upsample` al final de cada nivel para volver a la resolución de
      entrada.
    - **Salida**: ``GroupNorm → activación → Conv 3x3`` a ``in_channels``, sin
      activación final.

    El vector temporal lo produce :class:`TimeMLP` **una sola vez** por forward y lo
    comparten todos los :class:`ConvResBlock`. Satisface el contrato
    :class:`~diffusion.models.base.ScoreModel` estructuralmente (``(x, t) -> score``
    con la misma shape que ``x``), sin heredar de nada.

    Los defaults del constructor definen la arquitectura de referencia; con
    ``image_size`` 64 o 32 la misma config ``attn_resolutions=(16,)`` coloca atención
    en la resolución 16×16 (más el bottleneck).
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 64,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        embed_dim: int = 128,
        time_embed_dim: int = 256,
        attn_resolutions: tuple[int, ...] = (16,),
        groups: int = 8,
        activation: str = "silu",
        time_scale: float = 1.0,
    ) -> None:
        """Inicializa la red con la arquitectura de referencia por defecto.

        Args:
            in_channels: Canales de la imagen de entrada y de la salida (p. ej. 1
                para escala de grises, 3 para RGB).
            image_size: Resolución espacial de trabajo (``H == W``); fija las
                resoluciones por nivel (``image_size / 2**i``) y, con ellas, la
                colocación de la atención en construcción.
            base_channels: Canales del primer nivel (los demás son
                ``base_channels * channel_mults[i]``).
            channel_mults: Multiplicador de canales por nivel; su longitud es la
                cantidad de niveles del encoder/decoder.
            num_res_blocks: Cantidad de :class:`ConvResBlock` por nivel del encoder.
            embed_dim: Dimensión del embedding sinusoidal de tiempo (debe ser par;
                lo valida :class:`~diffusion.models.layers.SinusoidalEmbedding`).
            time_embed_dim: Dimensión del vector de condicionamiento que produce
                :class:`TimeMLP` y comparten los bloques.
            attn_resolutions: Resoluciones espaciales (absolutas) en las que se
                coloca :class:`AttentionBlock`; el bottleneck siempre la lleva.
            groups: Grupos de las capas :class:`~torch.nn.GroupNorm`; debe dividir a
                todos los anchos de canal.
            activation: Nombre de la activación, compartida por toda la red vía
                :func:`~diffusion.models.layers._make_activation`.
            time_scale: Factor de escala temporal; se pasa tal cual (vía
                :class:`TimeMLP`) al :class:`~diffusion.models.layers.SinusoidalEmbedding`
                compartido, que lo valida y lo aplica. Default ``1.0``
                (comportamiento previo).

        Raises:
            ValueError: Si ``image_size`` no es divisible por el factor total de
                reducción, si ``groups`` no divide a algún ancho de canal, o si
                ``time_scale`` no es finito y positivo (esta última la valida el
                embedding al construirse).
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(m) for m in channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.embed_dim = int(embed_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.attn_resolutions = tuple(int(r) for r in attn_resolutions)
        self.groups = int(groups)

        # --- Validaciones fail-fast (antes de construir sub-módulos, para cortar
        # con un mensaje propio y no con un error críptico de torch más abajo). ---
        # image_size debe soportar los len(channel_mults) - 1 Downsample (uno por
        # nivel salvo el último): cada uno divide la resolución por 2.
        reduction = 2 ** (len(self.channel_mults) - 1)
        if self.image_size % reduction != 0:
            raise ValueError(
                f"image_size debe ser divisible por {reduction} "
                f"(= 2**(len(channel_mults)-1), un Downsample por nivel salvo el "
                f"último); recibí image_size={self.image_size}"
            )
        # groups debe dividir a todos los anchos de canal base_channels*m que
        # normaliza GroupNorm; se reporta el primer nivel infractor y su ancho.
        for level, mult in enumerate(self.channel_mults):
            width = self.base_channels * mult
            if width % self.groups != 0:
                raise ValueError(
                    f"groups debe dividir a todos los anchos de canal; en el nivel "
                    f"{level} el ancho es base_channels*{mult}={width}, no divisible "
                    f"por groups={self.groups}"
                )

        num_levels = len(self.channel_mults)
        attn_set = set(self.attn_resolutions)

        # Vector temporal compartido: se computa una vez por forward y lo reusan
        # todos los ConvResBlock (cada uno lo re-proyecta a sus canales).
        self.time_mlp = TimeMLP(embed_dim, time_embed_dim, activation, time_scale)
        #: Escala temporal (introspección): el valor ya validado por el embedding.
        self.time_scale = self.time_mlp.time_scale

        # Convolución de entrada: lleva los in_channels a base_channels.
        self.conv_in = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )

        # --- Encoder: un nivel por multiplicador; se guardan los skips. ---
        # skip_channels lleva la cuenta de canales de cada activación guardada
        # (empezando por la salida de conv_in) para dimensionar el decoder.
        self.down_blocks = nn.ModuleList()
        ch = self.base_channels
        skip_channels: list[int] = [ch]
        for level, mult in enumerate(self.channel_mults):
            out_ch = self.base_channels * mult
            resolution = self.image_size // (2 ** level)
            use_attn = resolution in attn_set
            for _ in range(self.num_res_blocks):
                stage: list[nn.Module] = [
                    ConvResBlock(ch, out_ch, time_embed_dim, groups, activation)
                ]
                ch = out_ch
                if use_attn:
                    stage.append(AttentionBlock(ch, groups))
                self.down_blocks.append(nn.ModuleList(stage))
                skip_channels.append(ch)
            if level != num_levels - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)

        # --- Bottleneck: la atención va siempre acá. ---
        self.mid_block1 = ConvResBlock(ch, ch, time_embed_dim, groups, activation)
        self.mid_attn = AttentionBlock(ch, groups)
        self.mid_block2 = ConvResBlock(ch, ch, time_embed_dim, groups, activation)

        # --- Decoder: espejo del encoder; concatena el skip antes de cada bloque. ---
        # Cada nivel consume num_res_blocks + 1 skips (los num_res_blocks del nivel
        # más el del cambio de resolución / conv_in), y el ConvResBlock recibe
        # ch + skip_ch canales por la concatenación.
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(num_levels)):
            out_ch = self.base_channels * self.channel_mults[level]
            resolution = self.image_size // (2 ** level)
            use_attn = resolution in attn_set
            for i in range(self.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                stage = [
                    ConvResBlock(
                        ch + skip_ch, out_ch, time_embed_dim, groups, activation
                    )
                ]
                ch = out_ch
                if use_attn:
                    stage.append(AttentionBlock(ch, groups))
                if level != 0 and i == self.num_res_blocks:
                    stage.append(Upsample(ch))
                self.up_blocks.append(nn.ModuleList(stage))

        # --- Cabeza de salida: sin activación final (score no acotado). ---
        self.out_norm = nn.GroupNorm(groups, ch)
        self.out_act = _make_activation(activation)
        self.conv_out = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    @staticmethod
    def _apply_stage(
        stage: nn.ModuleList, x: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        """Aplica en orden las capas de una etapa (encoder o decoder).

        Solo los :class:`ConvResBlock` reciben el vector temporal; la atención y el
        cambio de resolución operan únicamente sobre el tensor espacial.

        Args:
            stage: Lista ordenada de capas de la etapa.
            x: Tensor de entrada de la etapa.
            t_emb: Vector temporal compartido ``(B, time_embed_dim)``.

        Returns:
            El tensor tras atravesar todas las capas de la etapa.
        """
        for layer in stage:
            if isinstance(layer, ConvResBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predice el score en ``(x, t)``.

        Args:
            x: Imagen ruidosa de shape ``(B, in_channels, image_size, image_size)``.
            t: Tiempo / nivel de ruido de shape ``(B,)`` o ``(B, 1)`` (lo normaliza
                el embedding reusado), en cualquier escala usada por las SDEs del
                repo.

        Returns:
            Score predicho de shape ``(B, in_channels, image_size, image_size)`` (la
            misma shape y dtype que ``x``), sin activación final.

        Raises:
            ValueError: Si ``x`` no tiene alto y ancho iguales a ``image_size`` (la
                arquitectura quedó fijada en construcción) o si su número de canales
                no coincide con ``in_channels``.
        """
        # --- Validaciones fail-fast del tensor de entrada. ---
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            raise ValueError(
                f"x debe tener alto y ancho iguales a image_size={self.image_size} "
                f"(la arquitectura quedó fijada en construcción); recibí "
                f"(H, W)=({x.shape[-2]}, {x.shape[-1]})"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"x debe tener in_channels={self.in_channels} canales; "
                f"recibí {x.shape[1]}"
            )

        t_emb = self.time_mlp(t)               # (B, time_embed_dim)

        # Encoder: guarda cada activación como skip (empezando por conv_in).
        h = self.conv_in(x)
        skips: list[torch.Tensor] = [h]
        for stage in self.down_blocks:
            h = self._apply_stage(stage, h, t_emb)
            skips.append(h)

        # Bottleneck.
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder: concatena el skip por canales antes de cada etapa.
        for stage in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = self._apply_stage(stage, h, t_emb)

        # Cabeza de salida sin activación final.
        h = self.out_act(self.out_norm(h))
        return self.conv_out(h)


if __name__ == "__main__":
    # Smoke test manual: instancia la red, corre un forward y reporta tamaños.
    # Correr (con diffusion-models/src en PYTHONPATH): python -m diffusion.models.unet
    # (los imports relativos exigen ejecutarlo como módulo, no como script suelto).
    net = ScoreUNet()
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    out = net(x, t)
    print(f"ScoreUNet(in_channels=3, image_size=64): salida {tuple(out.shape)}")

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")
