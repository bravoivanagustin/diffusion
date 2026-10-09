"""Loop de entrenamiento por denoising score matching (DSM).

El cuarto módulo del TP: el eslabón que une ``data_generation`` (los ``x_0``), ``models`` (la
red de score) y ``sde`` (el proceso forward). Enseña a una :class:`diffusion.models.ScoreModel`
cualquiera a aproximar ``s_θ(x, t) ≈ ∇_x log p_t(x)`` para una SDE dada, minimizando la pérdida
de DSM. :func:`train` es agnóstico a la red y a los datos: el caller construye la red y la
fuente infinita de tensores; el loop corre por pasos.

Uso típico (a mano)::

    from diffusion.sde import make_sde
    from diffusion.data_generation import infinite_bare, make_distribution
    from diffusion.models import ScoreMLP
    from diffusion.training import TrainConfig, train

    sde = make_sde("vp")
    net = ScoreMLP(data_dim=sde.data_dim)
    dist = make_distribution("mixture", dim=2, n_components=8, seed=0)
    data = infinite_bare(dist.dataloader(4000, 256, shuffle=True))
    result = train(sde, net, data, TrainConfig(num_steps=4000))

Uso típico (config-driven, una celda del estudio por archivo)::

    from diffusion.training import load_config, build_run, train, save_checkpoint
    spec = build_run(load_config("config/toy/vp_mixture.yaml"))
    result = train(spec.sde, spec.model, spec.data, spec.config)
    save_checkpoint(result, spec.checkpoint)
"""

from __future__ import annotations

from .config import DataSources, RunSpec, build_data_source, build_run, load_config
from .ema import EmaShadow
from .losses import dsm_loss, sample_timesteps
from .time_sampling import (
    LogUniformTimeSampler,
    TimeSampler,
    UniformTimeSampler,
    available_time_samplers,
    make_time_sampler,
)
from .resume import (
    ResumePlan,
    discover_snapshots,
    load_resume,
    prune_snapshots,
    resolve_resume,
    resume_sidecar_path,
    validate_compatible,
)
from .validation import VAL_EXAM_SEED, FixedValExam, ValPoint, evaluate_with_weights
from .trainer import (
    ResumeState,
    TrainConfig,
    TrainResult,
    TrainSnapshot,
    load_checkpoint,
    load_resume_state,
    save_checkpoint,
    save_resume_state,
    train,
)

__all__ = [
    "dsm_loss",
    "sample_timesteps",
    "TimeSampler",
    "UniformTimeSampler",
    "LogUniformTimeSampler",
    "make_time_sampler",
    "available_time_samplers",
    "EmaShadow",
    "FixedValExam",
    "evaluate_with_weights",
    "ValPoint",
    "VAL_EXAM_SEED",
    "TrainConfig",
    "TrainResult",
    "train",
    "save_checkpoint",
    "load_checkpoint",
    "ResumeState",
    "TrainSnapshot",
    "save_resume_state",
    "load_resume_state",
    "ResumePlan",
    "resolve_resume",
    "discover_snapshots",
    "prune_snapshots",
    "resume_sidecar_path",
    "validate_compatible",
    "load_resume",
    "RunSpec",
    "DataSources",
    "load_config",
    "build_run",
    "build_data_source",
]
