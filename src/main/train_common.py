import os
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from src.training.trainer import RLTrainer, SupervisedTrainer, TrainerConfig
from src.training.utils import (
    ModelName,
    ProblemName,
    build_loader,
    build_model,
    count_trainable_parameters,
    default_target_algorithm,
    resolve_device,
    set_seed,
)

ModeName = Literal["supervised", "rl"]


def run_pipeline(cfg: DictConfig) -> None:
    """Run one training job from a Hydra config."""

    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise TypeError("Hydra config must resolve to a mapping")

    model_cfg = _as_dict(resolved["model"])
    data_cfg = _as_dict(resolved["data"])
    paths_cfg = _as_dict(resolved["paths"])
    trainer_cfg = _as_dict(resolved["trainer"])

    model_name = _as_model_name(model_cfg["name"])
    problem = _as_problem_name(resolved["problem"])
    mode = _as_mode_name(resolved["mode"])

    set_seed(int(resolved["seed"]))
    device = resolve_device(str(resolved["device"]))

    target_algorithm = data_cfg.get("target_algorithm")
    if mode == "supervised" and target_algorithm is None:
        target_algorithm = default_target_algorithm(problem)

    train_loader = build_loader(
        problem=problem,
        path=str(paths_cfg["train"]),
        batch_size=int(data_cfg["batch_size"]),
        target_algorithm=str(target_algorithm) if mode == "supervised" else None,
        shuffle=bool(data_cfg["shuffle"]),
        num_workers=int(data_cfg["num_workers"]),
    )

    val_loader = None
    if paths_cfg.get("val") is not None:
        val_loader = build_loader(
            problem=problem,
            path=str(paths_cfg["val"]),
            batch_size=int(data_cfg["eval_batch_size"]),
            target_algorithm=str(target_algorithm) if mode == "supervised" else None,
            shuffle=False,
            num_workers=int(data_cfg["num_workers"]),
        )

    model = build_model(model_name, problem, model_options=model_cfg)
    print(
        f"model={model_name} problem={problem} mode={mode} "
        f"parameters={count_trainable_parameters(model)} device={device}"
    )

    output_dir = paths_cfg.get("output_dir") or os.path.join(
        "outputs",
        model_name,
        problem,
        mode,
    )
    config = TrainerConfig(
        problem=problem,
        output_dir=str(output_dir),
        n_epochs=int(trainer_cfg["epochs"]),
        steps_per_epoch=trainer_cfg.get("steps_per_epoch"),
        learning_rate=float(trainer_cfg["learning_rate"]),
        learning_rate_decay=float(trainer_cfg["learning_rate_decay"]),
        max_grad_norm=float(trainer_cfg["max_grad_norm"]),
        log_every=int(trainer_cfg["log_every"]),
        checkpoint_every=int(trainer_cfg["checkpoint_every"]),
        optimizer=str(trainer_cfg["optimizer"]),
        baseline=str(trainer_cfg["baseline"]),
        baseline_alpha=float(trainer_cfg["baseline_alpha"]),
        baseline_warmup_epochs=int(trainer_cfg["baseline_warmup_epochs"]),
        exp_baseline_beta=float(trainer_cfg["exp_baseline_beta"]),
    )

    trainer_cls = SupervisedTrainer if mode == "supervised" else RLTrainer
    trainer = trainer_cls(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    trainer.fit()


def _as_dict(value: object) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping config section, got {type(value).__name__}")
    return value


def _as_model_name(value: object) -> ModelName:
    if value not in ("am", "pn"):
        raise ValueError("model.name must be 'am' or 'pn'")
    return value


def _as_problem_name(value: object) -> ProblemName:
    if value not in ("tsp", "mis"):
        raise ValueError("problem must be 'tsp' or 'mis'")
    return value


def _as_mode_name(value: object) -> ModeName:
    if value not in ("supervised", "rl"):
        raise ValueError("mode must be 'supervised' or 'rl'")
    return value
