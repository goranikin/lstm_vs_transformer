import argparse
import os
from typing import Literal

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


def run_pipeline(
    model_name: ModelName,
    problem: ProblemName,
    mode: ModeName,
) -> None:
    args = _build_parser(model_name, problem, mode).parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    target_algorithm = args.target_algorithm
    if mode == "supervised" and target_algorithm is None:
        target_algorithm = default_target_algorithm(problem)

    train_loader = build_loader(
        problem=problem,
        path=args.train_path,
        batch_size=args.batch_size or _default_batch_size(model_name, mode),
        target_algorithm=target_algorithm if mode == "supervised" else None,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
    )
    val_loader = None
    if args.val_path is not None:
        val_loader = build_loader(
            problem=problem,
            path=args.val_path,
            batch_size=args.eval_batch_size,
            target_algorithm=target_algorithm if mode == "supervised" else None,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = build_model(model_name, problem)
    print(
        f"model={model_name} problem={problem} mode={mode} "
        f"parameters={count_trainable_parameters(model)} device={device}"
    )

    config = TrainerConfig(
        problem=problem,
        output_dir=args.output_dir
        or os.path.join("outputs", model_name, problem, mode),
        n_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate
        if args.learning_rate is not None
        else _default_learning_rate(model_name, mode),
        learning_rate_decay=args.learning_rate_decay
        if args.learning_rate_decay is not None
        else _default_learning_rate_decay(model_name, mode),
        max_grad_norm=args.max_grad_norm
        if args.max_grad_norm is not None
        else _default_max_grad_norm(model_name, mode),
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        optimizer=args.optimizer or _default_optimizer(model_name, mode),
        baseline=args.baseline,
        baseline_alpha=args.baseline_alpha,
        baseline_warmup_epochs=args.baseline_warmup_epochs,
        exp_baseline_beta=args.exp_baseline_beta,
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


def _build_parser(
    model_name: ModelName,
    problem: ProblemName,
    mode: ModeName,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Train {model_name.upper()} on {problem.upper()} with {mode}"
    )
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--target-algorithm", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-decay", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default=None)
    parser.add_argument("--baseline", choices=("rollout", "exponential"), default="rollout")
    parser.add_argument("--baseline-alpha", type=float, default=0.05)
    parser.add_argument("--baseline-warmup-epochs", type=int, default=1)
    parser.add_argument("--exp-baseline-beta", type=float, default=0.8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser


def _default_batch_size(model_name: ModelName, mode: ModeName) -> int:
    if model_name == "pn" and mode == "supervised":
        return 128
    return 512


def _default_learning_rate(model_name: ModelName, mode: ModeName) -> float:
    if model_name == "pn" and mode == "supervised":
        return 1.0
    if model_name == "pn" and mode == "rl":
        return 1e-3
    return 1e-4


def _default_learning_rate_decay(model_name: ModelName, mode: ModeName) -> float:
    if model_name == "pn" and mode == "rl":
        return 0.96
    return 1.0


def _default_max_grad_norm(model_name: ModelName, mode: ModeName) -> float:
    if model_name == "pn" and mode == "supervised":
        return 2.0
    return 1.0


def _default_optimizer(model_name: ModelName, mode: ModeName) -> str:
    if model_name == "pn" and mode == "supervised":
        return "sgd"
    return "adam"
