import argparse
from collections.abc import Iterator
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data_generating.common import instance_seed, write_jsonl
from src.data_generating.TSP.algorithms import solve_with_algorithms


class TSPGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_instances: int = Field(gt=0)
    num_nodes: int = Field(gt=0)
    seed: int
    output_path: str
    solvers: tuple[str, ...] = ("concorde", "lkh3")
    concorde_executable: str | None = None
    lkh3_executable: str | None = None
    lkh3_trials: int = Field(default=1000, gt=0)
    lkh3_runs: int = Field(default=10, gt=0)
    solver_timeout_sec: float | None = None
    skip_solvers: bool = False

    @model_validator(mode="after")
    def validate_solvers(self) -> Self:
        if not self.skip_solvers and not self.solvers:
            raise ValueError("at least one solver must be requested")
        return self


def generate_tsp_instance(num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((num_nodes, 2), dtype=np.float64)


def iter_tsp_records(config: TSPGenerationConfig) -> Iterator[dict[str, Any]]:
    for index in range(config.num_instances):
        seed = instance_seed(config.seed, index)
        coords = generate_tsp_instance(config.num_nodes, seed)
        record: dict[str, Any] = {
            "problem": "tsp",
            "index": index,
            "seed": seed,
            "num_nodes": config.num_nodes,
            "coordinates": coords.tolist(),
        }
        if not config.skip_solvers:
            record["solutions"] = solve_with_algorithms(
                coords,
                algorithms=config.solvers,
                concorde_executable=config.concorde_executable,
                lkh3_executable=config.lkh3_executable,
                lkh3_trials=config.lkh3_trials,
                lkh3_runs=config.lkh3_runs,
                seed=seed,
                timeout_sec=config.solver_timeout_sec,
            )
        yield record


def generate_tsp_dataset(config: TSPGenerationConfig) -> int:
    return write_jsonl(iter_tsp_records(config), config.output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic TSP JSONL data"
    )
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--solvers",
        type=str,
        default="concorde,lkh3",
        help="Comma-separated solver labels. Default: concorde,lkh3",
    )
    parser.add_argument("--concorde-executable", type=str, default=None)
    parser.add_argument("--lkh3-executable", type=str, default=None)
    parser.add_argument("--lkh3-trials", type=int, default=1000)
    parser.add_argument("--lkh3-runs", type=int, default=10)
    parser.add_argument("--solver-timeout-sec", type=float, default=None)
    parser.add_argument(
        "--skip-solvers",
        action="store_true",
        help="Generate instances only, without solver labels.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = TSPGenerationConfig(
        num_instances=args.num_instances,
        num_nodes=args.num_nodes,
        seed=args.seed,
        output_path=args.output_path,
        solvers=tuple(s.strip() for s in args.solvers.split(",") if s.strip()),
        concorde_executable=args.concorde_executable,
        lkh3_executable=args.lkh3_executable,
        lkh3_trials=args.lkh3_trials,
        lkh3_runs=args.lkh3_runs,
        solver_timeout_sec=args.solver_timeout_sec,
        skip_solvers=args.skip_solvers,
    )
    written = generate_tsp_dataset(config)
    print(f"Wrote {written} TSP instances to {config.output_path}")


if __name__ == "__main__":
    main()
