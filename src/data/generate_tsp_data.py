"""Generate TSP instances with optimal tours using Concorde or LKH.

Usage:
    uv run python -m src.data.generate_tsp_data
    uv run python -m src.data.generate_tsp_data --min-nodes 50 --max-nodes 50 --num-samples 1280 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import warnings
from typing import Any

import numpy as np

from src.data.generation.runner import log_generation_summary, run_batch_generation
from src.data.generation.tsp_solver import _init_worker, solve_tsp
from configs.tsp_config import TspGenerationConfig
from src.data.generation.types import TspInstance, TspSample, TspTour

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _build_tsp_batch(config: TspGenerationConfig) -> list[np.ndarray]:
    num_nodes = config.sample_num_nodes()
    return [np.random.random((num_nodes, 2)) for _ in range(config.batch_size)]


def _process_instance(coords: np.ndarray) -> str | None:
    tour = solve_tsp(coords)
    sample = TspSample(
        instance=TspInstance.from_coords_array(coords),
        tour=TspTour.from_solver_tour(tour),
    )
    if not sample.tour.is_valid(sample.instance.num_nodes):
        return None
    return sample.to_line()


def _build_arg_parser() -> argparse.ArgumentParser:
    defaults = TspGenerationConfig()
    parser = argparse.ArgumentParser(description="Generate TSP instances with optimal tours")
    for field_name, field_info in TspGenerationConfig.model_fields.items():
        default = getattr(defaults, field_name)
        arg_type: type[Any]
        if field_name == "filename":
            arg_type = str
        elif field_name == "solver":
            arg_type = str
        else:
            arg_type = type(default)
        parser.add_argument(
            f"--{field_name.replace('_', '-')}",
            type=arg_type,
            default=default,
            help=field_info.description or "",
        )
    return parser


def _config_from_args(args: argparse.Namespace) -> TspGenerationConfig:
    payload: dict[str, Any] = {}
    for field_name in TspGenerationConfig.model_fields:
        value = getattr(args, field_name)
        if value is not None or field_name == "filename":
            payload[field_name] = value
    return TspGenerationConfig.model_validate(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = _config_from_args(_build_arg_parser().parse_args())
    np.random.seed(config.seed)
    logger.info("Run options: %s", config.model_dump())

    stats = run_batch_generation(
        config,
        config.output_path,
        pool_initializer=_init_worker,
        pool_initargs=(config.model_dump(include={"solver", "lkh_trials"}),),
        build_tasks=lambda: _build_tsp_batch(config),
        process_task=_process_instance,
    )

    log_generation_summary(
        problem_label=f"TSP{config.min_nodes}-{config.max_nodes}",
        written=stats.written,
        requested=config.num_samples,
        elapsed_seconds=stats.elapsed_seconds,
    )


if __name__ == "__main__":
    main()
