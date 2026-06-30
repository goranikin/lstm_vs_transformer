import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing import Pool

import tqdm

from configs.tsp_config import BatchGenerationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationStats:
    written: int
    elapsed_seconds: float


def num_workers(batch_size: int) -> int:
    return min(batch_size, os.cpu_count() or 4)


def run_batch_generation[T](
    config: BatchGenerationConfig,
    output_path: str,
    *,
    pool_initializer: Callable[..., None] | None = None,
    pool_initargs: tuple = (),
    build_tasks: Callable[[], Iterable[T]],
    process_task: Callable[[T], str | None],
) -> GenerationStats:
    """Write one dataset line per successful task result."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    written = 0
    start_time = time.time()

    with (
        open(output_path, "w") as handle,
        Pool(
            num_workers(config.batch_size),
            initializer=pool_initializer,
            initargs=pool_initargs,
        ) as pool,
    ):
        for _ in tqdm.tqdm(range(config.num_batches)):
            lines = pool.map(process_task, list(build_tasks()))
            for line in lines:
                if line is None:
                    continue
                handle.write(line)
                written += 1

    return GenerationStats(written=written, elapsed_seconds=time.time() - start_time)


def log_generation_summary(
    *,
    problem_label: str,
    written: int,
    requested: int,
    elapsed_seconds: float,
    extra_lines: Iterable[str] = (),
) -> None:
    logger.info(
        "Completed generation of %s/%s %s instances.",
        written,
        requested,
        problem_label,
    )
    for line in extra_lines:
        logger.info("%s", line)
    logger.info("Total time: %.1fm", elapsed_seconds / 60)
    if written:
        logger.info("Average time: %.1fs", elapsed_seconds / written)
