import os
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BatchGenerationConfig(BaseModel):
    """Shared batch-generation settings."""

    model_config = ConfigDict(validate_assignment=True)

    num_samples: int = Field(default=128_000, gt=0)
    batch_size: int = Field(default=128, gt=0)
    seed: int = 1234
    filename: str | None = None

    @property
    def num_batches(self) -> int:
        if self.num_samples % self.batch_size != 0:
            raise ValueError(
                f"num_samples ({self.num_samples}) must be divisible by "
                f"batch_size ({self.batch_size})"
            )
        return self.num_samples // self.batch_size


class TspSolverConfig(BaseModel):
    """External TSP solver settings passed to worker processes."""

    model_config = ConfigDict(validate_assignment=True)

    solver: Literal["concorde", "lkh"] = "concorde"
    lkh_trials: int = Field(default=1000, gt=0)


class TspGenerationConfig(BatchGenerationConfig, TspSolverConfig):
    """TSP dataset generation (Appendix B.2 + Concorde/LKH optimal tours)."""

    model_config = ConfigDict(validate_assignment=True)

    min_nodes: int = Field(default=50, gt=0)
    max_nodes: int = Field(default=50, gt=0)

    @model_validator(mode="after")
    def validate_node_range(self) -> Self:
        if self.min_nodes > self.max_nodes:
            raise ValueError("min_nodes must be <= max_nodes")
        return self

    def sample_num_nodes(self) -> int:
        if self.min_nodes == self.max_nodes:
            return self.min_nodes
        return int(np.random.randint(self.min_nodes, self.max_nodes + 1))

    @property
    def output_path(self) -> str:
        if self.filename is not None:
            return self.filename
        graph_size = (
            self.min_nodes
            if self.min_nodes == self.max_nodes
            else f"{self.min_nodes}-{self.max_nodes}"
        )
        return os.path.join("data", "tsp", f"tsp{graph_size}_seed{self.seed}.txt")
