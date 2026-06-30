"""Problem definitions and cost functions."""

from enum import StrEnum
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict

from src.data import generate_data
from src.data.state import OPState, PCTSPState, TSPState, VRPState


class ProblemName(StrEnum):
    TSP = "tsp"
    CVRP = "cvrp"
    SDVRP = "sdvrp"
    OP = "op"
    PCTSP = "pctsp"
    SPCTSP = "spctsp"


class ProblemSpec(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    name: ProblemName
    node_dim: int
    step_context_dim: int
    has_depot: bool
    allow_split: bool = False
    op_distribution: Literal["distance", "const", "uniform"] = "distance"
    stochastic: bool = False


PROBLEM_SPECS: dict[ProblemName, ProblemSpec] = {
    ProblemName.TSP: ProblemSpec(
        name=ProblemName.TSP, node_dim=2, step_context_dim=2, has_depot=False
    ),
    ProblemName.CVRP: ProblemSpec(
        name=ProblemName.CVRP, node_dim=3, step_context_dim=1, has_depot=True
    ),
    ProblemName.SDVRP: ProblemSpec(
        name=ProblemName.SDVRP,
        node_dim=3,
        step_context_dim=1,
        has_depot=True,
        allow_split=True,
    ),
    ProblemName.OP: ProblemSpec(
        name=ProblemName.OP, node_dim=3, step_context_dim=1, has_depot=True
    ),
    ProblemName.PCTSP: ProblemSpec(
        name=ProblemName.PCTSP, node_dim=4, step_context_dim=1, has_depot=True
    ),
    ProblemName.SPCTSP: ProblemSpec(
        name=ProblemName.SPCTSP,
        node_dim=4,
        step_context_dim=1,
        has_depot=True,
        stochastic=True,
    ),
}


def get_problem(name: str) -> ProblemSpec:
    try:
        return PROBLEM_SPECS[ProblemName(name)]
    except ValueError as exc:
        supported = ", ".join(p.value for p in ProblemName)
        raise ValueError(f"Unknown problem '{name}'. Supported: {supported}") from exc


def sample_batch(
    problem: ProblemSpec,
    batch_size: int,
    graph_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if problem.name == ProblemName.TSP:
        return {"loc": generate_data.sample_tsp(batch_size, graph_size, device, generator)}
    if problem.name in (ProblemName.CVRP, ProblemName.SDVRP):
        return generate_data.sample_cvrp(batch_size, graph_size, device, generator)
    if problem.name == ProblemName.OP:
        return generate_data.sample_op(
            batch_size,
            graph_size,
            distribution=problem.op_distribution,
            device=device,
            generator=generator,
        )
    if problem.name in (ProblemName.PCTSP, ProblemName.SPCTSP):
        return generate_data.sample_pctsp(
            batch_size,
            graph_size,
            device=device,
            stochastic=problem.stochastic,
            generator=generator,
        )
    raise ValueError(problem.name)


def make_state(problem: ProblemSpec, batch: dict[str, torch.Tensor]) -> Any:
    if problem.name == ProblemName.TSP:
        return TSPState.initialize(batch["loc"])
    if problem.name in (ProblemName.CVRP, ProblemName.SDVRP):
        return VRPState.initialize(
            batch["loc"], batch["demand"], allow_split=problem.allow_split
        )
    if problem.name == ProblemName.OP:
        return OPState.initialize(
            batch["loc"], batch["prize"], batch["max_length"][0].item()
        )
    if problem.name in (ProblemName.PCTSP, ProblemName.SPCTSP):
        return PCTSPState.initialize(batch["loc"], batch["prize"], batch["penalty"])
    raise ValueError(problem.name)


def compute_cost(
    problem: ProblemSpec,
    batch: dict[str, torch.Tensor],
    pi: torch.Tensor,
) -> torch.Tensor:
    if problem.name == ProblemName.TSP:
        return TSPState.get_final_cost(batch["loc"], pi)
    if problem.name in (ProblemName.CVRP, ProblemName.SDVRP):
        return VRPState.get_final_cost(batch["loc"], pi)
    if problem.name == ProblemName.OP:
        return OPState.get_final_cost(batch["loc"], pi, batch["prize"])
    if problem.name in (ProblemName.PCTSP, ProblemName.SPCTSP):
        return PCTSPState.get_final_cost(
            batch["loc"], pi, batch["prize"], batch["penalty"]
        )
    raise ValueError(problem.name)


def max_decode_steps(problem: ProblemSpec, graph_size: int) -> int:
    if problem.name == ProblemName.TSP:
        return graph_size
    if problem.name in (ProblemName.CVRP, ProblemName.SDVRP):
        return graph_size * 4
    if problem.name in (ProblemName.OP, ProblemName.PCTSP, ProblemName.SPCTSP):
        return graph_size + 1
    return graph_size
