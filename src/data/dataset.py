"""PyTorch datasets for routing-problem training (Section 5, AM paper)."""

import os
import pickle
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.generate_data import (
    sample_cvrp,
    sample_op,
    sample_pctsp,
    sample_tsp,
)
from src.data.registry import ProblemName, ProblemSpec, get_problem


def collate_routing_instances(
    instances: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack per-instance tensors into a training batch."""
    if not instances:
        raise ValueError("Cannot collate an empty list of instances")
    keys = instances[0].keys()
    return {key: torch.stack([instance[key] for instance in instances]) for key in keys}


def sample_instance(
    problem: ProblemSpec,
    graph_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Sample a single routing instance as a feature dict."""
    device = device or torch.device("cpu")

    if problem.name == ProblemName.TSP:
        return {"loc": sample_tsp(1, graph_size, device, generator).squeeze(0)}

    if problem.name in (ProblemName.CVRP, ProblemName.SDVRP):
        batch = sample_cvrp(1, graph_size, device, generator)
        return {key: value.squeeze(0) for key, value in batch.items()}

    if problem.name == ProblemName.OP:
        batch = sample_op(
            1,
            graph_size,
            distribution=problem.op_distribution,
            device=device,
            generator=generator,
        )
        return {key: value.squeeze(0) for key, value in batch.items()}

    if problem.name in (ProblemName.PCTSP, ProblemName.SPCTSP):
        batch = sample_pctsp(
            1,
            graph_size,
            device=device,
            stochastic=problem.stochastic,
            generator=generator,
        )
        return {key: value.squeeze(0) for key, value in batch.items()}

    raise ValueError(problem.name)


def _check_extension(filename: str) -> str:
    if not filename.endswith(".pkl"):
        return f"{filename}.pkl"
    return filename


def save_dataset(instances: list[dict[str, torch.Tensor]], filename: str) -> None:
    """Persist a list of routing instances to a pickle file."""
    path = _check_extension(filename)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(instances, handle, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename: str) -> list[dict[str, torch.Tensor]]:
    """Load routing instances from a pickle file."""
    with open(_check_extension(filename), "rb") as handle:
        data: Any = pickle.load(handle)
    if not isinstance(data, list):
        raise ValueError("Pickle file must contain a list of instance dicts")
    return data


class RoutingDataset(Dataset):
    """Routing problem dataset with on-the-fly or file-backed instances.

    Training (Section 5) generates ``num_samples`` instances on demand each
    epoch. For validation or testing, pass ``filename`` to load a fixed set.
    """

    def __init__(
        self,
        problem: str | ProblemSpec,
        graph_size: int,
        num_samples: int,
        seed: int = 1234,
        offset: int = 0,
        filename: str | None = None,
    ) -> None:
        self.problem = get_problem(problem) if isinstance(problem, str) else problem
        self.graph_size = graph_size
        self.seed = seed
        self.offset = offset

        if filename is not None:
            all_instances = load_dataset(filename)
            end = offset + num_samples
            self._instances = all_instances[offset:end]
            self.num_samples = len(self._instances)
        else:
            self._instances = None
            self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)

        if self._instances is not None:
            return self._instances[idx]

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.offset + idx)
        return sample_instance(self.problem, self.graph_size, generator=generator)


class RoutingEpochDataset(RoutingDataset):
    """One epoch of on-the-fly training data (Section 5).

    Default size is 1_280_000 instances (2500 steps × batch size 512).
    """

    def __init__(
        self,
        problem: str | ProblemSpec,
        graph_size: int,
        epoch_size: int = 1_280_000,
        seed: int = 1234,
        epoch: int = 0,
    ) -> None:
        super().__init__(
            problem=problem,
            graph_size=graph_size,
            num_samples=epoch_size,
            seed=seed,
            offset=epoch * epoch_size,
        )


def make_dataset(
    problem: str | ProblemSpec,
    graph_size: int,
    num_samples: int,
    seed: int = 1234,
    offset: int = 0,
    filename: str | None = None,
) -> RoutingDataset:
    """Build a routing dataset for training or evaluation."""
    return RoutingDataset(
        problem=problem,
        graph_size=graph_size,
        num_samples=num_samples,
        seed=seed,
        offset=offset,
        filename=filename,
    )
