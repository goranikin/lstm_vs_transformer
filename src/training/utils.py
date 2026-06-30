import random
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.am_config import AMModelConfig
from configs.pn_config import PNModelConfig
from src.data_generating.MIS.dataset import MISDataset, collate_mis
from src.data_generating.TSP.dataset import TSPDataset, collate_tsp
from src.models.pointer_network.model import PointerNetwork
from src.models.transformer.model import AttentionModel

ModelName = Literal["am", "pn"]
ProblemName = Literal["tsp", "mis"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def build_dataset(
    problem: ProblemName,
    path: str,
    target_algorithm: str | None = None,
) -> TSPDataset | MISDataset:
    if problem == "tsp":
        return TSPDataset(path, target_algorithm=target_algorithm)
    if problem == "mis":
        return MISDataset(path, target_algorithm=target_algorithm)
    raise ValueError(f"Unsupported problem: {problem}")


def collate_for(problem: ProblemName) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    if problem == "tsp":
        return collate_tsp
    if problem == "mis":
        return collate_mis
    raise ValueError(f"Unsupported problem: {problem}")


def build_loader(
    problem: ProblemName,
    path: str,
    batch_size: int,
    target_algorithm: str | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        build_dataset(problem, path, target_algorithm=target_algorithm),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_for(problem),
    )


def build_model(
    model_name: ModelName,
    problem: ProblemName,
) -> torch.nn.Module:
    if model_name == "am":
        return AttentionModel(config=AMModelConfig(), default_problem=problem)
    if model_name == "pn":
        return PointerNetwork(config=PNModelConfig(), default_problem=problem)
    raise ValueError(f"Unsupported model: {model_name}")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def default_target_algorithm(problem: ProblemName) -> str:
    if problem == "tsp":
        return "concorde"
    if problem == "mis":
        return "gurobi"
    raise ValueError(f"Unsupported problem: {problem}")
