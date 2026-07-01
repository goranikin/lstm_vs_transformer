import random
from collections.abc import Callable, Mapping
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
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        else:
            return torch.device("cpu")
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


def collate_for(
    problem: ProblemName,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
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
    model_options: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    options = dict(model_options or {})
    interface = dict(options.get("interface") or {})
    if model_name == "am":
        return AttentionModel(
            config=AMModelConfig(**dict(options.get("am") or {})),
            default_problem=problem,
            tsp_input_size=int(options.get("tsp_input_size", 2)),
            mis_input_size=int(options.get("mis_input_size", 1)),
            mis_context_size=int(options.get("mis_context_size", 1)),
            loc_key=str(interface.get("loc_key", "loc")),
            adjacency_key=str(interface.get("adjacency_key", "adjacency")),
            target_tour_key=str(interface.get("target_tour_key", "target_tour")),
            target_set_key=str(interface.get("target_set_key", "target_set")),
        )
    if model_name == "pn":
        return PointerNetwork(
            input_size=int(options.get("input_size", 2)),
            config=PNModelConfig(**dict(options.get("pn") or {})),
            default_problem=problem,
            loc_key=str(interface.get("loc_key", "loc")),
            adjacency_key=str(interface.get("adjacency_key", "adjacency")),
            target_tour_key=str(interface.get("target_tour_key", "target_tour")),
            target_set_key=str(interface.get("target_set_key", "target_set")),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def default_target_algorithm(problem: ProblemName) -> str:
    if problem == "tsp":
        return "concorde"
    if problem == "mis":
        return "gurobi"
    raise ValueError(f"Unsupported problem: {problem}")
