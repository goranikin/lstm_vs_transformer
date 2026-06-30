from typing import Any

import torch
from torch.utils.data import Dataset

from src.data_generating.common import load_jsonl
from src.data_generating.MIS.algorithms import edges_to_adjacency


class MISDataset(Dataset):
    def __init__(
        self,
        path: str,
        target_algorithm: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.path = path
        self.target_algorithm = target_algorithm
        self.dtype = dtype
        self.records = load_jsonl(path)

        for index, record in enumerate(self.records):
            if record.get("problem") != "mis":
                raise ValueError(f"Record {index} is not an MIS instance")
            if (
                target_algorithm is not None
                and target_algorithm not in record["solutions"]
            ):
                raise ValueError(
                    f"Record {index} does not contain target algorithm "
                    f"'{target_algorithm}'"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        num_nodes = int(record["num_nodes"])
        edges = [tuple(edge) for edge in record["edges"]]
        adjacency = edges_to_adjacency(num_nodes, edges)

        item: dict[str, Any] = {
            "index": torch.tensor(record["index"], dtype=torch.long),
            "seed": torch.tensor(record["seed"], dtype=torch.long),
            "adjacency": torch.tensor(adjacency, dtype=self.dtype),
        }

        if self.target_algorithm is not None:
            solution = record["solutions"][self.target_algorithm]
            target = torch.zeros(num_nodes, dtype=self.dtype)
            target[torch.tensor(solution["nodes"], dtype=torch.long)] = 1.0
            item["target_set"] = target
            item["target_size"] = torch.tensor(solution["size"], dtype=torch.long)

        return item


def collate_mis(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty MIS batch")
    keys = items[0].keys()
    batch: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch
