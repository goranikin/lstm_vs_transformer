from typing import Any

import torch
from torch.utils.data import Dataset

from src.data_generating.common import load_jsonl


class TSPDataset(Dataset):
    """File-backed TSP dataset produced by ``src.data_generating.TSP.generate``."""

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
            if record.get("problem") != "tsp":
                raise ValueError(f"Record {index} is not a TSP instance")
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
        item: dict[str, Any] = {
            "index": torch.tensor(record["index"], dtype=torch.long),
            "seed": torch.tensor(record["seed"], dtype=torch.long),
            "loc": torch.tensor(record["coordinates"], dtype=self.dtype),
        }

        if self.target_algorithm is not None:
            solution = record["solutions"][self.target_algorithm]
            item["target_tour"] = torch.tensor(solution["tour"], dtype=torch.long)
            item["target_cost"] = torch.tensor(solution["cost"], dtype=self.dtype)

        return item


def collate_tsp(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty TSP batch")
    keys = items[0].keys()
    batch: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch
