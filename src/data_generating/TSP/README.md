# TSP Dataset (`src/data_generating/TSP`)

This directory contains deterministic data generation and dataset loading for the
Euclidean Travelling Salesman Problem (TSP).

## Problem

Each instance contains `num_nodes` points sampled uniformly from `[0, 1]^2`.
The label solvers return a tour, represented as a permutation of node indices.
Tour cost is the closed Euclidean tour length.

## Label Algorithms

Only the following solvers are supported for generated labels:

- `concorde`: Concorde TSP solver. This is treated as the exact optimal label.
- `lkh3`: LKH-3 solver. This is a high-quality heuristic label.

Concorde is resolved in this order:

1. Python package `concorde` / `pyconcorde`
2. `CONCORDE_EXECUTABLE` environment variable
3. `--concorde-executable /path/to/concorde`
4. `concorde` on `PATH`

LKH-3 is resolved in this order:

1. `LKH3_EXECUTABLE` environment variable
2. `--lkh3-executable /path/to/LKH`
3. `LKH`, `lkh`, or `LKH-3` on `PATH`

## Generate Data

```bash
uv run python -m src.data_generating.TSP.generate \
  --num-instances 1000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/tsp/tsp50_seed1234.jsonl
```

With explicit solver paths:

```bash
uv run python -m src.data_generating.TSP.generate \
  --num-instances 1000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/tsp/tsp50_seed1234.jsonl \
  --concorde-executable /path/to/concorde \
  --lkh3-executable /path/to/LKH
```

Useful options:

- `--solvers concorde,lkh3`: comma-separated solver list.
- `--lkh3-trials 1000`: LKH-3 `MAX_TRIALS`.
- `--lkh3-runs 10`: LKH-3 `RUNS`.
- `--solver-timeout-sec 300`: timeout per solver call.

## File Format

The generator writes JSONL. Each line is one instance:

```json
{
  "problem": "tsp",
  "index": 0,
  "seed": 1234,
  "num_nodes": 50,
  "coordinates": [[0.1, 0.2], [0.3, 0.4]],
  "solutions": {
    "concorde": {
      "algorithm": "concorde",
      "tour": [0, 1],
      "cost": 1.0,
      "is_exact": true
    },
    "lkh3": {
      "algorithm": "lkh3",
      "tour": [0, 1],
      "cost": 1.0,
      "is_exact": false
    }
  }
}
```

The per-instance seed is `base_seed + index`, so generation is reproducible.

## Dataset Usage

```python
from torch.utils.data import DataLoader

from src.data_generating.TSP.dataset import TSPDataset, collate_tsp

dataset = TSPDataset(
    "data/tsp/tsp50_seed1234.jsonl",
    target_algorithm="concorde",
)
loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_tsp)

batch = next(iter(loader))
loc = batch["loc"]                 # [batch, num_nodes, 2]
target_tour = batch["target_tour"] # [batch, num_nodes]
target_cost = batch["target_cost"] # [batch]
```

Use `target_algorithm="lkh3"` to train against the LKH-3 label instead.
