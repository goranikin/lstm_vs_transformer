# MIS Dataset

This directory contains deterministic data generation and dataset loading for the
Maximum Independent Set (MIS) problem.

## Problem

Each instance is an undirected Erdos-Renyi graph `G(n, p)` with `num_nodes`
vertices and edge probability `edge_probability`. The label solvers return an
independent set as a list of node indices.

## Label Algorithms

Only the following solvers are supported for generated labels:

- `kamis`: KaMIS maximum independent set solver.
- `gurobi`: Gurobi binary integer programming model for MIS.

The Gurobi model uses one binary variable per node and one constraint per edge:

```text
maximize sum_i x_i
subject to x_u + x_v <= 1 for every edge (u, v)
x_i in {0, 1}
```

`gurobi` is exact when Gurobi returns `OPTIMAL`. If a time limit is used and
Gurobi returns a feasible non-optimal solution, the record stores the returned
solution and solver metadata.

KaMIS is resolved in this order:

1. `KAMIS_EXECUTABLE` environment variable
2. `--kamis-executable /path/to/redumis`
3. `redumis`, `KaMIS`, `kamis`, or `branch_reduce` on `PATH`

Gurobi requires the `gurobipy` Python package and a valid local license.

## Generate Data

```bash
uv run python -m src.new_data.MIS.generate \
  --num-instances 1000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path data/new/mis100_p015_seed1234.jsonl
```

With explicit KaMIS path and a per-instance solver time limit:

```bash
uv run python -m src.new_data.MIS.generate \
  --num-instances 1000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path data/new/mis100_p015_seed1234.jsonl \
  --kamis-executable /path/to/redumis \
  --solver-time-limit-sec 300
```

Useful options:

- `--solvers kamis,gurobi`: comma-separated solver list.
- `--solver-time-limit-sec 300`: timeout/time limit per solver call.
- `--kamis-executable /path/to/redumis`: explicit KaMIS binary path.

## File Format

The generator writes JSONL. Each line is one instance:

```json
{
  "problem": "mis",
  "index": 0,
  "seed": 1234,
  "num_nodes": 5,
  "edge_probability": 0.15,
  "edges": [[0, 2], [1, 3]],
  "solutions": {
    "kamis": {
      "algorithm": "kamis",
      "nodes": [0, 1, 4],
      "size": 3,
      "is_exact": false
    },
    "gurobi": {
      "algorithm": "gurobi",
      "nodes": [0, 1, 4],
      "size": 3,
      "is_exact": true
    }
  }
}
```

The per-instance seed is `base_seed + index`, so graph generation is
reproducible.

## Dataset Usage

```python
from torch.utils.data import DataLoader

from src.new_data.MIS.dataset import MISDataset, collate_mis

dataset = MISDataset(
    "data/new/mis100_p015_seed1234.jsonl",
    target_algorithm="gurobi",
)
loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_mis)

batch = next(iter(loader))
adjacency = batch["adjacency"]   # [batch, num_nodes, num_nodes]
target_set = batch["target_set"] # [batch, num_nodes], 1.0 for selected nodes
target_size = batch["target_size"]
```

Use `target_algorithm="kamis"` to train against the KaMIS label instead.

