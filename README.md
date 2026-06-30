# lstm-vs-transformer

Comparison of Pointer Network and Attention Model on combinatorial optimization
problems, based on [Attention, Learn to Solve Routing Problems](docs/Attention,%20learn%20to%20solve%20routing%20problems.pdf).

## Layout

```text
data/                  Generated JSONL datasets (repo root; not source code)
src/data_generating/   Data generation scripts and dataset loaders
src/models/            Pointer Network and Attention Model
src/training/          Training loops and utilities
src/main/              Training entry points
configs/               Model and training hyperparameters
```

`data/` at the repo root stores generated instance files. `src/data_generating/`
contains the code that creates and loads those files.

## Setup

```bash
uv sync
uv sync --extra tsp-solver   # optional: pyconcorde for TSP labels
```

MIS labels with Gurobi require `gurobipy` and a valid Gurobi license.

## Generate data

See `src/data_generating/TSP/README.md` and `src/data_generating/MIS/README.md`
for details.

```bash
# TSP validation set (paper: 10k instances, n=50)
uv run python -m src.data_generating.TSP.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/tsp/tsp50_val_seed1234.jsonl \
  --solvers concorde

# MIS (example defaults: n=100, p=0.15)
uv run python -m src.data_generating.MIS.generate \
  --num-instances 1000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path data/mis/mis100_p015_val_seed1234.jsonl \
  --solvers gurobi
```

## Train

```bash
# Attention Model, TSP, REINFORCE
uv run python -m src.main.train_am_tsp_rl \
  --train-path data/tsp/tsp50_val_seed1234.jsonl \
  --val-path data/tsp/tsp50_val_seed1234.jsonl \
  --epochs 100 \
  --steps-per-epoch 2500 \
  --batch-size 512

# Attention Model, MIS, supervised
uv run python -m src.main.train_am_mis_supervised \
  --train-path data/mis/mis100_p015_val_seed1234.jsonl \
  --val-path data/mis/mis100_p015_val_seed1234.jsonl \
  --target-algorithm gurobi
```

Other entry points: `train_am_tsp_supervised`, `train_pn_tsp_rl`,
`train_pn_tsp_supervised`, `train_am_mis_rl`, `train_pn_mis_rl`,
`train_pn_mis_supervised`.
