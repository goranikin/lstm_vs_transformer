# lstm-vs-transformer

Experiments for comparing Pointer Network (PN) and Attention Model (AM) on
combinatorial optimization problems.

The project is based on two papers in `docs/`:

- `Pointer Networks.pdf`
- `Attention, learn to solve routing problems.pdf`

The first reproduction target is the AM paper's TSP comparison against PN. MIS
is included as an additional graph problem for this project; it is not part of
the original AM paper benchmark.

## Current Scope

Supported problems:

- `tsp`: Euclidean Travelling Salesman Problem.
- `mis`: Maximum Independent Set on undirected Erdos-Renyi graphs.

Supported models:

- `pn`: LSTM Pointer Network with additive attention and one glimpse.
- `am`: Transformer-style graph attention encoder with attention decoder.

Supported training modes:

- `supervised`: train against solver-produced labels in the dataset.
- `rl`: train with REINFORCE using sampled model solutions and a baseline.

The default PN hidden size is set so that PN and AM have similar trainable
parameter counts:

- PN default: about `795k` parameters.
- AM default: about `792k` parameters.

## Repository Layout

```text
configs/                 Pydantic model/training config defaults
docs/                    Reference papers
src/data_generating/     TSP/MIS generators, solver wrappers, dataset classes
src/models/              PN and AM implementations
src/training/            Shared supervised/RL trainer classes and utilities
src/main/                Concrete training entry points
data/                    Generated datasets and local experiment data
outputs/                 Training checkpoints and logs, when generated
```

Problem-specific dataset documentation is in:

- `src/data_generating/TSP/README.md`
- `src/data_generating/MIS/README.md`

## Setup

Install the Python environment:

```bash
uv sync
```

Optional TSP solver dependencies:

```bash
uv sync --extra tsp-solver
```

External solver requirements:

- TSP labels use `concorde` and/or `lkh3`.
- MIS labels use `kamis` and/or `gurobi`.
- Gurobi requires `gurobipy` and a valid local Gurobi license.

## Data Generation

Generated datasets are JSONL files. Each line is one problem instance plus any
solver labels generated for that instance.

TSP example:

```bash
uv run python -m src.data_generating.TSP.generate \
  --num-instances 10000 \
  --num-nodes 50 \
  --seed 1234 \
  --output-path data/tsp/tsp50_seed1234.jsonl \
  --solvers concorde,lkh3
```

MIS example:

```bash
uv run python -m src.data_generating.MIS.generate \
  --num-instances 10000 \
  --num-nodes 100 \
  --edge-probability 0.15 \
  --seed 1234 \
  --output-path data/mis/mis100_p015_seed1234.jsonl \
  --solvers gurobi,kamis
```

Generation is deterministic per instance: the seed for item `index` is
`base_seed + index`.

## Hydra Training Configuration

Training uses Hydra. The base config is:

```text
configs/hydra/base.yaml
```

Each model/problem/training-mode combination has its own config:

```text
configs/hydra/train_am_tsp_supervised.yaml
configs/hydra/train_am_tsp_rl.yaml
configs/hydra/train_am_mis_supervised.yaml
configs/hydra/train_am_mis_rl.yaml
configs/hydra/train_pn_tsp_supervised.yaml
configs/hydra/train_pn_tsp_rl.yaml
configs/hydra/train_pn_mis_supervised.yaml
configs/hydra/train_pn_mis_rl.yaml
```

The matching Python entry points are:

```text
src.main.train_am_tsp_supervised
src.main.train_am_tsp_rl
src.main.train_am_mis_supervised
src.main.train_am_mis_rl
src.main.train_pn_tsp_supervised
src.main.train_pn_tsp_rl
src.main.train_pn_mis_supervised
src.main.train_pn_mis_rl
```

Use Hydra overrides instead of argparse flags. Common override keys:

```text
paths.train
paths.val
paths.output_dir
data.target_algorithm
data.batch_size
data.eval_batch_size
data.num_workers
data.shuffle
trainer.epochs
trainer.steps_per_epoch
trainer.learning_rate
trainer.learning_rate_decay
trainer.max_grad_norm
trainer.optimizer
trainer.baseline
trainer.log_every
trainer.checkpoint_every
model.am.d_h
model.am.n_layers
model.am.n_heads
model.pn.hidden_size
model.pn.n_glimpses
```

Supervised TSP with AM:

```bash
uv run python -m src.main.train_am_tsp_supervised \
  paths.train=data/tsp/tsp50_seed1234.jsonl \
  paths.val=data/tsp/tsp50_val_seed1234.jsonl \
  data.target_algorithm=concorde \
  trainer.epochs=100 \
  data.batch_size=512
```

RL TSP with AM:

```bash
uv run python -m src.main.train_am_tsp_rl \
  paths.train=data/tsp/tsp50_seed1234.jsonl \
  paths.val=data/tsp/tsp50_val_seed1234.jsonl \
  trainer.epochs=100 \
  trainer.steps_per_epoch=2500 \
  data.batch_size=512 \
  trainer.baseline=rollout
```

Supervised MIS with PN:

```bash
uv run python -m src.main.train_pn_mis_supervised \
  paths.train=data/mis/mis100_p015_seed1234.jsonl \
  paths.val=data/mis/mis100_p015_val_seed1234.jsonl \
  data.target_algorithm=gurobi \
  trainer.epochs=100
```

RL MIS with PN:

```bash
uv run python -m src.main.train_pn_mis_rl \
  paths.train=data/mis/mis100_p015_seed1234.jsonl \
  paths.val=data/mis/mis100_p015_val_seed1234.jsonl \
  trainer.epochs=100 \
  trainer.steps_per_epoch=2500 \
  trainer.baseline=rollout
```

Print a resolved job config without training:

```bash
uv run python -m src.main.train_am_tsp_rl --cfg job
```

## Model Interface

Both PN and AM follow the same high-level interface:

```python
cost, log_likelihood = model(batch, problem="tsp")
cost, log_likelihood, solution = model(batch, problem="mis", return_pi=True)
loss = model.supervised_loss(batch, problem="tsp")
model.set_decode_type("sampling")  # or "greedy"
```

Default batch keys:

```text
TSP input:        loc
TSP label:        target_tour
MIS input:        adjacency
MIS label:        target_set
```

Both models allow these keys to be overridden in the constructor, for example:

```python
AttentionModel(loc_key="coords", target_tour_key="tour")
PointerNetwork(adjacency_key="graph", target_set_key="label_set")
```

## Training Behavior

Supervised mode:

- TSP uses pointer/tour imitation against `target_tour`.
- MIS uses binary node-membership loss against `target_set`.

RL mode:

- TSP cost is closed tour length.
- MIS cost is negative independent-set size, so lower is better.
- `rollout` baseline uses greedy decoding from a frozen model.
- `exponential` baseline uses an exponential moving average of sampled costs.

Default optimizer settings follow the intended reproduction split:

- AM RL: Adam, learning rate `1e-4`.
- PN RL: Adam, learning rate `1e-3`, decay `0.96`.
- PN supervised: SGD, learning rate `1.0`, gradient clip `2.0`.

## Verification

Basic import/compile check:

```bash
uv run python -m compileall -q src configs
```

Small one-step smoke run:

```bash
uv run python -m src.main.train_am_tsp_rl \
  paths.train=data/tsp/tsp50_seed1234.jsonl \
  trainer.epochs=1 \
  trainer.steps_per_epoch=1 \
  data.batch_size=2 \
  trainer.checkpoint_every=99
```
