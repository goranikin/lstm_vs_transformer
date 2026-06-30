# Data generation

Scripts and dataset loaders for problem instances written to JSONL files under
the repo-root `data/` directory.

| Problem | Module | Output docs |
|---------|--------|-------------|
| TSP | `src.data_generating.TSP` | [TSP/README.md](TSP/README.md) |
| MIS | `src.data_generating.MIS` | [MIS/README.md](MIS/README.md) |

Run generators as modules from the repo root, for example:

```bash
uv run python -m src.data_generating.TSP.generate --help
uv run python -m src.data_generating.MIS.generate --help
```

Training code loads generated files via `TSPDataset` / `MISDataset` in this
package (see `src/training/utils.py`).
