import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.data_generating.common import ExternalSolverError


class MisSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    nodes: list[int]
    size: int
    is_exact: bool
    metadata: dict[str, Any] | None = None

    def to_record(self) -> dict:
        record = {
            "algorithm": self.algorithm,
            "is_exact": self.is_exact,
            "nodes": self.nodes,
            "size": self.size,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record


def _validate_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.bool_)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if np.any(np.diag(adjacency)):
        raise ValueError("adjacency must not contain self-loops")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")
    return adjacency


def edges_to_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.bool_)
    for u, v in edges:
        if u == v:
            raise ValueError("MIS graph cannot contain self-loops")
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError("edge endpoint out of range")
        adjacency[u, v] = True
        adjacency[v, u] = True
    return adjacency


def adjacency_to_edges(adjacency: np.ndarray) -> list[tuple[int, int]]:
    adjacency = _validate_adjacency(adjacency)
    edges: list[tuple[int, int]] = []
    for u in range(adjacency.shape[0]):
        for v in range(u + 1, adjacency.shape[1]):
            if bool(adjacency[u, v]):
                edges.append((u, v))
    return edges


def is_independent_set(adjacency: np.ndarray, nodes: list[int]) -> bool:
    adjacency = _validate_adjacency(adjacency)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if bool(adjacency[u, v]):
                return False
    return True


def _write_metis_graph(path: Path, adjacency: np.ndarray) -> None:
    adjacency = _validate_adjacency(adjacency)
    num_nodes = adjacency.shape[0]
    num_edges = int(np.triu(adjacency, k=1).sum())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{num_nodes} {num_edges}\n")
        for node in range(num_nodes):
            neighbors = np.flatnonzero(adjacency[node]).tolist()
            handle.write(" ".join(str(neighbor + 1) for neighbor in neighbors))
            handle.write("\n")


def _parse_independent_set_file(path: Path, num_nodes: int) -> list[int]:
    values: list[int] = []
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in stripped.replace(",", " ").split():
                try:
                    values.append(int(token))
                except ValueError:
                    continue

    if not values:
        raise ExternalSolverError(f"KaMIS output file did not contain node ids: {path}")

    if len(values) == num_nodes and set(values).issubset({0, 1}):
        return [index for index, value in enumerate(values) if value == 1]

    if min(values) >= 1 and max(values) <= num_nodes:
        nodes = [value - 1 for value in values]
    elif min(values) >= 0 and max(values) < num_nodes:
        nodes = values
    else:
        raise ExternalSolverError(f"Independent-set node ids out of range: {path}")

    return sorted(set(nodes))


def _parse_independent_set_stdout(stdout: str, num_nodes: int) -> list[int] | None:
    for line in reversed(stdout.splitlines()):
        lower = line.lower()
        if "independent set" not in lower and "solution" not in lower:
            continue
        values: list[int] = []
        for token in line.replace(":", " ").replace(",", " ").split():
            try:
                values.append(int(token))
            except ValueError:
                continue
        if not values:
            continue
        if len(values) == num_nodes and set(values).issubset({0, 1}):
            return [index for index, value in enumerate(values) if value == 1]
        if min(values) >= 1 and max(values) <= num_nodes:
            return sorted({value - 1 for value in values})
        if min(values) >= 0 and max(values) < num_nodes:
            return sorted(set(values))
    return None


def _resolve_executable(
    explicit_path: str | None,
    env_var: str,
    candidate_names: Sequence[str],
) -> str | None:
    if explicit_path:
        return explicit_path
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path
    for candidate in candidate_names:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def solve_kamis(
    adjacency: np.ndarray,
    *,
    executable: str | None = None,
    seed: int | None = None,
    time_limit_sec: float | None = None,
) -> MisSolution:
    adjacency = _validate_adjacency(adjacency)
    executable = _resolve_executable(
        executable,
        "KAMIS_EXECUTABLE",
        ("redumis", "KaMIS", "kamis", "branch_reduce"),
    )
    if executable is None:
        raise ExternalSolverError(
            "KaMIS is not available. Put a KaMIS binary on PATH, set "
            "KAMIS_EXECUTABLE, or pass --kamis-executable /path/to/redumis."
        )

    with tempfile.TemporaryDirectory(prefix="mis_kamis_") as tmp:
        tmpdir = Path(tmp)
        graph_path = tmpdir / "instance.graph"
        solution_path = tmpdir / "solution.txt"
        _write_metis_graph(graph_path, adjacency)

        command = [
            executable,
            str(graph_path),
            f"--output={solution_path}",
        ]
        if seed is not None:
            command.append(f"--seed={seed}")
        if time_limit_sec is not None:
            command.append(f"--time_limit={time_limit_sec}")

        result = subprocess.run(
            command,
            cwd=tmpdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=time_limit_sec,
        )
        if result.returncode != 0:
            raise ExternalSolverError(
                "KaMIS failed with exit code "
                f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
            )

        if solution_path.exists():
            nodes = _parse_independent_set_file(solution_path, adjacency.shape[0])
        else:
            parsed = _parse_independent_set_stdout(result.stdout, adjacency.shape[0])
            if parsed is None:
                raise ExternalSolverError(
                    "KaMIS did not produce the configured output file and stdout "
                    "did not contain a parseable independent set"
                )
            nodes = parsed

    if not is_independent_set(adjacency, nodes):
        raise ExternalSolverError("KaMIS produced an invalid independent set")
    return MisSolution(
        algorithm="kamis",
        nodes=nodes,
        size=len(nodes),
        is_exact=False,
        metadata={"time_limit_sec": time_limit_sec},
    )


def solve_gurobi(
    adjacency: np.ndarray,
    *,
    seed: int | None = None,
    time_limit_sec: float | None = None,
) -> MisSolution:
    adjacency = _validate_adjacency(adjacency)
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ExternalSolverError(
            "gurobipy is not installed. Install Gurobi's Python package in this "
            "environment before generating MIS labels with --solvers gurobi."
        ) from exc

    num_nodes = adjacency.shape[0]
    model = gp.Model("maximum_independent_set")
    model.Params.OutputFlag = 0
    if seed is not None:
        model.Params.Seed = int(seed)
    if time_limit_sec is not None:
        model.Params.TimeLimit = float(time_limit_sec)

    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(x[node] for node in range(num_nodes)), GRB.MAXIMIZE)
    for u, v in adjacency_to_edges(adjacency):
        model.addConstr(x[u] + x[v] <= 1)

    model.optimize()
    if model.SolCount == 0:
        raise ExternalSolverError(
            f"Gurobi did not find a feasible solution: {model.Status}"
        )

    nodes = [node for node in range(num_nodes) if x[node].X > 0.5]
    if not is_independent_set(adjacency, nodes):
        raise ExternalSolverError("Gurobi produced an invalid independent set")

    is_exact = model.Status == GRB.OPTIMAL
    return MisSolution(
        algorithm="gurobi",
        nodes=nodes,
        size=len(nodes),
        is_exact=is_exact,
        metadata={
            "objective_bound": float(model.ObjBound),
            "status": int(model.Status),
        },
    )


def solve_with_algorithms(
    adjacency: np.ndarray,
    *,
    algorithms: Sequence[str] = ("kamis", "gurobi"),
    kamis_executable: str | None = None,
    seed: int | None = None,
    time_limit_sec: float | None = None,
) -> dict[str, dict]:
    adjacency = _validate_adjacency(adjacency)
    solutions: dict[str, dict] = {}

    for algorithm in algorithms:
        normalized = algorithm.lower().replace("-", "_")
        if normalized == "kamis":
            solution = solve_kamis(
                adjacency,
                executable=kamis_executable,
                seed=seed,
                time_limit_sec=time_limit_sec,
            )
        elif normalized == "gurobi":
            solution = solve_gurobi(
                adjacency,
                seed=seed,
                time_limit_sec=time_limit_sec,
            )
        else:
            raise ValueError(
                f"Unknown MIS solver algorithm: {algorithm}. Supported: kamis, gurobi."
            )
        if not is_independent_set(adjacency, solution.nodes):
            raise RuntimeError(f"{solution.algorithm} produced an invalid set")
        solutions[solution.algorithm] = solution.to_record()

    return solutions
