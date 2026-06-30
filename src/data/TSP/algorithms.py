import importlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.data.common import ExternalSolverError


class TspSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    tour: list[int]
    cost: float
    is_exact: bool
    metadata: dict[str, Any] | None = None

    def to_record(self) -> dict:
        record = {
            "algorithm": self.algorithm,
            "cost": self.cost,
            "is_exact": self.is_exact,
            "tour": self.tour,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record


def tour_length(coords: np.ndarray, tour: list[int] | np.ndarray) -> float:
    coords = np.asarray(coords, dtype=np.float64)
    route = np.asarray(tour, dtype=np.int64)
    if route.ndim != 1:
        raise ValueError("tour must be one-dimensional")
    if len(route) != len(coords):
        raise ValueError("tour length must equal number of coordinates")
    ordered = coords[route]
    edges = np.roll(ordered, shift=-1, axis=0) - ordered
    return float(np.linalg.norm(edges, axis=1).sum())


def _validate_tour(tour: list[int], num_nodes: int) -> None:
    if sorted(tour) != list(range(num_nodes)):
        raise ValueError("tour must be a permutation of node indices")


def _scaled_integer_coords(coords: np.ndarray, scale: int = 1_000_000) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape [num_nodes, 2]")
    return np.rint(coords * scale).astype(np.int64)


def _write_tsplib(path: Path, coords: np.ndarray, name: str = "instance") -> None:
    scaled = _scaled_integer_coords(coords)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"NAME: {name}\n")
        handle.write("TYPE: TSP\n")
        handle.write(f"DIMENSION: {len(scaled)}\n")
        handle.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        handle.write("NODE_COORD_SECTION\n")
        for index, (x, y) in enumerate(scaled, start=1):
            handle.write(f"{index} {int(x)} {int(y)}\n")
        handle.write("EOF\n")


def _parse_tour_file(path: Path, num_nodes: int) -> list[int]:
    values: list[int] = []
    in_section = False
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "TOUR_SECTION":
                in_section = True
                continue
            if upper == "EOF":
                break
            if ":" in line and not in_section:
                continue
            for token in line.split():
                try:
                    value = int(token)
                except ValueError:
                    continue
                if value == -1:
                    break
                values.append(value)

    if len(values) == num_nodes + 1 and values[0] == num_nodes:
        values = values[1:]
    if len(values) != num_nodes:
        raise ExternalSolverError(
            f"Could not parse {num_nodes} tour nodes from solver output: {path}"
        )

    if min(values) >= 1 and max(values) <= num_nodes:
        tour = [value - 1 for value in values]
    elif min(values) >= 0 and max(values) < num_nodes:
        tour = values
    else:
        raise ExternalSolverError(f"Tour indices out of range in solver output: {path}")

    _validate_tour(tour, num_nodes)
    return tour


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


def solve_concorde(
    coords: np.ndarray,
    *,
    executable: str | None = None,
    timeout_sec: float | None = None,
) -> TspSolution:
    """Solve TSP with Concorde, preferring pyconcorde and falling back to CLI."""
    coords = np.asarray(coords, dtype=np.float64)
    if executable is None:
        try:
            tsp_module = importlib.import_module("concorde.tsp")
        except ImportError:
            pass
        else:
            scaled = _scaled_integer_coords(coords)
            solver = tsp_module.TSPSolver.from_data(
                scaled[:, 0], scaled[:, 1], norm="EUC_2D"
            )
            solution = solver.solve(verbose=False)
            tour = [int(node) for node in solution.tour]
            if len(tour) > 1 and tour[0] == tour[-1]:
                tour = tour[:-1]
            _validate_tour(tour, len(coords))
            return TspSolution(
                algorithm="concorde",
                tour=tour,
                cost=tour_length(coords, tour),
                is_exact=True,
            )

    executable = _resolve_executable(executable, "CONCORDE_EXECUTABLE", ("concorde",))
    if executable is None:
        raise ExternalSolverError(
            "Concorde is not available. Install pyconcorde or pass "
            "--concorde-executable /path/to/concorde."
        )

    with tempfile.TemporaryDirectory(prefix="tsp_concorde_") as tmp:
        tmpdir = Path(tmp)
        tsp_path = tmpdir / "instance.tsp"
        _write_tsplib(tsp_path, coords)
        result = subprocess.run(
            [executable, "-x", str(tsp_path)],
            cwd=tmpdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            raise ExternalSolverError(
                "Concorde failed with exit code "
                f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
            )
        tour_files = sorted(tmpdir.glob("*.sol")) + sorted(tmpdir.glob("*.tour"))
        if not tour_files:
            raise ExternalSolverError("Concorde did not produce a .sol or .tour file")
        tour = _parse_tour_file(tour_files[0], len(coords))
        return TspSolution(
            algorithm="concorde",
            tour=tour,
            cost=tour_length(coords, tour),
            is_exact=True,
        )


def solve_lkh3(
    coords: np.ndarray,
    *,
    executable: str | None = None,
    max_trials: int = 1000,
    runs: int = 10,
    seed: int | None = None,
    timeout_sec: float | None = None,
) -> TspSolution:
    coords = np.asarray(coords, dtype=np.float64)
    executable = _resolve_executable(
        executable,
        "LKH3_EXECUTABLE",
        ("LKH", "lkh", "LKH-3"),
    )
    if executable is None:
        raise ExternalSolverError(
            "LKH-3 is not available. Put the LKH executable on PATH, set "
            "LKH3_EXECUTABLE, or pass --lkh3-executable /path/to/LKH."
        )
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")

    with tempfile.TemporaryDirectory(prefix="tsp_lkh3_") as tmp:
        tmpdir = Path(tmp)
        problem_path = tmpdir / "instance.tsp"
        output_path = tmpdir / "instance.tour"
        param_path = tmpdir / "params.par"
        _write_tsplib(problem_path, coords)
        with open(param_path, "w", encoding="utf-8") as handle:
            handle.write(f"PROBLEM_FILE = {problem_path}\n")
            handle.write(f"OUTPUT_TOUR_FILE = {output_path}\n")
            handle.write(f"MAX_TRIALS = {max_trials}\n")
            handle.write(f"RUNS = {runs}\n")
            handle.write("TRACE_LEVEL = 0\n")
            if seed is not None:
                handle.write(f"SEED = {seed}\n")

        result = subprocess.run(
            [executable, str(param_path)],
            cwd=tmpdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            raise ExternalSolverError(
                "LKH-3 failed with exit code "
                f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
            )
        if not output_path.exists():
            raise ExternalSolverError("LKH-3 did not produce the configured tour file")
        tour = _parse_tour_file(output_path, len(coords))
        return TspSolution(
            algorithm="lkh3",
            tour=tour,
            cost=tour_length(coords, tour),
            is_exact=False,
            metadata={"max_trials": max_trials, "runs": runs},
        )


def solve_with_algorithms(
    coords: np.ndarray,
    *,
    algorithms: Sequence[str] = ("concorde", "lkh3"),
    concorde_executable: str | None = None,
    lkh3_executable: str | None = None,
    lkh3_trials: int = 1000,
    lkh3_runs: int = 10,
    seed: int | None = None,
    timeout_sec: float | None = None,
) -> dict[str, dict]:
    coords = np.asarray(coords, dtype=np.float64)
    solutions: dict[str, dict] = {}

    for algorithm in algorithms:
        normalized = algorithm.lower().replace("-", "_")
        if normalized == "concorde":
            solution = solve_concorde(
                coords,
                executable=concorde_executable,
                timeout_sec=timeout_sec,
            )
        elif normalized in ("lkh", "lkh3", "lkh_3"):
            solution = solve_lkh3(
                coords,
                executable=lkh3_executable,
                max_trials=lkh3_trials,
                runs=lkh3_runs,
                seed=seed,
                timeout_sec=timeout_sec,
            )
        else:
            raise ValueError(
                f"Unknown TSP solver algorithm: {algorithm}. Supported: concorde, lkh3."
            )
        solutions[solution.algorithm] = solution.to_record()

    return solutions
