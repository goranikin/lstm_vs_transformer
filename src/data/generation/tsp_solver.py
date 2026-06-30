import os
import sys

import numpy as np

from configs.tsp_config import TspSolverConfig

_solver_config: TspSolverConfig | None = None


def _init_worker(config: dict) -> None:
    global _solver_config
    _solver_config = TspSolverConfig.model_validate(config)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)


def solve_tsp(nodes_coord: np.ndarray) -> list[int]:
    if _solver_config is None:
        raise RuntimeError("TSP solver config not initialized in worker process")

    num_nodes = nodes_coord.shape[0]

    if _solver_config.solver == "concorde":
        from concorde.tsp import TSPSolver

        scale = 1e6
        solver = TSPSolver.from_data(
            nodes_coord[:, 0] * scale,
            nodes_coord[:, 1] * scale,
            norm="EUC_2D",
        )
        solution = solver.solve(verbose=False)
        return list(solution.tour)

    if _solver_config.solver == "lkh":
        import lkh
        import tsplib95

        scale = 1e6
        lkh_path = "LKH-3.0.6/LKH"
        problem = tsplib95.models.StandardProblem(
            name="TSP",
            type="TSP",
            dimension=num_nodes,
            edge_weight_type="EUC_2D",
            node_coords={n + 1: nodes_coord[n] * scale for n in range(num_nodes)},
        )
        solution = lkh.solve(
            lkh_path,
            problem=problem,
            max_trials=_solver_config.lkh_trials,
            runs=10,
        )
        return [n - 1 for n in solution[0]]

    raise ValueError(f"Unknown solver: {_solver_config.solver}")
