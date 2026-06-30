from typing import Self

import numpy as np

from src.data.generation.constants import COORD_SCALE, OUTPUT_MARKER, ROUTE_SEPARATOR
from src.data.generation.types.base import Schema


class Coordinate(Schema):
    x: float
    y: float


class TspInstance(Schema):
    coordinates: list[Coordinate]

    @classmethod
    def from_coords_array(cls, coords: np.ndarray) -> Self:
        return cls(
            coordinates=[
                Coordinate(x=float(x), y=float(y)) for x, y in coords.tolist()
            ]
        )

    @property
    def num_nodes(self) -> int:
        return len(self.coordinates)


class TspTour(Schema):
    nodes: list[int]

    @classmethod
    def from_solver_tour(cls, tour: list[int]) -> Self:
        if len(tour) > 1 and tour[0] == tour[-1]:
            tour = tour[:-1]
        return cls(nodes=tour)

    def is_valid(self, num_nodes: int) -> bool:
        return len(self.nodes) == num_nodes and sorted(self.nodes) == list(
            range(num_nodes)
        )


class TspSample(Schema):
    instance: TspInstance
    tour: TspTour

    def to_line(self) -> str:
        coord_parts: list[str] = []
        for coordinate in self.instance.coordinates:
            coord_parts.append(str(int(round(coordinate.x * COORD_SCALE))))
            coord_parts.append(str(int(round(coordinate.y * COORD_SCALE))))
        route = ROUTE_SEPARATOR.join(str(node) for node in self.tour.nodes)
        return " ".join(coord_parts) + OUTPUT_MARKER + route + "\n"
