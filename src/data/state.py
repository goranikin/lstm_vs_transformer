"""Decoding state for routing problems."""

import torch
from pydantic import BaseModel, ConfigDict


class _RoutingState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)


class TSPState(_RoutingState):
    """State for Euclidean TSP (Section 3)."""

    loc: torch.Tensor
    i: int
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited: torch.Tensor

    @classmethod
    def initialize(cls, loc: torch.Tensor) -> "TSPState":
        batch_size, n, _ = loc.shape
        device = loc.device
        return cls(
            loc=loc,
            i=0,
            first_a=torch.zeros(batch_size, dtype=torch.long, device=device),
            prev_a=torch.zeros(batch_size, dtype=torch.long, device=device),
            visited=torch.zeros(batch_size, 1, n, device=device, dtype=torch.bool),
        )

    def get_mask(self) -> torch.Tensor:
        return self.visited

    def get_context_features(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return self.first_a, self.prev_a, self.i == 0

    def update(self, selected: torch.Tensor) -> None:
        if self.i == 0:
            self.first_a = selected
        self.prev_a = selected
        self.visited = self.visited.scatter(-1, selected[:, None, None], True)
        self.i += 1

    def all_finished(self) -> bool:
        return self.i >= self.loc.size(1)

    @staticmethod
    def get_final_cost(loc: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        d = loc.gather(1, pi.unsqueeze(-1).expand_as(loc))
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (
            d[:, 0] - d[:, -1]
        ).norm(p=2, dim=1)


class VRPState(_RoutingState):
    """State for CVRP / SDVRP (Appendix C)."""

    loc: torch.Tensor
    demand: torch.Tensor
    i: int
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited: torch.Tensor
    remaining_demand: torch.Tensor
    allow_split: bool

    @classmethod
    def initialize(
        cls,
        loc: torch.Tensor,
        demand: torch.Tensor,
        allow_split: bool = False,
    ) -> "VRPState":
        batch_size = loc.size(0)
        device = loc.device
        return cls(
            loc=loc,
            demand=demand,
            i=0,
            prev_a=torch.zeros(batch_size, dtype=torch.long, device=device),
            used_capacity=torch.zeros(batch_size, device=device),
            visited=torch.zeros(
                batch_size, 1, loc.size(1), device=device, dtype=torch.bool
            ),
            remaining_demand=demand.clone(),
            allow_split=allow_split,
        )

    def get_mask(self) -> torch.Tensor:
        batch_size, n = self.loc.size(0), self.loc.size(1)
        mask = torch.zeros(batch_size, 1, n, device=self.loc.device, dtype=torch.bool)

        capacity_left = (1.0 - self.used_capacity).unsqueeze(-1)

        if self.i > 0:
            mask[:, :, 0] = (self.prev_a == 0).unsqueeze(1)

        rd = self.remaining_demand[:, 1:]
        if self.allow_split:
            infeasible = rd <= 0
        else:
            infeasible = (rd <= 0) | (rd > capacity_left)
        mask[:, :, 1:] = infeasible.unsqueeze(1)

        return mask

    def get_context_features(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return self.prev_a, 1.0 - self.used_capacity, self.i == 0

    def update(self, selected: torch.Tensor) -> None:
        batch_size = self.loc.size(0)
        batch_idx = torch.arange(batch_size, device=self.loc.device)
        is_depot = selected == 0

        demand_at = self.remaining_demand[batch_idx, selected]
        capacity_left = (1.0 - self.used_capacity).clamp(min=0)
        delivered = torch.where(
            is_depot,
            torch.zeros_like(capacity_left),
            torch.minimum(demand_at, capacity_left),
        )

        self.remaining_demand[batch_idx, selected] = (
            self.remaining_demand[batch_idx, selected] - delivered
        ).clamp(min=0)

        self.used_capacity = torch.where(
            is_depot,
            torch.zeros_like(self.used_capacity),
            (self.used_capacity + delivered).clamp(max=1.0),
        )

        if not self.allow_split:
            self.visited = self.visited.scatter(
                -1, selected[:, None, None], (~is_depot)[:, None, None]
            )

        self.prev_a = selected
        self.i += 1

    @staticmethod
    def get_final_cost(loc: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(loc.size(0), device=loc.device)
        coords = loc[batch_idx.unsqueeze(1), pi]
        return (coords[:, 1:] - coords[:, :-1]).norm(p=2, dim=2).sum(1)


class OPState(_RoutingState):
    """State for Orienteering Problem (Appendix D)."""

    loc: torch.Tensor
    prize: torch.Tensor
    max_length: float
    i: int
    prev_a: torch.Tensor
    length_used: torch.Tensor
    visited: torch.Tensor

    @classmethod
    def initialize(
        cls,
        loc: torch.Tensor,
        prize: torch.Tensor,
        max_length: float,
    ) -> "OPState":
        batch_size = loc.size(0)
        device = loc.device
        return cls(
            loc=loc,
            prize=prize,
            max_length=max_length,
            i=0,
            prev_a=torch.zeros(batch_size, dtype=torch.long, device=device),
            length_used=torch.zeros(batch_size, device=device),
            visited=torch.zeros(
                batch_size, 1, loc.size(1), device=device, dtype=torch.bool
            ),
        )

    def get_mask(self) -> torch.Tensor:
        batch_size = self.loc.size(0)
        mask = self.visited.clone()
        batch_idx = torch.arange(batch_size, device=self.loc.device)
        remaining = self.max_length - self.length_used
        prev_loc = self.loc[batch_idx, self.prev_a]
        to_loc = self.loc
        d_prev = (to_loc - prev_loc.unsqueeze(1)).norm(p=2, dim=-1)
        d_depot = (self.loc[:, 0].unsqueeze(1) - to_loc).norm(p=2, dim=-1)
        infeasible = d_prev + d_depot > remaining.unsqueeze(1)
        mask = mask | infeasible.unsqueeze(1)
        return mask

    def get_context_features(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return self.prev_a, self.max_length - self.length_used, self.i == 0

    def update(self, selected: torch.Tensor) -> None:
        if self.i > 0:
            batch_idx = torch.arange(self.loc.size(0), device=self.loc.device)
            step_dist = (
                self.loc[batch_idx, selected] - self.loc[batch_idx, self.prev_a]
            ).norm(p=2, dim=-1)
            self.length_used = self.length_used + step_dist
        self.visited = self.visited.scatter(-1, selected[:, None, None], True)
        self.prev_a = selected
        self.i += 1

    @staticmethod
    def get_final_cost(
        loc: torch.Tensor, pi: torch.Tensor, prize: torch.Tensor
    ) -> torch.Tensor:
        batch_idx = torch.arange(loc.size(0), device=loc.device)
        collected = prize[batch_idx.unsqueeze(1), pi].sum(1)
        return -collected


class PCTSPState(_RoutingState):
    """State for Prize Collecting TSP (Appendix E)."""

    loc: torch.Tensor
    prize: torch.Tensor
    penalty: torch.Tensor
    i: int
    prev_a: torch.Tensor
    remaining_prize: torch.Tensor
    visited: torch.Tensor
    n_customers: int

    @classmethod
    def initialize(
        cls,
        loc: torch.Tensor,
        prize: torch.Tensor,
        penalty: torch.Tensor,
    ) -> "PCTSPState":
        batch_size = loc.size(0)
        device = loc.device
        return cls(
            loc=loc,
            prize=prize,
            penalty=penalty,
            i=0,
            prev_a=torch.zeros(batch_size, dtype=torch.long, device=device),
            remaining_prize=torch.ones(batch_size, device=device),
            visited=torch.zeros(
                batch_size, 1, loc.size(1), device=device, dtype=torch.bool
            ),
            n_customers=loc.size(1) - 1,
        )

    def get_mask(self) -> torch.Tensor:
        mask = self.visited.clone()
        must_collect = (self.remaining_prize > 0) & (self.i <= self.n_customers)
        mask[:, :, 0] = mask[:, :, 0] | must_collect.unsqueeze(1)
        return mask

    def get_context_features(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return self.prev_a, self.remaining_prize, self.i == 0

    def update(
        self, selected: torch.Tensor, real_prize: torch.Tensor | None = None
    ) -> None:
        batch_idx = torch.arange(self.loc.size(0), device=self.loc.device)
        p = real_prize if real_prize is not None else self.prize[batch_idx, selected]
        self.remaining_prize = (self.remaining_prize - p).clamp(min=0)
        self.visited = self.visited.scatter(-1, selected[:, None, None], True)
        self.prev_a = selected
        self.i += 1

    @staticmethod
    def get_final_cost(
        loc: torch.Tensor,
        pi: torch.Tensor,
        prize: torch.Tensor,
        penalty: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n = loc.size(0), loc.size(1)
        batch_idx = torch.arange(batch_size, device=loc.device)
        visited = torch.zeros(batch_size, n, dtype=torch.bool, device=loc.device)
        visited[batch_idx.unsqueeze(1), pi] = True

        coords = loc[batch_idx.unsqueeze(1), pi]
        tour_len = (coords[:, 1:] - coords[:, :-1]).norm(p=2, dim=2).sum(1)

        unvisited = visited.clone()
        unvisited[:, 0] = False
        penalty_sum = (penalty * unvisited.float()).sum(1)
        return tour_len + penalty_sum
