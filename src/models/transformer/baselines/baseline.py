import copy

import torch
from pydantic import BaseModel, ConfigDict
from scipy.stats import ttest_rel

from src.data.registry import ProblemSpec, sample_batch
from src.models.transformer.model import AttentionModel


class ExponentialBaseline:
    def __init__(self, beta: float = 0.8) -> None:
        self.beta = beta
        self.v: torch.Tensor | None = None

    def eval(self, cost: torch.Tensor) -> torch.Tensor:
        mean_cost = cost.mean()
        if self.v is None:
            self.v = mean_cost.detach()
        else:
            self.v = self.beta * self.v + (1.0 - self.beta) * mean_cost.detach()
        return self.v.expand_as(cost)


class RolloutBaseline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    problem: ProblemSpec
    graph_size: int
    device: torch.device
    alpha: float = 0.05
    val_size: int = 10_000
    baseline_model: AttentionModel | None = None
    val_batch: dict[str, torch.Tensor] | None = None

    def _ensure_val_batch(self) -> dict[str, torch.Tensor]:
        if self.val_batch is None:
            self.val_batch = sample_batch(
                self.problem, self.val_size, self.graph_size, self.device
            )
        return self.val_batch

    def init_from(self, model: AttentionModel) -> None:
        self.baseline_model = copy.deepcopy(model).to(self.device)
        self.baseline_model.eval()
        self.baseline_model.set_decode_type("greedy")

    @torch.no_grad()
    def greedy_costs(
        self, model: AttentionModel, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        model.eval()
        model.set_decode_type("greedy")
        cost, _ = model(batch)
        return cost.detach()

    def eval_batch(
        self,
        model: AttentionModel,
        batch: dict[str, torch.Tensor],
        warmup: ExponentialBaseline | None = None,
    ) -> torch.Tensor:
        if self.baseline_model is None:
            if warmup is None:
                return torch.zeros(batch["loc"].size(0), device=self.device)
            return warmup.eval(torch.zeros(batch["loc"].size(0), device=self.device))

        return self.greedy_costs(self.baseline_model, batch)

    @torch.no_grad()
    def maybe_update(
        self, model: AttentionModel, epoch: int, warmup_epochs: int
    ) -> None:
        if epoch < warmup_epochs:
            return

        if self.baseline_model is None:
            self.init_from(model)
            self.val_batch = None
            return

        val = self._ensure_val_batch()
        candidate = self.greedy_costs(model, val)
        baseline = self.greedy_costs(self.baseline_model, val)

        _, p_value = ttest_rel(candidate.cpu().numpy(), baseline.cpu().numpy())
        if p_value < self.alpha and candidate.mean() < baseline.mean():
            self.init_from(model)
            self.val_batch = None
