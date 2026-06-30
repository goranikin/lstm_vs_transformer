import copy
from collections.abc import Iterable

import torch
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import ttest_rel


class ExponentialBaseline:
    def __init__(self, beta: float = 0.8) -> None:
        self.beta = beta
        self.v: torch.Tensor | None = None

    def eval(self, cost: torch.Tensor) -> torch.Tensor:
        mean_cost = cost.detach().mean()
        if self.v is None:
            self.v = mean_cost
        else:
            self.v = self.beta * self.v + (1.0 - self.beta) * mean_cost
        return self.v.to(cost.device).expand_as(cost)


class RolloutBaseline(BaseModel):
    """Greedy rollout baseline for AM/PN policy-gradient training."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    problem: str
    device: torch.device
    alpha: float = Field(default=0.05, ge=0, le=1)
    baseline_model: torch.nn.Module | None = None

    def init_from(self, model: torch.nn.Module) -> None:
        self.baseline_model = copy.deepcopy(model).to(self.device)
        self.baseline_model.eval()
        if hasattr(self.baseline_model, "set_decode_type"):
            self.baseline_model.set_decode_type("greedy")

    @torch.no_grad()
    def greedy_costs(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        was_training = model.training
        previous_decode_type = getattr(model, "decode_type", None)
        model.eval()
        if hasattr(model, "set_decode_type"):
            model.set_decode_type("greedy")
        cost, _ = model(batch, problem=self.problem)
        if hasattr(model, "set_decode_type"):
            model.set_decode_type(previous_decode_type or "sampling")
        if was_training:
            model.train()
        return cost.detach()

    def eval_batch(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        warmup: ExponentialBaseline | None = None,
    ) -> torch.Tensor:
        if self.baseline_model is None:
            if warmup is None:
                cost, _ = model(batch, problem=self.problem)
                return cost.detach().mean().expand_as(cost)
            cost, _ = model(batch, problem=self.problem)
            return warmup.eval(cost)

        return self.greedy_costs(self.baseline_model, batch)

    @torch.no_grad()
    def maybe_update(
        self,
        model: torch.nn.Module,
        val_batches: Iterable[dict[str, torch.Tensor]],
        epoch: int,
        warmup_epochs: int,
    ) -> bool:
        if epoch < warmup_epochs:
            return False

        if self.baseline_model is None:
            self.init_from(model)
            return True

        candidate_costs = []
        baseline_costs = []
        for batch in val_batches:
            candidate_costs.append(self.greedy_costs(model, batch))
            baseline_costs.append(self.greedy_costs(self.baseline_model, batch))

        candidate = torch.cat(candidate_costs).cpu()
        baseline = torch.cat(baseline_costs).cpu()
        _, p_value = ttest_rel(candidate.numpy(), baseline.numpy())
        if p_value < self.alpha and candidate.mean() < baseline.mean():
            self.init_from(model)
            return True
        return False
