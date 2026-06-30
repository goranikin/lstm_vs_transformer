import os
from collections.abc import Iterable
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.models.transformer.baselines.baseline import ExponentialBaseline, RolloutBaseline
from src.training.utils import move_batch_to_device

ProblemName = Literal["tsp", "mis"]
OptimizerName = Literal["adam", "sgd"]
BaselineName = Literal["rollout", "exponential"]


class TrainerConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    problem: ProblemName
    output_dir: str = "outputs"
    n_epochs: int = Field(default=100, gt=0)
    steps_per_epoch: int | None = Field(default=None, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    learning_rate_decay: float = Field(default=1.0, gt=0, le=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    log_every: int = Field(default=50, gt=0)
    checkpoint_every: int = Field(default=1, gt=0)
    optimizer: OptimizerName = "adam"
    baseline: BaselineName = "rollout"
    baseline_alpha: float = Field(default=0.05, ge=0, le=1)
    baseline_warmup_epochs: int = Field(default=1, ge=0)
    exp_baseline_beta: float = Field(default=0.8, ge=0, le=1)


class SupervisedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        config: TrainerConfig,
        device: torch.device,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = self._build_optimizer()

    def fit(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        for epoch in range(self.config.n_epochs):
            if self.config.learning_rate_decay != 1.0:
                self._set_learning_rate(
                    self.config.learning_rate * (self.config.learning_rate_decay**epoch)
                )
            train_loss = self.train_epoch(epoch)
            message = f"epoch={epoch + 1} train_loss={train_loss:.6f}"
            if self.val_loader is not None:
                val_loss = self.evaluate_loss()
                message += f" val_loss={val_loss:.6f}"
            print(message)
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch + 1)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for step, batch in enumerate(self._epoch_batches(self.train_loader), start=1):
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model.supervised_loss(batch, problem=self.config.problem)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            steps += 1
            if step % self.config.log_every == 0:
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"loss={total_loss / steps:.6f}"
                )
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def evaluate_loss(self) -> float:
        if self.val_loader is None:
            raise ValueError("val_loader is not configured")
        self.model.eval()
        total = 0.0
        count = 0
        for batch in self.val_loader:
            batch = move_batch_to_device(batch, self.device)
            loss = self.model.supervised_loss(batch, problem=self.config.problem)
            batch_size = self._batch_size(batch)
            total += float(loss.item()) * batch_size
            count += batch_size
        return total / max(count, 1)

    def save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.config.output_dir, f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.model_dump(),
            },
            path,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.config.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _epoch_batches(
        self,
        loader: DataLoader,
    ) -> Iterable[dict[str, torch.Tensor]]:
        if self.config.steps_per_epoch is None:
            yield from loader
            return

        iterator = iter(loader)
        for _ in range(self.config.steps_per_epoch):
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(loader)
                yield next(iterator)

    @staticmethod
    def _batch_size(batch: dict[str, torch.Tensor]) -> int:
        for key in ("loc", "adjacency"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.size(0))
        raise ValueError("Cannot infer batch size")


class RLTrainer(SupervisedTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        config: TrainerConfig,
        device: torch.device,
        val_loader: DataLoader | None = None,
    ) -> None:
        super().__init__(model, train_loader, config, device, val_loader)
        self.exp_baseline = ExponentialBaseline(beta=config.exp_baseline_beta)
        self.rollout_baseline = RolloutBaseline(
            problem=config.problem,
            device=device,
            alpha=config.baseline_alpha,
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        if hasattr(self.model, "set_decode_type"):
            self.model.set_decode_type("sampling")
        total_cost = 0.0
        steps = 0
        for step, batch in enumerate(self._epoch_batches(self.train_loader), start=1):
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            cost, log_likelihood = self.model(batch, problem=self.config.problem)
            baseline = self._baseline_value(cost, batch, epoch)
            loss = ((cost - baseline).detach() * log_likelihood).mean()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_cost += float(cost.detach().mean().item())
            steps += 1
            if step % self.config.log_every == 0:
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"cost={total_cost / steps:.6f} loss={loss.item():.6f}"
                )
        return total_cost / max(steps, 1)

    def fit(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        for epoch in range(self.config.n_epochs):
            if self.config.learning_rate_decay != 1.0:
                self._set_learning_rate(
                    self.config.learning_rate * (self.config.learning_rate_decay**epoch)
                )
            train_cost = self.train_epoch(epoch)
            message = f"epoch={epoch + 1} train_cost={train_cost:.6f}"
            if self.val_loader is not None:
                val_cost = self.evaluate_cost()
                message += f" val_greedy_cost={val_cost:.6f}"
                if self.config.baseline == "rollout":
                    updated = self.rollout_baseline.maybe_update(
                        self.model,
                        self._validation_batches(),
                        epoch,
                        self.config.baseline_warmup_epochs,
                    )
                    message += f" rollout_updated={updated}"
            print(message)
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch + 1)

    @torch.no_grad()
    def evaluate_cost(self) -> float:
        if self.val_loader is None:
            raise ValueError("val_loader is not configured")
        self.model.eval()
        if hasattr(self.model, "set_decode_type"):
            self.model.set_decode_type("greedy")
        total = 0.0
        count = 0
        for batch in self.val_loader:
            batch = move_batch_to_device(batch, self.device)
            cost, _ = self.model(batch, problem=self.config.problem)
            batch_size = self._batch_size(batch)
            total += float(cost.sum().item())
            count += batch_size
        if hasattr(self.model, "set_decode_type"):
            self.model.set_decode_type("sampling")
        return total / max(count, 1)

    def _baseline_value(
        self,
        cost: torch.Tensor,
        batch: dict[str, torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        if self.config.baseline == "exponential":
            return self.exp_baseline.eval(cost)
        if self.config.baseline != "rollout":
            raise ValueError(f"Unsupported baseline: {self.config.baseline}")
        if epoch < self.config.baseline_warmup_epochs:
            return self.exp_baseline.eval(cost)
        if self.rollout_baseline.baseline_model is None:
            self.rollout_baseline.init_from(self.model)
        return self.rollout_baseline.eval_batch(self.model, batch)

    def _validation_batches(self) -> Iterable[dict[str, torch.Tensor]]:
        if self.val_loader is None:
            return []
        return (
            move_batch_to_device(batch, self.device)
            for batch in self.val_loader
        )
