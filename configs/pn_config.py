from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PNModelConfig(BaseModel):
    """Pointer Network architecture for AM-paper benchmarks (Section 5).

    Uses a single-layer LSTM encoder with additive attention and one glimpse,
    matching the reference implementation in attention-learn-to-route.
    """

    model_config = ConfigDict(validate_assignment=True)

    hidden_size: int = Field(default=128, gt=0)
    num_layers: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.0, ge=0, le=1)
    tanh_clip: float = Field(default=10.0, ge=0)
    n_glimpses: int = Field(default=1, ge=0)
    mask_inner: bool = True
    mask_logits: bool = True


class PNTrainingConfig(BaseModel):
    """REINFORCE training for Pointer Network (Algorithm 1, Section 5; Appendix B.5).

    PN is trained with a higher initial learning rate and per-epoch decay
    (η = 10⁻³ × 0.96^epoch), unlike the Attention Model which uses η = 10⁻⁴.
    """

    model_config = ConfigDict(validate_assignment=True)

    n_epochs: int = Field(default=100, gt=0)
    steps_per_epoch: int = Field(default=2500, gt=0)
    batch_size: int = Field(default=512, gt=0)
    batch_size_n100: int = Field(
        default=256,
        gt=0,
        description="Batch size for VRP/SDVRP with n=100 (memory constraint, Section 5).",
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0,
        description="Initial η; decayed each epoch for PN (Section 5.2).",
    )
    learning_rate_decay: float = Field(
        default=0.96,
        gt=0,
        le=1,
        description="Per-epoch multiplier applied to η (Section 5.2).",
    )
    max_grad_norm: float = Field(default=1.0, gt=0)
    seed: int = 1234
    teacher_forcing_ratio: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="0 during REINFORCE training (sampling decode).",
    )
    baseline: Literal["rollout", "critic", "exponential"] = "rollout"
    baseline_alpha: float = Field(default=0.05, ge=0, le=1)
    baseline_warmup_epochs: int = Field(default=1, ge=0)
    exp_baseline_beta: float = Field(default=0.8, ge=0, le=1)
    val_size: int = Field(default=10_000, gt=0)
    eval_batch_size: int = Field(default=1024, gt=0)
    n_sample_eval: int = Field(
        default=1280,
        gt=0,
        description="Number of sampled solutions at test time (Section 5).",
    )
    log_every: int = Field(default=50, gt=0)
    checkpoint_every: int = Field(default=1, gt=0)
    output_dir: str = "outputs"

    def learning_rate_at(self, epoch: int) -> float:
        """η after `epoch` completed epochs (0-indexed)."""
        return self.learning_rate * (self.learning_rate_decay**epoch)

    def batch_size_for(self, problem: str, graph_size: int) -> int:
        """Effective batch size; VRP n=100 uses a smaller batch (Section 5)."""
        if problem in ("cvrp", "sdvrp") and graph_size == 100:
            return self.batch_size_n100
        return self.batch_size


class PNBenchmarkConfig(BaseModel):
    """Problem instance settings for AM-paper benchmark reproduction."""

    model_config = ConfigDict(validate_assignment=True)

    problem: Literal["tsp", "cvrp", "sdvrp", "op", "pctsp", "spctsp"] = "tsp"
    graph_size: Literal[20, 50, 100] = 20
    op_distribution: Literal["distance", "const", "uniform"] = "distance"
