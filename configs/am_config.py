from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AMModelConfig(BaseModel):
    """Hyperparameters from Section 5 of the paper."""

    model_config = ConfigDict(validate_assignment=True)

    d_h: int = Field(default=128, gt=0)
    n_layers: int = Field(default=3, gt=0)
    n_heads: int = Field(default=8, gt=0)
    d_ff: int = Field(default=512, gt=0)
    tanh_clip: float = Field(default=10.0, ge=0)
    normalization: Literal["batch", "instance"] = "batch"

    @model_validator(mode="after")
    def validate_attention_dimensions(self) -> Self:
        if self.d_h % self.n_heads != 0:
            raise ValueError("d_h must be divisible by n_heads")
        return self


class AMTrainingConfig(BaseModel):
    """Training hyperparameters (Algorithm 1, Section 5)."""

    model_config = ConfigDict(validate_assignment=True)

    n_epochs: int = Field(default=100, gt=0)
    steps_per_epoch: int = Field(default=2500, gt=0)
    batch_size: int = Field(default=512, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    seed: int = 1234
    baseline_alpha: float = Field(default=0.05, ge=0, le=1)
    baseline_warmup_epochs: int = Field(default=1, ge=0)
    exp_baseline_beta: float = Field(default=0.8, ge=0, le=1)
    val_size: int = Field(default=10_000, gt=0)
    eval_batch_size: int = Field(default=1024, gt=0)
    n_sample_eval: int = Field(default=1280, gt=0)
    baseline: Literal["rollout", "exponential"] = "rollout"
    log_every: int = Field(default=50, gt=0)
    checkpoint_every: int = Field(default=1, gt=0)
    output_dir: str = "outputs"
