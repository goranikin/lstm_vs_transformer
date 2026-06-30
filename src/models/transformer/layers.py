import math

import torch
from torch import nn


def init_uniform_(param: torch.Tensor, d: int) -> None:
    bound = 1.0 / math.sqrt(d)
    nn.init.uniform_(param, -bound, bound)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim,
                out_features=ff_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=ff_dim,
                out_features=hidden_dim,
            ),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                init_uniform_(param=module.weight, d=module.in_features)
                init_uniform_(param=module.bias, d=module.bias.numel())

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


class SkipConnection(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x + self.module(x, *args, **kwargs)


class Normalization(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kind: str = "batch",
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "batch":
            self.norm = nn.BatchNorm1d(
                num_features=hidden_dim,
                affine=False,
            )
        elif kind == "instance":
            self.norm = nn.InstanceNorm1d(
                num_features=hidden_dim,
                affine=False,
            )
        else:
            raise ValueError(f"Unknown normalization: {kind}")

        self.w_bn = nn.Parameter(torch.ones(hidden_dim))
        self.b_bn = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, n, d_h)
        batch_size, n, hidden_dim = x.shape
        y = self.norm(x.reshape(batch_size * n, hidden_dim)).view(
            batch_size, n, hidden_dim
        )
        return self.w_bn * y + self.b_bn
