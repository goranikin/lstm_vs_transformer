import torch
from torch import nn

from src.models.transformer.attention_layer import AttentionLayer


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_h: int,
        d_ff: int,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            modules=[
                AttentionLayer(
                    n_heads=n_heads,
                    d_h=d_h,
                    d_ff=d_ff,
                    normalization=normalization,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h = layer(h)
        h_bar = h.mean(dim=1)
        return h, h_bar
