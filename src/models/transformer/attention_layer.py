import math

import torch
from torch import nn

from src.models.transformer.layers import (
    FeedForward,
    Normalization,
    SkipConnection,
    init_uniform_,
)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_h: int,
        input_dim: int | None = None,
        d_k: int | None = None,
        d_v: int | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_h = d_h
        self.input_dim = input_dim or d_h
        self.d_k = d_k or d_h // n_heads
        self.d_v = d_v or d_h // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.W_query = nn.Parameter(torch.empty(n_heads, self.input_dim, self.d_k))
        self.W_key = nn.Parameter(torch.empty(n_heads, self.input_dim, self.d_k))
        self.W_val = nn.Parameter(torch.empty(n_heads, self.input_dim, self.d_v))
        self.W_out = nn.Parameter(torch.empty(n_heads, self.d_v, d_h))

        for param in self.parameters():
            init_uniform_(param, param.size(-1))

    def forward(
        self,
        q: torch.Tensor,
        h: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h is None:
            h = q
        if value is None:
            value = h

        # Projections (eq. 10): b=batch, h=head, i=input_dim, k/d_k, v/d_v, n/key len, q/query len
        Q = torch.einsum("bqi,hik->hbqk", q, self.W_query)
        K = torch.einsum("bni,hik->hbnk", h, self.W_key)
        V = torch.einsum("bni,hiv->hbnv", value, self.W_val)

        # u_ij = q_i^T k_j / sqrt(d_k)  (eq. 11)
        compatibility = self.scale * torch.einsum("hbqk,hbnk->hbqn", Q, K)

        if mask is not None:
            compatibility = compatibility.masked_fill(
                mask.unsqueeze(0).expand_as(compatibility),
                float("-inf"),
            )

        # a_ij = softmax(u_ij)  (eq. 12)
        attn = torch.softmax(compatibility, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).expand_as(attn), 0.0)

        # h'_i = sum_j a_ij v_j  (eq. 13)
        heads = torch.einsum("hbqn,hbnv->hbqv", attn, V)

        out = torch.einsum("hbqv,hvd->bqd", heads, self.W_out)

        return out


class AttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_h: int,
        d_ff: int,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        self.mha = SkipConnection(
            MultiHeadAttention(n_heads=n_heads, d_h=d_h, input_dim=d_h)
        )
        self.norm1 = Normalization(d_h, normalization)
        self.ff = SkipConnection(FeedForward(d_h, d_ff))
        self.norm2 = Normalization(d_h, normalization)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # h: (B, n, d_h)
        h_hat = self.norm1(self.mha(h, mask=mask))
        return self.norm2(self.ff(h_hat))


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

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h = layer(h, mask=mask)
        h_bar = h.mean(dim=1)
        return h, h_bar
