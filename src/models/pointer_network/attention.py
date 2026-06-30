import torch
from torch import nn

from src.models.transformer.layers import init_uniform_


class AdditiveAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        tanh_clip: float | None = None,
    ):
        super().__init__()
        self.tanh_clip = tanh_clip
        self.W_encoder = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_decoder = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_score = nn.Parameter(torch.empty(1, hidden_size))

        for param in self.parameters():
            init_uniform_(param, param.size(-1))

    def forward(
        self,
        encoder_output_list: torch.Tensor,
        decoder_output: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Projections: b=batch, n=source len, d=hidden, e=proj hidden, o=1
        e_proj = torch.einsum("bnd,ed->bne", encoder_output_list, self.W_encoder)
        d_proj = torch.einsum("bd,ed->be", decoder_output, self.W_decoder).unsqueeze(1)

        # u_i = v^T * tanh(W1@e + W2@d_i)
        # (b, n, d) x (1, d) -> (b, n, 1) -> (b, n)
        u = torch.einsum(
            "bne,oe->bno",
            torch.tanh(e_proj + d_proj),
            self.W_score,
        ).squeeze(-1)
        if self.tanh_clip is not None and self.tanh_clip > 0:
            u = self.tanh_clip * torch.tanh(u)
        if mask is not None:
            u = u.masked_fill(mask, float("-inf"))
        return u
