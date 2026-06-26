import torch
from torch import nn

from src.models.transformer.layers import init_uniform_


class AdditiveAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()
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
        u = torch.einsum(
            "bne,oe->bno",
            torch.tanh(e_proj + d_proj),
            self.W_score,
        ).squeeze(-1)
        if mask is not None:
            u = u.masked_fill(mask, torch.finfo(u.dtype).min)
        return u
