import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()
        self.encoder_proj = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
        )
        self.decoder_proj = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
        )
        self.score_projection = nn.Linear(
            in_features=hidden_size,
            out_features=1,
            bias=False,
        )

    def forward(
        self,
        encoder_output_list: torch.Tensor,
        decoder_output: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # e = (e1, ..., eN), where N is the source sequence length
        e_proj = self.encoder_proj(encoder_output_list)
        d_i_proj = self.decoder_proj(decoder_output).unsqueeze(1)

        # u_i = v^T * tanh(W1@e + W2@d_i)
        u = self.score_projection(torch.tanh(e_proj + d_i_proj)).squeeze(1)
        if mask is not None:
            u = u.masked_fill(mask, torch.finfo(u.dtype).min)
        return u
