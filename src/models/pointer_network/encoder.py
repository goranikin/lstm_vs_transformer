import torch
from torch import nn


class PointerEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        input_list: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # output, (hidden states, cell states) = lstm(input_list)
        output, (hidden_states, cell_states) = self.lstm(input_list)
        return output, hidden_states, cell_states
