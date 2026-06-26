import torch
from torch import nn

from models.pointer_network.decoder import PointerDecoder
from models.pointer_network.encoder import PointerEncoder
from models.pointer_network.types import PointerNetworkOutput


class PointerNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = PointerEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = PointerDecoder(
            input_size=input_size,
            hidden_size=hidden_size,
        )

    def forward(
        self,
        input_list: torch.Tensor,
        target_list: torch.Tensor | None = None,
        output_length: int | None = None,
        teacher_forcing_ratio: float = 1.0,
        allow_repeats: bool = False,
    ) -> PointerNetworkOutput:

        # sanity check
        output_length = self._validate_inputs(
            input_list=input_list,
            target_list=target_list,
            output_length=output_length,
            teacher_forcing_ratio=teacher_forcing_ratio,
            allow_repeats=allow_repeats,
        )

        encoder_result: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.encoder(
            input_list
        )
        encoder_output_list, hidden_states, cell_states = encoder_result

        initial_hidden: torch.Tensor = hidden_states[-1]
        initial_cell: torch.Tensor = cell_states[-1]
        logit_list, pointer_list = self.decoder(
            input_list=input_list,
            encoder_output_list=encoder_output_list,
            initial_hidden=initial_hidden,
            initial_cell=initial_cell,
            output_length=output_length,
            target_list=target_list,
            teacher_forcing_ratio=teacher_forcing_ratio,
            allow_repeats=allow_repeats,
        )
        return PointerNetworkOutput(logit_list=logit_list, pointer_list=pointer_list)

    @staticmethod
    def loss(logit_list: torch.Tensor, target_list: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss for pointer logit_list."""
        if logit_list.ndim != 3:
            raise ValueError(
                "logit_list must have shape [batch, output_length, source_length]"
            )
        if target_list.shape != logit_list.shape[:2]:
            raise ValueError("target_list must have shape [batch, output_length]")
        return nn.functional.cross_entropy(
            input=logit_list.reshape(-1, logit_list.size(-1)),
            target=target_list.reshape(-1),
        )

    @staticmethod
    def _validate_inputs(
        input_list: torch.Tensor,
        target_list: torch.Tensor | None,
        output_length: int | None,
        teacher_forcing_ratio: float,
        allow_repeats: bool,
    ) -> int:
        if input_list.ndim != 3:
            raise ValueError(
                "input_list must have shape [batch, source_length, input_size]"
            )
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1]")

        if output_length is None:
            output_length = (
                target_list.size(dim=1)
                if target_list is not None
                else input_list.size(dim=1)
            )
        if output_length < 1:
            raise ValueError("output_length must be at least 1")
        if not allow_repeats and output_length > input_list.size(dim=1):
            raise ValueError(
                "output_length must be less than or equal to the input sequence length"
            )

        if target_list is None:
            return output_length
        if target_list.ndim != 2:
            raise ValueError("target_list must have shape [batch, output_length]")
        if target_list.dtype != torch.long:
            raise ValueError("target_list must be a torch.long tensor")
        if target_list.size(0) != input_list.size(0):
            raise ValueError("target_list batch size must match input_list batch size")
        if target_list.size(1) < output_length:
            raise ValueError("target_list must cover every requested decoding step")
        if target_list.min().item() < 0 or target_list.max().item() >= input_list.size(
            1
        ):
            raise ValueError("target_list must index positions in the input sequence")
        return output_length
