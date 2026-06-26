import torch
from torch import nn

from models.pointer_network.attention import AdditiveAttention


class PointerDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ):
        super().__init__()

        self.lstmcell = nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.attention = AdditiveAttention(
            hidden_size=hidden_size,
        )
        self.initial_input = nn.Parameter(torch.empty(input_size))
        nn.init.normal_(self.initial_input, mean=0.0, std=0.01)

    def forward(
        self,
        input_list: torch.Tensor,
        encoder_output_list: torch.Tensor,
        initial_hidden: torch.Tensor,
        initial_cell: torch.Tensor,
        output_length: int,
        target_list: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
        allow_repeats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs: (batch_size, source_length, input_size)
        batch_size, source_length, _ = input_list.shape

        # Initial decoder input
        decoder_input = self.initial_input.expand(batch_size, -1)
        hidden = initial_hidden
        cell = initial_cell
        selected_mask = torch.zeros(
            size=(batch_size, source_length),
            dtype=torch.bool,
            device=input_list.device,
        )

        logits: list[torch.Tensor] = []
        pointers: list[torch.Tensor] = []

        for step in range(output_length):
            hidden, cell = self.lstmcell(decoder_input, (hidden, cell))
            # a = (a_1, ..., a_N)
            # a_i = P{C_i | C_1, ..., C_{i-1}, P ; theta}
            step_logits: torch.Tensor = self.attention(
                encoder_output_list=encoder_output_list,
                decoder_output=hidden,
                mask=None if allow_repeats else selected_mask,
            )
            # \hat C^P_i = argmax_j a_j
            step_pointers = step_logits.argmax(dim=1)

            logits.append(step_logits)
            pointers.append(step_pointers)

            next_indices = self._next_input_indices(
                step=step,
                predicted=step_pointers,
                target_list=target_list,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

            decoder_input = self._gather_inputs(input_list, next_indices)

            if not allow_repeats:
                selected_mask = selected_mask.scatter(
                    dim=1,
                    index=next_indices.unsqueeze(1),
                    value=True,
                )

        return torch.stack(logits, dim=1), torch.stack(pointers, dim=1)

    @staticmethod
    def _gather_inputs(
        input_list: torch.Tensor, next_indices: torch.Tensor
    ) -> torch.Tensor:
        gather_index = next_indices.view(-1, 1, 1).expand(-1, 1, input_list.size(-1))
        return input_list.gather(dim=1, index=gather_index).squeeze(1)

    @staticmethod
    def _next_input_indices(
        step: int,
        predicted: torch.Tensor,
        target_list: torch.Tensor | None,
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        if target_list is None or teacher_forcing_ratio <= 0.0:
            return predicted
        if teacher_forcing_ratio >= 1.0:
            return target_list[:, step]

        use_teacher = (
            torch.rand(
                predicted.shape,
                device=predicted.device,
            )
            < teacher_forcing_ratio
        )
        return torch.where(use_teacher, target_list[:, step], predicted)
