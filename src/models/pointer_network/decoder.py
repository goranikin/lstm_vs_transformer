import torch
import torch.nn.functional as F
from torch import nn

from src.models.pointer_network.attention import AdditiveAttention


class PointerDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        tanh_clip: float = 10.0,
        n_glimpses: int = 1,
        mask_inner: bool = True,
        mask_logits: bool = True,
    ):
        super().__init__()
        self.n_glimpses = n_glimpses
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.lstmcell = nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.glimpse = AdditiveAttention(hidden_size=hidden_size)
        self.attention = AdditiveAttention(
            hidden_size=hidden_size,
            tanh_clip=tanh_clip,
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
        decode_type: str = "greedy",
        selected_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inputs: (batch_size, source_length, input_size)
        batch_size, source_length, _ = input_list.shape

        # Initial decoder input
        decoder_input = self.initial_input.expand(batch_size, -1)
        hidden = initial_hidden
        cell = initial_cell
        if selected_mask is None:
            selected_mask = torch.zeros(
                size=(batch_size, source_length),
                dtype=torch.bool,
                device=input_list.device,
            )

        logits: list[torch.Tensor] = []
        pointers: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []

        for step in range(output_length):
            hidden, cell = self.lstmcell(decoder_input, (hidden, cell))

            for _ in range(self.n_glimpses):
                glimpse_logits = self.glimpse(
                    encoder_output_list=encoder_output_list,
                    decoder_output=hidden,
                    mask=(
                        None if allow_repeats or not self.mask_inner else selected_mask
                    ),
                )
                glimpse_prob = F.softmax(glimpse_logits, dim=-1)
                glimpse_prob = torch.nan_to_num(glimpse_prob, nan=0.0)
                hidden = torch.einsum(
                    "bn,bnd->bd",
                    glimpse_prob,
                    encoder_output_list,
                )

            step_logits: torch.Tensor = self.attention(
                encoder_output_list=encoder_output_list,
                decoder_output=hidden,
                mask=(None if allow_repeats or not self.mask_logits else selected_mask),
            )

            step_pointers = self._select(step_logits, decode_type)
            next_indices = self._next_input_indices(
                step=step,
                predicted=step_pointers,
                target_list=target_list,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            step_log_prob = F.log_softmax(step_logits, dim=-1)
            step_log_prob = torch.nan_to_num(
                step_log_prob,
                nan=0.0,
                neginf=0.0,
            )
            step_log_prob = step_log_prob.gather(
                dim=1,
                index=next_indices.unsqueeze(1),
            ).squeeze(1)

            logits.append(step_logits)
            pointers.append(next_indices)
            log_probs.append(step_log_prob)

            decoder_input = self._gather_inputs(input_list, next_indices)

            if not allow_repeats:
                selected_mask = selected_mask.scatter(
                    dim=1,
                    index=next_indices.unsqueeze(1),
                    value=True,
                )

        return (
            torch.stack(logits, dim=1),
            torch.stack(pointers, dim=1),
            torch.stack(log_probs, dim=1).sum(dim=1),
        )

    @staticmethod
    def _select(logits: torch.Tensor, decode_type: str) -> torch.Tensor:
        if decode_type == "greedy":
            return logits.argmax(dim=1)
        if decode_type != "sampling":
            raise ValueError("decode_type must be 'greedy' or 'sampling'")
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        return torch.multinomial(probs, 1).squeeze(1)

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
