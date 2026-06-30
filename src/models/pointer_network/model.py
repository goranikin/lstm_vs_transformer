from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from configs.pn_config import PNModelConfig
from src.models.pointer_network.decoder import PointerDecoder
from src.models.pointer_network.encoder import PointerEncoder
from src.models.pointer_network.types import PointerNetworkOutput
from src.models.transformer.layers import init_uniform_

DecodeType = Literal["greedy", "sampling"]
ProblemType = Literal["tsp", "mis"]


class PointerNetwork(nn.Module):
    """Pointer Network with TSP/MIS helpers for supervised and RL experiments."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        config: PNModelConfig | None = None,
        default_problem: ProblemType | None = None,
        loc_key: str = "loc",
        adjacency_key: str = "adjacency",
        target_tour_key: str = "target_tour",
        target_set_key: str = "target_set",
    ) -> None:
        super().__init__()
        self.config = config or PNModelConfig()
        if input_size < 1:
            raise ValueError("input_size must be at least 1")
        self.input_size = input_size
        self.hidden_size = hidden_size or self.config.hidden_size
        self.num_layers = num_layers or self.config.num_layers
        self.dropout = self.config.dropout if dropout is None else dropout
        self.default_problem = default_problem
        self.loc_key = loc_key
        self.adjacency_key = adjacency_key
        self.target_tour_key = target_tour_key
        self.target_set_key = target_set_key

        self.encoder = PointerEncoder(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.decoder = PointerDecoder(
            input_size=input_size,
            hidden_size=self.hidden_size,
            tanh_clip=self.config.tanh_clip,
            n_glimpses=self.config.n_glimpses,
            mask_inner=self.config.mask_inner,
            mask_logits=self.config.mask_logits,
        )

        self.mis_stop_input = nn.Parameter(torch.empty(input_size))
        self.mis_classifier = nn.Linear(self.hidden_size, 1)
        nn.init.normal_(self.mis_stop_input, mean=0.0, std=0.01)
        init_uniform_(self.mis_classifier.weight, self.hidden_size)
        init_uniform_(self.mis_classifier.bias, self.mis_classifier.bias.numel())

        self.decode_type: DecodeType = "sampling"

    def set_decode_type(self, decode_type: DecodeType) -> None:
        if decode_type not in ("greedy", "sampling"):
            raise ValueError("decode_type must be 'greedy' or 'sampling'")
        self.decode_type = decode_type

    def forward(
        self,
        batch: dict[str, torch.Tensor] | torch.Tensor,
        problem: ProblemType | None = None,
        return_pi: bool = False,
        target_list: torch.Tensor | None = None,
        output_length: int | None = None,
        teacher_forcing_ratio: float = 0.0,
        allow_repeats: bool = False,
    ) -> (
        PointerNetworkOutput
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if isinstance(batch, torch.Tensor):
            return self.decode_sequence(
                batch,
                target_list=target_list,
                output_length=output_length,
                teacher_forcing_ratio=teacher_forcing_ratio,
                allow_repeats=allow_repeats,
            )

        problem = self._resolve_problem(batch, problem)
        if problem == "tsp":
            return self.solve_tsp(batch, return_pi=return_pi)
        if problem == "mis":
            return self.solve_mis(batch, return_pi=return_pi)
        raise ValueError(f"Unsupported problem: {problem}")

    def decode_sequence(
        self,
        input_list: torch.Tensor,
        target_list: torch.Tensor | None = None,
        output_length: int | None = None,
        teacher_forcing_ratio: float = 0.0,
        allow_repeats: bool = False,
    ) -> PointerNetworkOutput:
        output_length = self._validate_inputs(
            input_list=input_list,
            target_list=target_list,
            output_length=output_length,
            teacher_forcing_ratio=teacher_forcing_ratio,
            allow_repeats=allow_repeats,
        )

        encoder_output_list, hidden_states, cell_states = self.encoder(input_list)
        logit_list, pointer_list, log_likelihood = self.decoder(
            input_list=input_list,
            encoder_output_list=encoder_output_list,
            initial_hidden=hidden_states[-1],
            initial_cell=cell_states[-1],
            output_length=output_length,
            target_list=target_list,
            teacher_forcing_ratio=teacher_forcing_ratio,
            allow_repeats=allow_repeats,
            decode_type=self.decode_type,
        )
        return PointerNetworkOutput(
            logit_list=logit_list,
            pointer_list=pointer_list,
            log_likelihood=log_likelihood,
        )

    def solve_tsp(
        self,
        batch: dict[str, torch.Tensor],
        return_pi: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        loc = self._require_tensor(batch, self.loc_key)
        if loc.ndim != 3 or loc.size(-1) != self.input_size:
            raise ValueError(
                f"TSP batch['{self.loc_key}'] must have shape "
                f"[batch, nodes, {self.input_size}]"
            )

        output = self.decode_sequence(loc, teacher_forcing_ratio=0.0)
        if output.log_likelihood is None:
            raise RuntimeError("Pointer decoder did not return log likelihood")
        cost = self.tsp_cost(loc, output.pointer_list)

        if return_pi:
            return cost, output.log_likelihood, output.pointer_list
        return cost, output.log_likelihood

    def solve_mis(
        self,
        batch: dict[str, torch.Tensor],
        return_pi: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        adjacency = self._require_tensor(batch, self.adjacency_key).bool()
        if adjacency.ndim != 3 or adjacency.size(1) != adjacency.size(2):
            raise ValueError(
                f"MIS batch['{self.adjacency_key}'] must have shape "
                "[batch, nodes, nodes]"
            )

        batch_size, num_nodes, _ = adjacency.shape
        device = adjacency.device
        stop_index = num_nodes
        node_input = self._mis_node_features(adjacency)
        stop_input = self.mis_stop_input.view(1, 1, -1).expand(batch_size, 1, -1)
        input_list = torch.cat([node_input, stop_input], dim=1)

        encoder_output_list, hidden_states, cell_states = self.encoder(input_list)
        hidden = hidden_states[-1]
        cell = cell_states[-1]
        decoder_input = self.decoder.initial_input.expand(batch_size, -1)
        batch_idx = torch.arange(batch_size, device=device)

        unavailable = torch.zeros(
            batch_size,
            num_nodes,
            dtype=torch.bool,
            device=device,
        )
        selected_mask = torch.zeros_like(unavailable)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        log_p_steps: list[torch.Tensor] = []

        for _ in range(num_nodes + 1):
            hidden, cell = self.decoder.lstmcell(decoder_input, (hidden, cell))
            action_mask = torch.cat(
                [
                    unavailable,
                    torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
                ],
                dim=1,
            )
            action_mask = torch.where(
                finished.unsqueeze(1),
                self._stop_only_mask(batch_size, num_nodes, device),
                action_mask,
            )

            query = hidden
            for _ in range(self.decoder.n_glimpses):
                glimpse_logits = self.decoder.glimpse(
                    encoder_output_list=encoder_output_list,
                    decoder_output=query,
                    mask=(None if not self.decoder.mask_inner else action_mask),
                )
                glimpse_prob = F.softmax(glimpse_logits, dim=-1)
                glimpse_prob = torch.nan_to_num(glimpse_prob, nan=0.0)
                query = torch.bmm(
                    glimpse_prob.unsqueeze(1),
                    encoder_output_list,
                ).squeeze(1)

            logits = self.decoder.attention(
                encoder_output_list=encoder_output_list,
                decoder_output=query,
                mask=None if not self.decoder.mask_logits else action_mask,
            )
            selected = self.decoder._select(logits, self.decode_type)
            selected = torch.where(
                finished,
                torch.full_like(selected, stop_index),
                selected,
            )
            log_p = F.log_softmax(logits, dim=-1)
            log_p = torch.nan_to_num(log_p, nan=0.0, neginf=0.0)
            log_p = log_p.gather(1, selected.unsqueeze(1)).squeeze(1)
            log_p = torch.where(finished, torch.zeros_like(log_p), log_p)

            is_stop = selected == stop_index
            active = (~finished) & (~is_stop)
            node_selected = selected.clamp(max=num_nodes - 1)

            if active.any():
                active_idx = batch_idx[active]
                active_nodes = node_selected[active]
                selected_mask[active_idx, active_nodes] = True
                unavailable[active_idx] |= adjacency[active_idx, active_nodes]
                unavailable[active_idx, active_nodes] = True

            decoder_input = self.decoder._gather_inputs(input_list, selected)
            finished = finished | is_stop | unavailable.all(dim=-1)
            log_p_steps.append(log_p)

            if finished.all():
                break

        log_likelihood = torch.stack(log_p_steps, dim=1).sum(dim=1)
        cost = -selected_mask.float().sum(dim=1)
        if return_pi:
            return cost, log_likelihood, selected_mask
        return cost, log_likelihood

    def supervised_loss(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType | None = None,
    ) -> torch.Tensor:
        problem = self._resolve_problem(batch, problem)
        if problem == "tsp":
            return self.supervised_tsp_loss(batch)
        if problem == "mis":
            return self.supervised_mis_loss(batch)
        raise ValueError(f"Unsupported problem: {problem}")

    def supervised_tsp_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        loc = self._require_tensor(batch, self.loc_key)
        target_tour = self._require_tensor(batch, self.target_tour_key).long()
        output = self.decode_sequence(
            loc,
            target_list=target_tour,
            output_length=target_tour.size(1),
            teacher_forcing_ratio=1.0,
        )
        return self.loss(output.logit_list, target_tour)

    def supervised_mis_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.mis_node_logits(batch)
        target_set = self._require_tensor(batch, self.target_set_key).to(
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.binary_cross_entropy_with_logits(logits, target_set)

    def mis_node_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        adjacency = self._require_tensor(batch, self.adjacency_key).bool()
        h, _, _ = self.encoder(self._mis_node_features(adjacency))
        return self.mis_classifier(h).squeeze(-1)

    @staticmethod
    def loss(logit_list: torch.Tensor, target_list: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss for pointer logits."""
        if logit_list.ndim != 3:
            raise ValueError(
                "logit_list must have shape [batch, output_length, source_length]"
            )
        if target_list.shape != logit_list.shape[:2]:
            raise ValueError("target_list must have shape [batch, output_length]")
        return F.cross_entropy(
            input=logit_list.reshape(-1, logit_list.size(-1)),
            target=target_list.reshape(-1),
        )

    @staticmethod
    def tsp_cost(loc: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        ordered = loc.gather(1, pi.unsqueeze(-1).expand(-1, -1, loc.size(-1)))
        return (ordered[:, 1:] - ordered[:, :-1]).norm(p=2, dim=-1).sum(dim=1) + (
            ordered[:, 0] - ordered[:, -1]
        ).norm(p=2, dim=-1)

    def _mis_node_features(self, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = adjacency.shape
        degree = adjacency.float().sum(dim=-1) / max(num_nodes - 1, 1)
        features = torch.zeros(
            batch_size,
            num_nodes,
            self.input_size,
            dtype=degree.dtype,
            device=adjacency.device,
        )
        features[..., 0] = degree
        return features

    def _resolve_problem(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType | None,
    ) -> ProblemType:
        if problem is not None:
            return problem
        if self.default_problem is not None:
            return self.default_problem
        if self.loc_key in batch:
            return "tsp"
        if self.adjacency_key in batch:
            return "mis"
        raise ValueError("Pass problem='tsp' or problem='mis' to forward")

    @staticmethod
    def _require_tensor(batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        value = batch.get(key)
        if value is None:
            raise ValueError(f"Missing batch['{key}']")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch['{key}'] must be a torch.Tensor")
        return value

    @staticmethod
    def _stop_only_mask(
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.ones(batch_size, num_nodes + 1, dtype=torch.bool, device=device)
        mask[:, -1] = False
        return mask

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
