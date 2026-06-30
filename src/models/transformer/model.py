import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from configs.am_config import AMModelConfig
from src.models.transformer.attention_layer import (
    GraphAttentionEncoder,
    MultiHeadAttention,
)
from src.models.transformer.layers import init_uniform_

DecodeType = Literal["greedy", "sampling"]
ProblemType = Literal["tsp", "mis"]


class AttentionModel(nn.Module):
    """Attention encoder-decoder for file-backed TSP and MIS batches."""

    def __init__(
        self,
        config: AMModelConfig | None = None,
        default_problem: ProblemType | None = None,
        tsp_input_size: int = 2,
        mis_input_size: int = 1,
        mis_context_size: int = 1,
        loc_key: str = "loc",
        adjacency_key: str = "adjacency",
        target_tour_key: str = "target_tour",
        target_set_key: str = "target_set",
    ) -> None:
        super().__init__()
        self.config = config or AMModelConfig()
        if tsp_input_size < 1:
            raise ValueError("tsp_input_size must be at least 1")
        if mis_input_size < 1:
            raise ValueError("mis_input_size must be at least 1")
        if mis_context_size < 1:
            raise ValueError("mis_context_size must be at least 1")
        self.default_problem = default_problem
        self.tsp_input_size = tsp_input_size
        self.mis_input_size = mis_input_size
        self.mis_context_size = mis_context_size
        self.loc_key = loc_key
        self.adjacency_key = adjacency_key
        self.target_tour_key = target_tour_key
        self.target_set_key = target_set_key
        d_h = self.config.d_h

        self.tsp_node_embed = nn.Linear(in_features=tsp_input_size, out_features=d_h)
        self.tsp_node_bias = nn.Parameter(torch.zeros(d_h))
        init_uniform_(self.tsp_node_embed.weight, tsp_input_size)
        init_uniform_(self.tsp_node_bias, d_h)

        self.tsp_prev_placeholder = nn.Parameter(torch.empty(d_h))
        self.tsp_first_placeholder = nn.Parameter(torch.empty(d_h))
        init_uniform_(self.tsp_prev_placeholder, d_h)
        init_uniform_(self.tsp_first_placeholder, d_h)

        self.mis_node_embed = nn.Linear(in_features=mis_input_size, out_features=d_h)
        self.mis_node_bias = nn.Parameter(torch.zeros(d_h))
        init_uniform_(self.mis_node_embed.weight, mis_input_size)
        init_uniform_(self.mis_node_bias, d_h)

        self.mis_prev_placeholder = nn.Parameter(torch.empty(d_h))
        self.mis_stop_embedding = nn.Parameter(torch.empty(d_h))
        self.mis_stop_logit_key = nn.Parameter(torch.empty(d_h))
        init_uniform_(self.mis_prev_placeholder, d_h)
        init_uniform_(self.mis_stop_embedding, d_h)
        init_uniform_(self.mis_stop_logit_key, d_h)

        self.encoder = GraphAttentionEncoder(
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_h=d_h,
            d_ff=self.config.d_ff,
            normalization=self.config.normalization,
        )

        self.W_node_proj = nn.Linear(in_features=d_h, out_features=3 * d_h, bias=False)
        init_uniform_(self.W_node_proj.weight, d_h)

        self.tsp_step = nn.Linear(in_features=3 * d_h, out_features=d_h, bias=False)
        self.mis_step = nn.Linear(
            in_features=2 * d_h + mis_context_size,
            out_features=d_h,
            bias=False,
        )
        init_uniform_(self.tsp_step.weight, 3 * d_h)
        init_uniform_(self.mis_step.weight, 2 * d_h + mis_context_size)
        self.mis_classifier = nn.Linear(in_features=d_h, out_features=1)
        init_uniform_(self.mis_classifier.weight, d_h)
        init_uniform_(self.mis_classifier.bias, self.mis_classifier.bias.numel())

        self.glimpse_mha = MultiHeadAttention(
            n_heads=self.config.n_heads,
            d_h=d_h,
            input_dim=d_h,
        )

        self.decode_type: DecodeType = "sampling"

    def set_decode_type(self, decode_type: DecodeType) -> None:
        if decode_type not in ("greedy", "sampling"):
            raise ValueError("decode_type must be 'greedy' or 'sampling'")
        self.decode_type = decode_type

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        problem: ProblemType | None = None,
        return_pi: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        problem = self._resolve_problem(batch, problem)
        if problem == "tsp":
            return self.solve_tsp(batch, return_pi=return_pi)
        if problem == "mis":
            return self.solve_mis(batch, return_pi=return_pi)
        raise ValueError(f"Unsupported problem: {problem}")

    def solve_tsp(
        self,
        batch: dict[str, torch.Tensor],
        return_pi: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        loc = self._require_tensor(batch, self.loc_key)
        if loc.ndim != 3 or loc.size(-1) != self.tsp_input_size:
            raise ValueError(
                f"TSP batch['{self.loc_key}'] must have shape "
                f"[batch, nodes, {self.tsp_input_size}]"
            )

        batch_size, num_nodes, _ = loc.shape
        device = loc.device
        h, h_bar = self.encoder(self.tsp_node_embed(loc) + self.tsp_node_bias)
        glimpse_key, glimpse_value, logit_key = self.W_node_proj(h).chunk(3, dim=-1)

        selected_mask = torch.zeros(
            batch_size,
            num_nodes,
            dtype=torch.bool,
            device=device,
        )
        batch_idx = torch.arange(batch_size, device=device)
        first_a = torch.zeros(batch_size, dtype=torch.long, device=device)
        prev_a = torch.zeros(batch_size, dtype=torch.long, device=device)

        log_p_steps: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []

        for step in range(num_nodes):
            if step == 0:
                context = torch.cat(
                    [
                        h_bar,
                        self.tsp_prev_placeholder.expand(batch_size, -1),
                        self.tsp_first_placeholder.expand(batch_size, -1),
                    ],
                    dim=-1,
                )
            else:
                context = torch.cat(
                    [h_bar, h[batch_idx, prev_a], h[batch_idx, first_a]],
                    dim=-1,
                )

            query = self.tsp_step(context)
            mask = selected_mask.unsqueeze(1)
            glimpse = self.glimpse_mha(
                query.unsqueeze(1),
                glimpse_key,
                value=glimpse_value,
                mask=mask,
            ).squeeze(1)
            logits = self._attention_logits(glimpse, logit_key, selected_mask)
            selected, log_p = self._select(logits)

            if step == 0:
                first_a = selected
            prev_a = selected
            selected_mask = selected_mask.scatter(1, selected.unsqueeze(1), True)

            log_p_steps.append(log_p)
            actions.append(selected)

        pi = torch.stack(actions, dim=1)
        log_likelihood = torch.stack(log_p_steps, dim=1).sum(dim=1)
        cost = self.tsp_cost(loc, pi)

        if return_pi:
            return cost, log_likelihood, pi
        return cost, log_likelihood

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
        mis_features = self._mis_node_features(adjacency)

        eye = torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        graph_mask = ~(adjacency | eye)
        h, h_bar = self.encoder(
            self.mis_node_embed(mis_features) + self.mis_node_bias,
            mask=graph_mask,
        )
        node_glimpse_key, node_glimpse_value, node_logit_key = self.W_node_proj(h).chunk(
            3,
            dim=-1,
        )

        stop_index = num_nodes
        stop_embedding = self.mis_stop_embedding.view(1, 1, -1).expand(
            batch_size,
            1,
            -1,
        )
        stop_logit_key = self.mis_stop_logit_key.view(1, 1, -1).expand(
            batch_size,
            1,
            -1,
        )
        stop_glimpse_key = self.mis_stop_logit_key.view(1, 1, -1).expand(
            batch_size,
            1,
            -1,
        )
        decode_glimpse_key = torch.cat([node_glimpse_key, stop_glimpse_key], dim=1)
        decode_glimpse_value = torch.cat([node_glimpse_value, stop_embedding], dim=1)
        decode_keys = torch.cat([node_logit_key, stop_logit_key], dim=1)

        unavailable = torch.zeros(
            batch_size,
            num_nodes,
            dtype=torch.bool,
            device=device,
        )
        selected_mask = torch.zeros_like(unavailable)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        prev_emb = self.mis_prev_placeholder.expand(batch_size, -1)
        batch_idx = torch.arange(batch_size, device=device)

        log_p_steps: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []

        for _ in range(num_nodes + 1):
            available_ratio = (~unavailable).float().sum(dim=-1) / num_nodes
            mis_context = self._mis_context_features(available_ratio)
            context = torch.cat(
                [h_bar, prev_emb, mis_context], dim=-1
            )
            query = self.mis_step(context)

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

            glimpse = self.glimpse_mha(
                query.unsqueeze(1),
                decode_glimpse_key,
                value=decode_glimpse_value,
                mask=action_mask.unsqueeze(1),
            ).squeeze(1)
            logits = self._attention_logits(glimpse, decode_keys, action_mask)
            selected, log_p = self._select(logits)

            selected = torch.where(
                finished,
                torch.full_like(selected, stop_index),
                selected,
            )
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
                prev_emb = torch.where(
                    active.unsqueeze(1),
                    h[batch_idx, node_selected],
                    prev_emb,
                )

            finished = finished | is_stop | unavailable.all(dim=-1)
            log_p_steps.append(log_p)
            actions.append(selected)

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
        log_p_steps = self._tsp_target_log_probs(loc, target_tour)
        return -torch.stack(log_p_steps, dim=1).mean()

    def supervised_mis_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.mis_node_logits(batch)
        target_set = self._require_tensor(batch, self.target_set_key).to(
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.binary_cross_entropy_with_logits(logits, target_set)

    def mis_node_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        adjacency = self._require_tensor(batch, self.adjacency_key).bool()
        if adjacency.ndim != 3 or adjacency.size(1) != adjacency.size(2):
            raise ValueError(
                f"MIS batch['{self.adjacency_key}'] must have shape "
                "[batch, nodes, nodes]"
            )
        num_nodes = adjacency.size(1)
        eye = torch.eye(num_nodes, dtype=torch.bool, device=adjacency.device).unsqueeze(0)
        graph_mask = ~(adjacency | eye)
        h, _ = self.encoder(
            self.mis_node_embed(self._mis_node_features(adjacency)) + self.mis_node_bias,
            mask=graph_mask,
        )
        return self.mis_classifier(h).squeeze(-1)

    @staticmethod
    def tsp_cost(loc: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        ordered = loc.gather(1, pi.unsqueeze(-1).expand(-1, -1, loc.size(-1)))
        return (ordered[:, 1:] - ordered[:, :-1]).norm(p=2, dim=-1).sum(dim=1) + (
            ordered[:, 0] - ordered[:, -1]
        ).norm(p=2, dim=-1)

    def _tsp_target_log_probs(
        self,
        loc: torch.Tensor,
        target_tour: torch.Tensor,
    ) -> list[torch.Tensor]:
        if loc.ndim != 3 or loc.size(-1) != self.tsp_input_size:
            raise ValueError(
                f"TSP batch['{self.loc_key}'] must have shape "
                f"[batch, nodes, {self.tsp_input_size}]"
            )
        if target_tour.ndim != 2 or target_tour.size(0) != loc.size(0):
            raise ValueError(
                f"batch['{self.target_tour_key}'] must have shape [batch, nodes]"
            )
        if target_tour.size(1) != loc.size(1):
            raise ValueError(
                f"batch['{self.target_tour_key}'] must include every TSP node"
            )

        batch_size, num_nodes, _ = loc.shape
        device = loc.device
        h, h_bar = self.encoder(self.tsp_node_embed(loc) + self.tsp_node_bias)
        glimpse_key, glimpse_value, logit_key = self.W_node_proj(h).chunk(3, dim=-1)

        selected_mask = torch.zeros(
            batch_size,
            num_nodes,
            dtype=torch.bool,
            device=device,
        )
        batch_idx = torch.arange(batch_size, device=device)
        first_a = torch.zeros(batch_size, dtype=torch.long, device=device)
        prev_a = torch.zeros(batch_size, dtype=torch.long, device=device)
        log_p_steps: list[torch.Tensor] = []

        for step in range(num_nodes):
            if step == 0:
                context = torch.cat(
                    [
                        h_bar,
                        self.tsp_prev_placeholder.expand(batch_size, -1),
                        self.tsp_first_placeholder.expand(batch_size, -1),
                    ],
                    dim=-1,
                )
            else:
                context = torch.cat(
                    [h_bar, h[batch_idx, prev_a], h[batch_idx, first_a]],
                    dim=-1,
                )

            query = self.tsp_step(context)
            mask = selected_mask.unsqueeze(1)
            glimpse = self.glimpse_mha(
                query.unsqueeze(1),
                glimpse_key,
                value=glimpse_value,
                mask=mask,
            ).squeeze(1)
            logits = self._attention_logits(glimpse, logit_key, selected_mask)
            selected = target_tour[:, step]
            log_p = F.log_softmax(logits, dim=-1)
            log_p = torch.nan_to_num(log_p, nan=0.0, neginf=0.0)
            log_p_steps.append(log_p.gather(1, selected.unsqueeze(1)).squeeze(1))

            if step == 0:
                first_a = selected
            prev_a = selected
            selected_mask = selected_mask.scatter(1, selected.unsqueeze(1), True)

        return log_p_steps

    def _mis_node_features(self, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = adjacency.shape
        degree = adjacency.float().sum(dim=-1) / max(num_nodes - 1, 1)
        features = torch.zeros(
            batch_size,
            num_nodes,
            self.mis_input_size,
            dtype=degree.dtype,
            device=adjacency.device,
        )
        features[..., 0] = degree
        return features

    def _mis_context_features(self, available_ratio: torch.Tensor) -> torch.Tensor:
        features = torch.zeros(
            available_ratio.size(0),
            self.mis_context_size,
            dtype=available_ratio.dtype,
            device=available_ratio.device,
        )
        features[:, 0] = available_ratio
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

    def _attention_logits(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.d_h)
        logits = scale * torch.matmul(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(
            1
        )
        if self.config.tanh_clip > 0:
            logits = self.config.tanh_clip * torch.tanh(logits)
        return logits.masked_fill(mask, float("-inf"))

    def _select(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.decode_type == "greedy":
            selected = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            selected = torch.multinomial(probs, 1).squeeze(-1)

        log_p = F.log_softmax(logits, dim=-1)
        log_p = torch.nan_to_num(log_p, nan=0.0, neginf=0.0)
        log_p = log_p.gather(1, selected.unsqueeze(-1)).squeeze(-1)
        return selected, log_p

    @staticmethod
    def _stop_only_mask(
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.ones(batch_size, num_nodes + 1, dtype=torch.bool, device=device)
        mask[:, -1] = False
        return mask
