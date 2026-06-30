import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from configs.am_config import AMModelConfig
from src.data.registry import (
    ProblemName,
    ProblemSpec,
    compute_cost,
    make_state,
    max_decode_steps,
)
from src.models.transformer.attention_layer import AttentionLayer, MultiHeadAttention
from src.models.transformer.layers import init_uniform_

DecodeType = Literal["greedy", "sampling"]


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

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h = layer(h)
        h_bar = h.mean(dim=1)
        return h, h_bar


class AttentionModel(nn.Module):
    """
    Attention based encoder-decoder for routing problems.

    Policy factorization (eq. 1):
        p_θ(π|s) = ∏_t p_θ(π_t | s, π_{1:t-1})
    """

    def __init__(
        self, problem: ProblemSpec, config: AMModelConfig | None = None
    ) -> None:
        super().__init__()
        self.problem = problem
        self.config = config or AMModelConfig()
        d_h = self.config.d_h

        self.allow_split = problem.allow_split

        if problem.has_depot:
            self.W_x0 = nn.Linear(2, d_h)
            self.b_x0 = nn.Parameter(torch.zeros(d_h))
            init_uniform_(self.W_x0.weight, 2)
            init_uniform_(self.b_x0, d_h)
            node_in = (
                4 if problem.name in (ProblemName.PCTSP, ProblemName.SPCTSP) else 3
            )
            if self.allow_split:
                self.W_K_d = nn.Linear(1, d_h, bias=False)
                self.W_V_d = nn.Linear(1, d_h, bias=False)
        else:
            node_in = 2
            self.v_l = nn.Parameter(torch.empty(d_h))
            self.v_f = nn.Parameter(torch.empty(d_h))
            init_uniform_(self.v_l, d_h)
            init_uniform_(self.v_f, d_h)

        self.W_x = nn.Linear(node_in, d_h)
        self.b_x = nn.Parameter(torch.zeros(d_h))
        init_uniform_(self.W_x.weight, node_in)
        init_uniform_(self.b_x, d_h)

        self.encoder = GraphAttentionEncoder(
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_h=d_h,
            d_ff=self.config.d_ff,
            normalization=self.config.normalization,
        )

        self.W_node_proj = nn.Linear(d_h, 3 * d_h, bias=False)
        init_uniform_(self.W_node_proj.weight, d_h)

        step_context_dim = (2 * d_h + 1) if problem.has_depot else (3 * d_h)
        self.W_step = nn.Linear(step_context_dim, d_h, bias=False)
        init_uniform_(self.W_step.weight, step_context_dim)

        self.glimpse_mha = MultiHeadAttention(
            n_heads=self.config.n_heads,
            d_h=d_h,
            input_dim=d_h,
        )

        self.decode_type: DecodeType = "sampling"

    def set_decode_type(self, decode_type: DecodeType) -> None:
        self.decode_type = decode_type

    def _init_embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        loc = batch["loc"]
        batch_size, n, _ = loc.shape

        if not self.problem.has_depot:
            return self.W_x(loc) + self.b_x

        h0 = torch.zeros(batch_size, n, self.config.d_h, device=loc.device)
        h0[:, 0] = self.W_x0(loc[:, 0]) + self.b_x0

        if self.problem.name in (ProblemName.PCTSP, ProblemName.SPCTSP):
            node_feat = torch.cat(
                [
                    loc[:, 1:],
                    batch["prize"][:, 1:, None],
                    batch["penalty"][:, 1:, None],
                ],
                dim=-1,
            )
        elif self.problem.name == ProblemName.OP:
            node_feat = torch.cat([loc[:, 1:], batch["prize"][:, 1:, None]], dim=-1)
        else:
            node_feat = torch.cat([loc[:, 1:], batch["demand"][:, 1:, None]], dim=-1)

        h0[:, 1:] = self.W_x(node_feat) + self.b_x
        return h0

    def _context_embedding(self, h: torch.Tensor, state) -> torch.Tensor:
        batch_size = h.size(0)
        h_bar = h.mean(dim=1)
        batch_idx = torch.arange(batch_size, device=h.device)

        if self.problem.has_depot:
            prev_a, scalar_ctx, is_first = state.get_context_features()
            if is_first:
                prev_emb = h[:, 0]
            else:
                prev_emb = h[batch_idx, prev_a]
            ctx = torch.cat([h_bar, prev_emb, scalar_ctx.unsqueeze(-1)], dim=-1)
            return self.W_step(ctx)

        first_a, prev_a, is_first = state.get_context_features()
        if is_first:
            ctx = torch.cat(
                [
                    h_bar,
                    self.v_l.expand(batch_size, -1),
                    self.v_f.expand(batch_size, -1),
                ],
                dim=-1,
            )
        else:
            ctx = torch.cat(
                [h_bar, h[batch_idx, prev_a], h[batch_idx, first_a]], dim=-1
            )
        return self.W_step(ctx)

    def _attention_logits(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
        clip: bool,
    ) -> torch.Tensor:
        d_h = self.config.d_h
        q = query.unsqueeze(1)
        scale = 1.0 / math.sqrt(d_h)
        compat = scale * torch.matmul(q, keys.transpose(1, 2)).squeeze(1)
        if clip and self.config.tanh_clip > 0:
            compat = self.config.tanh_clip * torch.tanh(compat)
        node_mask = mask.squeeze(1) if mask.dim() == 3 else mask
        return compat.masked_fill(node_mask, float("-inf"))

    def _glimpse(
        self,
        query: torch.Tensor,
        node_emb: torch.Tensor,
        mask: torch.Tensor,
        remaining_demand: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.allow_split and remaining_demand is not None:
            rd = remaining_demand.unsqueeze(-1)
            keys = node_emb + self.W_K_d(rd)
            values = node_emb + self.W_V_d(rd)
            q = query.unsqueeze(1)
            scale = 1.0 / math.sqrt(self.config.d_h)
            compat = scale * torch.matmul(q, keys.transpose(1, 2))
            compat = compat.masked_fill(mask, float("-inf"))
            attn = torch.softmax(compat, dim=-1)
            attn = attn.masked_fill(mask, 0.0)
            return torch.matmul(attn, values).squeeze(1)

        return self.glimpse_mha(query.unsqueeze(1), node_emb, mask=mask).squeeze(1)

    def _decode_step(
        self,
        h: torch.Tensor,
        logit_key: torch.Tensor,
        state,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        remaining = getattr(state, "remaining_demand", None)
        query = self._context_embedding(h, state)
        glimpse = self._glimpse(query, h, mask, remaining)

        if self.allow_split and remaining is not None:
            rd = remaining.unsqueeze(-1)
            keys = logit_key + self.W_K_d(rd)
            logits = self._attention_logits(glimpse, keys, mask, clip=True)
        else:
            logits = self._attention_logits(glimpse, logit_key, mask, clip=True)

        if self.decode_type == "greedy":
            selected = logits.argmax(dim=-1)
            all_masked = ~torch.isfinite(logits).any(dim=-1)
            selected = torch.where(all_masked, torch.zeros_like(selected), selected)
        else:
            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            # If every action is masked, return to depot
            all_masked = ~torch.isfinite(logits).any(dim=-1)
            probs[all_masked, 0] = 1.0
            selected = torch.multinomial(probs, 1).squeeze(-1)

        log_p = F.log_softmax(logits, dim=-1)
        log_p = torch.nan_to_num(log_p, nan=0.0)
        log_p = log_p.gather(1, selected.unsqueeze(-1)).squeeze(-1)
        return selected, log_p

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_pi: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        h, _ = self.encoder(self._init_embed(batch))
        _, _, logit_key = self.W_node_proj(h).chunk(3, dim=-1)
        state = make_state(self.problem, batch)
        batch_size = h.size(0)
        device = h.device
        n_customers = h.size(1) - (1 if self.problem.has_depot else 0)

        batch_size = h.size(0)
        device = h.device
        n_customers = h.size(1) - (1 if self.problem.has_depot else 0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        log_p_steps: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []

        for _ in range(max_decode_steps(self.problem, n_customers)):
            mask = state.get_mask()
            selected, log_p = self._decode_step(h, logit_key, state, mask)
            selected = torch.where(finished, torch.zeros_like(selected), selected)
            log_p = torch.where(finished, torch.zeros_like(log_p), log_p)

            log_p_steps.append(log_p)
            actions.append(selected)

            if "real_prize" in batch and self.problem.stochastic:
                batch_idx = torch.arange(batch_size, device=device)
                state.update(
                    selected, real_prize=batch["real_prize"][batch_idx, selected]
                )
            else:
                state.update(selected)

            finished = finished | self._instance_done(state, selected)
            if finished.all():
                break

        pi = torch.stack(actions, dim=1)
        log_likelihood = torch.stack(log_p_steps, dim=1).sum(dim=1)
        cost = compute_cost(self.problem, batch, pi)

        if return_pi:
            return cost, log_likelihood, pi
        return cost, log_likelihood

    def _instance_done(self, state, selected: torch.Tensor) -> torch.Tensor:
        if self.problem.name == ProblemName.TSP:
            return torch.full(
                (selected.size(0),),
                state.i >= state.loc.size(1),
                device=selected.device,
            )
        if self.problem.name in (ProblemName.CVRP, ProblemName.SDVRP):
            return (state.remaining_demand[:, 1:].sum(-1) == 0) & (state.i > 1)
        if self.problem.name in (ProblemName.OP, ProblemName.PCTSP, ProblemName.SPCTSP):
            return (selected == 0) & (state.i > 1)
        return torch.zeros(selected.size(0), dtype=torch.bool, device=selected.device)
