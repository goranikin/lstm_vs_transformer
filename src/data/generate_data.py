"""Instance generation following the paper appendices."""

import torch


def sample_tsp(
    batch_size: int,
    n: int,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniform [0,1]^2 coordinates (Appendix B.2)."""
    device = device or torch.device("cpu")
    return torch.rand(batch_size, n, 2, device=device, generator=generator)


def sample_cvrp(
    batch_size: int,
    n: int,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """CVRP/SDVRP instances (Appendix C.1)."""
    device = device or torch.device("cpu")
    capacity_map = {20: 30, 50: 40, 100: 50}
    d_n = capacity_map.get(n, 40)

    depot = torch.rand(batch_size, 1, 2, device=device, generator=generator)
    loc = torch.rand(batch_size, n, 2, device=device, generator=generator)
    raw_demand = torch.randint(
        1,
        10,
        (batch_size, n),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    demand = raw_demand / d_n

    return {
        "loc": torch.cat([depot, loc], dim=1),
        "demand": torch.cat([torch.zeros(batch_size, 1, device=device), demand], dim=1),
    }


def _op_max_length(n: int) -> float:
    return {20: 2.0, 50: 3.0, 100: 4.0}.get(n, 3.0)


def sample_op(
    batch_size: int,
    n: int,
    distribution: str = "distance",
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Orienteering instances (Appendix D.1)."""
    device = device or torch.device("cpu")
    depot = torch.rand(batch_size, 1, 2, device=device, generator=generator)
    loc = torch.rand(batch_size, n, 2, device=device, generator=generator)
    all_loc = torch.cat([depot, loc], dim=1)

    if distribution == "const":
        prize = torch.ones(batch_size, n + 1, device=device)
    elif distribution == "uniform":
        prize = (
            torch.randint(
                1,
                101,
                (batch_size, n + 1),
                device=device,
                generator=generator,
            ).float()
            / 100.0
        )
    elif distribution == "distance":
        d0 = (all_loc - depot).norm(p=2, dim=-1)
        d0[:, 0] = 0
        max_d = d0.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        raw = 1.0 + torch.floor(99.0 * d0 / max_d)
        prize = raw / 100.0
    else:
        raise ValueError(f"Unknown OP prize distribution: {distribution}")

    return {
        "loc": all_loc,
        "prize": prize,
        "max_length": torch.full((batch_size,), _op_max_length(n), device=device),
    }


def sample_pctsp(
    batch_size: int,
    n: int,
    device: torch.device | None = None,
    stochastic: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """PCTSP / SPCTSP instances (Appendices E.1, F)."""
    device = device or torch.device("cpu")
    k_n = {20: 2.0, 50: 3.0, 100: 4.0}.get(n, 3.0)

    depot = torch.rand(batch_size, 1, 2, device=device, generator=generator)
    loc = torch.rand(batch_size, n, 2, device=device, generator=generator)
    all_loc = torch.cat([depot, loc], dim=1)

    raw_prize = torch.rand(batch_size, n + 1, device=device, generator=generator)
    raw_prize[:, 0] = 0
    prize = raw_prize * (4.0 / n)

    penalty = (
        torch.rand(batch_size, n + 1, device=device, generator=generator)
        * (3.0 * k_n / n)
    )
    penalty[:, 0] = 0

    out: dict[str, torch.Tensor] = {
        "loc": all_loc,
        "prize": prize,
        "penalty": penalty,
    }

    if stochastic:
        # ρ*_i ~ Uniform(0, 2 ρ_i)  (Appendix F)
        out["real_prize"] = (
            torch.rand(batch_size, n + 1, device=device, generator=generator)
            * 2.0
            * prize
        )

    return out
