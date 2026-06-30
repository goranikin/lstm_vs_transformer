import torch
from pydantic import BaseModel, ConfigDict


class PointerNetworkOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    logit_list: torch.Tensor
    pointer_list: torch.Tensor
    log_likelihood: torch.Tensor | None = None
