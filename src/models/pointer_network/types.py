import torch
from pydantic import BaseModel


class PointerNetworkOutput(BaseModel):
    logit_list: torch.Tensor
    pointer_list: torch.Tensor
