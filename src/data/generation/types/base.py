from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
