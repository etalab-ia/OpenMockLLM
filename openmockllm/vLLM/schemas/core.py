from pydantic import BaseModel, ConfigDict


class VllmBaseModel(BaseModel):
    """Base model for VLLM schemas"""

    model_config = ConfigDict(extra="allow")
