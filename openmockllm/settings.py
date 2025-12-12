from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    api_key: str | None = None
    tiktoken_encoder: str = "cl100k_base"
    faker_langage: str = "fr_FR"
    faker_seed: int | None = None
    reference_tps: int = 100
    simulate_latency: bool = False

    model_config = ConfigDict(extra="allow")


settings = Settings()
