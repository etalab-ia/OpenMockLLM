from typing import Optional


class Settings:
    api_key: Optional[str] = None
    tiktoken_encoder: str = "cl100k_base"
    faker_langage: str = "fr_FR"
    faker_seed_instance: Optional[int] = None


settings = Settings()
