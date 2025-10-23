class Settings:
    api_key: str | None = None
    tiktoken_encoder: str = "cl100k_base"
    faker_langage: str = "fr_FR"
    faker_seed: int | None = None


settings = Settings()
