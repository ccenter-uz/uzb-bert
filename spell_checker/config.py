from pathlib import Path
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model settings
    MODEL_NAME: str = "tahrirchi/tahrirchi-bert-base"
    MODEL_TOP_K: int = 50
    MAX_BATCH_SIZE: int = 32
    
    # Paths
    DICTIONARY_PATH: str = str(Path(__file__).parent / "uz_words.json")
    
    # API settings
    DEFAULT_MAX_SUGGESTIONS: int = 3
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()