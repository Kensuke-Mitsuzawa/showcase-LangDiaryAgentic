from pathlib import Path
import os
import logging
import typing as ty

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from . import static

logger = logging.getLogger(__name__)

DOTENV = Path(__file__).parent.parent / ".env"
assert Path(DOTENV).exists(), f".env file not found at {DOTENV}."


class Settings(BaseSettings):
    # Define variables with types and default values
    APP_NAME: str = "LinguaLog"

    Mode_Deployment: static.PossibleChoiceModeDeployment = "server_custom_hf"
    Server_API_Endpoint: ty.Optional[str] = None
    Cloud_API_Token: ty.Optional[str] = None

    MODEL_NAME_Embedding: str = "all-MiniLM-L6-v2"
    MODEL_NAME_Primary: str = "Qwen/Qwen2.5-7B-Instruct"

    DB_BASE_DIR: str = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.as_posix())
    DB_NAME_GENERATION: str = "diary_log.duckdb"
    DB_NAME_ERROR_VECTOR: str = "chroma_db"
    GENERATION_DB_PATH: ty.Optional[str] = None
    ErrorVectorDB_PATH: ty.Optional[str] = None

    # This sub-class tells Pydantic to look for a .env file
    model_config = SettingsConfigDict(env_file=DOTENV, extra='allow')

    def model_post_init(self, context: ty.Any) -> None:
        self.GENERATION_DB_PATH = (Path(self.DB_BASE_DIR).absolute() / "data" / "diary_log.duckdb").as_posix()
        self.ErrorVectorDB_PATH = (Path(self.DB_BASE_DIR).absolute() / "data" / "chroma_db").as_posix()

        if self.Mode_Deployment == 'cloud_api':
            assert self.Cloud_API_Token is not None, "`Cloud_API_Token` must be give."
        elif self.Mode_Deployment in ('server_custom_hf', 'server_ollama'):
            assert self.Server_API_Endpoint is not None, "`Server_API_Endpoint` must be give."
# end if


# Create a single instance to use across your app
settings = Settings()
logger.info(f'loaded settings: {settings}')