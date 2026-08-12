from pathlib import Path
import os
import logging
import typing as ty

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from . import static

logger = logging.getLogger(__name__)

# TODO: if the env var. is given, read from the env. var else use the default path.
_path_config_given = os.environ.get("PATH_CONFIG", None)
if _path_config_given is None:
    _path_config = Path(__file__) / ".env"
    assert Path(_path_config).exists(), f".env file not found at {_path_config}."
    logger.info("The config file is loaded from the .env")
else:
    assert Path(_path_config_given).exists(), f"The variable `PATH_CONFIG` is given. But no file found at {_path_config_given}"
    logger.info(f"The config file is loaded from the variable `PATH_CONFIG`.")
    _path_config = Path(_path_config_given)
# end if


ConfigDefaultSetting = {
    'gemini': {
        'Server_API_Endpoint': 'https://generativelanguage.googleapis.com/v1beta/openai/',
        'MODEL_NAME_Primary': 'gemini-3.5-flash',
        'MODEL_NAME_Embedding': 'models/gemini-embedding-2'
    }
}


class SettingsVariables(BaseSettings):

    # Define variables with types and default values
    APP_NAME: str = "LinguaLog"

    Mode_Deployment: static.PossibleChoiceModeDeployment = "cloud_api"
    Cloud_API_Provider: static.PossibleCloudLLMProvider = "gemini"

    Server_API_Endpoint: ty.Optional[str] = None
    Cloud_API_Token: ty.Optional[str] = None

    MODEL_NAME_Primary: str
    MODEL_NAME_Embedding: str    

    # DB_BASE_DIR: str = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.as_posix())
    DB_BASE_DIR: str
    DB_NAME_GENERATION: str = "diary_log.duckdb"
    DB_NAME_ERROR_VECTOR: str = "chroma_db"
    GENERATION_DB_PATH: ty.Optional[str] = None
    ErrorVectorDB_PATH: ty.Optional[str] = None

    # This sub-class tells Pydantic to look for a .env file
    model_config = SettingsConfigDict(env_file=_path_config, extra='allow')

    def model_post_init(self, context: ty.Any) -> None:
        self.GENERATION_DB_PATH = (Path(self.DB_BASE_DIR).absolute() / "data" / "diary_log.duckdb").as_posix()
        self.ErrorVectorDB_PATH = (Path(self.DB_BASE_DIR).absolute() / "data" / "chroma_db").as_posix()

        if self.Mode_Deployment == 'cloud_api':
            assert self.Cloud_API_Token is not None, "`Cloud_API_Token` must be give."
        elif self.Mode_Deployment in ('server_custom_hf', 'server_ollama'):
            assert self.Server_API_Endpoint is not None, "`Server_API_Endpoint` must be give."
# end if
