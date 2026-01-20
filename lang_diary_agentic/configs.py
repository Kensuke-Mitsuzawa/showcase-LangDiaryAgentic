from pathlib import Path
import os
import pycountry
import typing

PossibleChoiceModeDeployment = typing.Literal["local", "server"]

# Save DB in the same 'data' folder
BASE_DIR = Path(__file__).resolve().parent.parent
GENERATION_DB_PATH = os.path.join(BASE_DIR, "data", "diary_log.duckdb")
ErrorVectorDB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")


MODEL_NAME_Embedding = "all-MiniLM-L6-v2"
MODEL_NAME_Primary = "Qwen/Qwen2.5-7B-Instruct" # Or your preferred model

Mode_Deployment: PossibleChoiceModeDeployment = "server"

Server_API_Endpoint: str = "http://0.0.0.0:8000"


# ---- list of language codes ----
# Iterate through all languages and filter for those that have an alpha_2 code
Languages_Code = []
for lang in pycountry.languages:
    if hasattr(lang, 'alpha_2'):
        Languages_Code.append(lang.alpha_2)  # type: ignore
    # end if
# end for
assert len(Languages_Code) > 0, "No language codes are loaded."
# ---- END: list of language codes ----