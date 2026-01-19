from pathlib import Path
import os


# Save DB in the same 'data' folder
BASE_DIR = Path(__file__).resolve().parent.parent
GENERATION_DB_PATH = os.path.join(BASE_DIR, "data", "diary_log.duckdb")
ErrorVectorDB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
