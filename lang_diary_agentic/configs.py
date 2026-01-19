from pathlib import Path
import os
import pycountry


# Save DB in the same 'data' folder
BASE_DIR = Path(__file__).resolve().parent.parent
GENERATION_DB_PATH = os.path.join(BASE_DIR, "data", "diary_log.duckdb")
ErrorVectorDB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")


# Iterate through all languages and filter for those that have an alpha_2 code
languages_code = []
for lang in pycountry.languages:
    if hasattr(lang, 'alpha_2'):
        languages_code.append(lang.alpha_2)
        # languages.append({
        #     'code': lang.alpha_2,
        #     'name': lang.name
        # })
    # end if
# end for
