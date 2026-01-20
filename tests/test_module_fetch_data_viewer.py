from lang_diary_agentic import module_fetch_data_viewer

from lang_diary_agentic.db_handler import HandlerDairyDB
from lang_diary_agentic.models.generation_records import DiaryEntry, UnknownExpressionEntry
from lang_diary_agentic.vector_store import get_vector_store

from lang_diary_agentic.configs import GENERATION_DB_PATH



def test_module_fetch_data_viewer():
    handler = HandlerDairyDB(GENERATION_DB_PATH)
    vector_db = get_vector_store()

    seq_entry = module_fetch_data_viewer.fetch_records_language(
        language_annotation="en",
        language_daiary_body="fr",
        handler=handler,
        vector_db=vector_db
    )


if __name__ == "__main__":
    test_module_fetch_data_viewer()
