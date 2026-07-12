from ..models import AgentState, DiaryEntry
from ..db_handler import HandlerDairyDB
from ..configs import settings

def node_init_db_entry(state: AgentState):
    assert settings.GENERATION_DB_PATH is not None
    handler = HandlerDairyDB(settings.GENERATION_DB_PATH)
    handler.init_db()

    # Use today's date if not provided
    diary_date = state["diary_date"]
    created_at = state["created_at"]

    language_source = state.get("lang_diary_body", "Unknown")
    language_source = "Unknown" if language_source is None else language_source

    language_annotation = state.get("lang_annotation", "Unknown")
    language_annotation = "Unknown" if language_annotation is None else language_annotation

    diary_entry = DiaryEntry(
        date_diary=diary_date,
        language_source=language_source,
        language_annotation=language_annotation,
        diary_original=state["draft_text"],
        diary_replaced='',
        diary_rewritten='',
        created_at=created_at,
        primary_id=state["primary_id_DiaryEntry"],
        level_rewriting=state["level_rewriting"],
        model_id_tutor=settings.MODEL_NAME_Primary,
        title_diary=state["title_diary"],
        current_version=0,
        is_show=True
    )

    handler.save_diary_entry(diary_entry)

    return {
        "diary_entry": diary_entry
    }
