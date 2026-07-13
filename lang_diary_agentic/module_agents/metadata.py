from datetime import date, datetime
from ..models import AgentState

def node_set_metadata(state: AgentState) -> AgentState:
    # set the meta-info first

    _date_diary = state.diary_entry_input.date_diary
    if _date_diary is None:
        _date_diary = str(date.today())
        state.diary_entry_input.date_diary = _date_diary
    # end if

    _title_diary = state.diary_entry_input.title_diary
    if _title_diary is None:
        # date and language of diary body
        _title_diary = f"Title: {_date_diary} - {state.diary_entry_input.lang_diary_body}"
        state.diary_entry_input.title_diary = _title_diary
    # end if

    return state
