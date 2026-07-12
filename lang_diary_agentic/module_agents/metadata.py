from datetime import date, datetime
from ..models import AgentState

def node_set_metadata(state: AgentState):
    # set the meta-info first
    diary_date = state.get("date_diary", str(date.today()))
    created_at = datetime.now()
    datetime_str = created_at.isoformat()
    
    if state["primary_id_DiaryEntry"] is None:
        primary_id_DiaryEntry = f"{diary_date}_{datetime_str}"
    else:
        primary_id_DiaryEntry = state["primary_id_DiaryEntry"]
    # end if

    if state["title_diary"] is None:
        title_diary = ""
    else:
        title_diary = state["title_diary"]
    # end if

    return {
        "title_diary": title_diary,
        "primary_id_DiaryEntry": primary_id_DiaryEntry,
        "diary_date": diary_date,
        "created_at": created_at
    }
