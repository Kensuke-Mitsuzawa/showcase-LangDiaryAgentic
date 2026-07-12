import typing as ty
from ..models import AgentState
from ..static import PossibleLevelRewriting

def node_validation(state: AgentState):
    level_rewriting = state.get("level_rewriting")
    assert level_rewriting is not None
    assert level_rewriting in ty.get_args(PossibleLevelRewriting)
