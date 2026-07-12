from ..models import AgentState, TaskParameterConfig

def node_dynamic_parameter_adjustment(state: AgentState) -> AgentState:
    """Optimizing the parameter depending on the user's input parameter.
    
    Rule one: + 512 words to the input diary length.
    Rule two: if enable_think=True, double the max token length.
    """
    target_fields = ["config_translator", "config_archivist", "config_rewriter", "config_reviewer"]

    diary_length = len(state["draft_text"])

    dict_updated_config = {}
    for _filed_name in target_fields:
        _config_obj: TaskParameterConfig = state[_filed_name]
        
        _new_length = diary_length + 512
        if _config_obj.enable_thinking is True:
            _new_length = _new_length * 2
        # end if

        _config_obj.max_tokens = _new_length
        dict_updated_config[_filed_name] = _config_obj
    # end for

    return state
