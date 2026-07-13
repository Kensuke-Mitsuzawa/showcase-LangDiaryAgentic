import logging

from .models import (
    ParameterConfig,
    TaskParameterConfig, 
    TranslationReplacementInformation, 
    AgentState,
    DiaryEntry,
)
from .configs import SettingsVariables

# from .llm_clients import (
#     llm_large, 
#     client_embedding_model_server, 
#     create_compatible_chain
# )
from .module_agents import (
    node_validation,
    node_set_metadata,
    node_init_db_entry,
    node_dynamic_parameter_adjustment,
    node_detect_unknown_expressions,
    node_translator,
    node_error_analysis,
    node_rewriter,
    node_reviewer,
    node_save_duckdb
)
from .configs import SettingsVariables

logger = logging.getLogger(__name__)


def run_pipeline(
    state: AgentState,
    settings: SettingsVariables
    ) -> AgentState:
    # Create a single instance to use across your app
    logger.info(f'loaded settings: {settings}')
    
    state = node_validation(state)
    state = node_set_metadata(state)
    # state = node_dynamic_parameter_adjustment(state)
    state = node_detect_unknown_expressions(state)
    state = node_translator(state, settings=settings)
    state = node_rewriter(state, settings=settings)
    state = node_error_analysis(state, settings=settings)
    
    # state = node_reviewer(state, settings=settings)
    # state = node_save_duckdb(state)

    return state


# def init_graph():
#     # --- 4. Build Graph ---
#     workflow = StateGraph(AgentState)
#     workflow.add_node("validator", node_validation)
#     workflow.add_node("meta_data", node_set_metadata)
#     # workflow.add_node("init_db_entry", node_init_db_entry)    
#     workflow.add_node("adjuster_parameter", node_dynamic_parameter_adjustment)
#     workflow.add_node("detector", node_detect_unknown_expressions)
#     workflow.add_node("translator", node_translator)
#     workflow.add_node("archivist", node_archivist)
#     workflow.add_node("rewriter", node_rewriter)
#     workflow.add_node("reviewer", node_reviewer)
#     workflow.add_node("db_saver", node_save_duckdb)

#     workflow.set_entry_point("validator")
#     workflow.add_edge("validator", "meta_data")
#     # workflow.add_edge("meta_data", "init_db_entry")
#     # workflow.add_edge("init_db_entry", "adjuster_parameter")    
#     workflow.add_edge("meta_data", "adjuster_parameter")
#     workflow.add_edge("adjuster_parameter", "detector")    
#     workflow.add_edge("detector", "translator")
#     workflow.add_edge("translator", "archivist")
#     workflow.add_edge("archivist", "rewriter")
#     workflow.add_edge('rewriter', 'reviewer')
#     workflow.add_edge("reviewer", "db_saver")
#     workflow.add_edge("db_saver", END)

#     app_graph = workflow.compile()

#     return app_graph