import logging
from langgraph.graph import StateGraph, END

from .models import TaskParameterConfig, TranslationReplacementInformation, AgentState
from .llm_clients import llm_large, client_embedding_model_server, create_compatible_chain
from .module_agents import (
    node_validation,
    node_set_metadata,
    node_init_db_entry,
    node_dynamic_parameter_adjustment,
    node_language_detect,
    node_translator,
    node_archivist,
    node_rewriter,
    node_reviewer,
    node_save_duckdb
)

logger = logging.getLogger(__name__)


def init_graph():
    # --- 4. Build Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("meta_data", node_set_metadata)
    workflow.add_node("validator", node_validation)
    workflow.add_node("init_db_entry", node_init_db_entry)    
    workflow.add_node("adjuster_parameter", node_dynamic_parameter_adjustment)
    workflow.add_node("detector", node_language_detect)
    workflow.add_node("translator", node_translator)
    workflow.add_node("archivist", node_archivist)
    workflow.add_node("rewriter", node_rewriter)
    workflow.add_node("reviewer", node_reviewer)
    workflow.add_node("db_saver", node_save_duckdb)

    workflow.set_entry_point("validator")
    workflow.add_edge("validator", "meta_data")
    workflow.add_edge("meta_data", "init_db_entry")
    workflow.add_edge("init_db_entry", "adjuster_parameter")    
    workflow.add_edge("adjuster_parameter", "detector")    
    workflow.add_edge("detector", "translator")
    workflow.add_edge("translator", "archivist")
    workflow.add_edge("archivist", "rewriter")
    workflow.add_edge('rewriter', 'reviewer')
    workflow.add_edge("reviewer", "db_saver")
    workflow.add_edge("db_saver", END)

    app_graph = workflow.compile()

    return app_graph