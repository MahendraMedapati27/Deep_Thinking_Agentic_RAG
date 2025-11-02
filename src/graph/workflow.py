"""
LangGraph workflow assembly and compilation.
"""
from langgraph.graph import StateGraph, END
from .nodes import (
    RAGState,
    plan_node,
    retrieval_node,
    web_search_node,
    rerank_node,
    compression_node,
    reflection_node,
    final_answer_node
)
from .edges import route_by_tool, should_continue_node
from config.config import logger


def build_graph():
    """
    Build and compile the LangGraph workflow.
    
    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Building LangGraph workflow...")
    
    # Create the state graph
    graph = StateGraph(RAGState)
    
    # Add all nodes
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve_documents", retrieval_node)
    graph.add_node("retrieve_web", web_search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("compress", compression_node)
    graph.add_node("reflect", reflection_node)
    graph.add_node("generate_final_answer", final_answer_node)
    
    # Set entry point
    graph.set_entry_point("plan")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "plan",
        route_by_tool,
        {
            "search_documents": "retrieve_documents",
            "search_web": "retrieve_web",
        }
    )
    
    # Add sequential edges
    graph.add_edge("retrieve_documents", "rerank")
    graph.add_edge("retrieve_web", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "reflect")
    
    # Add main loop control
    graph.add_conditional_edges(
        "reflect",
        should_continue_node,
        {
            "continue": "plan",
            "finish": "generate_final_answer"
        }
    )
    
    graph.add_edge("generate_final_answer", END)
    
    # Compile with increased recursion limit
    compiled_graph = graph.compile()
    logger.info("✅ Graph compiled successfully!")
    
    return compiled_graph

