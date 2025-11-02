"""
LangGraph edge conditions for routing workflow.
"""
from typing import Dict
import json
from src.agents import policy_agent
from src.utils.helpers import get_past_context_str
from config.config import MAX_REASONING_ITERATIONS, logger


def route_by_tool(state: Dict) -> str:
    """Route to appropriate retrieval node based on tool selection"""
    try:
        current_step = state["plan"].steps[state["current_step_index"]]
        tool = current_step.tool
        logger.info(f"Routing to tool: {tool}")
        return tool
    except Exception as e:
        logger.error(f"Error in route_by_tool: {e}")
        return "search_documents"  # Default fallback


def should_continue_node(state: Dict) -> str:
    """Decide whether to continue or finish"""
    logger.info("🚦 Evaluating policy...")
    
    try:
        current_step_index = state["current_step_index"]
        
        # Check basic stopping conditions
        if current_step_index >= len(state["plan"].steps):
            logger.info("✓ Plan complete. Finishing.")
            return "finish"
        
        if current_step_index >= MAX_REASONING_ITERATIONS:
            logger.warning("⚠️ Max iterations reached. Finishing.")
            return "finish"
        
        # Ask policy agent
        history = get_past_context_str(state.get('past_steps', []))
        plan_str = json.dumps([s.dict() for s in state['plan'].steps])
        
        decision = policy_agent.invoke({
            "question": state["original_question"],
            "plan": plan_str,
            "history": history
        })
        
        logger.info(f"📊 Decision: {decision.next_action}")
        return "finish" if decision.next_action == "FINISH" else "continue"
        
    except Exception as e:
        logger.error(f"Error in should_continue_node: {e}")
        return "finish"  # Default to finish on error

