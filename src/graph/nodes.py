"""
LangGraph node implementations for the Deep-Thinking RAG workflow.
"""
from typing import Dict, List, TypedDict, Any
import json
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from src.agents import (
    planner_agent,
    query_rewriter_agent,
    retrieval_supervisor_agent,
    reflection_agent,
    policy_agent,
    final_answer_agent
)
from src.retrieval import (
    vector_search_only,
    bm25_search_only,
    hybrid_search,
    rerank_documents
)
from src.tools import web_search_function
from src.utils.helpers import get_past_context_str
from config.config import TOP_K_RETRIEVAL, REASONING_LLM, GROQ_API_KEY, logger


class PastStep(TypedDict):
    """Memory of completed research"""
    step_index: int
    sub_question: str
    retrieved_docs: List[Document]
    summary: str


class RAGState(TypedDict):
    """The agent's complete memory"""
    original_question: str
    plan: Any  # Plan object
    past_steps: List[PastStep]
    current_step_index: int
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    synthesized_context: str
    final_answer: str


def _extract_groq_error_message(error: Exception) -> str:
    """
    Extract meaningful error message from Groq API errors.
    """
    error_str = str(error)
    
    # Check for rate limit errors (429 or rate_limit_exceeded)
    if "rate_limit" in error_str.lower() or "429" in error_str:
        if "TPD" in error_str or "tokens per day" in error_str.lower():
            return "API rate limit exceeded (daily token limit reached). Please try again later or upgrade your Groq API plan."
        elif "RPM" in error_str or "requests per minute" in error_str.lower():
            return "API rate limit exceeded (too many requests per minute). Please wait a moment and try again."
        else:
            return "API rate limit exceeded. Please wait a moment and try again."
    
    # Check for 500 Internal Server Error
    if "500" in error_str or "internal_server_error" in error_str.lower():
        return "Groq API encountered an internal server error. This may be temporary. Please try again in a moment."
    
    # Check for authentication errors
    if "401" in error_str or "unauthorized" in error_str.lower() or "invalid api key" in error_str.lower():
        return "API authentication failed. Please check your GROQ_API_KEY in the .env file."
    
    # Check for model-specific errors
    if "model" in error_str.lower() and ("not found" in error_str.lower() or "unavailable" in error_str.lower()):
        return "The requested model is currently unavailable. Please try again later."
    
    # Return generic error message with details
    return f"API error occurred: {error_str[:200]}"


def plan_node(state: RAGState) -> Dict:
    """Generate the research plan"""
    logger.info("🧠 Generating plan...")
    try:
        plan = planner_agent.invoke({"question": state["original_question"]})
        logger.info(f"✅ Plan created with {len(plan.steps)} steps")
        return {"plan": plan, "current_step_index": 0, "past_steps": []}
    except Exception as e:
        logger.error(f"Error in plan node: {e}")
        raise


def retrieval_node(state: RAGState) -> Dict:
    """Retrieve from internal documents"""
    current_step = state["plan"].steps[state["current_step_index"]]
    logger.info(f"🔍 Retrieving from documents (Step {state['current_step_index'] + 1})...")
    
    try:
        # Rewrite query
        past_context = get_past_context_str(state['past_steps'])
        rewritten_query = query_rewriter_agent.invoke({
            "sub_question": current_step.sub_question,
            "keywords": ", ".join(current_step.keywords),
            "past_context": past_context
        })
        logger.info(f"📝 Rewritten Query: {rewritten_query[:100]}...")
        
        # Get supervisor decision
        decision = retrieval_supervisor_agent.invoke({"sub_question": rewritten_query})
        logger.info(f"📡 Strategy: {decision.strategy}")
        
        # Execute search
        if decision.strategy == 'vector_search':
            docs = vector_search_only(
                rewritten_query, 
                section_filter=current_step.document_section if hasattr(current_step, 'document_section') else None, 
                k=TOP_K_RETRIEVAL
            )
        elif decision.strategy == 'keyword_search':
            docs = bm25_search_only(rewritten_query, k=TOP_K_RETRIEVAL)
        else:
            docs = hybrid_search(
                rewritten_query, 
                section_filter=current_step.document_section if hasattr(current_step, 'document_section') else None, 
                k=TOP_K_RETRIEVAL
            )
        
        logger.info(f"🔍 Found {len(docs)} initial documents")
        return {"retrieved_docs": docs}
        
    except Exception as e:
        logger.error(f"Error in retrieval node: {e}")
        return {"retrieved_docs": []}


def web_search_node(state: RAGState) -> Dict:
    """Search the web"""
    current_step = state["plan"].steps[state["current_step_index"]]
    logger.info(f"🌐 Searching web (Step {state['current_step_index'] + 1})...")
    
    try:
        past_context = get_past_context_str(state['past_steps'])
        rewritten_query = query_rewriter_agent.invoke({
            "sub_question": current_step.sub_question,
            "keywords": ", ".join(current_step.keywords),
            "past_context": past_context
        })
        logger.info(f"📝 Rewritten Query: {rewritten_query[:100]}...")
        
        docs = web_search_function(rewritten_query)
        logger.info(f"🔍 Found {len(docs)} web results")
        return {"retrieved_docs": docs}
        
    except Exception as e:
        logger.error(f"Error in web search node: {e}")
        return {"retrieved_docs": []}


def rerank_node(state: RAGState) -> Dict:
    """Rerank for precision"""
    logger.info("🎯 Reranking documents...")
    try:
        current_step = state["plan"].steps[state["current_step_index"]]
        reranked = rerank_documents(current_step.sub_question, state["retrieved_docs"])
        logger.info(f"✅ Reranked to top {len(reranked)} documents")
        return {"reranked_docs": reranked}
    except Exception as e:
        logger.error(f"Error in rerank node: {e}")
        return {"reranked_docs": state.get("retrieved_docs", [])}


def compression_node(state: RAGState) -> Dict:
    """Distill context"""
    logger.info("✂️ Distilling context...")
    try:
        current_step = state["plan"].steps[state["current_step_index"]]
        context = "\n\n".join([doc.page_content for doc in state.get("reranked_docs", [])])
        
        if not context:
            logger.warning("No context to distill")
            return {"synthesized_context": ""}
        
        distiller_prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize these document snippets into one concise, coherent paragraph.
            
            🎯 Goal: Provide clear context that answers: '{question}'
            ⚠️ Focus: Remove redundancy, maintain all key facts
            📝 Output: Only the synthesized context (no extra commentary)"""),
            ("human", "Retrieved Documents:\n{context}")
        ])
        
        distiller_agent = distiller_prompt | ChatGroq(model=REASONING_LLM, temperature=0, groq_api_key=GROQ_API_KEY) | StrOutputParser()
        
        synthesized = distiller_agent.invoke({
            "question": current_step.sub_question,
            "context": context
        })
        logger.info("✅ Context distilled")
        return {"synthesized_context": synthesized}
        
    except Exception as e:
        error_msg = _extract_groq_error_message(e)
        logger.error(f"Error in compression node: {e}")
        logger.warning(f"Compression failed: {error_msg}. Returning original context.")
        # Return original context if compression fails
        context = "\n\n".join([doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content 
                              for doc in state.get("reranked_docs", [])])
        return {"synthesized_context": context}


def reflection_node(state: RAGState) -> Dict:
    """Reflect and update history"""
    logger.info("🤔 Reflecting on findings...")
    try:
        current_step_index = state["current_step_index"]
        current_step = state["plan"].steps[current_step_index]
        
        summary = reflection_agent.invoke({
            "sub_question": current_step.sub_question,
            "context": state.get('synthesized_context', '')
        })
        
        new_past_step = {
            "step_index": current_step_index + 1,
            "sub_question": current_step.sub_question,
            "retrieved_docs": state.get('reranked_docs', []),
            "summary": summary
        }
        
        logger.info(f"📝 Summary: {summary[:100]}...")
        
        return {
            "past_steps": state["past_steps"] + [new_past_step],
            "current_step_index": current_step_index + 1
        }
        
    except Exception as e:
        error_msg = _extract_groq_error_message(e)
        logger.error(f"Error in reflection node: {e}")
        logger.warning(f"Reflection failed: {error_msg}. Creating summary from context.")
        
        # Create a fallback summary from the synthesized context
        current_step_index = state.get("current_step_index", 0)
        current_step = state["plan"].steps[current_step_index] if state.get("plan") and current_step_index < len(state["plan"].steps) else None
        
        fallback_summary = state.get('synthesized_context', 'No context available.')[:200]
        if len(fallback_summary) == 200:
            fallback_summary += "..."
        
        new_past_step = {
            "step_index": current_step_index + 1,
            "sub_question": current_step.sub_question if current_step else "Unknown",
            "retrieved_docs": state.get('reranked_docs', []),
            "summary": f"⚠️ Reflection failed: {error_msg}. Context: {fallback_summary}"
        }
        
        return {
            "past_steps": state.get("past_steps", []) + [new_past_step],
            "current_step_index": current_step_index + 1
        }


def final_answer_node(state: RAGState) -> Dict:
    """Generate final, cited answer"""
    logger.info("✅ Generating final answer with citations...")
    try:
        # Gather all evidence
        final_context = ""
        for i, step in enumerate(state.get('past_steps', [])):
            final_context += f"\n--- Research Step {i+1} ---\n"
            for doc in step.get('retrieved_docs', []):
                source = doc.metadata.get('section') or doc.metadata.get('source')
                final_context += f"Source: {source}\nContent: {doc.page_content}\n\n"
        
        if not final_context:
            logger.warning("No context available for final answer")
            return {"final_answer": "Unable to generate answer due to insufficient information."}
        
        # Generate answer
        final_answer = final_answer_agent.invoke({
            "question": state['original_question'],
            "context": final_context
        })
        
        logger.info("✅ Final answer generated")
        return {"final_answer": final_answer}
        
    except Exception as e:
        error_msg = _extract_groq_error_message(e)
        logger.error(f"Error in final answer node: {e}")
        
        # Provide a more informative error response
        error_response = f"""⚠️ **Unable to generate final answer**

**Reason:** {error_msg}

**Available Information:** Based on the research steps completed, here's what was found:

"""
        # Include summaries from past steps if available
        if state.get('past_steps'):
            for i, step in enumerate(state.get('past_steps', []), 1):
                error_response += f"**Step {i}:** {step.get('sub_question', 'N/A')}\n"
                error_response += f"*Summary:* {step.get('summary', 'No summary available')}\n\n"
        else:
            error_response += "No research steps were completed."
        
        return {"final_answer": error_response}

