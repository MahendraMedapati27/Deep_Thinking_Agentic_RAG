"""
Helper utility functions for the Deep-Thinking RAG system.
"""
from typing import List
from langchain_core.documents import Document
from config.config import logger


def get_past_context_str(past_steps: List[dict]) -> str:
    """
    Convert past research steps into a formatted string.
    
    Args:
        past_steps: List of past research steps
        
    Returns:
        Formatted string summarizing past research
    """
    if not past_steps:
        return "No previous research steps."
    
    context_parts = []
    for i, step in enumerate(past_steps, 1):
        context_parts.append(f"Step {i}: {step.get('summary', 'No summary available')}")
    
    return "\n\n".join(context_parts)


def format_documents_for_display(documents: List[Document], max_chars: int = 500) -> str:
    """
    Format documents for display in the UI.
    
    Args:
        documents: List of Document objects
        max_chars: Maximum characters per document
        
    Returns:
        Formatted string
    """
    if not documents:
        return "No documents retrieved."
    
    formatted = []
    for i, doc in enumerate(documents, 1):
        content = doc.page_content[:max_chars]
        if len(doc.page_content) > max_chars:
            content += "..."
        
        source = doc.metadata.get('section') or doc.metadata.get('source', 'Unknown')
        formatted.append(f"**Document {i}** (Source: {source})\n{content}\n")
    
    return "\n---\n".join(formatted)


def calculate_retrieval_metrics(retrieved_docs: List[Document], 
                                relevant_docs: List[Document] = None) -> dict:
    """
    Calculate basic retrieval metrics.
    
    Args:
        retrieved_docs: Documents retrieved by the system
        relevant_docs: Actually relevant documents (for evaluation)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "total_retrieved": len(retrieved_docs),
        "avg_doc_length": sum(len(doc.page_content) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
    }
    
    if relevant_docs:
        relevant_ids = {doc.metadata.get('id') for doc in relevant_docs if doc.metadata.get('id')}
        retrieved_ids = {doc.metadata.get('id') for doc in retrieved_docs if doc.metadata.get('id')}
        
        precision = len(relevant_ids & retrieved_ids) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(relevant_ids & retrieved_ids) / len(relevant_ids) if relevant_ids else 0
        
        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        })
    
    return metrics

