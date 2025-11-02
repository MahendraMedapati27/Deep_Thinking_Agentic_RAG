"""
High-precision reranking using cross-encoder models.
"""
import sys
from typing import List, Optional
from langchain_core.documents import Document
from config.config import RERANKER_MODEL, TOP_N_RERANK, logger


# Global reranker instance (lazy initialization)
_reranker: Optional[object] = None
_initialization_attempted: bool = False


def _get_reranker():
    """
    Lazy initialization of the reranker model.
    Imports CrossEncoder only when needed to avoid atexit registration issues.
    """
    global _reranker, _initialization_attempted
    
    # Check if Python is shutting down (if available)
    try:
        if hasattr(sys, 'is_finalizing') and sys.is_finalizing():
            return None
    except (AttributeError, RuntimeError):
        pass
    
    # Return cached instance if already initialized
    if _reranker is not None:
        return _reranker
    
    # Only attempt initialization once
    if _initialization_attempted:
        return None
    
    _initialization_attempted = True
    
    try:
        # Lazy import to avoid issues during Python shutdown
        from sentence_transformers import CrossEncoder
        
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info(f"✅ Reranker initialized: {RERANKER_MODEL}")
        return _reranker
        
    except RuntimeError as e:
        # Handle shutdown-related errors gracefully
        if "atexit" in str(e).lower() or "shutdown" in str(e).lower():
            logger.debug("Reranker initialization skipped during shutdown")
            return None
        logger.error(f"Error initializing reranker: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing reranker: {e}")
        return None


def rerank_documents(query: str, documents: List[Document]) -> List[Document]:
    """
    Rerank documents using cross-encoder for high precision.
    
    Args:
        query: Search query
        documents: List of documents to rerank
        
    Returns:
        Top N reranked documents
    """
    if not documents:
        return []
    
    # Get reranker instance (lazy initialization)
    reranker = _get_reranker()
    
    if reranker is None:
        logger.warning("Reranker not available, returning original documents")
        return documents[:TOP_N_RERANK]
    
    try:
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = reranker.predict(pairs)
        
        # Sort and return top N
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = [doc for doc, score in doc_scores[:TOP_N_RERANK]]
        logger.debug(f"Reranked {len(documents)} documents to top {len(results)}")
        return results
        
    except RuntimeError as e:
        # Handle shutdown-related errors gracefully
        if "atexit" in str(e).lower() or "shutdown" in str(e).lower():
            logger.debug("Reranking skipped during shutdown")
            return documents[:TOP_N_RERANK]
        logger.error(f"Error in reranking: {e}")
        return documents[:TOP_N_RERANK]
    except Exception as e:
        logger.error(f"Error in reranking: {e}")
        return documents[:TOP_N_RERANK]

