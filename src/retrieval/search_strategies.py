"""
Multiple search strategies: vector, BM25, and hybrid search.
"""
from typing import List, Dict, Optional
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from config.config import logger


# Global variables (initialized by workflow)
vector_store: Optional[Chroma] = None
bm25: Optional[BM25Okapi] = None
doc_map: Dict[str, Document] = {}
doc_ids: List[str] = []


def initialize_search(vs: Chroma, documents: List[Document]):
    """
    Initialize search indices for all strategies.
    
    Args:
        vs: Chroma vector store
        documents: List of documents for BM25 indexing
    """
    global vector_store, bm25, doc_map, doc_ids
    
    try:
        vector_store = vs
        
        # Prepare BM25
        logger.info("Initializing BM25 index...")
        tokenized_corpus = [doc.page_content.split(" ") for doc in documents]
        doc_ids = [doc.metadata["id"] for doc in documents]
        doc_map = {doc.metadata["id"]: doc for doc in documents}
        bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(f"✅ Search strategies initialized with {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Error initializing search strategies: {e}")
        raise


def vector_search_only(query: str, section_filter: Optional[str] = None, k: int = 10) -> List[Document]:
    """
    Vector similarity search with optional metadata filtering.
    
    Args:
        query: Search query
        section_filter: Optional section filter
        k: Number of results
        
    Returns:
        List of documents
    """
    if vector_store is None:
        logger.error("Vector store not initialized")
        return []
    
    try:
        filter_dict = {"section": section_filter} if section_filter else None
        results = vector_store.similarity_search(query, k=k, filter=filter_dict)
        logger.debug(f"Vector search returned {len(results)} documents")
        return results
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []


def bm25_search_only(query: str, k: int = 10) -> List[Document]:
    """
    BM25 keyword search.
    
    Args:
        query: Search query
        k: Number of results
        
    Returns:
        List of documents
    """
    if bm25 is None:
        logger.error("BM25 index not initialized")
        return []
    
    try:
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(bm25_scores)[::-1][:k]
        results = [doc_map[doc_ids[i]] for i in top_k_indices]
        logger.debug(f"BM25 search returned {len(results)} documents")
        return results
        
    except Exception as e:
        logger.error(f"Error in BM25 search: {e}")
        return []


def hybrid_search(query: str, section_filter: Optional[str] = None, k: int = 10) -> List[Document]:
    """
    Hybrid search with Reciprocal Rank Fusion (RRF).
    
    Args:
        query: Search query
        section_filter: Optional section filter
        k: Number of results
        
    Returns:
        List of documents
    """
    try:
        bm25_docs = bm25_search_only(query, k=k)
        semantic_docs = vector_search_only(query, section_filter=section_filter, k=k)
        
        # Combine using RRF
        all_docs_dict = {doc.metadata["id"]: doc for doc in bm25_docs + semantic_docs}
        ranked_lists = [
            [doc.metadata["id"] for doc in bm25_docs],
            [doc.metadata["id"] for doc in semantic_docs]
        ]
        
        rrf_scores = {}
        for doc_list in ranked_lists:
            for i, doc_id in enumerate(doc_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1 / (i + 61)  # RRF formula
        
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = [all_docs_dict[doc_id] for doc_id in sorted_doc_ids[:k]]
        logger.debug(f"Hybrid search returned {len(results)} documents")
        return results
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        return []

