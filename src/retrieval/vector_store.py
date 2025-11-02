"""
Vector store initialization and management.
"""
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config.config import EMBEDDING_MODEL, VECTOR_STORE_DIR, logger


# Global vector store instance
_vector_store: Optional[Chroma] = None


def initialize_vector_store(documents: List[Document]) -> Chroma:
    """
    Initialize the vector store with documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Chroma vector store
    """
    global _vector_store
    
    logger.info(f"Initializing vector store with {len(documents)} documents")
    
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        _vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=str(VECTOR_STORE_DIR)
        )
        
        logger.info("✅ Vector store initialized successfully")
        return _vector_store
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise


def get_vector_store() -> Optional[Chroma]:
    """
    Get the global vector store instance.
    
    Returns:
        Chroma vector store or None if not initialized
    """
    return _vector_store


def load_existing_vector_store() -> Optional[Chroma]:
    """
    Load an existing vector store from disk.
    
    Returns:
        Chroma vector store or None if not found
    """
    global _vector_store
    
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        _vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embedding_function
        )
        
        # Verify the store has documents
        collection = _vector_store._collection
        count = collection.count()
        
        if count > 0:
            logger.info(f"✅ Loaded existing vector store with {count} documents")
            return _vector_store
        else:
            logger.warning("Vector store exists but contains no documents")
            _vector_store = None
            return None
            
    except Exception as e:
        logger.warning(f"Could not load existing vector store: {e}")
        _vector_store = None
        return None


def get_all_documents_from_store() -> List[Document]:
    """
    Retrieve all documents from the loaded vector store for BM25 initialization.
    
    Returns:
        List of Document objects
    """
    global _vector_store
    
    if _vector_store is None:
        logger.error("Vector store not initialized")
        return []
    
    try:
        # Get all documents by searching with a very broad query
        all_docs = _vector_store.similarity_search("", k=10000)  # Large k to get all docs
        
        # If we have IDs in metadata, we can reconstruct the exact documents
        if all_docs and 'id' in all_docs[0].metadata:
            logger.info(f"Retrieved {len(all_docs)} documents from vector store")
            return all_docs
        else:
            logger.warning("Could not retrieve all documents from vector store")
            return all_docs
            
    except Exception as e:
        logger.error(f"Error retrieving documents from vector store: {e}")
        return []

