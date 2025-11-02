"""
Document processing with section metadata extraction.
"""
import re
import uuid
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, logger


def process_documents(file_path: str) -> List[Document]:
    """
    Load and process a document with section metadata extraction.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of Document objects with metadata
    """
    logger.info(f"Processing document: {file_path}")
    
    try:
        # Load document
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        raw_text = documents[0].page_content
        
        # Extract section titles (e.g., "ITEM 1A. Risk Factors", "Chapter 1", etc.)
        section_pattern = r"(ITEM\s+\d[A-Z]?\.\s*.*?)(?=\nITEM\s+\d[A-Z]?\.|\Z)"
        sections = re.findall(section_pattern, raw_text, re.IGNORECASE | re.DOTALL)
        
        if not sections:
            logger.warning("No sections found using pattern matching. Creating generic chunks.")
            # Fallback: create chunks without section metadata
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_text(raw_text)
            
            return [
                Document(
                    page_content=chunk,
                    metadata={
                        "source_doc": file_path,
                        "id": str(uuid.uuid4()),
                        "chunk_index": idx
                    }
                ) for idx, chunk in enumerate(chunks)
            ]
        
        # Split text by sections
        section_titles = [t.strip().replace('\n', ' ') for t in sections]
        sections_content = re.split(section_pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
        sections_content = [
            c.strip() for c in sections_content 
            if c.strip() and not any(c.strip().lower().startswith(f"item {i}") for i in range(1, 16))
        ]
        
        # Create chunks with metadata
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        doc_chunks_with_metadata = []
        
        for i, content in enumerate(sections_content):
            if i >= len(section_titles):
                continue
                
            section_title = section_titles[i]
            section_chunks = text_splitter.split_text(content)
            
            for chunk in section_chunks:
                doc_chunks_with_metadata.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "section": section_title,
                            "source_doc": file_path,
                            "id": str(uuid.uuid4())
                        }
                    )
                )
        
        logger.info(f"✅ Created {len(doc_chunks_with_metadata)} metadata-enriched chunks")
        return doc_chunks_with_metadata
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise

