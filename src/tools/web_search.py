"""
Web search tool using Tavily API.
"""
from typing import List
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from config.config import TAVILY_API_KEY, logger


# Initialize web search tool
try:
    web_search_tool = TavilySearch(api_key=TAVILY_API_KEY)
    logger.info("✅ Web search tool initialized")
except Exception as e:
    logger.error(f"Error initializing web search tool: {e}")
    web_search_tool = None


def web_search_function(query: str) -> List[Document]:
    """
    Search the web and format results as Documents.
    
    Args:
        query: Search query
        
    Returns:
        List of Document objects with web results
    """
    if web_search_tool is None:
        logger.error("Web search tool not initialized")
        return []
    
    try:
        result = web_search_tool.invoke(query)
        
        # TavilySearch returns a dict with a 'results' key containing a list
        documents = []
        if isinstance(result, dict) and "results" in result:
            for res in result["results"]:
                content = res.get("content", res.get("title", ""))
                url = res.get("url", "Unknown")
                title = res.get("title", "")
                
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": url, "title": title}
                    )
                )
        
        logger.info(f"Web search returned {len(documents)} results for query: {query[:100]}")
        return documents
        
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

