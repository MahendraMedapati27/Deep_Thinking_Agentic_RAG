"""
Retrieval supervisor agent that chooses the best search strategy.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal
from config.config import REASONING_LLM, GROQ_API_KEY, logger


class RetrievalDecision(BaseModel):
    strategy: Literal["vector_search", "keyword_search", "hybrid_search"]
    justification: str


retrieval_supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval strategy expert. Choose the best approach:

    🔮 **Vector Search**: Best for conceptual, semantic queries
    📝 **Keyword Search**: Best for specific terms, exact phrases
    🔀 **Hybrid Search**: Balanced approach, combines both
    
    Decide based on query characteristics."""),
    ("human", "Query: {sub_question}")
])

reasoning_llm = ChatGroq(model=REASONING_LLM, temperature=0, groq_api_key=GROQ_API_KEY)
retrieval_supervisor_agent = retrieval_supervisor_prompt | reasoning_llm.with_structured_output(RetrievalDecision)

logger.info("✅ Retrieval supervisor agent initialized")

