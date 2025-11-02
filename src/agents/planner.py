"""
Planning agent that decomposes complex queries into multi-step research plans.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from config.config import REASONING_LLM, GROQ_API_KEY, logger


class Step(BaseModel):
    """A single step in the research plan"""
    sub_question: str = Field(description="Specific question for this step")
    justification: str = Field(description="Why this step is necessary")
    tool: Literal["search_documents", "search_web"] = Field(description="Tool to use")
    keywords: List[str] = Field(description="Search keywords")
    document_section: Optional[str] = Field(description="Target section (for search_documents)")


class Plan(BaseModel):
    """Complete research strategy"""
    steps: List[Step] = Field(description="Multi-step research plan")


planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research planner. Break down complex queries into steps.

🔧 Available Tools:
1. `search_documents`: Search uploaded documents (historical data, sections, general content)
2. `search_web`: Search the internet (recent news, current information, external sources)

📋 Your Task:
- Decompose the query into simple, sequential sub-questions
- Choose the appropriate tool for each step
- For `search_documents`, optionally identify relevant document sections if available
- Provide critical keywords for each search

💡 Tips:
- Use `search_documents` for information likely in the uploaded files
- Use `search_web` for recent events or external information
- Balance between thoroughness and efficiency"""),
    ("human", "User Query: {question}")
])

reasoning_llm = ChatGroq(model=REASONING_LLM, temperature=0, groq_api_key=GROQ_API_KEY)
planner_agent = planner_prompt | reasoning_llm.with_structured_output(Plan)

logger.info("✅ Planner agent initialized")

