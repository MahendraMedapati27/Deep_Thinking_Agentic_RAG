"""
Policy agent that decides when to continue research or finish and generate the final answer.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal
from config.config import REASONING_LLM, GROQ_API_KEY, logger


class Decision(BaseModel):
    next_action: Literal["CONTINUE_PLAN", "FINISH"]
    justification: str


policy_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a master strategist analyzing research progress.

    📊 You have:
    - Original question
    - Initial plan
    - History of completed steps with summaries
    
    🎯 Decide:
    - **FINISH**: If collected information sufficiently answers the original question
    - **CONTINUE_PLAN**: If more research steps are needed
    
    Think strategically about information completeness."""),
    ("human", """Original Question: {question}
    
    Initial Plan: {plan}
    
    Research History: {history}""")
])

reasoning_llm = ChatGroq(model=REASONING_LLM, temperature=0, groq_api_key=GROQ_API_KEY)
policy_agent = policy_prompt | reasoning_llm.with_structured_output(Decision)

logger.info("✅ Policy agent initialized")

