"""
Query rewriting agent that optimizes search queries for better retrieval.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config.config import REASONING_LLM, GROQ_API_KEY, logger


query_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a search optimization expert. Transform sub-questions into 
    highly effective search queries using:
    - Specific terminology from the domain
    - Keywords from the research plan
    - Context from previous findings
    
    Goal: Maximize retrieval of relevant documents."""),
    ("human", """Current sub-question: {sub_question}
    Keywords: {keywords}
    Past findings: {past_context}""")
])

reasoning_llm = ChatGroq(model=REASONING_LLM, temperature=0, groq_api_key=GROQ_API_KEY)
query_rewriter_agent = query_rewriter_prompt | reasoning_llm | StrOutputParser()

logger.info("✅ Query rewriter agent initialized")

