"""
Final answer agent that synthesizes research findings into a comprehensive, cited answer.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config.config import REASONING_LLM, GROQ_API_KEY, logger


final_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert analyst. Synthesize research findings into a 
    comprehensive answer. CITE sources after each claim:
    - For documents: [Source: <section/description>]
    - For web: [Source: <URL>]
    
    Make your answer clear, well-structured, and authoritative.
    Provide thorough analysis while being concise and precise."""),
    ("human", "Question: {question}\n\nContext: {context}")
])

reasoning_llm = ChatGroq(model=REASONING_LLM, temperature=0, groq_api_key=GROQ_API_KEY)
final_answer_agent = final_answer_prompt | reasoning_llm | StrOutputParser()

logger.info("✅ Final answer agent initialized")

