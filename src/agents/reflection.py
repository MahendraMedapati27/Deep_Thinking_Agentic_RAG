"""
Reflection agent that summarizes key findings from research steps.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config.config import REASONING_LLM, GROQ_API_KEY, logger


reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Summarize the key findings from the 
    retrieved context in ONE CONCISE SENTENCE. This summary will be added to our 
    research history."""),
    ("human", """Current sub-question: {sub_question}
    
    Distilled context: {context}""")
])

reasoning_llm = ChatGroq(model=REASONING_LLM, temperature=0, groq_api_key=GROQ_API_KEY)
reflection_agent = reflection_prompt | reasoning_llm | StrOutputParser()

logger.info("✅ Reflection agent initialized")

