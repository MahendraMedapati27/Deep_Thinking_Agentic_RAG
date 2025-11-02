"""
Configuration module for Deep-Thinking RAG system.
Centralizes all configuration settings and validates environment setup.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables (override=True to pick up changes after .env edits)
load_dotenv(override=True)

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
LOG_FILE = LOGS_DIR / "deep_thinking_rag.log"
logger.add(
    str(LOG_FILE),
    rotation="100 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
)

# LLM Configuration - Using Groq
# Note: llama-3.1-8b-instant has much better rate limits (no daily token limit on free tier)
# Valid Groq models: llama-3.1-70b-versatile, llama-3.1-8b-instant, llama-3.3-70b-versatile
# For free tier with rate limit issues, use llama-3.1-8b-instant
REASONING_LLM = os.getenv("REASONING_LLM", "llama-3.1-8b-instant")  # Better rate limits on free tier 
FAST_LLM = os.getenv("FAST_LLM", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Free sentence-transformers model
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval Configuration
TOP_K_RETRIEVAL = 10
TOP_N_RERANK = 3
MAX_REASONING_ITERATIONS = 7
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Debug: Log API key status (without exposing the full key)
if GROQ_API_KEY:
    api_key_preview = f"{GROQ_API_KEY[:10]}...{GROQ_API_KEY[-4:]}" if len(GROQ_API_KEY) > 14 else "***"
    logger.info(f"✅ GROQ_API_KEY loaded: {api_key_preview}")
    logger.info(f"✅ Using model: {REASONING_LLM}")
else:
    logger.warning("⚠️ GROQ_API_KEY not found in environment variables")

# LangSmith Configuration
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "Deep-Thinking-RAG")

# Validate API keys
def validate_api_keys():
    """Validate that required API keys are present."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    
    if missing:
        error_msg = f"Missing required API keys: {', '.join(missing)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("✅ All required API keys are present")

# Call validation on import
try:
    validate_api_keys()
except ValueError as e:
    logger.warning(f"API key validation failed: {e}. Please set up your .env file.")

# Export configuration
__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "VECTOR_STORE_DIR",
    "LOGS_DIR",
    "REASONING_LLM",
    "FAST_LLM",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "TOP_K_RETRIEVAL",
    "TOP_N_RERANK",
    "MAX_REASONING_ITERATIONS",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "GROQ_API_KEY",
    "LANGSMITH_API_KEY",
    "TAVILY_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "logger",
]

