@echo off
REM Deep-Thinking RAG Setup Script for Windows

echo ======================================================
echo Setting up Deep-Thinking RAG System
echo ======================================================
echo.

REM Create conda environment
echo Creating conda environment...
conda env create -f environment.yml

REM Activate environment
echo Activating environment...
call conda activate deep-thinking-rag

REM Copy .env.example to .env
if not exist .env (
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo Please edit .env file and add your API keys
) else (
    echo .env file already exists
)

REM Create necessary directories
echo Creating directories...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist vector_store mkdir vector_store

echo.
echo ======================================================
echo Setup complete!
echo ======================================================
echo.
echo Next steps:
echo 1. Edit .env file and add your API keys:
echo    - OPENAI_API_KEY
echo    - TAVILY_API_KEY
echo    - LANGSMITH_API_KEY (optional)
echo.
echo 2. Add your document to the data\ folder
echo.
echo 3. Run the system:
echo    streamlit run app.py
echo    OR
echo    python main.py
echo.
pause

