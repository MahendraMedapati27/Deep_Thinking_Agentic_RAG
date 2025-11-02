#!/bin/bash

# Deep-Thinking RAG Setup Script
echo "🚀 Setting up Deep-Thinking RAG System..."

# Create conda environment
echo "📦 Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo "✅ Environment created. Activating..."
conda activate deep-thinking-rag

# Copy .env.example to .env
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p vector_store

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - OPENAI_API_KEY"
echo "   - TAVILY_API_KEY"
echo "   - LANGSMITH_API_KEY (optional)"
echo ""
echo "2. Add your document to the data/ folder"
echo ""
echo "3. Run the system:"
echo "   streamlit run app.py"
echo "   OR"
echo "   python main.py"

