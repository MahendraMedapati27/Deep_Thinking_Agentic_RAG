"""
Main entry point for Deep-Thinking RAG system (command-line interface).
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import build_graph
from src.retrieval import (
    initialize_vector_store,
    load_existing_vector_store,
    initialize_search
)
from src.retrieval.document_processor import process_documents
from config.config import DATA_DIR, logger


def main():
    """Main execution function"""
    print("=" * 80)
    print("🧠 Deep-Thinking RAG System")
    print("=" * 80)
    print()
    
    # Check for documents
    data_files = list(DATA_DIR.glob("*.txt"))
    
    if not data_files:
        print("❌ No documents found in data/ directory")
        print("Please add a document (e.g., text file) to the data/ folder")
        return
    
    print(f"📁 Found {len(data_files)} document(s)")
    
    # Try to load existing vector store
    vector_store = load_existing_vector_store()
    
    if vector_store is None:
        print("\n🔄 Initializing vector store...")
        # Process first document
        file_path = data_files[0]
        documents = process_documents(str(file_path))
        
        # Initialize vector store
        vector_store = initialize_vector_store(documents)
        
        # Initialize search strategies
        initialize_search(vector_store, documents)
        print("✅ Vector store initialized")
    else:
        print("✅ Using existing vector store")
        # We need to reload documents for BM25
        file_path = data_files[0]
        documents = process_documents(str(file_path))
        initialize_search(vector_store, documents)
    
    # Build graph
    print("\n🔗 Building workflow graph...")
    graph = build_graph()
    
    # Get user query
    print("\n" + "=" * 80)
    query = input("\n🔍 Enter your question (or 'quit' to exit):\n> ")
    
    while query.lower() not in ['quit', 'exit', 'q']:
        print("\n🧠 Processing your query...")
        print("-" * 80)
        
        try:
            initial_state = {"original_question": query}
            final_state = None
            
            # Stream execution
            for chunk in graph.stream(initial_state, stream_mode="values"):
                final_state = chunk
            
            # Display answer
            if final_state and 'final_answer' in final_state:
                print("\n📝 Final Answer:")
                print("=" * 80)
                print(final_state['final_answer'])
                print("=" * 80)
                
                # Show research history
                if final_state.get('past_steps'):
                    print("\n📚 Research History:")
                    print("-" * 80)
                    for i, step in enumerate(final_state['past_steps'], 1):
                        print(f"\nStep {i}: {step['sub_question']}")
                        print(f"Summary: {step['summary']}")
            
            print("\n" + "=" * 80)
            query = input("\n🔍 Enter your next question (or 'quit' to exit):\n> ")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Error in main: {e}")
            query = input("\n🔍 Try another question (or 'quit' to exit):\n> ")
    
    print("\n👋 Thanks for using Deep-Thinking RAG!")


if __name__ == "__main__":
    main()

