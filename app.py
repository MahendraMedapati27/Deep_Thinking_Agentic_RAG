"""
Streamlit UI for Deep-Thinking RAG System.
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import build_graph
from src.retrieval import (
    initialize_vector_store,
    load_existing_vector_store,
    initialize_search,
    get_all_documents_from_store
)
from src.retrieval.document_processor import process_documents
from config.config import DATA_DIR, logger

# Page configuration
st.set_page_config(
    page_title="Deep-Thinking RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state.graph = None
    st.session_state.vector_store_initialized = False

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🧠 Deep-Thinking RAG System</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
    An intelligent agentic RAG system with multi-step reasoning and self-critique capabilities
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📁 Document Setup")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=['txt'],
        help="Upload any text document to analyze"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        file_path = DATA_DIR / uploaded_file.name
        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Initialize button
        if st.button("🔄 Initialize Vector Store", type="primary"):
            with st.spinner("Processing documents and initializing vector store..."):
                try:
                    # Process documents
                    documents = process_documents(str(file_path))
                    
                    # Initialize vector store
                    vector_store = initialize_vector_store(documents)
                    
                    # Initialize search strategies
                    initialize_search(vector_store, documents)
                    
                    st.session_state.vector_store_initialized = True
                    st.success("✅ Vector store initialized successfully!")
                    logger.info("Vector store initialized from Streamlit UI")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Error initializing vector store: {e}")
    
    # Check for existing vector store
    if not st.session_state.vector_store_initialized:
        existing_vs = load_existing_vector_store()
        if existing_vs is not None:
            # Initialize BM25 index from the loaded vector store
            docs = get_all_documents_from_store()
            if docs:
                initialize_search(existing_vs, docs)
                logger.info("BM25 index initialized from loaded vector store")
            st.session_state.vector_store_initialized = True
            st.success("✅ Found existing vector store!")
    
    st.divider()
    
    st.subheader("📊 Settings")
    max_iterations = st.slider(
        "Max Reasoning Iterations",
        min_value=3,
        max_value=15,
        value=7,
        help="Maximum number of research steps before forcing completion"
    )
    
    show_intermediate = st.checkbox(
        "Show Intermediate Steps",
        value=True,
        help="Display research process as it happens"
    )

# Main content area
if not st.session_state.vector_store_initialized:
    st.info("""
    👋 **Welcome to Deep-Thinking RAG!**
    
    To get started:
    1. Upload a document in the sidebar
    2. Click "Initialize Vector Store" to process the document
    3. Start asking questions!
    
    **Example queries:**
    - "What are the key themes mentioned in the document?"
    - "Summarize the main points and insights"
    - "Analyze the important sections"
    """)
else:
    st.success("✅ System ready! Start asking questions below.")
    
    # Query input
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_area(
            "🔍 Enter your question:",
            height=100,
            placeholder="Example: What are the key risks mentioned in the document and how do recent market developments affect them?",
            help="Ask complex, multi-step questions that require reasoning"
        )
    
    with col2:
        st.write("")
        st.write("")
        run_button = st.button("🚀 Go", type="primary", use_container_width=True)
    
    # Process query
    if run_button and query:
        # Initialize graph if not already done
        if st.session_state.graph is None:
            with st.spinner("Initializing system..."):
                st.session_state.graph = build_graph()
                logger.info("Graph built from Streamlit UI")
        
        # Run the RAG system
        with st.spinner("🧠 Processing your query..."):
            try:
                initial_state = {"original_question": query}
                final_state = None
                
                # Create containers for streaming output
                if show_intermediate:
                    steps_container = st.container()
                    with steps_container:
                        st.subheader("📊 Research Progress")
                        steps_placeholder = st.empty()
                
                # Stream execution with increased recursion limit
                steps_text = []
                for chunk in st.session_state.graph.stream(initial_state, stream_mode="values", config={"recursion_limit": 50}):
                    final_state = chunk
                    
                    if show_intermediate and 'plan' in chunk and chunk['plan']:
                        plan_str = f"**Research Plan Created:** {len(chunk['plan'].steps)} steps identified\n\n"
                        for i, step in enumerate(chunk['plan'].steps, 1):
                            plan_str += f"{i}. {step.sub_question} → {step.tool}\n"
                        steps_text.append(plan_str)
                    
                    if show_intermediate and 'current_step_index' in chunk:
                        step_idx = chunk['current_step_index']
                        if step_idx > 0:
                            steps_text.append(f"✅ Completed step {step_idx}")
                    
                    if show_intermediate and steps_placeholder:
                        steps_placeholder.markdown("\n".join(steps_text))
                
                # Display final answer
                st.divider()
                st.subheader("✅ Final Answer")
                
                if final_state and 'final_answer' in final_state:
                    st.markdown(final_state['final_answer'])
                    
                    # Show research history
                    if final_state.get('past_steps'):
                        with st.expander("📚 Research History"):
                            for i, step in enumerate(final_state['past_steps'], 1):
                                st.markdown(f"**Step {i}: {step['sub_question']}**")
                                st.markdown(f"Summary: {step['summary']}")
                                st.divider()
                    
                    logger.info("Query processed successfully")
                else:
                    st.error("Unable to generate an answer. Please try rephrasing your question.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error processing query: {e}")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9rem;">
    Built with 🧠 using LangGraph, LangChain, and Streamlit
</div>
""", unsafe_allow_html=True)

