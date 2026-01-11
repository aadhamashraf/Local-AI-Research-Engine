"""
Streamlit App - Professional Web Interface for Research Engine
"""

import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ResearchEngine


# Page configuration
st.set_page_config(
    page_title="Research Intelligence Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Elite Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 100%);
        color: #e4e4e7;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #1a1d2e 100%);
        border-right: 1px solid #2d3348;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: #e4e4e7;
    }
    
    /* Custom Headers */
    .elite-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .elite-subheader {
        font-size: 1.1rem;
        color: #a1a1aa;
        font-weight: 400;
        margin-bottom: 2.5rem;
        letter-spacing: 0.01em;
    }
    
    /* Card Components */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.3);
        border-color: #475569;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #e4e4e7;
        line-height: 1;
    }
    
    .metric-status {
        font-size: 0.875rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Citation Box */
    .citation-container {
        background: linear-gradient(135deg, #1e293b 0%, #27344a 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 3px solid #667eea;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .citation-container:hover {
        background: linear-gradient(135deg, #27344a 0%, #2d3a52 100%);
        border-left-color: #764ba2;
    }
    
    /* Answer Section */
    .answer-section {
        background: linear-gradient(135deg, #1a1d2e 0%, #252a42 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #2d3348;
        margin: 1.5rem 0;
        line-height: 1.7;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background: #1e293b;
        border: 1px solid #334155;
        color: #e4e4e7;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(102, 126, 234, 0.5);
    }
    
    /* Secondary Button */
    .stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid #475569;
        color: #e4e4e7;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #334155;
        border-color: #64748b;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 2rem 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 8px;
        border: 1px solid #334155;
        color: #e4e4e7;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: #27344a;
        border-color: #475569;
    }
    
    /* Code Block */
    .stCodeBlock {
        background: #0f1117;
        border: 1px solid #1e293b;
        border-radius: 8px;
    }
    
    /* Info/Warning/Success Boxes */
    .stAlert {
        background: #1e293b;
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #1e293b;
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #334155;
    }
    
    /* Navigation */
    .nav-item {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-item:hover {
        background: #1e293b;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    /* Document Item */
    .doc-item {
        background: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
    }
    
    .doc-item:hover {
        background: #27344a;
        border-color: #475569;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-success {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
    }
    
    .badge-warning {
        background: rgba(251, 146, 60, 0.2);
        color: #fb923c;
    }
    
    .badge-error {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1d2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    """Initialize and cache the research engine."""
    return ResearchEngine()


def render_metric_card(label, value, status=None, status_type="success"):
    """Render a professional metric card."""
    status_class = f"badge-{status_type}" if status else ""
    status_html = f'<div class="badge {status_class}">{status}</div>' if status else ""
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {status_html}
    </div>
    """


def render_search_page():
    """Render the main search interface."""
    st.markdown('<div class="elite-header">Research Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="elite-subheader">Advanced semantic search with cited answers from your document collection</div>', unsafe_allow_html=True)
    
    # Initialize engine
    engine = get_engine()
    
    # Search interface
    question = st.text_input(
        "QUERY",
        placeholder="Enter your research question...",
        key="search_query",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([2, 2, 8])
    
    with col1:
        search_button = st.button("SEARCH", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("CLEAR", use_container_width=True, key="clear_btn")
    
    if clear_button:
        st.session_state.search_query = ""
        st.session_state.last_result = None
        st.rerun()
    
    # Display results
    if search_button and question:
        with st.spinner("Analyzing documents and generating response..."):
            result = engine.query(question)
            st.session_state.last_result = result
    
    if "last_result" in st.session_state and st.session_state.last_result:
        result = st.session_state.last_result
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Metrics section
        confidence = result.get("confidence", 0)
        if confidence >= 0.8:
            confidence_status = "HIGH CONFIDENCE"
            confidence_type = "success"
        elif confidence >= 0.5:
            confidence_status = "MODERATE"
            confidence_type = "warning"
        else:
            confidence_status = "LOW CONFIDENCE"
            confidence_type = "error"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                render_metric_card(
                    "Confidence Score",
                    f"{confidence:.0%}",
                    confidence_status,
                    confidence_type
                ),
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                render_metric_card(
                    "Source Documents",
                    result.get("num_sources", 0)
                ),
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                render_metric_card(
                    "Citations Found",
                    len(result.get("citations", []))
                ),
                unsafe_allow_html=True
            )
        
        st.markdown("")
        
        # Answer section
        st.markdown("### GENERATED RESPONSE")
        st.markdown(
            f'<div class="answer-section">{result.get("answer", "No answer generated.")}</div>',
            unsafe_allow_html=True
        )
        
        # Citations
        if result.get("citations"):
            st.markdown("### SUPPORTING CITATIONS")
            
            for i, citation in enumerate(result["citations"], 1):
                with st.expander(f"Citation {i}: {citation['source_name']} — Section {citation.get('section', 'N/A')}"):
                    st.markdown("**Content Preview**")
                    st.text(citation.get("content", "No preview available"))
                    
                    if st.button(f"View Complete Content", key=f"view_{i}"):
                        st.markdown("**Full Document Content**")
                        st.text_area(
                            "Content",
                            citation.get("full_content", ""),
                            height=300,
                            key=f"full_{i}",
                            label_visibility="collapsed"
                        )


def render_document_management():
    """Render document management page."""
    st.markdown('<div class="elite-header">Document Repository</div>', unsafe_allow_html=True)
    st.markdown('<div class="elite-subheader">Manage and index your research document collection</div>', unsafe_allow_html=True)
    
    engine = get_engine()
    
    # Upload section
    st.markdown("### UPLOAD DOCUMENTS")
    
    uploaded_files = st.file_uploader(
        "Drag and drop files or click to browse",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("PROCESS AND INDEX", type="primary"):
            docs_dir = Path(engine.config["paths"]["documents"])
            docs_dir.mkdir(parents=True, exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save file
                file_path = docs_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("Indexing documents...")
            engine.ingest_documents()
            
            progress_bar.progress(1.0)
            status_text.empty()
            st.success("Documents successfully indexed and ready for search")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Indexed documents
    st.markdown("### INDEXED COLLECTION")
    
    sources = engine.vector_store.get_all_sources()
    
    if sources:
        st.markdown(
            render_metric_card("Total Documents", len(sources)),
            unsafe_allow_html=True
        )
        
        st.markdown("")
        
        for source in sources:
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(f'<div class="doc-item"><span>{source}</span></div>', unsafe_allow_html=True)
            with col2:
                if st.button("DELETE", key=f"del_{source}"):
                    engine.vector_store.delete_by_source(source)
                    st.success(f"Removed {source}")
                    st.rerun()
    else:
        st.info("No documents in the repository. Upload documents to begin.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Reindex all
    col1, col2, col3 = st.columns([2, 2, 8])
    with col1:
        if st.button("REBUILD INDEX"):
            with st.spinner("Rebuilding complete index..."):
                engine.vector_store.clear()
                engine.keyword_index.clear()
                engine.ingest_documents()
                st.success("Index rebuild complete")


def render_knowledge_graph():
    """Render knowledge graph visualization."""
    st.markdown('<div class="elite-header">Knowledge Graph</div>', unsafe_allow_html=True)
    st.markdown('<div class="elite-subheader">Interactive visualization of entity relationships and connections</div>', unsafe_allow_html=True)
    
    engine = get_engine()
    graph = engine.knowledge_graph.graph
    
    if graph.number_of_nodes() == 0:
        st.warning("Knowledge graph is empty. Index documents to populate the graph.")
        return
    
    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            render_metric_card("Entities", graph.number_of_nodes()),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            render_metric_card("Relationships", graph.number_of_edges()),
            unsafe_allow_html=True
        )
    
    st.markdown("")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        entity_types = set()
        for _, data in graph.nodes(data=True):
            entity_types.add(data.get("type", "Unknown"))
        
        selected_type = st.selectbox(
            "ENTITY TYPE FILTER",
            ["All Types"] + sorted(list(entity_types))
        )
    
    with col2:
        max_nodes = st.slider("MAXIMUM NODES", 10, 100, 50)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Build visualization
    nodes = []
    edges = []
    
    node_count = 0
    for node_id, node_data in graph.nodes(data=True):
        if node_count >= max_nodes:
            break
        
        node_type = node_data.get("type", "Unknown")
        
        if selected_type != "All Types" and node_type != selected_type:
            continue
        
        # Color scheme
        color_map = {
            "Concept": "#667eea",
            "Method": "#fb923c",
            "Author": "#4ade80",
            "Paper": "#f87171",
            "Tool": "#a78bfa"
        }
        
        nodes.append(Node(
            id=node_id,
            label=node_data.get("label", node_id)[:30],
            size=25,
            color=color_map.get(node_type, "#64748b")
        ))
        
        node_count += 1
    
    # Add edges
    for source, target, edge_data in graph.edges(data=True):
        if source in [n.id for n in nodes] and target in [n.id for n in nodes]:
            edges.append(Edge(
                source=source,
                target=target,
                label=edge_data.get("type", "")
            ))
    
    # Display graph
    config = Config(
        width=900,
        height=650,
        directed=True,
        physics=True,
        hierarchical=False
    )
    
    if nodes:
        agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.warning("No entities match the current filter criteria.")
    
    # Entity search
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### ENTITY SEARCH")
    
    search_term = st.text_input("Search entities by name or type", label_visibility="collapsed", placeholder="Enter search term...")
    
    if search_term:
        results = engine.knowledge_graph.search_entities(search_term, limit=10)
        
        if results:
            for result in results:
                with st.expander(f"{result['label']} — {result['type']}"):
                    st.markdown(f"**Entity Type:** {result['type']}")
                    st.markdown(f"**Identifier:** {result['entity_id']}")
                    
                    # Get related entities
                    related = engine.knowledge_graph.get_related_entities(
                        result['entity_id'],
                        max_depth=1,
                        max_results=5
                    )
                    
                    if related:
                        st.markdown("**Connected Entities:**")
                        for rel in related:
                            st.markdown(f"— {rel['label']} ({rel['relationship']})")
        else:
            st.info("No entities found matching your search.")


def render_settings():
    """Render settings and configuration page."""
    st.markdown('<div class="elite-header">System Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="elite-subheader">View system status and adjust operational parameters</div>', unsafe_allow_html=True)
    
    engine = get_engine()
    
    st.markdown("### INFRASTRUCTURE STATUS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Ollama Configuration**")
        st.code(f"""
Base URL: {engine.config['ollama']['base_url']}
LLM Model: {engine.config['ollama']['llm_model']}
Embedding Model: {engine.config['ollama']['embedding_model']}
        """, language="yaml")
    
    with col2:
        st.markdown("**Index Statistics**")
        st.code(f"""
Vector Store: {engine.vector_store.count()} documents
Keyword Index: {engine.keyword_index.count()} documents
Knowledge Graph: {engine.knowledge_graph.graph.number_of_nodes()} nodes
        """, language="yaml")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("### RETRIEVAL PARAMETERS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("Vector Search Top-K", value=engine.config['retrieval']['vector_top_k'], key="vector_k")
        st.number_input("Keyword Search Top-K", value=engine.config['retrieval']['keyword_top_k'], key="keyword_k")
    
    with col2:
        st.number_input("Final Results Top-K", value=engine.config['retrieval']['final_top_k'], key="final_k")
        st.number_input("Reranking Top-K", value=engine.config['retrieval']['rerank_top_k'], key="rerank_k")
    
    st.markdown("")
    
    if st.button("SAVE CONFIGURATION"):
        st.success("Configuration saved successfully. Restart required for changes to take effect.")


def render_advanced_page():
    """Render advanced features page."""
    st.markdown('<div class="elite-header">Advanced Capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="elite-subheader">Comparisons, reviews, and automated analysis</div>', unsafe_allow_html=True)
    
    engine = get_engine()
    
    tab1, tab2, tab3 = st.tabs(["⚖️ PAPER COMPARISON", "📖 LITERATURE REVIEW", "⚠️ CONTRADICTION AUDIT"])
    
    # -------------------------------------------------------------------------
    # Tab 1: Paper Comparison
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### COMPARE DOCUMENTS")
        st.info("Select two documents to compare side-by-side on specific aspects.")
        
        # Get documents
        sources = engine.vector_store.get_all_sources()
        
        if len(sources) < 2:
            st.warning("Need at least 2 documents indexed to compare.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                doc1 = st.selectbox("Document 1", sources, key="doc1")
            with col2:
                doc2 = st.selectbox("Document 2", sources, index=1 if len(sources) > 1 else 0, key="doc2")
            
            aspects = st.multiselect(
                "Aspects to compare:",
                ["Methodology", "Results", "Limitations", "Future Work", "Theory", "Dataset"],
                default=["Methodology", "Results"]
            )
            
            if st.button("RUN COMPARISON", type="primary"):
                if doc1 == doc2:
                    st.error("Please select different documents.")
                elif not aspects:
                    st.error("Please select at least one aspect.")
                else:
                    with st.spinner("Analyzing and comparing documents..."):
                        # Retrieve documents content (simple retrieval by source name)
                        res1 = engine.retriever.retrieve(doc1, final_top_k=1)
                        res2 = engine.retriever.retrieve(doc2, final_top_k=1)
                        
                        if res1 and res2:
                            comparison = engine.paper_comparator.compare(
                                res1[0], res2[0], aspects=aspects
                            )
                            
                            st.markdown("### COMPARISON ANALYSIS")
                            
                            for aspect in aspects:
                                aspect_lower = aspect.lower()
                                if aspect_lower in comparison.get("comparisons", {}):
                                    comp = comparison["comparisons"][aspect_lower]
                                    st.markdown(f"#### {aspect.upper()}")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.markdown(f"**Similarities**")
                                        st.markdown(f"_{comp.get('similarities', 'None')}_")
                                    with col_b:
                                        st.markdown(f"**Differences**")
                                        st.markdown(f"_{comp.get('differences', 'None')}_")
                                    
                                    st.markdown(f"**Verdict:** {comp.get('better_approach', 'Neutral')}")
                                    st.markdown("---")
                            
                            # Export
                            md_output = engine.paper_comparator.export_comparison(comparison, format="markdown")
                            st.download_button("DOWNLOAD REPORT", md_output, "comparison.md")
                        else:
                            st.error("Could not retrieve documents content.")

    # -------------------------------------------------------------------------
    # Tab 2: Literature Review
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### GENERATE LITERATURE REVIEW")
        st.info("Auto-generate a structured review from multiple documents.")
        
        topic = st.text_input("Research Topic:", placeholder="e.g., Deep Learning Optimization Methods")
        max_sources = st.slider("Max Sources Included:", 3, 10, 5)
        
        if st.button("GENERATE REVIEW", type="primary"):
            if not topic:
                st.error("Please enter a topic.")
            else:
                with st.spinner("Synthesizing literature review (this may take a minute)..."):
                    # 1. Search
                    results = engine.retriever.retrieve(topic, final_top_k=max_sources * 2)
                    
                    if not results:
                        st.error("No relevant documents found for this topic.")
                    else:
                        # 2. Rerank
                        reranked = engine.reranker.rerank(topic, results, top_k=max_sources)
                        
                        # 3. Generate
                        review = engine.lit_reviewer.generate_review(topic, reranked)
                        
                        st.markdown("### GENERATED REVIEW")
                        st.markdown(review)
                        
                        st.download_button("DOWNLOAD REVIEW", review, "literature_review.md")

    # -------------------------------------------------------------------------
    # Tab 3: Contradiction Detection
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### DETECT CONTRADICTIONS")
        st.info("Find conflicting information across your knowledge base.")
        
        query = st.text_input("Topic to audit:", placeholder="e.g., Learning rate effects")
        
        if st.button("SCAN FOR CONTRADICTIONS", type="primary"):
            if not query:
                st.error("Please enter a topic.")
            else:
                with st.spinner("Scanning for contradictions..."):
                    results = engine.retriever.retrieve(query, final_top_k=10)
                    contradictions = engine.contradiction_detector.find_contradictions(results)
                    
                    if contradictions:
                        st.warning(f"Found {len(contradictions)} potential contradictions.")
                        
                        for i, c in enumerate(contradictions, 1):
                            with st.expander(f"Conflict {i}: {c.get('severity', 'Unknown').upper()} SEVERITY"):
                                st.markdown(f"**Source 1:** {c.get('source1', 'Unknown')}")
                                st.markdown(f"**Source 2:** {c.get('source2', 'Unknown')}")
                                st.markdown(f"**Explanation:**\n{c.get('explanation', '')}")
                    else:
                        st.success("No obvious contradictions found in the top results.")


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown('<div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 2rem; color: #e4e4e7;">NAVIGATION</div>', unsafe_allow_html=True)
        
        page = st.radio(
            "Select Module",
            ["Research Intelligence", "Document Repository", "Knowledge Graph", "Advanced Capabilities", "System Configuration"],
            label_visibility="collapsed"
        )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("### PLATFORM INFO")
        st.markdown("""
        **Research Intelligence Platform**
        
        Privacy-focused research assistant powered by local AI infrastructure.
        
        **Core Capabilities**
        - Semantic document search
        - Knowledge graph analysis
        - Citation-backed responses
        - Complete data privacy
        
        **Version** 2.0.0
        """)
    
    # Route to pages
    if page == "Research Intelligence":
        render_search_page()
    elif page == "Document Repository":
        render_document_management()
    elif page == "Knowledge Graph":
        render_knowledge_graph()
    elif page == "Advanced Capabilities":
        render_advanced_page()
    elif page == "System Configuration":
        render_settings()


if __name__ == "__main__":
    main()