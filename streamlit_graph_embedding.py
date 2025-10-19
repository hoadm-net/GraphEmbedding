#!/usr/bin/env python3
"""
Streamlit UI for Graph Embedding Pipeline
"""

import streamlit as st
import pandas as pd
import os
from graph_embedding import GraphEmbeddingPipeline, load_text_from_file
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Cấu hình trang
st.set_page_config(
    page_title="Graph Embedding Pipeline",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("🎯 Graph Embedding Pipeline")
    st.markdown("Pipeline hoàn chỉnh từ văn bản tiếng Việt → Graph Embeddings")
    
    # Sidebar cho cấu hình
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Pipeline parameters
        st.subheader("Pipeline Parameters")
        window_size = st.slider("Window Size", 1, 10, 3, help="Kích thước cửa sổ co-occurrence")
        vector_size = st.selectbox("Vector Size", [32, 64, 128, 256], index=1, help="Kích thước embedding vector")
        walk_length = st.slider("Walk Length", 5, 50, 10, help="Độ dài mỗi random walk")
        num_walks = st.slider("Number of Walks per Node", 1, 20, 5, help="Số walks từ mỗi node")
        epochs = st.slider("Training Epochs", 5, 100, 20, help="Số epochs training Word2Vec")
        min_frequency = st.slider("Minimum Word Frequency", 1, 10, 1, help="Tần suất tối thiểu của từ")
        
        # Visualization parameters
        st.subheader("Visualization")
        viz_method = st.selectbox("Dimensionality Reduction", ["PCA", "t-SNE"], help="Phương pháp giảm chiều để visualization")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Enter text manually", "Upload file", "Select from data folder"])
        
        text_content = ""
        text_name = ""
        
        if input_method == "Enter text manually":
            text_content = st.text_area(
                "Enter Vietnamese text:", 
                height=200,
                placeholder="Nhập văn bản tiếng Việt ở đây..."
            )
            text_name = st.text_input("Name for output files:", "manual_text")
            
        elif input_method == "Upload file":
            uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
            if uploaded_file is not None:
                text_content = str(uploaded_file.read(), "utf-8")
                text_name = os.path.splitext(uploaded_file.name)[0]
                st.success(f"✅ Loaded file: {uploaded_file.name}")
                
        elif input_method == "Select from data folder":
            data_folder = "data"
            if os.path.exists(data_folder):
                files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
                if files:
                    selected_file = st.selectbox("Select a file:", files)
                    if selected_file:
                        file_path = os.path.join(data_folder, selected_file)
                        try:
                            text_content = load_text_from_file(file_path)
                            text_name = os.path.splitext(selected_file)[0]
                            st.success(f"✅ Loaded: {selected_file}")
                        except Exception as e:
                            st.error(f"❌ Error loading file: {e}")
                else:
                    st.warning("No .txt files found in data folder")
            else:
                st.warning("Data folder not found")
        
        # Preview text
        if text_content:
            with st.expander("📖 Text Preview"):
                st.text(text_content[:500] + ("..." if len(text_content) > 500 else ""))
    
    with col2:
        st.header("🚀 Actions")
        
        # Process button
        if st.button("🔄 Run Pipeline", type="primary", disabled=not text_content):
            if text_content and text_name:
                run_pipeline(text_content, text_name, window_size, vector_size, 
                           walk_length, num_walks, epochs, min_frequency, viz_method)
        
        # Results info
        if 'pipeline_results' in st.session_state:
            results = st.session_state.pipeline_results
            
            st.subheader("📊 Results")
            st.metric("Graph Nodes", results['stats']['graph_nodes'])
            st.metric("Graph Edges", results['stats']['graph_edges'])
            st.metric("Total Walks", results['stats']['total_walks'])
            st.metric("Vocabulary Size", results['stats']['vocab_size'])
            
            # Download buttons
            if 'node_file' in results and 'graph_file' in results:
                st.subheader("💾 Download Files")
                
                # Node embeddings download
                with open(results['node_file'], 'rb') as f:
                    st.download_button(
                        "📄 Download Node Embeddings",
                        f.read(),
                        file_name=os.path.basename(results['node_file']),
                        mime="text/csv"
                    )
                
                # Graph embedding download
                with open(results['graph_file'], 'rb') as f:
                    st.download_button(
                        "🔗 Download Graph Embedding",
                        f.read(),
                        file_name=os.path.basename(results['graph_file']),
                        mime="text/csv"
                    )

def run_pipeline(text, text_name, window_size, vector_size, walk_length, 
                num_walks, epochs, min_frequency, viz_method):
    """Run the graph embedding pipeline"""
    
    with st.spinner("🔄 Processing... This may take a few moments"):
        try:
            # Initialize pipeline
            pipeline = GraphEmbeddingPipeline(
                window_size=window_size,
                weight_method='frequency',
                walk_length=walk_length,
                num_walks=num_walks,
                vector_size=vector_size,
                epochs=epochs
            )
            
            # Process text
            node_embeddings, graph_embedding = pipeline.process_text_to_embeddings(
                text, min_frequency=min_frequency
            )
            
            # Save embeddings
            node_file, graph_file = pipeline.save_embeddings(text_name)
            
            # Get stats
            stats = pipeline.get_stats()
            
            # Store results in session state
            st.session_state.pipeline_results = {
                'stats': stats,
                'node_embeddings': node_embeddings,
                'graph_embedding': graph_embedding,
                'node_file': node_file,
                'graph_file': graph_file,
                'pipeline': pipeline
            }
            
            st.success("✅ Pipeline completed successfully!")
            
            # Display results
            display_results(viz_method)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

def display_results(viz_method):
    """Display pipeline results"""
    
    if 'pipeline_results' not in st.session_state:
        return
    
    results = st.session_state.pipeline_results
    node_embeddings = results['node_embeddings']
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "🎯 Node Embeddings", "📈 Visualization", "📋 Data Tables"])
    
    with tab1:
        st.subheader("📊 Pipeline Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Graph Nodes", results['stats']['graph_nodes'])
        with col2:
            st.metric("Graph Edges", results['stats']['graph_edges'])
        with col3:
            st.metric("Total Walks", results['stats']['total_walks'])
        with col4:
            st.metric("Embedding Dimension", results['stats']['embedding_dim'])
        
        # Graph info
        pipeline = results['pipeline']
        if pipeline.graph:
            st.subheader("🔗 Graph Properties")
            
            # Node degrees
            degrees = dict(pipeline.graph.degree())
            avg_degree = np.mean(list(degrees.values()))
            max_degree = max(degrees.values())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Degree", f"{avg_degree:.2f}")
            with col2:
                st.metric("Max Degree", max_degree)
            
            # Most connected nodes
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            st.write("**Top 5 Most Connected Nodes:**")
            for node, degree in top_nodes:
                st.write(f"- {node}: {degree} connections")
    
    with tab2:
        st.subheader("🎯 Node Embeddings Sample")
        
        # Sample embeddings
        sample_nodes = list(node_embeddings.keys())[:10]
        
        for node in sample_nodes:
            embedding = node_embeddings[node]
            with st.expander(f"Node: {node}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**Dimension:** {len(embedding)}")
                    st.write(f"**L2 Norm:** {np.linalg.norm(embedding):.4f}")
                with col2:
                    # Show first 10 values
                    st.write("**First 10 values:**")
                    st.write([f"{val:.4f}" for val in embedding[:10]])
    
    with tab3:
        st.subheader("📈 Embedding Visualization")
        
        if len(node_embeddings) > 1:
            # Prepare data for visualization
            nodes = list(node_embeddings.keys())
            embeddings_matrix = np.array([node_embeddings[node] for node in nodes])
            
            # Dimensionality reduction
            if viz_method == "PCA":
                reducer = PCA(n_components=2)
                reduced_embeddings = reducer.fit_transform(embeddings_matrix)
                explained_var = reducer.explained_variance_ratio_.sum()
                st.write(f"**PCA Explained Variance:** {explained_var:.2%}")
            else:  # t-SNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(nodes)-1))
                reduced_embeddings = reducer.fit_transform(embeddings_matrix)
            
            # Create scatter plot
            fig = go.Figure(data=go.Scatter(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                mode='markers+text',
                text=nodes,
                textposition="top center",
                marker=dict(
                    size=10,
                    color=np.arange(len(nodes)),
                    colorscale='Viridis',
                    showscale=True
                ),
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Node Embeddings Visualization ({viz_method})",
                xaxis_title=f"{viz_method} Component 1",
                yaxis_title=f"{viz_method} Component 2",
                height=600
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Need at least 2 nodes for visualization")
    
    with tab4:
        st.subheader("📋 Data Tables")
        
        # Node embeddings table
        st.write("**Node Embeddings (first 5 dimensions):**")
        
        table_data = []
        for node, embedding in list(node_embeddings.items())[:20]:  # First 20 nodes
            row = {'Node': node}
            for i in range(min(5, len(embedding))):
                row[f'Dim_{i+1}'] = f"{embedding[i]:.4f}"
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, width='stretch')
        
        # Graph embedding
        st.write("**Graph Embedding (first 10 dimensions):**")
        graph_emb = results['graph_embedding']
        graph_data = {f'Dim_{i+1}': f"{graph_emb[i]:.4f}" for i in range(min(10, len(graph_emb)))}
        st.json(graph_data)

if __name__ == "__main__":
    main()