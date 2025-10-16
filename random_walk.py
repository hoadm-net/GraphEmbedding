import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import numpy as np
import random
from collections import defaultdict
import tempfile
from text2graph import TextGraphBuilder, load_data_files

# Cấu hình trang
st.set_page_config(
    page_title="Random Walk Visualization",
    page_icon="🚶",
    layout="wide"
)

class RandomWalker:
    def __init__(self, graph, walk_length=10, num_walks=10):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.walks = []
        
    def single_walk(self, start_node):
        """Thực hiện một random walk từ node bắt đầu"""
        if start_node not in self.graph.nodes():
            return []
            
        walk = [start_node]
        current_node = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            if not neighbors:  # Nếu không có neighbors
                break
            # Chọn ngẫu nhiên một neighbor
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
            
        return walk
    
    def generate_walks(self, start_nodes=None):
        """Tạo multiple random walks"""
        self.walks = []
        
        if start_nodes is None:
            start_nodes = list(self.graph.nodes())
        
        for start_node in start_nodes:
            for _ in range(self.num_walks):
                walk = self.single_walk(start_node)
                if len(walk) > 1:  # Chỉ giữ walks có ít nhất 2 nodes
                    self.walks.append(walk)
        
        return self.walks
    
    def get_walk_statistics(self):
        """Thống kê về các walks"""
        if not self.walks:
            return {}
        
        walk_lengths = [len(walk) for walk in self.walks]
        node_frequencies = defaultdict(int)
        
        for walk in self.walks:
            for node in walk:
                node_frequencies[node] += 1
        
        return {
            'total_walks': len(self.walks),
            'avg_walk_length': np.mean(walk_lengths),
            'min_walk_length': min(walk_lengths),
            'max_walk_length': max(walk_lengths),
            'unique_nodes_visited': len(node_frequencies),
            'most_visited_nodes': dict(sorted(node_frequencies.items(), 
                                            key=lambda x: x[1], reverse=True)[:10])
        }



def visualize_graph_with_walks(graph, selected_nodes=None, walk_paths=None):
    """Trực quan hóa graph với highlighting cho selected nodes và walk paths"""
    if len(graph.nodes()) == 0:
        st.warning("Graph không có nodes.")
        return None
    
    net = Network(height="600px", width="100%", notebook=False, directed=False)
    net.barnes_hut()
    
    # Tạo sets để dễ check
    selected_set = set(selected_nodes) if selected_nodes else set()
    walk_nodes = set()
    if walk_paths:
        for walk in walk_paths:
            walk_nodes.update(walk)
    
    # Thêm nodes với màu sắc theo trạng thái
    for node in graph.nodes():
        freq = graph.nodes[node].get('frequency', 1)
        
        # Xác định màu node
        if node in selected_set:
            color = '#ff4444'  # Đỏ cho selected nodes
        elif node in walk_nodes:
            color = '#44ff44'  # Xanh lá cho nodes trong walk
        else:
            color = '#4444ff'  # Xanh dương cho nodes thông thường
        
        # Hiển thị label với khoảng trắng thay vì underscore
        display_label = node.replace('_', ' ')
        
        tooltip = f"""Từ: {display_label}
Tần suất: {freq}
Bậc: {graph.degree[node]}
Trạng thái: {"Selected" if node in selected_set else "In Walk" if node in walk_nodes else "Normal"}"""
        
        net.add_node(
            node,
            label=display_label,
            title=tooltip,
            size=25 if node in selected_set else 20,
            color=color
        )
    
    # Thêm edges
    for edge in graph.edges():
        w = graph[edge[0]][edge[1]].get('weight', 1)
        
        # Highlight edges nếu cả 2 nodes đều được chọn hoặc trong walk
        if (edge[0] in selected_set and edge[1] in selected_set):
            edge_color = '#ff4444'  # Đỏ cho edges giữa selected nodes
            width = 3
        elif (edge[0] in walk_nodes and edge[1] in walk_nodes):
            edge_color = '#44ff44'  # Xanh cho edges trong walk
            width = 2
        else:
            edge_color = '#808080'  # Xám cho edges thông thường
            width = 1
            
        net.add_edge(edge[0], edge[1], width=width, color=edge_color, 
                    title=f"Trọng số: {w:.2f}")
    
    # Cấu hình visualization
    net.set_options('''
    var options = {
        "nodes": {
            "font": {"size": 16},
            "borderWidth": 2
        },
        "edges": {
            "smooth": false
        },
        "physics": {
            "enabled": true,
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 146,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 150,
                "updateInterval": 25,
                "fit": true
            }
        }
    }
    ''')
    
    # Lưu và hiển thị
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        net.save_graph(f.name)
        html_content = open(f.name, 'r', encoding='utf-8').read()
    
    st.components.v1.html(html_content, height=650, scrolling=True)
    return None

def main():
    st.title("🚶 Random Walk on Text Graphs")
    st.markdown("**Khám phá đồ thị từ vựng bằng Random Walk và tạo Node Vectors**")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Cấu hình")
    
    # Load graph từ text data
    st.sidebar.subheader("1. Tạo Graph")
    files, texts = load_data_files()
    
    if not files:
        st.error("Không tìm thấy file dữ liệu trong thư mục 'data'")
        return
    
    # Chọn files
    selected_files = st.sidebar.multiselect(
        "Chọn files:",
        files,
        default=files[:2] if len(files) >= 2 else files
    )
    
    if not selected_files:
        st.warning("Vui lòng chọn ít nhất một file.")
        return
    
    # Tham số graph
    window_size = st.sidebar.slider("Window size:", 1, 10, 3)
    min_frequency = st.sidebar.slider("Min frequency:", 1, 10, 2)
    weight_method = st.sidebar.selectbox("Weight method:", ["frequency", "pmi"])
    
    # Tạo graph
    with st.spinner("Tạo graph..."):
        graph_builder = TextGraphBuilder(window_size=window_size, weight_method=weight_method)
        all_text = " ".join([texts[f] for f in selected_files])
        tokens = graph_builder.process_text(all_text)
        graph_builder.build_cooccurrence_matrix(tokens)
        G = graph_builder.build_graph(min_frequency=min_frequency)
    
    if len(G.nodes()) == 0:
        st.error("Graph không có nodes. Thử giảm min_frequency.")
        return
    
    # Random Walk parameters
    st.sidebar.subheader("2. Random Walk")
    walk_length = st.sidebar.slider("Walk length:", 3, 20, 10)
    num_walks = st.sidebar.slider("Walks per node:", 1, 10, 3)
    
    # Node selection
    st.sidebar.subheader("3. Chọn Nodes")
    all_nodes = sorted(list(G.nodes()))
    selected_nodes = st.sidebar.multiselect(
        "Chọn nodes để highlight:",
        all_nodes,
        default=all_nodes[:3] if len(all_nodes) >= 3 else all_nodes[:1]
    )
    
    # Thực hiện Random Walk
    walker = RandomWalker(G, walk_length=walk_length, num_walks=num_walks)
    
    # Chọn loại walk
    walk_mode = st.sidebar.radio(
        "Random walk từ:",
        ["Single walk demo", "Selected nodes only"]
    )
    
    if walk_mode == "Single walk demo":
        if selected_nodes:
            single_walk = walker.single_walk(selected_nodes[0])
            walks = [single_walk] if single_walk else []
        else:
            walks = []
    else:  # Selected nodes only
        walks = walker.generate_walks(start_nodes=selected_nodes)
    
    # Statistics
    stats = walker.get_walk_statistics()
    
    # Display
    st.subheader("📊 Graph Visualization")
    
    # Legend
    st.markdown("""
    **Legend:**
    - 🔴 **Selected nodes** (đã chọn)
    - 🟢 **Walk nodes** (trong random walks)  
    - 🔵 **Other nodes** (còn lại)
    """)
    
    visualize_graph_with_walks(G, selected_nodes, walks)
        
    
    # Walk details
    if walks:
        st.subheader("🚶 Walk Details")
        
        # Hiển thị một số walks mẫu
        num_display = min(5, len(walks))
        
        for i in range(num_display):
            walk = walks[i]
            walk_display = " → ".join([node.replace('_', ' ') for node in walk])
            st.write(f"**Walk {i+1}:** {walk_display}")
        
        if len(walks) > num_display:
            st.write(f"... và {len(walks) - num_display} walks khác")
    


if __name__ == "__main__":
    main()