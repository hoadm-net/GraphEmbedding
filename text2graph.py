import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from underthesea import word_tokenize
import os
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math

# Cấu hình trang
st.set_page_config(
    page_title="Text to Graph Visualization",
    page_icon="🕸️",
    layout="wide"
)

class TextGraphBuilder:
    def __init__(self, window_size=3, weight_method='frequency'):
        self.window_size = window_size
        self.weight_method = weight_method
        self.vocab = set()
        self.cooccurrence = defaultdict(int)
        self.word_counts = Counter()
        self.total_words = 0
        
    def process_text(self, text):
        """Tokenize và xử lý văn bản tiếng Việt"""
        # Tokenize bằng underthesea - giữ nguyên từ ghép với dấu gạch dưới
        tokens = word_tokenize(text.lower())
        
        # Lọc bỏ dấu câu và từ quá ngắn, giữ nguyên dấu gạch dưới cho từ ghép
        filtered_tokens = []
        for token in tokens:
            # Loại bỏ khoảng trắng đầu cuối
            token = token.strip()
            # Chỉ giữ từ có độ dài > 1 (bỏ isalnum để giữ lại từ ghép có dấu '_')
            if len(token) > 1:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def build_cooccurrence_matrix(self, tokens):
        """Xây dựng ma trận đồng xuất hiện"""
        self.vocab.update(tokens)
        self.word_counts.update(tokens)
        self.total_words += len(tokens)
        
        # Sliding window để tính đồng xuất hiện
        for i in range(len(tokens)):
            for j in range(max(0, i - self.window_size), 
                          min(len(tokens), i + self.window_size + 1)):
                if i != j:
                    word1, word2 = tokens[i], tokens[j]
                    # Sắp xếp để đảm bảo tính nhất quán
                    if word1 != word2:
                        pair = tuple(sorted([word1, word2]))
                        self.cooccurrence[pair] += 1
    
    def calculate_pmi(self, word1, word2, cooc_count):
        """Tính Pointwise Mutual Information (PMI)"""
        if self.total_words == 0:
            return 0
            
        # P(word1, word2)
        p_joint = cooc_count / self.total_words
        
        # P(word1) và P(word2)
        p_word1 = self.word_counts[word1] / self.total_words
        p_word2 = self.word_counts[word2] / self.total_words
        
        if p_word1 == 0 or p_word2 == 0 or p_joint == 0:
            return 0
            
        # PMI = log(P(word1, word2) / (P(word1) * P(word2)))
        pmi = math.log2(p_joint / (p_word1 * p_word2))
        return max(0, pmi)  # Positive PMI
    
    def build_graph(self, min_frequency=1):
        """Xây dựng đồ thị từ ma trận đồng xuất hiện"""
        G = nx.Graph()
        
        # Lọc từ theo tần suất (bỏ giới hạn số lượng nút)
        frequent_words = [word for word, count in self.word_counts.most_common() 
                         if count >= min_frequency]
        
        # Thêm nút
        for word in frequent_words:
            G.add_node(word, frequency=self.word_counts[word])
        
        # Thêm cạnh
        edges_added = 0
        for (word1, word2), cooc_count in self.cooccurrence.items():
            if word1 in frequent_words and word2 in frequent_words and cooc_count >= min_frequency:
                if self.weight_method == 'frequency':
                    weight = cooc_count
                else:  # PMI
                    weight = self.calculate_pmi(word1, word2, cooc_count)
                
                if weight > 0:
                    G.add_edge(word1, word2, weight=weight)
                    edges_added += 1
        
        return G

def load_data_files():
    """Load tất cả các file txt từ thư mục data"""
    data_folder = "data"
    files = []
    texts = {}
    
    if os.path.exists(data_folder):
        for filename in os.listdir(data_folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_folder, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        texts[filename] = content
                        files.append(filename)
                except Exception as e:
                    st.error(f"Không thể đọc file {filename}: {e}")
    
    return files, texts

def visualize_graph(G):
    """Trực quan hóa đồ thị bằng Pyvis và nhúng vào Streamlit với layout Spring"""
    if len(G.nodes()) == 0:
        st.warning("Đồ thị không có nút nào. Hãy thử giảm ngưỡng tần suất tối thiểu.")
        return None
    
    net = Network(height="600px", width="100%", notebook=False, directed=False)
    net.barnes_hut()
    
    # Thêm nút với kích thước bằng nhau
    for node in G.nodes():
        freq = G.nodes[node].get('frequency', 1)
        # Lấy danh sách các node kề cận
        neighbors = list(G.neighbors(node))
        # Hiển thị từ kề cận với khoảng trắng thay vì underscore
        neighbors_display = [n.replace('_', ' ') for n in neighbors[:10]]
        neighbors_str = ', '.join(neighbors_display)
        if len(neighbors) > 10:
            neighbors_str += f' ... (và {len(neighbors) - 10} từ khác)'
        
        # Thay thế _ bằng khoảng trắng chỉ khi hiển thị
        display_label = node.replace('_', ' ')
        
        # Tạo tooltip với HTML đúng format
        tooltip = f"""Từ: {display_label}
Tần suất: {freq}
Bậc: {G.degree[node]}
Các từ liên kết: {neighbors_str}"""
        
        net.add_node(
            node,
            label=display_label,  # Hiển thị với khoảng trắng
            title=tooltip,
            size=20,  # Kích thước cố định cho tất cả các nút
        )
    
    # Thêm cạnh với độ dày cố định và màu xám
    for edge in G.edges():
        w = G[edge[0]][edge[1]].get('weight', 1)
        net.add_edge(edge[0], edge[1], width=1, color='#808080', title=f"Trọng số: {w:.2f}")
    
    # Tùy chỉnh options: sử dụng Spring layout và tắt hiệu ứng xoay vòng
    net.set_options('''
    var options = {
        "nodes": {
            "font": {"size": 18},
            "size": 20,
            "borderWidth": 2
        },
        "edges": {
            "color": {"color": "#808080"},
            "width": 1,
            "smooth": false
        },
        "layout": {
            "improvedLayout": false
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
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        }
    }
    ''')
    
    # Lưu HTML tạm thời và nhúng vào Streamlit
    import tempfile
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        net.save_graph(f.name)
        html_content = open(f.name, 'r', encoding='utf-8').read()
    
    # Sử dụng width để tránh lỗi với st.components.v1.html
    st.components.v1.html(html_content, height=650, scrolling=True)
    return None

def main():
    st.title("🕸️ Text to Graph Visualization")
    st.markdown("**Trực quan hóa quá trình xây dựng đồ thị từ văn bản tiếng Việt**")
    
    # Sidebar để chỉnh tham số
    st.sidebar.header("⚙️ Tham số")
    
    # Load dữ liệu
    files, texts = load_data_files()
    
    if not files:
        st.error("Không tìm thấy file dữ liệu trong thư mục 'data'")
        return
    
    # Chọn file
    selected_files = st.sidebar.multiselect(
        "Chọn file để phân tích:",
        files,
        default=files[:3] if len(files) >= 3 else files
    )
    
    # Tham số sliding window
    window_size = st.sidebar.slider(
        "Kích thước cửa sổ ngữ cảnh (k):",
        min_value=1,
        max_value=10,
        value=3,
        help="Số từ xung quanh để tính đồng xuất hiện"
    )
    
    # Phương pháp tính trọng số
    weight_method = st.sidebar.selectbox(
        "Phương pháp tính trọng số cạnh:",
        ["frequency", "pmi"],
        format_func=lambda x: "Tần suất" if x == "frequency" else "PMI (Pointwise Mutual Information)"
    )
    
    # Tham số lọc
    min_frequency = st.sidebar.slider(
        "Tần suất tối thiểu:",
        min_value=1,
        max_value=10,
        value=2,
        help="Từ phải xuất hiện ít nhất bao nhiêu lần"
    )
    
    if not selected_files:
        st.warning("Vui lòng chọn ít nhất một file để phân tích.")
        return
    
    # Xử lý dữ liệu
    with st.spinner("Đang xử lý văn bản..."):
        # Khởi tạo TextGraphBuilder
        graph_builder = TextGraphBuilder(window_size=window_size, weight_method=weight_method)
        # Xử lý từng file được chọn
        all_text = ""
        for filename in selected_files:
            all_text += texts[filename] + " "
        # Tokenize toàn bộ văn bản
        tokens = graph_builder.process_text(all_text)
        # Xây dựng ma trận đồng xuất hiện
        graph_builder.build_cooccurrence_matrix(tokens)
        # Xây dựng đồ thị (không giới hạn số nút)
        G = graph_builder.build_graph(min_frequency=min_frequency)
    
    # Hiển thị thống kê
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Số file đã chọn", len(selected_files))
    
    with col2:
        st.metric("Tổng số từ", len(tokens), help="Tổng số từ trong văn bản (kể cả từ lặp lại)")
    
    with col3:
        st.metric("Từ vựng duy nhất", len(graph_builder.vocab), help="Số lượng từ khác nhau (không đếm trùng lặp)")
    
    with col4:
        st.metric("Cặp từ đồng xuất hiện", len(graph_builder.cooccurrence), help="Số cặp từ xuất hiện gần nhau trong ngữ cảnh")
    
    # Hiển thị đồ thị
    st.header("📊 Đồ thị từ vựng")
    if len(G.nodes()) > 0:
        visualize_graph(G)
        # Hiển thị thông tin đồ thị
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Thông tin đồ thị")
            st.write(f"**Số nút:** {len(G.nodes())}")
            st.write(f"**Số cạnh:** {len(G.edges())}")
            st.write(f"**Mật độ:** {nx.density(G):.4f}")
            if len(G.nodes()) > 0:
                st.write(f"**Bậc trung bình:** {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")
        with col2:
            st.subheader("🔥 Top từ có tần suất cao")
            top_words = graph_builder.word_counts.most_common(10)
            df_top = pd.DataFrame(top_words, columns=['Từ', 'Tần suất'])
            st.dataframe(df_top, width='stretch')
    else:
        st.warning("Không thể tạo đồ thị với các tham số hiện tại. Hãy thử giảm ngưỡng tần suất tối thiểu.")
    
    # Hiển thị một số cặp từ đồng xuất hiện
    if graph_builder.cooccurrence:
        st.subheader("🔗 Top cặp từ đồng xuất hiện")
        top_cooc = sorted(graph_builder.cooccurrence.items(), key=lambda x: x[1], reverse=True)[:15]
        
        cooc_data = []
        for (word1, word2), count in top_cooc:
            if weight_method == 'pmi':
                pmi_score = graph_builder.calculate_pmi(word1, word2, count)
                cooc_data.append([f"{word1} - {word2}", count, f"{pmi_score:.3f}"])
            else:
                cooc_data.append([f"{word1} - {word2}", count, count])
        
        columns = ['Cặp từ', 'Tần suất', 'PMI' if weight_method == 'pmi' else 'Trọng số']
        df_cooc = pd.DataFrame(cooc_data, columns=columns)
        st.dataframe(df_cooc, width='stretch')
    
    # Hiển thị nội dung file được chọn
    with st.expander("📄 Xem nội dung file đã chọn"):
        for filename in selected_files:
            st.subheader(f"File: {filename}")
            st.text_area("Nội dung:", texts[filename], height=150, key=filename)

if __name__ == "__main__":
    main()
