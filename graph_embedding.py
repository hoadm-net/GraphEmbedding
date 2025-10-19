#!/usr/bin/env python3
"""
Graph Embedding Pipeline: Text → Graph → Node Embeddings → Graph Embeddings

Combines text-to-graph conversion with DeepWalk to create both node and graph embeddings.
"""

import pandas as pd
import networkx as nx
from underthesea import word_tokenize
import os
import numpy as np
from collections import Counter, defaultdict
import math
import random
import datetime
from gensim.models import Word2Vec


class TextGraphBuilder:
    """Build co-occurrence graph from Vietnamese text"""
    
    def __init__(self, window_size=3, weight_method='frequency'):
        self.window_size = window_size
        self.weight_method = weight_method
        self.vocab = set()
        self.cooccurrence = defaultdict(int)
        self.word_counts = Counter()
        self.total_words = 0
        
    def process_text(self, text):
        """Tokenize và xử lý văn bản tiếng Việt"""
        tokens = word_tokenize(text.lower())
        
        # Lọc bỏ dấu câu và từ quá ngắn
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
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
                    if word1 != word2:
                        pair = tuple(sorted([word1, word2]))
                        self.cooccurrence[pair] += 1
    
    def calculate_pmi(self, word1, word2, cooc_count):
        """Tính Pointwise Mutual Information (PMI)"""
        if self.total_words == 0:
            return 0
            
        p_joint = cooc_count / self.total_words
        p_word1 = self.word_counts[word1] / self.total_words
        p_word2 = self.word_counts[word2] / self.total_words
        
        if p_word1 == 0 or p_word2 == 0 or p_joint == 0:
            return 0
            
        pmi = math.log2(p_joint / (p_word1 * p_word2))
        return max(0, pmi)
    
    def build_graph(self, min_frequency=1):
        """Xây dựng đồ thị từ ma trận đồng xuất hiện"""
        G = nx.Graph()
        
        # Lọc từ theo tần suất
        frequent_words = [word for word, count in self.word_counts.most_common() 
                         if count >= min_frequency]
        
        # Thêm nút
        for word in frequent_words:
            G.add_node(word, frequency=self.word_counts[word])
        
        # Thêm cạnh
        for (word1, word2), cooc_count in self.cooccurrence.items():
            if word1 in frequent_words and word2 in frequent_words and cooc_count >= min_frequency:
                if self.weight_method == 'frequency':
                    weight = cooc_count
                else:  # PMI
                    weight = self.calculate_pmi(word1, word2, cooc_count)
                
                if weight > 0:
                    G.add_edge(word1, word2, weight=weight)
        
        return G


class RandomWalker:
    """Generate random walks from graph"""
    
    def __init__(self, graph, walk_length=10, num_walks=5, random_seed=42):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def single_walk(self, start_node):
        """Thực hiện một random walk từ node bắt đầu"""
        walk = [start_node]
        current_node = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
            
        return walk
    
    def generate_walks(self):
        """Tạo tất cả walks từ tất cả nodes"""
        walks = []
        nodes = list(self.graph.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.single_walk(node)
                walks.append(walk)
        
        return walks


class DeepWalkEmbedding:
    """Train embeddings using Skip-gram on random walks"""
    
    def __init__(self, vector_size=64, window=5, min_count=1, epochs=20, 
                 workers=4, sg=1, negative=5, random_seed=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.sg = sg
        self.negative = negative
        self.random_seed = random_seed
        self.model = None
    
    def train(self, walks):
        """Huấn luyện mô hình Word2Vec trên walks"""
        print(f"🚀 Training Word2Vec model...")
        print(f"📊 Parameters: vector_size={self.vector_size}, window={self.window}, epochs={self.epochs}")
        
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
            negative=self.negative,
            seed=self.random_seed
        )
        
        print(f"✅ Training completed! Vocabulary size: {len(self.model.wv.key_to_index)}")
        return self.model
    
    def get_node_embeddings(self):
        """Lấy embeddings cho tất cả nodes"""
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện. Gọi train() trước.")
        
        node_embeddings = {}
        for word in self.model.wv.key_to_index:
            node_embeddings[word] = self.model.wv[word]
        
        return node_embeddings
    
    def get_graph_embedding(self, method='mean'):
        """Tạo graph embedding bằng cách aggregate node embeddings"""
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện. Gọi train() trước.")
        
        all_vectors = [self.model.wv[word] for word in self.model.wv.key_to_index]
        
        if method == 'mean':
            graph_embedding = np.mean(all_vectors, axis=0)
        elif method == 'sum':
            graph_embedding = np.sum(all_vectors, axis=0)
        elif method == 'max':
            graph_embedding = np.max(all_vectors, axis=0)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return graph_embedding


class GraphEmbeddingPipeline:
    """Complete pipeline from text to embeddings"""
    
    def __init__(self, window_size=3, weight_method='frequency', walk_length=10, 
                 num_walks=5, vector_size=64, epochs=20, random_seed=42):
        self.window_size = window_size
        self.weight_method = weight_method
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.vector_size = vector_size
        self.epochs = epochs
        self.random_seed = random_seed
        
        # Components
        self.graph_builder = None
        self.walker = None
        self.embedding_model = None
        
        # Results
        self.graph = None
        self.walks = None
        self.node_embeddings = None
        self.graph_embedding = None
    
    def process_text_to_embeddings(self, text, min_frequency=1):
        """Complete pipeline: text → graph → walks → embeddings"""
        print("🔄 Starting Graph Embedding Pipeline...")
        
        # 1. Text to Graph
        print("📝 Step 1: Converting text to graph...")
        self.graph_builder = TextGraphBuilder(
            window_size=self.window_size, 
            weight_method=self.weight_method
        )
        
        tokens = self.graph_builder.process_text(text)
        self.graph_builder.build_cooccurrence_matrix(tokens)
        self.graph = self.graph_builder.build_graph(min_frequency=min_frequency)
        
        print(f"✅ Graph created: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
        
        if len(self.graph.nodes()) == 0:
            raise ValueError("Đồ thị rỗng. Hãy giảm min_frequency hoặc kiểm tra text input.")
        
        # 2. Random Walks
        print("🚶 Step 2: Generating random walks...")
        self.walker = RandomWalker(
            self.graph, 
            walk_length=self.walk_length, 
            num_walks=self.num_walks,
            random_seed=self.random_seed
        )
        
        self.walks = self.walker.generate_walks()
        print(f"✅ Generated {len(self.walks)} walks")
        
        # 3. Train Embeddings
        print("🧠 Step 3: Training embeddings...")
        self.embedding_model = DeepWalkEmbedding(
            vector_size=self.vector_size,
            epochs=self.epochs,
            random_seed=self.random_seed
        )
        
        self.embedding_model.train(self.walks)
        
        # 4. Extract Results
        print("📊 Step 4: Extracting embeddings...")
        self.node_embeddings = self.embedding_model.get_node_embeddings()
        self.graph_embedding = self.embedding_model.get_graph_embedding()
        
        print("✅ Pipeline completed successfully!")
        return self.node_embeddings, self.graph_embedding
    
    def save_embeddings(self, text_name, output_dir="output"):
        """Lưu embeddings vào CSV files"""
        if self.node_embeddings is None or self.graph_embedding is None:
            raise ValueError("Chưa có embeddings. Chạy process_text_to_embeddings() trước.")
        
        # Tạo thư mục output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Tạo timestamp ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save node embeddings: Node,0,1,2,3,...
        node_file = os.path.join(output_dir, f"{text_name}-node_embedding.csv")
        
        # Tạo DataFrame cho node embeddings
        node_data = []
        for node, embedding in self.node_embeddings.items():
            row = {'node': node}  # Lowercase 'node'
            for i, val in enumerate(embedding):
                row[str(i)] = val  # Sử dụng index 0,1,2,... làm column name
            node_data.append(row)
        
        node_df = pd.DataFrame(node_data)
        node_df.to_csv(node_file, index=False, encoding='utf-8')
        
        # 2. Save graph embedding: 0,1,2,3,...
        graph_file = os.path.join(output_dir, f"{text_name}-graph_embedding.csv")
        
        # Tạo DataFrame cho graph embedding (1 dòng duy nhất)
        graph_data = {}
        for i, val in enumerate(self.graph_embedding):
            graph_data[str(i)] = [val]  # List với 1 phần tử, sử dụng index làm column name
        
        graph_df = pd.DataFrame(graph_data)
        graph_df.to_csv(graph_file, index=False, encoding='utf-8')
        
        print(f"💾 Embeddings saved:")
        print(f"📄 Node embeddings: {node_file} ({len(node_df)} nodes, {len(node_df.columns)-1} dimensions)")
        print(f"🔗 Graph embedding: {graph_file} (1 graph, {len(graph_df.columns)} dimensions)")
        
        return node_file, graph_file
    
    def get_stats(self):
        """Lấy thống kê về pipeline"""
        stats = {
            'graph_nodes': len(self.graph.nodes()) if self.graph else 0,
            'graph_edges': len(self.graph.edges()) if self.graph else 0,
            'total_walks': len(self.walks) if self.walks else 0,
            'embedding_dim': self.vector_size,
            'vocab_size': len(self.node_embeddings) if self.node_embeddings else 0
        }
        return stats


def load_text_from_file(filepath):
    """Load text from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def interactive_pipeline():
    """Interactive pipeline với user input"""
    print("🎯 Interactive Graph Embedding Pipeline")
    print("=" * 50)
    
    # User can choose input method
    print("\n📝 Choose input method:")
    print("1. Enter text manually")
    print("2. Load from file")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n📝 Enter your Vietnamese text:")
        text = input("Text: ")
    elif choice == "2":
        filepath = input("Enter file path: ").strip()
        try:
            text = load_text_from_file(filepath)
            print(f"✅ Loaded text from {filepath}")
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return
    else:
        print("❌ Invalid choice")
        return
    
    # Get text name for output files
    text_name = input("\n📁 Enter name for output files (e.g., 'my_text'): ").strip()
    if not text_name:
        text_name = "untitled"
    
    # Pipeline parameters
    print("\n⚙️ Configure parameters (press Enter for defaults):")
    
    try:
        window_size = input("Window size (default: 3): ").strip()
        window_size = int(window_size) if window_size else 3
        
        vector_size = input("Vector size (default: 64): ").strip()
        vector_size = int(vector_size) if vector_size else 64
        
        walk_length = input("Walk length (default: 10): ").strip()
        walk_length = int(walk_length) if walk_length else 10
        
        num_walks = input("Number of walks per node (default: 5): ").strip()
        num_walks = int(num_walks) if num_walks else 5
        
        epochs = input("Training epochs (default: 20): ").strip()
        epochs = int(epochs) if epochs else 20
        
        min_freq = input("Minimum word frequency (default: 1): ").strip()
        min_freq = int(min_freq) if min_freq else 1
        
    except ValueError as e:
        print(f"❌ Invalid parameter: {e}")
        return
    
    # Initialize pipeline
    pipeline = GraphEmbeddingPipeline(
        window_size=window_size,
        weight_method='frequency', 
        walk_length=walk_length,
        num_walks=num_walks,
        vector_size=vector_size,
        epochs=epochs
    )
    
    try:
        print("\n🚀 Starting pipeline...")
        
        # Process text to embeddings
        node_embeddings, graph_embedding = pipeline.process_text_to_embeddings(
            text, 
            min_frequency=min_freq
        )
        
        # Show stats
        stats = pipeline.get_stats()
        print(f"\n📊 Results:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save embeddings
        node_file, graph_file = pipeline.save_embeddings(text_name)
        
        print(f"\n✅ Pipeline completed! Files saved:")
        print(f"  📄 {node_file}")
        print(f"  🔗 {graph_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Demo function"""
    print("🎯 Graph Embedding Pipeline Demo")
    print("=" * 50)
    
    # Sample text
    sample_text = """
    Học sinh đi học tại trường đại học. Sinh viên học bài tại nhà.
    Giáo viên dạy học sinh. Học sinh làm bài tập về nhà.
    Trường đại học có nhiều sinh viên. Giáo viên dạy bài học mới.
    Học bài rất quan trọng đối với học sinh.
    """
    
    # Initialize pipeline
    pipeline = GraphEmbeddingPipeline(
        window_size=3,
        weight_method='frequency', 
        walk_length=10,
        num_walks=5,
        vector_size=64,
        epochs=20
    )
    
    try:
        # Process text to embeddings
        node_embeddings, graph_embedding = pipeline.process_text_to_embeddings(
            sample_text, 
            min_frequency=1
        )
        
        # Show stats
        stats = pipeline.get_stats()
        print(f"\n📊 Pipeline Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show sample embeddings
        print(f"\n📄 Sample Node Embeddings:")
        for i, (node, embedding) in enumerate(list(node_embeddings.items())[:3]):
            print(f"  {node}: [{embedding[:3]}...] (dim: {len(embedding)})")
        
        print(f"\n🔗 Graph Embedding:")
        print(f"  Dimension: {len(graph_embedding)}")
        print(f"  Sample values: [{graph_embedding[:5]}...]")
        
        # Save embeddings
        node_file, graph_file = pipeline.save_embeddings("demo_text")
        
        print(f"\n✅ Demo completed successfully!")
        
        # Ask if user wants to try interactive mode
        print(f"\n🔄 Want to try with your own text? Run with --interactive flag")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_pipeline()
    else:
        main()