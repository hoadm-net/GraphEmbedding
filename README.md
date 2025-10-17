# 🕸️ Graph Embedding for Text Analysis

Project trực quan hóa và phân tích đồ thị từ văn bản tiếng Việt sử dụng các phương pháp Graph Embedding cơ bản.

## 📋 Tổng quan

Project này thực hiện pipeline hoàn chỉnh từ văn bản đến phân loại đồ thị:

```
Văn bản → Đồ thị từ vựng → Graph Embedding → Phân loại/Phân tích
   (1)          (2)              (3)                (4)
```

1. **Text to Graph**: Chuyển đổi văn bản thành đồ thị đồng xuất hiện (co-occurrence graph)
2. **Visualization**: Trực quan hóa đồ thị tương tác với Pyvis
3. **Graph Embedding**: Tạo vector representation cho nodes/graphs bằng Random Walk và DeepWalk
4. **Classification**: Sử dụng học máy để phân loại đồ thị

## 🎯 Mục tiêu

- **Trực quan hóa**: Hiểu cấu trúc và mối quan hệ từ vựng trong văn bản tiếng Việt
- **Graph Embedding**: Học representation của nodes và graphs trong không gian vector
- **Machine Learning**: Áp dụng các thuật toán ML truyền thống trên graph embeddings
- **Giáo dục**: Minh họa các khái niệm cơ bản về Graph Neural Networks

## ✨ Các modules chính

### 1. Text to Graph Visualization (`text2graph.py`)

Chuyển đổi văn bản tiếng Việt thành đồ thị đồng xuất hiện (co-occurrence graph) và trực quan hóa tương tác.

**Khái niệm:**
- **Tokenization** tiếng Việt với `underthesea`
- **Sliding window** để xây dựng co-occurrence matrix
- Trọng số cạnh: **Frequency** (tần suất) và **PMI** (Pointwise Mutual Information)
- Trực quan hóa tương tác với Pyvis
- Thống kê và phân tích cấu trúc đồ thị

👉 [Chi tiết Text2Graph](README_TEXT2GRAPH.md)

**Chạy ứng dụng:**
```bash
streamlit run text2graph.py
```

---

### 2. Random Walk (`random_walk.py`)

Thuật toán duyệt đồ thị ngẫu nhiên để tạo sequences, nền tảng cho Graph Embedding.

**Khái niệm:**
- **Random Walk algorithm**: Di chuyển ngẫu nhiên trên đồ thị
- **Markov Chain**: Chuỗi Markov với transition probabilities
- Visualization tương tác với node coloring
- Sequence generation: Single demo và selected nodes
- Tích hợp với text2graph module

👉 [Chi tiết Random Walk](README_RANDOM_WALK.md)

**Chạy ứng dụng:**
```bash
streamlit run random_walk.py
```

---

### 3. DeepWalk - Graph Embedding (`deepwalk_notebook.ipynb`)

Học vector representations cho nodes sử dụng Random Walks và Skip-gram.

**Khái niệm:**
- **Pipeline**: Text → Graph → Random Walks → Skip-gram Training
- **Skip-gram model**: Học embeddings từ sequences (Word2Vec)
- **Node embeddings**: Vector representation trong không gian liên tục
- Phân tích similarity và clustering
- Visualization với t-SNE và K-means

👉 [Chi tiết Skip-gram](README_SKIP_GRAM.md)

**Chạy notebook:**
```bash
jupyter notebook deepwalk_notebook.ipynb
```

---

### 4. Text Classification (`text_classification_notebook.ipynb`)

Phân loại văn bản sử dụng Graph Embeddings và KNN classifier.

**Khái niệm:**
- **Document-level approach**: Document → Graph → Vector representation
- **Mean pooling**: Aggregate node embeddings thành document vector
- **KNN classification**: Phân loại dựa trên cosine similarity
- **t-SNE visualization**: Trực quan hóa trong không gian 2D
- **Nearest neighbors analysis**: Giải thích prediction

**Pipeline:**
```
Text → Graph → Node Embeddings → Mean Pooling → Document Vector → KNN
```

**Chạy notebook:**
```bash
jupyter notebook text_classification_notebook.ipynb
```

## 🚀 Cài đặt nhanh

```bash
# Clone repository
git clone https://github.com/hoadm-net/GraphEmbedding.git
cd GraphEmbedding

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy Streamlit apps
streamlit run text2graph.py
streamlit run random_walk.py

# Hoặc mở Jupyter notebooks
jupyter notebook
```

## 📊 Dataset

Project sử dụng dữ liệu văn bản tiếng Việt trong thư mục `data/`:
- **Training**: Files 1-10 (1-5: núi, 6-10: biển)
- **Test**: `test.txt` (cho classification demo)

Bạn có thể thêm files `.txt` của riêng mình vào thư mục này.

## 📁 Cấu trúc project

```
GraphEmbedding/
├── README.md                          # Tổng quan project
├── README_TEXT2GRAPH.md               # Chi tiết Text to Graph
├── README_RANDOM_WALK.md              # Chi tiết Random Walk  
├── README_SKIP_GRAM.md                # Chi tiết Skip-gram
├── text2graph.py                      # Streamlit: Text → Graph
├── random_walk.py                     # Streamlit: Random Walk
├── deepwalk_notebook.ipynb            # Notebook: DeepWalk pipeline
├── text_classification_notebook.ipynb # Notebook: Text Classification
├── requirements.txt                   # Python dependencies
├── .gitignore                 
└── data/                              # Văn bản đầu vào
    ├── 1.txt - 10.txt                 # Training data (núi & biển)
    └── test.txt                       # Test data
```


## 📚 Tài liệu tham khảo

### Papers
1. **DeepWalk**: Perozzi et al., 2014 - [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
2. **Node2Vec**: Grover & Leskovec, 2016 - [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
3. **Word2Vec**: Mikolov et al., 2013 - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

### Libraries
- [NetworkX](https://networkx.org/) - Graph analysis
- [Gensim](https://radimrehurek.com/gensim/) - Word2Vec implementation
- [Underthesea](https://github.com/undertheseanlp/underthesea) - Vietnamese NLP
- [Pyvis](https://pyvis.readthedocs.io/) - Interactive graph visualization
- [scikit-learn](https://scikit-learn.org/) - Machine learning
