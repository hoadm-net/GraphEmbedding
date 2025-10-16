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

## ✨ Tính năng

### ✅ 1. Text to Graph Visualization (`text2graph.py`)

**Status**: ✅ Hoàn thành

- Tokenization tiếng Việt với `underthesea`
- Xây dựng co-occurrence graph với sliding window
- Tính trọng số cạnh: Frequency và PMI
- Trực quan hóa tương tác với Pyvis
- Thống kê và phân tích đồ thị

👉 [Chi tiết Text2Graph](README_TEXT2GRAPH.md)

**Sử dụng:**
```bash
streamlit run text2graph.py
```

---

### ✅ 2. Random Walk (`random_walk.py`)

**Status**: ✅ Hoàn thành

- Random Walk algorithm với RandomWalker class
- Interactive visualization với node coloring
- Walk modes: Single demo và Selected nodes
- Sequence generation và display
- Integration với text2graph module

👉 [Chi tiết Random Walk](README_RANDOM_WALK.md)

**Sử dụng:**
```bash
streamlit run random_walk.py
```

---

### 🚧 3. DeepWalk

**Status**: 🚧 Đang phát triển

*Placeholder: Học node embeddings từ random walks sử dụng Skip-gram model*

---

### 🚧 4. Graph Classification

**Status**: 🚧 Đang phát triển

*Placeholder: Phân loại đồ thị sử dụng graph embeddings và ML classifiers*

## 🚀 Cài đặt nhanh

```bash
# Clone repository
git clone <repository-url>
cd GraphEmbedding

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Cài đặt dependencies
pip install streamlit pandas networkx pyvis underthesea numpy

# Chạy ứng dụng
streamlit run text2graph.py
```

## 📁 Cấu trúc project

```
GraphEmbedding/
├── README.md                    # Tổng quan project
├── README_TEXT2GRAPH.md         # Chi tiết Text to Graph
├── README_RANDOM_WALK.md        # Chi tiết Random Walk
├── text2graph.py               # ✅ Streamlit app: Text → Graph
├── random_walk.py              # ✅ Streamlit app: Random Walk
├── deepwalk.py                 # 🚧 DeepWalk (coming soon)
├── graph_classification.py     # 🚧 Classification (coming soon)
├── requirements.txt            # Python dependencies
├── .gitignore                 
└── data/                      # Thư mục văn bản đầu vào
    ├── 1.txt
    ├── 2.txt
    └── ...
```

##  Tài liệu tham khảo

1. **DeepWalk**: Perozzi et al., 2014 - [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
2. **Node2Vec**: Grover & Leskovec, 2016 - [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
3. **Word2Vec**: Mikolov et al., 2013 - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

