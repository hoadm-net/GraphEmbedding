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

### ✅ 3. DeepWalk (`deepwalk_notebook.ipynb`)

**Status**: ✅ Hoàn thành

- Pipeline hoàn chỉnh: Text → Graph → Random Walks → Skip-gram Training
- Word2Vec implementation với gensim
- Node embeddings analysis và similarity
- t-SNE visualization
- Clustering với K-means
- Export embeddings và model

👉 [Chi tiết Skip-gram](README_SKIP_GRAM.md)

**Sử dụng:**
```bash
# Mở Jupyter Notebook
jupyter notebook deepwalk_notebook.ipynb
```

---

### ✅ 4. Text Classification (`text_classification_notebook.ipynb`)

**Status**: ✅ Hoàn thành

- **Document-level approach**: Mỗi document → 1 graph → 1 vector
- **Pipeline**: Doc2Graph → Embeddings → Mean Pooling → KNN
- **Dataset**: 10 training files (5 núi, 5 biển) + 1 test file
- **Visualization**: t-SNE cho 11 documents với nearest neighbors
- **Educational**: Step-by-step annotations cho seminar

**Features:**
- Mean pooling để tạo document vectors từ node embeddings
- KNN classification (k=3) với cosine similarity
- Interactive visualization: training + test documents
- Nearest neighbors analysis với distance metrics
- Prediction confidence scores

**Sử dụng:**
```bash
# Mở Jupyter Notebook
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
├── text2graph.py                      # ✅ Streamlit: Text → Graph
├── random_walk.py                     # ✅ Streamlit: Random Walk
├── deepwalk_notebook.ipynb            # ✅ Notebook: DeepWalk pipeline
├── text_classification_notebook.ipynb # ✅ Notebook: Text Classification
├── requirements.txt                   # Python dependencies
├── .gitignore                 
└── data/                             # Văn bản đầu vào
    ├── 1.txt - 10.txt                # Training data (núi & biển)
    └── test.txt                      # Test data
```

## 🎓 Use Cases

### 1. **Seminar/Teaching**
- Interactive demos với Streamlit apps
- Step-by-step notebooks với detailed annotations
- Visual explanations của graph concepts

### 2. **Research**
- Baseline implementations cho graph embedding methods
- Easy experimentation với different parameters
- Export results cho further analysis

### 3. **Vietnamese NLP**
- Co-occurrence graphs cho Vietnamese text
- Integration với underthesea tokenizer
- Domain-specific vocabulary analysis

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

## 🤝 Contributing

Contributions are welcome! Vui lòng tạo issue hoặc pull request.

## 📝 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 👨‍💻 Author

**Hoa Dinh**
- GitHub: [@hoadm-net](https://github.com/hoadm-net)
- Repository: [GraphEmbedding](https://github.com/hoadm-net/GraphEmbedding)

---

⭐ **Star this repo** nếu bạn thấy hữu ích cho việc học và nghiên cứu!

