# 🎯 Thuật Toán Skip-gram

Tài liệu chi tiết về thuật toán Skip-gram - nền tảng của Word2Vec và DeepWalk.

## 📋 Tổng quan

Skip-gram là thuật toán **mạng neural** để học **word embeddings** (hoặc node embeddings trong DeepWalk). Ý tưởng cốt lõi: **dự đoán các từ ngữ cảnh từ từ trung tâm**.

### **Khác biệt với CBOW:**
- **Skip-gram**: Từ trung tâm → Dự đoán các từ ngữ cảnh
- **CBOW**: Các từ ngữ cảnh → Dự đoán từ trung tâm

## 🎯 Mục tiêu của Skip-gram

Cho một **từ trung tâm**, tối đa hóa xác suất dự đoán các **từ ngữ cảnh** xung quanh nó.

### **Hàm mục tiêu toán học:**

$$
\text{maximize} \quad \mathcal{L} = \sum_{w \in \text{corpus}} \sum_{c \in C(w)} \log P(c | w)
$$

Trong đó:
- $w$: từ trung tâm  
- $C(w)$: các từ ngữ cảnh của $w$
- $P(c | w)$: xác suất dự đoán từ ngữ cảnh $c$ khi biết từ trung tâm $w$

## 📖 Ví dụ minh họa cụ thể

### **Câu đầu vào:**
```
"Học sinh đi học tại trường đại học"
```

### **Sau khi tách từ:**
```
[học, sinh, đi, học, tại, trường, đại, học]
```

### **Kích thước cửa sổ ngữ cảnh = 2:**

| Vị trí | Từ trung tâm | Cửa sổ ngữ cảnh | Các từ ngữ cảnh |
|--------|-------------|----------------|-----------------|
| 0 | học | [-, -, **học**, sinh, đi] | [sinh, đi] |
| 1 | sinh | [-, **học**, **sinh**, đi, học] | [học, đi, học] |
| 2 | đi | [**học**, **sinh**, **đi**, học, tại] | [học, sinh, học, tại] |
| 3 | học | [**sinh**, **đi**, **học**, tại, trường] | [sinh, đi, tại, trường] |
| 4 | tại | [**đi**, **học**, **tại**, trường, đại] | [đi, học, trường, đại] |

### **Ví dụ huấn luyện:**
```python
# Từ vị trí 1 (từ trung tâm = "sinh"):
Ví dụ huấn luyện:
- Đầu vào: "sinh" → Mục tiêu: "học" 
- Đầu vào: "sinh" → Mục tiêu: "đi"
- Đầu vào: "sinh" → Mục tiêu: "học" (lần 2)

# Từ vị trí 2 (từ trung tâm = "đi"):  
Ví dụ huấn luyện:
- Đầu vào: "đi" → Mục tiêu: "học"
- Đầu vào: "đi" → Mục tiêu: "sinh" 
- Đầu vào: "đi" → Mục tiêu: "học" (lần 2)
- Đầu vào: "đi" → Mục tiêu: "tại"
```

## 🧮 Kiến trúc Skip-gram

### **Cấu trúc mạng Neural Network:**

```
Lớp đầu vào      Lớp ẩn        Lớp đầu ra
(One-hot)       (Embeddings)    (Softmax)

   học              [0.2]           P(học|sinh)
   sinh      →      [0.5]      →    P(đi|sinh)  
   đi               [0.1]           P(tại|sinh)
   tại              [0.8]           ...
   ...              [...]
```

### **Công thức toán học:**

**Đầu vào**: Vector one-hot $\mathbf{x}_w$ cho từ trung tâm $w$

**Lớp ẩn**: Vector embedding
$$
\mathbf{h} = \mathbf{W}^T \mathbf{x}_w = \mathbf{v}_w
$$

**Lớp đầu ra**: Phân phối xác suất
$$
P(c | w) = \frac{\exp(\mathbf{u}_c^T \mathbf{v}_w)}{\sum_{j=1}^{|V|} \exp(\mathbf{u}_j^T \mathbf{v}_w)}
$$

Trong đó:
- $\mathbf{v}_w$: **input embedding** của từ $w$
- $\mathbf{u}_c$: **output embedding** của từ ngữ cảnh $c$  
- $|V|$: kích thước từ điển (vocabulary size)

## 🔢 Ví dụ tính toán cụ thể

### **Thiết lập:**
```python
vocabulary = ["học", "sinh", "đi", "tại", "trường"]
vocab_size = 5
embedding_dim = 3
```

### **Bước 1: Khởi tạo embeddings**
```python
# Input embeddings (W_in) - ma trận 5x3
W_in = [
    [0.1, 0.2, 0.3],  # học
    [0.4, 0.5, 0.6],  # sinh  
    [0.7, 0.8, 0.9],  # đi
    [0.2, 0.3, 0.4],  # tại
    [0.5, 0.6, 0.7]   # trường
]

# Output embeddings (W_out) - ma trận 5x3  
W_out = [
    [0.9, 0.8, 0.7],  # học
    [0.6, 0.5, 0.4],  # sinh
    [0.3, 0.2, 0.1],  # đi  
    [0.7, 0.6, 0.5],  # tại
    [0.4, 0.3, 0.2]   # trường
]
```

### **Bước 2: Lan truyền xuôi (Forward pass)**
**Ví dụ huấn luyện**: Từ trung tâm = "sinh" (chỉ số 1), Mục tiêu = "đi" (chỉ số 2)

```python
# Lấy input embedding cho "sinh"
v_sinh = W_in[1] = [0.4, 0.5, 0.6]

# Tính điểm cho tất cả các từ
scores = []
for i in range(vocab_size):
    u_i = W_out[i]
    score = dot_product(u_i, v_sinh)
    scores.append(score)

# Ví dụ tính toán:
# score("học") = [0.9, 0.8, 0.7] · [0.4, 0.5, 0.6] = 0.36 + 0.40 + 0.42 = 1.18
# score("sinh") = [0.6, 0.5, 0.4] · [0.4, 0.5, 0.6] = 0.24 + 0.25 + 0.24 = 0.73
# score("đi") = [0.3, 0.2, 0.1] · [0.4, 0.5, 0.6] = 0.12 + 0.10 + 0.06 = 0.28
```

### **Bước 3: Áp dụng Softmax**
```python
scores = [1.18, 0.73, 0.28, 1.02, 0.53]

# Áp dụng softmax
exp_scores = [exp(s) for s in scores] = [3.25, 2.07, 1.32, 2.77, 1.70]
sum_exp = sum(exp_scores) = 11.11

# Tính xác suất
probabilities = [s/sum_exp for s in exp_scores]
P = [0.293, 0.186, 0.119, 0.249, 0.153]

# P(đi | sinh) = 0.119
```

### **Bước 4: Tính toán hàm mất mát**
```python
# Cross-entropy loss cho mục tiêu "đi" (chỉ số 2)
target_prob = P[2] = 0.119
loss = -log(target_prob) = -log(0.119) = 2.13
```

## 🔄 Tối ưu hóa Negative Sampling

### **Vấn đề với vanilla softmax:**
Tính toán softmax cho từ điển lớn (10K-100K từ) rất tốn kém về mặt tính toán!

### **Giải pháp: Negative Sampling**

Thay vì dự đoán trên toàn bộ từ điển, chỉ phân biệt giữa:
- 1 **ví dụ dương** (từ ngữ cảnh thật)
- $k$ **ví dụ âm** (các từ ngẫu nhiên)

### **Hàm mục tiêu:**
$$
\log \sigma(\mathbf{u}_c^T \mathbf{v}_w) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-\mathbf{u}_{w_i}^T \mathbf{v}_w)]
$$

Trong đó:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$: hàm sigmoid
- $P_n(w) \propto U(w)^{3/4}$: phân phối lấy mẫu âm
- $k = 5-20$: số lượng negative samples

### **Ví dụ Negative Sampling:**
```python
# Ví dụ huấn luyện: "sinh" → "đi" (dương)
positive_pair = ("sinh", "đi")

# Các mẫu âm ngẫu nhiên (k=3):
negative_pairs = [
    ("sinh", "trường"),  # từ ngẫu nhiên 1
    ("sinh", "học"),     # từ ngẫu nhiên 2  
    ("sinh", "tại")      # từ ngẫu nhiên 3
]

# Mục tiêu:
# Tối đa hóa: P("đi" | "sinh") = sigmoid(u_đi · v_sinh)
# Tối thiểu hóa: P("trường" | "sinh") = sigmoid(u_trường · v_sinh)  
# Tối thiểu hóa: P("học" | "sinh") = sigmoid(u_học · v_sinh)
# Tối thiểu hóa: P("tại" | "sinh") = sigmoid(u_tại · v_sinh)
```

## 🚀 Skip-gram trong DeepWalk

### **Chuỗi Random Walk:**
```python
walks = [
    ["học", "sinh", "đi", "học", "tại"],
    ["sinh", "học", "trường", "đại"],
    ["đi", "tại", "trường", "học"]
]
```

### **Chuyển đổi thành dữ liệu huấn luyện Skip-gram:**
```python
training_pairs = []

for walk in walks:
    for i, center_word in enumerate(walk):
        for j in range(max(0, i-window), min(len(walk), i+window+1)):
            if i != j:
                context_word = walk[j]
                training_pairs.append((center_word, context_word))

# Kết quả:
training_pairs = [
    ("học", "sinh"), ("học", "đi"),        # từ "học" tại vị trí 0
    ("sinh", "học"), ("sinh", "đi"), ("sinh", "học"),  # từ "sinh" tại vị trí 1
    ("đi", "học"), ("đi", "sinh"), ("đi", "học"), ("đi", "tại"), # từ "đi" tại vị trí 2
    # ...
]
```

## 💻 Triển khai với Gensim

### **Sử dụng cơ bản:**
```python
from gensim.models import Word2Vec

# Chuẩn bị corpus (danh sách các câu đã tách từ)
corpus = [
    ["học", "sinh", "đi", "học", "tại"],
    ["sinh", "học", "trường", "đại"],
    ["đi", "tại", "trường", "học"]
]

# Huấn luyện mô hình Skip-gram
model = Word2Vec(
    sentences=corpus,
    vector_size=128,        # số chiều embedding
    window=5,               # kích thước cửa sổ ngữ cảnh  
    min_count=1,           # tần suất từ tối thiểu
    sg=1,                  # 1=skip-gram, 0=CBOW
    negative=15,           # negative sampling
    epochs=10,             # số epoch huấn luyện
    alpha=0.025,           # tốc độ học ban đầu
    workers=4              # số worker song song
)

# Lấy embeddings
embedding_học = model.wv['học']        # numpy array [128,]
similarity = model.wv.similarity('học', 'sinh')  # độ tương đồng cosine

# Tìm từ tương tự
similar_words = model.wv.most_similar('học', topn=5)
```

### **Tham số nâng cao:**
```python
model = Word2Vec(
    sentences=corpus,
    vector_size=300,           # embedding kích thước lớn hơn
    window=10,                 # ngữ cảnh rộng hơn  
    min_count=5,              # lọc từ hiếm
    sg=1,                     # skip-gram
    hs=0,                     # 0=negative sampling, 1=hierarchical softmax
    negative=20,              # nhiều negative samples hơn
    ns_exponent=0.75,         # số mũ negative sampling
    alpha=0.025,              # tốc độ học
    min_alpha=0.0001,         # tốc độ học tối thiểu
    epochs=30,                # nhiều epoch huấn luyện hơn
    batch_words=10000,        # số từ mỗi batch
    workers=8,                # xử lý song song
    callbacks=[callback]      # callback huấn luyện
)
```

## 🎯 Điều chỉnh tham số (Hyperparameter Tuning)

### **Kích thước Vector:**
- **Nhỏ** (50-100): Huấn luyện nhanh, phù hợp dữ liệu nhỏ
- **Trung bình** (100-300): Lựa chọn chuẩn, cân bằng tốt
- **Lớn** (300-1000): Chất lượng tốt hơn, cần nhiều dữ liệu hơn

### **Kích thước cửa sổ ngữ cảnh:**
- **Nhỏ** (2-5): Nắm bắt mối quan hệ cú pháp  
- **Lớn** (10-20): Nắm bắt mối quan hệ ngữ nghĩa
- **Nguyên tắc**: 5-10 cho hầu hết ứng dụng

### **Negative Sampling:**
- **Nhỏ** (5-10): Huấn luyện nhanh
- **Lớn** (15-25): Chất lượng tốt hơn, huấn luyện chậm hơn
- **Công thức**: $k = \text{min}(20, \text{max}(5, \frac{\text{vocab\_\_size}}{10000}))$

### **Tốc độ học (Learning Rate):**
- **Ban đầu**: 0.025 (mặc định của Gensim)
- **Giảm dần**: Giảm tuyến tính xuống min_alpha
- **Alpha tối thiểu**: 0.0001

## 📊 Phương pháp đánh giá

### **Đánh giá nội tại (Intrinsic Evaluation):**
1. **Độ tương đồng từ**: Tương quan với đánh giá của con người
2. **Loại suy từ**: $\mathbf{v}_{\text{vua}} - \mathbf{v}_{\text{đàn ông}} + \mathbf{v}_{\text{phụ nữ}} \approx \mathbf{v}_{\text{nữ hoàng}}$
3. **Phân cụm**: K-means trên embeddings

### **Đánh giá ngoại tại (Extrinsic Evaluation):**  
1. **Tác vụ hạ nguồn**: Phân loại, nhận dạng thực thể, gán nhãn từ loại
2. **Truy vấn thông tin**: Khớp query-document
3. **Hệ thống gợi ý**: Độ tương đồng item

### **Ví dụ đánh giá:**
```python
# Độ tương đồng từ
similarity_score = model.wv.similarity('học', 'sinh')
print(f"Độ tương đồng: {similarity_score:.3f}")

# Loại suy từ
result = model.wv.most_similar(
    positive=['sinh', 'giáo'], 
    negative=['học'], 
    topn=1
)
print(f"học : sinh :: giáo : {result[0][0]}")

# Đánh giá phân cụm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

embeddings = [model.wv[word] for word in model.wv.key_to_index]
kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(embeddings)
score = silhouette_score(embeddings, labels)
print(f"Điểm silhouette: {score:.3f}")
```

## 🔍 So sánh Skip-gram với các phương pháp khác

| Phương pháp | Ưu điểm | Nhược điểm | Trường hợp sử dụng |
|-------------|---------|------------|-------------------|
| **Skip-gram** | Tốt với từ hiếm, nắm bắt độ tương đồng ngữ nghĩa | Tính toán tốn kém | Đa năng, tác vụ ngữ nghĩa |
| **CBOW** | Huấn luyện nhanh, tốt với từ phổ biến | Kém với từ hiếm | Tạo mẫu nhanh, tác vụ cú pháp |
| **FastText** | Xử lý từ ngoài từ điển, thông tin subword | Phức tạp hơn, mô hình lớn hơn | Ngôn ngữ giàu hình thái |
| **GloVe** | Thống kê toàn cục, tính xác định | Giả định mối quan hệ tuyến tính | Tác vụ loại suy, khả năng giải thích |

## 📚 Tài liệu tham khảo

### **Bài báo gốc:**
1. **Word2Vec**: [Mikolov et al., 2013 - Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781)
2. **Negative Sampling**: [Mikolov et al., 2013 - Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546)
3. **DeepWalk**: [Perozzi et al., 2014 - DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)

### **Hướng dẫn học tập:**
- [Word2Vec Tutorial - Chris McCormick](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [The Illustrated Word2Vec - Jay Alammar](https://jalammar.github.io/illustrated-word2vec/)
- [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/models/word2vec.html)

### **Chủ đề nâng cao:**
- **Hierarchical Softmax**: Thay thế cho negative sampling
- **Thông tin Subword**: Mở rộng FastText
- **Contextualized Embeddings**: Phát triển BERT, GPT
