# 🚶 Random Walk Visualization

Tài liệu về thuật toán Random Walk và trực quan hóa sequences trên đồ thị từ vựng.

## 📋 Tổng quan

Random Walk là thuật toán duyệt đồ thị ngẫu nhiên, tạo ra **sequences** (chuỗi) các nodes bằng cách di chuyển ngẫu nhiên từ node này sang node kề cận. Trong project này, chúng ta sử dụng Random Walk để:

- **Khám phá cấu trúc đồ thị**: Hiểu cách các từ kết nối với nhau
- **Tạo sequences**: Chuỗi các từ có thể được sử dụng sau này cho Graph Embedding
- **Trực quan hóa**: Minh họa quá trình duyệt đồ thị một cách interactive
- **Giáo dục**: Hiểu nguyên lý cơ bản trước khi áp dụng DeepWalk/Node2Vec

## 🧮 Định nghĩa toán học

### 1. Đồ thị

Cho đồ thị vô hướng $G = (V, E)$ với:
- $V$: Tập hợp các đỉnh (nodes)
- $E$: Tập hợp các cạnh (edges)
- $|V| = n$ nodes, $|E| = m$ edges

### 2. Ma trận kề cận (Adjacency Matrix)

$$
A_{ij} = \begin{cases} 
1 & \text{nếu } (v_i, v_j) \in E \\
0 & \text{ngược lại}
\end{cases}
$$

### 3. Bậc của đỉnh (Degree)

$$
d(v_i) = \sum_{j=1}^{n} A_{ij}
$$

## 🎯 Random Walk Algorithm

### 1. Simple Random Walk

**Xác suất chuyển đổi (Transition Probability)**:

$$
P(X_{t+1} = v_j | X_t = v_i) = \begin{cases} 
\frac{1}{d(v_i)} & \text{nếu } (v_i, v_j) \in E \\
0 & \text{ngược lại}
\end{cases}
$$

**Ma trận chuyển đổi (Transition Matrix)**:

$$
P_{ij} = \frac{A_{ij}}{d(v_i)}
$$

Trong đó:
- $X_t$: Node tại thời điểm $t$
- $P_{ij}$: Xác suất di chuyển từ node $i$ đến node $j$
- Mỗi hàng của ma trận $P$ có tổng bằng 1

### 2. Weighted Random Walk

Với đồ thị có trọng số $w_{ij}$ trên cạnh $(v_i, v_j)$:

$$
P(X_{t+1} = v_j | X_t = v_i) = \frac{w_{ij}}{\sum_{k \in N(v_i)} w_{ik}}
$$

Trong đó $N(v_i)$ là tập neighbors của $v_i$.

### 3. Algorithm Implementation

```python
def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    current = start_node
    
    for t in range(walk_length - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        # Uniform random selection
        next_node = random.choice(neighbors)
        walk.append(next_node)
        current = next_node
    
    return walk
```

## 📊 Thuộc tính Markov

Random Walk là **Markov Chain** với tính chất:

$$
P(X_{t+1} = v_j | X_0, X_1, ..., X_t) = P(X_{t+1} = v_j | X_t)
$$

**Stationary Distribution**:

Phân phối dừng $\pi$ thỏa mãn:
$$
\pi^T P = \pi^T
$$

Với simple random walk trên đồ thị liên thông:
$$
\pi_i = \frac{d(v_i)}{2m}
$$

## 🎲 Tính chất thống kê

### 1. Expected Return Time

Thời gian kỳ vọng quay lại node $i$:
$$
E[T_i] = \frac{1}{\pi_i} = \frac{2m}{d(v_i)}
$$

### 2. Hitting Time

Thời gian kỳ vọng từ node $i$ đến node $j$ lần đầu:
$$
H_{ij} = E[T_j | X_0 = i]
$$

### 3. Mixing Time

Thời gian để phân phối hội tụ về stationary distribution:
$$
T_{mix}(\epsilon) = \min\{t : \max_i ||P^t(i, \cdot) - \pi||_{TV} \leq \epsilon\}
$$

## 🔄 Biến thể của Random Walk

### 1. Lazy Random Walk

Với xác suất $\alpha$ ở lại node hiện tại:
$$
P'_{ij} = \begin{cases} 
\alpha + (1-\alpha) \frac{1}{d(v_i)} & \text{nếu } i = j \\
(1-\alpha) \frac{A_{ij}}{d(v_i)} & \text{nếu } i \neq j
\end{cases}
$$

### 2. Random Walk with Restart

Với xác suất $c$ quay về node khởi đầu:
$$
P(X_{t+1} = v_j | X_t = v_i) = \begin{cases} 
c \cdot \mathbf{1}_{j=start} + (1-c) \frac{A_{ij}}{d(v_i)} & \text{nếu } (v_i, v_j) \in E \\
c \cdot \mathbf{1}_{j=start} & \text{ngược lại}
\end{cases}
$$

### 3. Biased Random Walk (Node2Vec style)

Với tham số $p$ (return) và $q$ (in-out):
$$
P(X_{t+1} = v_j | X_{t-1} = v_{t-1}, X_t = v_i) = \frac{\alpha_{pq}(v_{t-1}, v_j) \cdot w_{ij}}{Z}
$$

Trong đó:
$$
\alpha_{pq}(t, x) = \begin{cases} 
\frac{1}{p} & \text{nếu } d_{tx} = 0 \text{ (return to } t \text{)} \\
1 & \text{nếu } d_{tx} = 1 \text{ (same distance)} \\
\frac{1}{q} & \text{nếu } d_{tx} = 2 \text{ (move away)}
\end{cases}
$$

## 🎮 Ứng dụng trong Streamlit App

### 1. Node Sequence Generation

Random walk tạo ra sequences của nodes:
$$
S = \{s_1, s_2, ..., s_L\}
$$

Ví dụ: `["học", "sinh", "đi", "học", "bài"]`

### 2. Interactive Visualization

- **🔴 Selected nodes**: Nodes được người dùng chọn để bắt đầu walks
- **🟢 Walk nodes**: Nodes xuất hiện trong các walks được tạo
- **🔵 Other nodes**: Các nodes còn lại trong đồ thị
- **Edge highlighting**: Cạnh được tô đậm nếu kết nối selected nodes

### 3. Walk Modes

**Single walk demo**: Tạo 1 walk từ node đầu tiên trong selected nodes
```python
walk = walker.single_walk(start_node)
# Output: ["học", "sinh", "đi", "trường"]
```

**Selected nodes only**: Tạo nhiều walks từ tất cả selected nodes
```python
for start_node in selected_nodes:
    for _ in range(num_walks):
        walk = walker.single_walk(start_node)
        walks.append(walk)
```

### 4. Sequence Display

Walks được hiển thị dưới dạng dễ đọc:
```
Walk 1: học sinh → giáo viên → dạy → học
Walk 2: sinh viên → trường → đại học → học
Walk 3: học → bài → khó → làm
```

## � Implementation trong Script

### 1. RandomWalker Class

```python
class RandomWalker:
    def __init__(self, graph, walk_length=10, num_walks=10):
        self.graph = graph
        self.walk_length = walk_length  # Độ dài mỗi walk
        self.num_walks = num_walks      # Số walks từ mỗi node
        
    def single_walk(self, start_node):
        """Thực hiện một random walk từ node bắt đầu"""
        walk = [start_node]
        current_node = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)  # Uniform random
            walk.append(next_node)
            current_node = next_node
            
        return walk
```

### 2. Graph Loading từ Text

```python
# Sử dụng TextGraphBuilder từ text2graph.py
graph_builder = TextGraphBuilder(window_size=3, weight_method="frequency")
tokens = graph_builder.process_text(text)
graph_builder.build_cooccurrence_matrix(tokens)
G = graph_builder.build_graph(min_frequency=2)

# Tạo walker
walker = RandomWalker(G, walk_length=10, num_walks=3)
```

### 3. Visualization với Pyvis

```python
def visualize_graph_with_walks(graph, selected_nodes, walk_paths):
    """Trực quan hóa với color coding"""
    
    for node in graph.nodes():
        if node in selected_nodes:
            color = '#ff4444'  # Đỏ cho selected
        elif node in walk_nodes:
            color = '#44ff44'  # Xanh cho walk nodes
        else:
            color = '#4444ff'  # Xanh dương cho others
            
        net.add_node(node, color=color, size=20)
```

## ⚙️ Parameters trong UI

### 1. Graph Parameters
- **Window size**: Kích thước cửa sổ cho co-occurrence graph (từ text2graph)
- **Min frequency**: Tần suất tối thiểu của từ để được thêm vào graph
- **Weight method**: `frequency` hoặc `pmi` cho trọng số cạnh

### 2. Random Walk Parameters

**Walk length**: Độ dài mỗi walk sequence
```python
walk_length = 10  # Walk có tối đa 10 nodes
# Ví dụ: ["học", "sinh", "đi", "trường", "đại_học", ...]
```

**Walks per node**: Số walks được tạo từ mỗi selected node
```python
num_walks = 3     # Tạo 3 walks từ mỗi node được chọn
# Nếu chọn 2 nodes → tổng cộng 6 walks
```

### 3. Node Selection
- **Multiselect**: Chọn nhiều nodes để bắt đầu walks
- **Default**: Tự động chọn 3 nodes đầu tiên
- **Interactive**: Click để thêm/bỏ nodes

### 4. Walk Modes

**Single walk demo**: 
- Tạo 1 walk duy nhất từ node đầu tiên
- Dùng để demo algorithm step-by-step

**Selected nodes only**:
- Tạo nhiều walks từ tất cả selected nodes
- Phù hợp để khám phá neighborhood

## 📚 Tài liệu tham khảo

1. **Random Walks on Graphs**: Lovász, L. (1993) - [Classical theory](https://web.cs.elte.hu/~lovasz/erdos.pdf)
2. **DeepWalk**: Perozzi et al. (2014) - [Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
3. **Node2Vec**: Grover & Leskovec (2016) - [Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
4. **Graph Representation Learning**: Hamilton, W.L. (2020) - [Synthesis Lectures](https://www.cs.mcgill.ca/~wlh/grl_book/)

## 🎯 Sử dụng App

### 1. Setup
```bash
# Chạy ứng dụng
streamlit run random_walk.py
```

### 2. Workflow
1. **Chọn files**: Select text files từ thư mục `data/`
2. **Tạo graph**: Điều chỉnh parameters (window size, frequency, weight method)
3. **Chọn nodes**: Multiselect các nodes để highlight và tạo walks
4. **Configure walks**: Set walk length và số walks per node
5. **Run walks**: Chọn walk mode và xem visualization
6. **Analyze**: Xem walk sequences và patterns

### 3. Interpretation

**Visualization Colors:**
- 🔴 **Red nodes**: Nodes được bạn chọn (starting points)
- 🟢 **Green nodes**: Nodes xuất hiện trong generated walks  
- 🔵 **Blue nodes**: Nodes khác trong graph

**Walk Sequences:**
```
Walk 1: học sinh → đi → học → bài → khó
Walk 2: sinh viên → trường → đại học → học  
Walk 3: học → giáo viên → dạy → sinh viên
```

### 4. Use Cases
- **Graph exploration**: Hiểu structure và connectivity
- **Educational**: Học nguyên lý Random Walk algorithm
- **Preprocessing**: Tạo sequences cho Graph Embedding models
- **Analysis**: Khám phá semantic relationships trong văn bản tiếng Việt
