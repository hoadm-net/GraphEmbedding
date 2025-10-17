# 🕸️ Text to Graph - Chuyển văn bản thành đồ thị

Chuyển đổi văn bản tiếng Việt thành đồ thị từ vựng (word co-occurrence graph) với trực quan hóa tương tác.

## 📋 Mô tả

Module này chuyển đổi văn bản tiếng Việt thành **co-occurrence graph**, trong đó:
- **Nodes (Nút)**: Các từ hoặc từ ghép trong văn bản
- **Edges (Cạnh)**: Mối quan hệ đồng xuất hiện giữa các từ trong cùng ngữ cảnh

Sử dụng thuật toán **sliding window** để xác định các từ xuất hiện gần nhau và xây dựng ma trận đồng xuất hiện (co-occurrence matrix).

## 🎯 Các khái niệm chính

### 1. **Tokenization tiếng Việt**
- Sử dụng thư viện `underthesea` để tách từ chính xác cho tiếng Việt
- Hỗ trợ từ ghép: `trí_tuệ_nhân_tạo`, `học_sinh`, `đại_học`
- Tự động lọc bỏ dấu câu và từ quá ngắn

### 3. **Phương pháp tính trọng số**

Có hai phương pháp để tính trọng số cho các cạnh:

#### **a) Frequency (Tần suất)**
Số lần hai từ xuất hiện gần nhau trong cửa sổ ngữ cảnh:

$$w_{ij} = \text{count}(w_i, w_j)$$

#### **b) PMI (Pointwise Mutual Information)**
Đo lường mức độ liên kết ngữ nghĩa giữa hai từ:
Cửa sổ trượt để xác định ngữ cảnh của từ:

```
Văn bản: "học sinh đi học mỗi ngày"
Window size k=2:

Position 0 (học):     [học, sinh, đi]
Position 1 (sinh):    [học, sinh, đi, học]
Position 2 (đi):      [sinh, đi, học, mỗi]
...
```

Mỗi cặp từ trong cùng cửa sổ được ghi nhận là một lần đồng xuất hiện.

PMI đo lường mức độ liên kết ngữ nghĩa giữa hai từ:

$$\text{PMI}(w_1, w_2) = \log_2 \frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}$$

Trong đó:
- $P(w_1, w_2)$: Xác suất đồng xuất hiện của $w_1$ và $w_2$
- $P(w_1)$, $P(w_2)$: Xác suất xuất hiện của mỗi từ

PMI cao → Hai từ có xu hướng xuất hiện cùng nhau nhiều hơn ngẫu nhiên

**Công thức tính xác suất:**

$$P(w_1, w_2) = \frac{\text{count}(w_1, w_2)}{N}$$

$$P(w_i) = \frac{\text{count}(w_i)}{N}$$

Với $N$ là tổng số từ trong văn bản.

## 🎨 Trực quan hóa

Module cung cấp trực quan hóa tương tác với Pyvis:
- **Layout**: Spring (ForceAtlas2) tự động phân bố nodes
- **Node size**: Tỷ lệ với tần suất xuất hiện
- **Edge thickness**: Tỷ lệ với trọng số cạnh
- **Hover effects**: Hiển thị thông tin chi tiết:
  - Tần suất xuất hiện của từ
  - Bậc của node (degree)
  - Danh sách các từ kề cận (neighbors)

## 📊 Thống kê đồ thị

Phân tích cấu trúc đồ thị qua các metrics:
- **Số nodes**: Từ vựng duy nhất
- **Số edges**: Cặp từ đồng xuất hiện
- **Density**: Mức độ kết nối trong đồ thị
- **Average degree**: Số kết nối trung bình mỗi node
- **Top words**: Từ có tần suất cao nhất
- **Top pairs**: Cặp từ đồng xuất hiện nhiều nhất

## �️ Thư viện sử dụng

- **Streamlit**: Web framework để xây dựng UI
- **NetworkX**: Thư viện xử lý và phân tích đồ thị
- **Pyvis**: Trực quan hóa đồ thị tương tác
- **underthesea**: Tokenization tiếng Việt
- **Pandas**: Xử lý và hiển thị dữ liệu dạng bảng
- **NumPy**: Tính toán số học
