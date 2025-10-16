# 🕸️ Text to Graph Visualization

Ứng dụng trực quan hóa đồ thị từ văn bản tiếng Việt sử dụng Streamlit, NetworkX và Pyvis.

## 📋 Mô tả

Ứng dụng này chuyển đổi văn bản tiếng Việt thành đồ thị từ vựng (word co-occurrence graph), trong đó:
- **Nút (Nodes)**: Các từ hoặc từ ghép trong văn bản
- **Cạnh (Edges)**: Mối quan hệ đồng xuất hiện giữa các từ trong cùng một ngữ cảnh

Ứng dụng sử dụng thuật toán **sliding window** để xác định các từ xuất hiện gần nhau và xây dựng ma trận đồng xuất hiện.

## ✨ Tính năng

### 1. **Tokenization tiếng Việt**
- Sử dụng thư viện `underthesea` để tách từ chính xác
- Hỗ trợ từ ghép tiếng Việt (ví dụ: `trí_tuệ_nhân_tạo`, `học_sinh`)
- Tự động lọc bỏ dấu câu và từ quá ngắn

### 2. **Phương pháp tính trọng số cạnh**
- **Frequency (Tần suất)**: Số lần hai từ xuất hiện gần nhau
- **PMI (Pointwise Mutual Information)**: Đo lường mức độ liên kết ngữ nghĩa giữa hai từ

### 3. **Trực quan hóa tương tác**
- Đồ thị tương tác với Pyvis
- Layout Spring (ForceAtlas2) tự động
- Hiệu ứng hover hiển thị thông tin chi tiết:
  - Tần suất xuất hiện của từ
  - Bậc của nút (số lượng kết nối)
  - Danh sách các từ kề cận
- Tùy chỉnh kích thước cửa sổ ngữ cảnh
- Lọc từ theo tần suất tối thiểu

### 4. **Thống kê và phân tích**
- Số file được chọn
- Tổng số từ (bao gồm từ lặp lại)
- Từ vựng duy nhất (số từ khác nhau)
- Số cặp từ đồng xuất hiện
- Thông tin đồ thị (số nút, số cạnh, mật độ, bậc trung bình)
- Top từ có tần suất cao
- Top cặp từ đồng xuất hiện

## 📊 Phương pháp và Thuật toán

### 1. Sliding Window Co-occurrence

Thuật toán sử dụng cửa sổ trượt để xác định các từ xuất hiện gần nhau:

```
Văn bản: "học sinh đi học mỗi ngày"
Window size k=2:

Position 0 (học):     [học, sinh, đi]
Position 1 (sinh):    [học, sinh, đi, học]
Position 2 (đi):      [sinh, đi, học, mỗi]
...
```

Mỗi cặp từ trong cùng cửa sổ được ghi nhận là một lần đồng xuất hiện.

### 2. Pointwise Mutual Information (PMI)

PMI đo lường mức độ liên kết ngữ nghĩa giữa hai từ:

$$
\text{PMI}(w_1, w_2) = \log_2 \frac{P(w_1, w_2)}{P(w_1) \cdot P(w_2)}
$$

Trong đó:
- $P(w_1, w_2)$: Xác suất đồng xuất hiện của $w_1$ và $w_2$
- $P(w_1)$, $P(w_2)$: Xác suất xuất hiện của mỗi từ

PMI cao → Hai từ có xu hướng xuất hiện cùng nhau nhiều hơn ngẫu nhiên

**Công thức tính xác suất:**

$$
P(w_1, w_2) = \frac{\text{count}(w_1, w_2)}{N}
$$

$$
P(w_i) = \frac{\text{count}(w_i)}{N}
$$

Với $N$ là tổng số từ trong văn bản.

## 📚 Thư viện sử dụng

- **Streamlit**: Web framework để xây dựng UI
- **NetworkX**: Thư viện xử lý và phân tích đồ thị
- **Pyvis**: Trực quan hóa đồ thị tương tác
- **underthesea**: Tokenization tiếng Việt
- **Pandas**: Xử lý và hiển thị dữ liệu dạng bảng
- **NumPy**: Tính toán số học
