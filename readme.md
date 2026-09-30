# Bài tập Machine Learning: Perceptron & Gradient Descent

Repository này chứa các đoạn code Python giải quyết các bài tập thực hành về thuật toán Gradient Descent và mạng nơ-ron nhân tạo cơ bản (Perceptron) từ bài 3.26 đến 3.30.


## Phương pháp giải các bài tập

* **Bài 3.26 (Gradient Descent):**
  Bài toán yêu cầu tìm cực tiểu của hàm số $f(x) = x^2 - 4x + 5$. Cách làm là tính đạo hàm $f'(x) = 2x - 4$, sau đó sử dụng vòng lặp trong Python để mô phỏng 4 bước cập nhật giá trị $x$ theo công thức: $x_{new} = x_{old} - \eta \cdot f'(x_{old})$.

* **Bài 3.27 & 3.28 (Tính toán và cập nhật Perceptron):**
  Giải quyết bài toán phân lớp bằng cách dùng `numpy.dot()` để tính tích vô hướng $w^Tx$. Nhãn dự đoán được xác định bằng hàm `numpy.sign()`. Khi phát hiện điểm dữ liệu bị phân lớp sai (nhãn dự đoán khác nhãn thực tế $y$), thuật toán sẽ tự động cập nhật lại vector trọng số theo quy tắc: $w_{new} = w_{old} + y \cdot x$.

* **Bài 3.29 (Xây dựng Class Perceptron):**
  Mô phỏng lại mã nguồn của các thư viện ML. Code tự xây dựng một Class `BasicPerceptron` thuần túy bằng thư viện Toán học `numpy`. Trong đó:
  * **Hàm `fit`**: Lặp qua các epochs, kiểm tra từng điểm dữ liệu và cập nhật trọng số nếu bị phân lớp sai.
  * **Hàm `predict`**: Nhận dữ liệu mới, tính tích vô hướng với trọng số $w$ đã huấn luyện và trả về nhãn tương ứng.

* **Bài 3.30 (Áp dụng phân lớp dữ liệu):**
  Sử dụng mô hình `Perceptron` có sẵn của thư viện `scikit-learn` áp dụng vào một tập dữ liệu giả lập (bài toán phát hiện giao dịch gian lận). Hiệu suất của thuật toán được đo lường chi tiết thông qua 4 chỉ số: Accuracy, Precision, Recall và F1-score.

