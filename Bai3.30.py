from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- BÀI 3.30: BÀI TOÁN PHÂN LỚP GIAO DỊCH GIAN LẬN ---

# 1. Tạo tập dữ liệu giả lập (vd: 1000 giao dịch, 20% là gian lận)
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, 
                           weights=[0.8, 0.2], random_state=42)

# Đổi nhãn 0 thành -1 để đúng chuẩn toán học của Perceptron
import numpy as np
y = np.where(y == 0, -1, 1)

# 2. Chia tập Train / Test (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Khởi tạo mô hình Perceptron và huấn luyện (fit)
model = Perceptron(eta0=0.1, max_iter=100, random_state=42)
model.fit(X_train, y_train)

# 4. Dự báo dữ liệu mới (predict)
y_pred = model.predict(X_test)

# 5. Đánh giá mô hình bằng 4 độ đo (Accuracy, Precision, Recall, F1)
print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PERCEPTRON (GIAO DỊCH GIAN LẬN) ===")
print(f"- Accuracy  (Độ chính xác tổng thể): {accuracy_score(y_test, y_pred):.4f}")
print(f"- Precision (Độ chuẩn xác)         : {precision_score(y_test, y_pred):.4f}")
print(f"- Recall    (Độ bao phủ)           : {recall_score(y_test, y_pred):.4f}")
print(f"- F1-score  (Trung bình hài hòa)   : {f1_score(y_test, y_pred):.4f}")