import numpy as np

# --- BÀI 3.29: CLASS PERCEPTRON CƠ BẢN ---

class BasicPerceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None

    def fit(self, X, y):
        # Số lượng mẫu (N) và số đặc trưng (d)
        N, d = X.shape
        
        # Khởi tạo trọng số w ban đầu bằng 0 (kích thước d, vì giả sử X đã có bias)
        self.w = np.zeros(d)

        # Lặp qua các epochs
        for epoch in range(self.epochs):
            for i in range(N):
                # Tính dự đoán: wTx
                wTx = np.dot(X[i], self.w)
                y_pred = np.sign(wTx) if wTx != 0 else 1
                
                # Cập nhật nếu phân lớp sai
                if y_pred != y[i]:
                    self.w = self.w + self.lr * y[i] * X[i]

    def predict(self, X_new):
        # Tính tích vô hướng và lấy dấu để trả về nhãn dự báo
        wTx = np.dot(X_new, self.w)
        return np.where(wTx >= 0, 1, -1)