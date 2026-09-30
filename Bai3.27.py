import numpy as np

# --- BÀI 3.27: PHƯƠNG PHÁP PERCEPTRON ---

# Khai báo các vector w, x và nhãn thực tế y
w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
y_true = -1

print("\n=== KẾT QUẢ BÀI 3.27 ===")

# 1. Tính giá trị wTx (Tích vô hướng của w và x)
wTx = np.dot(w, x)
print(f"1. Giá trị wTx: {wTx}")

# 2. Xác định nhãn dự đoán (Lấy dấu của wTx)
# np.sign(x) trả về 1 nếu x > 0, -1 nếu x < 0, và 0 nếu x = 0
y_pred = np.sign(wTx)
print(f"2. Nhãn dự đoán: {y_pred}")

# 3. Kiểm tra xem điểm dữ liệu có bị phân lớp sai không
if y_pred != y_true:
    print(f"3. Kết luận: Điểm dữ liệu BỊ PHÂN LỚP SAI (Dự đoán: {y_pred}, Thực tế: {y_true})")
else:
    print(f"3. Kết luận: Điểm dữ liệu ĐƯỢC PHÂN LỚP ĐÚNG (Dự đoán: {y_pred}, Thực tế: {y_true})")