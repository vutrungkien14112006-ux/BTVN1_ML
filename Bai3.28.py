import numpy as np

# --- BÀI 3.28: CẬP NHẬT PERCEPTRON ---

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y_true = 1

print("=== KẾT QUẢ BÀI 3.28 ===")

# 1. Kiểm tra mẫu có bị phân lớp sai hay không
wTx = np.dot(w, x)
y_pred = np.sign(wTx)

print(f"1. wTx ban đầu = {wTx} -> Nhãn dự đoán = {y_pred}")

if y_pred != y_true:
    print("-> Điểm dữ liệu BỊ PHÂN LỚP SAI. Tiến hành cập nhật...")
    
    # 2. Thực hiện cập nhật Perceptron (Learning rate = 1)
    w_new = w + y_true * x
    print(f"2. Trọng số w sau cập nhật: {w_new}")
    
    # 3. Tính lại wTx sau khi cập nhật
    wTx_new = np.dot(w_new, x)
    y_pred_new = np.sign(wTx_new)
    print(f"3. wTx sau cập nhật = {wTx_new} -> Nhãn dự đoán mới = {y_pred_new} (Đã khớp với thực tế!)")
else:
    print("-> Điểm dữ liệu đã phân lớp đúng, không cần cập nhật.")