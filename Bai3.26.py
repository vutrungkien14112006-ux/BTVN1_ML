# --- BÀI 3.26: GRADIENT DESCENT ---

# Định nghĩa hàm số f(x) = x^2 - 4x + 5
def f(x):
    return x**2 - 4*x + 5

# Định nghĩa đạo hàm f'(x) = 2x - 4
def derivative_f(x):
    return 2*x - 4

# Khởi tạo tham số
x = 5.0              # Điểm khởi tạo x(0)
learning_rate = 0.2  # Hệ số học (eta)
steps = 4            # Số bước cập nhật

print("=== KẾT QUẢ BÀI 3.26 ===")
print(f"Bước 0 (Khởi tạo): x = {x:.4f}, f(x) = {f(x):.4f}")

# Chạy vòng lặp cập nhật
for i in range(1, steps + 1):
    # Tính đạo hàm tại điểm x hiện tại
    grad = derivative_f(x)
    
    # Cập nhật x mới
    x = x - learning_rate * grad
    
    # Tính giá trị hàm số mới
    val = f(x)
    
    print(f"Bước {i}: Đạo hàm = {grad:.4f} | x_mới = {x:.4f} | f(x_mới) = {val:.4f}")