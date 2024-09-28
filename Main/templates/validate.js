function validateForm() {
    const weight = parseFloat(document.getElementById("weight").value);
    const height = parseFloat(document.getElementById("height").value);
    const bodyLength = parseFloat(document.getElementById("body_length").value);
    const age = parseFloat(document.getElementById("age").value);
    const errorMessageElement = document.getElementById("error-message");

    // Kiểm tra xem tất cả các giá trị có lớn hơn 0 không
    if (weight <= 0 || height <= 0 || bodyLength <= 0 || age <= 0) {
        errorMessageElement.innerText = "Nhập sai dữ liệu. Vui lòng nhập lại với giá trị lớn hơn 0.";
        return false; // Ngăn không cho gửi biểu mẫu
    }

    // Nếu tất cả các giá trị hợp lệ, xóa thông báo lỗi
    errorMessageElement.innerText = "";
    return true; // Cho phép gửi biểu mẫu
}
