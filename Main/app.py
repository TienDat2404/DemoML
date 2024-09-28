from flask import Flask, render_template, request, redirect, url_for
import joblib
from sklearn.metrics import accuracy_score
import wheel
import pandas as pd 

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Lấy dữ liệu từ form HTML
        data = {key: value for key, value in request.form.items()}
        print(data)
        selected_model = request.form.get("model")  # Lấy mô hình mà người dùng đã chọn
        
        # Chuẩn bị dữ liệu (data preprocessing)
        df = pd.DataFrame([data])
        df.drop(columns='model')
        
        # Load scaler và label encoders
        scaler = joblib.load("./artifact/scaler.pkl")
        label_encoders = joblib.load("./artifact/label_encoders.pkl")

        # Xử lý label
        for col in df.select_dtypes('object').columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])
        
        # Chuẩn hóa dữ liệu
        df = scaler.transform(df)

        # Load mô hình và dự đoán
        model = None
        if selected_model == "perceptron":
            model = joblib.load("./model/perceptron_model.py")
        elif selected_model == "logistic_regression":
            model = joblib.load("./model/logistic_model.py")
        elif selected_model == "neural_network":
            model = joblib.load("./model/neural_network_model.py")

        # Dự đoán
        prediction = model.predict(df)
        # Trả về trang kết quả với thông tin mô hình và dự đoán
        return render_template("result.html", 
                               selected_model=selected_model,
                               prediction=prediction[0])

    return render_template("index.html")

@app.route("/back", methods=["GET"])
def back():
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
