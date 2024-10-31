from flask import Flask, render_template, request 
import joblib
import pandas as pd

app = Flask(__name__)

# Load các mô hình và công cụ tiền xử lý
perceptron_model = joblib.load('models/perceptron_model.pkl')
logistic_model = joblib.load('models/logistic_model.pkl')
mlp_model = joblib.load('models/mlp_model.pkl')
bagging_model = joblib.load('models/bagging_model.pkl')
scaler = joblib.load('artifacts/scaler.pkl')
label_encoders = joblib.load('artifacts/label_encoders.pkl')

# Load dữ liệu 
data = pd.read_csv('data/animal_gender_classification.csv', sep=';')
animal_names = data['Species'].unique()  # lấy danh sách tên động vật
fur_colors = data['Fur Color'].unique()  # Lấy danh sách màu lông

@app.route('/')
def index():
    return render_template('index.html', animal_names=animal_names, fur_colors=fur_colors)

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    body_length = float(request.form['body_length'])
    has_horns = int(request.form['has_horns'])
    fur_color = request.form['fur_color']
    lays_eggs = int(request.form['lays_eggs'])
    age = float(request.form['age'])
    model_choice = request.form['model']  

    # Mã hóa và chuẩn hóa dữ liệu đầu vào
    fur_color_encoded = label_encoders['Fur Color'].transform([fur_color])[0]
    input_data = [[weight, height, body_length, has_horns, fur_color_encoded, lays_eggs, age]]
    input_data_scaled = scaler.transform(input_data)

    # Lựa chọn mô hình dự đoán
    if model_choice == 'perceptron':
        model = perceptron_model
    elif model_choice == 'logistic':
        model = logistic_model
    elif model_choice == 'mlp':
        model = mlp_model
    elif model_choice == 'bagging':
        model = bagging_model
    else:
        return "Model not found", 404

    # Dự đoán
    prediction = model.predict(input_data_scaled)[0]
    
    # Chuyển kết quả dự đoán thành tên có thể đọc được
    prediction_label = label_encoders['Gender'].inverse_transform([prediction])[0]

    # Lưu trữ độ chính xác mô hình (nếu cần)
    perceptron_accuracy = joblib.load('models/perceptron_accuracy.pkl') * 100
    logistic_accuracy = joblib.load('models/logistic_accuracy.pkl') * 100
    mlp_accuracy = joblib.load('models/mlp_accuracy.pkl') * 100
    bagging_accuracy = joblib.load('models/bagging_accuracy.pkl') * 100

    # Trả về kết quả hiển thị
    return render_template(
        'index.html',
        animal_names=animal_names,
        fur_colors=fur_colors,
        result={
            'prediction': prediction_label,
            'perceptron_accuracy': perceptron_accuracy,
            'logistic_accuracy': logistic_accuracy,
            'mlp_accuracy': mlp_accuracy,
            'bagging_accuracy': bagging_accuracy,
        }
    )

if __name__ == '__main__':
    app.run(debug=True)
