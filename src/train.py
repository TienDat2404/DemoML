import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Đọc dữ liệu
data = pd.read_csv('data/animal_gender_classification.csv', sep=';')

# Tiền xử lý dữ liệu
label_encoders = {}
for column in ['Has Horns', 'Fur Color', 'Lays Eggs', 'Gender']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
joblib.dump(label_encoders, 'artifacts/label_encoders.pkl')

# Chia thành tập huấn luyện - tập kiểm tra
X = data[['Weight (kg)', 'Height (cm)', 'Body Length (cm)', 'Has Horns', 'Fur Color', 'Lays Eggs', 'Age (years)']]
y = data['Gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)        
joblib.dump(scaler, 'artifacts/scaler.pkl')

# Huấn luyện mô hình
perceptron_model = Perceptron()
perceptron_model.fit(X_train, y_train)
joblib.dump(perceptron_model, 'models/perceptron_model.pkl')

logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train, y_train)
joblib.dump(logistic_model, 'models/logistic_model.pkl')

mlp_model = MLPClassifier(max_iter=3000)
mlp_model.fit(X_train, y_train)
joblib.dump(mlp_model, 'models/mlp_model.pkl')

bagging_model = BaggingClassifier(estimator=logistic_model, n_estimators=10)
bagging_model.fit(X_train, y_train)
joblib.dump(bagging_model, 'models/bagging_model.pkl')

# Dự đoán trên tập kiểm tra
y_pred_perceptron = perceptron_model.predict(X_test)
y_pred_logistic = logistic_model.predict(X_test)
y_pred_mlp = mlp_model.predict(X_test)
y_pred_bagging = bagging_model.predict(X_test)

# Đánh giá mô hình
print("Perceptron Model Accuracy:", accuracy_score(y_test, y_pred_perceptron))
print("Logistic Regression Model Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("MLP Model Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Bagging Model Accuracy:", accuracy_score(y_test, y_pred_bagging))

# In kết quả
print("\nClassification Report for Perceptron:\n", classification_report(y_test, y_pred_perceptron))
print("Classification Report for Logistic Regression:\n", classification_report(y_test, y_pred_logistic))
print("Classification Report for MLP:\n", classification_report(y_test, y_pred_mlp))
print("Classification Report for Bagging:\n", classification_report(y_test, y_pred_bagging))
