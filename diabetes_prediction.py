import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("diabetes.csv")

# Display Dataset Information
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Separate Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Accuracy
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("\nTraining Accuracy:",
      round(accuracy_score(y_train, train_pred) * 100, 2), "%")

print("Testing Accuracy:",
      round(accuracy_score(y_test, test_pred) * 100, 2), "%")

# User Input
print("\n===== Diabetes Prediction =====")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

input_data = np.array([
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]).reshape(1, -1)

input_data = scaler.transform(input_data)

prediction = model.predict(input_data)

if prediction[0] == 0:
    print("\nResult: The person is NOT diabetic.")
else:
    print("\nResult: The person IS diabetic.")