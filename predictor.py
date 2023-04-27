import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
dataset = pd.read_csv("heart.csv")

# Input from user
age = int(input("Enter your age: "))
sex = int(input("Enter your sex (0 for female, 1 for male): "))
cp = int(input("Enter your chest pain type (1 to 4): "))
trestbps = int(input("Enter your resting blood pressure: "))
chol = int(input("Enter your serum cholesterol level: "))
fbs = int(
    input("Enter your fasting blood sugar (0 for < 120 mg/dl, 1 for > 120 mg/dl): "))
restecg = int(
    input("Enter your resting electrocardiographic results (0 to 2): "))
thalach = int(input("Enter your maximum heart rate achieved: "))
exang = int(input("Enter your exercise-induced angina (0 for no, 1 for yes): "))
oldpeak = float(
    input("Enter your ST depression induced by exercise relative to rest: "))
slope = int(
    input("Enter your the slope of the peak exercise ST segment (1 to 3): "))
ca = int(input("Enter the number of major vessels (0 to 3) colored by fluoroscopy: "))
thal = int(input("Enter your thalassemia (1 to 3): "))

# Prepare input data for prediction
user_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                     restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Initialize classifier and fit data
lr = LogisticRegression()
lr.fit(dataset.drop("target", axis=1), dataset["target"])

# Make prediction on user data
prediction = lr.predict(user_data)

# Print prediction
if prediction == 0:
    print("You are not likely to have heart disease.")
else:
    print("You are likely to have heart disease.")
