# training/src/predict.py
import mlflow
import pandas as pd
import os

# --------------------------------------------------
# Set MLflow tracking
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
MLRUNS_PATH = os.path.join(PROJECT_ROOT, "mlruns")

mlflow.set_tracking_uri("file:///" + MLRUNS_PATH.replace("\\", "/"))

MODEL_NAME = "Student_Grade_Model"
MODEL_STAGE = "Staging"

# --------------------------------------------------
# Load model from MLflow Registry (STAGING)
# --------------------------------------------------
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

print(f"Loading model from: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)

print("✅ Model loaded successfully from STAGING")

# --------------------------------------------------
# Example input data for prediction
# --------------------------------------------------
sample_input = pd.DataFrame({
    "midterm_score": [67.0],
    "attendance": [68.0],
    "study_hours": [5.0],
    "no_of_projects": [2.0]
})

# --------------------------------------------------
# Predict
# --------------------------------------------------
prediction = model.predict(sample_input)

# --------------------------------------------------
# Convert numeric grade → letter grade
# --------------------------------------------------
grade_map = {
    0.0: "F",
    1.0: "D",
    2.0: "C",
    3.0: "B",
    4.0: "A"
}

predicted_grade = grade_map.get(float(prediction[0]), "Unknown")

print("🔮 Prediction Result")
print("-------------------")
print(f"Numeric Grade : {prediction[0]}")
print(f"Final Grade   : {predicted_grade}")
