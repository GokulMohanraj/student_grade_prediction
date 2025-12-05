# model_training.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import os

# --------------------------------------------------------
# 1️⃣ Get project root (outside training/src)
# --------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MLRUNS_PATH = os.path.join(PROJECT_ROOT, "mlruns")

# Fix Windows path format for MLflow
mlflow.set_tracking_uri("file:///" + MLRUNS_PATH.replace("\\", "/"))

# Set experiment
mlflow.set_experiment("Student_Grade_Prediction")

# --------------------------------------------------------
# 2️⃣ Ensure models folder exists
# --------------------------------------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------------
# 3️⃣ Load processed data
# --------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))['grade']
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))['grade']

print("Data Loaded Successfully!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# --------------------------------------------------------
# 4️⃣ MLflow Run
# --------------------------------------------------------
with mlflow.start_run():

    # Model parameters
    n_estimators = 100
    max_depth = None
    random_state = 42

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # Train
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    print(f"\nAccuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Log model to MLflow

    # Use a small sample of your training data as example
    X_train_float = X_train.astype(float)

    mlflow.sklearn.log_model(rf_model, name="random_forest_model", input_example=X_train_float.head(2))

    # Save model locally
    MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_grade_model.pkl")
    joblib.dump(rf_model, MODEL_PATH)
    print(f"\nModel saved locally at {MODEL_PATH}")

print("\nMLflow run completed!")
print(f"MLruns stored at: {MLRUNS_PATH}")
