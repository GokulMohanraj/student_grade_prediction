# training/main.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.model_training import train_model
from src.register_model import register_best_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")

# Load data
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["grade"]
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))["grade"]

models = {
    "RandomForest": RandomForestClassifier(n_estimators=10, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=300),
}

best_accuracy = 0
best_model_name = ""
best_run_id = None

for name, model in models.items():
    result = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name=name
    )

    if result["accuracy"] > best_accuracy:
        best_accuracy = result["accuracy"]
        best_model_name = name
        best_run_id = result["run_id"]
print("\n====================")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.3f}")
print(f"Run ID: {best_run_id}")
print("====================")

# Register the best model
register_best_model(
    run_id=best_run_id,
    model_name=best_model_name
)