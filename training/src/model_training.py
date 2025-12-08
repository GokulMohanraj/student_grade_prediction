# training/src/model_training.py
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MLRUNS_PATH = os.path.join(PROJECT_ROOT, "mlruns")

os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri("file:///" + MLRUNS_PATH.replace("\\", "/"))
experiment_name = f"Student_Grade_prediction"
mlflow.set_experiment(experiment_name)

# -----------------------------
# Training function
# -----------------------------
def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train model, log to MLflow, save model, return metrics + run_id
    """
    run_name = f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    with mlflow.start_run(run_name=run_name) as run:

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(report)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(
            model,
            name=model_name,
            input_example=X_train.head(2)
        )

        # Save locally
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)

        print(f"Model saved at {model_path}")

        run_id = run.info.run_id   # ✅ SAFE

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "model_object": model,
        "run_id": run_id           # ✅ RETURN RUN ID
    }
