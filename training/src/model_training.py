# training/src/model_training.py

import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
# from common.config import MODEL_DIR

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(model, X_train, y_train, X_test, y_test, model_name):

    run_name = f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"

    with mlflow.start_run(run_name=run_name) as run:

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(report)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(model.get_params())

        mlflow.sklearn.log_model(
            model,
            name=model_name,
            input_example=X_train.head(2)
        )

        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)

        print(f"Model saved at {model_path}")

        run_id = run.info.run_id

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "model_object": model,
        "run_id": run_id
    }
