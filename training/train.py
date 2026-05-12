# training/train.py
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import your pipeline modules
from src.data_processing import process_data
from src.model_training import train_model
from src.register_model import register_best_model
from src.promote_model import rotate_model_lifecycle
from src.config import MLFLOW_TRACKING_URI 


# --------------------------------------------------
# MLflow Configuration (Environment Based)
# --------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Student_Grade_prediction_v2")


def main():

    # --------------------------------------------------
    # Load Processed Data (No manual CSV loading)
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = process_data()

    # --------------------------------------------------
    # Define Models
    # --------------------------------------------------
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=500
        ),
    }

    best_accuracy = 0
    best_model_name = None
    best_run_id = None
    # --------------------------------------------------
    # Train All Models
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Print Best Model Info
    # --------------------------------------------------
    print("\n==============================")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📊 Best Accuracy: {best_accuracy:.3f}")
    print(f"🆔 Run ID: {best_run_id}")
    print("==============================\n")

    # --------------------------------------------------
    # Register Best Model
    # --------------------------------------------------
    registered_version = register_best_model(
        run_id=best_run_id,
        model_name=best_model_name
    )

    print("✅ Model registered successfully.")

    # --------------------------------------------------
    # Move to Staging
    # --------------------------------------------------
    rotate_model_lifecycle(registered_version)

    print("🚀 Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
