import mlflow

def register_best_model(run_id, model_name):
    """
    Register the best model from MLflow run
    """

    model_uri = f"runs:/{run_id}/{model_name}"

    result = mlflow.register_model(
        model_uri=model_uri,
        name="Student_Grade_Model"
    )

    print(f"✅ Model registered successfully")
    print(f"Model name: Student_Grade_Model")
    print(f"Version: {result.version}")
