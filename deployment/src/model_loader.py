import mlflow
import pandas as pd
from common.config import MODEL_NAME, MLFLOW_TRACKING_URI

print("Tracking URI:", mlflow.get_tracking_uri())

# --------------------------------------------------
# Configure MLflow Tracking URI
# --------------------------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --------------------------------------------------
# Always load PRODUCTION alias
# --------------------------------------------------
MODEL_ALIAS = "production"


def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])
    preds = model.predict(df)
    return int(preds[0])
