import mlflow
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MLRUNS_PATH = os.path.join(PROJECT_ROOT, "mlruns")

mlflow.set_tracking_uri("file:///" + MLRUNS_PATH.replace("\\", "/"))

MODEL_NAME = "Student_Grade_Model"
MODEL_STAGE = "Staging"

def load_model():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])
    preds = model.predict(df)
    return int(preds[0])
