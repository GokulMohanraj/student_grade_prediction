import os

# --------------------------------------------------
# Project Paths
# --------------------------------------------------
# config.py is inside common/
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MLRUNS_PATH = os.path.join(PROJECT_ROOT, "mlruns")

MLFLOW_TRACKING_URI = "file:///" + MLRUNS_PATH.replace("\\", "/")

MODEL_NAME = "Student_Grade_Model"

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------
LOCAL_MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")

# MLFLOW_TRACKING_URI = os.getenv(
#     "MLFLOW_TRACKING_URI",
#     "file:///" + LOCAL_MLRUNS_DIR.replace("\\", "/")
# )
# MODEL_NAME = os.getenv("MODEL_NAME", "Student_Grade_Model")
# --------------------------------------------------
# Promotion Settings
# --------------------------------------------------
ACCURACY_THRESHOLD = float(
    os.getenv("ACCURACY_THRESHOLD", 0.85)
)


print("✅ Configuration Loaded:")
print(f"   MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
print(f"   MODEL_NAME: {MODEL_NAME}")
print(f"   ACCURACY_THRESHOLD: {ACCURACY_THRESHOLD}")

print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
print(f"   MODEL_DIR: {MODEL_DIR}")
print(f"   MLRUNS_DIR: {MLRUNS_DIR}")
