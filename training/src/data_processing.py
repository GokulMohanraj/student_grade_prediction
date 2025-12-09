import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def create_processed_data():
    """Reads raw CSV, preprocesses, and saves processed CSVs."""
    df = pd.read_csv(os.path.join(RAW_DIR, "student_data.csv"))
    print("Raw data shape:", df.shape)

    grade_map = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}
    df["grade_encoded"] = df["grade"].map(grade_map)

    X = df[["midterm_score", "attendance", "study_hours", "no_of_projects"]].astype(float)   
    y = df["grade_encoded"].astype(float)

    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pd.DataFrame(X_train, columns=X.columns).to_csv(
        os.path.join(PROCESSED_DIR, "X_train.csv"), index=False
    )
    pd.DataFrame(X_test, columns=X.columns).to_csv(
        os.path.join(PROCESSED_DIR, "X_test.csv"), index=False
    )
    pd.DataFrame({"grade": y_train}).to_csv(
        os.path.join(PROCESSED_DIR, "y_train.csv"), index=False
    )
    pd.DataFrame({"grade": y_test}).to_csv(
        os.path.join(PROCESSED_DIR, "y_test.csv"), index=False
    )

    print("Processed data created successfully.")


def process_data():
    """Loads already processed CSVs (used by main.py)."""
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["grade"]
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["grade"]

    return X_train, X_test, y_train, y_test
