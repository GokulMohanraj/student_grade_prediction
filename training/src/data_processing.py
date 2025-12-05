import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# --------------------------------------------------------
# Project root and data directories
# --------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --------------------------------------------------------
# Load raw data
# --------------------------------------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "student_data.csv"))
print("Raw data shape:", df.shape)

# --------------------------------------------------------
# Encode target 'grade' to numeric
# --------------------------------------------------------
grade_mapping = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}
df['grade_encoded'] = df['grade'].map(grade_mapping)
print("Grade mapping used:", grade_mapping)

# --------------------------------------------------------
# Features and target
# --------------------------------------------------------
X = df[['midterm_score', 'attendance', 'study_hours', 'no_of_projects']]
y = df['grade_encoded']

# --------------------------------------------------------
# Feature scaling
# --------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert features to float to avoid MLflow warnings
X_scaled = X_scaled.astype(float)

# --------------------------------------------------------
# Train-test split
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------------
# Convert target to float
# --------------------------------------------------------
y_train_df = pd.DataFrame({'grade': y_train.astype(float)})
y_test_df = pd.DataFrame({'grade': y_test.astype(float)})

# --------------------------------------------------------
# Save processed features and target
# --------------------------------------------------------
X_train_path = os.path.join(PROCESSED_DIR, "X_train.csv")
X_test_path = os.path.join(PROCESSED_DIR, "X_test.csv")
y_train_path = os.path.join(PROCESSED_DIR, "y_train.csv")
y_test_path = os.path.join(PROCESSED_DIR, "y_test.csv")

pd.DataFrame(X_train, columns=X.columns).to_csv(X_train_path, index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv(X_test_path, index=False)
y_train_df.to_csv(y_train_path, index=False)
y_test_df.to_csv(y_test_path, index=False)

# --------------------------------------------------------
# Print summary
# --------------------------------------------------------
print("Processed data saved successfully!")
print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

