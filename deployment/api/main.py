from fastapi import FastAPI
from deployment.src.model_loader import load_model, predict
from deployment.src.schemas import StudentInput, PredictionOutput

app = FastAPI(
    title="Student Grade Prediction API",
    version="1.0"
)

model = load_model()

GRADE_MAP = {0: "F", 1: "D", 2: "C", 3: "B", 4: "A"}

@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: StudentInput):
    prediction = model.predict([[ 
        data.midterm_score,
        data.attendance,
        data.study_hours,
        data.no_of_projects
    ]])

    grade_map = {0: "F", 1: "D", 2: "C", 3: "B", 4: "A"}

    return {"grade": grade_map[int(prediction[0])]}
