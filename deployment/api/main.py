from fastapi import FastAPI
from deployment.src.model_loader import load_model, predict as model_predict
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


@app.post("/predict", response_model=PredictionOutput)
def predict_grade(data: StudentInput):
    prediction = model_predict(
        model,
        {
            "midterm_score": data.midterm_score,
            "attendance": data.attendance,
            "study_hours": data.study_hours,
            "no_of_projects": data.no_of_projects
        }
    )
    return {"predicted_grade": GRADE_MAP[int(prediction)]}
