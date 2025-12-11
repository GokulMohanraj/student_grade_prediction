from pydantic import BaseModel

class StudentInput(BaseModel):
    midterm_score: float
    attendance: float
    study_hours: float
    no_of_projects: float


class PredictionOutput(BaseModel):
    predicted_grade: str
