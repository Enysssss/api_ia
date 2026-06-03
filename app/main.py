from fastapi import FastAPI

from app.schemas import NutritionInput, NutritionOutput
from app.service import NutritionService

app = FastAPI(title="API ML Recommandation Fitness")
service = NutritionService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/nutrition/predict", response_model=NutritionOutput)
def predict(data: NutritionInput) -> NutritionOutput:
    return service.predict(data)
