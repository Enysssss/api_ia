from pydantic import BaseModel, Field


class NutritionInput(BaseModel):
    age: int = Field(..., gt=0, lt=120)
    poids_kg: float = Field(..., gt=20, lt=300)
    taille_cm: float = Field(..., gt=100, lt=250)
    taux_masse_grasse: float = Field(..., gt=1.0, lt=70.0)


class NutritionOutput(BaseModel):
    imc: float
    imc_categorie: str
    label: str
    confidence: float
    message: str
