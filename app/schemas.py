from pydantic import BaseModel, Field, model_validator


class NutritionInput(BaseModel):
    age: int = Field(..., ge=18, le=59,
        description="Âge (18-59 — range du dataset NHANES)")
    poids_kg: float = Field(..., gt=20, lt=300)
    taille_cm: float = Field(..., gt=100, lt=250)
    taux_masse_grasse: float = Field(..., ge=12.0, le=57.0,
        description="% masse grasse (12-57 — range du dataset NHANES)")

    @model_validator(mode="after")
    def check_imc_range(self):
        imc = self.poids_kg / (self.taille_cm / 100) ** 2
        if imc < 15 or imc > 64:
            raise ValueError(
                f"IMC calculé ({imc:.1f}) hors du range du dataset d'entraînement (15-64). "
                "Vérifiez poids_kg et taille_cm."
            )
        return self


class NutritionOutput(BaseModel):
    imc: float
    imc_categorie: str
    label: str
    confidence: float
    message: str
