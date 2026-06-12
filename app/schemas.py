from pydantic import BaseModel, Field, model_validator


class RecommendInput(BaseModel):
    age: int = Field(..., ge=18, le=65, description="Âge en années (18-65)")
    gender: str = Field(..., pattern="^(male|female)$", description="Genre : 'male' ou 'female'")
    weight_kg: float = Field(..., gt=30, lt=200, description="Poids en kg")
    height_cm: float = Field(..., gt=140, lt=215, description="Taille en cm")
    body_fat_pct: float = Field(..., ge=4, le=55, description="Taux de masse grasse (%)")
    resting_bpm: int = Field(..., ge=40, le=105, description="Fréquence cardiaque au repos (bpm)")
    experience_level: str = Field(
        ...,
        pattern="^(beginner|intermediate|advanced)$",
        description="Niveau d'expérience sportive",
    )

    @model_validator(mode="after")
    def check_bmi(self):
        bmi = self.weight_kg / (self.height_cm / 100) ** 2
        if bmi < 14 or bmi > 48:
            raise ValueError(
                f"IMC calculé ({bmi:.1f}) hors plage acceptable (14-48). "
                "Vérifiez poids_kg et taille_cm."
            )
        return self

    def gender_encoded(self) -> int:
        return 1 if self.gender == "male" else 0

    def experience_encoded(self) -> int:
        return {"beginner": 1, "intermediate": 2, "advanced": 3}[self.experience_level]


class ProgramOutput(BaseModel):
    sessions_per_week: int
    session_duration_min: int
    focus: str
    activities: list[str]
    intensity: str
    weekly_volume_h: float
    progression: str
    nutrition_tip: str
    objective: str


class RecommendOutput(BaseModel):
    profile: str
    confidence: float
    bmi: float
    bmi_category: str
    program: ProgramOutput


# Rétrocompatibilité avec l'ancien endpoint /nutrition/predict
class NutritionInput(BaseModel):
    age: int = Field(..., ge=18, le=65)
    poids_kg: float = Field(..., gt=20, lt=300)
    taille_cm: float = Field(..., gt=100, lt=250)
    taux_masse_grasse: float = Field(..., ge=4.0, le=55.0)


class NutritionOutput(BaseModel):
    imc: float
    imc_categorie: str
    label: str
    confidence: float
    message: str
