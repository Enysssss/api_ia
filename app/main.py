from fastapi import FastAPI

from app.schemas import RecommendInput, RecommendOutput, NutritionInput, NutritionOutput
from app.service import FitnessService

app = FastAPI(
    title="HealthAI Coach — API ML",
    description="Recommandation de programmes sportifs personnalisés par Machine Learning.",
    version="2.0.0",
)

service = FitnessService()


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/recommend", response_model=RecommendOutput)
def recommend(data: RecommendInput) -> RecommendOutput:
    """
    Prédit le profil fitness de l'utilisateur et retourne un programme sportif personnalisé.
    """
    return service.recommend(data)


@app.get("/profiles")
def list_profiles():
    """Liste les profils fitness disponibles."""
    from ml.src.recommendation_engine.engine import list_profiles, _PROGRAMS
    return {
        "profiles": [
            {
                "id": p,
                "focus": _PROGRAMS[p].focus,
                "sessions_per_week": _PROGRAMS[p].sessions_per_week,
            }
            for p in list_profiles()
        ]
    }


# Rétrocompatibilité
@app.post("/nutrition/predict", response_model=NutritionOutput, deprecated=True)
def predict_legacy(data: NutritionInput) -> NutritionOutput:
    """Ancien endpoint — utiliser /recommend à la place."""
    return service.predict_legacy(data)
