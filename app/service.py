import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from app.schemas import RecommendInput, RecommendOutput, ProgramOutput
from app.schemas import NutritionInput, NutritionOutput

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "ml" / "models"

sys_path_appended = False
try:
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ml"))
    from ml.src.preprocessing.engineer import engineer
    from ml.src.preprocessing.pipeline import get_feature_names
    from ml.src.recommendation_engine.engine import get_program, program_to_dict
    sys_path_appended = True
except Exception as e:
    sys_path_appended = False
    _IMPORT_ERROR = str(e)


_BMI_CATEGORIES = [
    (18.5, "Sous-poids"),
    (25.0, "Normal"),
    (30.0, "Surpoids"),
    (float("inf"), "Obésité"),
]

_LEGACY_MESSAGES = {
    "perte_poids": "Privilégiez un déficit calorique modéré et augmentez votre activité physique.",
    "maintien":    "Continuez sur votre lancée : alimentation équilibrée et activité régulière.",
    "prise_masse": "Augmentez votre apport calorique avec des protéines de qualité et un entraînement en résistance.",
}


def _bmi_category(bmi: float) -> str:
    for threshold, label in _BMI_CATEGORIES:
        if bmi < threshold:
            return label
    return "Obésité"


class FitnessService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model   = joblib.load(MODELS_DIR / "model.pkl")
            cls._instance._encoder = joblib.load(MODELS_DIR / "encoder.pkl")
        return cls._instance

    def recommend(self, data: RecommendInput) -> RecommendOutput:
        if not sys_path_appended:
            raise RuntimeError(f"Import ML modules failed: {_IMPORT_ERROR}")

        bmi = data.weight_kg / (data.height_cm / 100) ** 2

        row = pd.DataFrame([{
            "age":              data.age,
            "gender":           data.gender_encoded(),
            "weight_kg":        data.weight_kg,
            "height_cm":        data.height_cm,
            "bmi":              round(bmi, 2),
            "body_fat_pct":     data.body_fat_pct,
            "resting_bpm":      data.resting_bpm,
            "experience_level": data.experience_encoded(),
        }])

        row = engineer(row)
        X = row[get_feature_names()]

        encoded   = self._model.predict(X)[0]
        profile   = self._encoder.inverse_transform([encoded])[0]
        confidence = float(np.max(self._model.predict_proba(X)))

        prog = get_program(profile)

        return RecommendOutput(
            profile=profile,
            confidence=round(confidence, 4),
            bmi=round(bmi, 2),
            bmi_category=_bmi_category(bmi),
            program=ProgramOutput(**program_to_dict(prog)),
        )

    def predict_legacy(self, data: NutritionInput) -> NutritionOutput:
        """Rétrocompatibilité avec l'ancien endpoint /nutrition/predict."""
        imc = data.poids_kg / (data.taille_cm / 100) ** 2

        # Mapping simple vers les 3 anciens labels via règle
        if imc >= 27.5:
            label = "perte_poids"
        elif imc < 22:
            label = "prise_masse"
        else:
            label = "maintien"

        return NutritionOutput(
            imc=round(imc, 2),
            imc_categorie=_bmi_category(imc),
            label=label,
            confidence=0.0,
            message=_LEGACY_MESSAGES[label],
        )
