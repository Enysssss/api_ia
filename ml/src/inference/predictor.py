"""
Classe d'inférence — utilisée par l'API FastAPI.
Charge le modèle une seule fois (singleton) et expose predict().
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.preprocessing.engineer import engineer
from ml.src.preprocessing.pipeline import get_feature_names

MODELS_DIR = Path(__file__).parent.parent.parent.parent / "ml" / "models"


class FitnessPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model   = joblib.load(MODELS_DIR / "model.pkl")
            cls._instance._encoder = joblib.load(MODELS_DIR / "encoder.pkl")
        return cls._instance

    def predict(self, age: int, gender: int, weight_kg: float, height_cm: float,
                body_fat_pct: float, resting_bpm: int, experience_level: int
                ) -> tuple[str, float]:
        """
        Retourne (profile_label, confidence).
        """
        bmi = weight_kg / (height_cm / 100) ** 2

        row = pd.DataFrame([{
            "age":             age,
            "gender":          gender,
            "weight_kg":       weight_kg,
            "height_cm":       height_cm,
            "bmi":             round(bmi, 2),
            "body_fat_pct":    body_fat_pct,
            "resting_bpm":     resting_bpm,
            "experience_level": experience_level,
        }])

        row = engineer(row)
        X = row[get_feature_names()]

        encoded = self._model.predict(X)[0]
        profile = self._encoder.inverse_transform([encoded])[0]
        proba   = float(np.max(self._model.predict_proba(X)))

        return profile, round(proba, 4)
