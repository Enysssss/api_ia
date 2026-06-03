import joblib
import numpy as np

from app.schemas import NutritionInput, NutritionOutput

_MESSAGES = {
    "perte_poids": "Privilégiez un déficit calorique modéré et augmentez votre activité physique.",
    "maintien": "Continuez sur votre lancée : alimentation équilibrée et activité régulière.",
    "prise_masse": "Augmentez votre apport calorique avec des protéines de qualité et un entraînement en résistance.",
}

_IMC_CATEGORIES = [
    (18.5, "Sous-poids"),
    (25.0, "Normal"),
    (30.0, "Surpoids"),
    (float("inf"), "Obésité"),
]


def _imc_categorie(imc: float) -> str:
    for threshold, label in _IMC_CATEGORIES:
        if imc < threshold:
            return label
    return "Obésité"


class NutritionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = joblib.load("models/model.pkl")
            cls._instance._encoder = joblib.load("models/encoder.pkl")
        return cls._instance

    def predict(self, data: NutritionInput) -> NutritionOutput:
        imc = data.poids_kg / (data.taille_cm / 100) ** 2
        X = [[imc, data.taux_masse_grasse, data.age]]

        encoded_pred = self._model.predict(X)[0]
        label = self._encoder.inverse_transform([encoded_pred])[0]
        confidence = float(np.max(self._model.predict_proba(X)))

        return NutritionOutput(
            imc=round(imc, 2),
            imc_categorie=_imc_categorie(imc),
            label=label,
            confidence=round(confidence, 4),
            message=_MESSAGES[label],
        )
