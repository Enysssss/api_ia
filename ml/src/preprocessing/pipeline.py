"""
Pipeline sklearn complet : encodage + normalisation.
Séparé du feature engineering (qui reste en pandas) pour faciliter l'inférence.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder


# Features numériques brutes
NUM_FEATURES = [
    "age", "weight_kg", "height_cm", "bmi", "body_fat_pct",
    "resting_bpm", "experience_level", "fitness_score",
    "bmi_x_exp", "fat_per_exp",
]

# Features catégorielles ordinales (ordre défini)
ORD_FEATURES = {
    "bmi_category":  ["sous_poids", "normal", "surpoids", "obesite"],
    "fat_category":  ["essentiel", "athlete", "fitness", "acceptable", "obese"],
    "hr_zone":       ["excellent", "bon", "moyen", "sous_optimal", "mauvais"],
    "age_group":     ["18-25", "26-35", "36-45", "46-55", "56-65"],
}

# Features binaires (gender)
BIN_FEATURES = ["gender"]


def get_feature_names() -> list[str]:
    return NUM_FEATURES + list(ORD_FEATURES.keys()) + BIN_FEATURES


def build_preprocessor() -> ColumnTransformer:
    ord_encoder = OrdinalEncoder(
        categories=list(ORD_FEATURES.values()),
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("ord", ord_encoder, list(ORD_FEATURES.keys())),
            ("bin", "passthrough", BIN_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_label_encoder(y: pd.Series) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(y)
    return le


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[get_feature_names()].copy()
    y = df["fitness_profile"].copy()
    return X, y
