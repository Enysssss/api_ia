"""
Feature engineering : crée des features métier à partir des variables brutes.
Toutes les features dérivées sont explicables et défendables métier.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Catégories BMI (OMS)
# ---------------------------------------------------------------------------
BMI_BINS   = [0, 18.5, 25.0, 30.0, np.inf]
BMI_LABELS = ["sous_poids", "normal", "surpoids", "obesite"]


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=BMI_BINS, labels=BMI_LABELS, right=False
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Catégories masse grasse (ACSM 2022, stratifié genre)
# ---------------------------------------------------------------------------
def _fat_category(row: pd.Series) -> str:
    f = row["body_fat_pct"]
    m = row["gender"]
    if m == 1:  # homme
        if f < 6:    return "essentiel"
        if f < 14:   return "athlete"
        if f < 18:   return "fitness"
        if f < 25:   return "acceptable"
        return "obese"
    else:  # femme
        if f < 14:   return "essentiel"
        if f < 21:   return "athlete"
        if f < 25:   return "fitness"
        if f < 32:   return "acceptable"
        return "obese"


def add_fat_category(df: pd.DataFrame) -> pd.DataFrame:
    df["fat_category"] = df.apply(_fat_category, axis=1)
    return df


# ---------------------------------------------------------------------------
# Zone cardiaque au repos (indicateur de forme cardio-vasculaire)
# Bradycardie (<60) → forme excellente / Tachycardie (>100) → mauvaise forme
# ---------------------------------------------------------------------------
HR_BINS   = [0, 60, 70, 80, 90, np.inf]
HR_LABELS = ["excellent", "bon", "moyen", "sous_optimal", "mauvais"]


def add_hr_zone(df: pd.DataFrame) -> pd.DataFrame:
    df["hr_zone"] = pd.cut(
        df["resting_bpm"], bins=HR_BINS, labels=HR_LABELS, right=False
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Score de forme global (0–100) — composite normalisé
# Plus élevé = meilleure forme de base
# ---------------------------------------------------------------------------
def add_fitness_score(df: pd.DataFrame) -> pd.DataFrame:
    # Normalisation min-max sur les composantes
    bmi_score  = 1 - (df["bmi"] - 21).abs().clip(0, 15) / 15   # optimum ~21
    fat_norm   = df["body_fat_pct"]
    fat_score  = 1 - (fat_norm - fat_norm.quantile(0.25)).clip(0) / (fat_norm.max() - fat_norm.min())
    hr_score   = 1 - (df["resting_bpm"] - 50).clip(0, 50) / 50
    exp_score  = (df["experience_level"] - 1) / 2   # 0→1

    df["fitness_score"] = (
        0.30 * bmi_score +
        0.25 * fat_score +
        0.25 * hr_score  +
        0.20 * exp_score
    ).clip(0, 1).round(4) * 100
    return df


# ---------------------------------------------------------------------------
# Tranche d'âge
# ---------------------------------------------------------------------------
AGE_BINS   = [17, 25, 35, 45, 55, 66]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65"]


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df["age_group"] = pd.cut(
        df["age"], bins=AGE_BINS, labels=AGE_LABELS
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Interaction BMI × expérience (feature polynomiale métier)
# Capture la différence entre un débutant obèse et un athlète lourd
# ---------------------------------------------------------------------------
def add_bmi_exp_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df["bmi_x_exp"] = df["bmi"] * df["experience_level"]
    return df


# ---------------------------------------------------------------------------
# Rapport masse grasse / expérience
# Débutant avec beaucoup de graisse vs confirmé avec même graisse → besoins différents
# ---------------------------------------------------------------------------
def add_fat_exp_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["fat_per_exp"] = (df["body_fat_pct"] / df["experience_level"]).round(3)
    return df


# ---------------------------------------------------------------------------
# Pipeline complet feature engineering
# ---------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    print("[engineer] Feature engineering ...")
    df = add_bmi_category(df)
    df = add_fat_category(df)
    df = add_hr_zone(df)
    df = add_fitness_score(df)
    df = add_age_group(df)
    df = add_bmi_exp_interaction(df)
    df = add_fat_exp_ratio(df)
    feats = ["bmi_category", "fat_category", "hr_zone", "fitness_score",
             "age_group", "bmi_x_exp", "fat_per_exp"]
    print(f"[engineer] {len(feats)} features créées : {feats}")
    return df
