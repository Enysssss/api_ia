"""
Nettoyage du dataset : doublons, valeurs manquantes, outliers.
"""

import pandas as pd
import numpy as np


FEATURE_BOUNDS = {
    "age":            (18, 65),
    "weight_kg":      (30, 200),
    "height_cm":      (145, 215),
    "bmi":            (14, 48),
    "body_fat_pct":   (4,  52),
    "resting_bpm":    (40, 105),
    "experience_level": (1, 3),
}


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates()
    dropped = n_before - len(df)
    if dropped:
        print(f"  [cleaner] Doublons supprimés : {dropped}")
    return df.reset_index(drop=True)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    if missing.any():
        print(f"  [cleaner] Valeurs manquantes :\n{missing[missing > 0]}")
        # Imputation médiane pour les numériques
        for col in df.select_dtypes(include="number").columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        # Imputation mode pour les catégorielles
        for col in df.select_dtypes(exclude="number").columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list[str], factor: float = 3.0) -> pd.DataFrame:
    """Supprime les outliers extrêmes (factor=3 → outliers vraiment impossibles)."""
    n_before = len(df)
    mask = pd.Series([True] * len(df))
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask &= df[col].between(lower, upper)
    df = df[mask].reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped:
        print(f"  [cleaner] Outliers IQR supprimés : {dropped}")
    return df


def clip_physiological_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip sur les bornes physiologiques (valeurs impossibles)."""
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[cleaner] Début nettoyage ...")
    df = remove_duplicates(df)
    df = handle_missing(df)
    numeric_cols = [c for c in FEATURE_BOUNDS if c in df.columns]
    df = remove_outliers_iqr(df, numeric_cols, factor=3.0)
    df = clip_physiological_bounds(df)
    print(f"[cleaner] Dataset propre : {len(df)} lignes")
    return df
