"""
Évaluation post-training du meilleur modèle sauvegardé.
-------------------------------------------------------
- Charge model.pkl + encoder.pkl
- Recharge le dataset, applique le même pipeline
- Génère : rapport de classification, matrice de confusion, rapport JSON
- Usage : python -m ml.src.evaluation.evaluate
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
)

ROOT       = Path(__file__).parent.parent.parent.parent
DATA_PATH  = ROOT / "ml" / "data" / "processed" / "healthai_dataset.csv"
MODELS_DIR = ROOT / "ml" / "models"
ARTIFACTS  = ROOT / "ml" / "artifacts"

sys.path.insert(0, str(ROOT / "ml"))
from src.preprocessing.cleaner  import clean
from src.preprocessing.engineer import engineer
from src.preprocessing.pipeline import build_label_encoder, get_X_y


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = clean(df.copy())
    df = engineer(df)
    return df


def main():
    print("=" * 60)
    print(" ÉVALUATION DU MODÈLE — HealthAI Coach")
    print("=" * 60)

    df = load_data()
    X, y_str = get_X_y(df)

    model   = joblib.load(MODELS_DIR / "model.pkl")
    encoder = joblib.load(MODELS_DIR / "encoder.pkl")
    y       = encoder.transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    y_test_labels = encoder.inverse_transform(y_test)
    y_pred_labels = encoder.inverse_transform(y_pred)

    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_wei = f1_score(y_test, y_pred, average="weighted")

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="f1_macro", n_jobs=-1)

    print(f"\n  Accuracy          : {acc:.4f}")
    print(f"  F1-macro (test)   : {f1_mac:.4f}")
    print(f"  F1-weighted (test): {f1_wei:.4f}")
    if auc:
        print(f"  AUC-ROC (macro)   : {auc:.4f}")
    print(f"  CV F1-macro       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Classification Report :")
    report = classification_report(y_test_labels, y_pred_labels)
    print(report)

    # Sauvegarde rapport JSON
    report_dict = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    summary = {
        "accuracy":    round(acc,    4),
        "f1_macro":    round(f1_mac, 4),
        "f1_weighted": round(f1_wei, 4),
        "auc_roc":     round(auc, 4) if auc else None,
        "cv_f1_macro": round(float(cv_scores.mean()), 4),
        "cv_f1_std":   round(float(cv_scores.std()),  4),
        "per_class":   {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall":    round(report_dict[cls]["recall"],    4),
                "f1-score":  round(report_dict[cls]["f1-score"],  4),
                "support":   int(report_dict[cls]["support"]),
            }
            for cls in encoder.classes_
        },
    }

    report_path = ARTIFACTS / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[evaluate] Rapport JSON : {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=35)
    ax.set_title("Matrice de confusion — meilleur modèle (test set)", fontsize=13)
    plt.tight_layout()
    cm_path = ARTIFACTS / "confusion_matrix_best.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[evaluate] Matrice de confusion : {cm_path}")

    # Barplot support par classe
    classes   = list(summary["per_class"].keys())
    f1_scores = [summary["per_class"][c]["f1-score"] for c in classes]
    supports  = [summary["per_class"][c]["support"]  for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.RdYlGn([s / max(f1_scores) for s in f1_scores])
    bars = ax1.barh(classes, f1_scores, color=colors)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("F1-score")
    ax1.set_title("F1-score par classe")
    ax1.axvline(f1_mac, color="navy", linestyle="--", label=f"F1-macro={f1_mac:.3f}")
    ax1.legend()
    for bar, val in zip(bars, f1_scores):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

    ax2.barh(classes, supports, color="#4C72B0")
    ax2.set_xlabel("Nombre d'échantillons (test)")
    ax2.set_title("Distribution du test set par classe")
    for i, val in enumerate(supports):
        ax2.text(val + 0.3, i, str(val), va="center", fontsize=9)

    plt.suptitle("Analyse par classe — meilleur modèle", fontsize=14, fontweight="bold")
    plt.tight_layout()
    per_class_path = ARTIFACTS / "per_class_analysis.png"
    plt.savefig(per_class_path, dpi=150)
    plt.close()
    print(f"[evaluate] Analyse par classe : {per_class_path}")

    print("\n[evaluate] Évaluation terminée.")
    return summary


if __name__ == "__main__":
    main()
