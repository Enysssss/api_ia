"""
Script d'entraînement HealthAI Coach ML
=========================================
- Compare Random Forest, Gradient Boosting, XGBoost (si disponible)
- RandomizedSearchCV par modèle (n_iter=50, cv=5, f1_macro)
- MLflow : log params, métriques, modèle, artifacts
- Sauvegarde joblib du meilleur modèle
"""

import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline as SkPipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Chemins
ROOT       = Path(__file__).parent.parent.parent.parent
DATA_PATH  = ROOT / "ml" / "data" / "processed" / "healthai_dataset.csv"
MODELS_DIR = ROOT / "ml" / "models"
ARTIFACTS  = ROOT / "ml" / "artifacts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "ml"))

from src.preprocessing.cleaner  import clean
from src.preprocessing.engineer import engineer
from src.preprocessing.pipeline import (
    build_preprocessor, build_label_encoder, get_X_y, get_feature_names,
)


# ---------------------------------------------------------------------------
# MLflow config
# ---------------------------------------------------------------------------
MLFLOW_DB  = str(ROOT / "ml" / "artifacts" / "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("HealthAI_Coach")


# ---------------------------------------------------------------------------
# Hyperparamètres à rechercher
# ---------------------------------------------------------------------------
SEARCH_SPACES = {
    "RandomForest": {
        "clf__n_estimators":    [100, 200, 300, 500],
        "clf__max_depth":       [None, 5, 10, 15, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__class_weight":    ["balanced", None],
        "clf__max_features":    ["sqrt", "log2", None],
    },
    "GradientBoosting": {
        "clf__n_estimators":    [100, 150, 200],
        "clf__learning_rate":   [0.05, 0.1, 0.15, 0.2],
        "clf__max_depth":       [3, 4, 5, 6],
        "clf__subsample":       [0.7, 0.8, 0.9, 1.0],
        "clf__min_samples_leaf": [1, 2, 5],
    },
}

BASE_MODELS = {
    "RandomForest":     RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

try:
    from xgboost import XGBClassifier
    BASE_MODELS["XGBoost"] = XGBClassifier(
        random_state=42, eval_metric="mlogloss", verbosity=0,
    )
    SEARCH_SPACES["XGBoost"] = {
        "clf__n_estimators":  [100, 200, 300],
        "clf__max_depth":     [3, 4, 5, 6],
        "clf__learning_rate": [0.05, 0.1, 0.15, 0.2],
        "clf__subsample":     [0.7, 0.8, 0.9],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
    }
    print("[train] XGBoost détecté — inclus dans la comparaison")
except ImportError:
    print("[train] XGBoost non disponible — comparaison RF vs GB uniquement")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(cm, classes, name: str, run_id: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    path = ARTIFACTS / f"confusion_matrix_{name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def _save_feature_importance(model, feature_names: list[str], name: str):
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None

    preprocessor = model.named_steps["pre"]
    try:
        feat_names_out = preprocessor.get_feature_names_out()
    except Exception:
        feat_names_out = feature_names

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feat_names_out[i] for i in idx[:15]],
        importances[idx[:15]],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importances — {name} (Top 15)")
    ax.invert_yaxis()
    plt.tight_layout()
    path = ARTIFACTS / f"feature_importance_{name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def _save_learning_curve(model, X_train, y_train, name: str):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=5, scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), label="Train", color="steelblue")
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.2, color="steelblue")
    ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation", color="darkorange")
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.2, color="darkorange")
    ax.set_xlabel("Taille du jeu d'entraînement")
    ax.set_ylabel("F1-macro")
    ax.set_title(f"Learning Curve — {name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = ARTIFACTS / f"learning_curve_{name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preprocessor = build_preprocessor()

    for model_name, base_clf in BASE_MODELS.items():
        print(f"\n{'='*60}")
        print(f" Modèle : {model_name}")
        print(f"{'='*60}")

        pipe = SkPipeline([
            ("pre", preprocessor),
            ("clf", base_clf),
        ])

        search_space = SEARCH_SPACES.get(model_name, {})

        with mlflow.start_run(run_name=model_name):
            # RandomizedSearchCV
            search = RandomizedSearchCV(
                pipe,
                param_distributions=search_space,
                n_iter=50,
                cv=cv,
                scoring="f1_macro",
                random_state=42,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            # Métriques test
            y_pred = best_model.predict(X_test)
            y_pred_labels = label_encoder.inverse_transform(y_pred)
            y_test_labels  = label_encoder.inverse_transform(y_test)

            acc     = accuracy_score(y_test, y_pred)
            f1_mac  = f1_score(y_test, y_pred, average="macro")
            f1_wei  = f1_score(y_test, y_pred, average="weighted")

            # Cross-val sur train
            cv_scores = cross_val_score(best_model, X_train, y_train,
                                        cv=cv, scoring="f1_macro", n_jobs=-1)

            print(f"  Best params : {search.best_params_}")
            print(f"  Accuracy    : {acc:.4f}")
            print(f"  F1-macro    : {f1_mac:.4f}")
            print(f"  CV F1-macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"\n  Classification Report :")
            print(classification_report(y_test_labels, y_pred_labels))

            # MLflow log
            mlflow.log_params({k.replace("clf__", ""): v
                               for k, v in search.best_params_.items()})
            mlflow.log_metrics({
                "accuracy":       acc,
                "f1_macro":       f1_mac,
                "f1_weighted":    f1_wei,
                "cv_f1_macro":    cv_scores.mean(),
                "cv_f1_std":      cv_scores.std(),
            })

            # Artifacts
            cm = confusion_matrix(y_test, y_pred)
            cm_path = _save_confusion_matrix(cm, label_encoder.classes_, model_name, mlflow.active_run().info.run_id)
            fi_path = _save_feature_importance(best_model, get_feature_names(), model_name)
            lc_path = _save_learning_curve(best_model, X_train, y_train, model_name)

            for p in [cm_path, fi_path, lc_path]:
                if p:
                    mlflow.log_artifact(p)

            mlflow.sklearn.log_model(best_model, artifact_path="model")

            results[model_name] = {
                "model":    best_model,
                "f1_macro": f1_mac,
                "accuracy": acc,
                "cv_f1":    cv_scores.mean(),
                "report":   classification_report(y_test_labels, y_pred_labels, output_dict=True),
            }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[train] Chargement et préparation des données ...")
    df_raw = pd.read_csv(DATA_PATH)
    df = clean(df_raw.copy())
    df = engineer(df)

    X, y_str = get_X_y(df)
    label_encoder = build_label_encoder(y_str)
    y = label_encoder.transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train : {len(X_train)}  |  Test : {len(X_test)}")
    print(f"Classes : {list(label_encoder.classes_)}")

    results = train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder)

    # Meilleur modèle sur F1-macro
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best = results[best_name]
    print(f"\n{'='*60}")
    print(f" MEILLEUR MODÈLE : {best_name}")
    print(f" F1-macro  : {best['f1_macro']:.4f}")
    print(f" Accuracy  : {best['accuracy']:.4f}")
    print(f" CV F1     : {best['cv_f1']:.4f}")
    print(f"{'='*60}")

    # Sauvegarde
    joblib.dump(best["model"],  MODELS_DIR / "model.pkl")
    joblib.dump(label_encoder,  MODELS_DIR / "encoder.pkl")
    print(f"\n[train] Modèle sauvegardé dans {MODELS_DIR}/")

    # Tableau comparatif final
    print("\n Comparaison des modèles :")
    print(f"{'Modèle':<22} {'F1-macro':>10} {'Accuracy':>10} {'CV F1':>10}")
    print("-" * 56)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["f1_macro"]):
        marker = " ← SÉLECTIONNÉ" if name == best_name else ""
        print(f"{name:<22} {r['f1_macro']:>10.4f} {r['accuracy']:>10.4f} {r['cv_f1']:>10.4f}{marker}")

    _save_model_comparison(results, best_name)


def _save_model_comparison(results: dict, best_name: str):
    """Barplot comparatif F1-macro / Accuracy / CV F1 pour tous les modèles."""
    names   = list(results.keys())
    f1_mac  = [results[n]["f1_macro"] for n in names]
    acc     = [results[n]["accuracy"]  for n in names]
    cv_f1   = [results[n]["cv_f1"]     for n in names]

    x = np.arange(len(names))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, f1_mac, width, label="F1-macro (test)",  color=colors[0])
    bars2 = ax.bar(x,         acc,    width, label="Accuracy (test)",   color=colors[1])
    bars3 = ax.bar(x + width, cv_f1,  width, label="CV F1-macro (5-fold)", color=colors[2])

    # Valeurs au-dessus des barres
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    # Mettre en évidence le meilleur modèle
    best_idx = names.index(best_name)
    ax.axvspan(best_idx - 0.4, best_idx + 0.4, alpha=0.08, color="gold", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparaison des modèles — HealthAI Coach", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Badge "SÉLECTIONNÉ"
    ax.annotate("★ SÉLECTIONNÉ",
                xy=(best_idx, max(f1_mac[best_idx], acc[best_idx], cv_f1[best_idx]) + 0.06),
                ha="center", fontsize=10, color="goldenrod", fontweight="bold")

    plt.tight_layout()
    path = ARTIFACTS / "model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[train] Barplot comparatif sauvegardé : {path}")


if __name__ == "__main__":
    main()
