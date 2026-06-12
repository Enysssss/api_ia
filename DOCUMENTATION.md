# HealthAI Coach — Documentation Technique Complète

> Plateforme de coaching santé par Machine Learning  
> MSPR CDA / Développeur IA

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Structure du projet](#2-structure-du-projet)
3. [Pipeline Machine Learning](#3-pipeline-machine-learning)
   - [Dataset](#31-dataset)
   - [Nettoyage](#32-nettoyage)
   - [Feature Engineering](#33-feature-engineering)
   - [Preprocessing sklearn](#34-preprocessing-sklearn)
   - [Modélisation](#35-modélisation)
   - [Évaluation](#36-évaluation)
   - [Explicabilité](#37-explicabilité)
   - [MLflow](#38-mlflow)
4. [Moteur de recommandation](#4-moteur-de-recommandation)
5. [API FastAPI](#5-api-fastapi)
   - [Endpoints](#51-endpoints)
   - [Schémas de données](#52-schémas-de-données)
   - [Architecture interne](#53-architecture-interne)
6. [Guide de démarrage](#6-guide-de-démarrage)
7. [Décisions techniques](#7-décisions-techniques)
8. [Défense orale — points clés](#8-défense-orale--points-clés)

---

## 1. Vue d'ensemble

**HealthAI Coach** est une plateforme de coaching santé qui utilise le Machine Learning pour recommander des programmes sportifs personnalisés.

### Flux principal

```
Utilisateur (données biométriques)
        ↓
   POST /recommend
        ↓
   FitnessService
        ↓
   Feature Engineering   ← créer les features dérivées
        ↓
   Modèle ML (Random Forest / GBM / XGBoost)
        ↓
   Profil fitness prédit (6 classes)
        ↓
   Moteur de recommandation (règles ACSM)
        ↓
   Programme sportif détaillé (JSON)
```

### Choix de conception — séparation ML / moteur de règles

Le modèle ML **prédit un profil**, pas un programme. Le programme est généré par un **moteur de règles métier** indépendant. Cette architecture est intentionnelle :

- Le modèle reste interprétable (6 classes nommées)
- Le programme peut être modifié sans réentraîner le modèle
- En production, les programmes peuvent être enrichis par des experts

---

## 2. Structure du projet

```
2MPR/
│
├── app/                            # API FastAPI
│   ├── __init__.py
│   ├── main.py                     # Définition des routes
│   ├── schemas.py                  # Modèles Pydantic (input/output)
│   └── service.py                  # Logique métier, singleton ML
│
├── ml/                             # Pipeline Machine Learning complet
│   │
│   ├── data/
│   │   ├── generate_dataset.py     # Génère le dataset synthétique
│   │   ├── processed/
│   │   │   └── healthai_dataset.csv   # Dataset généré (2000 lignes)
│   │   └── raw/                    # Données brutes (vide, pour futurs datasets)
│   │
│   ├── notebooks/
│   │   └── HealthAI_Coach_ML.ipynb # Notebook complet (EDA → Interprétation)
│   │
│   ├── src/
│   │   ├── preprocessing/
│   │   │   ├── cleaner.py          # Nettoyage : doublons, NaN, outliers
│   │   │   ├── engineer.py         # Feature engineering métier
│   │   │   └── pipeline.py         # Pipeline sklearn (encodage + normalisation)
│   │   │
│   │   ├── training/
│   │   │   └── train.py            # Entraînement : 3 modèles + RandomSearch + MLflow
│   │   │
│   │   ├── evaluation/             # (extensible : métriques personnalisées)
│   │   │
│   │   ├── inference/
│   │   │   └── predictor.py        # Classe d'inférence pour l'API
│   │   │
│   │   └── recommendation_engine/
│   │       └── engine.py           # Profil → Programme sportif (règles ACSM)
│   │
│   ├── models/
│   │   ├── model.pkl               # Meilleur modèle sérialisé (joblib)
│   │   └── encoder.pkl             # LabelEncoder des 6 profils
│   │
│   ├── tests/                      # Tests unitaires (à compléter)
│   │
│   └── artifacts/
│       ├── mlflow.db               # Base SQLite MLflow (runs, params, métriques)
│       ├── confusion_matrix_*.png  # Matrices de confusion par modèle
│       ├── feature_importance_*.png
│       └── learning_curve_*.png
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── DOCUMENTATION.md
```

---

## 3. Pipeline Machine Learning

### 3.1 Dataset

**Fichier :** `ml/data/generate_dataset.py`

#### Pourquoi un dataset synthétique ?

Les datasets publics disponibles (NHANES, Gym Members Exercise Dataset) ont des problèmes rédhibitoires pour ce cas d'usage :

| Dataset | Problème |
|---|---|
| NHANES 2017-2018 | Le label est une **auto-perception** ("pensez-vous être en surpoids ?"), pas une prescription |
| Gym Members Exercise Dataset | Les features `Avg_BPM`, `Calories_Burned`, `Session_Duration` sont mesurées **pendant l'entraînement** — leakage temporel |

**Solution :** Dataset synthétique basé sur les **distributions cliniques publiées** (NHANES, ACSM, OMS).

#### Structure du dataset

```
2000 lignes × 9 colonnes

Features (disponibles à l'inscription) :
  age               int     [18, 65]      Distribution uniforme
  gender            int     {0, 1}        48% F / 52% M (NHANES)
  weight_kg         float   [39, 137]     Corrélé à BMI + taille
  height_cm         float   [150, 210]    N(176,7) hommes / N(163,6) femmes
  bmi               float   [16, 45]      Calculé, corrélé au niveau d'expérience
  body_fat_pct      float   [5, 50]       Corrélé au BMI + genre (ACSM)
  resting_bpm       int     [45, 100]     Corrélé au niveau d'expérience
  experience_level  int     {1, 2, 3}     50% débutant / 35% inter / 15% avancé

Cible :
  fitness_profile   str     6 classes     Voir section 3.5
```

#### Corrélations physiologiques simulées

Le générateur reproduit des corrélations réalistes :

- `bmi` plus élevé pour les débutants (exp=1) → `N(27.5, 4.5)` vs `N(23, 2.8)` pour avancés
- `body_fat_pct` corrélé au BMI avec offset genre : hommes `bmi × 1.05 - 10`, femmes `bmi × 1.10 - 5`
- `resting_bpm` plus bas pour les sportifs confirmés : débutants `N(76,6)`, avancés `N(62,6)`

#### Labeling — règles ACSM 2022

```python
if bmi >= 27.5 OR fat% élevé (ACSM) :
    exp == 1  →  perte_poids_debutant
    exp >= 2  →  perte_poids_confirme

elif bmi < 22.5 OR fat% bas :
    exp == 1  →  prise_masse_debutant
    exp >= 2  →  prise_masse_confirme

elif resting_bpm > 72 AND 22 <= bmi <= 28 :
    →  amelioration_cardio

else :
    →  maintien_bien_etre
```

**Bruit réaliste :** 12% des labels sont flippés vers une classe adjacente pour simuler l'ambiguïté des cas limites. Cela empêche le modèle d'apprendre une règle déterministe parfaite et force l'apprentissage de patterns statistiques réels.

#### Distribution finale des classes

| Profil | Effectif | % |
|---|---|---|
| prise_masse_confirme | ~492 | 24.6% |
| perte_poids_debutant | ~433 | 21.6% |
| amelioration_cardio  | ~337 | 16.9% |
| prise_masse_debutant | ~296 | 14.8% |
| maintien_bien_etre   | ~294 | 14.7% |
| perte_poids_confirme | ~148 | 7.4%  |

Léger déséquilibre sur `perte_poids_confirme` → géré via `class_weight='balanced'` dans RandomForest.

---

### 3.2 Nettoyage

**Fichier :** `ml/src/preprocessing/cleaner.py`

```python
clean(df) → df_clean
```

**Étapes :**

1. **Suppression des doublons** — `drop_duplicates()` sur toutes les colonnes
2. **Gestion des NaN** — imputation médiane (numériques) / mode (catégorielles)
3. **Outliers IQR×3** — supprime uniquement les outliers physiologiquement impossibles (factor=3 est très permissif — seules les valeurs absurdes sont retirées)
4. **Clip physiologique** — bornes dures par variable (ex: BPM repos 40-105)

**Borne par variable :**

| Variable | Min | Max | Source |
|---|---|---|---|
| age | 18 | 65 | Critère d'inclusion HealthAI |
| weight_kg | 30 | 200 | Borne physiologique |
| height_cm | 145 | 215 | Borne physiologique |
| bmi | 14 | 48 | OMS (en deçà = anorexie sévère) |
| body_fat_pct | 4 | 52 | ACSM (minimum essentiel) |
| resting_bpm | 40 | 105 | Clinique (bradycardie/tachycardie) |

---

### 3.3 Feature Engineering

**Fichier :** `ml/src/preprocessing/engineer.py`

Crée 7 features dérivées à partir des 8 variables brutes.

#### Tableau complet

| Feature créée | Méthode | Justification métier |
|---|---|---|
| `bmi_category` | Seuils OMS : <18.5/18.5-25/25-30/30+ | Standard médical international |
| `fat_category` | Seuils ACSM stratifiés par genre | Interprétation clinique du fat% |
| `hr_zone` | 5 zones : excellent(<60) → mauvais(>90) | Indicateur forme cardio-vasculaire |
| `fitness_score` | Composite normalisé BMI+fat%+BPM+exp | Score synthétique 0-100 |
| `age_group` | Tranches 18-25/26-35/36-45/46-55/56-65 | Métabolisme change par décennie |
| `bmi_x_exp` | `bmi × experience_level` | Interaction : débutant obèse ≠ athlète lourd |
| `fat_per_exp` | `body_fat_pct / experience_level` | Masse grasse relative à l'expérience |

#### Détail du `fitness_score`

```python
bmi_score  = 1 - |bmi - 21| / 15        # optimum OMS ~21
fat_score  = 1 - (fat - Q25) / range    # moins de graisse = meilleur score
hr_score   = 1 - (resting_bpm - 50) / 50  # BPM bas = bonne forme
exp_score  = (experience_level - 1) / 2   # 0→1

fitness_score = (0.30 × bmi_score +
                 0.25 × fat_score  +
                 0.25 × hr_score   +
                 0.20 × exp_score) × 100
```

Les poids (30/25/25/20) sont basés sur la littérature en médecine du sport (BMI et fat% ont le plus d'impact sur la capacité fonctionnelle).

#### Feature `bmi_x_exp` — pourquoi ?

Un BMI de 28 signifie des choses très différentes selon l'expérience :
- `exp=1` (débutant) → probable surpoids, programme perte de poids
- `exp=3` (avancé) → probable muscle, programme maintien/masse

Cette interaction n'est pas capturée par les features séparées. `bmi × exp_level` crée une dimension orthogonale.

#### Catégories `fat_category` (ACSM 2022)

| Catégorie | Hommes | Femmes |
|---|---|---|
| essentiel | < 6% | < 14% |
| athlete | 6-14% | 14-21% |
| fitness | 14-18% | 21-25% |
| acceptable | 18-25% | 25-32% |
| obese | > 25% | > 32% |

---

### 3.4 Preprocessing sklearn

**Fichier :** `ml/src/preprocessing/pipeline.py`

Construit un `ColumnTransformer` sklearn qui traite les 3 types de features :

```python
ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),      # 10 features numériques → normalisées
    ("ord", OrdinalEncoder(...), ORD_FEATURES),   # 4 features catégorielles → entiers ordonnés
    ("bin", "passthrough", BIN_FEATURES),          # gender → déjà encodé 0/1
])
```

**Features numériques (10) :**
`age, weight_kg, height_cm, bmi, body_fat_pct, resting_bpm, experience_level, fitness_score, bmi_x_exp, fat_per_exp`

**Features catégorielles ordinales (4) avec ordre explicite :**

| Feature | Ordre |
|---|---|
| `bmi_category` | sous_poids → normal → surpoids → obesite |
| `fat_category` | essentiel → athlete → fitness → acceptable → obese |
| `hr_zone` | excellent → bon → moyen → sous_optimal → mauvais |
| `age_group` | 18-25 → 26-35 → 36-45 → 46-55 → 56-65 |

**Pourquoi `OrdinalEncoder` et pas `OneHotEncoder` ?**

Ces catégories ont un ordre naturel et médical. `OrdinalEncoder` respecte cet ordre, ce qui permet aux arbres de décision de faire des splits sémantiquement corrects (ex: `bmi_category > 1` = surpoids ou obèse).

**Pourquoi `StandardScaler` ?**

Bien que les arbres de décision (RF, GBM, XGBoost) ne soient pas sensibles à l'échelle, normaliser permet :
- Une Permutation Importance et des SHAP values comparables entre features
- Une compatibilité future avec d'autres algorithmes (SVM, Logistic Regression)

---

### 3.5 Modélisation

**Fichier :** `ml/src/training/train.py`

#### Les 6 profils (classes cibles)

| Profil | Signification métier |
|---|---|
| `perte_poids_debutant` | Surpoids + peu d'expérience sportive |
| `perte_poids_confirme` | Surpoids + expérience existante |
| `prise_masse_debutant` | Sous-poids/maigreur + débutant |
| `prise_masse_confirme` | Sous-poids/maigreur + expérimenté |
| `amelioration_cardio` | BPM repos élevé, profil cardio-vasculaire à améliorer |
| `maintien_bien_etre` | Profil équilibré, objectif bien-être |

#### Algorithmes comparés

**Random Forest**
- Ensemble de N arbres de décision entraînés en parallèle (bagging)
- Prédiction = vote majoritaire
- Avantages : robuste aux outliers, résistant à l'overfitting, interprétable via feature importance
- Inconvénients : lourd en mémoire sur N élevé

**Gradient Boosting**
- Arbres entraînés séquentiellement (chaque arbre corrige les erreurs du précédent)
- Minimise une fonction de perte par descente de gradient
- Avantages : généralement plus précis que RF
- Inconvénients : risque d'overfitting si `n_estimators` élevé sans régularisation

**XGBoost**
- Implémentation optimisée de Gradient Boosting avec régularisation L1/L2
- Gestion native des valeurs manquantes
- Avantages : état de l'art sur les données tabulaires
- Inconvénients : plus de paramètres à tuner, moins interprétable

#### Optimisation des hyperparamètres — RandomizedSearchCV

```python
RandomizedSearchCV(
    pipeline,
    param_distributions=SEARCH_SPACE,
    n_iter=50,          # 50 combinaisons aléatoires (vs >1000 pour GridSearch)
    cv=StratifiedKFold(n_splits=5),
    scoring='f1_macro', # Métrique de sélection = F1-macro (robuste au déséquilibre)
    random_state=42,
    n_jobs=-1,          # Parallélisation sur tous les cœurs
)
```

**Pourquoi RandomizedSearch ?** Sur un espace à 4-6 hyperparamètres avec 3-4 valeurs chacun, GridSearch = 4⁴ = 256+ combinaisons × 5 folds = 1280+ fits. RandomizedSearch avec n_iter=50 couvre statistiquement 95%+ de la performance optimale en 50 fits seulement.

**Pourquoi F1-macro ?** La classe `perte_poids_confirme` représente 7.4% du dataset. L'accuracy favoriserait un modèle qui l'ignore. F1-macro donne le même poids à chaque classe, indépendamment de son effectif.

#### Train/Test split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

**80/20 stratifié :** La stratification garantit que chaque classe est représentée dans les deux splits dans les mêmes proportions que le dataset original. Critique pour la classe minoritaire `perte_poids_confirme`.

---

### 3.6 Évaluation

#### Métriques produites

| Métrique | Pertinence | À défendre |
|---|---|---|
| **Accuracy** | ⚠️ Trompeuse si déséquilibre | Oui, avec nuance |
| **F1-macro** | ✅ Métrique principale | Oui — expliquer pourquoi |
| **F1 par classe** | ✅ Granularité métier | Oui — commenter les écarts |
| **Confusion Matrix** | ✅ Analyse des erreurs | Oui — analyse métier |
| **Learning Curve** | ✅ Détecte overfitting | Oui — montre la rigueur |
| **CV F1 ± std** | ✅ Robustesse de la mesure | Oui — montre la stabilité |

#### Lecture de la Confusion Matrix — approche métier

Les erreurs ne sont pas toutes égales. Un bon commentaire de jury :

> "Le modèle confond parfois `perte_poids_debutant` et `amelioration_cardio`. C'est une erreur **acceptable** métier — les deux profils impliquent une augmentation de l'activité physique. En revanche, une confusion entre `perte_poids_debutant` et `prise_masse_debutant` serait une erreur critique, car les recommandations nutritionnelles sont opposées. Notre matrice montre que ces erreurs critiques sont quasi-nulles."

#### Lecture de la Learning Curve

- **Train ≈ Validation, tous deux élevés** → bonne généralisation ✅
- **Grand écart Train >> Validation** → overfitting, réduire la complexité
- **Train et Validation tous deux bas** → underfitting, augmenter la complexité ou le volume de données

---

### 3.7 Explicabilité

#### Feature Importance (sklearn)

Calculée lors de l'entraînement (impureté Gini). Rapide et intégrée au modèle. Limitée : les features corrélées se "volent" mutuellement de l'importance.

#### Permutation Importance

```python
permutation_importance(model, X_test, y_test, n_repeats=20, scoring='f1_macro')
```

Mélange aléatoirement chaque feature et mesure la dégradation du F1-macro. Plus fiable que l'importance Gini pour les features corrélées (BMI et body_fat_pct). Le `n_repeats=20` donne une estimation robuste avec intervalle de confiance.

#### SHAP (TreeExplainer)

- **Global (beeswarm)** : impact moyen de chaque feature sur toutes les prédictions
- **Local (waterfall)** : décomposition de la prédiction d'un individu spécifique

SHAP est particulièrement pertinent pour HealthAI Coach car il permet d'expliquer une recommandation à l'utilisateur : *"Votre BMI de 28.3 contribue fortement à la recommandation perte_poids_debutant."*

---

### 3.8 MLflow

**Tracking URI :** `sqlite:///ml/artifacts/mlflow.db`

Chaque run logge :
- **Paramètres** : hyperparamètres du meilleur modèle après RandomSearch
- **Métriques** : accuracy, f1_macro, f1_weighted, cv_f1_macro, cv_f1_std
- **Artifacts** : confusion matrix, feature importance, learning curve (PNG)
- **Modèle** : sérialisé sklearn via `mlflow.sklearn.log_model`

**Lancer l'UI MLflow :**
```bash
cd /home/enys/2MPR
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///ml/artifacts/mlflow.db
# Ouvrir http://localhost:5000
```

---

## 4. Moteur de recommandation

**Fichier :** `ml/src/recommendation_engine/engine.py`

Traduit un profil ML en programme sportif structuré. **Séparé intentionnellement du modèle ML.**

### Profils et programmes

| Profil | Sessions/sem | Durée | Focus |
|---|---|---|---|
| `perte_poids_debutant` | 3 | 40 min | Cardio modéré + renforcement léger |
| `perte_poids_confirme` | 4 | 50 min | HIIT + musculation + cardio |
| `prise_masse_debutant` | 3 | 50 min | Renforcement fondamental |
| `prise_masse_confirme` | 4 | 60 min | Hypertrophie split push/pull |
| `amelioration_cardio`  | 4 | 45 min | VO2max + endurance |
| `maintien_bien_etre`   | 3 | 45 min | Yoga + jogging + fonctionnel |

### Interface

```python
from ml.src.recommendation_engine.engine import get_program, program_to_dict

program = get_program("perte_poids_debutant")
# → Program(sessions_per_week=3, session_duration_min=40, activities=[...], ...)

data = program_to_dict(program)
# → dict compatible avec l'API
```

---

## 5. API FastAPI

### 5.1 Endpoints

#### `GET /health`

Healthcheck. Retourne le statut et la version.

```json
{"status": "ok", "version": "2.0.0"}
```

---

#### `POST /recommend` ← Endpoint principal

**Description :** Prédit le profil fitness et retourne un programme sportif complet.

**Request body :**

```json
{
  "age": 32,
  "gender": "male",
  "weight_kg": 88.0,
  "height_cm": 178.0,
  "body_fat_pct": 26.0,
  "resting_bpm": 74,
  "experience_level": "beginner"
}
```

| Champ | Type | Contraintes | Description |
|---|---|---|---|
| `age` | int | [18, 65] | Âge en années |
| `gender` | str | "male" / "female" | Genre |
| `weight_kg` | float | (30, 200) | Poids en kg |
| `height_cm` | float | (140, 215) | Taille en cm |
| `body_fat_pct` | float | [4, 55] | Taux de masse grasse (%) |
| `resting_bpm` | int | [40, 105] | Fréquence cardiaque au repos |
| `experience_level` | str | "beginner" / "intermediate" / "advanced" | Niveau sportif |

**Réponse :**

```json
{
  "profile": "perte_poids_debutant",
  "confidence": 0.8734,
  "bmi": 27.76,
  "bmi_category": "Surpoids",
  "program": {
    "sessions_per_week": 3,
    "session_duration_min": 40,
    "focus": "Cardio modéré + renforcement léger",
    "activities": [
      "Marche rapide ou vélo 20 min (zone 2)",
      "Circuit renforcement corps entier 15 min",
      "Étirements 5 min"
    ],
    "intensity": "Modérée — 55-65% FCmax (zone 2-3)",
    "weekly_volume_h": 2.0,
    "progression": "Ajouter 5 min de cardio chaque semaine. Après 4 semaines : passer à 4 séances.",
    "nutrition_tip": "Déficit calorique de 300-400 kcal/j. Priorité aux protéines (1.6 g/kg).",
    "objective": "Perte de poids progressive — 0.5 kg/semaine est l'objectif réaliste."
  }
}
```

---

#### `GET /profiles`

Liste les 6 profils disponibles avec résumé.

```json
{
  "profiles": [
    {"id": "perte_poids_debutant", "focus": "Cardio modéré + renforcement léger", "sessions_per_week": 3},
    ...
  ]
}
```

---

#### `POST /nutrition/predict` *(deprecated)*

Ancien endpoint maintenu pour rétrocompatibilité. Retourne les 3 anciennes classes (perte_poids / maintien / prise_masse) via une règle déterministe simple. **Ne pas utiliser en production.**

---

### 5.2 Schémas de données

**Fichier :** `app/schemas.py`

```
RecommendInput   →  validation Pydantic de l'entrée utilisateur
RecommendOutput  →  profil + BMI + programme complet
ProgramOutput    →  détail du programme sportif
NutritionInput   →  (legacy) ancien format d'entrée
NutritionOutput  →  (legacy) ancien format de sortie
```

**Validation personnalisée :**

```python
@model_validator(mode="after")
def check_bmi(self):
    bmi = self.weight_kg / (self.height_cm / 100) ** 2
    if bmi < 14 or bmi > 48:
        raise ValueError(f"IMC calculé ({bmi:.1f}) hors plage...")
```

Le BMI est calculé dès la validation pour rejeter les combinaisons poids/taille physiologiquement impossibles, avant même d'atteindre le modèle.

---

### 5.3 Architecture interne

**Fichier :** `app/service.py`

```python
class FitnessService:
    _instance = None           # Singleton

    def __new__(cls):
        # Charge model.pkl + encoder.pkl une seule fois au démarrage
        ...

    def recommend(self, data: RecommendInput) -> RecommendOutput:
        # 1. Calculer BMI
        # 2. Construire DataFrame
        # 3. Feature engineering (engineer())
        # 4. Sélectionner features (get_feature_names())
        # 5. predict() + predict_proba()
        # 6. Décoder label (LabelEncoder)
        # 7. get_program() → moteur de règles
        # 8. Retourner RecommendOutput
```

**Singleton :** Le modèle (6.6 MB) est chargé une seule fois en mémoire au démarrage du serveur. Chaque requête réutilise le même objet. Évite de charger/décharger le modèle à chaque requête (latence ~500ms → ~2ms).

---

## 6. Guide de démarrage

### Prérequis

- Python 3.11+
- Linux / WSL2

### Installation

```bash
# Créer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Générer le dataset

```bash
python ml/data/generate_dataset.py
```

Crée `ml/data/processed/healthai_dataset.csv` (2000 lignes).

### Entraîner les modèles

```bash
python ml/src/training/train.py
```

- Compare Random Forest, Gradient Boosting, XGBoost
- RandomizedSearchCV (50 itérations, 5-fold CV)
- Log tous les runs dans MLflow
- Sauvegarde le meilleur modèle dans `ml/models/`

Durée typique : 5-10 minutes selon la machine.

### Lancer l'API

```bash
uvicorn app.main:app --reload --port 8000
```

Documentation interactive : http://localhost:8000/docs

### Visualiser les runs MLflow

```bash
mlflow ui --backend-store-uri sqlite:///ml/artifacts/mlflow.db
```

Interface : http://localhost:5000

### Ouvrir le notebook

```bash
jupyter notebook ml/notebooks/HealthAI_Coach_ML.ipynb
```

### Exemple de requête API

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32,
    "gender": "male",
    "weight_kg": 88,
    "height_cm": 178,
    "body_fat_pct": 26,
    "resting_bpm": 74,
    "experience_level": "beginner"
  }'
```

---

## 7. Décisions techniques

### Leakage temporel — pourquoi c'est crucial

Les features `Avg_BPM`, `Calories_Burned`, `Session_Duration` du Gym Members Dataset sont mesurées **pendant** l'entraînement. Un nouvel utilisateur qui s'inscrit ne les a pas. Si ces features avaient été utilisées, le modèle aurait appris à partir d'informations futures — c'est du **data leakage temporel**.

**Conséquence en production :** Un modèle leaké donne d'excellentes métriques en développement mais est inutilisable en production (les features ne sont pas disponibles). HealthAI Coach utilise uniquement des features disponibles **à l'inscription**.

### Dataset synthétique — justification

Construire les labels depuis des **règles cliniques reconnues** (ACSM, OMS) est une pratique courante quand les données réelles ne contiennent pas de labels prescriptifs validés. C'est ce que font les systèmes de santé numériques réels. La valeur ajoutée du ML est d'**apprendre les frontières floues** entre ces règles (cas limites, interactions non-linéaires) et de **généraliser** à de nouveaux profils.

### Séparation ML / moteur de règles

Le modèle prédit une classe nommée (`perte_poids_debutant`). Le programme est généré par un moteur de règles indépendant. Cette séparation permet :
- Modifier les programmes sans réentraîner le modèle
- Intégrer des recommandations d'experts médicaux dans le moteur
- Tester A/B des programmes différents pour le même profil

### Choix de SQLite pour MLflow

MLflow 2.x a déprécié le file store. SQLite est la solution la plus légère pour une MSPR — pas besoin de serveur de base de données, le fichier est portable et versionnable.

---

## 8. Défense orale — points clés

### Ce qui impressionne un jury CDA

1. **Justifier le rejet du dataset NHANES** — "Le label est une auto-perception, pas une prescription. Aucune valeur métier pour une recommandation sportive."

2. **Expliquer le leakage temporel** — "Nous avons identifié et éliminé les features temporellement leakées avant modélisation. Un Data Scientist qui ne fait pas cette analyse expose le projet à un échec en production."

3. **Analyser la Confusion Matrix de manière métier** — Pas juste des chiffres, mais "ces erreurs sont acceptables / ces erreurs seraient critiques".

4. **Montrer MLflow en live** — Ouvrir l'UI, comparer les 3 runs, expliquer pourquoi on a choisi ce modèle sur ce score.

5. **Expliquer la séparation ML / moteur de règles** — "Le modèle classe, le moteur prescrit. C'est une architecture propre, maintenable, et conforme aux bonnes pratiques MLOps."

6. **Learning Curve** — "La faible différence entre courbe train et validation confirme l'absence d'overfitting."

### Ce qu'il faut éviter

- Dire "précision 95%" sans mentionner les classes déséquilibrées
- Présenter SHAP sans pouvoir expliquer une valeur négative
- Ne pas avoir de réponse sur "pourquoi ce dataset ?"
- Confondre accuracy et F1 dans l'analyse

### Questions jury fréquentes et réponses

**"Pourquoi un dataset synthétique ?"**
> "Aucun dataset public ne propose de labels prescriptifs validés médicalement. Nous avons construit les labels depuis les guidelines ACSM 2022, ce qui est une pratique standard en santé numérique. Le ML apprend les interactions complexes et les cas limites que les règles simples ne capturent pas."

**"Le modèle ne fait pas juste reproduire vos règles ?"**
> "Non, pour deux raisons : 1) Le bruit de 12% simule l'ambiguïté réelle des cas limites. 2) Le feature engineering crée des dimensions non présentes dans les règles (fitness_score, bmi_x_exp). Le modèle apprend des frontières de décision non-linéaires dans un espace à 15 dimensions."

**"Pourquoi F1-macro et pas accuracy ?"**
> "La classe `perte_poids_confirme` représente 7.4% du dataset. Un modèle qui l'ignorerait complètement aurait une accuracy de 92.6% — excellente sur le papier, inutile en pratique. F1-macro donne un poids identique à chaque classe."
