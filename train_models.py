import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("dataset_fitness_ml.csv")
print(f"Dataset : {len(df)} lignes\n")

FEATURES = ["imc", "taux_masse_grasse", "age"]
CIBLE = "label"

X = df[FEATURES]
y = df[CIBLE]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print(f"Classes encodées : {list(zip(encoder.classes_, range(len(encoder.classes_))))}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train : {len(X_train)} lignes  |  Test : {len(X_test)} lignes\n")

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report :")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

print("Importance des features :")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(encoder, "models/encoder.pkl")
print("\n✓ Modèles sauvegardés dans models/")
