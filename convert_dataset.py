"""
Fusion des fichiers NHANES 2017-2018 → dataset_fitness_ml.csv

Jointure sur SEQN (identifiant unique de chaque participant) :
  DEMO_J.xpt  →  age (RIDAGEYR)
  BMX_J.xpt   →  taille (BMXHT) + poids (BMXWT) → calcul IMC
  DXX_J.xpt   →  taux de masse grasse total % (DXDTOPF)  ← données DXA cliniques
  WHQ_J.xpt   →  perception du poids (WHQ030)            ← label

Label WHQ030 — question posée par un enquêteur médical :
  1 = "Overweight"              → perte_poids
  2 = "Underweight"             → prise_masse
  3 = "About the right weight"  → maintien
  7 / 9 = Refused / Don't know  → supprimé

Indépendance features / label :
  Features : IMC (height+weight), taux_masse_grasse (DXA), age — mesures biométriques
  Label    : perception clinique déclarée                    — source complètement distincte
"""

import pandas as pd

OUTPUT = "dataset_fitness_ml.csv"

WHQ030_MAP = {1.0: "perte_poids", 2.0: "prise_masse", 3.0: "maintien"}

# ── chargement ─────────────────────────────────────────────────────────────
print("Chargement des fichiers NHANES...")

demo = pd.read_sas("DEMO_J.xpt", format="xport")[["SEQN", "RIDAGEYR"]]
bmx  = pd.read_sas("BMX_J.xpt",  format="xport")[["SEQN", "BMXHT", "BMXWT"]]
dxx  = pd.read_sas("DXX_J.xpt",  format="xport")[["SEQN", "DXDTOPF"]]
whq  = pd.read_sas("WHQ_J.xpt",  format="xport")[["SEQN", "WHQ030"]]

print(f"  DEMO : {len(demo):>5} participants")
print(f"  BMX  : {len(bmx):>5} participants")
print(f"  DXX  : {len(dxx):>5} participants (DXA — sous-échantillon)")
print(f"  WHQ  : {len(whq):>5} participants")

# ── merge sur SEQN ─────────────────────────────────────────────────────────
print("\nFusion sur SEQN...")
merged = demo.merge(bmx, on="SEQN") \
             .merge(dxx, on="SEQN") \
             .merge(whq, on="SEQN")
print(f"  Après merge          : {len(merged)} lignes")

# ── nettoyage ──────────────────────────────────────────────────────────────
merged = merged.dropna(subset=["RIDAGEYR", "BMXHT", "BMXWT", "DXDTOPF", "WHQ030"])
print(f"  Après dropna         : {len(merged)} lignes")

merged = merged[merged["WHQ030"].isin([1.0, 2.0, 3.0])]
print(f"  Après filtre label   : {len(merged)} lignes (refuse/don't know exclus)")

merged = merged[merged["RIDAGEYR"] >= 18]
print(f"  Après filtre adultes : {len(merged)} lignes")

# ── construction des colonnes finales ──────────────────────────────────────
out = pd.DataFrame()
out["age"]               = merged["RIDAGEYR"].astype(int)
out["imc"]               = (merged["BMXWT"] / (merged["BMXHT"] / 100) ** 2).round(2)
out["taux_masse_grasse"] = merged["DXDTOPF"].round(2)
out["label"]             = merged["WHQ030"].map(WHQ030_MAP)

# filtres biologiques de cohérence
out = out[(out["imc"] >= 13) & (out["imc"] <= 70)]
out = out[(out["taux_masse_grasse"] >= 3) & (out["taux_masse_grasse"] <= 70)]
out = out.dropna()

out.to_csv(OUTPUT, index=False)

# ── rapport ────────────────────────────────────────────────────────────────
print(f"\n{'═'*50}")
print(f"Lignes finales         : {len(out)}")
print()
print("Distribution des labels :")
vc = out["label"].value_counts()
for lbl, n in vc.items():
    bar = "█" * int(n / len(out) * 40)
    print(f"  {lbl:<15} {n:>5}  ({n/len(out)*100:.1f}%)  {bar}")

print()
print("Statistiques des features :")
print(out[["age", "imc", "taux_masse_grasse"]].describe().round(2))
print(f"\n✓ Sauvegardé → {OUTPUT}")
