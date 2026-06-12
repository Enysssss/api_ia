"""
Moteur de recommandation : profil fitness → programme sportif détaillé.
Séparé du ML : le modèle prédit le profil, ce module traduit en programme.
Basé sur les recommandations ACSM 2022 et OMS.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Program:
    profile: str
    sessions_per_week: int
    session_duration_min: int
    focus: str
    activities: list[str]
    intensity: str
    weekly_volume_h: float
    progression: str
    nutrition_tip: str
    objective: str


_PROGRAMS: dict[str, Program] = {

    "perte_poids_debutant": Program(
        profile="perte_poids_debutant",
        sessions_per_week=3,
        session_duration_min=40,
        focus="Cardio modéré + renforcement léger",
        activities=[
            "Marche rapide ou vélo 20 min (zone 2)",
            "Circuit renforcement corps entier 15 min",
            "Étirements 5 min",
        ],
        intensity="Modérée — 55-65% FCmax (zone 2-3)",
        weekly_volume_h=2.0,
        progression="Ajouter 5 min de cardio chaque semaine. Après 4 semaines : passer à 4 séances.",
        nutrition_tip="Déficit calorique de 300-400 kcal/j. Priorité aux protéines (1.6 g/kg). Éviter les ultra-transformés.",
        objective="Perte de poids progressive — 0.5 kg/semaine est l'objectif réaliste.",
    ),

    "perte_poids_confirme": Program(
        profile="perte_poids_confirme",
        sessions_per_week=4,
        session_duration_min=50,
        focus="HIIT + musculation + cardio soutenu",
        activities=[
            "HIIT 20 min (30s effort / 30s repos × 10)",
            "Musculation : squat, deadlift, rowing 20 min",
            "Cardio steady-state 10 min",
        ],
        intensity="Élevée — 70-85% FCmax (zone 3-4) sur les séances HIIT",
        weekly_volume_h=3.3,
        progression="Augmenter l'intensité HIIT (ratio 40/20) après 3 semaines. Ajouter charge +5% sur musculation.",
        nutrition_tip="Déficit calorique de 400-500 kcal/j. Protéines 1.8-2.0 g/kg. Timing : protéines post-workout dans les 2h.",
        objective="Remodelage corporel — perte de graisse avec préservation musculaire.",
    ),

    "prise_masse_debutant": Program(
        profile="prise_masse_debutant",
        sessions_per_week=3,
        session_duration_min=50,
        focus="Renforcement musculaire fondamental",
        activities=[
            "Squat 3×8-12",
            "Développé couché ou pompes 3×8-12",
            "Rowing barre ou tirage 3×8-12",
            "Gainage 3×30s",
        ],
        intensity="Modérée-élevée — charge à 65-75% 1RM. Tempo contrôlé.",
        weekly_volume_h=2.5,
        progression="Programme linéaire : +2.5 kg chaque séance sur les exercices de base. Passer à 4 séances après 6 semaines.",
        nutrition_tip="Surplus calorique de 200-300 kcal/j. Protéines 1.8 g/kg. Glucides avant et après l'entraînement.",
        objective="Construction des bases musculaires — progression des charges toutes les séances.",
    ),

    "prise_masse_confirme": Program(
        profile="prise_masse_confirme",
        sessions_per_week=4,
        session_duration_min=60,
        focus="Hypertrophie — programme split push/pull",
        activities=[
            "Jour 1 (Push) : développé couché, épaules, triceps",
            "Jour 2 (Pull) : rowing, tirage, biceps, trapèzes",
            "Jour 3 : repos ou cardio léger",
            "Jour 4 (Legs) : squat, leg press, fentes, mollets",
            "Jour 5 : full body ou répétition push/pull",
        ],
        intensity="Élevée — 70-80% 1RM, 6-12 reps, surcharge progressive.",
        weekly_volume_h=4.0,
        progression="Périodisation ondulante : alterner semaines force (5×5) et hypertrophie (4×10).",
        nutrition_tip="Surplus calorique de 300-400 kcal/j. Protéines 2.0-2.2 g/kg. Créatine monohydrate 5g/j (evidence level A).",
        objective="Hypertrophie musculaire avancée avec minimisation du gras pris.",
    ),

    "amelioration_cardio": Program(
        profile="amelioration_cardio",
        sessions_per_week=4,
        session_duration_min=45,
        focus="Développement VO2max et endurance cardio-vasculaire",
        activities=[
            "Course / vélo zone 2 : 30 min (60-70% FCmax)",
            "Intervalles courts : 6×3min à 80-85% FCmax",
            "Natation ou vélo elliptique (low impact) 30 min",
            "Cardio récupération active 20 min zone 1",
        ],
        intensity="Variable — 60-85% FCmax selon séance. Alternance easy/hard.",
        weekly_volume_h=3.0,
        progression="80% du volume en zone 2, 20% en zone 4-5 (polarized training). Augmenter le volume de 10%/semaine max.",
        nutrition_tip="Glucides comme carburant principal sur les séances longues. Hydratation : 500ml/h d'effort. Récupération prioritaire.",
        objective="Baisser le BPM repos de 5-10 bpm en 8 semaines. Améliorer l'endurance générale.",
    ),

    "maintien_bien_etre": Program(
        profile="maintien_bien_etre",
        sessions_per_week=3,
        session_duration_min=45,
        focus="Équilibre forme physique et récupération",
        activities=[
            "Yoga ou Pilates 30 min",
            "Marche / jogging léger 30 min",
            "Renforcement fonctionnel 20 min",
        ],
        intensity="Légère à modérée — 50-65% FCmax. Plaisir et régularité prioritaires.",
        weekly_volume_h=2.25,
        progression="Maintenir le volume. Varier les activités pour éviter la monotonie. Écouter la récupération.",
        nutrition_tip="Alimentation équilibrée. Pas de restriction. Hydratation et sommeil sont les leviers prioritaires.",
        objective="Maintenir la condition physique actuelle. Bien-être général et prévention du déconditionnement.",
    ),
}


def get_program(profile: str) -> Program:
    if profile not in _PROGRAMS:
        raise ValueError(f"Profil inconnu : {profile}. Valeurs valides : {list(_PROGRAMS.keys())}")
    return _PROGRAMS[profile]


def list_profiles() -> list[str]:
    return list(_PROGRAMS.keys())


def program_to_dict(program: Program) -> dict:
    return {
        "sessions_per_week":     program.sessions_per_week,
        "session_duration_min":  program.session_duration_min,
        "focus":                 program.focus,
        "activities":            program.activities,
        "intensity":             program.intensity,
        "weekly_volume_h":       program.weekly_volume_h,
        "progression":           program.progression,
        "nutrition_tip":         program.nutrition_tip,
        "objective":             program.objective,
    }
