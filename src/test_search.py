"""
Script de test pour valider les ameliorations du moteur de recherche.
Charge TOUTES les saisons et teste des requetes Q1 avec episodes attendus.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import data_loader as dl
import moteur as mt
import q1_search as q1
import pandas as pd

print("=" * 70)
print("  TEST DU MOTEUR DE RECHERCHE TYPE Q1")
print("=" * 70)

# Chargement de toutes les saisons
print("\n[1] Chargement de toutes les saisons...")
df = dl.chargerDonnees("../datasets")

# Construction de l'index
print("\n[2] Construction de l'index TF-IDF...")
vectorizer, matrice_tfidf, df_docs, mots_vides = mt.construireIndex(df, par_scene=False)

# --- Requetes de test Q1 avec episodes attendus ---
questions_test = [
    {"id": 1, "question": "Monica and Chandler announce their engagement.",  "expected_episode": "S07E01"},
    {"id": 2, "question": "Rachel's first day at her new job with Mark.",     "expected_episode": "S03E12"},
    {"id": 3, "question": "Joey learns to speak French for an audition.",     "expected_episode": "S10E13"},
    {"id": 4, "question": "Phoebe wants to sing at Monica's wedding.",        "expected_episode": "S07E01"},
    {"id": 5, "question": "Ross is jealous of the gifts sent to Rachel's workplace.", "expected_episode": "S03E12"},
]

print(f"\n[3] Execution de {len(questions_test)} requetes Q1...\n")

nb_ok = 0
for test in questions_test:
    qid = test["id"]
    question = test["question"]
    expected = test["expected_episode"]

    print("=" * 70)
    print(f"  Q{qid}: {question}")
    print(f"  Attendu: {expected}")
    print("-" * 70)

    resultats = q1.rechercherQ1(
        question, df, vectorizer, matrice_tfidf, df_docs,
        top_k=5, motsVidesRecherche=mots_vides
    )

    if resultats.empty:
        print("  -> Aucun resultat")
        found = False
    else:
        found = False
        for rank, (_, row) in enumerate(resultats.iterrows(), 1):
            ep_str = f"S{row['saison']}E{row['episode']}"
            marker = ""
            if ep_str == expected:
                marker = " <-- MATCH"
                found = True
            print(f"  #{rank}  {ep_str}  score={row['score']:.4f}{marker}")
            if 'apercu' in row and row['apercu']:
                print(f"       > {row['apercu'][:150]}")

    status = "OK" if found else "MISS"
    if found:
        nb_ok += 1
    print(f"\n  Resultat: [{status}]")
    print()

print("=" * 70)
print(f"  SCORE FINAL : {nb_ok}/{len(questions_test)} episodes attendus trouves dans le top 5")
print("=" * 70)
