import os
import sys

# Ajout du path pour importer les modules proprement
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import data_loader as dl
import sentiment_classifier as sc
import fiches_acteurs as fa

def main():
    print("="*50)
    print(" Lancement du pipeline d'Analyse (Groupe 1)")
    print("="*50)
    
    # 1. Chargement des données
    # On se place généralement à la racine ou dans src
    dossier_datasets = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    if not os.path.exists(dossier_datasets):
        print(f"Erreur : le dossier {dossier_datasets} n'existe pas.")
        return
        
    print("\n[1/4] Chargement des données...")
    df = dl.chargerDonnees(dossier_datasets)
    
    # Suppression des entités qui ne sont pas de vrais personnages (apres tests)
    df = df[~df['acteur'].isin(['All', 'Woman'])]
    
    # 2. Sentiments
    print("\n[2/4] Analyse des Sentiments...")
    df = sc.predict_sentiments_vader(df)
    
    dataset_kag = os.path.join(dossier_datasets, "sentiment_analysis.csv")
    vectorizer, model = sc.train_ml_classifier(dataset_kag)
    if vectorizer and model:
        df = sc.predict_sentiments_ml(df, vectorizer, model)
    else:
        print("=> Modèle ML ignoré faute de dataset. Les fiches acteurs auront les champs ML vides.")
        
    # 3. Génération des fiches acteurs (inclus les rôles et le graphe/réseau)
    print("\n[3/4] Génération des fiches acteurs, réseaux et extraction de sujets...")
    dossier_sortie = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "fiches_acteurs"))
    # minLines à 100 pour accélérer un peu sur les personnages principaux
    fiches = fa.genererFichesActeurs(df, outputDir=dossier_sortie, minLines=100)
    
    print("\n[4/4] Terminé ! Les fiches ont été générées dans le dossier :", dossier_sortie)

if __name__ == "__main__":
    main()
