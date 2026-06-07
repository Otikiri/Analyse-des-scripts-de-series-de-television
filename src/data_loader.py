import pandas as pd
from parser import parserRepertoire
from utils import nettoyerTexte,tokeniserTexte
import os

def chargerDonnees(path): 
    """Charge les données textuelles brutes depuis un répertoire, applique les pipelines de nettoyage et de tokenisation ou charge directement un cache sérialisé PKL.

    Args:
        path (str): Emplacement du dossier contenant l'ensemble des fichiers scripts (.txt).

    Returns:
        pd.DataFrame: DataFrame traité contenant le texte d'origine, nettoyé, les tokens, ainsi que le décompte de mots.
    """
    cache_file = os.path.join(path, "dataset_cache.pkl")
    
    if os.path.exists(cache_file):
        print(f"Chargement depuis le cache ({cache_file})...")
        return pd.read_pickle(cache_file)
        
    print("Parsing des fichiers originaux (première exécution)...")
    df = parserRepertoire(path)
    df['texte nettoyer'] = df['ligne'].apply(nettoyerTexte)
    df['token'] = df['texte nettoyer'].apply(tokeniserTexte)
    df['nombres de tokens'] = df['token'].apply(len)
    df['nombres de mots'] = df['texte nettoyer'].str.split().str.len()
    
    print(f"Sauvegarde dans le cache ({cache_file})...")
    df.to_pickle(cache_file)
    return df

def prendreDonneeParEp(df):
    """Calcule des indicateurs statistiques condensés regroupés par épisode unique (nombre d'acteurs distincts, échanges, totaux et moyennes de mots).
    Args:
        df (pd.DataFrame): Le DataFrame complet des dialogues nettoyés.
    Returns:
        pd.DataFrame: Tableau agrégé par ['saison', 'episode'] contenant les métriques calculées.
    """
    return df.groupby(['saison','episode']).agg(
        nbActeurParEp =('acteur','nunique'), 
        nbEchangesParEp = ('texte nettoyer','count'),
        nbMotsTotal = ('nombres de mots','sum'), 
        moyenneMotsParEchanges = ('nombres de mots','mean')
    ).reset_index()

def prendreDonneeParActeur(df):
    """Génère un récapitulatif quantitatif de la prise de parole par intervenant (nombre total de répliques et volume cumulé de mots prononcés).
    Args:
        df (pd.DataFrame): Le DataFrame complet enrichi des répliques.
    Returns:
        pd.DataFrame: Tableau récapitulatif indexé par la variable unique 'acteur'.
    """
    return df.groupby(['acteur']).agg(
        nb_lignes = ('texte nettoyer','count'), 
        nb_mots_total = ('nombres de mots','sum')
    ).reset_index()

