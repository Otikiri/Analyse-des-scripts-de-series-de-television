import pandas as pd
from parser import parserRepertoire
from utils import nettoyerTexte,tokeniserTexte
import os

def chargerDonnees(path): 
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
    return df.groupby(['saison','episode']).agg(
        nbActeurParEp =('acteur','nunique'), 
        nbEchangesParEp = ('texte nettoyer','count'),
        nbMotsTotal = ('nombres de mots','sum'), 
        moyenneMotsParEchanges = ('nombres de mots','mean')
    ).reset_index()

def prendreDonneeParActeur(df):
    return df.groupby(['acteur']).agg(
        nb_lignes = ('texte nettoyer','count'), 
        nb_mots_total = ('nombres de mots','sum')
    ).reset_index()

