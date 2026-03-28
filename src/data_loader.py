import pandas as pd
from parser import parserRepertoire
from utils import nettoyerTexte,tokeniserTexte

def chargerDonnees(path): 
    df = parserRepertoire(path)
    df['texte nettoyer'] = df['ligne'].apply(nettoyerTexte)
    df['token'] = df['texte nettoyer'].apply(tokeniserTexte)
    df['nombres de tokens'] = df['token'].apply(len)
    df['nombres de mots'] = df['texte nettoyer'].str.split().str.len()
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

