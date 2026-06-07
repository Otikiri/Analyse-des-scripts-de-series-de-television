import numpy as np
import pandas as pd
import matplotlib as plt
from nltk.tokenize import word_tokenize

import data_loader as dl
import cluster as cl

df = dl.chargerDonnees("./datasets")

df['code episode'] = df['nom fichier'].str.extract(r"(S\d+E\d+)")

saison = input("Numéro de la saison : ").zfill(2)
episode = input("Numero de l'épisode : ").zfill(2)
personnage = input("Nom du personnage : ")

resultat = df[
    (df['code episode'] == "S"+saison+"E"+episode) &
    (df['acteur'].str.lower() == personnage.lower())
]

model, dfCluster, vecteur = cl.clusteringW2v(df, resultat)
dfCluster.to_csv("result.csv")


total_mots = resultat['nombres de mots'].sum()
print(personnage + " a prononcé " + str(total_mots) + " mots dans l'épisode " + episode + " de la saison " + saison)