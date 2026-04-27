# LIBRAIRIE 
from gensim import models
from gensim import corpora
import pandas as pd 
from tqdm import tqdm 

# FICHIER PY
import utils as ut
import stats_lexical as sl
import data_loader as dl

#============================================================================
#                               EXPORT DE DONNEES
#============================================================================

# DE DATA LOADER 

df = dl.chargerDonnees("../datasets")
etapes = [
    ("Donnees par acteur", lambda : dl.prendreDonneeParActeur(df)),
    ("Donnees par episodes", lambda : dl.prendreDonneeParEp(df)), 
    ("Stats mots", lambda : sl.donnerStatsMots(df)), 
    ("Frequence mots par acteur", lambda : sl.freqMotsParActeur(df)), 
    ("Frequence mots par episodes", lambda : sl.freqMotsParEpisode(df)),
    ("Frequence mots par conversation", lambda : sl.freqMotsParConversation(df)),
]

resultats = {}
for nom, fn in tqdm(etapes, desc="Chargement de donnees",unit="etape"): 
    resultats[nom] = fn()

df_episodes = resultats["Donnees par episodes"]
df_acteur = resultats["Donnees par acteur"]
stats_mots = resultats["Stats mots"]
fma = resultats["Frequence mots par acteur"]
fme = resultats["Frequence mots par episodes"]
fmc = resultats["Frequence mots par conversation"]
topMots = stats_mots['mot'].head(10).tolist()
freq = sl.freqMotsAuCoursDuTemps(df,topMots)

# EXPORT EN CSV

df.to_csv("df.csv")
df_acteur.to_csv("df_acteur.csv")
df_episodes.to_csv("df_episodes.csv")
stats_mots.to_csv("stats_mots.csv")
fma.to_csv("fma.csv")
fme.to_csv("fme.csv")
fmc.to_csv("fmc.csv")

# EXPORT EN PNG 
ut.plotEvolutionMots(freq)
ut.plotFreqMotsParActeur(fma)