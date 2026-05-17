# LIBRAIRIE 
from gensim import models
from gensim import corpora
import pandas as pd 
from tqdm import tqdm 

# FICHIER PY
import utils as ut
import stats_lexical as sl
import data_loader as dl
import cluster as cl
import moteur as mt 

#to ignore warnings from KeyBERT
import warnings
warnings.filterwarnings('ignore') 
#============================================================================
#                               EXPORT DE DONNEES
#============================================================================

# df = dl.chargerDonnees("../datasets")
# etapes = [
#     ("Donnees par acteur", lambda : dl.prendreDonneeParActeur(df)),
#     ("Donnees par episodes", lambda : dl.prendreDonneeParEp(df)), 
#     ("Stats mots", lambda : sl.donnerStatsMots(df)), 
#     ("Frequence mots par acteur", lambda : sl.freqMotsParActeur(df)), 
#     ("Frequence mots par episodes", lambda : sl.freqMotsParEpisode(df)),
#     ("Frequence mots par conversation", lambda : sl.freqMotsParConversation(df)),
# ]

# resultats = {}
# for nom, fn in tqdm(etapes, desc="Chargement de donnees",unit="etape"): 
#     resultats[nom] = fn()

# df_episodes = resultats["Donnees par episodes"]
# df_acteur = resultats["Donnees par acteur"]
# stats_mots = resultats["Stats mots"]
# fma = resultats["Frequence mots par acteur"]
# fme = resultats["Frequence mots par episodes"]
# fmc = resultats["Frequence mots par conversation"]
# topMots = stats_mots['mot'].head(10).tolist()
# freq = sl.freqMotsAuCoursDuTemps(df,topMots)

# # EXPORT EN CSV

# df.to_csv("df.csv")
# df_acteur.to_csv("df_acteur.csv")
# df_episodes.to_csv("df_episodes.csv")
# stats_mots.to_csv("stats_mots.csv")
# fma.to_csv("fma.csv")
# fme.to_csv("fme.csv")
# fmc.to_csv("fmc.csv")

# # EXPORT EN PNG 
# ut.plotEvolutionMots(freq)
# ut.plotFreqMotsParActeur(fma)
# for i in range(1,10):
#     df = dl.chargerDonnees("../datasets/S0"+str(i))
#     ut.trouverNbOptimalTopics(df,min_topics=2,max_topics=8,min_tokens_par_scene=20,alpha_values=[0.01,0.05,0.1,0.3,'auto'],id=str(i))
# df = dl.chargerDonnees("../datasets/S10")
# ut.trouverNbOptimalTopics(df,min_topics=2,max_topics=8,min_tokens_par_scene=20,alpha_values=[0.01,0.05,0.1,0.3,'auto'],id='10')

#============================================================================
#                             MOTEUR DE RECHERCHE
#============================================================================
# df = dl.chargerDonnees("../datasets/S01") 
# meilleur = pd.read_csv("m_csv1.csv")
# print(meilleur)
# meilleurs_par_saison = {
#     str(int(row['saison'])).zfill(2): {
#         'n_topics': int(row['n_topics']),
#         'alpha': row['alpha'],
#         'coherence': row['coherence'],
#         'perplexite': row['perplexite']
#     }
#     for _, row in meilleur.iterrows()
# }
# print(meilleurs_par_saison)
# res = cl.clusteringLDA(df, meilleurs_par_saison)
# print("lda\n",mt.construireSujetsEpLDA(df,res))
# print("bert\n",mt.construireSujetsEpBERTopic(df))
# print("keybert\n",mt.construireSujetsEpKeyBERT(df))
# print("tfidf\n",mt.construireSujetsEpTFIDF(df))


df = dl.chargerDonnees("../datasets")
print(df.columns)
q1_test = [
    "Monica and Chandler announce their engagement. ", 
    "Rachel's first day at her new job with Mark.",
    "Joey learns to speak French for an audition.",
    "Phoebe wants to sing at Monica's wedding.", 
    "Ross is jealous of the gifts sent to Rachel's workplace."
]

q2_test = [
    "Drinking a gallon of milk in ten seconds.",
    "A poem about an empty vase written by a waiter.",
    "Playing a racing video game on PlayStation while dressing like a nineteen-year-old.",
    "Someone puts a turkey on their head to make people laugh.",
    "Eating a stolen cheesecake off the floor in the hallway."
]

for i in q2_test:
    print(i)
    print(mt.determiner_type_question(i,df))