# LIBRAIRIE 
from gensim import models
from gensim import corpora
import pandas as pd 
from tqdm import tqdm 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# FICHIER PY
import utils as ut
import stats_lexical as sl
import data_loader as dl

#============================================================================
#                             EXPORT DE DONNEES
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
for nom, fn in tqdm(etapes, desc="Chargement de donnees", unit="etape"): 
    resultats[nom] = fn()

df_episodes = resultats["Donnees par episodes"]
df_acteur = resultats["Donnees par acteur"]

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


# Fonction pour catégoriser le score
def get_sentiment_label(score):
    if score >= 0.05:
        return 'Positif'
    elif score <= -0.05:
        return 'Négatif'
    else:
        return 'Neutre'

analyzer = SentimentIntensityAnalyzer()
df['score_compound'] = df['ligne'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['sentiment_ligne'] = df['score_compound'].apply(get_sentiment_label)
df_sentiment_par_acteur = df.groupby('acteur')['score_compound'].mean().reset_index()
df_fusion = pd.merge(df_acteur, df_sentiment_par_acteur, on='acteur')
df_final = df_fusion[df_fusion['nb_lignes'] >= 167].copy()
df_final['sentiment_global'] = df_final['score_compound'].apply(get_sentiment_label)
df_final = df_final.sort_values(by='score_compound', ascending=False)

# ============================================================================
# COMPTAGE DES PHRASES POSITIVES, NEGATIVES ET NEUTRES
# ============================================================================

df_comptage = df.groupby(['acteur', 'sentiment_ligne']).size().unstack(fill_value=0).reset_index()

df_comptage.columns.name = None 

for col in ['Négatif', 'Neutre', 'Positif']:
    if col not in df_comptage.columns:
        df_comptage[col] = 0

df_final = pd.merge(df_final, df_comptage, on='acteur', how='left')

# ============================================================================
# CALCUL DES POURCENTAGES
# ============================================================================

df_final['%_Neutre'] = (df_final['Neutre'] / df_final['nb_lignes'] * 100).round(2)
df_final['%_Négatif'] = (df_final['Négatif'] / df_final['nb_lignes'] * 100).round(2)
df_final['%_Positif'] = (df_final['Positif'] / df_final['nb_lignes'] * 100).round(2)

# ============================================================================
# AFFICHAGE FILTRÉ
# ============================================================================

colonnes_a_afficher = [
    'acteur', 
    'Neutre', 
    'Négatif', 
    'Positif', 
    '%_Neutre', 
    '%_Négatif', 
    '%_Positif'
]

df_affichage = df_final[colonnes_a_afficher]

print("\n=== Dataframe Final Acteur Principaux ===")
print(df_affichage.to_string(index=False))
