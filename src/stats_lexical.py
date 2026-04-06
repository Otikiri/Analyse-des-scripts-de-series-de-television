# ==========================================================
# LIBRAIRIES ET MODULES
# ==========================================================
import pandas as pd
from collections import Counter
from gensim import corpora
from gensim import models
import matplotlib.pyplot as plt
import numpy as np
# Module dependencies 
import utils as ut

#---------------------------------------------------------------
# STATS GLOBAL DE MOTS
#---------------------------------------------------------------

# Tableau des statistiques des mots
# Donne les statistiques globale sur tout les mots de la serie/saison
# Retourne un dataframe [mot, nb_fois,frequence]
def donnerStatsMots(df): 
    ttMots = ut.retournerTtMots(df)
    compteur = Counter(ttMots)
    total = sum(compteur.values())
    
    ligne = []
    for mot,fois in compteur.most_common(): 
        ligne.append({
                'mot' : mot,
                'nb_fois': fois,
                'frequence': round( fois/total *100,4)
            }
        )
    return pd.DataFrame(ligne)

#---------------------------------------------------------------
# FREQ DE MOTS PAR CONVERSATIONS
#---------------------------------------------------------------

# Analyse de frequence de mots par conversation
# Frequence des mots dans toute les conversations de la serie/saison
# Retourne un dataFrame avec colonne [id_conv, acteur, nb_fois, frequence_%]
def freqMotsParConversation(df):
    lignes = []
    for id, ligne in df.iterrows(): 
        compteur = Counter(ligne['token'])
        total = sum(compteur.values())
        for mot, comp in compteur.items(): 
            lignes.append({
                'id_conv' : id,
                'acteur' : ligne['acteur'],
                'mot': mot,
                'nb_fois' : comp,
                'frequence_%' : round(comp/total * 100,4) if total > 0 else 0  
            })
    fmc=pd.DataFrame(lignes)
    return fmc

#---------------------------------------------------------------
# FREQ DE MOTS PAR ACTEUR
#---------------------------------------------------------------

# Analyse de mots par acteur
# Frequence relative par rapport au nombre total des mots utilise par l'acteur 
# Retourne un dataframe [acteur,mot,nb_fois,frequence_%]
def freqMotsParActeur(df): 
    lignes = []

    for acteur,groupe in df.groupby('acteur'): 
        ttMots = ut.retournerTtMots(groupe)
        compteur = Counter(ttMots)
        total = sum(compteur.values())
    
        for mot, compt in compteur.most_common(20): 
            lignes.append({
                'acteur' : acteur,
                'mot': mot,
                'nb_fois' : compt,
                'frequence_%' : round(compt/total * 100,4)  
            })
    fma = pd.DataFrame(lignes)
    return fma

#---------------------------------------------------------------
# FREQ DE MOTS PAR EPISODES
#---------------------------------------------------------------

# Analyse de mots par episode
# Frequence relative par rapport au nombre total de mots par episode
# Retourne un dataframe [saison,episode,mot,nb_fois,frequence]
def freqMotsParEpisode(df):
    lignes = []
    for (saison,episode), groupe in df.groupby(['saison','episode']):
        ttMots = ut.retournerTtMots(groupe)
        compteur = Counter(ttMots)
        total = sum(compteur.values())

        for mot, compt in compteur.most_common(20):
            lignes.append({
                'saison' : saison,
                'episode' : episode,
                'mot' : mot,
                'nb_fois' : compt, 
                'frequence_%' : round(compt/total * 100,4)  
            })
    fme = pd.DataFrame(lignes)
    return fme


#---------------------------------------------------------------
# ANALYSE FREQUENTIELLES DES MOTS AU LONG DU TEMPS
#---------------------------------------------------------------

# Analyse frequentielle de au cours de la serie
# Compte tout les mots dans la serie qui appartient a l'array 
# Retourne un dataframe [saison,mot,nb_fois,frequence]
def freqMotsAuCoursDuTemps(df, motsASuivre=['love', 'wedding', 'coffee', 'baby', 'job']):
    lignes = []
    for saison, groupe in df.groupby('saison'): 
        ttMots = ut.retournerTtMots(groupe)
        compteur = Counter(ttMots)
        total = sum(compteur.values())

        for mot in motsASuivre:
            compt = compteur.get(mot,0)
            lignes.append({
                'saison' : saison,
                'mot' : mot,
                'nb_fois' : compteur.get(mot,0), 
                'frequence_%' : round(compt/total * 100,4)  
            })

    nvDf = pd.DataFrame(lignes)
    ut.plotEvolutionMots(nvDf)
    return nvDf


# def retrouverAlphaLDA(df,nTopics=5,alpha_values=
#     [0.01,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,'auto']): 
#     df = df[df['token'].apply(len)>0]
#     textes = df['token'].tolist()
#     dictionnaire = corpora.Dictionary(textes)
#     dictionnaire.filter_extremes(no_below=5, no_above=0.5)
#     corpus = [dictionnaire.doc2bow(texte) for texte in textes]

#     scores = []
#     for alpha in (alpha_values): 
#         model = models.LdaModel(
#             corpus=corpus, 
#             num_topics=nTopics, 
#             id2word=dictionnaire,
#             passes=50, 
#             alpha=alpha,
#             eta='auto',
#             random_state=42,
#         )
#         coherence = models.CoherenceModel(
#             model=model, 
#             texts=textes, 
#             dictionary=dictionnaire, 
#             coherence='c_v'
#         ).get_coherence()
#         bound =model.log_perplexity(corpus)
#         perplexity = np.exp2(-bound)
#         scores.append({'alpha':alpha,'coherence':round(coherence,4),'perplexity':round(perplexity,4)})
#         print(f"alpha={alpha} → coherence={round(coherence, 4)} | perplexity={round(perplexity,4)}")
    
#     #fonction de plot 
#     scores_df = pd.DataFrame(scores)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#     # use range(len()) for x positions instead of the alpha values directly
#     x = range(len(scores_df))
#     labels = scores_df['alpha'].astype(str)  # convert all to string for labels

#     ax1.bar(x, scores_df['perplexity'], color='steelblue')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(labels, rotation=45)
#     ax1.set_title('Perplexity by Alpha (lower = better)')
#     ax1.set_xlabel('Alpha')
#     ax1.set_ylabel('Perplexity')

#     ax2.bar(x, scores_df['coherence'], color='green')
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(labels, rotation=45)
#     ax2.set_title('Coherence by Alpha (higher = better)')
#     ax2.set_xlabel('Alpha')
#     ax2.set_ylabel('Coherence')

#     plt.suptitle(f'LDA Evaluation (n_topics={nTopics})')
#     plt.tight_layout()
#     plt.show()

#     return scores_df

def trouverNbOptimalTopics(df, min_topics=5, max_topics=15, min_tokens_par_scene=0,
    alpha_values=[0.3, 0.4, 0.5, 0.6, 0.7, 'auto']):

    # Grouper les tokens filtres par scene : utilises pour entrainer LDA
    tokens_par_scene = (
        df.groupby(['saison', 'episode', 'scene_num'])['token']
        .apply(lambda rows: [t for tokens in rows for t in tokens])
        .tolist()
    )
    # Grouper les mots bruts par scene : utilises comme reference pour la coherence
    # c_v a besoin de textes riches pour calculer les co-occurrences dans une fenetre glissante
    mots_par_scene = (
        df.groupby(['saison', 'episode', 'scene_num'])['texte nettoyer']
        .apply(lambda rows: ' '.join(rows).split())
        .tolist()
    )
    # On garde uniquement les scenes avec assez de tokens
    scenes = [(tok, mots) for tok, mots in zip(tokens_par_scene, mots_par_scene)
              if len(tok) >= min_tokens_par_scene]
    textes = [s[0] for s in scenes]      # tokens filtres → entrainement LDA
    textes_ref = [s[1] for s in scenes]  # mots bruts → reference coherence
    print(f"  Nombre de scenes utilisees : {len(textes)}")

    # Construire le dictionnaire gensim et filtrer les mots trop rares ou trop frequents
    # no_below=2 : mot doit apparaitre dans au moins 2 scenes
    # no_above=0.85 : mot ne doit pas apparaitre dans plus de 85% des scenes
    # seuils adaptes aux documents par scene (plus longs, mots plus repetes)
    dictionnaire = corpora.Dictionary(textes)
    dictionnaire.filter_extremes(no_below=2, no_above=0.85)

    # Construire le corpus : chaque scene devient une liste de (id_mot, frequence)
    corpus = [dictionnaire.doc2bow(texte) for texte in textes]

    # Construire le dictionnaire de reference une seule fois avant la boucle
    # evite de le reconstruire a chaque iteration (n x alpha fois)
    dictionnaire_ref = corpora.Dictionary(textes_ref)

    scores = []
    meilleur = {'coherence': -1}

    for n in range(min_topics, max_topics + 1):
        print(f"\n  n_topics={n}:")
        for alpha in alpha_values:
            # Entrainer le modele LDA avec n topics et alpha donne
            modele = models.LdaModel(
                corpus=corpus,
                id2word=dictionnaire,
                num_topics=n,
                passes=20,
                alpha=alpha,
                eta='auto',
                random_state=42,
            )

            # Calculer la coherence c_v
            # textes_ref : mots bruts pour les co-occurrences (plus riche que les tokens filtres)
            # processes=1 pour eviter les problemes de multiprocessing sur macOS
            coherence = models.CoherenceModel(
                model=modele,
                texts=textes_ref,
                dictionary=dictionnaire_ref,
                coherence='c_v',
                processes=1
            ).get_coherence()

            # Calculer la perplexite : exp(-log_perplexity)
            # Plus c'est bas, meilleur est le modele
            perplexite = round(np.exp(-modele.log_perplexity(corpus)), 2)

            scores.append({
                'n_topics': n, 'alpha': alpha,
                'coherence': round(coherence, 4),
                'perplexite': perplexite
            })
            print(f"    alpha={alpha} → coherence={round(coherence, 4)} | perplexite={perplexite:.2f}")

            # Garder la meilleure combinaison selon la coherence
            if coherence > meilleur['coherence']:
                meilleur = {
                    'n_topics': n, 'alpha': alpha,
                    'coherence': round(coherence, 4),
                    'perplexite': perplexite
                }

    print(f"\nMeilleure combinaison : n_topics={meilleur['n_topics']}, alpha={meilleur['alpha']} "
          f"(coherence={meilleur['coherence']}, perplexite={meilleur['perplexite']})")
    return meilleur, scores