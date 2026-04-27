# ==========================================================
# LIBRAIRIES ET MODULES
# ==========================================================
import pandas as pd
from tqdm import tqdm
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
    ttMots = ut.retournerTokens(df)
    compteur = Counter(ttMots)
    total = sum(compteur.values())
    
    ligne = []
    for mot,fois in compteur.most_common(): 
        ligne.append({
                'mot' : mot,
                'nb_fois': fois,
                'frequence_%': round( fois/total *100,4)
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
# Frequence relative par rapport au nombre total des tokens utilise par l'acteur 
# Retourne un dataframe [acteur,mot,nb_fois,frequence_%]
def freqMotsParActeur(df): 
    lignes = []

    for acteur,groupe in df.groupby('acteur'): 
        ttMots = ut.retournerTokens(groupe)
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
# Frequence relative par rapport au nombre total de tokens par episode
# Retourne un dataframe [saison,episode,mot,nb_fois,frequence]
def freqMotsParEpisode(df):
    lignes = []
    for (saison,episode), groupe in df.groupby(['saison','episode']):
        ttMots = ut.retournerTokens(groupe)
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
        ttMots = ut.retournerTokens(groupe)
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
    return nvDf

#======================================================================
# LDA : LATENT DIRICHLET ALLOCATION 
#======================================================================

def trouverNbOptimalTopics(df, min_topics=5, max_topics=15, min_tokens_par_scene=0,
    alpha_values=[0.3, 0.4, 0.5, 0.6, 0.7, 'auto']):

    tokens_par_scene = (
        df.groupby(['saison', 'episode', 'scene_num'])['token']
        .apply(lambda rows: [t for tokens in rows for t in tokens])
        .tolist()
    )
    mots_par_scene = (
        df.groupby(['saison', 'episode', 'scene_num'])['texte nettoyer']
        .apply(lambda rows: ' '.join(rows).split())
        .tolist()
    )
    scenes = [(tok, mots) for tok, mots in zip(tokens_par_scene, mots_par_scene)
              if len(tok) >= min_tokens_par_scene]
    textes = [s[0] for s in scenes]
    textes_ref = [s[1] for s in scenes]

    dictionnaire = corpora.Dictionary(textes)
    dictionnaire.filter_extremes(no_below=5, no_above=0.45)
    corpus = [dictionnaire.doc2bow(texte) for texte in textes]
    dictionnaire_ref = corpora.Dictionary(textes_ref)

    scores = []
    meilleur = {'coherence': -1}
    total = (max_topics - min_topics + 1) * len(alpha_values)

    with tqdm(total=total, desc="  LDA", unit="combi") as pbar:
        for n in range(min_topics, max_topics + 1):
            for alpha in alpha_values:
                pbar.set_postfix(n_topics=n, alpha=str(alpha))

                modele = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionnaire,
                    num_topics=n,
                    passes=20,
                    alpha=alpha,
                    eta='auto',
                    random_state=42,
                    iterations=200
                )
                coherence = models.CoherenceModel(
                    model=modele,
                    texts=textes_ref,
                    dictionary=dictionnaire_ref,
                    coherence='c_v',
                    processes=1
                ).get_coherence()
                perplexite = round(np.exp(-modele.log_perplexity(corpus)), 2)
                scores.append({
                    'n_topics': n, 'alpha': alpha,
                    'coherence': round(coherence, 4),
                    'perplexite': perplexite
                })
                if coherence > meilleur['coherence']:
                    meilleur = {
                        'n_topics': n, 'alpha': alpha,
                        'coherence': round(coherence, 4),
                        'perplexite': perplexite
                    }
                pbar.update(1)

    print(f"\nMeilleure combinaison : n_topics={meilleur['n_topics']}, alpha={meilleur['alpha']} "
          f"(coherence={meilleur['coherence']}, perplexite={meilleur['perplexite']})")
    return meilleur, scores


def ldaParSaison(df, meilleurs_par_saison, min_tokens_par_scene=20):
    resultats = {}

    for season_id, df_saison in df.groupby('saison'):
        print(f"\n--- Saison {season_id} ---")

        if season_id not in meilleurs_par_saison:
            print(f"  Pas de parametres optimaux, saison ignoree.")
            continue

        meilleur = meilleurs_par_saison[season_id]

        # Grouper les tokens par scene
        textes = (
            df_saison.groupby(['episode', 'scene_num'])['token']
            .apply(lambda rows: [t for tokens in rows for t in tokens])
            .tolist()
        )
        textes = [t for t in textes if len(t) >= min_tokens_par_scene]

        if not textes:
            print(f"  Aucune scene suffisante, saison ignoree.")
            continue

        print(f"  Scenes utilisees : {len(textes)}")
        print(f"  Parametres : n={meilleur['n_topics']}, alpha={meilleur['alpha']}")

        dictionnaire = corpora.Dictionary(textes)
        dictionnaire.filter_extremes(no_below=5, no_above=0.45)
        corpus = [dictionnaire.doc2bow(t) for t in textes]

        try:
            modele = models.LdaModel(
                corpus=corpus, id2word=dictionnaire,
                num_topics=meilleur['n_topics'], passes=20, iterations=200,
                alpha=meilleur['alpha'], eta='auto', random_state=42,
            )
            resultats[season_id] = modele

            for idx, topic in modele.print_topics(num_words=10):
                print(f"  Topic {idx}: {topic}")

        except Exception as e:
            print(f"  Erreur saison {season_id}: {e}")

    print("\nEntrainement termine.")
    return resultats
