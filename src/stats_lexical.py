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


def retrouverAlphaLDA(df,nTopics=5,alpha_values=
    [0.01,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,'auto']): 
    df = df[df['token'].apply(len)>0]
    textes = df['token'].tolist()
    dictionnaire = corpora.Dictionary(textes)
    dictionnaire.filter_extremes(no_below=10, no_above=0.3)
    corpus = [dictionnaire.doc2bow(texte) for texte in textes]

    scores = []
    for alpha in (alpha_values): 
        model = models.LdaModel(
            corpus=corpus, 
            num_topics=nTopics, 
            id2word=dictionnaire,
            passes=20, 
            alpha=alpha,
            random_state=42,
        )
        coherence = models.CoherenceModel(
            model=model, 
            texts=textes, 
            dictionary=dictionnaire, 
            coherence='c_v'
        ).get_coherence()
        bound =model.log_perplexity(corpus)
        perplexity = np.exp2(-bound)
        scores.append({'alpha':alpha,'coherence':round(coherence,4),'perplexity':round(perplexity,4)})
        print(f"alpha={alpha} → coherence={round(coherence, 4)} | perplexity={round(perplexity,4)}")
    
    #fonction de plot 
    scores_df = pd.DataFrame(scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # use range(len()) for x positions instead of the alpha values directly
    x = range(len(scores_df))
    labels = scores_df['alpha'].astype(str)  # convert all to string for labels

    ax1.bar(x, scores_df['perplexity'], color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_title('Perplexity by Alpha (lower = better)')
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Perplexity')

    ax2.bar(x, scores_df['coherence'], color='green')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_title('Coherence by Alpha (higher = better)')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Coherence')

    plt.suptitle(f'LDA Evaluation (n_topics={nTopics})')
    plt.tight_layout()
    plt.show()

    return scores_df

def retrouverNTopics(df, topic_range=range(2, 15)):
    df = df[df['token'].apply(len)>0]
    textes = df['token'].tolist()
    dictionnaire = corpora.Dictionary(textes)
    dictionnaire.filter_extremes(no_below=10, no_above=0.3)
    corpus = [dictionnaire.doc2bow(texte) for texte in textes]

    scores = []
    for n in topic_range:
        model =models.LdaModel(
            corpus=corpus,
            num_topics=n,
            id2word=dictionnaire,
            passes=50,
            alpha=0.3,      # best from your results
            random_state=42
        )
        coherence = models.CoherenceModel(
            model=model,
            texts=textes,
            dictionary=dictionnaire,
            coherence='c_v'
        ).get_coherence()

        scores.append({'n_topics': n, 'coherence': round(coherence, 4)})
        print(f"n_topics={n} → coherence={round(coherence, 4)}")

    scores_df = pd.DataFrame(scores)

    plt.figure(figsize=(10, 5))
    plt.plot(scores_df['n_topics'], scores_df['coherence'], marker='o', color='steelblue')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Coherence by Number of Topics')
    plt.tight_layout()
    plt.show()

    best = scores_df.loc[scores_df['coherence'].idxmax()]
    print(f"\nBest n_topics={best['n_topics']} → coherence={best['coherence']}")
    return scores_df