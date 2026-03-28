# ==========================================================
# LIBRAIRIES ET MODULES
# ==========================================================
import pandas as pd
from collections import Counter
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


