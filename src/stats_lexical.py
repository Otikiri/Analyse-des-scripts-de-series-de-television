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

def donnerStatsMots(df): 
    """Calcule les statistiques globales d'occurrences et de fréquences relatives de l'intégralité des mots du corpus.
    Args:
        df (pd.DataFrame): Dataframe source contenant la colonne 'token'.
    Returns:
        pd.DataFrame: Tableau structuré contenant les colonnes ['mot', 'nb_fois', 'frequence_%'] classé par ordre décroissant de présence.
    """

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

def freqMotsParConversation(df):
    """Analyse la composition lexicale de chaque interaction/ligne de dialogue de manière isolée pour en extraire la fréquence relative locale.
    Args:
        df (pd.DataFrame): Dataframe source contenant les listes de tokens et l'identité des acteurs.
    Returns:
        pd.DataFrame: Un dictionnaire tabulaire structuré contenant les colonnes : ['id_conv', 'acteur', 'mot', 'nb_fois', 'frequence_%'].
    """

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

def freqMotsParActeur(df): 
    """Identifie le top 20 des expressions et mots les plus caractéristiques prononcés par chaque personnage par rapport à son vocabulaire global.
    Args:
        df (pd.DataFrame): Le DataFrame complet des lignes de scripts.
    Returns:
        pd.DataFrame: Tableau associatif structuré contenant les colonnes ['acteur', 'mot', 'nb_fois', 'frequence_%'].
    """
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

def freqMotsParEpisode(df):
    """Calcule le profil de distribution des 20 mots les plus denses pour chaque épisode unique de la série.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée regroupant les saisons et épisodes.

    Returns:
        pd.DataFrame: Tableau détaillé contenant les variables ['saison', 'episode', 'mot', 'nb_fois', 'frequence_%'].
    """

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

def freqMotsAuCoursDuTemps(df, motsASuivre=['love', 'wedding', 'coffee', 'baby', 'job']):
    """Mesure et suit l'évolution macro-temporelle de la fréquence d'apparition d'un groupe de marqueurs thématiques spécifiques à travers les saisons.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée structuré par saison.
        motsASuivre (list, optional): Liste des chaînes de caractères à pister dans l'index. Valeur par défaut : ['love', 'wedding', 'coffee', 'baby', 'job'].

    Returns:
        pd.DataFrame: Un tableau chronologique contenant les colonnes ['saison', 'mot', 'nb_fois', 'frequence_%'].
    """
    
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

