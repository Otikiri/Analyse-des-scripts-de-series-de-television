import cluster as cl 
import pandas as pd
import numpy as np 

from keybert import KeyBERT
# from bertopic import BERTopic

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import nettoyerTexte, tokeniserTexteRecherche
import spacy 

nlp = spacy.load("en_core_web_sm")

import warnings
warnings.filterwarnings('ignore') 

#======================================================================
#                          CONSTRUCTION DE SUJET
#======================================================================

def construireSujetsEpTFIDF(df):
    """Génère une synthèse des sujets principaux par épisode en extrayant les 5 mots possédant la valeur TF-IDF la plus élevée, puis l'enregistre au format CSV.

    Args:
        df (pd.DataFrame): Dataframe source contenant les lignes de scripts.

    Returns:
        pd.DataFrame: Un DataFrame contenant les colonnes ['saison', 'episode', 'sujet'].
    """

    km, docs, X_2d,vectorizer,X = cl.clusteringTfidf(
        df,
        groupbyCols=['saison', 'episode'],
        ngramRange=(1,1)
    )
    
    terms = np.array(vectorizer.get_feature_names_out())
    
    sujets = []
    for idx, row in docs.iterrows():
        

        tfidf_top = X[idx].toarray()[0].argsort()[-5:][::-1]
        tfidf_words = [terms[i] for i in tfidf_top]
        
        sujets.append({
            'saison': row['saison'],
            'episode': row['episode'],
            'sujet': ', '.join(tfidf_words)
        })
    
    sujets_df = pd.DataFrame(sujets)
    sujets_df.to_csv('sujets_episodes_tfidf.csv', index=False)
    return sujets_df

def construireSujetsEpKeyBERT(df):
    """Extrait des mots-clés thématiques diversifiés par épisode en utilisant des représentations d'embeddings KeyBERT (modèle MMR), puis l'enregistre au format CSV.

    Args:
        df (pd.DataFrame): Dataframe source des dialogues de la série.

    Returns:
        pd.DataFrame: Un DataFrame contenant les colonnes ['saison', 'episode', 'titre', 'sujet'].
    """

    kw_model = KeyBERT()
    
    # reuse clusteringTfidf to get docs with episode texts
    km, docs, X_2d,vectorizer,X = cl.clusteringTfidf(
        df,
        groupbyCols=['saison', 'episode'],
        ngramRange=(1,1)
    )

    sujets = []
    for idx, row in docs.iterrows():
        keywords = kw_model.extract_keywords(
            row['doc'],                      # episode text already built
            keyphrase_ngram_range=(1,1),
            stop_words='english',
            top_n=5,
            use_mmr=True,    # forces diverse results
            diversity=0.7    # 0=no diversity, 1=maximum diversity
        )
        sujets.append({
            'saison': row['saison'],
            'episode': row['episode'],
            'titre': df[(df['episode']==row['episode']) & (df['saison']==row["saison"])]['nom fichier'].values[0],
            'sujet': ', '.join([kw for kw, score in keywords])
        })
    
    sujets_df = pd.DataFrame(sujets)
    sujets_df.to_csv('sujet_par_ep.csv', index=False)
    return sujets_df


#======================================================================
#                          RECHERCHE 
#======================================================================


def combinerTokens(listes_de_tokens):
    """Fusionne une structure imbriquée de listes de jetons textuels en une unique chaîne de caractères délimitée par des espaces.

    Args:
        listes_de_tokens (list): Une liste contenant des sous-listes de tokens (mots).

    Returns:
        str: Le texte complet regroupé.
    """

    tous_les_mots = []
    for liste in listes_de_tokens:
        for mot in liste:
            tous_les_mots.append(mot)
    return ' '.join(tous_les_mots)


def construireIndex(df, par_scene=False):
    """Construit la matrice et l'index de recherche inversé TF-IDF sur le corpus textuel, nettoyé de ses entités de personnages.

    Args:
        df (pd.DataFrame): Dataframe global contenant le texte d'origine et la variable 'acteur'.
        par_scene (bool, optional): Si True, indexe au niveau granulaire de la scène, sinon agrège par épisode. Valeur par défaut : False.

    Returns:
        tuple: Un ensemble de variables d'indexation contenant :
            - vectorizer (TfidfVectorizer) : Le modèle de transformation vectoriel ajusté.
            - matrice_tfidf (scipy.sparse.csr_matrix) : La matrice de poids TF-IDF résultante.
            - df_docs (pd.DataFrame) : Le référentiel des documents associés contenant les textes consolidés.
            - motsVidesRecherche (set) : L'ensemble des stop-words appliqués à l'exclusion des protagonistes.
    """
    # On choisit de travailler par scene ou par episode
    if par_scene:
        colonnes_groupe = ['saison', 'episode', 'scene_num']
    else:
        colonnes_groupe = ['saison', 'episode']

    # On regroupe les tokens et on les combine en un seul texte par document
    from utils import motVide
    nomsPersonnages = set(df['acteur'].str.lower().unique())
    motsVidesRecherche = motVide - nomsPersonnages

    df_temp = df.copy()
    df_temp['token_recherche'] = df_temp['texte nettoyer'].apply(
        lambda t: tokeniserTexteRecherche(t, motsVidesRecherche=motsVidesRecherche)
    )
    df_docs = df_temp.groupby(colonnes_groupe)['token_recherche'].apply(combinerTokens).reset_index()
    df_docs = df_docs.rename(columns={'token_recherche': 'texte'})

    documents_non_vides = df_docs['texte'].str.strip() != ''
    df_docs = df_docs[documents_non_vides].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'(?u)\b\w+\b',
        sublinear_tf=True
    )
    matrice_tfidf = vectorizer.fit_transform(df_docs['texte'])

    print(f"Index TF-IDF construit : {matrice_tfidf.shape[0]} documents, {matrice_tfidf.shape[1]} termes")
    return vectorizer, matrice_tfidf, df_docs, motsVidesRecherche


def vectoriserRequete(requete, vectorizer, motsVidesRecherche):
    """Nettoie, normalise et projette une requête textuelle formulée en langage naturel dans l'espace vectoriel TF-IDF configuré.

    Args:
        requete (str): La phrase ou question brute de l'utilisateur.
        vectorizer (TfidfVectorizer): L'indexeur TF-IDF ajusté au préalable.
        motsVidesRecherche (set): Le dictionnaire d'exclusion de mots.

    Returns:
        scipy.sparse.csr_matrix: Un vecteur creux à une ligne représentant le profil TF-IDF de la requête.
    """

    texte_propre = nettoyerTexte(requete)
    tokens = tokeniserTexteRecherche(texte_propre, motsVidesRecherche=motsVidesRecherche)
    texte_final = ' '.join(tokens)
    vecteur = vectorizer.transform([texte_final])
    return vecteur


def similariteCosinus(vecteur_requete, matrice_documents):
    """Calcule la métrique d'alignement ou de proximité angulaire par cosinus entre un vecteur requête et l'ensemble des documents de l'index.

    Args:
        vecteur_requete (scipy.sparse.csr_matrix): Le vecteur de la question d'entrée.
        matrice_documents (scipy.sparse.csr_matrix): La matrice globale de documents TF-IDF.

    Returns:
        np.ndarray: Un tableau 1D regroupant l'ensemble des scores de similarité compris entre 0.0 et 1.0.
    """
    scores = cosine_similarity(vecteur_requete, matrice_documents)
    return scores.flatten() 


def rechercher(requete, vectorizer, matrice_tfidf, df_docs, df, top_k=5, motsVidesRecherche=None):
    """Orchestre la chaîne complète d'analyse pour retourner les documents du corpus les plus pertinents vis-à-vis d'une requête utilisateur.

    Args:
        requete (str): La requête saisie en langage naturel.
        vectorizer (TfidfVectorizer): Le vectoriseur entraîné.
        matrice_tfidf (scipy.sparse.csr_matrix): L'index des poids de termes.
        df_docs (pd.DataFrame): Le tableau référentiel de correspondances de documents.
        df (pd.DataFrame): Dataframe original de travail.
        top_k (int, optional): Nombre maximal de résultats ordonnés à extraire. Valeur par défaut : 5.
        motsVidesRecherche (set, optional): Le dictionnaire d'exclusion de stop-words personnalisé. Valeur par défaut : None.

    Returns:
        pd.DataFrame: Extrait trié par score décroissant restreint aux correspondances strictes (score > 0).
    """
    vecteur = vectoriserRequete(requete, vectorizer, motsVidesRecherche)
    scores = similariteCosinus(vecteur, matrice_tfidf)

    resultats = df_docs.copy()
    resultats['score'] = scores

    resultats = resultats.sort_values('score', ascending=False).head(top_k)
    resultats = resultats[resultats['score'] > 0]
    resultats['question']=requete
    return resultats.reset_index(drop=True)


#======================================================================
#                       MISE EN FORME RESULTAT
#======================================================================

def miseEnFormeRes(res,sujet_df):
    """Met en forme les résultats bruts d'un moteur de recherche en y injectant les rangs, titres d'épisodes et mots-clés thématiques associés.

    Args:
        res (pd.DataFrame): Résultats retournés par la fonction `rechercher`.
        sujet_df (pd.DataFrame): Référentiel thématique préalablement généré contenant les résumés ou titres.

    Returns:
        pd.DataFrame: Un DataFrame structuré et réordonné contenant les colonnes : ['rank', 'question', 'saison', 'episode', 'titre', 'score', 'sujet'].
    """
    d = {
        "question":res['question'],
        "saison":res['saison'],
        "episode":res['episode'],
        "score":res['score']
        }
    df = pd.DataFrame(d)
    df['rank'] = range(1,len(df)+1)

    df = df.merge(sujet_df[['saison', 'episode', 'sujet', 'titre']], 
                  on=['saison', 'episode'], 
                  how='left')
    
    # reorder columns
    df = df[['rank', 'question', 'saison', 'episode','titre', 'score', 'sujet']]
    return df

#======================================================================
#                        EVALUATION DU MODELE
#======================================================================

def calculerMRR(question_verite,resultats_df):
    """Calcule le score de performance MRR (Mean Reciprocal Rank) du système de recherche face à un référentiel de vérité terrain.

    Args:
        question_verite (dict): Un dictionnaire associant chaque question à la liste des identifiants uniques attendus (`'saison_episode'`).
        resultats_df (pd.DataFrame): L'ensemble agrégé des résultats prédits par l'application pour ces requêtes.

    Returns:
        float: La moyenne globale des rangs réciproques calculée (valeur entre 0 et 1).
    """
    score_rr = []

    for question, v_ep in question_verite.items():

        res_question = resultats_df[resultats_df['question']==question].copy()
        res_question['ep_id'] = res_question['saison']+'_'+res_question['episode']

        # print(f"Q: {question}")
        # print(f"  expected: {v_ep}")
        # print(f"  got ep_ids: {res_question['ep_id'].tolist()}")
        # print(f"  ranks: {res_question['rank'].tolist()}")
              
        rr = 0
        for _, row in res_question.iterrows():
            if row['ep_id'] in v_ep: 
                rr = 1/ row['rank']
                break 
        
        score_rr.append(rr)
        print(f"Q: {question}")
        print(f"   RR: {round(rr, 4)}")
    
    mrr = np.average(score_rr)
    print(f"\nMRR global: {round(mrr, 4)}")
    return mrr