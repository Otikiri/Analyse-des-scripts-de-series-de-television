import re 
import cluster as cl 
import pandas as pd
import numpy as np 
from collections import Counter
from keybert import KeyBERT
# from bertopic import BERTopic

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import nettoyerTexte, tokeniserTexteRecherche
import q1_search as q1 
import q2_search as q2
import spacy 

nlp = spacy.load("en_core_web_sm")

import warnings
warnings.filterwarnings('ignore') 

#======================================================================
#                          CONSTRUCTION DE SUJET
#======================================================================

def construireSujetsEpTFIDF(df):
    
    km, docs, X_2d,vectorizer,X = cl.clusteringTfidf(
        df,
        groupby_cols=['saison', 'episode'],
        ngram_range=(1,1)
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
    
    kw_model = KeyBERT()
    
    # reuse clusteringTfidf to get docs with episode texts
    km, docs, X_2d,vectorizer,X = cl.clusteringTfidf(
        df,
        groupby_cols=['saison', 'episode'],
        ngram_range=(1,1)
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
            'sujet': ', '.join([kw for kw, score in keywords])
        })
    
    sujets_df = pd.DataFrame(sujets)
    sujets_df.to_csv('sujet_par_ep.csv', index=False)
    return sujets_df


#======================================================================
#                          RECHERCHE 
#======================================================================

# 1 - Construction de l'index tf-idf
# Combiner tous les token d'un groupe (episode ou scene) en texte
def combinerTokens(listes_de_tokens):
    tous_les_mots = []
    for liste in listes_de_tokens:
        for mot in liste:
            tous_les_mots.append(mot)
    return ' '.join(tous_les_mots)


# Creation de la matrice TF-IDF
# A appeler en premier avant toute recherche, construit l'index TF-IDF
def construireIndex(df, par_scene=False):

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


# 2 - Vectorisation d'une requete
# Prend une question en langage naturel,
# la nettoie et la tokenise avec les memes fonctions que le reste du projet,
# puis la transforme en vecteur TF-IDF avec le vectorizer précédent.

def vectoriserRequete(requete, vectorizer, motsVidesRecherche):
    texte_propre = nettoyerTexte(requete)
    tokens = tokeniserTexteRecherche(texte_propre, motsVidesRecherche=motsVidesRecherche)
    texte_final = ' '.join(tokens)
    vecteur = vectorizer.transform([texte_final])
    return vecteur


# 3 - Similarité par cos
# Calcule la similarite cosinus entre un vecteur requete et chaque ligne de la matrice documents.
# Formule : cos(A, B) = (A . B) / (||A|| * ||B||)
def similariteCosinus(vecteur_requete, matrice_documents):
    scores = cosine_similarity(vecteur_requete, matrice_documents)
    return scores.flatten() 


# 4 - Recherche
# question -> nettoyage -> vectorisation -> similarite cosinus -> tri -> resultats
# Cherche les top_k documents les plus proches d'une requete en langage naturel
def rechercher(requete, vectorizer, matrice_tfidf, df_docs, df, top_k=5, motsVidesRecherche=None):
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
# fonction d'affichage commune des resulats
def miseEnFormeRes(res,sujet_df):
    d = {
        "question":res['question'],
        "saison":res['saison'],
        "episode":res['episode'],
        "score":res['score']
        }
    df = pd.DataFrame(d)
    df['rank'] = range(1,len(df)+1)

    df = df.merge(sujet_df[['saison', 'episode', 'sujet']], 
                  on=['saison', 'episode'], 
                  how='left')
    
    # reorder columns
    df = df[['rank', 'question', 'saison', 'episode', 'score', 'sujet']]
    return df

#======================================================================
#                        EVALUATION DU MODELE
#======================================================================

def calculerMRR(question_verite,resultats_df):
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