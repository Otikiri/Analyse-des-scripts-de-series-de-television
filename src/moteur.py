
import re 
import cluster as cl 
import pandas as pd
import numpy as np 
from collections import Counter
from keybert import KeyBERT
from bertopic import BERTopic

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import nettoyerTexte, tokeniserTexte, tokeniserTexteRecherche
import q1_search as q1 
import q2_search as q2
import spacy

nlp = spacy.load("en_core_web_sm")

import warnings
warnings.filterwarnings('ignore') 

#======================================================================
#                          CONSTRUCTION DE SUJET
#======================================================================
def construireSujetsEpLDA(df, resLDA):
    sujets = []

    for (saison, episode), groupe in df.groupby(['saison', 'episode']):

        if saison not in resLDA:
            continue

        modele = resLDA[saison]
        dictionnaire = modele.id2word
        voteSuj = Counter()

        for scene, grpScn in groupe.groupby('scene_num'):
            tokens = [w for tokens in grpScn['token'] for w in tokens]
            if len(tokens) < 20:
                continue
            bow = dictionnaire.doc2bow(tokens)  
            topics = modele.get_document_topics(bow)
            if topics:
                dominant = max(topics, key=lambda x: x[1])
                voteSuj[dominant[0]] += 1

        if voteSuj:
            best_topic = voteSuj.most_common(1)[0][0]
            top_words = [w for w, p in modele.show_topic(best_topic, topn=5)]
            sujets.append({
                'saison': saison,
                'episode': episode,
                'sujet': ', '.join(top_words)
            })

    sujets_df = pd.DataFrame(sujets)
    sujets_df.to_csv('sujets_episodes.csv', index=False)
    return sujets_df

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
    sujets_df.to_csv('sujets_episodes.csv', index=False)
    return sujets_df

def construireSujetsEpBERTopic(df):
    
    # reuse clusteringTfidf to get docs
    km, docs, X, vectorizer, X_tfidf = cl.clusteringTfidf(
        df,
        groupby_cols=['saison', 'episode'],
        ngram_range=(1,1)
    )
    
    corpus = docs['doc'].tolist()
    
    # train BERTopic
    # reduce min_topic_size for small datasets
    topic_model = BERTopic(
        language='english',
        min_topic_size=2,      # default is 10, too high for 24 docs
        calculate_probabilities=True
)
    topics, probs = topic_model.fit_transform(corpus)
    
    # get topic labels
    docs['topic_id'] = topics
    
    sujets = []
    for idx, row in docs.iterrows():
        topic_id = row['topic_id']
        
        if topic_id == -1:  # -1 means outlier in BERTopic
            sujet = 'divers'
        else:
            top_words = [w for w, score in topic_model.get_topic(topic_id)[:5]]
            sujet = ', '.join(top_words)
        
        sujets.append({
            'saison': row['saison'],
            'episode': row['episode'],
            'sujet': sujet
        })
    
    sujets_df = pd.DataFrame(sujets)
    sujets_df.to_csv('sujets_episodes.csv', index=False)
    return sujets_df, topic_model

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
    return resultats.reset_index(drop=True)



#======================================================================
#                        DETERMINATION DE TYPE
#======================================================================

ACTEURS_FRIENDS = ["ross", "rachel", "monica", "chandler", "joey", "phoebe", "central perk"]

def determiner_type_question(question,df):
    """
    Détermine si une question est de type Q1 (avec entités) ou Q2 (sans entités).
    Retourne le type ("Q1" ou "Q2") et la liste des entités trouvées.
    """
    if nlp is None:
        return "Erreur", []
        
    doc = nlp(question)
    entites_trouvees = []
    
    # 1. Extraction via Spacy (Personnes et Lieux)
    # On se limite à PER (Personne) et LOC (Lieu) pour être plus strict et éviter
    # les faux positifs comme 'Playstation' (classé en MISC).
    for ent in doc.ents:
        # On vérifie si l'entité n'est pas un verbe, une erreur de classification courante
        # pour les mots en majuscule en début de phrase (ex: "Manger", "Jouer").
        if ent.root.pos_ == 'VERB':
            continue
        if ent.label_ in ["PER", "LOC"]: 
            entites_trouvees.append(ent.text)
            
    # 2. Ajout d'une vérification par mots-clés pour le domaine de Friends
    question_lower = question.lower()
    for entite in ACTEURS_FRIENDS:
        # On cherche le mot entier (avec \b) pour éviter les faux positifs (ex: "Ross" dans "Cross")
        if re.search(r'\b' + re.escape(entite) + r'\b', question_lower):
            # On vérifie qu'une entité plus complète n'existe pas déjà (ex: ne pas ajouter "Ross" si "Ross Geller" est déjà là)
            if not any(entite in e.lower() for e in entites_trouvees):
                 entites_trouvees.append(entite.capitalize())

    # Filtrer les doublons
    entites_trouvees = list(set(entites_trouvees))
            
    # Si on a trouvé au moins une entité, c'est Q1. Sinon Q2.
    if len(entites_trouvees) > 0:
        vectorizer, matrice_tfidf, df_docs, mots_vides = construireIndex(df, par_scene=False)
        q1.rechercherQ1(question, df, vectorizer, matrice_tfidf, df_docs, motsVidesRecherche=mots_vides)
    else:
        q2.search_Q2(question,vectorizer,matrice_tfidf,df_docs)