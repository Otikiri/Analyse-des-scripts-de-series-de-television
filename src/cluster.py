# ==========================================================
# LIBRAIRIES ET MODULES
# ==========================================================
import os
import numpy as np
import pandas as pd

from gensim import corpora
from gensim import models
from gensim.models import Word2Vec
from gensim.models.coherencemodel import CoherenceModel

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Module dependencies
import utils as ut

def nommerClustersCentroide(labels, vecteurs, motsVocab, nMotsNom=3):
    """Approche par centroïde : trouve les mots les plus proches du centre géométrique.

    Args:
        labels (list or np.ndarray): Liste des labels de cluster associés à chaque mot.
        vecteurs (list or np.ndarray): Matrice des vecteurs de mots.
        motsVocab (list): Liste ordonnée des mots du vocabulaire correspondant aux vecteurs.
        nMotsNom (int, optional): Nombre de mots-clés à extraire pour composer le nom. Valeur par défaut : 3.

    Returns:
        dict: Un dictionnaire associant l'identifiant du cluster à son nom généré (str).
    """
    vecteurs = np.array(vecteurs)
    noms = {}
    for c in sorted(set(labels)):
        indices = np.where(np.array(labels) == c)[0]
        if len(indices) == 0:
            noms[c] = f"Cluster {c}"
            continue
        vecsC = vecteurs[indices]
        centroide = np.mean(vecsC, axis=0)
        distances = np.linalg.norm(vecsC - centroide, axis=1)
        n = min(nMotsNom, len(distances))
        indicesTop = np.argsort(distances)[:n]
        motsTop = [motsVocab[indices[i]] for i in indicesTop]
        noms[c] = " / ".join(motsTop)
    return noms

def nommerClustersTFIDF(labels, vecteurs, motsVocab, nMotsNom=3):
    """Approche par maximum de poids (TF-IDF / LDA) : trouve les mots ayant la plus forte probabilité/poids pour ce cluster.

    Args:
        labels (list or np.ndarray): Liste des labels de cluster associés à chaque mot.
        vecteurs (list or np.ndarray): Matrice des poids/probabilités (mots en lignes, dimensions/clusters en colonnes).
        motsVocab (list): Liste ordonnée des mots du vocabulaire.
        nMotsNom (int, optional): Nombre de mots-clés à extraire pour composer le nom. Valeur par défaut : 3.

    Returns:
        dict: Un dictionnaire associant l'identifiant du cluster à son nom généré (str).
    """
    vecteurs = np.array(vecteurs)
    noms = {}
    for c in sorted(set(labels)):
        indices = np.where(np.array(labels) == c)[0]
        if len(indices) == 0:
            noms[c] = f"Cluster {c}"
            continue
        # Poids des mots appartenant à ce cluster pour la dimension c
        poids = vecteurs[indices, c]
        n = min(nMotsNom, len(poids))
        # Trier par poids décroissant
        indicesTopLocaux = np.argsort(poids)[::-1][:n]
        motsTop = [motsVocab[indices[i]] for i in indicesTopLocaux]
        noms[c] = " / ".join(motsTop)
    return noms

def nommerClusters(labels, vecteurs, motsVocab, nMotsNom=3, methode='tfidf'):
    """Fonction chapeau qui redirige vers la bonne méthode de nommage de cluster (TF-IDF ou Centroïde).

    Args:
        labels (list or np.ndarray): Liste des labels de cluster associés à chaque mot.
        vecteurs (list or np.ndarray): Matrice de vecteurs ou de poids.
        motsVocab (list): Liste ordonnée des mots du vocabulaire.
        nMotsNom (int, optional): Nombre de mots-clés à extraire pour composer le nom. Valeur par défaut : 3.
        methode (str, optional): Algorithme choisi ('tfidf' ou autre pour 'centroide'). Valeur par défaut : 'tfidf'.

    Returns:
        dict: Un dictionnaire contenant les identifiants de cluster et leurs noms calculés.
    """
    if methode == "tfidf":
        return nommerClustersTFIDF(labels, vecteurs, motsVocab, nMotsNom)
    else:
        return nommerClustersCentroide(labels, vecteurs, motsVocab, nMotsNom)

# ==========================================================
# CLUSTERING ET W2VEC
# ==========================================================

# Prend en parametre un dataframe avec des tokens
# Entraine un modele Word2Vec et cluster les mots
# renvoie le modele, un dataframe des clusters, les vecteurs et le score de coherence
def clusteringW2v(df, numClusters=5, vectorSize=50, window=5, minCount=10, methodeNommage="centroide"):
    if 'token' not in df.columns:
        raise ValueError("Le dataframe doit contenir une colonne 'token'")
    phrases_train = df_train['token'].tolist()
    phrases_app = df_app['token'].tolist()

    print("Entrainement de Word2Vec en cours...")
    model = Word2Vec(sentences=phrases_train, vector_size=vectorSize, window=window, min_count=minCount, epochs=30, workers=4)

    mots_personnage_uniques = set([mot for phrase in phrases_app for mot in phrase])
    mots_finaux = [mot for mot in mots_personnage_uniques if mot in model.wv]

    if not mots_finaux:
        raise ValueError("Aucun mot n'a ete trouve par Word2Vec avec ce min_count")
    vecteursBruts = [model.wv[mot] for mot in mots_finaux]
    vecteurs = normalize(vecteursBruts, norm='l2')

    # --- KMeans clustering ---
    print(f"Clustering KMeans en cours (K={numClusters})...")
    kmeans = KMeans(n_clusters=numClusters, random_state=42, n_init='auto')
    labels_finaux = kmeans.fit_predict(vecteurs)

    # --- Calcul des Metriques ---
    inertie = kmeans.inertia_
    score_silhouette = silhouette_score(vecteurs, labels_finaux)

    topics = []
    for c in range(numClusters):
        indices = np.where(labels_finaux == c)[0]
        if len(indices) == 0: continue
        vecteurs_cluster = vecteurs[indices]
        centroide = np.mean(vecteurs_cluster, axis=0)
        distances = np.linalg.norm(vecteurs_cluster - centroide, axis=1)
        n_mots = min(10, len(distances))
        indices_top = np.argsort(distances)[:n_mots]
        mots_topic = [mots_finaux[indices[i]] for i in indices_top]
        topics.append(mots_topic)

    dictionary = corpora.Dictionary(phrases_app)
    cm = CoherenceModel(topics=topics, texts=phrases_app, dictionary=dictionary, coherence='c_v')
    score_coherence = cm.get_coherence()

    print(f"\n--- SCORES POUR K={numClusters} ---")
    print(f"Inertie    : {inertie:.1f}")
    print(f"Silhouette : {score_silhouette:.4f}")
    print(f"Coherence  : {score_coherence:.4f}\n")

    nomsBruts = nommerClusters(labels_finaux, vecteurs, mots_finaux, nMotsNom=3, methode=methodeNommage)
    noms = {k: v.replace(' / ', ', ') for k, v in nomsBruts.items()}
    
    dfClusters = pd.DataFrame({
        'Mot': mots_finaux,
        'Cluster': labels_finaux
    })
    dfClusters['Nom_Cluster'] = dfClusters['Cluster'].map(noms)
    return model, dfClusters, vecteurs

#======================================================================
# LDA : LATENT DIRICHLET ALLOCATION
#======================================================================




def clusteringLDA(df, meilleursParSaison, minTokensParScene=20, methodeNommage="tfidf"):
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

            # Adaptation pour nommerClusters
            topicWordMatrix = modele.get_topics() # (num_topics, num_words)
            wordVectors = topicWordMatrix.T # (num_words, num_topics)
            wordLabels = np.argmax(wordVectors, axis=1)
            motsVocab = [dictionnaire[i] for i in range(len(dictionnaire))]
            
            nomsBruts = nommerClusters(wordLabels, wordVectors, motsVocab, nMotsNom=3, methode=methodeNommage)
            noms = {k: f"Topic {k} - {v.replace(' / ', ', ')}" for k, v in nomsBruts.items()}
            
            print("\n  --- Noms des topics générés ---")
            for tId, nom in noms.items():
                print(f"  * {nom}")
            print()

            for idx in range(meilleur['n_topics']):
                nomTopic = noms[idx]
                motsDuNom = [m.strip() for m in nomTopic.split('-')[-1].split(',')]
                
                # get_topic_terms returns list of (word_id, probability)
                termsProbs = modele.get_topic_terms(idx, topn=20)
                motsRestants = []
                for wordId, prob in termsProbs:
                    mot = dictionnaire[wordId]
                    if mot not in motsDuNom:
                        motsRestants.append(f'{prob:.3f}*"{mot}"')
                    if len(motsRestants) == 10:
                        break
                print(f"  Détail Topic {idx} (contexte): {' + '.join(motsRestants)}")

        except Exception as e:
            print(f"  Erreur saison {season_id}: {e}")
        
        vis = gensimvis.prepare(modele,corpus,dictionnaire)
        pyLDAvis.save_html(vis,'saison_'+str(season_id)+'_lda.html')
        print('saison_'+str(season_id)+'_lda.html : saved')


    print("\nEntrainement termine.")
    return resultats

# ==========================================================
# TF-IDF ET KMEANS
# ==========================================================

def clusteringTfidf(df, groupbyCols=None, resultsDir='results_tfidf', kMin=2, kMax=10, topNWords=10, maxDf=1.0, ngramRange=(1,1), useSvd=False, methodeNommage="tfidf"):
    """
    Applique TF-IDF et KMeans sur les tokens.
    Si groupbyCols est specifie, agrege les tokens par ces colonnes (ex: ['saison', 'episode']).
    """
    os.makedirs(resultsDir, exist_ok=True)
    
    # Copie pour ne pas modifier l'original
    docs = df.copy()
    
    if groupbyCols:
        docs = docs.groupby(groupbyCols)['token'].sum().reset_index()
    else:
        docs = docs.reset_index(drop=True)
        
    docs['doc'] = docs['token'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    docs = docs[docs['doc'].str.strip() != ''].copy()
    docs = docs.reset_index(drop=True)
    
    corpus = docs['doc'].tolist()
    print(f"Nombre de documents à analyser : {len(corpus)}")
    
    if len(corpus) < kMin:
        print("Pas assez de documents pour le clustering.")
        return None
        
    # --- TF-IDF ---
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'(?u)\b\w+\b',
        min_df=1,
        max_df=maxDf,
        sublinear_tf=True,
        ngram_range=ngramRange
    )
    X = vectorizer.fit_transform(corpus)
    terms = np.array(vectorizer.get_feature_names_out())
    
    print("Matrice TF-IDF :", X.shape)

    ks,inertias,silhouettes,best_k,km,labels = ut.rechercheBestK(kMin,kMax,corpus=corpus,results_dir=resultsDir,docs=docs,X=X)

    ut.coudeTFIDF(ks,inertias,silhouettes,resultsDir)

    # Adaptation pour nommerClusters
    wordVectors = km.cluster_centers_.T # (num_words, num_clusters)
    wordLabels = np.argmax(wordVectors, axis=1)
    
    nomsBruts = nommerClusters(wordLabels, wordVectors, terms, nMotsNom=3, methode=methodeNommage)
    noms = {k: v.replace(' / ', ', ') for k, v in nomsBruts.items()}
    
    if 'cluster' in docs.columns:
        docs['nom_cluster'] = docs['cluster'].map(noms)

    ut.topMotsParClusters(km,best_k,topNWords,terms,resultsDir,docs)

    X_2d = ut.visualizationTFIDF(useSvd,best_k,labels,docs,groupbyCols,resultsDir,X)

    print(f"Résultats enregistrés dans {resultsDir}/")
    return km, docs, X_2d,vectorizer,X
