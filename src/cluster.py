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


def clusteringW2v(df, numClusters=5, vectorSize=50, window=5, minCount=10, methodeNommage="centroide"):

    """Entraîne un modèle Word2Vec, applique un partitionnement KMeans sur les vecteurs de mots et évalue la cohérence.

    Args:
        df (pd.DataFrame): Dataframe d'entrée devant posséder une colonne 'token' contenant des listes de chaînes.
        numClusters (int, optional): Nombre de clusters cibles pour KMeans (K). Valeur par défaut : 5.
        vectorSize (int, optional): Dimension de l'espace de plongement vectoriel. Valeur par défaut : 50.
        window (int, optional): Fenêtre contextuelle maximale entre le mot cible et ses voisins. Valeur par défaut : 5.
        minCount (int, optional): Fréquence minimale d'apparition d'un mot pour être retenu. Valeur par défaut : 10.
        methodeNommage (str, optional): Approche de nommage des clusters ('centroide' ou 'tfidf'). Valeur par défaut : "centroide".

    Raises:
        ValueError: Si la colonne 'token' est absente ou si aucun mot ne respecte le filtre `minCount`.

    Returns:
        tuple: Un tuple de 3 éléments contenant :
            - model (Word2Vec) : Le modèle de plongement de mots entraîné.
            - dfClusters (pd.DataFrame) : Un DataFrame associant chaque mot à son cluster et au nom de ce cluster.
            - vecteurs (np.ndarray) : La matrice normalisée des vecteurs de mots calculés.
    """

    if 'token' not in df.columns:
        raise ValueError("Le dataframe doit contenir une colonne 'token'")
    phrases = df['token'].tolist()
    print("Entrainement de Word2Vec en cours...")
    model = Word2Vec(sentences=phrases, vector_size=vectorSize, window=window, min_count=minCount, epochs=50, workers=4)
    mots = list(model.wv.index_to_key)
    if not mots:
        raise ValueError("Aucun mot n'a ete trouve par Word2Vec avec ce min_count")
    vecteursBruts = [model.wv[mot] for mot in mots]
    vecteurs = normalize(vecteursBruts, norm='l2')

    # --- KMeans clustering ---
    print(f"Clustering KMeans en cours (K={numClusters})...")
    kmeans = KMeans(n_clusters=numClusters, random_state=42, n_init='auto')
    labelsFinaux = kmeans.fit_predict(vecteurs)

    # --- Calcul des Metriques ---
    inertie = kmeans.inertia_
    scoreSilhouette = silhouette_score(vecteurs, labelsFinaux)

    topics = []
    for c in range(numClusters):
        indices = np.where(labelsFinaux == c)[0]
        if len(indices) == 0: continue
        vecteursCluster = vecteurs[indices]
        centroide = np.mean(vecteursCluster, axis=0)
        distances = np.linalg.norm(vecteursCluster - centroide, axis=1)
        nMots = min(10, len(distances))
        indicesTop = np.argsort(distances)[:nMots]
        motsTopic = [mots[indices[i]] for i in indicesTop]
        topics.append(motsTopic)

    dictionary = corpora.Dictionary(phrases)
    cm = CoherenceModel(topics=topics, texts=phrases, dictionary=dictionary, coherence='c_v')
    scoreCoherence = cm.get_coherence()

    print(f"\n--- SCORES POUR K={numClusters} ---")
    print(f"Inertie    : {inertie:.1f}")
    print(f"Silhouette : {scoreSilhouette:.4f}")
    print(f"Coherence  : {scoreCoherence:.4f}\n")

    nomsBruts = nommerClusters(labelsFinaux, vecteurs, mots, nMotsNom=3, methode=methodeNommage)
    noms = {k: v.replace(' / ', ', ') for k, v in nomsBruts.items()}
    
    dfClusters = pd.DataFrame({
        'Mot': mots,
        'Cluster': labelsFinaux
    })
    dfClusters['Nom_Cluster'] = dfClusters['Cluster'].map(noms)
    return model, dfClusters, vecteurs

#======================================================================
# LDA : LATENT DIRICHLET ALLOCATION
#======================================================================




def clusteringLDA(df, meilleursParSaison, minTokensParScene=20, methodeNommage="tfidf"):
    """Exécute la modélisation de sujets (LDA) par saison, nomme les thématiques et exporte les visualisations interactives HTML.

    Args:
        df (pd.DataFrame): Dataframe contenant les lignes de dialogue nettoyées et tokenisées, indexées par saison, épisode et scène.
        meilleursParSaison (dict): Configuration optimisée par saison contenant le nombre de topics et l'hyperparamètre alpha.
        minTokensParScene (int, optional): Seuil de tokens minimal requis sous lequel une scène est écartée de l'entraînement. Valeur par défaut : 20.
        methodeNommage (str, optional): Algorithme utilisé pour labelliser les topics générés ('tfidf' ou 'centroide'). Valeur par défaut : "tfidf".

    Returns:
        dict: Un dictionnaire associant l'identifiant de la saison (seasonId) au modèle LdaModel correspondant.
    """


    resultats = {}

    for seasonId, dfSaison in df.groupby('saison'):
        print(f"\n--- Saison {seasonId} ---")

        if seasonId not in meilleursParSaison:
            print(f"  Pas de parametres optimaux, saison ignoree.")
            continue

        meilleur = meilleursParSaison[seasonId]

        # Grouper les tokens par scene
        textes = (
            dfSaison.groupby(['episode', 'scene_num'])['token']
            .apply(lambda rows: [t for tokens in rows for t in tokens])
            .tolist()
        )

        textes = [t for t in textes if len(t) >= minTokensParScene]

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
            resultats[seasonId] = modele

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
            print(f"  Erreur saison {seasonId}: {e}")
        
        vis = gensimvis.prepare(modele,corpus,dictionnaire)
        pyLDAvis.save_html(vis,'saison_'+seasonId+'_lda.html')
        print('saison_'+seasonId+'_lda.html : saved')


    print("\nEntrainement termine.")
    return resultats

# ==========================================================
# TF-IDF ET KMEANS
# ==========================================================

def clusteringTfidf(df, groupbyCols=None, resultsDir='results_tfidf', kMin=2, kMax=10, topNWords=10, maxDf=1.0, ngramRange=(1,1), useSvd=False, methodeNommage="tfidf"):

    """Applique une vectorisation TF-IDF suivie d'un clustering KMeans automatique basé sur l'optimisation du score Silhouette.

    Args:
        df (pd.DataFrame): Dataframe d'entrée contenant la colonne 'token'.
        groupbyCols (list, optional): Liste des colonnes de regroupement (ex: `['saison', 'episode']`) pour fusionner les tokens en un seul document. Valeur par défaut : None.
        resultsDir (str, optional): Répertoire de sauvegarde des scores d'optimisation et figures. Valeur par défaut : 'results_tfidf'.
        kMin (int, optional): Nombre minimal de clusters à tester. Valeur par défaut : 2.
        kMax (int, optional): Nombre maximal de clusters à tester. Valeur par défaut : 10.
        topNWords (int, optional): Nombre de mots principaux à tracer par graphique de cluster. Valeur par défaut : 10.
        maxDf (float, optional): Seuil de fréquence de document maximale appliqué au TfidfVectorizer. Valeur par défaut : 1.0.
        ngramRange (tuple, optional): Limites inférieures et supérieures de la taille des n-grammes à extraire. Valeur par défaut : (1,1).
        useSvd (bool, optional): Utilise l'algorithme SVD (LSA) plutôt que le PCA par défaut pour projeter les graphiques en 2D. Valeur par défaut : False.
        methodeNommage (str, optional): Stratégie de désignation linguistique des clusters. Valeur par défaut : "tfidf".

    Returns:
        tuple or None: Si les conditions de données sont réunies, retourne un tuple contenant :
            - km (KMeans) : L'instance d'ajustement du modèle KMeans retenu.
            - docs (pd.DataFrame) : Le jeu de données agrégé avec ses labels de cluster appliqués et traduits.
            - X2d (np.ndarray) : Coordonnées réduites en 2 dimensions pour l'affichage graphique.
            - vectorizer (TfidfVectorizer) : L'extracteur de caractéristiques TF-IDF ajusté au corpus.
            - X (scipy.sparse.csr_matrix) : La matrice creuse des fréquences de termes TF-IDF.
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

    ks,inertias,silhouettes,bestK,km,labels = ut.rechercheBestK(kMin,kMax,corpus=corpus,results_dir=resultsDir,docs=docs,X=X)

    ut.coudeTFIDF(ks,inertias,silhouettes,resultsDir)

    # Adaptation pour nommerClusters
    wordVectors = km.cluster_centers_.T # (num_words, num_clusters)
    wordLabels = np.argmax(wordVectors, axis=1)
    
    nomsBruts = nommerClusters(wordLabels, wordVectors, terms, nMotsNom=3, methode=methodeNommage)
    noms = {k: v.replace(' / ', ', ') for k, v in nomsBruts.items()}
    
    docs['nom_cluster'] = docs['cluster'].map(noms)

    ut.topMotsParClusters(km,bestK,topNWords,terms,resultsDir,docs)

    X2d = ut.visualizationTFIDF(useSvd,bestK,labels,docs,groupbyCols,resultsDir,X)

    print(f"Résultats enregistrés dans {resultsDir}/")
    return km, docs, X2d,vectorizer,X
