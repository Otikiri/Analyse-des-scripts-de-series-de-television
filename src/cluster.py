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

# ==========================================================
# CLUSTERING ET W2VEC
# ==========================================================

# Prend en parametre un dataframe avec des tokens
# Entraine un modele Word2Vec et cluster les mots
# renvoie le modele, un dataframe des clusters, les vecteurs et le score de coherence
def clusteringW2v(df, numClusters=5, vectorSize=50, window=5, minCount=10):
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
        mots_topic = [mots[indices[i]] for i in indices_top]
        topics.append(mots_topic)

    dictionary = corpora.Dictionary(phrases)
    cm = CoherenceModel(topics=topics, texts=phrases, dictionary=dictionary, coherence='c_v')
    score_coherence = cm.get_coherence()

    print(f"\n--- SCORES POUR K={numClusters} ---")
    print(f"Inertie    : {inertie:.1f}")
    print(f"Silhouette : {score_silhouette:.4f}")
    print(f"Coherence  : {score_coherence:.4f}\n")

    dfClusters = pd.DataFrame({
        'Mot': mots,
        'Cluster': labels_finaux
    })
    return model, dfClusters, vecteurs

#======================================================================
# LDA : LATENT DIRICHLET ALLOCATION
#======================================================================

def clusteringLDA(df, meilleurs_par_saison, min_tokens_par_scene=20):
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
        
        vis = gensimvis.prepare(modele,corpus,dictionnaire)
        pyLDAvis.save_html(vis,'saison_'+season_id+'_lda.html')
        print('saison_'+season_id+'_lda.html : saved')


    print("\nEntrainement termine.")
    return resultats

# ==========================================================
# TF-IDF ET KMEANS
# ==========================================================

def clusteringTfidf(df, groupby_cols=None, results_dir='results_tfidf', k_min=2, k_max=10, top_n_words=10, max_df=1.0, ngram_range=(1,1), use_svd=False):
    """
    Applique TF-IDF et KMeans sur les tokens.
    Si groupby_cols est specifie, agrege les tokens par ces colonnes (ex: ['saison', 'episode']).
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Copie pour ne pas modifier l'original
    docs = df.copy()
    
    if groupby_cols:
        docs = docs.groupby(groupby_cols)['token'].sum().reset_index()
    else:
        docs = docs.reset_index(drop=True)
        
    docs['doc'] = docs['token'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    docs = docs[docs['doc'].str.strip() != ''].copy()
    docs = docs.reset_index(drop=True)
    
    corpus = docs['doc'].tolist()
    print(f"Nombre de documents à analyser : {len(corpus)}")
    
    if len(corpus) < k_min:
        print("Pas assez de documents pour le clustering.")
        return None
        
    # --- TF-IDF ---
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'(?u)\b\w+\b',
        min_df=1,
        max_df=max_df,
        sublinear_tf=True,
        ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(corpus)
    terms = np.array(vectorizer.get_feature_names_out())
    
    print("Matrice TF-IDF :", X.shape)

    ks,inertias,silhouettes,best_k,km,labels = ut.rechercheBestK(k_min,k_max,corpus=corpus,results_dir=results_dir,docs=docs,X=X)

    ut.coudeTFIDF(ks,inertias,silhouettes,results_dir)

    ut.topMotsParClusters(km,best_k,top_n_words,terms,results_dir,docs)

    X_2d = ut.visualizationTFIDF(use_svd,best_k,labels,docs,groupby_cols,results_dir,X)

    print(f"Résultats enregistrés dans {results_dir}/")
    return km, docs, X_2d,vectorizer,X



