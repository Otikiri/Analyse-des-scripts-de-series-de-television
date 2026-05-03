# ==========================================================
# LIBRAIRIES ET MODULES
# ==========================================================
import glob
import os
import re # pour obtenir le texte brut
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from gensim import corpora
from gensim import models
from gensim.models import Word2Vec
from gensim.models.coherencemodel import CoherenceModel

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

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


# Prend en parametre le dataframe de clusters et les vecteurs
# Affiche un tableau d'echantillons et un nuage PCA/TSNE
def affichageW2v(dfClusters, vecteurs, outputPath='results/w2v_pca_clusters.png'):
    print("\n=== 5 Mots Pertinents par Cluster ===")
    top_mots_par_cluster = {}
    for c in sorted(dfClusters['Cluster'].unique()):
        # Obtenir les index des mots de ce cluster
        indices = dfClusters.index[dfClusters['Cluster'] == c].tolist()
        if len(indices) == 0: continue
        vecteurs_cluster = vecteurs[indices]
        # Calcul du centroide moyen
        centroide = np.mean(vecteurs_cluster, axis=0)
        # Distances euclidiennes au centroide
        distances = np.linalg.norm(vecteurs_cluster - centroide, axis=1)
        # Obtenir les indices des 5 mots les plus proches
        n_mots = min(5, len(distances))
        indices_locaux_top = np.argsort(distances)[:n_mots]
        indices_globaux_top = [indices[i] for i in indices_locaux_top]
        top_mots_par_cluster[c] = indices_globaux_top
        mots = dfClusters.loc[indices_globaux_top, 'Mot'].tolist()
        print(f"Cluster {c} : {', '.join(mots)}")
    print("==============================================\n")

    # Utilisation de PCA pour la visualisation
    pca = PCA(n_components=2, random_state=42)
    vecteurs2d = pca.fit_transform(vecteurs)
    dfClusters['Dim1'] = vecteurs2d[:, 0]
    dfClusters['Dim2'] = vecteurs2d[:, 1]
    plt.figure(figsize=(14, 10))
    clustersUniques = dfClusters['Cluster'].unique()

    for c in clustersUniques:
        subset = dfClusters[dfClusters['Cluster'] == c]
        plt.scatter(subset['Dim1'], subset['Dim2'], alpha=0.7, edgecolors='none', label=f"Cluster {c}", s=50)
    # Annoter uniquement les 5 mots les plus pertinents
    for c in clustersUniques:
        if c in top_mots_par_cluster:
            subset = dfClusters.loc[top_mots_par_cluster[c]]
            for _, row in subset.iterrows():
                plt.annotate(row['Mot'], (row['Dim1'], row['Dim2']), fontsize=11, alpha=0.9, weight='bold')

    plt.title('Clustering Word2Vec - Visualisation PCA')
    plt.xlabel('Dimension PCA 1')
    plt.ylabel('Dimension PCA 2')

    # Place la legende à l'exterieur
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)

    if not os.path.exists(os.path.dirname(outputPath)):
        try:
            os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        except:
            pass

    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graphique PCA sauvegarde dans {outputPath}")



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

# ==========================================================
# TF-IDF ET KMEANS
# ==========================================================

def clusteringTfidf(df, groupby_cols=None, results_dir='results_tfidf', k_min=2, k_max=10, top_n_words=10, max_df=1.0, ngram_range=(1,2), use_svd=False):
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
    
    # --- RECHERCHE DU MEILLEUR K ---
    inertias = []
    silhouettes = []
    ks = list(range(k_min, min(k_max, len(corpus) - 1) + 1))
    models_dict = {}
    
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        models_dict[k] = (km, labels)
        inertias.append(km.inertia_)
        
        if len(set(labels)) > 1:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(-1)
            
    scores = pd.DataFrame({'k': ks, 'inertia': inertias, 'silhouette': silhouettes})
    scores.to_csv(os.path.join(results_dir, 'k_optimization_scores.csv'), index=False)
    
    best_k = int(scores.sort_values(['silhouette', 'k'], ascending=[False, True]).iloc[0]['k'])
    km, labels = models_dict[best_k]
    docs['cluster'] = labels
    print("Meilleur k retenu =", best_k)
    
    # --- GRAPHE COUDE + SILHOUETTE ---
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(ks, inertias, marker='o', color='tab:blue', label='Méthode du coude')
    ax1.set_xlabel('Nombre de clusters k')
    ax1.set_ylabel('Inertie (coude)', color='tab:blue')
    ax1.set_title('Optimisation du nombre de clusters')

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, marker='s', color='tab:orange', label='Silhouette')
    ax2.set_ylabel('Score silhouette', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'elbow_silhouette.png'), dpi=200)
    plt.close()

    # --- TOP MOTS PAR CLUSTER ---
    centers = km.cluster_centers_
    rows = []
    for i in range(best_k):
        idx = centers[i].argsort()[::-1][:top_n_words]
        top_words = terms[idx]
        rows.append({'cluster': i, 'top_words': ', '.join(top_words)})

        plt.figure(figsize=(10,5))
        vals = centers[i][idx][::-1]
        words = top_words[::-1]
        plt.barh(words, vals)
        plt.title(f'Top mots cluster {i}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'cluster_{i}_top_words.png'), dpi=200)
        plt.close()

    pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'top_words_clusters.csv'), index=False)
    docs.to_csv(os.path.join(results_dir, 'clusters_assignes.csv'), index=False)

    # --- DIMENSION REDUCTION ET VISUALISATION ---
    if use_svd:
        reducer = TruncatedSVD(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X.toarray())
        
    var_exp = reducer.explained_variance_ratio_ * 100

    plt.figure(figsize=(12, 8))
    for i in range(best_k):
        mask = labels == i
        subset = docs.loc[mask]
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f'Cluster {i}', alpha=0.75, s=28)
        
        for j, (_, row) in enumerate(subset.iterrows()):
            label_txt = ""
            if groupby_cols == ['saison', 'episode']:
                try:
                    label_txt = f"S{int(row['saison']):02d}E{int(row['episode']):02d}"
                except:
                    pass
            elif groupby_cols == ['saison']:
                try:
                    label_txt = f"S{int(row['saison'])}"
                except:
                    pass
            else:
                try:
                    label_txt = f"S{int(row['saison']):02d}E{int(row['episode']):02d}"
                except:
                    pass
            
            if label_txt:
                plt.text(X_2d[mask, 0][j], X_2d[mask, 1][j], label_txt, fontsize=5, alpha=0.6)

    plt.xlabel(f'Dim 1 ({var_exp[0]:.1f}%)')
    plt.ylabel(f'Dim 2 ({var_exp[1]:.1f}%)')
    plt.title(f'Projection PCA/SVD des clusters (k={best_k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pca_clusters.png'), dpi=250)
    plt.close()

    print(f"Résultats enregistrés dans {results_dir}/")
    return km, docs, X_2d
