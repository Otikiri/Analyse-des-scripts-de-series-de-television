import re 

import nltk
nltk.download('stopwords',quiet=True)
nltk.download('punkt_tab',quiet=True)
nltk.download('wordnet',quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import numpy as np 

import os
from sklearn.decomposition import PCA, TruncatedSVD
from tqdm import tqdm
from gensim import corpora
from gensim import models
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#===================================================================
# MOT_VIDE : ELEMENTS A ENLEVER DES TOKENS
#===================================================================

# Les mots vides doivent etre enleves pour avoir une analyse correcte 
motVide = set(stopwords.words('english')) 

# Mots vides additionnels
motVide.update([
    'dont', 'doesnt', 'didnt', 'cant', 'wont', 'wouldnt', 'couldnt', 'shouldnt',
    'yeah', 'okay', 'well', 'right', 'hey', 'know', 'yknow', 'youre', 'going',
    'really', 'like', 'get', 'gon', 'got', 'oh', 'uh', 'um', 'yes', 'no',
    'just', 'actually', 'wait', 'look', 'come', 'think', 'said', 'say', 'tell',
    'let', 'want', 'way', 'good', 'mean', 'little', 'thing', 'something','theres','thats','could'
    ,'would', 'back', 'time', 'great', 'guys', 'make','play','dude','theyre','shes','hes','whats'
])
motVide.update([
    'anyway', 'cause', 'ever', 'funny', 'getting', 'guess', 
    'hard', 'hell', 'hello', 'honey', 'isnt', 'kinda', 'leave',
    'might', 'morning', 'next', 'nice', 'night', 'nothing', 
    'pretty', 'ready', 'since', 'still', 'stop', 'stuff', 'sure',
    'together', 'totally', 'understand', 'whole', 'wrong', 'used',
    'wanted', 'went', 'tonight', 'thinking', 'talking',
    'umm', 'god', 'see', 'one', 'ive', 'guy', 'ill', 'got', 'know', 'like','wan','man',
    'na', 'nt', 'im', 'ur','man', 'wow', 'much', 'give',
    'always', 'another', 'anything', 'around', 'away',         
    'better', 'cause', 'even', 'every', 'fine', 'first',
    'sorry', 'thanks', 'thank', 'never', 'take', 'need',
    'listen', 'thought', 'best', 'place',"yeah","okay","ok","hey","hi","oh",
    "uh","um","well","thing","something","going","gonna","want","look","looking",
    "find","trying","start","keep",'huh'])
motVide.update([
    'people', 'talk', 'maybe', 'call', 'big', 'two', 
    'please', 'name', 'happened',
    'believe', 'day', 'girl', 'bad', 'whoa','alright','uhm'
])
motVide.update([
    'ross', 'rachel', 'monica', 'chandler', 'joey', 'phoebe',
])

FRIENDS_FILLER = {
    # Exclamations, laughs & reactions
    "yeah", "oh", "okay", "ok", "hey", "well", "ooh", "wow", "aww",
    "uh", "uhh", "hmm", "um", "uhm", "umm", "ohh", "ahh", "oooh", "ah", "huh", "god", "yes", "sure", "fine",
    "great", "sorry", "please", "thank", "thanks", "dude", "alright", "ha", "haha",
    "whoa", "oops", "shh", "totally", "kinda", "anyway", "hell", "yknow",

    # Contractions without apostrophes (non-standard transcription formats)
    "'s", "'t", "'re", "'m", "'d", "'ve", "'ll", "n't", "ca", "wo", "kay", "lookin", "cha",
    "gon", "na", "wan", "ta", "gonna", "wanna", "gotta",
    "dont", "doesnt", "didnt", "cant", "wont", "wouldnt", "couldnt", "shouldnt", "isnt",
    "youre", "theres", "thats", "theyre", "shes", "hes", "whats", "ive", "ill",

    # Generic verbs
    "go", "get", "last","year", "got", "cut", "come", "let", "see", "say", "said",
    "think", "know", "look", "want", "tell", "make", "take",
    "give", "try", "put", "call", "stop", "feel", "wait", "would", "could", "do",
    "need", "talk", "hear", "listen", "happen", "leave", "keep", "show", "find",
    "believe", "care", "ask", "bring", "start", "sound", "turn", "hold", "use",
    "understand", "thought", "used", "went","told","things","made"

    # Generic adverbs / modifiers / time
    "really", "right", "like", "just", "back", "still", "never", "first",
    "little", "way", "maybe", "guess", "mean", "good", "thing", "time",
    "one", "stuff", "even", "much", "actually", "already", "else", "another",
    "also", "always", "around", "away", "two", "three", "us", "bad", "big",
    "old", "new", "long", "hi", "hello", "bye", "goodbye", "morning", "night",
    "tonight", "today", "tomorrow", "yesterday",
    "ever", "next", "yet", "lot", "pretty", "nice", "cool", "minute", "cause", "bet",
    "better", "every", "whole", "wrong", "together", "might", "since", "funny",
    "best", "place", "people", "day", "name", "happened",

    # Vague pronouns & Endearments
    "something", "anything", "nothing", "everything", "someone",
    "guy", "guys", "honey", "sweetie", "babe", "man", "boy", "girl", "woman", "lady",

    "rach", "mon", "joe", "pheebs", "bing", "geller", "green", 
    "buffay", "tribbiani", "hannigan", "zelner", "mr", "mrs", "dr", "sir", "miss"
}

motVide.update(FRIENDS_FILLER)

# lemmatizer : permet de reduire des mots a leur racine
lemmatizer = WordNetLemmatizer()

# ==============================================================
# FONCTIONS POUR DATA LOADER 
# ==============================================================

def nettoyerTexte(texte): 
    """Nettoie une réplique brute en retirant la mise en scène, le nom de l'acteur, la ponctuation, et en convertissant le texte en minuscules.

    Args:
        texte (str): La chaîne de caractères brute issue du script.

    Returns:
        str: Le texte nettoyé et normalisé.
    """
    texte = re.sub(r'\(.*?\)','',texte) # on envele la mise en scene
    texte = re.sub(r'^[A-Z\s]+:\s', '', texte,flags=re.IGNORECASE) # on enleve le nom de la personne qui parle
    texte = texte.lower() # on mets le texte en miniscule
    texte = re.sub(r'[^a-z\s]',' ', texte) #on remplace la ponctuation par des espaces
    texte = re.sub(r'\s+', ' ', texte).strip() # on enleve les espaces pas necesaires 
    texte = re.sub(r'\b(a+h+|o+h+|u+h+|m+h+)\b', '', texte)
    return texte 


def tokeniserTexte(texte): 
    """Transforme une chaîne de texte nettoyée en une liste de tokens (mots) filtrés par longueur et débarrassés des mots vides.

    Args:
        texte (str): Le texte d'entrée préalablement nettoyé.

    Returns:
        list: Une liste de chaînes de caractères (tokens).
    """
    tokens = word_tokenize(texte.lower())
    
    return [w for w in tokens if w.isalnum() and w not in motVide and len(w)>2]

# ==============================================================
# FONCTIONS DIVERS
# ==============================================================


def retournerTtMots(df):
    """Parcourt l'ensemble des textes nettoyés d'un DataFrame pour en extraire et lister tous les mots individuels.

    Args:
        df (pd.DataFrame): Dataframe d'entrée possédant la colonne 'texte nettoyer'.

    Returns:
        list: Une liste plate contenant l'ensemble des mots extraits.
    """
    ttMots = []
    for texte in df['texte nettoyer']: 
        for word in texte.split(): 
            ttMots.append(word)
    return ttMots

def retournerTokens(df):
    """Parcourt et aplatit les listes de tokens de chaque ligne d'un DataFrame en un unique tableau de mots.

    Args:
        df (pd.DataFrame): Dataframe d'entrée possédant la colonne 'token'.

    Returns:
        list: Une liste globale de tous les tokens accumulés.
    """
    ttMots = []
    for texte in df['token']: 
        for word in texte: 
            ttMots.append(word)
    return ttMots

def tokeniserTexteRecherche(texte, df=None, motsVidesRecherche=None):
    """Tokenise et applique une lemmatisation (racine morphologique) sur un texte destiné aux requêtes de recherche.

    Args:
        texte (str): La phrase de recherche d'origine.
        df (pd.DataFrame, optional): Dataframe de référence pour l'extraction dynamique des noms des personnages. Valeur par défaut : None.
        motsVidesRecherche (set, optional): Set de stop-words à exclure de la recherche. Valeur par défaut : None.

    Returns:
        list: Liste de tokens normalisés et lemmatisés pour le moteur de recherche.
    """
    if motsVidesRecherche is None:
        nomsPersonnages = set(df['acteur'].str.lower().unique()) if df is not None else set()
        motsVidesRecherche = motVide - nomsPersonnages
    tokens= word_tokenize(texte)
    tokenNettoye = []
    for t in tokens:
        if t not in motsVidesRecherche and len(t)>2 :
            tokenNettoye.append(lemmatizer.lemmatize(t))
    return [t for t in tokenNettoye if t not in motsVidesRecherche]

#==============================================================
# FONCTIONS DE PARAMETRAGE
#==============================================================

def trouverNbOptimalTopics(df, min_topics=5, max_topics=15, min_tokens_par_scene=0,
    alpha_values=[0.3, 0.4, 0.5, 0.6, 0.7, 'auto'],id=0):
    """Évalue itérativement différentes configurations de modèles LDA (nombre de thèmes et hyperparamètre alpha) en calculant leur score de cohérence intrinsèque.

    Args:
        df (pd.DataFrame): Dataframe source contenant les données de scripts.
        min_topics (int, optional): Nombre minimal de topics à explorer. Valeur par défaut : 5.
        max_topics (int, optional): Nombre maximal de topics à explorer. Valeur par défaut : 15.
        min_tokens_par_scene (int, optional): Taille limite de tokens par scène filtrée. Valeur par défaut : 0.
        alpha_values (list, optional): Liste des valeurs d'hyperparamètre alpha à tester. Valeur par défaut : [0.3, 0.4, 0.5, 0.6, 0.7, 'auto'].
        id (str or int, optional): Identifiant unique (ex: numéro de saison) servant à nommer les fichiers d'export CSV. Valeur par défaut : 0.

    Returns:
        tuple: Un tuple de 2 éléments contenant :
            - meilleur (dict) : Les hyperparamètres optimaux retenus (cohérence maximale).
            - scores (list) : L'historique complet des performances calculées pour chaque combinaison.
    """

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
    dictionnaire.filter_extremes(no_below=5, no_above=0.85)
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
                    iterations=400
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
                    'perplexite': perplexite,
                    'saison' : id,
                })
                if coherence > meilleur['coherence']:
                    meilleur = {
                        'n_topics': n, 'alpha': alpha,
                        'coherence': round(coherence, 4),
                        'perplexite': perplexite,
                        'saison' : id,
                    }
                pbar.update(1)

    print(f"\nMeilleure combinaison : n_topics={meilleur['n_topics']}, alpha={meilleur['alpha']} "
          f"(coherence={meilleur['coherence']}, perplexite={meilleur['perplexite']})")
    
    m_df = pd.DataFrame([meilleur])
    s_df = pd.DataFrame(scores)

    m_df.to_csv("m_csv"+id+".csv",index=False)
    s_df.to_csv("s_csv"+id+".csv",index=False)

    return meilleur, scores

def coudeTFIDF(ks,inertias,silhouettes,results_dir): 
    """Génère un graphique d'optimisation double-axe combinant la méthode du coude (Inertie) et l'indice Silhouette, puis l'enregistre au format PNG.

    Args:
        ks (list): Liste des nombres de clusters (k) évalués.
        inertias (list): Liste des inerties intra-classes correspondantes.
        silhouettes (list): Liste des coefficients de Silhouette calculés.
        results_dir (str): Dossier de destination de l'image de synthèse générée.
    """

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

def topMotsParClusters(km,best_k,top_n_words,terms,results_dir,docs):

    """Extrait et trace sous forme d'histogrammes horizontaux les mots prépondérants pour chaque cluster du modèle KMeans ajusté.

    Args:
        km (KMeans): Instance ajustée du partitionnement KMeans.
        best_k (int): Le nombre optimal de clusters sélectionné.
        top_n_words (int): Nombre de termes les plus significatifs à exporter par figure.
        terms (np.ndarray): Liste ordonnée de l'ensemble des termes de l'index TF-IDF.
        results_dir (str): Dossier d'export des figures PNG et synthèses au format CSV.
        docs (pd.DataFrame): Le DataFrame de documents mis à jour avec les affectations de groupes.
    """
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

def rechercheBestK(k_min,k_max,corpus,results_dir,docs,X): 
    """Parcourt une plage définie de valeurs K, instancie et ajuste les modèles KMeans associés et détermine automatiquement la structure optimale via le coefficient Silhouette.

    Args:
        k_min (int): Borne de départ du nombre de clusters.
        k_max (int): Borne de fin maximale de clusters admissibles.
        corpus (list): Liste brute de documents textuels traités.
        results_dir (str): Répertoire d'archivage des scores d'optimisation globaux.
        docs (pd.DataFrame): Dataframe cible recevant l'indexation de groupe.
        X (scipy.sparse.csr_matrix): Matrice de caractéristiques d'apprentissage TF-IDF.

    Returns:
        tuple: Un ensemble d'éléments d'optimisation contenant :
            - ks (list) : La plage ordonnée de k testés.
            - inertias (list) : Les inerties calculées.
            - silhouettes (list) : L'historique des indices silhouette calculés.
            - best_k (int) : La valeur k sélectionnée.
            - km (KMeans) : Le meilleur estimateur ajusté associé.
            - labels (np.ndarray) : Les étiquettes de partitionnement calculées.
    """
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
    return ks,inertias,silhouettes,best_k,km,labels
# ==============================================================
# FONCTIONS D'AFFICHAGE
# ==============================================================

def plotEvolutionMots(df):
    """Génère et sauvegarde une courbe d'évolution temporelle montrant la fluctuation de fréquence relative de mots-clés spécifiques au fil des saisons.
    Args:
        df (pd.DataFrame): Dataframe structuré issu de l'analyse chronologique.
    """
    plt.figure(figsize=(10, 6))
    for mot in df['mot'].unique():
        data = df[df['mot']==mot]
        plt.plot(data['saison'],data['frequence_%'],label=mot,marker='o')
    plt.xlabel('Saison')
    plt.ylabel('Frequence %')
    plt.title('Evolution des frequences des 10 mots les plus frequent au cours de la serie')
    plt.legend()
    plt.tight_layout()
    plt.savefig('freqMotsCoursTemps.png')
    plt.close()


def plotFreqMotsParActeur(fma, top_n_acteurs=5):
    """Crée une figure multi-panneaux présentant les diagrammes à barres des distributions lexicales pour les principaux locuteurs de la série.

    Args:
        fma (pd.DataFrame): Dataframe contenant l'évaluation de fréquences de mots par acteur.
        top_n_acteurs (int, optional): Nombre d'intervenants les plus bavards à retenir dans la visualisation. Valeur par défaut : 5.
    """
    totalActeurs = (fma.groupby('acteur')['nb_fois'].sum().sort_values(ascending = False))

    topActeurs = totalActeurs.head(top_n_acteurs).index

    fig,axes = plt.subplots(top_n_acteurs,1,figsize=(12,top_n_acteurs*4))

    for ax,acteur in zip(axes,topActeurs): 
        data = fma[fma["acteur"] == acteur].sort_values('nb_fois',ascending=False)
        ax.bar(data['mot'],data['frequence_%'])
        ax.set_title(f"{acteur}")
        ax.set_xlabel("Mots")
        ax.set_ylabel("Frequence %")
        ax.tick_params(axis='x',rotation=45)
    
    fig.suptitle("Top 20 mots par acteur qui parle le plus", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("freqMotsParActeur.png", bbox_inches='tight')
    plt.close()

def affichageW2v(dfClusters, vecteurs, outputPath='results/w2v_pca_clusters.png'):
    """Imprime les termes centraux caractéristiques de chaque cluster Word2Vec et génère une projection graphique plane bidimensionnelle via l'algorithme PCA.

    Args:
        dfClusters (pd.DataFrame): Dataframe cartographiant les associations mots-clusters.
        vecteurs (np.ndarray): Matrice géométrique des plongements de mots normalisés.
        outputPath (str, optional): Fichier image de destination cible. Valeur par défaut : 'results/w2v_pca_clusters.png'.
    """
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

def visualizationTFIDF(use_svd,best_k,labels,docs,groupby_cols,results_dir,X): 
    """Génère et enregistre une image de projection plane en 2D (SVD ou PCA) des clusters TF-IDF de documents avec des annotations d'étiquettes chronologiques d'épisodes.

    Args:
        use_svd (bool): Utilise la décomposition en valeurs singulières (TruncatedSVD) si True, sinon utilise la PCA standard.
        best_k (int): Nombre total de partitions sélectionnées.
        labels (np.ndarray): Tableau d'affectation des clusters par document.
        docs (pd.DataFrame): Dataframe contenant les métadonnées de structure (saison, épisode).
        groupby_cols (list): Liste des colonnes d'agrégation d'origine.
        results_dir (str): Dossier d'export de l'image résultante.
        X (scipy.sparse.csr_matrix or np.ndarray): Matrice de caractéristiques textuelles d'entrée.

    Returns:
        np.ndarray: La matrice réduite des coordonnées en deux dimensions (n_chantillons, 2).
    """
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
    return X_2d
