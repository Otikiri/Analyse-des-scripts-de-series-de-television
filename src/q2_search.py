from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def prepare_question(question, vectorizer):
    """Préparation de la question pour la recherche Q2.
    
    La question est transformée en minuscule, nettoyée, puis transformée 
    en vecteur TF-IDF avec le même vectorizer que celui utilisé pour les scripts.

    Args:
        question (str): La question de l'utilisateur.
        vectorizer (TfidfVectorizer): Le modèle TF-IDF entraîné sur le corpus.

    Returns:
        scipy.sparse.csr_matrix: Le vecteur TF-IDF représentant la question.
    """

    question = question.strip().lower()

    question_vector = vectorizer.transform([question])

    return question_vector

def vectorize_Q2(df, groupby_cols=None, max_df=0.95, ngram_range=(1, 2)):
    """Vectorisation TF-IDF pour le moteur de recherche Q2.
    
    On regroupe les lignes selon les colonnes spécifiées (ex: par saison + épisode) 
    afin d'obtenir un document par unité de recherche.

    Args:
        df (pd.DataFrame): Le DataFrame source contenant les données textuelles.
        groupby_cols (list of str, optional): Colonnes utilisées pour grouper les documents. Defaults to None.
        max_df (float, optional): Seuil de fréquence documentaire maximale. Defaults to 0.95.
        ngram_range (tuple, optional): Plage de n-grammes à utiliser. Defaults to (1, 2).

    Returns:
        tuple: (tfidf_matrix, vectorizer, docs) où docs est le DataFrame regroupé.
    """

    docs = df.copy()

    if groupby_cols:
        docs = (
            docs.groupby(groupby_cols, as_index=False)
            .agg({
                "texte nettoyer": lambda textes: " ".join(textes.astype(str)),
                "ligne": lambda lignes: " ".join(lignes.astype(str)),
                "nom fichier": "first",
                "nombres de mots": "sum"
            })
        )
    else:
        docs = docs.reset_index(drop=True)

    docs["doc"] = docs["texte nettoyer"].fillna("").astype(str)

    docs = docs[docs["doc"].str.strip() != ""].copy()
    docs = docs.reset_index(drop=True)

    corpus = docs["doc"].tolist()

    #print(f"Nombre de documents à analyser : {len(corpus)}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
        max_df=max_df,
        sublinear_tf=True,
        ngram_range=ngram_range,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)

    #print("Matrice TF-IDF :", tfidf_matrix.shape)

    return tfidf_matrix, vectorizer, docs

def cosine_similarity_Q2(question_vector, tfidf_matrix):
    """Calcul du score de similarité cosinus entre la question et les documents.

    Args:
        question_vector (scipy.sparse.csr_matrix): Le vecteur de la question.
        tfidf_matrix (scipy.sparse.csr_matrix): La matrice TF-IDF des documents.

    Returns:
        np.ndarray: Un tableau 1D contenant les scores de similarité.
    """

    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()

    return similarities

def euclidean_distance_Q2(question_vector, tfidf_matrix):
    """Calcul du score de similarité basé sur la distance euclidienne.

    Convertit la distance en score de similarité en utilisant la formule : 1 / (1 + distance).

    Args:
        question_vector (scipy.sparse.csr_matrix): Le vecteur de la question.
        tfidf_matrix (scipy.sparse.csr_matrix): La matrice TF-IDF des documents.

    Returns:
        np.ndarray: Un tableau 1D contenant les scores de similarité.
    """

    distances = euclidean_distances(question_vector, tfidf_matrix).flatten()

    scores = 1 / (1 + distances)

    return scores

def manhattan_distance_Q2(question_vector, tfidf_matrix):
    """Calcul du score de similarité basé sur la distance de Manhattan.

    Convertit la distance en score de similarité en utilisant la formule : 1 / (1 + distance).

    Args:
        question_vector (scipy.sparse.csr_matrix): Le vecteur de la question.
        tfidf_matrix (scipy.sparse.csr_matrix): La matrice TF-IDF des documents.

    Returns:
        np.ndarray: Un tableau 1D contenant les scores de similarité.
    """

    distances = manhattan_distances(question_vector, tfidf_matrix).flatten()

    scores = 1 / (1 + distances)

    return scores

def dot_product_Q2(question_vector, tfidf_matrix):
    """Calcul du score de similarité basé sur le produit scalaire.

    Args:
        question_vector (scipy.sparse.csr_matrix): Le vecteur de la question.
        tfidf_matrix (scipy.sparse.csr_matrix): La matrice TF-IDF des documents.

    Returns:
        np.ndarray: Un tableau 1D contenant les scores (produits scalaires).
    """

    scores = question_vector.dot(tfidf_matrix.T).toarray().flatten()

    return scores

def search_Q2_lines(question, vectorizer, tfidf_matrix, docs, tfidf_matrix_lines, vectorizer_lines, docs_lines, score_calculation, top_k=5):
    """Recherche de type Q2 : recherche par contenu au niveau des lignes.

    Calcule d'abord la similarité au niveau des épisodes complets, puis trouve 
    la ligne spécifique la plus pertinente au sein des épisodes retenus.

    Args:
        question (str): La question posée.
        vectorizer (TfidfVectorizer): Le vectoriseur au niveau des épisodes.
        tfidf_matrix (scipy.sparse.csr_matrix): La matrice TF-IDF au niveau des épisodes.
        docs (pd.DataFrame): Les documents groupés par épisode.
        tfidf_matrix_lines (scipy.sparse.csr_matrix): La matrice TF-IDF au niveau des lignes.
        vectorizer_lines (TfidfVectorizer): Le vectoriseur au niveau des lignes.
        docs_lines (pd.DataFrame): Les documents non groupés (une ligne = une réplique).
        score_calculation (callable): Fonction de calcul de similarité (ex: cosine_similarity_Q2).
        top_k (int, optional): Nombre d'épisodes à retourner. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame contenant les résultats formatés avec l'extrait exact.
    """

    docs_lines = docs_lines.reset_index(drop=True)  
    
    docs = docs.reset_index(drop=True)        

    question_vector = prepare_question(question, vectorizer)

    question_vector_lines = prepare_question(question, vectorizer_lines)

    similarities = score_calculation(question_vector, tfidf_matrix)

    similarities_lines = score_calculation(question_vector_lines, tfidf_matrix_lines)

    ranked_indices_lines = similarities_lines.argsort()[::-1]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    #print("Top indices lignes :", ranked_indices_lines[:top_k])
    #rint("Top ranked line extract : ", docs_lines.iloc[ranked_indices_lines[:top_k]][["saison", "episode", "ligne"]])
    for idx in ranked_indices[:top_k]:
        row = docs.iloc[idx]

        saison = row.get("saison")
        episode = row.get("episode")

        matching_lines = docs_lines[
            (docs_lines["saison"] == saison) &
            (docs_lines["episode"] == episode)
        ].copy()

        ranked_indices_matching_lines = similarities_lines[matching_lines.index].argsort()[::-1]

        result = {
            "saison": row.get("saison"),
            "episode": row.get("episode"),
            "titre": row.get("nom fichier"),
            "score": round(float(similarities[idx]), 4),
            "question" : question,
            "line": matching_lines.iloc[ranked_indices_matching_lines[0]].get("ligne") if not matching_lines.empty and "ligne" in docs_lines.columns else None,
            "nombre_mots": row.get("nombres de mots") if "nombres de mots" in docs.columns else None
        }

        results.append(result)
        df_res = pd.DataFrame(results)
        # df_res['rank'] = range(1,top_k+1)
    return df_res

def utiliser_moteur_Q2(question, vectorizer, tfidf_matrix, docs, tfidf_matrix_lines, vectorizer_lines, docs_lines, score_calculation=cosine_similarity_Q2, top_k=5):
    """Fonction principale du moteur Q2 utilisant des matrices pré-calculées.

    Les matrices étant déjà construites (via search_engine.py), cette fonction 
    délègue directement la recherche à `search_Q2_lines`.

    Args:
        question (str): La question posée par l'utilisateur.
        vectorizer (TfidfVectorizer): Vectoriseur pré-calculé (épisodes).
        tfidf_matrix (scipy.sparse.csr_matrix): Matrice pré-calculée (épisodes).
        docs (pd.DataFrame): DataFrame des épisodes.
        tfidf_matrix_lines (scipy.sparse.csr_matrix): Matrice pré-calculée (lignes).
        vectorizer_lines (TfidfVectorizer): Vectoriseur pré-calculé (lignes).
        docs_lines (pd.DataFrame): DataFrame des lignes.
        score_calculation (callable, optional): Méthode de calcul de distance. Defaults to cosine_similarity_Q2.
        top_k (int, optional): Nombre de résultats. Defaults to 5.

    Returns:
        pd.DataFrame: Les résultats de la recherche.
    """
    # Les matrices sont déjà construites (par search_engine.py), on passe directement à la recherche
    # print("q2_docs shape during search:", docs.shape)
    # print("q2_matrix shape during search:", tfidf_matrix.shape)
    results = search_Q2_lines(
        question,
        vectorizer,
        tfidf_matrix,
        docs,
        tfidf_matrix_lines,
        vectorizer_lines,
        docs_lines,
        score_calculation=score_calculation,
        top_k=top_k
    )
    return results


