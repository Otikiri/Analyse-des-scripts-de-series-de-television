from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def prepare_question(question, vectorizer):
    """
    Préparation de la question pour la recherche Q2.
    La question est transformée en vecteur TF-IDF avec le même vectorizer
    que celui utilisé pour les scripts.
    """

    question = question.strip().lower()

    question_vector = vectorizer.transform([question])

    return question_vector

def vectorize_Q2(df, groupby_cols=None, max_df=0.95, ngram_range=(1, 2)):
    """
    Vectorisation TF-IDF pour le moteur de recherche Q2.
    On regroupe les lignes par saison + épisode afin d'obtenir un document par épisode.
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
    """
    Calcul du score de similarité cosinus entre la question et les documents.
    """

    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()

    return similarities

def euclidean_distance_Q2(question_vector, tfidf_matrix):
    """
    Calcul du score de similarité basé sur la distance euclidienne entre la question et les documents.
    """

    distances = euclidean_distances(question_vector, tfidf_matrix).flatten()

    scores = 1 / (1 + distances)

    return scores

def manhattan_distance_Q2(question_vector, tfidf_matrix):
    """
    Calcul du score de similarité basé sur la distance de Manhattan entre la question et les documents.
    """

    distances = manhattan_distances(question_vector, tfidf_matrix).flatten()

    scores = 1 / (1 + distances)

    return scores

def dot_product_Q2(question_vector, tfidf_matrix):
    """
    Calcul du score de similarité basé sur le produit scalaire entre la question et les documents.
    """

    scores = question_vector.dot(tfidf_matrix.T).toarray().flatten()

    return scores


#search_q2 obsolete ?? 
def search_Q2(question, vectorizer, tfidf_matrix, docs, score_calculation, top_k=5):
    """
    Recherche de type Q2 : recherche par contenu sans filtrage.
    """

    question_vector = prepare_question(question, vectorizer)

    similarities = score_calculation(question_vector, tfidf_matrix)

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for idx in ranked_indices[:top_k]:
        row = docs.iloc[idx]

        result = {
            "score": round(float(similarities[idx]), 4),
            "saison": row.get("saison"),
            "episode": row.get("episode"),
            "title": row.get("nom fichier"),
            "line": row.get("ligne") if "ligne" in docs.columns else None,
            "nombre_mots": row.get("nombres de mots") if "nombres de mots" in docs.columns else None
        }

        results.append(result)

    return results

def search_Q2_lines(question, vectorizer, tfidf_matrix, docs, tfidf_matrix_lines, vectorizer_lines, docs_lines, score_calculation, top_k=5):
    """
    Recherche de type Q2 : recherche par contenu au niveau des lignes.
    """

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
            "title": row.get("nom fichier"),
            "score": round(float(similarities[idx]), 4),
            "question" : question
            #"line": matching_lines.iloc[ranked_indices_matching_lines[0]].get("ligne") if not matching_lines.empty and "ligne" in docs_lines.columns else None,
            #"nombre_mots": row.get("nombres de mots") if "nombres de mots" in docs.columns else None
        }

        results.append(result)
        df_res = pd.DataFrame(results)
        # df_res['rank'] = range(1,top_k+1)
    return df_res

def utiliser_moteur_Q2(df, question, score_calculation = cosine_similarity_Q2, top_k=5):
    # print("Recherche de type Q2...")

    tfidf_matrix, vectorizer, docs = vectorize_Q2(
        df,
        groupby_cols=["saison", "episode"],
        max_df=0.95,
        ngram_range=(1, 2)
    )
    tfidf_matrix_lines, vectorizer_lines, docs_lines = vectorize_Q2(
        df,
        groupby_cols=["saison", "episode", "ligne"],
        max_df=0.95,
        ngram_range=(1, 2)
    )

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

def afficher_resultats_Q2(results):
    """
    Affichage propre des résultats Q2.
    """

    print("Résultats de la recherche :")

    for idx, result in enumerate(results):
        print(f"\nRésultat {idx + 1} :")
        print(f"  Score de similarité : {result['score']}")
        print(f"  Saison : {result['saison']}")
        print(f"  Episode : {result['episode']}")
        print(f"  Titre du script : {result['title']}")

        # if result["nombre_mots"] is not None:
        #     print(f"  Nombre de mots : {result['nombre_mots']}")

        # if result["line"]:
        #     print(f"  Extrait du script : {result['line'][:300]}...")

