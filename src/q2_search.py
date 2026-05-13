from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from data_loader import chargerDonnees


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

    print(f"Nombre de documents à analyser : {len(corpus)}")

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

    print("Matrice TF-IDF :", tfidf_matrix.shape)

    return tfidf_matrix, vectorizer, docs


def search_Q2(question, vectorizer, tfidf_matrix, docs, top_k=5):
    """
    Recherche de type Q2 : recherche par contenu sans filtrage.
    """

    question_vector = prepare_question(question, vectorizer)

    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()

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

        if result["nombre_mots"] is not None:
            print(f"  Nombre de mots : {result['nombre_mots']}")

        if result["line"]:
            print(f"  Extrait du script : {result['line'][:300]}...")
