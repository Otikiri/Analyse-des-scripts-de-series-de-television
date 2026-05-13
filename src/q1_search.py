import spacy
import moteur as se

nlp = spacy.load("en_core_web_sm")


# Extrait les noms de personnages et lieux mentionnes dans la question
def extraireEntites(question, df=None):
    doc = nlp(question)
    entites = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "GPE", "LOC", "FAC", "ORG"}:
            entites.append(ent.text.strip())

    # si spaCy a rate des noms, on verifie chaque mot de la question dans le dataframe
    if df is not None:
        for mot in question.split():
            if mot not in entites and df['acteur'].str.lower().eq(mot.lower()).any():
                entites.append(mot)

    return entites


# Moteur de recherche pour les questions de type Q1
# Filtre d'abord les episodes où les entites mentionnees apparaissent,
# puis classe ces episodes par similarite avec la question
def rechercherQ1(question, df, vectorizer, matrice_tfidf, df_docs, top_k=5):
    entites = extraireEntites(question, df)
    print(f"Entites detectees : {entites}")

    if not entites:
        print("Aucune entite trouvee, recherche sans filtrage.")
        return se.rechercher(question, vectorizer, matrice_tfidf, df_docs, df, top_k)

    premiere_entite = entites[0]
    episodes_filtres = df[df['acteur'].str.lower() == premiere_entite.lower()][['saison', 'episode']].drop_duplicates()

    for entite in entites[1:]:
        episodes_entite = df[df['acteur'].str.lower() == entite.lower()][['saison', 'episode']].drop_duplicates()
        episodes_filtres = episodes_filtres.merge(episodes_entite, on=['saison', 'episode'])

    print(f"Episodes filtres : {len(episodes_filtres)}")

    if episodes_filtres.empty:
        print("Aucun episode commun trouve, recherche sans filtrage.")
        return se.rechercher(question, vectorizer, matrice_tfidf, df_docs, df, top_k)

    df_docs_filtre = df_docs.merge(episodes_filtres, on=['saison', 'episode'])

    if df_docs_filtre.empty:
        print("Aucun document apres filtrage, recherche sans filtrage.")
        return se.rechercher(question, vectorizer, matrice_tfidf, df_docs, df, top_k)

    indices_filtres = df_docs_filtre.index.tolist()
    matrice_filtree = matrice_tfidf[indices_filtres]
    df_docs_filtre = df_docs_filtre.reset_index(drop=True)

    return se.rechercher(question, vectorizer, matrice_filtree, df_docs_filtre, df, top_k)
