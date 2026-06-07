import spacy
import moteur as mt

nlp = spacy.load("en_core_web_sm")


# ======================================================================
#  Extraction d'entites (personnages + lieux)
# ======================================================================

def extraireEntites(question, df=None):
    """Extrait les noms de personnages et lieux mentionnés dans la question.

    Combine l'outil de reconnaissance d'entités nommées (NER) de spaCy et le DataFrame 
    des acteurs en guise de solution de repli (fallback). Ignore les verbes majuscules 
    pour éviter les faux positifs.

    Args:
        question (str): La question posée par l'utilisateur.
        df (pd.DataFrame, optional): DataFrame contenant la liste des acteurs connus pour le fallback. Defaults to None.

    Returns:
        list of str: Une liste des entités uniques détectées, en conservant l'ordre d'apparition.
    """
    doc = nlp(question)
    entites = []

    # 1. spaCy NER
    for ent in doc.ents:
        if ent.root.pos_ == 'VERB':
            continue
        if ent.label_ in {"PERSON", "GPE", "LOC", "FAC", "ORG"}:
            entites.append(ent.text.strip())

    # 2. Fallback : verification dans le DataFrame des acteurs
    if df is not None:
        acteurs_connus = set(df['acteur'].str.lower().unique())
        for mot in question.split():
            mot_lower = mot.lower().strip('?.,!;:')
            
            if mot_lower in acteurs_connus:
                nom_titre = mot_lower.title()
                
                # Verifie si le nom n'est pas deja dans la liste des entites
                deja_present = False
                for e in entites:
                    if nom_titre.lower() in e.lower() or e.lower() in nom_titre.lower():
                        deja_present = True
                        break
                        
                if not deja_present:
                    entites.append(nom_titre)

    # Deduplication en preservant l'ordre
    seen = set()
    entites_uniques = []
    for e in entites:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            entites_uniques.append(e)

    return entites_uniques


# ======================================================================
#  Filtrage d'episodes par entites
# ======================================================================

def filtrer_par_scene(df, entites):
    """Trouve les épisodes où TOUTES les entités apparaissent dans la même scène.

    Parcourt le DataFrame groupé par saison, épisode et numéro de scène pour 
    vérifier la cooccurrence des entités spécifiées au niveau microscopique (la scène).

    Args:
        df (pd.DataFrame): Le DataFrame contenant le script des épisodes.
        entites (list of str): Les entités à rechercher.

    Returns:
        set of tuple: Un ensemble de tuples (saison, episode) correspondants.
    """
    episodes = set()
    
    for (saison, episode), groupe in df.groupby(['saison', 'episode']):
        for scene, grp_scene in groupe.groupby('scene_num'):
            texte_scene = ' '.join(grp_scene['ligne'].str.lower())
            acteurs_scene = set(grp_scene['acteur'].str.lower())
            
            # On verifie que chaque entite est dans la scene
            toutes_presentes = True
            for e in entites:
                if e.lower() not in texte_scene and e.lower() not in acteurs_scene:
                    toutes_presentes = False
                    break
                    
            if toutes_presentes:
                episodes.add((saison, episode))
                break  # On a trouve une scene, on passe a l'episode suivant
                
    return episodes


def filtrer_par_episode(df, entites):
    """Trouve les épisodes où TOUTES les entités apparaissent (pas forcément dans la même scène).

    Parcourt le DataFrame groupé par saison et épisode pour vérifier la cooccurrence 
    des entités spécifiées au niveau macroscopique (l'épisode entier).

    Args:
        df (pd.DataFrame): Le DataFrame contenant le script des épisodes.
        entites (list of str): Les entités à rechercher.

    Returns:
        set of tuple: Un ensemble de tuples (saison, episode) correspondants.
    """
    episodes = set()
    
    for (saison, episode), groupe in df.groupby(['saison', 'episode']):
        texte_episode = ' '.join(groupe['ligne'].str.lower())
        acteurs_episode = set(groupe['acteur'].str.lower())
        
        # On verifie que chaque entite est dans l'episode
        toutes_presentes = True
        for e in entites:
            if e.lower() not in texte_episode and e.lower() not in acteurs_episode:
                toutes_presentes = False
                break
                
        if toutes_presentes:
            episodes.add((saison, episode))
            
    return episodes


# ======================================================================
#  Moteur de recherche Q1
# ======================================================================

def rechercherQ1(question, df, vectorizer, matrice_tfidf, df_docs,
                 top_k=5, motsVidesRecherche=None, entites=None):
    """Moteur de recherche pour les questions de type Q1 (avec entités).
    
    Stratégie de recherche en cascade :
      1. Filtre par scène (toutes entités dans la même scène)
      2. Si aucun résultat -> filtre par épisode (toutes entités dans le même épisode)
      3. Si aucun résultat -> recherche globale sans filtrage
    
    Puis classe les épisodes filtrés par similarité TF-IDF avec la question.

    Args:
        question (str): La question posée.
        df (pd.DataFrame): Le DataFrame source des scripts.
        vectorizer (TfidfVectorizer): Le vectoriseur pré-entraîné.
        matrice_tfidf (scipy.sparse.csr_matrix): La matrice TF-IDF pré-calculée.
        df_docs (pd.DataFrame): Le DataFrame des documents groupés par épisode.
        top_k (int, optional): Nombre de résultats à retourner. Defaults to 5.
        motsVidesRecherche (list, optional): Mots vides à ignorer. Defaults to None.
        entites (list of str, optional): Entités pré-extraites. Si None, elles seront extraites. Defaults to None.

    Returns:
        pd.DataFrame: Un DataFrame contenant les top_k résultats classés.
    """
    if entites is None:
        entites = extraireEntites(question, df)
    print(f"Entites detectees : {entites}")

    if not entites:
        print("Aucune entite trouvee, recherche sans filtrage.")
        return mt.rechercher(question, vectorizer, matrice_tfidf, df_docs, df,
                             top_k, motsVidesRecherche=motsVidesRecherche)

    # --- Cascade de filtrage ---

    # Niveau 1 : meme scene
    episodes_set = filtrer_par_scene(df, entites)
    niveau = "scene"

    # Niveau 2 : meme episode
    if not episodes_set:
        episodes_set = filtrer_par_episode(df, entites)
        niveau = "episode"

    # Niveau 3 : aucun filtrage
    if not episodes_set:
        print("Aucun episode trouve avec toutes les entites, recherche sans filtrage.")
        return mt.rechercher(question, vectorizer, matrice_tfidf, df_docs, df,
                             top_k, motsVidesRecherche=motsVidesRecherche)

    print(f"Episodes filtres ({niveau}) : {len(episodes_set)}")

    indices_a_garder = []
    
    for index, row in df_docs.iterrows():
        if (row['saison'], row['episode']) in episodes_set:
            indices_a_garder.append(index)

    matrice_filtree = matrice_tfidf[indices_a_garder]
    df_docs_filtre = df_docs.iloc[indices_a_garder].reset_index(drop=True)

    if df_docs_filtre.empty:
        print("Aucun document apres filtrage, recherche sans filtrage.")
        return mt.rechercher(question, vectorizer, matrice_tfidf, df_docs, df,
                             top_k, motsVidesRecherche=motsVidesRecherche)

    return mt.rechercher(question, vectorizer, matrice_filtree, df_docs_filtre, df,
                         top_k, motsVidesRecherche=motsVidesRecherche)