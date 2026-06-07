"""
SearchEngine : couche OOP fine qui délègue le travail à moteur.py,
q1_search.py et q2_search.py.
"""

import os
import sys

# S'assurer que le dossier src est dans le path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader as dl
import moteur as mt
import q1_search as q1
import q2_search as q2

PERSONNAGES = {'ross', 'rachel', 'monica', 'chandler', 'joey', 'phoebe',
    'mike', 'richard', 'emily', 'gunther', 'janice', 'carol'}

class SearchEngine:
    def __init__(self, data_path):
        """Charge les données et construit les index TF-IDF une seule fois au démarrage.

        Args:
            data_path (str): Le chemin vers le dossier contenant les fichiers de données.

        Raises:
            ValueError: Si le chargement des données échoue ou si le dataframe est vide.
            
        Returns:
            None
        """
        print("Chargement des données pour le moteur de recherche...")
        self.df = dl.chargerDonnees(data_path)
        if self.df is None or self.df.empty:
            raise ValueError("Le chargement des données a échoué ou le dataframe est vide.")

        print("Construction de l'index TF-IDF Q1 (moteur.py)...")
        self.vectorizer, self.matrice_tfidf, self.df_docs, self.mots_vides = (
            mt.construireIndex(self.df, par_scene=False)
        )
        
        print("Construction de l'index TF-IDF Q2 (q2_search.py)...")
        self.q2_matrix, self.q2_vectorizer, self.q2_docs = q2.vectorize_Q2(
            self.df, groupby_cols=["saison", "episode"], max_df=0.95, ngram_range=(1, 2)
        )
        self.q2_matrix_lines, self.q2_vectorizer_lines, self.q2_docs_lines = q2.vectorize_Q2(
            self.df, groupby_cols=["saison", "episode", "ligne"], max_df=0.95, ngram_range=(1, 2)
        )
        print("Moteur de recherche prêt.")

    # ------------------------------------------------------------------
    #  Classification de la question
    # ------------------------------------------------------------------


    def classify_question(self, question):
        """Détermine si une question est de type Q1 (avec entités) ou Q2 (sans entités).

        Utilise l'extraction d'entités de q1_search.py et vérifie si les entités 
        trouvées correspondent à des personnages principaux connus.

        Args:
            question (str): La question posée par l'utilisateur.

        Returns:
            tuple: Un tuple (type_str, liste_entites) où type_str est "Q1" ou "Q2".
        """

        entities = q1.extraireEntites(question, self.df)

        #on ne garde que les personnages "reels" => Waiter pas une entite
        entites_valides = [e for e in entities if e.lower() in PERSONNAGES]

        if entites_valides:
            return "Q1", entities
        return "Q2", []

    # ------------------------------------------------------------------
    #  Normalisation des résultats
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(records):
        """Normalise le format des dictionnaires de résultats pour l'interface graphique.

        Garantit que chaque dictionnaire possède les mêmes clés : score, saison, episode, 
        title, et texte, peu importe s'il provient d'une recherche Q1 ou Q2.

        Args:
            records (list of dict): Les résultats bruts issus de la recherche.

        Returns:
            list of dict: Les résultats formatés et normalisés.
        """
        out = []
        for r in records:
            # Pour Q1, 'texte' contient 'texte nettoyer' (sans ponctuation). On l'ignore pour l'interface.
            # Pour Q2, on récupère la ligne exacte via 'line'.
            texte_affiche = r.get("line") or ""
            
            out.append({
                "score":   r.get("score", 0),
                "saison":  r.get("saison", "?"),
                "episode": r.get("episode", "?"),
                "title":   r.get("title") or r.get("nom fichier") or "",
                "texte":   texte_affiche,
            })
        return out

    # ------------------------------------------------------------------
    #  Recherche Q1 – par entités (q1_search.rechercherQ1)
    # ------------------------------------------------------------------

    def search_q1(self, question, top_k=5, entites=None):
        """Effectue une recherche de type Q1 basée sur le filtrage par entités.

        Filtre les documents par les entités détectées, puis classe les résultats 
        restants en utilisant la similarité TF-IDF.

        Args:
            question (str): La question de l'utilisateur.
            top_k (int, optional): Le nombre maximal de résultats à retourner. Defaults to 5.
            entites (list of str, optional): Liste des entités préalablement extraites. Defaults to None.

        Returns:
            list of dict: Les top_k résultats correspondants, au format normalisé.
        """
        results_df = q1.rechercherQ1(
            question, self.df,
            self.vectorizer, self.matrice_tfidf, self.df_docs,
            top_k=top_k, motsVidesRecherche=self.mots_vides,
            entites=entites
        )
        return self._normalize(results_df.to_dict("records"))

    # ------------------------------------------------------------------
    #  Recherche Q2 – sémantique (q2_search.utiliser_moteur_Q2)
    # ------------------------------------------------------------------

    def search_q2(self, question, top_k=5):
        """Effectue une recherche de type Q2 par similarité sémantique globale.

        Recherche la similarité TF-IDF sur le corpus complet sans filtrage préalable 
        par entités, idéal pour les questions d'action ou d'objets.

        Args:
            question (str): La question de l'utilisateur.
            top_k (int, optional): Le nombre maximal de résultats à retourner. Defaults to 5.

        Returns:
            list of dict: Les top_k résultats correspondants, au format normalisé.
        """
        print("q2_docs shape before search:", self.q2_docs.shape)
        print("q2_matrix shape before search:", self.q2_matrix.shape)

        results = q2.utiliser_moteur_Q2(
            question, 
            vectorizer=self.q2_vectorizer,tfidf_matrix=self.q2_matrix,docs=self.q2_docs,
            tfidf_matrix_lines=self.q2_matrix_lines,vectorizer_lines=self.q2_vectorizer_lines, 
            docs_lines=self.q2_docs_lines,top_k=top_k
        )

        if hasattr(results, 'to_dict'):
            records = results.to_dict("records")
        else:
            records = results
        return self._normalize(records)