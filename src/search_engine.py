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
        """
        Charge les données et construit l'index TF-IDF une seule fois.
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
        """
        Détermine si une question est de type Q1 (avec entités) ou Q2 (sans).
        Utilise l'extraction d'entités de q1_search.py.
        Retourne (type_str, liste_entites).
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
        """
        Garantit que chaque dict a les clés : score, saison, episode, title, texte.
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
        """
        Recherche de type Q1 : filtre par entités puis classe par TF-IDF.
        Retourne une liste de dicts normalisés.
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
        """
        Recherche de type Q2 : similarité TF-IDF sur le corpus complet.
        Retourne une liste de dicts normalisés.
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