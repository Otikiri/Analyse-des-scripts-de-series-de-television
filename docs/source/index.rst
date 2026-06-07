.. Analyse des scripts de la series FRIENDS documentation master file, created by
   sphinx-quickstart on Sun Jun  7 13:00:25 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Analyse des scripts de la series FRIENDS documentation
======================================================

Bienvenue dans la documentation officielle du projet d'analyse textuelle et de traitement automatique du langage naturel (TAL) appliqué aux scripts complets de la série télévisée **FRIENDS**.

Description du Projet
---------------------
Ce projet explore la dynamique linguistique, thématique et comportementale des personnages emblématiques de la série. À l'aide de modèles statistiques et algorithmiques avancés, l'application permet de :

* **Analyser le lexique :** Extraire les statistiques fréquentielles globales, par épisode et par protagoniste.
* **Modéliser des thématiques :** Identifier des sujets latents par saison via l'allocation de Dirichlet latente (LDA) et KeyBERT.
* **Regrouper le vocabulaire :** Effectuer du clustering sémantique de mots à l'aide de représentations vectorielles Word2Vec et KMeans.
* **Moteur de recherche intelligent :** Explorer le corpus de manière granulaire grâce à une indexation TF-IDF inversée évaluant la similarité cosinus.

.. toctree::
   :maxdepth: 2
   :caption: Index des Modules Générés:

   cluster
   data_loader
   fiches_acteurs
   graphe_acteurs
   gui
   main
   moteur
   parser
   q1_search
   q2_search
   search_engine
   stats_lexical
   test_search
   utils

Indices et Tables
-----------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

