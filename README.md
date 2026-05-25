# Analyse de Scripts de Séries Télévisées - Friends

## Introduction

Ce projet a pour objectif l'analyse approfondie d'un corpus de scripts de la série télévisée "Friends" à l'aide de techniques de Traitement Automatique du Langage Naturel (NLP). Il se divise en deux grandes parties :

1. **Analyse exploratoire du corpus** pour en extraire des statistiques et des thèmes récurrents.
2. **Développement d'un moteur de recherche** capable de répondre à des questions en langage naturel sur le contenu des épisodes.

L'ensemble est accessible via une interface graphique simple et intuitive.

---

## Partie 1 : Exploration et Analyse du Corpus

Cette partie se concentre sur la compréhension des données textuelles. Les scripts ont été parsés pour extraire les dialogues, les personnages et la structure des scènes.

### Tâches Réalisées

- **Analyse Statistique et Dictionnaire Lexical** (`src/stats_lexical.py`, `src/data_loader.py`)
  
  - Création de tableaux statistiques : nombre d'acteurs par épisode, nombre de mots, nombre d'échanges moyen.
  - Calcul des fréquences de mots (tokens) à différentes granularités :
    - Fréquence globale sur toute la série.
    - Fréquence par acteur, pour caractériser le langage de chaque personnage.
    - Fréquence par épisode.
    - Évolution de la fréquence de certains mots-clés au fil des saisons.
- **Extraction de Sujets (Topic Modeling)** (`src/cluster.py`)
  
  - **Clustering TF-IDF & KMeans** : Les documents (scènes ou épisodes) sont représentés par des vecteurs TF-IDF, puis regroupés en clusters pour identifier des thèmes basés sur les mots-clés discriminants.
  - **Clustering Word2Vec & KMeans** : Les mots sont plongés dans un espace vectoriel sémantique avec Word2Vec. Le clustering de ces vecteurs permet de regrouper les mots par contexte d'utilisation.
  - **Modélisation par LDA (Latent Dirichlet Allocation)** : Une approche probabiliste pour découvrir les "sujets" abstraits qui composent les documents du corpus.

---

## Partie 2 : Moteur de Recherche d'Information

L'objectif est de permettre à un utilisateur de retrouver des scènes ou des épisodes en posant des questions en langage naturel.

### Fonctionnalités

- **Classification des Questions** (`src/question_classifier.py`)
  
  - Le système analyse la question de l'utilisateur pour la classer en deux types :
    - **Type Q1** : La question contient une ou plusieurs entités nommées (personnages, lieux). *Ex : "Le mariage de Monica et Chandler"*.
    - **Type Q2** : La question décrit un événement ou une scène sans nommer de personnage. *Ex : "quelqu'un met une dinde sur la tête"*.
  - L'extraction d'entités est réalisée avec la bibliothèque **SpaCy** et enrichie d'un dictionnaire spécifique à l'univers de Friends.
- **Mécanismes de Recherche** (`src/search_engine.py`)
  
  - **Pour les questions Q1** : Le moteur filtre les scènes pour ne garder que celles où les personnages mentionnés sont présents et interagissent.
  - **Pour les questions Q2** : Le moteur utilise une approche de recherche sémantique. Les scripts et la question sont transformés en vecteurs TF-IDF, et la **similarité cosinus** est utilisée pour classer les épisodes les plus pertinents.
- **Interface Graphique** (`gui.py`)
  
  - Une application de bureau développée avec Tkinter permet de poser des questions et d'afficher les résultats de manière claire et lisible.

---

## Structure et Description des Fichiers

```
.
├── datasets/             # Contient les scripts de la série par saison
├── src/                  # Code source du projet
│   ├── parser.py         # Module pour parser les fichiers de script
│   ├── data_loader.py    # Chargement et pré-traitement des données
│   ├── utils.py          # Fonctions utilitaires (nettoyage de texte, etc.)
│   ├── stats_lexical.py  # Fonctions pour les statistiques lexicales
│   ├── cluster.py        # Algorithmes de clustering (W2V, LDA, TF-IDF)
│   ├── search_engine.py  # Classe principale du moteur de recherche
│   └── question_classifier.py # Classification des questions (Q1/Q2)
├── gui.py                # Point d'entrée de l'application (Interface Graphique)
└── README.md             # Ce fichier
```

## Installation et Lancement

### Prérequis

- Python 3.8 ou supérieur
- `pip` et `virtualenv` (recommandé)

### Étapes d'installation

1. **Cloner le dépôt** (ou télécharger les fichiers) :
  
  ```bash
  git clone <url-du-depot>
  cd Analyse-des-scripts-de-series-de-television
  ```
  
2. **(Recommandé) Créer un environnement virtuel** :
  
  ```bash
  python -m venv venv
  source venv/bin/activate  # Sur Windows: venv\Scripts\activate
  ```
  
3. **Installer les dépendances** :
  Le fichier `requirements.txt` contient toutes les bibliothèques nécessaires.
  
  ```bash
  pip install -r requirements.txt
  ```
  
4. **Télécharger les modèles de langue SpaCy** :
  Ces modèles sont nécessaires pour la classification des questions.
  
  ```bash
  python -m spacy download fr_core_news_sm
  python -m spacy download en_core_web_sm
  ```
  

### Lancement de l'application

Une fois l'installation terminée, lancez l'interface graphique avec la commande suivante à la racine du projet :

```bash
python gui.py
```

Une fenêtre s'ouvrira, vous permettant d'interagir avec le moteur de recherche. Le premier lancement peut prendre un moment, le temps de charger et de vectoriser les données.

## Comment utiliser le moteur de recherche

1. Lancez l'application `gui.py`.
2. Dans la zone de texte, tapez votre question en français.
3. Cliquez sur "Rechercher" ou appuyez sur `Entrée`.

### Exemples de questions

- **Type Q1 (avec personnages)** :
  
  - `Joey apprend le français pour une audition`
  - `La fois où Monica et Chandler se fiancent`
  - `Scènes avec Ross et Rachel à l'hôpital`
- **Type Q2 (description de scène)** :
  
  - `quelqu'un met une dinde sur la tête pour faire rire`
  - `boire un gallon de lait en dix secondes`
  - `le canapé qui ne passe pas dans l'escalier`

Les résultats s'afficheront dans la zone principale, formatés pour une lecture facile.