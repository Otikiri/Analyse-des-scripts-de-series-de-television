import tkinter as tk
from tkinter import ttk, scrolledtext
import os
import sys
import pandas as pd

# S'assurer que le dossier src est dans le path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search_engine import SearchEngine
from moteur import miseEnFormeRes

class App(tk.Tk):
    """
    Classe principale de l'application GUI pour la recherche dans les scripts.
    """
    def __init__(self, search_engine, sujet_df):
        """Initialise la fenêtre principale de l'application.

        Args:
            search_engine (SearchEngine): L'instance du moteur de recherche pré-chargée.
            sujet_df (pd.DataFrame): Le DataFrame contenant les sujets des épisodes.

        Returns:
            None
        """
        super().__init__()
        self.search_engine = search_engine
        self.sujet_df = sujet_df
        self.title("Moteur de Recherche - Scripts de Friends")
        self.geometry("800x600")
        
        self.create_widgets()

    def create_widgets(self):
        """Crée les widgets de l'interface graphique.
        
        Initialise et place les différents éléments de l'interface utilisateur, 
        incluant la zone de saisie, le bouton de recherche, la barre de statut 
        et la zone d'affichage des résultats.
        
        Args:
            None

        Returns:
            None
        """
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Zone de saisie ---
        input_frame = ttk.LabelFrame(main_frame, text="Posez votre question", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        self.search_button = ttk.Button(input_frame, text="Rechercher", command=self.perform_search)
        self.search_button.pack(side=tk.RIGHT)
        
        self.question_entry = ttk.Entry(input_frame, width=80)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.question_entry.focus()
        
        self.question_entry.bind("<Return>", self.perform_search)
        
        # --- Barre de statut ---
        self.status_label = ttk.Label(main_frame, text="Prêt.")
        self.status_label.pack(fill=tk.X, pady=5)
        
        # --- Zone de résultats ---
        results_frame = ttk.LabelFrame(main_frame, text="Résultats", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, state='disabled', font=('Courier New', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def perform_search(self, event=None):
        """Lance le processus de recherche à partir de la question de l'utilisateur.
        
        Récupère la question saisie, détermine son type (Q1 ou Q2) via le SearchEngine, 
        effectue la recherche correspondante, et met à jour l'interface avec les résultats.
        
        Args:
            event (tk.Event, optional): L'événement déclencheur (ex: appui sur Entrée). Defaults to None.

        Returns:
            None
        """
        question = self.question_entry.get()
        if not question:
            return
            
        self.search_button.config(state='disabled')
        self.status_label.config(text="Analyse de la question...")
        self.update_idletasks()
        
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', tk.END)
        
        # Classification via SearchEngine (délègue à q1_search.extraireEntites)
        q_type, entities = self.search_engine.classify_question(question)
        
        self.status_label.config(text=f"Question de type '{q_type}' détectée. Recherche en cours...")
        self.update_idletasks()
        
        if q_type == "Q1":
            results = self.search_engine.search_q1(question, entites=entities)
            header = f"Recherche par entités : {', '.join(entities)}"
        else:
            results = self.search_engine.search_q2(question)
            header = "Recherche par contenu (similarité sémantique)"
        res_df = pd.DataFrame(results)
        res_df['question'] = question
        self.display_results(miseEnFormeRes(res_df,self.sujet_df), header)

        self.status_label.config(text="Recherche terminée.")
        self.search_button.config(state='normal')
        self.results_text.config(state='disabled')

    def display_results(self, results_df, header):
        """Affiche les résultats de recherche (format unifié Q1/Q2).
        
        Prend un DataFrame contenant les résultats formatés et les insère dans 
        la zone de texte de l'interface graphique avec un en-tête approprié.
        
        Args:
            results_df (pd.DataFrame): Le DataFrame contenant les résultats (score, saison, episode, etc.).
            header (str): Le titre ou l'en-tête à afficher au-dessus des résultats.

        Returns:
            None
        """
        self.results_text.insert(tk.END, f"{header}\n")
        self.results_text.insert(tk.END, "="*80 + "\n\n")
        
        if results_df is None or results_df.empty:
            self.results_text.insert(tk.END, "Aucun résultat trouvé.")
            return

        for _, row in results_df.iterrows():
            score   = row.get('score', 0)
            saison  = row.get('saison', '?')
            episode = row.get('episode', '?')
            titre   = row.get('titre', '')
            sujet   = row.get('sujet', '')
            rank    = row.get('rank', '?')

            ep_label = f"S{saison}E{episode}"
            if titre:
                ep_label += f" ({titre})"

            self.results_text.insert(tk.END, f"--- Résultat #{rank} (Score: {score:.4f}) ---\n")
            self.results_text.insert(tk.END, f"Épisode : {ep_label}\n")
            self.results_text.insert(tk.END, f"Sujet   : {sujet}\n")
            self.results_text.insert(tk.END, "\n")


# if __name__ == "__main__":
#     DATA_PATH = "../datasets"
#     print("Initialisation du moteur de recherche... (cela peut prendre un moment)")
#     search_engine = SearchEngine(DATA_PATH)
#     print("Lancement de l'interface graphique...")
#     sujet_df = pd.read_csv("sujet_par_ep.csv")
#     sujet_df['saison'] = sujet_df['saison'].astype(str).str.zfill(2)
#     sujet_df['episode'] = sujet_df['episode'].astype(str).str.zfill(2)
#     app = App(search_engine,sujet_df)
#     app.mainloop()