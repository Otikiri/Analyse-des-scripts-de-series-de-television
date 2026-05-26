import tkinter as tk
from tkinter import ttk, scrolledtext
import os
import sys

# S'assurer que le dossier src est dans le path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search_engine import SearchEngine


class App(tk.Tk):
    """
    Classe principale de l'application GUI pour la recherche dans les scripts.
    """
    def __init__(self, search_engine):
        super().__init__()
        self.search_engine = search_engine
        
        self.title("Moteur de Recherche - Scripts de Friends")
        self.geometry("800x600")
        
        self.create_widgets()

    def create_widgets(self):
        """Crée les widgets de l'interface graphique."""
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
        """Lance le processus de recherche à partir de la question de l'utilisateur."""
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

        self.display_results(results, header)

        self.status_label.config(text="Recherche terminée.")
        self.search_button.config(state='normal')
        self.results_text.config(state='disabled')

    def display_results(self, results, header):
        """Affiche les résultats de recherche (format unifié Q1/Q2)."""
        self.results_text.insert(tk.END, f"{header}\n")
        self.results_text.insert(tk.END, "="*80 + "\n\n")
        
        if not results:
            self.results_text.insert(tk.END, "Aucun résultat trouvé.")
            return

        for i, res in enumerate(results):
            score = res.get('score', 0)
            saison = res.get('saison', '?')
            episode = res.get('episode', '?')
            title = res.get('title', '')
            texte = res.get('texte', '')

            ep_label = f"S{saison}E{episode}"
            if title:
                ep_label += f" ({title})"

            self.results_text.insert(tk.END, f"--- Résultat #{i+1} (Score: {score:.4f}) ---\n")
            self.results_text.insert(tk.END, f"Épisode : {ep_label}\n")

            if texte:
                snippet = texte[:500] + '...' if len(texte) > 500 else texte
                self.results_text.insert(tk.END, f"Extrait : {snippet}\n")
            self.results_text.insert(tk.END, "\n")


if __name__ == "__main__":
    DATA_PATH = "../datasets"
    print("Initialisation du moteur de recherche... (cela peut prendre un moment)")
    search_engine = SearchEngine(DATA_PATH)
    print("Lancement de l'interface graphique...")
    app = App(search_engine)
    app.mainloop()