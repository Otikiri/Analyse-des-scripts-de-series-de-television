# ==========================================================
# LIBRAIRIES ET MODULES
# ==========================================================
import re
import pandas as pd

# Noms a exclure : ne sont pas de vrais personnages
EXCLURE = {"Written by", "Both", "All", "Everyone", "Cut to",
           "Scene", "Note", "Commercial Break", "Opening Credits"}

# Prend en parametre le nom d'un fichier sous forme d'un path
# retourne un dataframe [acteur, ligne, scene_num]
def parserScript(nomFich):
    dialogues = []

    with open(nomFich, 'r', encoding='utf-8', errors='replace') as fich:
        lignes = fich.read().replace('\r\n', '\n').split('\n')

    scene_num = 0
    derniere_entree = None  # garde en memoire la derniere ligne de dialogue pour la continuation

    for ligne in lignes[4:]:  # on ignore les 4 premieres lignes : nom script + auteur + 2 \n
        ligne = ligne.strip()
        if not ligne:
            continue

        # Detection des scenes : les lignes de scene commencent par [
        if ligne.startswith('['):
            scene_num += 1
            derniere_entree = None
            continue

        # On ignore les lignes qui sont uniquement de la mise en scene entre parentheses
        if ligne.startswith('(') and ligne.endswith(')'):
            derniere_entree = None
            continue

        # Detection d'une nouvelle ligne de dialogue : pattern NOM: texte
        dialogue = re.match(r'^([A-Za-z\s]+):\s(.+)', ligne, flags=re.IGNORECASE)
        if dialogue:
            acteurBrute = dialogue.group(1).strip()
            texte = dialogue.group(2).strip()

            # On ignore les labels qui ne sont pas des personnages
            if acteurBrute in EXCLURE:
                derniere_entree = None
                continue

            # On enleve la mise en scene dans le texte : (rires) etc.
            texte = re.sub(r'\([^)]*\)', '', texte).strip()

            # On split les acteurs quand ils parlent en meme temps
            acteurs = re.split(r'\s+and\s+|&|,', acteurBrute, flags=re.IGNORECASE)
            acteurs = [a.strip() for a in acteurs]  # enleve les espaces vides

            # On cree une entree par acteur et on garde la derniere pour la continuation
            derniere_entree = None
            for acteur in acteurs:
                entree = {'acteur': acteur, 'ligne': texte, 'scene_num': scene_num}
                dialogues.append(entree)
                derniere_entree = entree  # on garde la derniere entree pour la continuation

        # Continuation d'une ligne de dialogue sur plusieurs lignes
        elif derniere_entree is not None:
            suite = re.sub(r'\([^)]*\)', '', ligne).strip()
            if suite:
                derniere_entree['ligne'] += ' ' + suite

    return pd.DataFrame(dialogues)


# Prend en parametre le path du fichier des datasets
# retourne un dataframe [acteur, ligne, scene_num, saison, episode, nom_fichier]
def parserRepertoire(path):
    import os, glob
    toutDonnes = []

    for pathFichier in glob.glob(os.path.join(path, '**', '*.txt'), recursive=True):
        # on ignore les fichiers Zone.Identifier generes par Windows sur macOS
        if ':' in pathFichier:
            continue
        nomBase = os.path.basename(pathFichier)  # recuperer le nom du fichier du path
        fich = re.match(r'S(\d+)E(\d+)', nomBase)  # pattern match SXXEXX ...
        if fich:
            saison = fich.group(1)
            episode = fich.group(2)
            df = parserScript(pathFichier)
            df['saison'] = saison       # rajoute la colonne pour la saison
            df['episode'] = episode     # rajoute la colonne pour le numero de l'episode
            df['nom fichier'] = nomBase # rajoute le nom de l'episode
            toutDonnes.append(df)

    return pd.concat(toutDonnes, ignore_index=True)  # pour avoir un seul tableau de donnees
