import os
import glob
import re
import pandas as pd

# Noms a exclure : ne sont pas de vrais personnages
EXCLURE = {"Written By", "Everyone", "Everybody", "Cut To",
           "Scene", "Note", "Commercial Break", "Opening Credits",
           "Writer", "Intercom", "Announcer",
           "Directed By", "Produced By", "Story By", "Teleplay By",
           "Answering Machine", "Hypnosis Tape", "Machine", "Message",
           "Oven", "Radio", "Smoke Detector", "Tape", "Second Message",
           "Commercial", "Commercial Voiceover"}

# Verbes d'action et connecteurs : si present dans le nom de l'acteur, on coupe avant
# ex: "Monica To Ross" → "Monica", "Phoebe Shakes Her Hand" → "Phoebe"
VERBES_ACTION = {
    "looks", "turns", "walks", "storms", "starts", "raises", "gets",
    "comes", "goes", "runs", "sits", "stands", "enters", "exits",
    "screams", "screaming", "cries", "shakes", "crying", "laughs", "laughing",
    "smiles", "smiling", "grabs", "pushes", "pulls", "opens", "closes",
    "puts", "takes", "moves", "stares", "staring", "points", "pointing",
    "holds", "holding", "fiddling", "hums", "tries", "trying",
    "to", "at", "but", "while", "with", "about", "almost", "simultaneously", "says", "except","coming","starting"
}

def clean_text(text):
    """Supprime les balises HTML ou XML imbriquées ainsi que toutes les annotations textuelles placées entre parenthèses.
    Args:
        text (str): La chaîne de caractères brute à nettoyer.
    Returns:
        str: Le texte filtré et débarrassé des espaces superflus.
    """
    return re.sub(r'<[^>]*>', '', re.sub(r'\([^)]*\)', '', text)).strip()

def parse_actor_name(raw):
    """Analyse, tronque et nettoie le nom brut d'un intervenant en éliminant les indications de mise en scène ou de mouvement corporel jointes.
    Args:
        raw (str): Le nom extrait brut du fichier de script.
    Returns:
        str or None: Le nom de l'acteur nettoyé ou None si la chaîne commence directement par un verbe ou un modificateur d'action invalide.
    """
    mots = raw.split()
    idx = next((i for i, w in enumerate(mots) if w.lower() in VERBES_ACTION), None)
    if idx is None:
        return raw
    if idx == 0:
        return None
    before = mots[:idx]
    if before[0].lower() not in {'the', 'a', 'an', 'woman', 'man', 'girl', 'boy', 'lady', 'guy', 'person'}:
        return ' '.join(before)
    return raw

def parserScript(nomFich):
    """Analyse un fichier script texte individuel d'un épisode pour extraire les répliques chronologiques de chaque personnage et l'index de scène associé.

    Args:
        nomFich (str): Chemin d'accès absolu ou relatif vers le fichier de script `.txt`.

    Returns:
        pd.DataFrame: Un DataFrame structuré contenant les colonnes : ['acteur', 'ligne', 'scene_num'].
    """
    dialogues = []

    with open(nomFich, 'r', encoding='utf-8', errors='replace') as fich:
        lignes = fich.read().replace('\r\n', '\n').split('\n')

    scene_num = 0
    derniere_entree = None  # derniere replique, pour gerer les continuations multi-lignes

    for ligne in lignes[4:]:  # ignore les 4 premieres lignes : titre + auteur + 2 \n
        ligne = ligne.strip()
        if not ligne:
            continue

        if ligne.startswith('['):
            scene_num += 1
            derniere_entree = None
            continue

        if ligne.startswith('<') and ligne.endswith('>'):
            derniere_entree = None
            continue

        if ligne.startswith('(') and ligne.endswith(')'):
            derniere_entree = None
            continue

        dialogue = re.match(r'^([A-Za-z\s]+):\s(.+)', ligne, flags=re.IGNORECASE)
        if dialogue:
            acteurBrute = dialogue.group(1).strip().title()
            texte = dialogue.group(2).strip()

            if acteurBrute in EXCLURE:
                derniere_entree = None
                continue

            acteurBrute = parse_actor_name(acteurBrute)
            if acteurBrute is None:
                derniere_entree = None
                continue

            texte = clean_text(texte)

            # Split quand plusieurs acteurs parlent en meme temps : "Ross And Rachel", "Joey & Chandler"
            acteurs = re.split(r'\s+and\s+|&|,', acteurBrute, flags=re.IGNORECASE)
            acteurs = [a.strip() for a in acteurs if a.strip()]

            derniere_entree = None
            for acteur in acteurs:
                if acteur in EXCLURE:
                    continue
                entree = {'acteur': acteur, 'ligne': texte, 'scene_num': scene_num}
                dialogues.append(entree)
                derniere_entree = entree

        # Continuation d'une replique sur plusieurs lignes
        elif derniere_entree is not None:
            # Une ligne avec un verbe d'action est de la mise en scene, pas du dialogue
            if any(w.lower() in VERBES_ACTION for w in ligne.split()):
                derniere_entree = None
                continue
            suite = clean_text(ligne)
            if suite:
                derniere_entree['ligne'] += ' ' + suite

    return pd.DataFrame(dialogues)

def parserRepertoire(path):
    """Parcourt de manière récursive un répertoire à la recherche de scripts conformes à un format de nommage saison/épisode (ex: S01E01.txt) et consolide le tout.
    Args:
        path (str): Emplacement racine du dossier hébergeant les scripts.
    Returns:
        pd.DataFrame: Un DataFrame unifié comprenant les colonnes : ['acteur', 'ligne', 'scene_num', 'saison', 'episode', 'nom fichier'].
    """
    toutDonnes = []

    for pathFichier in glob.glob(os.path.join(path, '**', '*.txt'), recursive=True):
        if ':' in pathFichier:  # ignore les fichiers Zone.Identifier generes par Windows sur macOS
            continue
        nomBase = os.path.basename(pathFichier)
        fich = re.match(r'S(\d+)E(\d+)', nomBase)
        if fich:
            saison = fich.group(1)
            episode = fich.group(2)
            df = parserScript(pathFichier)
            df['saison'] = saison
            df['episode'] = episode
            df['nom fichier'] = nomBase
            toutDonnes.append(df)

    return pd.concat(toutDonnes, ignore_index=True)
