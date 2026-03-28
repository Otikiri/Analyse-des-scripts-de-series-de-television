# ==========================================================
# LIBRAIRIES ET MODULES
# ==========================================================
import re # pour obtenir le texte brut
import pandas as pd

# Prend en parametre le nom d'un ficheir sous forme d'un path 
# retourne un dataframe [acteur,ligne]
def parserScript(nomFich): 
    dialogues = []
    fich = open(nomFich,'r')
    #print("fichier:",nomFich,"ouvert")
    lignes = fich.readlines()
    for ligne in lignes[4:]: # on ignore les 4 premieres lignes : nom script + auteur + 2 \n
        dialogue = re.match(r'^([A-Za-z\s]+):\s(.+)',ligne,flags=re.IGNORECASE)
        if dialogue: 
            acteurBrute = dialogue.group(1) # Nom du personnage
            ligne = dialogue.group(2) # Ce qu'il dit

            acteurs = re.split(r'\s+and\s+|&|,', acteurBrute, flags=re.IGNORECASE) # On split les acteurs quand ils parlent en meme temps 
            acteurs = [a.strip() for a in acteurs]   # ["ROSS", "JOEY"] + enleve les espaces vides

            for acteur in acteurs:
                dialogues.append({'acteur':acteur,'ligne':ligne})
    #print("parse done:",nomFich)
    return pd.DataFrame(dialogues)

# Prend en parametre le path du fichier des datasets
# retourne un dataframe [acteur,ligne,saison,episode,nom_fichier]
def parserRepertoire(path):
    import os,glob
    toutDonnes = []

    for pathFichier in glob.glob(os.path.join(path, '**', '*.txt'), recursive=True): 
        nomBase = os.path.basename(pathFichier) #recuperer le nom du fichier du path 
        fich = re.match(r'S(\d+)E(\d+)',nomBase) #pattern match SXXEXX ... 
        if fich:
            saison = fich.group(1) 
            episode = fich.group(2) 
            df = parserScript(pathFichier)
            df['saison'] = saison  #rajoute la colone pour la saison
            df['episode'] = episode #rajoute la colone pour le numero de l'episode
            df['nom fichier'] = nomBase #rajoute le nom de l'episode
            toutDonnes.append(df) 
    return pd.concat(toutDonnes,ignore_index=True)  #pour avoir un seul tableau de donnees

