import os
import sys
import re
import glob
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import data_loader as dl
import utils as ut
import cluster as cl
import networkx as nx
import matplotlib.pyplot as plt


def parserEntetesScenes(cheminDossier):
    """Extrait les textes des en-têtes de scènes depuis les fichiers du dossier."""
    entetes = {}
    if not cheminDossier:
        return entetes
    for pathFichier in glob.glob(os.path.join(cheminDossier, '**', '*.txt'), recursive=True):
        if ':' in pathFichier:
            continue
        nomBase = os.path.basename(pathFichier)
        match = re.match(r'S(\d+)E(\d+)', nomBase)
        if not match:
            continue
        saison  = match.group(1)
        episode = match.group(2)
        with open(pathFichier, 'r', encoding='utf-8', errors='replace') as f:
            lignes = f.read().replace('\r\n', '\n').split('\n')
        scene_num = 0
        for ligne in lignes[4:]:
            ligne = ligne.strip()
            if ligne.startswith('['):
                scene_num += 1
                entetes[(saison, episode, scene_num)] = ligne
    return entetes


def trouverActeursDansTexte(texte, acteursConnus):
    """Identifie quels acteurs d'une liste connue sont mentionnés dans un texte."""
    trouves = set()
    texte_lower = texte.lower()
    for acteur in acteursConnus:
        pattern = r'\b' + re.escape(acteur.lower()) + r'\b'
        if re.search(pattern, texte_lower):
            trouves.add(acteur)
    return trouves


def construirePresenceParScene(df, entetes, acteursConnus):
    """Combine les présences issues des dialogues et des en-têtes pour chaque scène."""
    presence = defaultdict(set)
    for _, row in df.iterrows():
        cle = (row['saison'], row['episode'], row['scene_num'])
        presence[cle].add(row['acteur'])
    for cle, entete in entetes.items():
        acteurs_entete = trouverActeursDansTexte(entete, acteursConnus)
        presence[cle].update(acteurs_entete)
    return dict(presence)


def calculerLiaisons(df, presenceParScene):
    """Compte les scènes partagées et regroupe les dialogues pour chaque paire d'acteurs."""
    liaisons      = defaultdict(int)
    textes_paires = defaultdict(list)
    grouped = df.groupby(['saison', 'episode', 'scene_num'])
    for (saison, ep, scene), group in grouped:
        cle_scene = (saison, ep, scene)
        presents  = presenceParScene.get(cle_scene, set())
        
        acteurs_presents = sorted(list(presents))
        for i in range(len(acteurs_presents)):
            for j in range(i + 1, len(acteurs_presents)):
                paire = (acteurs_presents[i], acteurs_presents[j])
                liaisons[paire] += 1

        for _, row in group.iterrows():
            acteur  = row['acteur']
            ligne   = row['ligne']
            for autre in presents:
                if autre != acteur:
                    paire = tuple(sorted([acteur, autre]))
                    textes_paires[paire].append(ligne)
    return liaisons, textes_paires


def analyserReseau(df, cheminDossier):
    """Filtre les acteurs principaux, normalise leurs liaisons et extrait leurs sujets."""
    MAPPING_ACTEURS = {
        "Rach": "Rachel",
        "Rache": "Rachel",
        "Mnca": "Monica",
        "Phoe": "Phoebe",
        "Chan": "Chandler",
        "Steph": "Stephanie"
    }
    df['acteur'] = df['acteur'].replace(MAPPING_ACTEURS)

    ACTEURS_PRINCIPAUX = ["Rachel", "Monica", "Phoebe", "Joey", "Chandler", "Ross"]

    def nettoyer_nom_acteur(nom):
        nom_str = str(nom).strip()
        nom_lower = nom_str.lower()
        for acteur in ACTEURS_PRINCIPAUX:
            if nom_lower.startswith(acteur.lower()):
                return acteur
        return nom_str

    df['acteur'] = df['acteur'].apply(nettoyer_nom_acteur)
    df = df[df['acteur'].isin(ACTEURS_PRINCIPAUX)].copy()

    acteursConnus = ACTEURS_PRINCIPAUX

    entetes = parserEntetesScenes(cheminDossier)

    presence = construirePresenceParScene(df, entetes, acteursConnus)

    liaisons, textes_paires = calculerLiaisons(df, presence)

    episodes_par_acteur  = defaultdict(set)
    mots_par_acteur      = defaultdict(int)
    repliques_par_acteur = defaultdict(int)
    for _, row in df.iterrows():
        episodes_par_acteur[row['acteur']].add((row['saison'], row['episode']))
        mots_par_acteur[row['acteur']] += row['nombres de mots']
        repliques_par_acteur[row['acteur']] += 1

    resultats = []
    for paire, nb_scenes_communes in liaisons.items():
        actA, actB = paire
        eps_communs = episodes_par_acteur[actA].intersection(episodes_par_acteur[actB])
        nb_eps = len(eps_communs)
        if nb_eps == 0:
            continue
            
        tot_repliques = repliques_par_acteur[actA] + repliques_par_acteur[actB]
        poids_relatif = nb_scenes_communes / tot_repliques if tot_repliques > 0 else 0

        resultats.append({
            'Acteur 1':            actA,
            'Acteur 2':            actB,
            'Poids Relatif Moyen': round(poids_relatif, 6),
            'Scenes Communes':     nb_scenes_communes,
            'Nb Episodes Communs': nb_eps,
            '_textes':             " ".join(textes_paires[paire]),
        })

    df_res = pd.DataFrame(resultats)
    if df_res.empty:
        print("(attention) - Aucune interaction trouvée.")
        return pd.DataFrame(), acteursConnus, (mots_par_acteur, episodes_par_acteur)

    corpus = df_res['_textes'].tolist()
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=lambda txt: ut.tokeniserTexte(ut.nettoyerTexte(txt)),
            lowercase=False,
            token_pattern=None
        )
        tfidf_mat = vectorizer.fit_transform(corpus)
        features  = np.array(vectorizer.get_feature_names_out())
        
        # Préparation des données pour utiliser cl.nommerClusters
        wordVectors = tfidf_mat.T.toarray()
        wordLabels = np.argmax(wordVectors, axis=1)
        
        # Appel de la fonction du module cluster
        nomsBruts = cl.nommerClusters(wordLabels, wordVectors, features, nMotsNom=5, methode='tfidf')
        
        sujets = []
        for idx in range(len(df_res)):
            nom_cluster = nomsBruts.get(idx, f"Cluster {idx}")
            if nom_cluster.startswith("Cluster "):
                # Repli si aucun mot spécifique n'a été attribué à cette paire
                tokens = ut.tokeniserTexte(ut.nettoyerTexte(corpus[idx]))
                counts = Counter(tokens)
                top_mots = [w for w, _ in counts.most_common(5)]
                sujets.append(", ".join(top_mots) if top_mots else "aucun sujet")
            else:
                sujets.append(nom_cluster.replace(" / ", ", "))
                
    except Exception as e:
        print(f"tf-idf impossible ({e}). repli sur fréquence simple")
        sujets = []
        for txt in corpus:
            tokens   = ut.tokeniserTexte(ut.nettoyerTexte(txt))
            counts   = Counter(tokens)
            top_mots = [w for w, _ in counts.most_common(5)]
            sujets.append(", ".join(top_mots) if top_mots else "aucun sujet")

    df_res['Sujets Echanges'] = sujets
    df_res = df_res.drop(columns=['_textes'])
    df_res = df_res.sort_values('Poids Relatif Moyen', ascending=False).reset_index(drop=True)
    return df_res, acteursConnus, (mots_par_acteur, episodes_par_acteur)


def afficherGraphe(df_inter, metrics, output_png=None, limit_nodes=15, titre_saison=""):
    """Génère et sauvegarde une image du réseau d'interactions avec NetworkX."""
    mots_par_acteur, _ = metrics

    top_actors = sorted(mots_par_acteur, key=mots_par_acteur.get, reverse=True)[:limit_nodes]
    df_f = df_inter[
        df_inter['Acteur 1'].isin(top_actors) &
        df_inter['Acteur 2'].isin(top_actors)
    ]
    if df_f.empty:
        print("pas assez d'interactions pour le graphe")
        return False

    G = nx.Graph()
    for a in top_actors:
        G.add_node(a, size=mots_par_acteur.get(a, 100))

    for _, row in df_f.iterrows():
        G.add_edge(row['Acteur 1'], row['Acteur 2'],
                   weight=row['Poids Relatif Moyen'])

    fig, ax = plt.subplots(figsize=(14, 11), facecolor='#0f0f17')
    ax.set_facecolor('#0f0f17')
    pos = nx.kamada_kawai_layout(G, weight='weight')

    sizes = np.array([G.nodes[n]['size'] for n in G.nodes])
    sizes = 300 + (sizes / sizes.max()) * 2500

    nx.draw_networkx_nodes(G, pos, node_size=sizes,
                           node_color='#6366f1', edgecolors='#a5b4fc',
                           linewidths=1.5, alpha=0.9, ax=ax)

    edges   = G.edges(data=True)
    weights = np.array([e[2]['weight'] for e in edges])
    widths  = 1 + (weights / weights.max()) * 8 if len(weights) else []

    nx.draw_networkx_edges(G, pos, width=widths,
                           edge_color='#cbd5e1', alpha=0.25, ax=ax)

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        pct = round(d['weight'] * 100, 2)
        edge_labels[(u, v)] = f"{pct}%"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=7, font_color='#94a3b8',
                                  font_family='sans-serif', ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=11, font_color='#f1f5f9',
                            font_weight='bold', font_family='sans-serif', ax=ax)

    titre = f"Réseau d'Interactions — {titre_saison}\n" \
            f"Épaisseur = intensité (scènes communes / répliques) — Taille = volume de parole"
    ax.set_title(titre, color='#f1f5f9', fontsize=13, fontweight='bold', pad=18)
    ax.axis('off')
    fig.tight_layout()

    if output_png:
        os.makedirs(os.path.dirname(output_png) or '.', exist_ok=True)
        fig.savefig(output_png, dpi=200, facecolor='#0f0f17', edgecolor='none')
        print(f"graphe sauvegardé dans '{output_png}'")
    
    plt.close(fig)
    return True


def obtenirTopInterlocuteurs(df, acteur_cible, top_n=3):
    """
    Calcule les interactions pour un dataframe donné et renvoie le top N 
    des interlocuteurs pour un acteur précis (avec pourcentage et sujets).
    """
    tous_acteurs = df['acteur'].dropna().unique().tolist()
    if acteur_cible not in tous_acteurs:
        return []
        
    entetes = {}
    presenceParScene = construirePresenceParScene(df, entetes, tous_acteurs)
    
    liaisons, textes_paires = calculerLiaisons(df, presenceParScene)
    
    episodes_par_acteur  = defaultdict(set)
    repliques_par_acteur = defaultdict(int)
    for _, row in df.iterrows():
        episodes_par_acteur[row['acteur']].add((row['saison'], row['episode']))
        repliques_par_acteur[row['acteur']] += 1

    resultats = []
    for paire, nb_scenes_communes in liaisons.items():
        actA, actB = paire
        if actA != acteur_cible and actB != acteur_cible:
            continue
            
        eps_communs = episodes_par_acteur[actA].intersection(episodes_par_acteur[actB])
        nb_eps = len(eps_communs)
        if nb_eps == 0:
            continue
            
        tot_repliques = repliques_par_acteur[actA] + repliques_par_acteur[actB]
        poids_relatif = nb_scenes_communes / tot_repliques if tot_repliques > 0 else 0

        resultats.append({
            'Acteur 1':            actA,
            'Acteur 2':            actB,
            'Poids Relatif Moyen': round(poids_relatif, 6),
            '_textes':             " ".join(textes_paires[paire]),
        })
        
    df_act = pd.DataFrame(resultats)
    if df_act.empty:
        return []
        
    df_act = df_act.sort_values(by='Poids Relatif Moyen', ascending=False).head(top_n)
    
    corpus = df_act['_textes'].tolist()
    sujets = []
    for txt in corpus:
        tokens = ut.tokeniserTexte(ut.nettoyerTexte(txt))
        counts = Counter(tokens)
        top_mots = [w for w, _ in counts.most_common(5)]
        sujets.append(", ".join(top_mots) if top_mots else "aucun sujet")
        
    df_act['Sujets'] = sujets
    
    resultats_finaux = []
    for _, row in df_act.iterrows():
        other = row['Acteur 2'] if row['Acteur 1'] == acteur_cible else row['Acteur 1']
        pct = f"{round(row['Poids Relatif Moyen'] * 100, 2)}%"
        sujets = row.get('Sujets', '')
        resultats_finaux.append({
            'acteur': other,
            'pct': pct,
            'sujets': sujets
        })
    return resultats_finaux


def grapheEpisode(df_episode, dossier_sortie, numSaison, numEpisode):
    """Génère l'analyse réseau et le graphe pour un épisode."""
    print(f"\nGÉNÉRATION GRAPHE S{str(numSaison).zfill(2)}E{str(numEpisode).zfill(2)}")
    df_inter, acteurs, metrics = analyserReseau(df_episode, None) # chemin_dossier peut être None si on se base sur les dialogues uniquement
    
    if df_inter.empty:
        return False
        
    os.makedirs(dossier_sortie, exist_ok=True)
    csv_path = os.path.join(dossier_sortie, f"interactions_S{str(numSaison).zfill(2)}E{str(numEpisode).zfill(2)}.csv")
    df_inter.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"csv → {csv_path}")

    png_path = os.path.join(dossier_sortie, f"graphe_S{str(numSaison).zfill(2)}E{str(numEpisode).zfill(2)}.png")
    afficherGraphe(df_inter, metrics, output_png=png_path, titre_saison=f"Saison {numSaison} Episode {numEpisode}")
    
    return True


def grapheSaison(df_saison, dossier_sortie, numSaison):
    """Génère l'analyse réseau et le graphe pour une saison."""
    print(f"\nGÉNÉRATION GRAPHE SAISON {numSaison}")
    df_inter, acteurs, metrics = analyserReseau(df_saison, None)
    
    if df_inter.empty:
        return False
        
    os.makedirs(dossier_sortie, exist_ok=True)
    csv_path = os.path.join(dossier_sortie, f"interactions_S{str(numSaison).zfill(2)}.csv")
    df_inter.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"csv → {csv_path}")

    png_path = os.path.join(dossier_sortie, f"graphe_S{str(numSaison).zfill(2)}.png")
    afficherGraphe(df_inter, metrics, output_png=png_path, titre_saison=f"Saison {numSaison}")
    
    return True


def grapheAll(df_total, dossier_sortie):
    """Génère l'analyse réseau et le graphe pour toutes les saisons."""
    print(f"\nGÉNÉRATION GRAPHE GLOBAL")
    df_inter, acteurs, metrics = analyserReseau(df_total, None)
    
    if df_inter.empty:
        return False
        
    os.makedirs(dossier_sortie, exist_ok=True)
    csv_path = os.path.join(dossier_sortie, "interactions_all.csv")
    df_inter.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"csv → {csv_path}")

    png_path = os.path.join(dossier_sortie, "graphe_all.png")
    afficherGraphe(df_inter, metrics, output_png=png_path, limit_nodes=20, titre_saison="Global (Toutes saisons)")
    
    return True
