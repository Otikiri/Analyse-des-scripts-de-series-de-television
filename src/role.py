import networkx as nx
import pandas as pd

def calculer_importance_df(df):
    """Calcule l'importance à partir d'un DataFrame déjà chargé en mémoire."""
    results = []
    
    # Création d'une colonne temporaire 'nom fichier' si elle n'existe pas
    if 'nom fichier' not in df.columns:
        df['nom fichier'] = "S" + df['saison'].astype(str) + "E" + df['episode'].astype(str)

    grouped = df.groupby(['saison', 'episode'])

    for (saison, episode), ep_df in grouped:
        ep_name = ep_df["nom fichier"].iloc[0] if "nom fichier" in ep_df.columns else f"S{saison}E{episode}"

        v_brut = ep_df.groupby("acteur")["nombres de mots"].sum().to_dict()
        p_brut = ep_df.groupby("acteur")["scene_num"].nunique().to_dict()

        graphe = nx.Graph()
        scenes = ep_df.groupby("scene_num")["acteur"].unique().to_dict()
        for s_num, acteurs in scenes.items():
            liste_acteurs = list(set(acteurs))
            for i in range(len(liste_acteurs)):
                p1 = liste_acteurs[i]
                if not graphe.has_node(p1):
                    graphe.add_node(p1)
                for j in range(i + 1, len(liste_acteurs)):
                    p2 = liste_acteurs[j]
                    if graphe.has_edge(p1, p2):
                        graphe[p1][p2]["weight"] += 1
                    else:
                        graphe.add_edge(p1, p2, weight=1)

        try:
            centralite = nx.pagerank(graphe, weight="weight") if len(graphe) > 0 else {}
        except:
            centralite = nx.degree_centrality(graphe)

        max_v = max(v_brut.values()) if v_brut else 1
        max_p = max(p_brut.values()) if p_brut else 1
        max_c = max(centralite.values()) if centralite else 1

        for acteur in v_brut:
            v_norm = v_brut[acteur] / max_v if max_v > 0 else 0
            p_norm = p_brut[acteur] / max_p if max_p > 0 else 0
            c_norm = centralite.get(acteur, 0) / max_c if max_c > 0 else 0

            score_si = (0.3 * v_norm) + (0.3 * p_norm) + (0.4 * c_norm)

            if score_si > 0.66:
                statut = "Principal"
            elif score_si > 0.33:
                statut = "Secondaire"
            else:
                statut = "Tertiaire"

            results.append({
                "Episode_File": ep_name,
                "Saison": saison,
                "Episode": episode,
                "Acteur": acteur,
                "Statut Rôle": statut,
            })

    return pd.DataFrame(results)


if __name__ == '__main__':
    def calculer_importance_all_saisons(chemin_csv):
        df = pd.read_csv(chemin_csv)
        return calculer_importance_df(df)

    def obtenir_ep_principaux(df_scores, personnage):
        filtre = (df_scores["Acteur"].str.lower() == personnage.lower()) & (df_scores["Statut Rôle"] == "Principal")
        df_trie = df_scores[filtre].sort_values(by=["Saison", "Episode"])
        liste_ep = df_trie["Episode_File"].tolist()
        return len(liste_ep), liste_ep 

    scores_df = calculer_importance_all_saisons("all_saisons.csv")
    personnage = input("Nom du personnage : ")
    nb_ep, lst_ep = obtenir_ep_principaux(scores_df, personnage)
    print(f"{personnage} est un personnage principal dans {nb_ep} épisodes")