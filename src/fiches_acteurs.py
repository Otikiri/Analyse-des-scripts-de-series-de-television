import os, sys, shutil
import pandas as pd
import numpy as np
import data_loader as dl
import cluster as cl
import graphe_acteurs as ga
import role 
import sentiment_classifier as sc 

def menu():
    print("\n" + "="*40)
    print("  FICHES ACTEURS")
    print("  1. Par saison   2. Par épisode   3. Tout   0. Quitter")
    print("="*40)

def dmdInt(label):
    valeur = int(input(label))
    return valeur

def getChemin(base, numSaison):
    return os.path.join(base, f"S{str(numSaison).zfill(2)}")

def dmdActeur():
    valeur = input("Nom de l'acteur (Entrée = tous) : ").strip()
    return valeur or None

def _calcStats(df):
    statsEp = df.groupby(['acteur','saison','episode'])['nombres de mots'].sum().reset_index().rename(columns={'nombres de mots':'nb_mots_acteur_ep'})
    totalEp = df.groupby(['saison','episode'])['nombres de mots'].sum().reset_index().rename(columns={'nombres de mots':'nb_mots_total_ep'})
    totalSaison = df.groupby('saison')['nombres de mots'].sum().reset_index().rename(columns={'nombres de mots':'nb_mots_total_saison'})
    statsActeurSaison = df.groupby(['acteur','saison'])['nombres de mots'].sum().reset_index().rename(columns={'nombres de mots':'nb_mots_acteur_saison'})
    
    repEp = df.groupby(['acteur','saison','episode']).size().reset_index(name='nb_rep_acteur_ep')
    totalRepSaison = df.groupby('saison').size().reset_index(name='nb_rep_total_saison')
    repActeurSaison = df.groupby(['acteur','saison']).size().reset_index(name='nb_rep_acteur_saison')

    totalRepGlobal = len(df)
    repActeurGlobal = df.groupby('acteur').size().reset_index(name='nb_rep_acteur_global')

    stats = statsEp.merge(totalEp, on=['saison','episode']).merge(totalSaison, on='saison').merge(statsActeurSaison, on=['acteur','saison'])
    stats = stats.merge(repEp, on=['acteur','saison','episode']).merge(totalRepSaison, on='saison').merge(repActeurSaison, on=['acteur','saison'])
    stats = stats.merge(repActeurGlobal, on='acteur')
    
    stats['pct_parole_episode'] = (stats['nb_mots_acteur_ep']/stats['nb_mots_total_ep'] * 100).round(2)
    stats['pct_parole_saison']  = (stats['nb_mots_acteur_saison']/stats['nb_mots_total_saison'] * 100).round(2)
    stats['pct_rep_saison']     = (stats['nb_rep_acteur_saison']/stats['nb_rep_total_saison'] * 100).round(2)
    stats['pct_rep_global']     = (stats['nb_rep_acteur_global']/totalRepGlobal * 100).round(2)
    return stats


def genererFichesActeurs(df, outputDir="fiches_acteurs", method="tfidf", minLines=50, groupbyCols=None):
    if groupbyCols is None:
        groupbyCols = ['saison', 'episode']
    os.makedirs(outputDir, exist_ok=True)
    
    stats = _calcStats(df)
    
    ## NOUVEAU : Calcul des rôles pour toute la série UNE SEULE FOIS pour éviter de ralentir le script
    print("Calcul de l'importance des rôles en cours...")
    df_roles = role.calculer_importance_df(df)
    
    print("Calcul des statistiques de sentiments en cours...")
    stats_sent_ep = sc.calculer_statistiques_sentiments(df, groupby_cols=['acteur', 'saison', 'episode'])
    stats_sent_saison = sc.calculer_statistiques_sentiments(df, groupby_cols=['acteur', 'saison'])
    stats_sent_global = sc.calculer_statistiques_sentiments(df, groupby_cols=['acteur'])
    
    acteurs = df['acteur'].value_counts()
    acteurs = acteurs[acteurs >= minLines].index.tolist()
    print(f"{len(acteurs)} acteur(s) qualifié(s) (>= {minLines} répliques)")
    fiches = []
    
    for acteur in acteurs:
        if pd.isna(acteur) or not str(acteur).strip():
            continue
            
        dfActeur = df[df['acteur'] == acteur].copy()
        colsDispo = [c for c in groupbyCols if c in dfActeur.columns]
        nbDocs = dfActeur[colsDispo].drop_duplicates().shape[0]
        if nbDocs < 2:
            print(f"skip {acteur} ({nbDocs} document(s))")
            continue
        kMax = min(5, nbDocs - 1)
        if kMax < 2:
            print(f"skip {acteur} (clustering impossible)")
            continue
        tmpDir = os.path.join(outputDir, f".tmp_{acteur.replace(' ','_').replace('/','_')}")
        try:
            km, docs, _, vectorizer, X = cl.clusteringTfidf(
                dfActeur, groupbyCols=colsDispo,
                resultsDir=tmpDir, kMin=2, kMax=kMax,
                useSvd=False, methodeNommage=method
            )
            termes = np.array(vectorizer.get_feature_names_out())
            statsActeur = stats[stats['acteur'] == acteur]
            lignes = []
            
            # 1. Boucle sur les épisodes
            for idx, row in docs.iterrows():
                saisonRow  = row.get('saison', df['saison'].iloc[0])
                episodeRow = row.get('episode', df['episode'].iloc[0])
                sceneRow   = row.get('scene_num', None)
                indices = X[idx].toarray()[0].argsort()[-5:][::-1]
                motsCles = ', '.join(termes[indices])
                ligneStats = statsActeur[
                    (statsActeur['saison'] == saisonRow) &
                    (statsActeur['episode'] == episodeRow)
                ]
                
                ## NOUVEAU : Récupération du statut du rôle pour cet acteur, cette saison et cet épisode
                role_info = df_roles[
                    (df_roles['Acteur'] == acteur) & 
                    (df_roles['Saison'] == saisonRow) & 
                    (df_roles['Episode'] == episodeRow)
                ]
                statut_role = role_info['Statut Rôle'].iloc[0] if not role_info.empty else "Non défini"
                
                # Sentiments
                sent_ep_info = stats_sent_ep[
                    (stats_sent_ep['acteur'] == acteur) & 
                    (stats_sent_ep['saison'] == saisonRow) & 
                    (stats_sent_ep['episode'] == episodeRow)
                ]
                
                df_doc = df.copy()
                for col in colsDispo:
                    if col in row and not pd.isna(row[col]):
                        df_doc = df_doc[df_doc[col] == row[col]]
                top_inter = ga.obtenirTopInterlocuteurs(df_doc, acteur, top_n=3)
                
                dossier_graphes = os.path.join(outputDir, "graphes")
                png_path = os.path.join(dossier_graphes, f"graphe_S{str(saisonRow).zfill(2)}E{str(episodeRow).zfill(2)}.png")
                if not os.path.exists(png_path):
                    ga.grapheEpisode(df_doc, dossier_graphes, str(saisonRow).zfill(2), str(episodeRow).zfill(2))
                
                ligne_dict = {
                    'acteur': acteur,
                    'saison': saisonRow,
                    'episode': episodeRow,
                    'scene': sceneRow,
                    ## NOUVEAU : Ajout de la colonne rôle
                    'statut_role': statut_role,
                    'mots_cles': motsCles,
                    'sujet_serie (cluster)': row['nom_cluster'],
                    'nb_mots_acteur_ep': int(ligneStats['nb_mots_acteur_ep'].iloc[0]) if not ligneStats.empty else 0,
                    'pct_parole_episode_%': ligneStats['pct_parole_episode'].iloc[0] if not ligneStats.empty else np.nan,
                    'pct_parole_saison_%':  ligneStats['pct_parole_saison'].iloc[0]  if not ligneStats.empty else np.nan,
                    'pct_repliques_saison_%': '',
                    'pct_repliques_global_%': '',
                    'vader_Positif_%': sent_ep_info['vader_pct_Positif'].iloc[0] if not sent_ep_info.empty and 'vader_pct_Positif' in sent_ep_info.columns else '',
                    'vader_Négatif_%': sent_ep_info['vader_pct_Négatif'].iloc[0] if not sent_ep_info.empty and 'vader_pct_Négatif' in sent_ep_info.columns else '',
                    'vader_Neutre_%': sent_ep_info['vader_pct_Neutre'].iloc[0] if not sent_ep_info.empty and 'vader_pct_Neutre' in sent_ep_info.columns else '',
                    'ml_Positif_%': sent_ep_info['ml_pct_Positif'].iloc[0] if not sent_ep_info.empty and 'ml_pct_Positif' in sent_ep_info.columns else '',
                    'ml_Négatif_%': sent_ep_info['ml_pct_Négatif'].iloc[0] if not sent_ep_info.empty and 'ml_pct_Négatif' in sent_ep_info.columns else '',
                    'ml_Neutre_%': sent_ep_info['ml_pct_Neutre'].iloc[0] if not sent_ep_info.empty and 'ml_pct_Neutre' in sent_ep_info.columns else '',
                    'chemin_graphe': f"../results/fiches_acteurs/graphes/graphe_S{str(saisonRow).zfill(2)}E{str(episodeRow).zfill(2)}.png"
                }
                for i in range(1, 4):
                    if i <= len(top_inter):
                        ligne_dict[f'topActeur{i}'] = top_inter[i-1]['acteur']
                        ligne_dict[f'pctLiaison{i}'] = top_inter[i-1]['pct']
                        ligne_dict[f'sujetsLiaison{i}'] = top_inter[i-1]['sujets']
                    else:
                        ligne_dict[f'topActeur{i}'] = ""
                        ligne_dict[f'pctLiaison{i}'] = ""
                        ligne_dict[f'sujetsLiaison{i}'] = ""
                lignes.append(ligne_dict)
            
            # 2. Boucle sur les bilans de saisons
            saisons_acteur = statsActeur['saison'].unique()
            for s in saisons_acteur:
                ligne_s = statsActeur[statsActeur['saison'] == s].iloc[0]
                df_saison = df[df['saison'] == s]
                top_inter_s = ga.obtenirTopInterlocuteurs(df_saison, acteur, top_n=3)
                
                dossier_graphes = os.path.join(outputDir, "graphes")
                png_path = os.path.join(dossier_graphes, f"graphe_S{str(s).zfill(2)}.png")
                if not os.path.exists(png_path):
                    ga.grapheSaison(df_saison, dossier_graphes, str(s).zfill(2))
                
                sent_saison_info = stats_sent_saison[
                    (stats_sent_saison['acteur'] == acteur) & 
                    (stats_sent_saison['saison'] == s)
                ]
                
                ligne_dict = {
                    'acteur': acteur,
                    'saison': s,
                    'episode': 'BILAN SAISON',
                    'scene': '',
                    ## NOUVEAU : Vide pour le bilan saison, car le rôle est calculé par épisode
                    'statut_role': '',
                    'mots_cles': '',
                    'sujet_serie (cluster)': '',
                    'nb_mots_acteur_ep': '',
                    'pct_parole_episode_%': '',
                    'pct_parole_saison_%': ligne_s['pct_parole_saison'],
                    'pct_repliques_saison_%': ligne_s['pct_rep_saison'],
                    'pct_repliques_global_%': '',
                    'vader_Positif_%': sent_saison_info['vader_pct_Positif'].iloc[0] if not sent_saison_info.empty and 'vader_pct_Positif' in sent_saison_info.columns else '',
                    'vader_Négatif_%': sent_saison_info['vader_pct_Négatif'].iloc[0] if not sent_saison_info.empty and 'vader_pct_Négatif' in sent_saison_info.columns else '',
                    'vader_Neutre_%': sent_saison_info['vader_pct_Neutre'].iloc[0] if not sent_saison_info.empty and 'vader_pct_Neutre' in sent_saison_info.columns else '',
                    'ml_Positif_%': sent_saison_info['ml_pct_Positif'].iloc[0] if not sent_saison_info.empty and 'ml_pct_Positif' in sent_saison_info.columns else '',
                    'ml_Négatif_%': sent_saison_info['ml_pct_Négatif'].iloc[0] if not sent_saison_info.empty and 'ml_pct_Négatif' in sent_saison_info.columns else '',
                    'ml_Neutre_%': sent_saison_info['ml_pct_Neutre'].iloc[0] if not sent_saison_info.empty and 'ml_pct_Neutre' in sent_saison_info.columns else '',
                    'chemin_graphe': f"../results/fiches_acteurs/graphes/graphe_S{str(s).zfill(2)}.png"
                }
                for i in range(1, 4):
                    if i <= len(top_inter_s):
                        ligne_dict[f'topActeur{i}'] = top_inter_s[i-1]['acteur']
                        ligne_dict[f'pctLiaison{i}'] = top_inter_s[i-1]['pct']
                        ligne_dict[f'sujetsLiaison{i}'] = top_inter_s[i-1]['sujets']
                    else:
                        ligne_dict[f'topActeur{i}'] = ""
                        ligne_dict[f'pctLiaison{i}'] = ""
                        ligne_dict[f'sujetsLiaison{i}'] = ""
                lignes.append(ligne_dict)
            
            # 3. Bilan global
            ligne_g = statsActeur.iloc[0]
            top_inter_g = ga.obtenirTopInterlocuteurs(df, acteur, top_n=3)
            
            dossier_graphes = os.path.join(outputDir, "graphes")
            png_path = os.path.join(dossier_graphes, "graphe_all.png")
            if not os.path.exists(png_path):
                ga.grapheAll(df, dossier_graphes)
            
            sent_global_info = stats_sent_global[
                (stats_sent_global['acteur'] == acteur)
            ]
            
            ligne_dict = {
                    'acteur': acteur,
                    'saison': 'TOUTES',
                    'episode': 'BILAN GLOBAL',
                    'scene': '',
                    ## NOUVEAU : Vide pour le bilan global
                    'statut_role': '',
                    'mots_cles': '',
                    'sujet_serie (cluster)': '',
                    'nb_mots_acteur_ep': '',
                    'pct_parole_episode_%': '',
                    'pct_parole_saison_%': '',
                    'pct_repliques_saison_%': '',
                    'pct_repliques_global_%': ligne_g['pct_rep_global'],
                    'vader_Positif_%': sent_global_info['vader_pct_Positif'].iloc[0] if not sent_global_info.empty and 'vader_pct_Positif' in sent_global_info.columns else '',
                    'vader_Négatif_%': sent_global_info['vader_pct_Négatif'].iloc[0] if not sent_global_info.empty and 'vader_pct_Négatif' in sent_global_info.columns else '',
                    'vader_Neutre_%': sent_global_info['vader_pct_Neutre'].iloc[0] if not sent_global_info.empty and 'vader_pct_Neutre' in sent_global_info.columns else '',
                    'ml_Positif_%': sent_global_info['ml_pct_Positif'].iloc[0] if not sent_global_info.empty and 'ml_pct_Positif' in sent_global_info.columns else '',
                    'ml_Négatif_%': sent_global_info['ml_pct_Négatif'].iloc[0] if not sent_global_info.empty and 'ml_pct_Négatif' in sent_global_info.columns else '',
                    'ml_Neutre_%': sent_global_info['ml_pct_Neutre'].iloc[0] if not sent_global_info.empty and 'ml_pct_Neutre' in sent_global_info.columns else '',
                    'chemin_graphe': "../results/fiches_acteurs/graphes/graphe_all.png"
            }
            for i in range(1, 4):
                if i <= len(top_inter_g):
                    ligne_dict[f'topActeur{i}'] = top_inter_g[i-1]['acteur']
                    ligne_dict[f'pctLiaison{i}'] = top_inter_g[i-1]['pct']
                    ligne_dict[f'sujetsLiaison{i}'] = top_inter_g[i-1]['sujets']
                else:
                    ligne_dict[f'topActeur{i}'] = ""
                    ligne_dict[f'pctLiaison{i}'] = ""
                    ligne_dict[f'sujetsLiaison{i}'] = ""
            lignes.append(ligne_dict)

            nomFichier = os.path.join(outputDir, f"fiche_{acteur.replace(' ','_').replace('/','_')}.csv")
            pd.DataFrame(lignes).to_csv(nomFichier, index=False, encoding='utf-8')
            print(f"fichier enregistré: {nomFichier}")
            fiches.append(nomFichier)

        except Exception as erreur:
            print(f"{acteur} : {erreur}")
        finally:
            shutil.rmtree(tmpDir, ignore_errors=True)

    print(f"{len(fiches)} fiche(s) dans '{outputDir}'")
    return fiches
