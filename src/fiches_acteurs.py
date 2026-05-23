import os, sys, shutil
import pandas as pd
import numpy as np
import data_loader as dl
import cluster as cl

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
    stats = statsEp.merge(totalEp, on=['saison','episode']).merge(totalSaison, on='saison').merge(statsActeurSaison, on=['acteur','saison'])
    stats['pct_parole_episode'] = (stats['nb_mots_acteur_ep']/stats['nb_mots_total_ep'] * 100).round(2)
    stats['pct_parole_saison']  = (stats['nb_mots_acteur_saison']/stats['nb_mots_total_saison'] * 100).round(2)
    return stats


def genererFichesActeurs(df, outputDir="fiches_acteurs", method="tfidf", minLines=50, groupbyCols=None):
    if groupbyCols is None:
        groupbyCols = ['saison', 'episode']
    os.makedirs(outputDir, exist_ok=True)
    stats = _calcStats(df)
    acteurs = df['acteur'].value_counts()
    acteurs = acteurs[acteurs >= minLines].index.tolist()
    print(f"\n[INFO] {len(acteurs)} acteur(s) qualifié(s) (>= {minLines} répliques).")
    fiches = []
    for acteur in acteurs:
        if pd.isna(acteur) or not str(acteur).strip():
            continue
        dfActeur = df[df['acteur'] == acteur].copy()
        colsDispo = [c for c in groupbyCols if c in dfActeur.columns]
        nbDocs = dfActeur[colsDispo].drop_duplicates().shape[0]
        if nbDocs < 2:
            print(f"[SKIP] {acteur} ({nbDocs} document(s))")
            continue
        kMax = min(5, nbDocs - 1)
        if kMax < 2:
            print(f"[SKIP] {acteur} (clustering impossible)")
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
                lignes.append({
                    'acteur': acteur,
                    'saison': saisonRow,
                    'episode': episodeRow,
                    'scene': sceneRow,
                    'mots_cles': motsCles,
                    'sujet_serie (cluster)': row['nom_cluster'],
                    'nb_mots_acteur_ep': int(ligneStats['nb_mots_acteur_ep'].iloc[0]) if not ligneStats.empty else 0,
                    'pct_parole_episode_%': ligneStats['pct_parole_episode'].iloc[0] if not ligneStats.empty else np.nan,
                    'pct_parole_saison_%':  ligneStats['pct_parole_saison'].iloc[0]  if not ligneStats.empty else np.nan,
                })
            nomFichier = os.path.join(outputDir, f"fiche_{acteur.replace(' ','_').replace('/','_')}.csv")
            pd.DataFrame(lignes).to_csv(nomFichier, index=False, encoding='utf-8')
            print(f"[OK]  {nomFichier}")
            fiches.append(nomFichier)

        except Exception as erreur:
            print(f"{acteur} : {erreur}")
        finally:
            shutil.rmtree(tmpDir, ignore_errors=True)

    print(f"\n[DONE] {len(fiches)} fiche(s) dans '{outputDir}'.")
    return fiches

if __name__ == "__main__":
    BASE = "../datasets"
    while True:
        menu()
        choix = input("Choix : ").strip()
        if choix == "0":
            sys.exit(0)
        elif choix == "1":
            numSaison = dmdInt("Saison : ")
            if numSaison is None:
                print("Saison invalide."); continue
            acteur = dmdActeur()
            chemin = getChemin(BASE, numSaison)
            if not os.path.isdir(chemin):
                print(f"{chemin} introuvable."); continue
            df = dl.chargerDonnees(chemin)
            if acteur:
                df = df[df['acteur'].str.lower() == acteur.lower()].copy()
                if df.empty:
                    print(f"Acteur '{acteur}' introuvable."); continue
            print(f"[INFO] {len(df)} répliques chargées.")
            genererFichesActeurs(df, "fiches_acteurs", minLines=1 if acteur else 50)
        elif choix == "2":
            numSaison = dmdInt("Saison : ")
            if numSaison is None:
                print("Saison invalide."); continue
            numEpisode = dmdInt("Épisode : ")
            if numEpisode is None:
                print("Épisode invalide."); continue
            acteur = dmdActeur()
            chemin = getChemin(BASE, numSaison)
            if not os.path.isdir(chemin):
                print(f"{chemin} introuvable."); continue
            df = dl.chargerDonnees(chemin)
            epStr = str(numEpisode).zfill(2)
            df = df[df['episode'] == epStr].copy()
            if df.empty:
                print(f"Aucune donnée pour E{epStr}."); continue
            if acteur:
                df = df[df['acteur'].str.lower() == acteur.lower()].copy()
                if df.empty:
                    print(f"Acteur '{acteur}' introuvable dans cet épisode."); continue
            print(f"[INFO] {len(df)} répliques pour S{str(numSaison).zfill(2)}E{epStr}.")
            genererFichesActeurs(df, "fiches_acteurs", minLines=1, groupbyCols=['scene_num'])
        elif choix == "3":
            acteur = dmdActeur()
            saisonsDispos = sorted([
                d for d in os.listdir(BASE)
                if os.path.isdir(os.path.join(BASE, d)) and d.startswith("S")
            ])
            if not saisonsDispos:
                print(f"Aucune saison dans '{BASE}'."); continue
            print(f"[INFO] {len(saisonsDispos)} saison(s) : {', '.join(saisonsDispos)}")
            for dossier in saisonsDispos:
                print(f"\n--- {dossier} ---")
                try:
                    df = dl.chargerDonnees(os.path.join(BASE, dossier))
                    if acteur:
                        df = df[df['acteur'].str.lower() == acteur.lower()].copy()
                    genererFichesActeurs(df, "fiches_acteurs", minLines=1 if acteur else 50)
                except Exception as erreur:
                    print(f"{dossier} : {erreur}")
        else:
            print("Choix invalide.")
