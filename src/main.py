import os
import sys
import pandas as pd
import data_loader as dl
import fiches_acteurs as fa

def analyserToutesLesSaisons(base_dir="datasets", out_dir="fichesActeurs"):
    """
    Parcourt toutes les saisons disponibles, génère les fiches et graphes
    pour chaque saison, puis fait de même pour la série complète.
    """
    if not os.path.isdir(base_dir) and os.path.isdir("../datasets"):
        base_dir = "../datasets"
        out_dir = "../fichesActeurs"

    saisons = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("S")
    ])
    
    if not saisons:
        print(f"Aucune saison trouvée dans '{base_dir}'")
        return
        
    dfs = []
    
    # 1. Traitement saison par saison
    for saison in saisons:
        print(f"\n{'='*50}")
        print(f"  TRAITEMENT DE LA SAISON {saison}")
        print(f"{'='*50}")
        
        chemin = os.path.join(base_dir, saison)
        try:
            df = dl.chargerDonnees(chemin)
            dfs.append(df)
            
            dossier_sortie = os.path.join(out_dir, saison)
            # L'appel à genererFichesActeurs va créer le dossier SXX,
            # générer le CSV pour chaque acteur, et générer les graphes associés
            fa.genererFichesActeurs(df, outputDir=dossier_sortie)
            
        except Exception as e:
            print(f"Erreur lors du traitement de la saison {saison}: {e}")
            import traceback
            traceback.print_exc()
            
    # 2. Traitement global (toutes les saisons combinées)
    if dfs:
        print(f"\n{'='*50}")
        print(f"  TRAITEMENT DE TOUTE LA SÉRIE COMBINÉE")
        print(f"{'='*50}")
        
        df_total = pd.concat(dfs, ignore_index=True)
        dossier_sortie = os.path.join(out_dir, "toutes_saisons")
        
        try:
            fa.genererFichesActeurs(df_total, outputDir=dossier_sortie)
        except Exception as e:
            print(f"Erreur lors du traitement global: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\n✅ TERMINÉ ! Tous les dossiers ont été générés dans '{out_dir}'.")

if __name__ == "__main__":
    analyserToutesLesSaisons()