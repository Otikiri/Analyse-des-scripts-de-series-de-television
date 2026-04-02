import stats_lexical as sl
import data_loader as dl

# Chargements de donnees
df = dl.chargerDonnees("/home/sv/Study/Analyse-des-scripts-de-series-de-television/datasets")
# print(df)
# df_ep = dl.prendreDonneeParEp(df)
# print(df_ep)
# df_ac = dl.prendreDonneeParActeur(df)
# df_ac.to_csv("/home/sv/Study/Analyse-des-scripts-de-series-de-television/results/df_ac",index=False)

# print("stats total:")
# print(sl.donnerStatsMots(df))
# print("\nfreq mots par conversation:")
# print(sl.freqMotsParConversation(df))
# print("\nfreq mots par acteur:")
# print(sl.freqMotsParActeur(df))
# print("\nfreq mots par episode:")
# print(sl.freqMotsParEpisode(df))
# print("\nfreq mots au cours du temps:")
# print(sl.freqMotsAuCoursDuTemps(df))

# sl.retrouverAlphaLDA(df,10)
# print(df['token'].head(10))
sl.retrouverNTopics(df)