import stats_lexical as sl
import data_loader as dl
from gensim import models
from gensim import corpora

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

# sl.retrouverAlphaLDA(df,16)
# print(df['token'].head(10))
# topic_range=range(12,17)
# sl.retrouverNTopics(df,topic_range)


df = df[df['token'].apply(len)>0]
textes = df['token'].tolist()

dictionnaire = corpora.Dictionary(textes)
corpus = [dictionnaire.doc2bow(texte) for texte in textes]
model = models.LdaModel(
    corpus=corpus,
    num_topics=14,
    id2word=dictionnaire,
    passes=50,
    alpha='auto',
    eta='auto',
    random_state=42
)

for i, topic in model.show_topics(num_topics=16, num_words=8, formatted=False):
    words = [w for w, p in topic]
    print(f"Topic {i+1}: {words}")