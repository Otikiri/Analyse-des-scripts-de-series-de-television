import stats_lexical as sl
import data_loader as dl
from gensim import models
from gensim import corpora

# Chargements de donnees
df = dl.chargerDonnees("/home/sv/Study/Analyse-des-scripts-de-series-de-television/datasets")
print(df)



# print(sl.trouverNbOptimalTopics(df,min_tokens_par_scene=20))


#Grouper les tokens par scene : documents plus longs = meilleure qualite LDA
textes = (
    df.groupby(['saison', 'episode', 'scene_num'])['token']
    .apply(lambda rows: [t for tokens in rows for t in tokens])
    .tolist()
)
# # On garde uniquement les scenes avec assez de tokens
textes = [t for t in textes if len(t) >= 20]
print(f"  Nombre de scenes utilisees : {len(textes)}")

    # Construire le dictionnaire gensim et filtrer les mots trop rares ou trop frequents
    # no_below=5 : mot doit apparaitre dans au moins 5 scenes
    # no_above=0.5 : mot ne doit pas apparaitre dans plus de 50% des scenes
dictionnaire = corpora.Dictionary(textes)
dictionnaire.filter_extremes(no_below=5, no_above=0.5)

# Construire le corpus : chaque scene devient une liste de (id_mot, frequence)
corpus = [dictionnaire.doc2bow(texte) for texte in textes]

# dictionnaire = corpora.Dictionary(textes)
# corpus = [dictionnaire.doc2bow(texte) for texte in textes]
model = models.LdaModel(
    corpus=corpus,
    num_topics=13,
    id2word=dictionnaire,
    passes=50,
    alpha=0.5,
    eta='auto',
    random_state=42
)

for i, topic in model.show_topics(num_topics=13, num_words=8, formatted=False):
    words = [w for w, p in topic]
    print(f"Topic {i+1}: {words}")


import pyLDAvis
import pyLDAvis.gensim_models

# 1. Prepare the visualization data
# Note: 'sort_topics=False' keeps the topic IDs consistent with your print statements
vis_data = pyLDAvis.gensim_models.prepare(
    model, 
    corpus, 
    dictionnaire, 
    sort_topics=False
)

# 2. Export to a standalone HTML file
pyLDAvis.save_html(vis_data, 'friends_lda_vis.html')

print("Visualization exported to 'friends_lda_vis.html'. Open this file in your browser.")