import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'<[^>]*>', '', re.sub(r'\([^)]*\)', '', text)).strip()


# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("sentiment_analysis.csv")

# nettoyage sécurisé
df["text"] = df["text"].apply(clean_text)

# suppression des valeurs manquantes
df = df.dropna(subset=["text", "sentiment"])


# =========================
# SPLIT TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)


# =========================
# TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =========================
# TRAIN MODEL
# =========================
model = ComplementNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.metrics import f1_score
print("F1-macro:", f1_score(y_test, y_pred, average="macro"))
print(confusion_matrix(y_test, y_pred))


# =========================
# FRIENDS DATASET
# =========================
df_friends = pd.read_csv("../results/df.csv")

# =========================
# CLEAN FRIENDS TEXT (IMPORTANT)
# =========================
df_friends["texte nettoyer"] = df_friends["texte nettoyer"].fillna("")

df_friends["texte nettoyer"] = df_friends["texte nettoyer"].apply(clean_text)


def safe_predict(text):

    if not isinstance(text, str):
        return "neutral"

    text = text.strip()

    if text == "":
        return "neutral"

    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)

    return prediction[0]

df_friends["sentiment_pred"] = df_friends["texte nettoyer"].apply(safe_predict)


# =========================
# GROUP BY ACTOR
# =========================
df_grouped = df_friends.groupby("acteur")

# Exemple d'analyse utile
pivot = df_friends.pivot_table(
    index="acteur",
    columns="sentiment_pred",
    aggfunc="size",
    fill_value=0
)
pivot.to_csv("stats_sentiments_par_acteur.csv")

print(pivot)
acteurs_principaux = ["Ross", "Rachel", "Monica", "Chandler", "Joey", "Phoebe"]

df_main = df_friends[df_friends["acteur"].isin(acteurs_principaux)]
pivot_main = df_main.pivot_table(
    index="acteur",
    columns="sentiment_pred",
    aggfunc="size",
    fill_value=0
)

print(pivot_main)
pivot_main_percent = pivot_main.div(pivot_main.sum(axis=1), axis=0)

print(pivot_main_percent)


print(df["sentiment"].value_counts())

print(df["sentiment"].value_counts(normalize=True))

positive_examples = df_friends[
    df_friends["sentiment_pred"] == "positive"
][["acteur", "texte nettoyer"]]

print(positive_examples.sample(20))

negative_examples = df_friends[
    df_friends["sentiment_pred"] == "negative"
][["acteur", "texte nettoyer"]]

print(negative_examples.sample(20))

neutral_examples = df_friends[
    df_friends["sentiment_pred"] == "neutral"
][["acteur", "texte nettoyer"]]

print(neutral_examples.sample(20))

positive_examples.to_csv("positive_examples.csv", index=False)
negative_examples.to_csv("negative_examples.csv", index=False)
neutral_examples.to_csv("neutral_examples.csv", index=False)