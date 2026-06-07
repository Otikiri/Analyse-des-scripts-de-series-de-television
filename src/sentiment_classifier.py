import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Enlève les tags HTML et les textes entre parenthèses
    return re.sub(r'<[^>]*>', '', re.sub(r'\([^)]*\)', '', text)).strip()

def train_ml_classifier(dataset_path="datasets/sentiment_analysis.csv"):
    """
    Entraîne un classifieur Naive Bayes (ComplementNB) sur le dataset fourni.
    
    Méthodologie:
    1. Chargement du dataset Kaggle.
    2. Nettoyage du texte.
    3. Split Train (80%) / Test (20%).
    4. Vectorisation TF-IDF.
    5. Entraînement avec ComplementNB.
    6. Évaluation sur le set de test (Accuracy).
    """
    if not os.path.exists(dataset_path):
        print(f"⚠️ Dataset non trouvé: {dataset_path}")
        print("Veuillez le télécharger depuis https://www.kaggle.com/datasets/mdismielhossenabir/sentiment-analysis")
        print("et le placer dans ce chemin.")
        return None, None

    print(f"\n[ML] Chargement du dataset depuis {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text", "sentiment"])

    print("[ML] Séparation Train/Test (80% / 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    print("[ML] Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("[ML] Entraînement du modèle ComplementNB...")
    model = ComplementNB()
    model.fit(X_train_tfidf, y_train)

    print("[ML] Évaluation du modèle sur le set de test...")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"[ML] => Accuracy: {acc:.4f}")
    
    return vectorizer, model

def predict_sentiments_ml(df, vectorizer, model):
    """
    Ajoute les prédictions de sentiments du modèle ML au DataFrame.
    """
    print("[ML] Inférence sur les données FRIENDS...")
    
    # On va utiliser 'ligne' (ou 'texte nettoyer' si disponible)
    col_text = 'texte nettoyer' if 'texte nettoyer' in df.columns else 'ligne'
    
    # Copie locale pour nettoyer et vectoriser en batch
    textes = df[col_text].fillna("").apply(clean_text).tolist()
    
    X_tfidf = vectorizer.transform(textes)
    preds = model.predict(X_tfidf)
    
    # Mapping des prédictions (positive, negative, neutral) vers Français (Positif, Négatif, Neutre)
    mapping = {"positive": "Positif", "negative": "Négatif", "neutral": "Neutre"}
    df['sentiment_ML'] = [mapping.get(p.lower(), "Neutre") for p in preds]
    
    return df

def predict_sentiments_vader(df):
    """
    Ajoute les prédictions de sentiments VADER au DataFrame.
    """
    print("[VADER] Inférence sur les données FRIENDS...")
    analyzer = SentimentIntensityAnalyzer()
    
    def get_vader_label(text):
        if not isinstance(text, str):
            return "Neutre"
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            return 'Positif'
        elif score <= -0.05:
            return 'Négatif'
        else:
            return 'Neutre'
            
    df['sentiment_VADER'] = df['ligne'].apply(get_vader_label)
    
    return df

def calculer_statistiques_sentiments(df, groupby_cols=['acteur']):
    """
    Calcule les pourcentages de sentiments Positif/Négatif/Neutre par groupe.
    """
    stats = []
    grouped = df.groupby(groupby_cols)
    for name, group in grouped:
        total = len(group)
        if total == 0:
            continue
            
        dict_stat = {col: name[i] if isinstance(name, tuple) else name for i, col in enumerate(groupby_cols)}
        
        # Stats VADER
        if 'sentiment_VADER' in df.columns:
            v_counts = group['sentiment_VADER'].value_counts()
            dict_stat['vader_pct_Positif'] = round((v_counts.get('Positif', 0) / total) * 100, 2)
            dict_stat['vader_pct_Négatif'] = round((v_counts.get('Négatif', 0) / total) * 100, 2)
            dict_stat['vader_pct_Neutre'] = round((v_counts.get('Neutre', 0) / total) * 100, 2)
            
        # Stats ML
        if 'sentiment_ML' in df.columns:
            m_counts = group['sentiment_ML'].value_counts()
            dict_stat['ml_pct_Positif'] = round((m_counts.get('Positif', 0) / total) * 100, 2)
            dict_stat['ml_pct_Négatif'] = round((m_counts.get('Négatif', 0) / total) * 100, 2)
            dict_stat['ml_pct_Neutre'] = round((m_counts.get('Neutre', 0) / total) * 100, 2)
            
        stats.append(dict_stat)
        
    return pd.DataFrame(stats)
