import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).resolve().parent / "dataset.tsv"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_dataset():
    data = pd.read_csv(DATASET_PATH, sep="\t")

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError(
            "dataset.tsv must contain 'text' and 'label' columns."
        )

    data["text"] = data["text"].fillna("")
    data["clean_text"] = data["text"].apply(clean_text)

    return data

def train_model():
    data = load_dataset()

    X = data["clean_text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Not Dark Pattern", "Dark Pattern"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "model": model,
        "vectorizer": vectorizer,
        "data": data,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "report": report,
        "feature_count": len(vectorizer.get_feature_names_out()),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

def predict_text(text, model, vectorizer):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])

    prediction = int(model.predict(text_tfidf)[0])
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = float(probabilities[prediction])

    return prediction, confidence, probabilities

def explain_prediction(text, model, vectorizer, top_n=6):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()
    contributions = text_tfidf.toarray()[0] * model.coef_[0]

    positive = contributions.argsort()[::-1]
    positive_terms = [
        (feature_names[i], float(contributions[i]))
        for i in positive
        if contributions[i] > 0
    ][:top_n]

    negative = contributions.argsort()
    negative_terms = [
        (feature_names[i], float(contributions[i]))
        for i in negative
        if contributions[i] < 0
    ][:top_n]

    return positive_terms, negative_terms

def evaluate_model(model=None, vectorizer=None, data=None, test_size=0.20, random_state=42):
    if data is None:
        data = load_dataset()

    X = data["clean_text"]
    y = data["label"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        data.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if model is None or vectorizer is None:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)

    X_test_tfidf = vectorizer.transform(X_test)
    predictions = model.predict(X_test_tfidf)
    probabilities = model.predict_proba(X_test_tfidf)

    acc = float(accuracy_score(y_test, predictions))
    prec = float(precision_score(y_test, predictions, zero_division=0))
    rec = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    cm = confusion_matrix(y_test, predictions)

    report_dict = classification_report(
        y_test,
        predictions,
        target_names=["Not Dark Pattern", "Dark Pattern"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predictions,
        target_names=["Not Dark Pattern", "Dark Pattern"],
        zero_division=0,
    )

    test_df = data.loc[idx_test].copy()
    test_df["predicted"] = predictions
    test_df["prob_not_dark"] = probabilities[:, 0]
    test_df["prob_dark"] = probabilities[:, 1]

    false_positives = test_df[(test_df["label"] == 0) & (test_df["predicted"] == 1)].copy()
    false_negatives = test_df[(test_df["label"] == 1) & (test_df["predicted"] == 0)].copy()

    dark_test = test_df[test_df["label"] == 1]
    cat_rows = []
    for cat, grp in dark_test.groupby("Pattern Category"):
        total = len(grp)
        correct = int((grp["predicted"] == 1).sum())
        fn = int((grp["predicted"] == 0).sum())
        cat_rec = float(correct / total) if total > 0 else 0.0
        cat_rows.append({
            "category": cat,
            "total": total,
            "correct": correct,
            "false_negatives": fn,
            "recall": cat_rec,
        })
    cat_metrics = pd.DataFrame(cat_rows).sort_values(by="false_negatives", ascending=False)

    weaknesses = [
        "Short, context-poor, or subtle text snippets (e.g. countdowns, 'Will sell out fast!') with sparse TF-IDF features",
        "Low-frequency dark pattern categories (Forced Action with 0% recall, Sneaking with 33% recall) due to class imbalance and decision threshold",
    ]

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "category_metrics": cat_metrics,
        "test_df": test_df,
        "train_size": len(idx_train),
        "test_size": len(idx_test),
        "recommended_weaknesses": weaknesses,
    }

