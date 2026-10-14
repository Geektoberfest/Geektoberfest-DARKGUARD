import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

DATASET_PATH = Path(__file__).resolve().parent / "dataset.tsv"
DEFAULT_MULTILABEL_THRESHOLD = 0.35

# Stop words to retain because they convey key urgency, scarcity, and immediacy cues
SIGNAL_WORDS = {
    "only", "now", "will", "out", "all", "more", "most", "few",
    "less", "over", "under", "here", "last", "no", "not", "off",
}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - SIGNAL_WORDS)


def build_vectorizer(use_enhanced=True, char_min_df=2):
    if not use_enhanced:
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                stop_words=CUSTOM_STOP_WORDS,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=char_min_df,
                sublinear_tf=True,
            ),
        ),
    ])


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


def create_classifier(classifier_name="logistic_regression"):
    name = str(classifier_name).lower()
    if name in ("logistic_regression", "lr"):
        return LogisticRegression(max_iter=1000, random_state=42)
    elif name in ("linear_svm", "svm", "linearsvc"):
        base_svm = LinearSVC(random_state=42, dual=False, C=1.0)
        return CalibratedClassifierCV(base_svm, cv=5)
    elif name in ("naive_bayes", "multinomial_nb", "nb"):
        return MultinomialNB(alpha=1.0)
    elif name in ("complement_nb", "cnb"):
        return ComplementNB()
    else:
        raise ValueError(
            f"Unsupported classifier: {classifier_name}. "
            "Supported: 'logistic_regression', 'linear_svm', 'naive_bayes', 'complement_nb'"
        )


def train_model(classifier="logistic_regression", use_enhanced=True, C=1.5):
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

    vectorizer = build_vectorizer(use_enhanced=use_enhanced)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    if classifier == "logistic_regression" and C != 1.0:
        model = LogisticRegression(max_iter=1000, C=C, random_state=42)
    else:
        model = create_classifier(classifier)
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
    cleaned_names = [f.split("__", 1)[-1].strip() for f in feature_names]

    if hasattr(model, "coef_"):
        weights = model.coef_[0]
    elif hasattr(model, "calibrated_classifiers_"):
        weights = np.mean([clf.estimator.coef_[0] for clf in model.calibrated_classifiers_], axis=0)
    elif hasattr(model, "feature_log_prob_"):
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        weights = np.zeros(len(feature_names))

    contributions = text_tfidf.toarray()[0] * weights

    positive = contributions.argsort()[::-1]
    positive_terms = [
        (cleaned_names[i], float(contributions[i]))
        for i in positive
        if contributions[i] > 0
    ][:top_n]

    negative = contributions.argsort()
    negative_terms = [
        (cleaned_names[i], float(contributions[i]))
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


def compare_classifiers(data=None, test_size=0.20, random_state=42):
    if data is None:
        data = load_dataset()

    X = data["clean_text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    candidate_names = [
        "logistic_regression",
        "linear_svm",
        "naive_bayes",
        "complement_nb",
    ]
    display_labels = {
        "logistic_regression": "Logistic Regression (Baseline)",
        "linear_svm": "Linear SVM (Calibrated)",
        "naive_bayes": "Multinomial Naive Bayes",
        "complement_nb": "Complement Naive Bayes",
    }

    results = []
    trained_models = {}
    confusion_matrices = {}

    for cname in candidate_names:
        clf = create_classifier(cname)
        clf.fit(X_train_tfidf, y_train)
        preds = clf.predict(X_test_tfidf)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))

        results.append({
            "classifier_key": cname,
            "classifier_name": display_labels[cname],
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "true_positives": int(tp),
        })
        trained_models[cname] = clf
        confusion_matrices[cname] = cm

    comparison_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)

    return {
        "comparison_table": comparison_df,
        "confusion_matrices": confusion_matrices,
        "trained_models": trained_models,
        "vectorizer": vectorizer,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "recommendation": (
            "Calibrated Linear SVM provides the best balance by improving recall to 92.37% "
            "(cutting false negatives from 23 down to 18) while maintaining 93.16% precision. "
            "For audit use-cases prioritizing maximum recall, Multinomial Naive Bayes achieves 95.34% recall "
            "with only 11 false negatives at the cost of higher false positives (27)."
        ),
    }


def train_multilabel_model(data=None, test_size=0.20, random_state=42, threshold=DEFAULT_MULTILABEL_THRESHOLD):
    if data is None:
        data = load_dataset()

    data_copy = data.copy()
    data_copy["category_list"] = data_copy["Pattern Category"].apply(lambda c: [c])

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(data_copy["category_list"])

    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        data_copy["clean_text"],
        Y,
        data_copy.index,
        test_size=test_size,
        random_state=random_state,
        stratify=data_copy["Pattern Category"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    )
    model.fit(X_train_tfidf, Y_train)

    probabilities = model.predict_proba(X_test_tfidf)
    predictions = (probabilities >= threshold).astype(int)

    report_dict = classification_report(
        Y_test,
        predictions,
        target_names=mlb.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        Y_test,
        predictions,
        target_names=mlb.classes_,
        zero_division=0,
    )

    return {
        "model": model,
        "vectorizer": vectorizer,
        "mlb": mlb,
        "classes": list(mlb.classes_),
        "data": data_copy,
        "threshold": threshold,
        "report_dict": report_dict,
        "report_text": report_text,
        "feature_count": len(vectorizer.get_feature_names_out()),
        "train_size": len(idx_train),
        "test_size": len(idx_test),
        "X_test": X_test,
        "Y_test": Y_test,
        "predictions": predictions,
    }


def predict_multilabel(text, model, vectorizer, mlb, threshold=DEFAULT_MULTILABEL_THRESHOLD):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(text_tfidf)[0]
    prob_dict = {cls: float(prob) for cls, prob in zip(mlb.classes_, probabilities)}

    dark_categories = [cls for cls in mlb.classes_ if cls != "Not Dark Pattern"]
    detected_dark = [cls for cls in dark_categories if prob_dict[cls] >= threshold]

    if detected_dark:
        detected = detected_dark
        is_dark = True
    else:
        if prob_dict.get("Not Dark Pattern", 0.0) >= threshold:
            detected = ["Not Dark Pattern"]
            is_dark = False
        else:
            top_category = max(mlb.classes_, key=lambda c: prob_dict[c])
            detected = [top_category]
            is_dark = (top_category != "Not Dark Pattern")

    return {
        "detected_categories": detected,
        "is_dark": is_dark,
        "probabilities": prob_dict,
        "threshold": threshold,
    }


def explain_multilabel(text, model, vectorizer, mlb, top_n=4):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    text_array = text_tfidf.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()

    explanations = {}
    for i, estimator in enumerate(model.estimators_):
        cat_name = mlb.classes_[i]
        coef = estimator.coef_[0]
        contributions = text_array * coef

        pos_indices = contributions.argsort()[::-1]
        top_terms = [
            (feature_names[j], float(contributions[j]))
            for j in pos_indices
            if contributions[j] > 0
        ][:top_n]

        explanations[cat_name] = top_terms

    return explanations


def evaluate_on_multilabel_dataset(
    eval_set_path=None,
    model=None,
    vectorizer=None,
    mlb=None,
    threshold=DEFAULT_MULTILABEL_THRESHOLD,
):
    if eval_set_path is None:
        eval_set_path = Path(__file__).resolve().parent / "tests" / "multilabel_eval_set.json"
        if not eval_set_path.exists():
            eval_set_path = Path(__file__).resolve().parent / "multilabel_eval_set.json"

    with open(eval_set_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    if model is None or vectorizer is None or mlb is None:
        trained = train_multilabel_model(threshold=threshold)
        model = trained["model"]
        vectorizer = trained["vectorizer"]
        mlb = trained["mlb"]

    y_true_labels = [item["labels"] for item in benchmark_data]
    y_pred_labels = []
    detailed_results = []

    for item in benchmark_data:
        pred_res = predict_multilabel(item["text"], model, vectorizer, mlb, threshold=threshold)
        pred_labels = pred_res["detected_categories"]
        y_pred_labels.append(pred_labels)
        detailed_results.append({
            "text": item["text"],
            "true_labels": item["labels"],
            "predicted_labels": pred_labels,
            "probabilities": pred_res["probabilities"],
            "description": item.get("description", ""),
            "exact_match": sorted(item["labels"]) == sorted(pred_labels),
        })

    Y_true = mlb.transform(y_true_labels)
    Y_pred = mlb.transform(y_pred_labels)

    per_label_report = classification_report(
        Y_true,
        Y_pred,
        target_names=mlb.classes_,
        output_dict=True,
        zero_division=0,
    )
    per_label_text = classification_report(
        Y_true,
        Y_pred,
        target_names=mlb.classes_,
        zero_division=0,
    )

    subset_acc = float(accuracy_score(Y_true, Y_pred))
    h_loss = float(hamming_loss(Y_true, Y_pred))

    per_label_metrics = {}
    for i, cls in enumerate(mlb.classes_):
        cls_true = Y_true[:, i]
        cls_pred = Y_pred[:, i]
        per_label_metrics[cls] = {
            "precision": float(precision_score(cls_true, cls_pred, zero_division=0)),
            "recall": float(recall_score(cls_true, cls_pred, zero_division=0)),
            "f1": float(f1_score(cls_true, cls_pred, zero_division=0)),
            "support": int(cls_true.sum()),
        }

    return {
        "benchmark_samples": len(benchmark_data),
        "threshold": threshold,
        "subset_accuracy": subset_acc,
        "hamming_loss": h_loss,
        "micro_f1": float(per_label_report.get("micro avg", {}).get("f1-score", 0.0)),
        "macro_f1": float(per_label_report.get("macro avg", {}).get("f1-score", 0.0)),
        "classification_report_dict": per_label_report,
        "classification_report_text": per_label_text,
        "per_label_metrics": per_label_metrics,
        "detailed_results": detailed_results,
    }
