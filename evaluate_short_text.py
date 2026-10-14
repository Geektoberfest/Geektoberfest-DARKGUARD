"""
Evaluation script comparing baseline vs enhanced models on short and subtle text.
Addresses Issue #2: [DARKGUARD] Improve short and subtle text detection.
"""

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from model import load_dataset, build_vectorizer, clean_text
from sklearn.linear_model import LogisticRegression

def evaluate_models(eval_set_path="tests/short_subtle_eval_set.json", output_path="SHORT_TEXT_IMPROVEMENT_REPORT.md"):
    data = load_dataset()
    X = data["clean_text"]
    y = data["label"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, data.index,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 1. Baseline Model
    v_base = build_vectorizer(use_enhanced=False)
    X_train_b = v_base.fit_transform(X_train)
    X_test_b = v_base.transform(X_test)
    m_base = LogisticRegression(max_iter=1000).fit(X_train_b, y_train)

    # 2. Enhanced Model
    v_enh = build_vectorizer(use_enhanced=True)
    X_train_e = v_enh.fit_transform(X_train)
    X_test_e = v_enh.transform(X_test)
    m_enh = LogisticRegression(max_iter=1000, C=1.5).fit(X_train_e, y_train)

    # Evaluation subsets
    # A. Curated Benchmark Set
    with open(eval_set_path, "r", encoding="utf-8") as f:
        curated_data = json.load(f)
    curated_texts = [clean_text(item["text"]) for item in curated_data]
    curated_labels = [item["label"] for item in curated_data]

    curated_pred_base = m_base.predict(v_base.transform(curated_texts))
    curated_pred_enh = m_enh.predict(v_enh.transform(curated_texts))

    # B. Test Set Short Texts (<= 4 words)
    test_df = data.loc[idx_test].copy()
    test_df["word_count"] = test_df["clean_text"].apply(lambda x: len(x.split()))
    short_test_df = test_df[test_df["word_count"] <= 4]
    short_idx = short_test_df.index
    short_y = short_test_df["label"]

    short_pred_base = m_base.predict(v_base.transform(short_test_df["clean_text"]))
    short_pred_enh = m_enh.predict(v_enh.transform(short_test_df["clean_text"]))

    # C. Full Held-Out Test Set
    full_pred_base = m_base.predict(X_test_b)
    full_pred_enh = m_enh.predict(X_test_e)

    def calc_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "fn": int(fn),
            "fp": int(fp),
            "cm": cm,
        }

    res_curated_b = calc_metrics(curated_labels, curated_pred_base)
    res_curated_e = calc_metrics(curated_labels, curated_pred_enh)

    res_short_b = calc_metrics(short_y, short_pred_base)
    res_short_e = calc_metrics(short_y, short_pred_enh)

    res_full_b = calc_metrics(y_test, full_pred_base)
    res_full_e = calc_metrics(y_test, full_pred_enh)

    print("=" * 80)
    print("🛡️ DARKGUARD: SHORT & SUBTLE TEXT DETECTION BENCHMARK REPORT")
    print("=" * 80)
    print(f"\n1. Curated Short & Subtle Benchmark (n = {len(curated_data)}):")
    print(f"   Baseline: Acc={res_curated_b['accuracy']:.2%}, Prec={res_curated_b['precision']:.4f}, Rec={res_curated_b['recall']:.4f}, F1={res_curated_b['f1']:.4f}, FN={res_curated_b['fn']}, FP={res_curated_b['fp']}")
    print(f"   Enhanced: Acc={res_curated_e['accuracy']:.2%}, Prec={res_curated_e['precision']:.4f}, Rec={res_curated_e['recall']:.4f}, F1={res_curated_e['f1']:.4f}, FN={res_curated_e['fn']}, FP={res_curated_e['fp']}")

    print(f"\n2. Held-Out Test Set - Short Texts (<= 4 words, n = {len(short_test_df)}):")
    print(f"   Baseline: Acc={res_short_b['accuracy']:.2%}, Prec={res_short_b['precision']:.4f}, Rec={res_short_b['recall']:.4f}, F1={res_short_b['f1']:.4f}, FN={res_short_b['fn']}, FP={res_short_b['fp']}")
    print(f"   Enhanced: Acc={res_short_e['accuracy']:.2%}, Prec={res_short_e['precision']:.4f}, Rec={res_short_e['recall']:.4f}, F1={res_short_e['f1']:.4f}, FN={res_short_e['fn']}, FP={res_short_e['fp']}")

    print(f"\n3. Overall Held-Out Test Set (n = {len(y_test)}):")
    print(f"   Baseline: Acc={res_full_b['accuracy']:.2%}, Prec={res_full_b['precision']:.4f}, Rec={res_full_b['recall']:.4f}, F1={res_full_b['f1']:.4f}, FN={res_full_b['fn']}, FP={res_full_b['fp']}")
    print(f"   Enhanced: Acc={res_full_e['accuracy']:.2%}, Prec={res_full_e['precision']:.4f}, Rec={res_full_e['recall']:.4f}, F1={res_full_e['f1']:.4f}, FN={res_full_e['fn']}, FP={res_full_e['fp']}")
    print("=" * 80)

    # Generate Markdown Report
    md = []
    md.append("# 🛡️ Short and Subtle Text Detection: Improvement Report\n")
    md.append("## Objective\n")
    md.append("Address Issue #2 by improving detection of dark patterns in short, context-poor, and subtle website snippets without degrading overall performance on the held-out test set.\n")

    md.append("## Diagnostic Root Cause\n")
    md.append("1. **Stop-Word Erosion:** Scikit-learn's default `stop_words='english'` stripped crucial dark-pattern signal words (`only`, `now`, `few`, `will`, `out`, `all`, `last`). In short snippets such as 'Only 2 left', the primary scarcity marker 'only' was completely deleted.\n")
    md.append("2. **Sparse TF-IDF Representation:** With word n-grams alone, short snippets yield only 1 to 3 non-zero features. Out-of-vocabulary variations or punctuation boundaries fail to trigger trained weights.\n")
    md.append("3. **Solution:** We designed a hybrid `FeatureUnion` combining:\n")
    md.append("   - Word TF-IDF with curated stop-words (preserving urgency/scarcity signal words) and sublinear term frequency scaling.\n")
    md.append("   - Character boundary n-grams (`char_wb`, range 3-5, min_df=2) that capture morphology, abbreviations, numerical countdowns, and subwords.\n")
    md.append("   - Tuned inverse regularization parameter `C=1.5`.\n")

    md.append("## Benchmark Results\n")
    md.append("### 1. Curated Short & Subtle Evaluation Set (n = 30)\n")
    md.append("| Metric | Baseline Model | Enhanced Model | Delta |")
    md.append("|---|---|---|---|")
    md.append(f"| **Accuracy** | {res_curated_b['accuracy']:.2%} | {res_curated_e['accuracy']:.2%} | **+{res_curated_e['accuracy'] - res_curated_b['accuracy']:.2%}** |")
    md.append(f"| **Precision** | {res_curated_b['precision']:.4f} | {res_curated_e['precision']:.4f} | **+{res_curated_e['precision'] - res_curated_b['precision']:.4f}** |")
    md.append(f"| **Recall** | {res_curated_b['recall']:.4f} | {res_curated_e['recall']:.4f} | **+{res_curated_e['recall'] - res_curated_b['recall']:.4f}** |")
    md.append(f"| **F1-Score** | {res_curated_b['f1']:.4f} | {res_curated_e['f1']:.4f} | **+{res_curated_e['f1'] - res_curated_b['f1']:.4f}** |")
    md.append(f"| **False Negatives** | {res_curated_b['fn']} | {res_curated_e['fn']} | **-{res_curated_b['fn'] - res_curated_e['fn']}** |")
    md.append(f"| **False Positives** | {res_curated_b['fp']} | {res_curated_e['fp']} | **{res_curated_e['fp'] - res_curated_b['fp']:+d}** |\n")

    md.append("### 2. Held-Out Test Set - Short Texts (<= 4 words, n = 221)\n")
    md.append("| Metric | Baseline Model | Enhanced Model | Delta |")
    md.append("|---|---|---|---|")
    md.append(f"| **Accuracy** | {res_short_b['accuracy']:.2%} | {res_short_e['accuracy']:.2%} | **+{res_short_e['accuracy'] - res_short_b['accuracy']:.2%}** |")
    md.append(f"| **Precision** | {res_short_b['precision']:.4f} | {res_short_e['precision']:.4f} | **+{res_short_e['precision'] - res_short_b['precision']:.4f}** |")
    md.append(f"| **Recall** | {res_short_b['recall']:.4f} | {res_short_e['recall']:.4f} | **+{res_short_e['recall'] - res_short_b['recall']:.4f}** |")
    md.append(f"| **F1-Score** | {res_short_b['f1']:.4f} | {res_short_e['f1']:.4f} | **+{res_short_e['f1'] - res_short_b['f1']:.4f}** |")
    md.append(f"| **False Negatives** | {res_short_b['fn']} | {res_short_e['fn']} | **-{res_short_b['fn'] - res_short_e['fn']}** |")
    md.append(f"| **False Positives** | {res_short_b['fp']} | {res_short_e['fp']} | **{res_short_e['fp'] - res_short_b['fp']:+d}** |\n")

    md.append("### 3. Overall Held-Out Test Set (n = 472)\n")
    md.append("| Metric | Baseline Model | Enhanced Model | Delta |")
    md.append("|---|---|---|---|")
    md.append(f"| **Accuracy** | {res_full_b['accuracy']:.2%} | {res_full_e['accuracy']:.2%} | **+{res_full_e['accuracy'] - res_full_b['accuracy']:.2%}** |")
    md.append(f"| **Precision** | {res_full_b['precision']:.4f} | {res_full_e['precision']:.4f} | **+{res_full_e['precision'] - res_full_b['precision']:.4f}** |")
    md.append(f"| **Recall** | {res_full_b['recall']:.4f} | {res_full_e['recall']:.4f} | **+{res_full_e['recall'] - res_full_b['recall']:.4f}** |")
    md.append(f"| **F1-Score** | {res_full_b['f1']:.4f} | {res_full_e['f1']:.4f} | **+{res_full_e['f1'] - res_full_b['f1']:.4f}** |")
    md.append(f"| **False Negatives** | {res_full_b['fn']} | {res_full_e['fn']} | **-{res_full_b['fn'] - res_full_e['fn']}** |")
    md.append(f"| **False Positives** | {res_full_b['fp']} | {res_full_e['fp']} | **{res_full_e['fp'] - res_full_b['fp']:+d}** |\n")

    md.append("## Example Level Predictions on Curated Benchmark\n")
    md.append("| Snippet | Label | Baseline Pred | Enhanced Pred | Status |")
    md.append("|---|---|---|---|---|")
    for item, pb, pe in zip(curated_data, curated_pred_base, curated_pred_enh):
        lbl = "Dark Pattern (1)" if item["label"] == 1 else "Not Dark (0)"
        p_b_str = "Dark (1)" if pb == 1 else "Not Dark (0)"
        p_e_str = "Dark (1)" if pe == 1 else "Not Dark (0)"
        if pe == item["label"] and pb != item["label"]:
            status = "✅ Fixed by Enhanced"
        elif pe == item["label"]:
            status = "✓ Correct"
        else:
            status = "⚠️ Error remaining"
        md.append(f"| `{item['text']}` | {lbl} | {p_b_str} | {p_e_str} | {status} |")
    md.append("")

    report_str = "\n".join(md)
    Path(output_path).write_text(report_str, encoding="utf-8")
    print(f"\nSaved report to {output_path}")

if __name__ == "__main__":
    evaluate_models()
