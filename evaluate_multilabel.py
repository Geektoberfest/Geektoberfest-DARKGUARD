"""
Multi-label detection, explainability, and benchmark evaluation script.
Addresses Issue #4: [DARKGUARD] Add multi-label detection and explainability.
"""

from pathlib import Path
import pandas as pd
from model import (
    DEFAULT_MULTILABEL_THRESHOLD,
    evaluate_on_multilabel_dataset,
    explain_multilabel,
    load_dataset,
    predict_multilabel,
    train_multilabel_model,
)

def run_multilabel_evaluation(output_path="MULTILABEL_EVALUATION_REPORT.md"):
    print("=" * 80)
    print("🛡️ DARKGUARD: MULTI-LABEL DETECTION & EXPLAINABILITY EVALUATION")
    print("=" * 80)
    print(f"Consistent Prediction & Evaluation Threshold: {DEFAULT_MULTILABEL_THRESHOLD}")

    res = train_multilabel_model(threshold=DEFAULT_MULTILABEL_THRESHOLD)
    rep = res["report_dict"]

    print(f"\n1. Held-Out Category-Level Evaluation (Test Size: {res['test_size']} samples, 80/20 Stratified Split):")
    print("-" * 80)
    print(f"{'Category':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    for cls in res["classes"]:
        m = rep[cls]
        print(f"{cls:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1-score']:<12.4f} {int(m['support']):<10}")
    print("-" * 80)
    print(f"{'Micro Avg':<20} {rep['micro avg']['precision']:<12.4f} {rep['micro avg']['recall']:<12.4f} {rep['micro avg']['f1-score']:<12.4f} {int(rep['micro avg']['support']):<10}")
    print(f"{'Macro Avg':<20} {rep['macro avg']['precision']:<12.4f} {rep['macro avg']['recall']:<12.4f} {rep['macro avg']['f1-score']:<12.4f} {int(rep['macro avg']['support']):<10}")
    print(f"{'Weighted Avg':<20} {rep['weighted avg']['precision']:<12.4f} {rep['weighted avg']['recall']:<12.4f} {rep['weighted avg']['f1-score']:<12.4f} {int(rep['weighted avg']['support']):<10}")
    print("=" * 80)

    # 2. Genuine Multi-Label Benchmark Evaluation
    bench_res = evaluate_on_multilabel_dataset(threshold=DEFAULT_MULTILABEL_THRESHOLD)
    bench_rep = bench_res["classification_report_dict"]

    print(f"\n2. Genuine Multi-Label Benchmark Evaluation ({bench_res['benchmark_samples']} samples, tests/multilabel_eval_set.json):")
    print("-" * 80)
    print(f"Subset Accuracy (Exact Match): {bench_res['subset_accuracy']:.4f}")
    print(f"Hamming Loss (Label Error Rate): {bench_res['hamming_loss']:.4f}")
    print(f"Micro-Averaged F1-Score:       {bench_res['micro_f1']:.4f}")
    print(f"Macro-Averaged F1-Score:       {bench_res['macro_f1']:.4f}")
    print("-" * 80)
    print(f"{'Category':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    for cls in res["classes"]:
        if cls in bench_rep:
            m = bench_rep[cls]
            print(f"{cls:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1-score']:<12.4f} {int(m['support']):<10}")
    print("=" * 80)

    demonstrations = [
        ("Only 2 items left in stock!", "Single-Label Scarcity"),
        ("Hurry! This flash deal expires in 10 minutes.", "Single-Label Urgency"),
        ("127 people are viewing this product right now!", "Single-Label Social Proof"),
        ("I WANT TO PAY FULL PRICE LATER ;)", "Single-Label Misdirection"),
        ("Hurry! Only 2 items left in stock. This flash sale ends in 10 minutes!", "Compound (Scarcity + Urgency)"),
        ("Only 2 items left in stock! 28 people are viewing this property right now.", "Compound (Scarcity + Social Proof)"),
        ("Hurry! Only 2 items left in stock. This offer expires in 5 minutes. 15 people are looking right now!", "Triple Compound (Scarcity + Urgency + Social Proof)"),
        ("This shirt is made of 100% organic cotton. Available in navy and white.", "Neutral Non-Dark Control"),
    ]

    print("\nDemonstrations of Multi-Label Detection & Category-Level Explanations:")
    demo_records = []
    for text, desc in demonstrations:
        pred_res = predict_multilabel(text, res["model"], res["vectorizer"], res["mlb"], threshold=DEFAULT_MULTILABEL_THRESHOLD)
        expl_res = explain_multilabel(text, res["model"], res["vectorizer"], res["mlb"], top_n=3)

        print(f"\nSnippet: \"{text}\"")
        print(f"Target / Description: {desc}")
        print(f"Detected Categories:  {pred_res['detected_categories']}")
        print("Active Category Features:")
        active_explanations = {}
        for cat in pred_res["detected_categories"]:
            if cat in expl_res and expl_res[cat]:
                terms = [f"{t} (+{w:.2f})" for t, w in expl_res[cat]]
                active_explanations[cat] = terms
                print(f"   [{cat}]: {', '.join(terms)}")
            elif cat == "Not Dark Pattern":
                print("   [Not Dark Pattern]: No deceptive features triggered.")

        demo_records.append({
            "text": text,
            "desc": desc,
            "detected": pred_res["detected_categories"],
            "probabilities": pred_res["probabilities"],
            "explanations": active_explanations,
        })

    # Generate Markdown Report
    md = []
    md.append("# 🛡️ DARKGUARD: Multi-Label Detection and Explainability Report\n")
    md.append("## Objective\n")
    md.append("Address Issue #4 by extending DARKGUARD from binary classification to category-level and multi-label detection with interpretable category-specific explanations, unified prediction thresholds, and evaluation on a genuine multi-label benchmark dataset.\n")

    md.append("## Consistent Prediction and Evaluation Threshold\n")
    md.append(f"A single, standardized threshold (`DEFAULT_MULTILABEL_THRESHOLD = {DEFAULT_MULTILABEL_THRESHOLD}`) is used across all components:")
    md.append(f"- Model training and held-out validation in `train_multilabel_model(threshold={DEFAULT_MULTILABEL_THRESHOLD})`")
    md.append(f"- Live single and batch inference in `predict_multilabel(..., threshold={DEFAULT_MULTILABEL_THRESHOLD})`")
    md.append(f"- Benchmark evaluation in `evaluate_on_multilabel_dataset(..., threshold={DEFAULT_MULTILABEL_THRESHOLD})`")
    md.append("This eliminates discrepancy between reporting metrics and runtime application behavior.\n")

    md.append("## Dataset Analysis & Ground-Truth Limitations\n")
    md.append("A thorough analysis of `dataset.tsv` reveals critical architectural constraints:")
    md.append("1. **Strictly Single-Label Annotations in Base Dataset:** Every sample in `dataset.tsv` (2,356 samples) is mapped to exactly one category string. Zero multi-label records exist in the original corpus.")
    md.append("2. **Label Distribution:**")
    md.append("   - `Not Dark Pattern`: 1,178 (50.0%)")
    md.append("   - `Scarcity`: 418 (17.7%)")
    md.append("   - `Social Proof`: 312 (13.2%)")
    md.append("   - `Urgency`: 210 (8.9%)")
    md.append("   - `Misdirection`: 195 (8.3%)")
    md.append("   - `Obstruction`: 27 (1.1%)")
    md.append("   - `Sneaking`: 12 (0.5%)")
    md.append("   - `Forced Action`: 4 (0.2%)")
    md.append("3. **Severe Minority Imbalance:** `Forced Action` (4 samples) and `Sneaking` (12 samples) possess insufficient support for standalone generalization.\n")

    md.append("## Multi-Label Architecture: One-Vs-Rest (OvR) Binary Relevance\n")
    md.append("To support true multi-label prediction where a webpage snippet exhibits multiple simultaneous dark patterns (e.g. Scarcity + Urgency), we implemented a `OneVsRestClassifier` with balanced Logistic Regression sub-models:")
    md.append("- Each category possesses its own calibrated decision hyperplane.")
    md.append(f"- Predictions are independent per category via sigmoid probabilities with threshold {DEFAULT_MULTILABEL_THRESHOLD}.")
    md.append("- Multiple categories can be simultaneously triggered when compound dark patterns are present.")
    md.append("- Category-specific explanations are generated by decomposing TF-IDF feature activations against each category's learned linear weight vector.\n")

    md.append("## Multi-Label Benchmark Evaluation (tests/multilabel_eval_set.json)\n")
    md.append("To thoroughly evaluate the model's ability to detect overlapping categories on realistic inputs, a dedicated multi-label benchmark dataset (`tests/multilabel_eval_set.json`, 26 curated samples) was constructed:")
    md.append(f"- **Benchmark Size:** {bench_res['benchmark_samples']} samples")
    md.append(f"- **Decision Threshold:** {DEFAULT_MULTILABEL_THRESHOLD}")
    md.append(f"- **Subset Accuracy (Exact Match):** {bench_res['subset_accuracy']:.2%}")
    md.append(f"- **Hamming Loss (Error Rate):** {bench_res['hamming_loss']:.4f}")
    md.append(f"- **Micro-Averaged F1-Score:** {bench_res['micro_f1']:.4f}")
    md.append(f"- **Macro-Averaged F1-Score:** {bench_res['macro_f1']:.4f}\n")

    md.append("### Per-Label Benchmark Metrics\n")
    md.append("| Category | Precision | Recall | F1-Score | Support | Status |")
    md.append("|---|---|---|---|---|---|")
    for cls in res["classes"]:
        if cls in bench_rep:
            m = bench_rep[cls]
            status = "Strong (>0.85 F1)" if m["f1-score"] >= 0.85 else ("Moderate (>0.65 F1)" if m["f1-score"] >= 0.65 else "Data-Constrained")
            md.append(f"| **{cls}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} | {status} |")
    md.append(f"| **Micro Average** | {bench_rep['micro avg']['precision']:.4f} | {bench_rep['micro avg']['recall']:.4f} | {bench_rep['micro avg']['f1-score']:.4f} | {int(bench_rep['micro avg']['support'])} | - |")
    md.append(f"| **Macro Average** | {bench_rep['macro avg']['precision']:.4f} | {bench_rep['macro avg']['recall']:.4f} | {bench_rep['macro avg']['f1-score']:.4f} | {int(bench_rep['macro avg']['support'])} | - |\n")

    md.append("## Held-Out Base Test Set Evaluation (80/20 Stratified Split)\n")
    md.append("| Dark Pattern Category | Test Support | Precision | Recall | F1-Score | Status |")
    md.append("|---|---|---|---|---|---|")
    for cls in res["classes"]:
        m = rep[cls]
        status = "Strong (>0.85 F1)" if m["f1-score"] >= 0.85 else ("Moderate (>0.70 F1)" if m["f1-score"] >= 0.70 else "Data-Constrained (<0.70 F1)")
        md.append(f"| **{cls}** | {int(m['support'])} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {status} |")
    md.append(f"| **Micro Average** | {int(rep['micro avg']['support'])} | {rep['micro avg']['precision']:.4f} | {rep['micro avg']['recall']:.4f} | {rep['micro avg']['f1-score']:.4f} | - |")
    md.append(f"| **Macro Average** | {int(rep['macro avg']['support'])} | {rep['macro avg']['precision']:.4f} | {rep['macro avg']['recall']:.4f} | {rep['macro avg']['f1-score']:.4f} | - |\n")

    md.append("## Compound & Multi-Label Demonstrations\n")
    for rec in demo_records:
        cats_str = ", ".join(f"`{c}`" for c in rec["detected"])
        md.append(f"### Example: \"{rec['text']}\"")
        md.append(f"- **Intent:** {rec['desc']}")
        md.append(f"- **Detected Categories:** {cats_str}")
        if rec["explanations"]:
            md.append("- **Category Feature Explanations:**")
            for cat, terms in rec["explanations"].items():
                md.append(f"  - **{cat}:** {', '.join(terms)}")
        md.append("")

    md.append("## Annotation Recommendations for True Multi-Label Extension\n")
    md.append("To move from binary relevance OvR to fully verified multi-label training:")
    md.append("1. **Multi-Annotator Protocol:** Collect realistic e-commerce banners that bundle countdown timers with stock counters and re-annotate with multi-select checkboxes.")
    md.append("2. **Category Hierarchy:** Group rare patterns (`Sneaking`, `Obstruction`, `Forced Action`) into an aggregate `Deceptive Interaction` super-category or perform targeted web scraping to expand minority classes beyond <30 examples.")
    md.append("3. **Threshold Calibration:** Utilize category-specific F1-maximizing thresholds rather than a uniform threshold across imbalanced classes.\n")

    report_str = "\n".join(md)
    Path(output_path).write_text(report_str, encoding="utf-8")
    print(f"\nSaved report to {output_path}")

if __name__ == "__main__":
    run_multilabel_evaluation()
