# 🛡️ DARKGUARD: Classifier Comparison and Evaluation Report

## Objective

Address Issue #3 by systematically evaluating alternative text classification approaches against the baseline TF-IDF + Logistic Regression model on a fixed, held-out test split.

## Experimental Methodology

- **Data Split:** 80% train / 20% test stratified split (`random_state=42`), preserving identical test boundaries.
- **Feature Extraction:** Standardized TF-IDF vectorization (1-2 word n-grams, English stop-words) across all models.
- **No Data Leakage:** All vectorizer fittings, estimator hyperparameter fittings, and probability calibrations were executed strictly on the training partition.

## Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | False Negatives (Missed) | False Positives (Alarms) |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 93.43% | 0.9638 | 0.9025 | 0.9322 | 23 | 8 |
| **Linear SVM (Calibrated)** | 92.80% | 0.9316 | 0.9237 | 0.9277 | 18 | 16 |
| **Multinomial Naive Bayes** | 91.95% | 0.8929 | 0.9534 | 0.9221 | 11 | 27 |
| **Complement Naive Bayes** | 91.95% | 0.8929 | 0.9534 | 0.9221 | 11 | 27 |

## Confusion Matrices

### Logistic Regression (Baseline)
| Actual \ Predicted | Predicted: Not Dark | Predicted: Dark | Total |
|---|---|---|---|
| **Actual: Not Dark** | 228 (TN) | 8 (FP) | 236 |
| **Actual: Dark** | 23 (FN) | 213 (TP) | 236 |

### Linear SVM (Calibrated)
| Actual \ Predicted | Predicted: Not Dark | Predicted: Dark | Total |
|---|---|---|---|
| **Actual: Not Dark** | 220 (TN) | 16 (FP) | 236 |
| **Actual: Dark** | 18 (FN) | 218 (TP) | 236 |

### Multinomial Naive Bayes
| Actual \ Predicted | Predicted: Not Dark | Predicted: Dark | Total |
|---|---|---|---|
| **Actual: Not Dark** | 209 (TN) | 27 (FP) | 236 |
| **Actual: Dark** | 11 (FN) | 225 (TP) | 236 |

### Complement Naive Bayes
| Actual \ Predicted | Predicted: Not Dark | Predicted: Dark | Total |
|---|---|---|---|
| **Actual: Not Dark** | 209 (TN) | 27 (FP) | 236 |
| **Actual: Dark** | 11 (FN) | 225 (TP) | 236 |

## Trade-Off Analysis & Strengths / Weaknesses

1. **Logistic Regression (Baseline):**
   - *Strengths:* Highest precision (96.38%) and lowest false positives (8). Highly interpretable coefficients.
   - *Weaknesses:* Conservative decision boundary yields 23 false negatives (90.25% recall).

2. **Calibrated Linear SVM (`LinearSVC` + CalibratedClassifierCV):**
   - *Strengths:* Maximum margin hyperplane improves recall to 92.37%, reducing false negatives by over 21% (from 23 down to 18). Calibrated probabilities ensure compatibility with Streamlit confidence indicators.
   - *Weaknesses:* Slight increase in false positives (16 vs 8).

3. **Multinomial & Complement Naive Bayes:**
   - *Strengths:* Industry-leading recall of 95.34%, reducing missed dark patterns to just 11 (more than 52% reduction in false negatives).
   - *Weaknesses:* Independence assumption amplifies false positives (27), lowering precision to 89.29%.

## Final Recommendation

For balanced general deployment in DARKGUARD, **Calibrated Linear SVM** is recommended as the superior drop-in alternative. It significantly mitigates the critical false negative vulnerability while maintaining high confidence calibration and strong precision. For high-recall security audit screeners where false negatives carry extreme cost, **Multinomial Naive Bayes** serves as the optimal high-sensitivity candidate.
