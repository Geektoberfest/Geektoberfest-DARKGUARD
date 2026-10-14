# 🛡️ Short and Subtle Text Detection: Improvement Report

## Objective

Address Issue #2 by improving detection of dark patterns in short, context-poor, and subtle website snippets without degrading overall performance on the held-out test set.

## Diagnostic Root Cause

1. **Stop-Word Erosion:** Scikit-learn's default `stop_words='english'` stripped crucial dark-pattern signal words (`only`, `now`, `few`, `will`, `out`, `all`, `last`). In short snippets such as 'Only 2 left', the primary scarcity marker 'only' was completely deleted.

2. **Sparse TF-IDF Representation:** With word n-grams alone, short snippets yield only 1 to 3 non-zero features. Out-of-vocabulary variations or punctuation boundaries fail to trigger trained weights.

3. **Solution:** We designed a hybrid `FeatureUnion` combining:

   - Word TF-IDF with curated stop-words (preserving urgency/scarcity signal words) and sublinear term frequency scaling.

   - Character boundary n-grams (`char_wb`, range 3-5, min_df=2) that capture morphology, abbreviations, numerical countdowns, and subwords.

   - Tuned inverse regularization parameter `C=1.5`.

## Benchmark Results

### 1. Curated Short & Subtle Evaluation Set (n = 30)

| Metric | Baseline Model | Enhanced Model | Delta |
|---|---|---|---|
| **Accuracy** | 83.33% | 90.00% | **+6.67%** |
| **Precision** | 0.8571 | 0.9286 | **+0.0714** |
| **Recall** | 0.8000 | 0.8667 | **+0.0667** |
| **F1-Score** | 0.8276 | 0.8966 | **+0.0690** |
| **False Negatives** | 3 | 2 | **-1** |
| **False Positives** | 2 | 1 | **-1** |

### 2. Held-Out Test Set - Short Texts (<= 4 words, n = 221)

| Metric | Baseline Model | Enhanced Model | Delta |
|---|---|---|---|
| **Accuracy** | 93.67% | 95.02% | **+1.36%** |
| **Precision** | 0.9123 | 0.9310 | **+0.0188** |
| **Recall** | 0.8525 | 0.8852 | **+0.0328** |
| **F1-Score** | 0.8814 | 0.9076 | **+0.0262** |
| **False Negatives** | 9 | 7 | **-2** |
| **False Positives** | 5 | 4 | **-1** |

### 3. Overall Held-Out Test Set (n = 472)

| Metric | Baseline Model | Enhanced Model | Delta |
|---|---|---|---|
| **Accuracy** | 93.43% | 94.92% | **+1.48%** |
| **Precision** | 0.9638 | 0.9690 | **+0.0052** |
| **Recall** | 0.9025 | 0.9280 | **+0.0254** |
| **F1-Score** | 0.9322 | 0.9481 | **+0.0159** |
| **False Negatives** | 23 | 17 | **-6** |
| **False Positives** | 8 | 7 | **-1** |

## Example Level Predictions on Curated Benchmark

| Snippet | Label | Baseline Pred | Enhanced Pred | Status |
|---|---|---|---|---|
| `Only 2 left!` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Hurry!` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Limited time offer` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Will sell out fast!` | Dark Pattern (1) | Not Dark (0) | Not Dark (0) | ⚠️ Error remaining |
| `Sale ends today!` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Almost gone` | Dark Pattern (1) | Not Dark (0) | Not Dark (0) | ⚠️ Error remaining |
| `Just a few left` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Low in stock` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `High demand` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Offer expires soon` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `12 people viewing now` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Selling fast!` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Only 1 item remaining` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Last chance to save` | Dark Pattern (1) | Dark (1) | Dark (1) | ✓ Correct |
| `Claim your discount now` | Dark Pattern (1) | Not Dark (0) | Dark (1) | ✅ Fixed by Enhanced |
| `Size: Medium` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Color: Blue` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Standard shipping available` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Write a customer review` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Product specifications` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Free returns within 30 days` | Not Dark (0) | Dark (1) | Not Dark (0) | ✅ Fixed by Enhanced |
| `Add to wish list` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Contact customer support` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `100% cotton fabric` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `View item details` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Available in multiple sizes` | Not Dark (0) | Dark (1) | Dark (1) | ⚠️ Error remaining |
| `Machine wash cold` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Track your order` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Frequently asked questions` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
| `Customer ratings` | Not Dark (0) | Not Dark (0) | Not Dark (0) | ✓ Correct |
