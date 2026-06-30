# FactoDual-X Experimental Results

This document summarizes the classification performance of **FactoDual-X** on the Indian Pines hyperspectral dataset and compares it against other state-of-the-art architectures.

## Quantitative Metrics

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy (OA)** | 95.99% |
| **Average Accuracy (AA)** | 96.54% |
| **Cohen's Kappa ($\kappa$)** | 0.9541 |

---

## Comparison with Baseline Models

The table below showcases the comparison of FactoDual-X against baseline hyperspectral image classification models (SpectralFormer, MAEST, and FactoFormer) on the Indian Pines dataset:

| Model | Overall Accuracy (OA %) | Average Accuracy (AA %) | Cohen's Kappa ($\kappa$) |
| :--- | :---: | :---: | :---: |
| SpectralFormer | 81.76 | 87.81 | 0.7919 |
| MAEST | 84.15 | 90.97 | 0.8200 |
| FactoFormer | 91.30 | 94.30 | 0.9006 |
| **FactoDual-X (Ours)** | **95.99** | **96.54** | **0.9541** |

---

## Qualitative Visualizations

Detailed qualitative results are available in the repository assets:
- **Ground Truth Map**: `images/ground_truth.png`
- **FactoDual-X Prediction Map**: `images/prediction_map.png`
- **Training curves (Loss & Accuracy)**: `images/training_curves.png`

Detailed classification reports and confusion matrices generated from local runs can be found at `results/classification_report.txt`.
