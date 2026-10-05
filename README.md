# CTG-BERT: Multi-Task Clinical Language Modeling for Fetal Health Classification and Delivery Mode Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-4.40+-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official repository for **CTG-BERT**, a multi-task clinical Transformer framework pre-trained on domain-specific cardiotocography (CTG) text representations and jointly fine-tuned for:
1. **Task A: 3-Class Fetal Health Status Classification** (*Normal*, *Suspect*, *Pathological*)
2. **Task B: Intrapartum Delivery Mode Prediction** (*Vaginal* vs. *Cesarean*)

---

## Key Highlights

- **Joint Architecture**: Domain-adapted 12-layer Transformer encoder ($H=768, A=12$, 110M params) with dual classification heads.
- **Objective Balancing**: Class-weighted Focal Loss ($\gamma = 2.0$) dynamically balanced via **Kendall Homoscedastic Uncertainty Weighting** (Kendall et al., CVPR 2018).
- **Statistical Rigor**: 5-fold stratified cross-validation on natural clinical prevalence with Student's $t$ 95% confidence intervals (zero validation leakage).
- **Clinical Calibration**: Bayesian prior calibration reduces delivery Expected Calibration Error (ECE) from $0.1201 \to 0.0185$.
- **Cooperative Explainability**: Exact game-theoretic Shapley attributions across $2^8 = 256$ feature coalitions per patient, aligned with FIGO 2015 and ACOG guidelines.

---

## Repository Structure

```text
CTG_BERT/
├── CTG_BERT_BASE/              # Pretrained domain-adapted CTG-BERT Base model weights & config
├── CTG_BERT_MULTITASK/         # Fine-tuned multi-task model weights & tokenizer
├── my_perfect_tokenizer/       # Domain WordPiece tokenizer vocabulary
├── ctg_bert_results/           # 5-fold cross-validation checkpoints, evaluation metrics, and figures
│   ├── fold_1/ ... fold_5/     # Saved best model weights per fold (pytorch_model.bin)
│   ├── multitask_cv_fold_metrics.csv
│   ├── multitask_cv_evaluation_summary.json
│   ├── hyperparameter_tuning_results.csv
│   └── *.png                   # Publication figures (ROC, PR, Calibration, SHAP heatmaps)
├── ctg_full_text.csv           # CTG dataset with clinical text serialization
├── delivery.csv                # Maternal demographics and labor progression dataset
├── finetune_multitask.py       # Core multi-task 5-fold training and evaluation engine
├── tune_hyperparameters.py     # Hyperparameter grid evaluation script
├── run_hyperparameter_suite.py # Orchestrator for hyperparameter search
├── evaluate_clinical_shap.py   # Exact cooperative SHAP evaluation for CTG physiological features
├── evaluate_delivery_shap.py   # Exact cooperative SHAP evaluation for obstetric delivery features
├── train_tokenizer_and_mlm.py  # Domain WordPiece tokenizer training and MLM pre-training script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (excludes virtualenvs, caches, checkpoints)
└── README.md                   # Project documentation
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/CTG_BERT.git
   cd CTG_BERT
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Quick Start & Reproduction

### 1. Run 5-Fold Stratified Cross-Validation
Train and evaluate CTG-BERT across 5 stratified folds with training-only balancing and natural-prevalence validation:
```bash
python finetune_multitask.py \
    --ctg_data ctg_full_text.csv \
    --delivery_data delivery.csv \
    --base_model ./CTG_BERT_BASE \
    --output_dir ./ctg_bert_results \
    --epochs 15 \
    --lr 3e-5 \
    --batch_size 16 \
    --weight_decay 0.10 \
    --fetal_dropout 0.20 \
    --delivery_dropout 0.30 \
    --n_splits 5
```

### 2. Run Exact Cooperative Game-Theoretic SHAP Analysis
Compute exact Shapley attributions across $2^8 = 256$ coalitions:
```bash
# CTG Physiological Feature Explainability
python evaluate_clinical_shap.py

# Obstetric Delivery Feature Explainability
python evaluate_delivery_shap.py
```

### 3. Hyperparameter Optimization & Search Suite
Explore hyperparameter configurations across learning rates, dropouts, and Kendall uncertainty weighting:
```bash
python tune_hyperparameters.py
# Or run the multi-configuration automated suite:
python run_hyperparameter_suite.py
```

---

## Experimental Results Summary

### 5-Fold Stratified Cross-Validation (Trial 14 Configuration)
*Evaluated on unaltered natural clinical prevalence ($N=1,700$ CTG, $N=441$ delivery):*

| Task / Metric | Mean $\pm$ SD | 95% Confidence Interval |
| :--- | :---: | :---: |
| **CTG Overall Accuracy** | **86.29 $\pm$ 1.83%** | $[84.02\%, 88.57\%]$ |
| **CTG Macro $F_1$ Score** | **0.7995 $\pm$ 0.0185** | $[0.7765, 0.8224]$ |
| **Pathological Sensitivity (Critical)** | **88.62 $\pm$ 4.71%** | $[82.77\%, 94.47\%]$ |
| **Pathological Specificity** | **97.24 $\pm$ 0.86%** | $[96.17\%, 98.32\%]$ |
| **Delivery Accuracy (Optimal $\tau^*=0.64$)** | **90.99%** | — |
| **Delivery Expected Calibration Error (ECE)** | **0.0185** | — |

### External Held-Out Test Cohort ($N = 537$)
* **Single Primary Model**: CTG Accuracy $84.04\%$, Pathological Recall **$94.29\%$** (33 of 35 critical hypoxia cases detected).
* **5-Fold Ensemble**: CTG Accuracy $85.21\%$ ($87.79\%$ prior-calibrated), Macro ROC-AUC **$0.9433$**.
* **Delivery Prediction ($\tau^* = 0.64$)**: Accuracy **$90.99\%$** ($91.89\%$ prior-calibrated), Brier score **$0.1501$**.

---

## Citation

If you use CTG-BERT in your research, please cite:

```bibtex
@article{ctgbert2026,
  title={CTG-BERT: Multi-Task Clinical Language Modeling for Fetal Health Classification and Delivery Mode Prediction},
  author={Research Team},
  journal={arXiv preprint},
  year={2026}
}
```

