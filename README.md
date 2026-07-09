# CTG-BERT Final Thesis Project

This repository contains a thesis project for clinical text modeling using a custom CTG-BERT pipeline. It includes:

- A shared tokenizer trained on CTG and delivery-related text
- Shared masked-language-model pretraining
- Multi-task fine-tuning for:
  - CTG fetal health classification
  - Delivery type classification
- Evaluation, confusion matrices, prediction utilities, and explainability analysis

## Project Files

- CTG_BERT_final.ipynb — main notebook for training, evaluation, and explainability
- Feature_to_Text.py — supporting script
- ctg_full_text.csv — CTG text data
- ctg_full_text_balanced.csv — balanced CTG dataset
- delivery.csv — delivery text data
- delivery_balanced.csv — balanced delivery dataset
- fetal_health(kaggle).csv — original CTG dataset source

## Requirements

Python 3.10 or newer is recommended.

Install dependencies using either pip or conda.

## Setup

### With pip
```bash
pip install -r requirements.txt
```

### With conda
```bash
conda env create -f environment.yml
conda activate ctg-bert
```

## How to Run

1. Open CTG_BERT_final.ipynb
2. Run the tokenizer training cell
3. Run the shared MLM pretraining cell
4. Run the multi-task fine-tuning cell
5. Run the evaluation and explainability cells

## Outputs

The notebook saves model artifacts such as:

- my_perfect_tokenizer
- CTG_BERT_BASE
- CTG_BERT_MULTITASK

These are generated files and do not need to be committed unless you want to share trained weights.

## Notes

- If the datasets contain sensitive or restricted medical data, do not publish the raw CSV files publicly.
- For a public thesis repository, it is often better to share code, documentation, and a small sample dataset only.
- If you want reproducibility, include instructions for how to obtain the original data.

## Suggested Repo Contents

Recommended to include:
- CTG_BERT_final.ipynb
- Feature_to_Text.py
- README.md
- requirements.txt
- environment.yml

Optional:
- Sample data
- Figures and exported results
- A short thesis summary or poster PDF
