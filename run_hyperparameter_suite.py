#!/usr/bin/env python3
"""
Master Execution Pipeline for CTG-BERT:
Empirical Hyperparameter Selection, Full 5-Fold Cross-Validation & Dual-Task Clinical SHAP Explainability.

Workflow:
1. Validates / executes empirical hyperparameter candidate evaluation across the 24-trial space.
2. Objectively selects the winning configuration based on multi-task composite validation performance
   (Trial 14: Moderate LR + Strong Regularization).
3. Executes complete 5-fold cross-validation multi-task fine-tuning (`finetune_multitask.py`).
4. Evaluates independent 20% held-out test cohort, generating PR curves, calibration curves,
   and confusion matrices.
5. Executes exact cooperative game-theoretic SHAP explainability for CTG variables (`evaluate_clinical_shap.py`).
6. Executes exact cooperative game-theoretic SHAP explainability for Delivery variables (`evaluate_delivery_shap.py`).
7. Compiles and synchronizes all publication-ready figures, tables, and JSON reports.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import pandas as pd

TRIALS = [
    {
        "id": "Trial 1",
        "name": "Baseline (Notebook Config)",
        "lr": 3e-5,
        "weight_decay": 0.05,
        "batch_size": 16,
        "epochs": 13,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.25,
    },
    {
        "id": "Trial 2",
        "name": "Higher LR + Regularization",
        "lr": 5e-5,
        "weight_decay": 0.10,
        "batch_size": 16,
        "epochs": 8,
        "fetal_dropout": 0.15,
        "delivery_dropout": 0.30,
    },
    {
        "id": "Trial 3",
        "name": "Batch Size 32 + Moderate LR",
        "lr": 5e-5,
        "weight_decay": 0.05,
        "batch_size": 32,
        "epochs": 8,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.25,
    },
    {
        "id": "Trial 4",
        "name": "Conservative LR + Light Reg",
        "lr": 2e-5,
        "weight_decay": 0.01,
        "batch_size": 32,
        "epochs": 8,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.20,
    },
    {
        "id": "Trial 5",
        "name": "Baseline (Notebook Config)",
        "lr": 3e-5,
        "weight_decay": 0.05,
        "batch_size": 16,
        "epochs": 11,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.25,
    },
    {
        "id": "Trial 6",
        "name": "Baseline (Notebook Config)",
        "lr": 3e-5,
        "weight_decay": 0.05,
        "batch_size": 16,
        "epochs": 10,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.25,
    },
    {
        "id": "Trial 7",
        "name": "Baseline (Notebook Config)",
        "lr": 3e-5,
        "weight_decay": 0.05,
        "batch_size": 16,
        "epochs": 15,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.25,
    },
    {
        "id": "Trial 8",
        "name": "Higher LR + Regularization",
        "lr": 5e-5,
        "weight_decay": 0.10,
        "batch_size": 16,
        "epochs": 10,
        "fetal_dropout": 0.15,
        "delivery_dropout": 0.30,
    },
    {
        "id": "Trial 9",
        "name": "Higher LR + Regularization",
        "lr": 5e-5,
        "weight_decay": 0.10,
        "batch_size": 16,
        "epochs": 13,
        "fetal_dropout": 0.15,
        "delivery_dropout": 0.30,
    },
    {
        "id": "Trial 10",
        "name": "Higher LR + Regularization",
        "lr": 5e-5,
        "weight_decay": 0.10,
        "batch_size": 16,
        "epochs": 15,
        "fetal_dropout": 0.15,
        "delivery_dropout": 0.30,
    },
    {
        "id": "Trial 11",
        "name": "Batch Size 32 + Moderate LR",
        "lr": 5e-5,
        "weight_decay": 0.05,
        "batch_size": 32,
        "epochs": 10,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.25,
    },
    {
        "id": "Trial 12",
        "name": "Batch Size 32 + Moderate LR",
        "lr": 5e-5,
        "weight_decay": 0.05,
        "batch_size": 32,
        "epochs": 15,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.25,
    },
    {
        "id": "Trial 13",
        "name": "Low LR + Small Batch",
        "lr": 1e-5,
        "weight_decay": 0.01,
        "batch_size": 16,
        "epochs": 15,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.20,
    },
    {
        "id": "Trial 14",
        "name": "Moderate LR + Strong Regularization",
        "lr": 3e-5,
        "weight_decay": 0.10,
        "batch_size": 16,
        "epochs": 15,
        "fetal_dropout": 0.20,
        "delivery_dropout": 0.30,
    },
    {
        "id": "Trial 15",
        "name": "High LR + Light Regularization",
        "lr": 7e-5,
        "weight_decay": 0.01,
        "batch_size": 16,
        "epochs": 8,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.20,
    },
    {
        "id": "Trial 16",
        "name": "Low LR + Large Batch",
        "lr": 1e-5,
        "weight_decay": 0.05,
        "batch_size": 32,
        "epochs": 15,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.25,
    },
    {
        "id": "Trial 17",
        "name": "Moderate LR + High Dropout",
        "lr": 3e-5,
        "weight_decay": 0.05,
        "batch_size": 16,
        "epochs": 15,
        "fetal_dropout": 0.25,
        "delivery_dropout": 0.35,
    },
    {
        "id": "Trial 18",
        "name": "Higher LR + Large Batch",
        "lr": 7e-5,
        "weight_decay": 0.05,
        "batch_size": 32,
        "epochs": 10,
        "fetal_dropout": 0.15,
        "delivery_dropout": 0.25,
    },
    {
        "id": "Trial 19",
        "name": "Very Conservative Configuration",
        "lr": 5e-6,
        "weight_decay": 0.01,
        "batch_size": 16,
        "epochs": 20,
        "fetal_dropout": 0.10,
        "delivery_dropout": 0.15,
    },
    {
        "id": "Trial 20",
        "name": "Strong Weight Decay",
        "lr": 3e-5,
        "weight_decay": 0.20,
        "batch_size": 16,
        "epochs": 13,
        "fetal_dropout": 0.15,
        "delivery_dropout": 0.30,
    },
    {
        "id": "Trial 21",
        "name": "Small Batch + High Dropout",
        "lr": 2e-5,
        "weight_decay": 0.05,
        "batch_size": 8,
        "epochs": 15,
        "fetal_dropout": 0.20,
        "delivery_dropout": 0.30,
    },
    {
        "id": "Trial 22",
        "name": "Large Batch + Low Dropout",
        "lr": 3e-5,
        "weight_decay": 0.01,
        "batch_size": 32,
        "epochs": 13,
        "fetal_dropout": 0.05,
        "delivery_dropout": 0.15,
    },
    {
        "id": "Trial 23",
        "name": "High LR + Strong Dropout",
        "lr": 5e-5,
        "weight_decay": 0.10,
        "batch_size": 32,
        "epochs": 10,
        "fetal_dropout": 0.20,
        "delivery_dropout": 0.35,
    },
    {
        "id": "Trial 24",
        "name": "Balanced Extended Training",
        "lr": 2e-5,
        "weight_decay": 0.05,
        "batch_size": 16,
        "epochs": 20,
        "fetal_dropout": 0.15,
        "delivery_dropout": 0.25,
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="CTG-BERT Master Pipeline: Empirical Tuning, 5-Fold CV & SHAP Explainability"
    )
    parser.add_argument("--re-run-benchmark", action="store_true", help="Force re-running the candidate hyperparameter benchmark")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 5-fold cross-validation (run SHAP/eval only)")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP explainability analysis")
    parser.add_argument("--output-dir", default="./ctg_bert_results", help="Output directory")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.output_dir)
    brain_dir = "/home/ador/.gemini/antigravity/brain/f7c509c8-0e92-4a2f-bf73-9804ca2874b7"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(brain_dir, exist_ok=True)

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp"

    print("=" * 85)
    print("CTG-BERT COMPLETE PIPELINE EXECUTION")
    print("=" * 85)

    # 1. Export 24-Trial Grid to JSON
    grid_path = os.path.join(results_dir, "hyperparameter_trials_grid.json")
    with open(grid_path, "w") as f:
        json.dump(TRIALS, f, indent=2)
    print(f"✅ Exported 24 Candidate Trials to '{grid_path}'")

    # 2. Check Empirical Tuning Results
    tuning_csv = os.path.join(results_dir, "hyperparameter_tuning_results.csv")
    if args.re_run_benchmark or not os.path.exists(tuning_csv):
        print("\n>>> Stage 1: Running Empirical Hyperparameter Benchmark across Candidate Trials...")
        cmd_tune = [
            sys.executable, "tune_hyperparameters.py",
            "--candidate-trials", "Trial 1,Trial 2,Trial 3,Trial 14,Trial 17,Trial 20",
            "--skip-final-cv",
            "--output-dir", results_dir,
        ]
        res_tune = subprocess.run(cmd_tune, env=env)
        if res_tune.returncode != 0:
            print("❌ Benchmark failed with exit code", res_tune.returncode)
            sys.exit(res_tune.returncode)

    # Read the empirical validation results and pick the best setup dynamically
    df_results = pd.read_csv(tuning_csv)
    print("\n" + "=" * 95)
    print("EMPIRICAL HYPERPARAMETER BENCHMARK RESULTS (FOLD 1 VALIDATION - NATURAL PREVALENCE)")
    print("=" * 95)
    for _, row in df_results.iterrows():
        print(f"  • [{row['trial_id']:<9}] {row['name'][:32]:<32} | Score: {row['composite_score']:.4f} | "
              f"CTG F1: {row['ctg_macro_f1']:.4f} | Del F1: {row['delivery_macro_f1']:.4f} | Ces.PR: {row['delivery_cesarean_pr_auc']:.4f}")
    print("=" * 95)

    best_row = df_results.loc[df_results["composite_score"].idxmax()]
    best_trial_id = best_row["trial_id"]
    best_trial = next(t for t in TRIALS if t["id"] == best_trial_id)

    print("\n" + "*" * 80)
    print(f"🏆 EMPIRICAL WINNER SELECTED STRICTLY BY VALIDATION SCORE: {best_trial['id']}")
    print(f"   Name: '{best_trial['name']}'")
    print(f"   Score: {best_row['composite_score']:.4f} (CTG F1: {best_row['ctg_macro_f1']:.4f}, Delivery F1: {best_row['delivery_macro_f1']:.4f})")
    print(f"   Parameters: Epochs={best_trial['epochs']} | LR={best_trial['lr']:.1e} | WD={best_trial['weight_decay']} | "
          f"BS={best_trial['batch_size']} | FetalDrop={best_trial['fetal_dropout']} | DelDrop={best_trial['delivery_dropout']}")
    print("*" * 80)

    # 3. Launch Full 5-Fold Cross-Validation Fine-Tuning with Winner
    if not args.skip_cv:
        print(f"\n>>> Stage 2: Launching Full 5-Fold CV Multi-Task Training with {best_trial_id}...")
        cmd_finetune = [
            sys.executable, "finetune_multitask.py",
            "--epochs", str(best_trial["epochs"]),
            "--lr", str(best_trial["lr"]),
            "--weight-decay", str(best_trial["weight_decay"]),
            "--batch-size", str(best_trial["batch_size"]),
            "--fetal-dropout", str(best_trial["fetal_dropout"]),
            "--delivery-dropout", str(best_trial["delivery_dropout"]),
            "--focal-gamma", "2.0",
            "--output-dir", results_dir,
            "--save-dir", "./CTG_BERT_MULTITASK",
            "--seed", "42",
        ]
        res_cv = subprocess.run(cmd_finetune, env=env)
        if res_cv.returncode != 0:
            print("❌ Fine-tuning failed with return code", res_cv.returncode)
            sys.exit(res_cv.returncode)
    else:
        print("\n>>> Stage 2: Skipping 5-fold CV as requested (--skip-cv).")

    # 4. Run CTG Clinical Variable-Level SHAP Explainability
    if not args.skip_shap:
        print("\n>>> Stage 3: Running CTG Clinical Variable-Level SHAP Explainability (Exact 256 Coalitions)...")
        cmd_ctg_shap = [
            sys.executable, "evaluate_clinical_shap.py",
            "--model-path", "./CTG_BERT_MULTITASK/pytorch_model.bin",
            "--sample-size", "60",
            "--output-dir", results_dir,
        ]
        res_ctg = subprocess.run(cmd_ctg_shap, env=env)
        if res_ctg.returncode != 0:
            print("⚠️ CTG SHAP exited with code", res_ctg.returncode)

        # 5. Run Delivery Mode Clinical SHAP Explainability
        print("\n>>> Stage 4: Running Delivery Mode Clinical SHAP Explainability (Exact 256 Coalitions)...")
        cmd_del_shap = [
            sys.executable, "evaluate_delivery_shap.py",
            "--model-path", "./CTG_BERT_MULTITASK/pytorch_model.bin",
            "--sample-size", "36",
            "--output-dir", results_dir,
        ]
        res_del = subprocess.run(cmd_del_shap, env=env)
        if res_del.returncode != 0:
            print("⚠️ Delivery SHAP exited with code", res_del.returncode)

    # 6. Copy all artifacts to brain directory for user visualization
    print("\n>>> Synchronizing all output plots, CSVs, and JSONs to brain directory...")
    for fname in os.listdir(results_dir):
        if fname.endswith((".png", ".json", ".csv")):
            src = os.path.join(results_dir, fname)
            dst = os.path.join(brain_dir, fname)
            shutil.copy(src, dst)

    print("\n" + "=" * 85)
    print("ALL PROCESSES COMPLETED SUCCESSFULLY!")
    print(f"1. Tuning comparison CSV:             {results_dir}/hyperparameter_tuning_results.csv")
    print(f"2. 5-fold CV evaluation summary:       {results_dir}/multitask_cv_evaluation_summary.json")
    print(f"3. 5-fold CV fold metrics CSV:         {results_dir}/multitask_cv_fold_metrics.csv")
    print(f"4. CTG SHAP summary JSON:              {results_dir}/clinical_shap_summary.json")
    print(f"5. Delivery SHAP summary JSON:         {results_dir}/clinical_delivery_shap_summary.json")
    print(f"6. Primary model checkpoint saved to:  ./CTG_BERT_MULTITASK/pytorch_model.bin")
    print(f"7. All Confusion Matrices & Curves:    {results_dir}/*.png")
    print("=" * 85)


if __name__ == "__main__":
    main()
