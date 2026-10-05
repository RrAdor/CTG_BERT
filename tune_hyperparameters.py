#!/usr/bin/env python3
"""
Hyperparameter Evaluation and Automated Empirical Model Selection Pipeline for CTG-BERT.

Workflow:
1. Loads candidate configurations from 'ctg_bert_results/hyperparameter_trials_grid.json'.
2. Evaluates candidate configurations across learning rate, weight decay, batch size,
   epochs, and task-specific dropout heads on the Fold 1 validation partition
   (strictly under natural clinical prevalence, with zero held-out test leakage).
3. Computes comprehensive multi-task clinical metrics for each candidate:
   - CTG Accuracy, Macro F1, Pathological Recall, ROC-AUC
   - Delivery Accuracy, Macro F1, Cesarean Recall, Cesarean PR-AUC, ECE
   - Clinical Composite Multi-Task Score
4. Formats and prints an objective side-by-side comparison table and saves to CSV.
5. Programmatically selects the best-performing configuration (no default bias/hardcoding).
6. Runs the full 5-fold cross-validation fine-tuning using the empirically selected winner.
7. Executes dual-task clinical SHAP explainability pipelines for CTG and Delivery Mode.
"""

import argparse
import os
import sys
import json
import time
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# Import shared modules and functions from finetune_multitask
from finetune_multitask import (
    load_raw_datasets,
    split_train_test,
    get_5fold_splits,
    prepare_hf_dataset,
    CTGBertMultiTask,
    MultiTaskTrainer,
    evaluate_clinical_metrics,
    print_evaluation_report,
    plot_clinical_evaluation_curves,
    plot_clinical_confusion_matrix,
    optimize_binary_threshold,
    predict_dataset,
    calculate_ci_and_summary,
    adjust_logits_for_prior,
    CTG_CLASS_NAMES,
    DELIVERY_CLASS_NAMES,
    main as run_full_finetuning,
)


def evaluate_candidate(
    trial,
    fold_info,
    tokenizer,
    base_model_path="./CTG_BERT_BASE",
    seed=42,
    scratch_dir="./ctg_bert_tune_scratch",
):
    """
    Train and evaluate a single candidate hyperparameter configuration on Fold 1 validation.
    """
    trial_id = trial["id"]
    name = trial.get("name", trial_id)
    lr = float(trial["lr"])
    weight_decay = float(trial["weight_decay"])
    batch_size = int(trial["batch_size"])
    epochs = int(trial["epochs"])
    fetal_dropout = float(trial.get("fetal_dropout", 0.10))
    delivery_dropout = float(trial.get("delivery_dropout", 0.25))

    print("\n" + "=" * 80)
    print(f"EVALUATING CANDIDATE: {trial_id} - '{name}'")
    print(f"  Params: Epochs={epochs} | LR={lr:.1e} | WD={weight_decay} | BS={batch_size} | "
          f"FetalDrop={fetal_dropout} | DelDrop={delivery_dropout}")
    print("=" * 80)

    clean_id = trial_id.replace(" ", "_").lower()
    trial_dir = os.path.join(scratch_dir, f"trial_{clean_id}")
    os.makedirs(trial_dir, exist_ok=True)

    set_seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CTGBertMultiTask(
        base_model_path=base_model_path,
        use_focal_loss=True,
        focal_gamma=2.0,
        use_uncertainty_weighting=True,
        fetal_dropout=fetal_dropout,
        delivery_dropout=delivery_dropout,
    )

    tokenized_train = prepare_hf_dataset(fold_info["train_balanced"], tokenizer)
    tokenized_val = prepare_hf_dataset(fold_info["val_natural"], tokenizer)

    warmup_steps = min(100, int(len(tokenized_train) / batch_size * epochs * 0.1))

    training_args = TrainingArguments(
        output_dir=trial_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        eval_strategy="no",       # Disable per-epoch eval during search to accelerate CPU throughput
        save_strategy="no",       # Do not save checkpoints during tuning to conserve disk space
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=None,
        data_collator=default_data_collator,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # Run validation inference under natural clinical prevalence
    val_preds = predict_dataset(model, tokenized_val, device=device, batch_size=batch_size)

    ctg_metrics = evaluate_clinical_metrics(
        y_true=val_preds[0]["y_true"],
        y_score=val_preds[0]["y_score"],
        class_names=CTG_CLASS_NAMES,
        task_name=f"CTG Validation ({trial_id})",
    )

    del_metrics = evaluate_clinical_metrics(
        y_true=val_preds[1]["y_true"],
        y_score=val_preds[1]["y_score"],
        class_names=DELIVERY_CLASS_NAMES,
        task_name=f"Delivery Validation ({trial_id})",
    )

    ctg_acc = ctg_metrics["accuracy"]
    ctg_f1 = ctg_metrics["macro_f1"]
    pathological_recall = ctg_metrics["class_sensitivities"].get("Pathological", 0.0)
    ctg_roc_auc = ctg_metrics.get("macro_roc_auc", 0.0)

    del_acc = del_metrics["accuracy"]
    del_f1 = del_metrics["macro_f1"]
    cesarean_recall = del_metrics["class_sensitivities"].get("Cesarean", 0.0)
    cesarean_pr_auc = del_metrics["class_pr_auc"].get("Cesarean", 0.0)
    del_ece = del_metrics["ece"]

    # Multi-task composite clinical score:
    # 40% CTG Macro F1 + 30% Delivery Macro F1 + 15% Pathological Recall + 15% Cesarean PR-AUC
    composite_score = (
        0.40 * ctg_f1 +
        0.30 * del_f1 +
        0.15 * pathological_recall +
        0.15 * cesarean_pr_auc
    )
    harmonic_f1 = (2 * ctg_f1 * del_f1) / (ctg_f1 + del_f1 + 1e-8)

    print(f"\n--> RESULTS FOR {trial_id} ('{name}'): Runtime = {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"    CTG Accuracy:       {ctg_acc*100:.2f}% | Macro F1: {ctg_f1:.4f} | Pathological Recall: {pathological_recall*100:.2f}% | ROC-AUC: {ctg_roc_auc:.4f}")
    print(f"    Delivery Accuracy:  {del_acc*100:.2f}% | Macro F1: {del_f1:.4f} | Cesarean Recall:    {cesarean_recall*100:.2f}% | PR-AUC:  {cesarean_pr_auc:.4f}")
    print(f"    ★ Clinical Composite Score: {composite_score:.4f} (Harmonic Mean F1: {harmonic_f1:.4f})")

    # Clean up trial scratch directory
    if os.path.exists(trial_dir):
        shutil.rmtree(trial_dir, ignore_errors=True)

    return {
        "trial_id": trial_id,
        "name": name,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "fetal_dropout": fetal_dropout,
        "delivery_dropout": delivery_dropout,
        "train_runtime_sec": round(elapsed, 1),
        "ctg_accuracy": ctg_acc,
        "ctg_macro_f1": ctg_f1,
        "ctg_pathological_recall": pathological_recall,
        "ctg_roc_auc": ctg_roc_auc,
        "delivery_accuracy": del_acc,
        "delivery_macro_f1": del_f1,
        "delivery_cesarean_recall": cesarean_recall,
        "delivery_cesarean_pr_auc": cesarean_pr_auc,
        "delivery_ece": del_ece,
        "harmonic_f1": harmonic_f1,
        "composite_score": composite_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CTG-BERT Hyperparameter Evaluation & Automated Empirical Model Selection"
    )
    parser.add_argument("--ctg-path", default="ctg_full_text.csv", help="Path to CTG CSV")
    parser.add_argument("--delivery-path", default="delivery.csv", help="Path to Delivery CSV")
    parser.add_argument("--tokenizer-dir", default="./my_perfect_tokenizer", help="Path to tokenizer")
    parser.add_argument("--base-model", default="./CTG_BERT_BASE", help="Path to base BERT model")
    parser.add_argument("--output-dir", default="./ctg_bert_results", help="Output directory for final results")
    parser.add_argument("--trials-file", default="./ctg_bert_results/hyperparameter_trials_grid.json", help="Path to trials JSON file")
    parser.add_argument("--candidate-trials", default="Trial 1,Trial 2,Trial 3,Trial 4,Trial 14,Trial 17,Trial 20", help="Comma-separated trial IDs to evaluate (or 'all')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--skip-final-cv", action="store_true", help="Skip final full 5-fold CV run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(f"CTG-BERT EMPIRICAL HYPERPARAMETER SELECTION PIPELINE (Device: {device})")
    print("=" * 85)

    # 1. Load Data and Tokenizer
    ctg_df, del_df = load_raw_datasets(args.ctg_path, args.delivery_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)

    # 2. Prepare 5-Fold Splits (Zero-leakage: Fold 1 validation partition is untouched by test set)
    ctg_train, ctg_test, del_train, del_test = split_train_test(
        ctg_df, del_df, test_size=0.2, random_state=args.seed
    )
    folds = get_5fold_splits(ctg_train, del_train, n_splits=5, random_state=args.seed)
    fold_1 = folds[0]

    # 3. Load Trials Grid
    if not os.path.exists(args.trials_file):
        raise FileNotFoundError(f"Trials grid file '{args.trials_file}' not found.")

    with open(args.trials_file, "r") as f:
        all_trials = json.load(f)

    # Filter candidate trials
    if args.candidate_trials.strip().lower() == "all":
        selected_trials = all_trials
    else:
        req_ids = {t.strip() for t in args.candidate_trials.split(",") if t.strip()}
        selected_trials = [t for t in all_trials if t["id"] in req_ids]

    if not selected_trials:
        raise ValueError(f"No matching trials found for candidate list: '{args.candidate_trials}'")

    print(f"\nLoaded {len(selected_trials)} Candidate Hyperparameter Setups to evaluate:")
    for t in selected_trials:
        print(f"  • [{t['id']}] {t.get('name', ''):<35} | LR={float(t['lr']):.1e} | WD={t['weight_decay']} | "
              f"BS={t['batch_size']:<2d} | Ep={t['epochs']:<2d} | FetalDrop={t.get('fetal_dropout', 0.1):.2f} | DelDrop={t.get('delivery_dropout', 0.25):.2f}")

    # 4. Evaluate Each Candidate Setup on Fold 1
    tuning_results = []
    for i, t in enumerate(selected_trials, 1):
        print(f"\n>>> Running Benchmark {i}/{len(selected_trials)}: {t['id']} ...")
        res = evaluate_candidate(
            trial=t,
            fold_info=fold_1,
            tokenizer=tokenizer,
            base_model_path=args.base_model,
            seed=args.seed,
        )
        tuning_results.append(res)
        # Save progress incrementally
        pd.DataFrame(tuning_results).to_csv(os.path.join(args.output_dir, "hyperparameter_tuning_results.csv"), index=False)

    # 5. Format & Display Results Comparison Table
    df_results = pd.DataFrame(tuning_results)
    tuning_csv_path = os.path.join(args.output_dir, "hyperparameter_tuning_results.csv")
    df_results.to_csv(tuning_csv_path, index=False)
    brain_dir = "/home/ador/.gemini/antigravity/brain/f7c509c8-0e92-4a2f-bf73-9804ca2874b7"
    shutil.copy(tuning_csv_path, os.path.join(brain_dir, "hyperparameter_tuning_results.csv"))

    print("\n" + "=" * 115)
    print("HYPERPARAMETER BENCHMARK & COMPARISON TABLE (FOLD 1 VALIDATION - NATURAL PREVALENCE)")
    print("=" * 115)
    header = (
        f"{'Trial ID':<9} | {'Name':<30} | {'LR':<7} | {'WD':<5} | {'BS':<3} | {'Ep':<3} | "
        f"{'CTG F1':<7} | {'Path.Rec':<8} | {'Del F1':<7} | {'Ces.PR':<7} | {'Score':<7}"
    )
    print(header)
    print("-" * len(header))
    for r in tuning_results:
        row_str = (
            f"{r['trial_id']:<9} | {r['name'][:30]:<30} | {r['learning_rate']:<7.1e} | "
            f"{r['weight_decay']:<5.2f} | {r['batch_size']:<3d} | {r['epochs']:<3d} | "
            f"{r['ctg_macro_f1']:<7.4f} | {r['ctg_pathological_recall']*100:<7.2f}% | "
            f"{r['delivery_macro_f1']:<7.4f} | {r['delivery_cesarean_pr_auc']:<7.4f} | "
            f"{r['composite_score']:<7.4f}"
        )
        print(row_str)
    print("=" * 115)
    print(f"✅ Tuning results table saved to: '{tuning_csv_path}'")

    # 6. Programmatically Select Best Candidate Based on Empirical Score
    best_row = df_results.loc[df_results["composite_score"].idxmax()]
    best_trial_id = best_row["trial_id"]
    best_trial_name = best_row["name"]
    best_epochs = int(best_row["epochs"])
    best_lr = float(best_row["learning_rate"])
    best_wd = float(best_row["weight_decay"])
    best_bs = int(best_row["batch_size"])
    best_fd = float(best_row["fetal_dropout"])
    best_dd = float(best_row["delivery_dropout"])

    print("\n" + "*" * 80)
    print(f"🏆 BEST HYPERPARAMETER CONFIGURATION SELECTED EMPIRICALLY: {best_trial_id}")
    print(f"   Name: '{best_trial_name}'")
    print("*" * 80)
    print(f"  • Selected Learning Rate:      {best_lr:.1e}")
    print(f"  • Selected Weight Decay:       {best_wd}")
    print(f"  • Selected Batch Size:         {best_bs}")
    print(f"  • Selected Fine-Tuning Epochs: {best_epochs}")
    print(f"  • Selected Fetal Dropout:      {best_fd}")
    print(f"  • Selected Delivery Dropout:   {best_dd}")
    print(f"  • Validation Composite Score:  {best_row['composite_score']:.4f}")
    print(f"  • CTG Macro F1:                {best_row['ctg_macro_f1']:.4f} (Accuracy: {best_row['ctg_accuracy']*100:.2f}%, Pathological Recall: {best_row['ctg_pathological_recall']*100:.2f}%)")
    print(f"  • Delivery Macro F1:           {best_row['delivery_macro_f1']:.4f} (Accuracy: {best_row['delivery_accuracy']*100:.2f}%, Cesarean PR-AUC: {best_row['delivery_cesarean_pr_auc']:.4f})")
    print("*" * 80)

    # Update trials grid file with selection metadata
    for t in all_trials:
        t["selected_best"] = (t["id"] == best_trial_id)
    with open(args.trials_file, "w") as f:
        json.dump(all_trials, f, indent=2)
    shutil.copy(args.trials_file, os.path.join(brain_dir, "hyperparameter_trials_grid.json"))

    # 7. Execute Full 5-Fold Cross-Validation Fine-Tuning with Winning Parameters
    if not args.skip_final_cv:
        print("\n" + "=" * 80)
        print(f"STARTING FULL 5-FOLD CROSS-VALIDATION USING EMPIRICAL WINNER ({best_trial_id})")
        print("=" * 80)

        sys.argv = [
            "finetune_multitask.py",
            "--ctg-path", args.ctg_path,
            "--delivery-path", args.delivery_path,
            "--tokenizer-dir", args.tokenizer_dir,
            "--base-model", args.base_model,
            "--output-dir", args.output_dir,
            "--epochs", str(best_epochs),
            "--lr", str(best_lr),
            "--weight-decay", str(best_wd),
            "--batch-size", str(best_bs),
            "--fetal-dropout", str(best_fd),
            "--delivery-dropout", str(best_dd),
            "--seed", str(args.seed),
        ]
        run_full_finetuning()

        print("\n" + "=" * 80)
        print("STAGE 7 COMPLETE: 5-FOLD CV & EXTERNAL HELD-OUT EVALUATION FINISHED.")
        print("=" * 80)

        # 8. Re-run Dual-Task Clinical Variable-Level SHAP Explainability
        print("\n" + "=" * 80)
        print("STAGE 8: RUNNING DUAL-TASK CLINICAL SHAP EXPLAINABILITY PIPELINE")
        print("=" * 80)
        shap_ctg_cmd = (
            f"MPLCONFIGDIR=/tmp {sys.executable} evaluate_clinical_shap.py "
            f"--model-path ./CTG_BERT_MULTITASK/pytorch_model.bin --output-dir {args.output_dir}"
        )
        print(f"Executing: {shap_ctg_cmd}")
        os.system(shap_ctg_cmd)

        shap_del_cmd = (
            f"MPLCONFIGDIR=/tmp {sys.executable} evaluate_delivery_shap.py "
            f"--model-path ./CTG_BERT_MULTITASK/pytorch_model.bin --output-dir {args.output_dir}"
        )
        print(f"Executing: {shap_del_cmd}")
        os.system(shap_del_cmd)

        print("\n" + "=" * 80)
        print("ALL WORKFLOWS SUCCESSFULLY COMPLETED!")
        print(f"1. Tuning comparison CSV:             {tuning_csv_path}")
        print(f"2. 5-fold CV evaluation summary:       {args.output_dir}/multitask_cv_evaluation_summary.json")
        print(f"3. 5-fold CV fold metrics CSV:         {args.output_dir}/multitask_cv_fold_metrics.csv")
        print(f"4. CTG SHAP summary JSON:              {args.output_dir}/clinical_shap_summary.json")
        print(f"5. Delivery SHAP summary JSON:         {args.output_dir}/clinical_delivery_shap_summary.json")
        print(f"6. All Confusion Matrices & Curves:    {args.output_dir}/*.png")
        print("=" * 80)


if __name__ == "__main__":
    main()
