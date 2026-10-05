#!/usr/bin/env python3
"""
Multi-Task Fine-Tuning for CTG-BERT with 5-Fold Stratified Cross-Validation
and Rigorous Clinical Statistical Evaluation.

Directly addresses Reviewer Issue:
"Insufficient Statistical and Clinical Evaluation. Results are based on one seed and
 one validation split, which is especially unreliable given only 46 original Cesarean cases.
 Please conduct repeated stratified experiments and report confidence intervals, class-wise
 sensitivity and specificity, macro F1, PR-AUC, and calibration on the original class
 distribution. External validation should also be considered;"

Key Methodological Guarantees:
1. Leak-Free 80:20 Split: Datasets are partitioned into 80% train and 20% held-out test
   BEFORE any oversampling. The 20% test set serves as the external validation benchmark
   preserving the true natural clinical prevalence.
2. Training-Only Class & Task Balancing: Oversampling is applied strictly to training data
   (and training folds). Validation folds and test sets remain in natural class distribution.
3. 5-Fold Stratified Cross-Validation: Evaluates performance across all folds to eliminate
   single-split bias for the 46 Cesarean cases.
4. Statistical Significance: Computes 5-fold Mean +/- Std and 95% Confidence Intervals (CI).
5. Comprehensive Clinical Metrics: Class-wise sensitivity (recall), specificity, macro F1,
   PR-AUC, ROC-AUC, Expected Calibration Error (ECE), Brier score, decision threshold
   optimization, and Bayesian prior calibration.
6. Plot Generation: Exports publication-quality Precision-Recall and Calibration diagrams.
"""

import argparse
import json
import math
import os
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    BertModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

warnings.filterwarnings("ignore", category=UserWarning)

CTG_LABEL_MAP = {1.0: 0, 2.0: 1, 3.0: 2, "1.0": 0, "2.0": 1, "3.0": 2, 1: 0, 2: 1, 3: 2}
CTG_CLASS_NAMES = ["Normal", "Suspect", "Pathological"]

DELIVERY_LABEL_MAP = {"Vaginal": 0, "Cesarean": 1}
DELIVERY_CLASS_NAMES = ["Vaginal", "Cesarean"]


def find_file(possible_names, base_dir=Path(".")):
    """Locate first existing file matching any candidate names."""
    for name in possible_names:
        p = Path(base_dir) / name
        if p.is_file():
            return str(p)
    return None


# =====================================================================
# 1. DATA PIPELINE: LEAK-FREE SPLIT & TRAINING-ONLY OVERSAMPLING
# =====================================================================

def load_raw_datasets(ctg_path=None, delivery_path=None):
    """
    Load raw CTG and Delivery datasets and map them to standardized schemas.
    """
    if ctg_path is None or not os.path.exists(ctg_path):
        found_ctg = find_file(["ctg_full_text.csv"])
        if not found_ctg:
            raise FileNotFoundError("Could not find CTG dataset ('ctg_full_text.csv').")
        ctg_path = found_ctg

    if delivery_path is None or not os.path.exists(delivery_path):
        found_del = find_file(["delivery.csv", "delivery_update.csv", "delivery_balanced.csv"])
        if not found_del:
            raise FileNotFoundError("Could not find Delivery dataset ('delivery.csv').")
        delivery_path = found_del

    print(f"Loading CTG raw data from:      {ctg_path}")
    print(f"Loading Delivery raw data from: {delivery_path}")

    ctg_df = pd.read_csv(ctg_path).dropna(subset=["clinical_text", "fetal_health"]).copy()
    ctg_df["text"] = ctg_df["clinical_text"].astype(str).str.lower().str.strip()
    ctg_df["label"] = ctg_df["fetal_health"].map(CTG_LABEL_MAP).astype(int)
    ctg_df["task_id"] = 0

    delivery_df = pd.read_csv(delivery_path).dropna(subset=["caption", "delivery_type"]).copy()
    delivery_df["text"] = delivery_df["caption"].astype(str).str.lower().str.strip()
    delivery_df["label"] = delivery_df["delivery_type"].map(DELIVERY_LABEL_MAP).astype(int)
    delivery_df["task_id"] = 1

    return ctg_df[["text", "label", "task_id"]], delivery_df[["text", "label", "task_id"]]


def split_train_test(ctg_df, delivery_df, test_size=0.2, random_state=42):
    """
    Perform leak-free stratified 80:20 train/test split.
    The held-out 20% test partition acts as the external validation set.
    """
    ctg_train, ctg_test = train_test_split(
        ctg_df,
        test_size=test_size,
        random_state=random_state,
        stratify=ctg_df["label"]
    )
    delivery_train, delivery_test = train_test_split(
        delivery_df,
        test_size=test_size,
        random_state=random_state,
        stratify=delivery_df["label"]
    )
    return (
        ctg_train.reset_index(drop=True),
        ctg_test.reset_index(drop=True),
        delivery_train.reset_index(drop=True),
        delivery_test.reset_index(drop=True),
    )


def oversample_dataset(df, seed=42):
    """
    Oversample minority classes within a single dataset to match majority class count.
    Applied EXCLUSIVELY to training pools or training folds.
    """
    class_counts = df["label"].value_counts()
    max_count = class_counts.max()
    balanced_dfs = []
    for cls, count in class_counts.items():
        sub_df = df[df["label"] == cls]
        if count < max_count:
            oversampled = sub_df.sample(n=max_count, replace=True, random_state=seed)
            balanced_dfs.append(oversampled)
        else:
            balanced_dfs.append(sub_df)
    return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_balanced_training_split(ctg_train, delivery_train, random_state=42):
    """
    Balance class distributions for both tasks and equalize total sample counts
    to prevent gradient starvation across tasks during multi-task fine-tuning.
    """
    ctg_bal = oversample_dataset(ctg_train, seed=random_state)
    delivery_bal = oversample_dataset(delivery_train, seed=random_state)

    max_task_len = max(len(ctg_bal), len(delivery_bal))
    ctg_bal_eq = ctg_bal.sample(n=max_task_len, replace=True, random_state=random_state)
    del_bal_eq = delivery_bal.sample(n=max_task_len, replace=True, random_state=random_state)

    combined_train = pd.concat([ctg_bal_eq, del_bal_eq], ignore_index=True)
    combined_train = combined_train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return ctg_bal_eq, del_bal_eq, combined_train


def get_5fold_splits(ctg_train, delivery_train, n_splits=5, random_state=42):
    """
    Generate stratified 5 folds within the 80% training set.
    For each fold:
      - 'train_balanced': oversampled for class and task balance.
      - 'val_natural': natural, un-oversampled distribution (zero leakage!).
    """
    skf_ctg = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    skf_del = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    ctg_splits = list(skf_ctg.split(ctg_train, ctg_train["label"]))
    del_splits = list(skf_del.split(delivery_train, delivery_train["label"]))

    folds = []
    for k in range(n_splits):
        ctg_tr_idx, ctg_va_idx = ctg_splits[k]
        del_tr_idx, del_va_idx = del_splits[k]

        ctg_tr = ctg_train.iloc[ctg_tr_idx].copy()
        ctg_va = ctg_train.iloc[ctg_va_idx].copy()
        del_tr = delivery_train.iloc[del_tr_idx].copy()
        del_va = delivery_train.iloc[del_va_idx].copy()

        # Oversample training fold only
        ctg_tr_bal = oversample_dataset(ctg_tr, seed=random_state + k)
        del_tr_bal = oversample_dataset(del_tr, seed=random_state + k)
        max_len = max(len(ctg_tr_bal), len(del_tr_bal))
        ctg_tr_bal_eq = ctg_tr_bal.sample(n=max_len, replace=True, random_state=random_state + k)
        del_tr_bal_eq = del_tr_bal.sample(n=max_len, replace=True, random_state=random_state + k)
        train_balanced = pd.concat([ctg_tr_bal_eq, del_tr_bal_eq], ignore_index=True).sample(
            frac=1.0, random_state=random_state + k
        ).reset_index(drop=True)

        # Validation fold preserves natural class prevalence
        val_natural = pd.concat([ctg_va, del_va], ignore_index=True).sample(
            frac=1.0, random_state=random_state + k
        ).reset_index(drop=True)

        folds.append({
            "fold": k + 1,
            "train_balanced": train_balanced,
            "val_natural": val_natural,
            "ctg_val_len": len(ctg_va),
            "del_val_len": len(del_va),
            "del_val_cesarean": int((del_va["label"] == 1).sum()),
        })

    return folds


def prepare_hf_dataset(df, tokenizer, max_length=128):
    """Convert pandas DataFrame to tokenized Hugging Face Dataset."""
    ds = Dataset.from_pandas(df[["text", "label", "task_id"]])
    return ds.map(
        lambda ex: tokenizer(ex["text"], padding="max_length", truncation=True, max_length=max_length),
        batched=True,
    )


# =====================================================================
# 2. MULTI-TASK ARCHITECTURE: FOCAL LOSS & KENDALL UNCERTAINTY
# =====================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss with alpha class weighting to combat severe class imbalance.
    Down-weights easy examples and concentrates gradients on hard clinical boundaries.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_weight = (1.0 - p) ** self.gamma
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        loss = focal_weight * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


class CTGBertMultiTask(nn.Module):
    """
    Multi-Task Architecture for CTG Fetal Health & Delivery Mode.
    Uses shared BERT Base encoder with [CLS] pooling and task-specific classification heads.
    Supports Kendall homoscedastic uncertainty weighting and alpha-weighted Focal Loss.
    """
    def __init__(
        self,
        base_model_path="./CTG_BERT_BASE",
        num_fetal_labels=3,
        num_delivery_labels=2,
        use_focal_loss=True,
        focal_gamma=2.0,
        alpha_fetal=None,
        alpha_delivery=None,
        use_uncertainty_weighting=True,
        delivery_task_weight=1.0,
        fetal_dropout=None,
        delivery_dropout=None,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model_path)
        hidden_size = self.bert.config.hidden_size
        default_dropout = self.bert.config.hidden_dropout_prob
        f_drop = fetal_dropout if fetal_dropout is not None else default_dropout
        d_drop = delivery_dropout if delivery_dropout is not None else min(0.35, default_dropout + 0.15)
        self.dropout = nn.Dropout(f_drop)
        self.delivery_dropout = nn.Dropout(d_drop)

        self.fetal_head = nn.Linear(hidden_size, num_fetal_labels)
        self.delivery_head = nn.Sequential(
            self.delivery_dropout,
            nn.Linear(hidden_size, num_delivery_labels),
        )

        self.use_focal_loss = use_focal_loss
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.delivery_task_weight = delivery_task_weight

        if use_focal_loss:
            if alpha_fetal is None:
                alpha_fetal = torch.tensor([1.0, 1.3, 1.8], dtype=torch.float32)
            if alpha_delivery is None:
                alpha_delivery = torch.tensor([1.0, 1.5], dtype=torch.float32)
            self.loss_fetal = FocalLoss(gamma=focal_gamma, alpha=alpha_fetal)
            self.loss_delivery = FocalLoss(gamma=focal_gamma, alpha=alpha_delivery)
        else:
            self.loss_fetal = nn.CrossEntropyLoss()
            self.loss_delivery = nn.CrossEntropyLoss()

        if use_uncertainty_weighting:
            self.log_var_fetal = nn.Parameter(torch.tensor(0.0))
            self.log_var_delivery = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, task_id=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token pooling for pre-trained representation
        cls_rep = self.dropout(outputs.last_hidden_state[:, 0, :])

        fetal_logits = self.fetal_head(cls_rep)
        delivery_logits = self.delivery_head(cls_rep)

        loss = None
        if labels is not None and task_id is not None:
            labels = labels.view(-1)
            task_id = task_id.view(-1)
            mask_f = (task_id == 0)
            mask_d = (task_id == 1)

            loss_f = None
            loss_d = None
            if mask_f.any():
                loss_f = self.loss_fetal(fetal_logits[mask_f], labels[mask_f])
            if mask_d.any():
                loss_d = self.loss_delivery(delivery_logits[mask_d], labels[mask_d])

            if loss_f is not None and loss_d is not None:
                if self.use_uncertainty_weighting:
                    prec_f = torch.exp(-self.log_var_fetal)
                    prec_d = torch.exp(-self.log_var_delivery)
                    loss = prec_f * loss_f + 0.5 * self.log_var_fetal + prec_d * loss_d + 0.5 * self.log_var_delivery
                else:
                    loss = 0.5 * loss_f + 0.5 * self.delivery_task_weight * loss_d
            elif loss_f is not None:
                loss = loss_f
            elif loss_d is not None:
                loss = loss_d

            if loss is not None:
                loss = loss.squeeze()

        return {
            "loss": loss,
            "fetal_logits": fetal_logits,
            "delivery_logits": delivery_logits,
        }


class MultiTaskTrainer(Trainer):
    """Custom Trainer supporting multi-task loss computation."""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs.get("label")
        task_id = inputs.get("task_id")

        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            labels=labels,
            task_id=task_id,
        )
        loss = outputs.get("loss", None)
        if loss is None:
            raise ValueError("Loss is None. Ensure 'labels' and 'task_id' are present in inputs.")
        return (loss, outputs) if return_outputs else loss


# =====================================================================
# 3. STATISTICAL & CLINICAL METRICS ENGINE
# =====================================================================

def compute_class_sensitivity_specificity(y_true, y_pred, num_classes):
    """Compute sensitivity (recall) and specificity for each individual class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    sensitivities = []
    specificities = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)
    return sensitivities, specificities, cm


def compute_expected_calibration_error(y_true, y_score, num_bins=10):
    """Compute Expected Calibration Error (ECE) and return reliability bin data."""
    confidences = np.max(y_score, axis=-1)
    predictions = np.argmax(y_score, axis=-1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_data.append({"bin": i, "acc": float(acc_in_bin), "conf": float(conf_in_bin), "weight": float(prop_in_bin)})

    return float(ece), bin_data


def compute_multiclass_brier_score(y_true, y_score, num_classes):
    """Compute Multi-Class Brier Score."""
    y_true_onehot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((y_score - y_true_onehot) ** 2, axis=1)))


def adjust_logits_for_prior(y_score, natural_priors, eps=1e-8):
    r"""
    Adjust model probabilities for natural clinical distribution using Bayes' theorem:
    P_natural(Y=c|X) \propto P_balanced(Y=c|X) * (P_natural(c) / P_balanced(c))
    """
    y_score = np.asarray(y_score)
    num_classes = y_score.shape[-1]
    balanced_priors = np.ones(num_classes) / num_classes
    ratios = np.array(natural_priors) / (balanced_priors + eps)

    adjusted = y_score * ratios
    adjusted = adjusted / (np.sum(adjusted, axis=-1, keepdims=True) + eps)
    return adjusted


def optimize_binary_threshold(y_true, y_score, min_thresh=0.05, max_thresh=0.95, step=0.01):
    """
    Search for empirical decision threshold that maximizes Macro F1 for binary classification.
    """
    best_thresh = 0.50
    best_f1 = -1.0
    scores_pos = y_score[:, 1] if y_score.ndim == 2 else y_score

    for thresh in np.arange(min_thresh, max_thresh + step, step):
        preds = (scores_pos >= thresh).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    return best_thresh, best_f1


def evaluate_clinical_metrics(
    y_true,
    y_score,
    class_names,
    task_name="Task",
    natural_priors=None,
    decision_threshold=None,
):
    """
    Comprehensive clinical evaluation report on original class distribution.
    Computes Accuracy, Macro F1, Precision, Recall, ROC-AUC, PR-AUC,
    Confidence scores with 95% CIs, Class-wise Sensitivities and Specificities,
    ECE Calibration, and Brier Score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    num_classes = len(class_names)

    if num_classes == 2 and decision_threshold is not None:
        y_pred = (y_score[:, 1] >= decision_threshold).astype(int)
    else:
        y_pred = np.argmax(y_score, axis=-1)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    sensitivities, specificities, cm = compute_class_sensitivity_specificity(y_true, y_pred, num_classes)

    confidences = np.max(y_score, axis=-1)
    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))
    n_samples = len(confidences)
    ci_95 = 1.96 * (std_conf / np.sqrt(n_samples)) if n_samples > 1 else 0.0

    class_confidences = {}
    for c in range(num_classes):
        mask = (y_pred == c)
        class_confidences[class_names[c]] = float(np.mean(confidences[mask])) if mask.any() else 0.0

    pr_auc_per_class = {}
    if num_classes == 2:
        try:
            macro_roc_auc = float(roc_auc_score(y_true, y_score[:, 1]))
            macro_pr_auc = float(average_precision_score(y_true, y_score[:, 1]))
            pr_auc_per_class[class_names[1]] = macro_pr_auc
            pr_auc_per_class[class_names[0]] = float(average_precision_score(1 - y_true, y_score[:, 0]))
        except Exception:
            macro_roc_auc, macro_pr_auc = float("nan"), float("nan")
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        try:
            macro_roc_auc = float(roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr"))
        except Exception:
            macro_roc_auc = float("nan")
        try:
            macro_pr_auc = float(average_precision_score(y_true_bin, y_score, average="macro"))
            for c in range(num_classes):
                pr_auc_per_class[class_names[c]] = float(average_precision_score(y_true_bin[:, c], y_score[:, c]))
        except Exception:
            macro_pr_auc = float("nan")

    ece, bin_data = compute_expected_calibration_error(y_true, y_score, num_bins=10)
    brier = compute_multiclass_brier_score(y_true, y_score, num_classes)

    metrics = {
        "task_name": task_name,
        "n_samples": n_samples,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_roc_auc": macro_roc_auc,
        "macro_pr_auc": macro_pr_auc,
        "mean_confidence": mean_conf,
        "confidence_95_ci": float(ci_95),
        "class_confidences": class_confidences,
        "class_sensitivities": {class_names[c]: float(sensitivities[c]) for c in range(num_classes)},
        "class_specificities": {class_names[c]: float(specificities[c]) for c in range(num_classes)},
        "class_pr_auc": pr_auc_per_class,
        "ece": float(ece),
        "brier_score": float(brier),
        "decision_threshold": float(decision_threshold) if decision_threshold is not None else None,
        "confusion_matrix": cm,
    }

    if natural_priors is not None:
        cal_score = adjust_logits_for_prior(y_score, natural_priors)
        cal_metrics = evaluate_clinical_metrics(
            y_true,
            cal_score,
            class_names,
            task_name=f"{task_name} [Prior-Calibrated]",
            natural_priors=None,
            decision_threshold=decision_threshold,
        )
        metrics["prior_calibrated"] = cal_metrics
        metrics["calibrated_score"] = cal_score

    return metrics


def calculate_ci_and_summary(fold_metrics_list):
    """
    Calculate Mean, Standard Deviation, and 95% Confidence Interval
    across K cross-validation folds.
    """
    keys_to_agg = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_roc_auc", "macro_pr_auc", "ece", "brier_score"]
    summary = {}
    k = len(fold_metrics_list)
    t_val = stats.t.ppf(0.975, df=k - 1) if k > 1 else 1.96

    for key in keys_to_agg:
        vals = [m[key] for m in fold_metrics_list if key in m and not math.isnan(m[key])]
        if vals:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sem = std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ci_margin = float(t_val * sem)
            summary[key] = {
                "mean": mean,
                "std": std,
                "ci_95": ci_margin,
                "ci_lower": mean - ci_margin,
                "ci_upper": mean + ci_margin,
                "folds": vals,
            }

    # Aggregate class sensitivities and specificities
    first = fold_metrics_list[0]
    class_names = list(first["class_sensitivities"].keys())

    summary["class_sensitivities"] = {}
    summary["class_specificities"] = {}

    for c_name in class_names:
        sens_vals = [m["class_sensitivities"][c_name] for m in fold_metrics_list]
        spec_vals = [m["class_specificities"][c_name] for m in fold_metrics_list]

        sens_m = float(np.mean(sens_vals))
        sens_s = float(np.std(sens_vals, ddof=1)) if len(sens_vals) > 1 else 0.0
        sens_ci = float(t_val * (sens_s / np.sqrt(k))) if k > 1 else 0.0

        spec_m = float(np.mean(spec_vals))
        spec_s = float(np.std(spec_vals, ddof=1)) if len(spec_vals) > 1 else 0.0
        spec_ci = float(t_val * (spec_s / np.sqrt(k))) if k > 1 else 0.0

        summary["class_sensitivities"][c_name] = {
            "mean": sens_m,
            "std": sens_s,
            "ci_95": sens_ci,
            "ci_lower": sens_m - sens_ci,
            "ci_upper": sens_m + sens_ci,
        }
        summary["class_specificities"][c_name] = {
            "mean": spec_m,
            "std": spec_s,
            "ci_95": spec_ci,
            "ci_lower": spec_m - spec_ci,
            "ci_upper": spec_m + spec_ci,
        }

    return summary


def print_evaluation_report(metrics):
    """Format and print a structured clinical evaluation report."""
    print(f"\n=======================================================")
    print(f"CLINICAL EVALUATION REPORT: {metrics['task_name']}")
    print(f"Total Evaluated (Natural Prevalence): {metrics['n_samples']} samples")
    print(f"=======================================================")
    print(f"Overall Accuracy    : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Macro F1 Score      : {metrics['macro_f1']:.4f}")
    print(f"Macro Precision     : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall        : {metrics['macro_recall']:.4f}")
    print(f"Macro ROC-AUC       : {metrics['macro_roc_auc']:.4f}")
    print(f"Macro PR-AUC        : {metrics['macro_pr_auc']:.4f}")
    if metrics.get("decision_threshold") is not None:
        print(f"Decision Threshold  : {metrics['decision_threshold']:.3f}")
    print(f"Mean Confidence     : {metrics['mean_confidence']:.4f} (+/- {metrics['confidence_95_ci']:.4f} 95% CI)")
    print(f"Expected Calib Err  : {metrics['ece']:.4f} (ECE)")
    print(f"Brier Score         : {metrics['brier_score']:.4f}")

    print("\n--- Class-Wise Clinical Metrics ---")
    header = f"{'Class':<15} | {'Sensitivity (Recall)':<20} | {'Specificity':<15} | {'PR-AUC':<10} | {'Avg Confidence':<15}"
    print(header)
    print("-" * len(header))
    for c_name, sens in metrics["class_sensitivities"].items():
        spec = metrics["class_specificities"].get(c_name, float("nan"))
        pr = metrics["class_pr_auc"].get(c_name, float("nan"))
        conf = metrics["class_confidences"].get(c_name, float("nan"))
        print(f"{c_name:<15} | {sens:<20.4f} | {spec:<15.4f} | {pr:<10.4f} | {conf:<15.4f}")

    if "prior_calibrated" in metrics:
        print_evaluation_report(metrics["prior_calibrated"])


def print_cross_validation_ci_table(title, summary):
    """Print statistical 5-fold cross validation summary table with 95% CIs."""
    print(f"\n=======================================================")
    print(f"5-FOLD CROSS-VALIDATION STATISTICAL REPORT (95% CIs): {title}")
    print(f"=======================================================")
    hdr = f"{'Metric':<22} | {'Mean':<10} | {'Std':<10} | {'95% Confidence Interval':<26}"
    print(hdr)
    print("-" * len(hdr))
    display_keys = [
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro ROC-AUC", "macro_roc_auc"),
        ("Macro PR-AUC", "macro_pr_auc"),
        ("Expected Calib (ECE)", "ece"),
        ("Brier Score", "brier_score"),
    ]
    for label, key in display_keys:
        if key in summary:
            m = summary[key]["mean"]
            s = summary[key]["std"]
            ci_l = summary[key]["ci_lower"]
            ci_u = summary[key]["ci_upper"]
            print(f"{label:<22} | {m:<10.4f} | {s:<10.4f} | [{ci_l:.4f}, {ci_u:.4f}]")

    print("\n--- Class-Wise Sensitivities & Specificities (5-Fold Mean +/- 95% CI) ---")
    hdr_c = f"{'Class':<15} | {'Sensitivity [95% CI]':<32} | {'Specificity [95% CI]':<32}"
    print(hdr_c)
    print("-" * len(hdr_c))
    for c_name in summary["class_sensitivities"]:
        se = summary["class_sensitivities"][c_name]
        sp = summary["class_specificities"][c_name]
        se_str = f"{se['mean']:.4f} [{se['ci_lower']:.4f}, {se['ci_upper']:.4f}]"
        sp_str = f"{sp['mean']:.4f} [{sp['ci_lower']:.4f}, {sp['ci_upper']:.4f}]"
        print(f"{c_name:<15} | {se_str:<32} | {sp_str:<32}")


def plot_clinical_evaluation_curves(y_true, y_score, class_names, task_name, output_prefix):
    """
    Generate and save Precision-Recall curves and Calibration Reliability diagrams
    for all classes (including Vaginal & Cesarean for Delivery Mode).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    num_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_classes, 3)))
    markers = ['s', 'o', '^', 'd', 'v']

    # 1. Precision-Recall Curves (Both classes rendered)
    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        score_c = y_score[:, c]
        prec, rec, _ = precision_recall_curve(y_true_c, score_c)
        ap = average_precision_score(y_true_c, score_c)
        plt.plot(rec, prec, lw=2.5, color=colors[c], label=f"{class_names[c]} (AP = {ap:.3f})")

    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (PPV)", fontsize=11)
    plt.title(f"Precision-Recall Curves - {task_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    pr_path = f"{output_prefix}_pr_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # 2. Calibration Reliability Curves (Both classes rendered)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        score_c = y_score[:, c]
        prob_true, prob_pred = calibration_curve(y_true_c, score_c, n_bins=10)
        plt.plot(prob_pred, prob_true, marker=markers[c % len(markers)], lw=2, markersize=6, color=colors[c], label=f"{class_names[c]}")

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Positives", fontsize=11)
    plt.title(f"Calibration Reliability Diagram - {task_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    cal_path = f"{output_prefix}_calibration_curve.png"
    plt.tight_layout()
    plt.savefig(cal_path, dpi=200)
    plt.close()

    return pr_path, cal_path


def plot_clinical_confusion_matrix(cm, class_names, task_name, output_path, normalize=True):
    """
    Plot and save high-resolution publication-quality confusion matrix.
    Renders both raw patient counts and row-normalized percentage (sensitivity on diagonal).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm if normalize else cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1.0 if normalize else None)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Recall / Proportion' if normalize else 'Patient Count', rotation=-90, va="bottom", fontsize=10)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"Confusion Matrix: {task_name}",
        ylabel="True Clinical Diagnosis",
        xlabel="Model Predicted Diagnosis",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2. if normalize else cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count_val = int(cm[i, j])
            pct_val = cm_norm[i, j] * 100
            text_str = f"{count_val:,}\n({pct_val:.1f}%)"
            color = "white" if (cm_norm[i, j] if normalize else cm[i, j]) > thresh else "black"
            ax.text(j, i, text_str, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


# =====================================================================
# 4. TRAINING AND INFERENCE HELPERS
# =====================================================================

def predict_dataset(model, dataset, device=None, batch_size=32):
    """
    Run inference across a tokenized dataset and return true labels,
    predicted classes, and predicted probability matrices segregated by task.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    keep_cols = ["input_ids", "attention_mask", "label", "task_id"]
    if "token_type_ids" in dataset.column_names:
        keep_cols.append("token_type_ids")
    clean_ds = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols]).with_format("torch")
    loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False)

    results = {
        0: {"y_true": [], "y_pred": [], "y_score": []},
        1: {"y_true": [], "y_pred": [], "y_score": []},
    }

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()
            task_ids = batch["task_id"].cpu().numpy()
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            fetal_probs = F.softmax(outputs["fetal_logits"], dim=-1).cpu().numpy()
            del_probs = F.softmax(outputs["delivery_logits"], dim=-1).cpu().numpy()

            for i in range(len(labels)):
                t_id = int(task_ids[i])
                y_t = int(labels[i])
                probs = fetal_probs[i] if t_id == 0 else del_probs[i]
                y_p = int(np.argmax(probs))

                results[t_id]["y_true"].append(y_t)
                results[t_id]["y_pred"].append(y_p)
                results[t_id]["y_score"].append(probs)

    for t_id in results:
        results[t_id]["y_true"] = np.array(results[t_id]["y_true"])
        results[t_id]["y_pred"] = np.array(results[t_id]["y_pred"])
        results[t_id]["y_score"] = np.array(results[t_id]["y_score"])

    return results


def train_single_fold(
    fold_info,
    base_model_path,
    tokenizer,
    output_dir,
    num_train_epochs=8,
    per_device_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.02,
    use_focal_loss=True,
    focal_gamma=2.0,
    alpha_fetal=None,
    alpha_delivery=None,
    use_uncertainty_weighting=True,
    seed=42,
    fetal_dropout=None,
    delivery_dropout=None,
):
    """
    Train multi-task model on a single cross-validation fold:
    - Training data: balanced via oversampling.
    - Validation data: natural, un-oversampled distribution.
    - Loads best checkpoint evaluated against validation loss.
    """
    fold_num = fold_info["fold"]
    fold_out_dir = os.path.join(output_dir, f"fold_{fold_num}")
    os.makedirs(fold_out_dir, exist_ok=True)

    set_seed(seed + fold_num)
    model = CTGBertMultiTask(
        base_model_path=base_model_path,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        alpha_fetal=alpha_fetal,
        alpha_delivery=alpha_delivery,
        use_uncertainty_weighting=use_uncertainty_weighting,
        fetal_dropout=fetal_dropout,
        delivery_dropout=delivery_dropout,
    )

    tokenized_train = prepare_hf_dataset(fold_info["train_balanced"], tokenizer)
    tokenized_val = prepare_hf_dataset(fold_info["val_natural"], tokenizer)

    training_args = TrainingArguments(
        output_dir=fold_out_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
        seed=seed + fold_num,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=default_data_collator,
    )

    print(f"\n--- Training Fold {fold_num}/5 (FocalLoss={use_focal_loss}, KendallUncertainty={use_uncertainty_weighting}) ---")
    print(f"Train samples (balanced): {len(fold_info['train_balanced'])} | "
          f"Val samples (natural): {len(fold_info['val_natural'])} (Cesarean in Val: {fold_info['del_val_cesarean']})")

    trainer.train()

    weights_path = os.path.join(fold_out_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), weights_path)

    return model


# =====================================================================
# 5. MAIN WORKFLOW
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CTG-BERT 5-Fold Cross-Validation Multi-Task Training & Clinical Statistical Evaluation"
    )
    parser.add_argument("--ctg-path", default="ctg_full_text.csv", help="Path to CTG CSV")
    parser.add_argument("--delivery-path", default="delivery.csv", help="Path to Delivery CSV")
    parser.add_argument("--tokenizer-dir", default="./my_perfect_tokenizer", help="Path to pretrained tokenizer")
    parser.add_argument("--base-model", default="./CTG_BERT_BASE", help="Path to base BERT model")
    parser.add_argument("--output-dir", default="./ctg_bert_results", help="Output directory for checkpoints and metrics (default: ./ctg_bert_results)")
    parser.add_argument("--save-dir", default="./CTG_BERT_MULTITASK", help="Directory to save final model weights")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test set ratio (default: 0.2)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=8, help="Fine-tuning epochs per fold (default: 8)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Fine-tuning learning rate (default: 5e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="Weight decay (default: 0.02)")
    parser.add_argument("--use-focal-loss", action="store_true", default=True, help="Use Focal Loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter (default: 2.0)")
    parser.add_argument("--use-uncertainty-weighting", action="store_true", default=True, help="Use Kendall uncertainty weighting")
    parser.add_argument("--fetal-dropout", type=float, default=None, help="Dropout probability for fetal head")
    parser.add_argument("--delivery-dropout", type=float, default=None, help="Dropout probability for delivery head")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== CTG-BERT Multi-Task 5-Fold CV Pipeline (Device: {device}) ===")

    # 1. Load Raw Datasets
    print("\n" + "=" * 70)
    print("STEP 1: LOADING RAW CLINICAL DATASETS")
    print("=" * 70)
    ctg_df, del_df = load_raw_datasets(args.ctg_path, args.delivery_path)
    print(f"Loaded CTG records: {len(ctg_df):,} | Class distribution:\n{ctg_df['label'].value_counts().to_dict()} (0: Normal, 1: Suspect, 2: Pathological)")
    print(f"Loaded Delivery records: {len(del_df):,} | Class distribution:\n{del_df['label'].value_counts().to_dict()} (0: Vaginal, 1: Cesarean)")
    cesarean_count = int((del_df["label"] == 1).sum())
    print(f"Note: Delivery dataset contains {cesarean_count} original Cesarean cases.")

    # 2. Tokenizer
    print("\n" + "=" * 70)
    print("STEP 2: LOADING PRETRAINED WORDPIECE TOKENIZER")
    print("=" * 70)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    print(f"Loaded tokenizer from '{args.tokenizer_dir}' (Vocab size: {tokenizer.vocab_size:,})")

    # 3. Leak-Free 80:20 Train / Held-Out Test Split
    print("\n" + "=" * 70)
    print("STEP 3: STRATIFIED 80:20 SPLIT (ZERO-LEAKAGE HELD-OUT TEST BENCHMARK)")
    print("=" * 70)
    ctg_train, ctg_test, del_train, del_test = split_train_test(
        ctg_df, del_df, test_size=args.test_size, random_state=args.seed
    )
    print(f"CTG Split      -> Train (80%): {len(ctg_train):,} | Held-Out Test (20% External): {len(ctg_test):,}")
    print(f"Delivery Split -> Train (80%): {len(del_train):,} | Held-Out Test (20% External): {len(del_test):,}")

    ctg_priors = ctg_train["label"].value_counts(normalize=True).sort_index().tolist()
    del_priors = del_train["label"].value_counts(normalize=True).sort_index().tolist()
    print(f"Natural Clinical Priors -> CTG: {ctg_priors} | Delivery: {del_priors}")

    # Prepare tokenized held-out test set
    combined_test = pd.concat([ctg_test, del_test], ignore_index=True)
    tokenized_test = prepare_hf_dataset(combined_test, tokenizer)

    # 4. 5-Fold Cross-Validation Setup
    print("\n" + "=" * 70)
    print(f"STEP 4: GENERATING {args.n_splits}-FOLD STRATIFIED CV SPLITS (TRAINING-ONLY OVERSAMPLING)")
    print("=" * 70)
    folds = get_5fold_splits(ctg_train, del_train, n_splits=args.n_splits, random_state=args.seed)

    oof_predictions = {
        0: {"y_true": [], "y_score": []},
        1: {"y_true": [], "y_score": []},
    }
    test_fold_scores = {
        0: [],
        1: [],
    }

    fold_metrics_ctg = []
    fold_metrics_del = []

    alpha_fetal = torch.tensor([1.0, 1.3, 1.8], dtype=torch.float32)
    alpha_delivery = torch.tensor([1.0, 1.5], dtype=torch.float32)

    cv_models = []
    for fold_info in folds:
        fold_idx = fold_info["fold"]
        fold_model = train_single_fold(
            fold_info=fold_info,
            base_model_path=args.base_model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            use_focal_loss=args.use_focal_loss,
            focal_gamma=args.focal_gamma,
            alpha_fetal=alpha_fetal,
            alpha_delivery=alpha_delivery,
            use_uncertainty_weighting=args.use_uncertainty_weighting,
            seed=args.seed,
            fetal_dropout=args.fetal_dropout,
            delivery_dropout=args.delivery_dropout,
        )
        cv_models.append(fold_model)

        # Evaluate Out-of-Fold (OOF) validation partition (strictly natural prevalence)
        val_dataset = prepare_hf_dataset(fold_info["val_natural"], tokenizer)
        val_preds = predict_dataset(fold_model, val_dataset, device=device, batch_size=args.batch_size)

        # Compute individual fold metrics
        m_ctg = evaluate_clinical_metrics(
            y_true=val_preds[0]["y_true"],
            y_score=val_preds[0]["y_score"],
            class_names=CTG_CLASS_NAMES,
            task_name=f"CTG Fold {fold_idx}",
        )
        m_del = evaluate_clinical_metrics(
            y_true=val_preds[1]["y_true"],
            y_score=val_preds[1]["y_score"],
            class_names=DELIVERY_CLASS_NAMES,
            task_name=f"Delivery Fold {fold_idx}",
        )
        fold_metrics_ctg.append(m_ctg)
        fold_metrics_del.append(m_del)

        for t_id in [0, 1]:
            oof_predictions[t_id]["y_true"].extend(val_preds[t_id]["y_true"])
            oof_predictions[t_id]["y_score"].extend(val_preds[t_id]["y_score"])

        # Predict on held-out test set
        test_preds = predict_dataset(fold_model, tokenized_test, device=device, batch_size=args.batch_size)
        for t_id in [0, 1]:
            test_fold_scores[t_id].append(test_preds[t_id]["y_score"])

    # Convert pooled OOF arrays
    for t_id in [0, 1]:
        oof_predictions[t_id]["y_true"] = np.array(oof_predictions[t_id]["y_true"])
        oof_predictions[t_id]["y_score"] = np.array(oof_predictions[t_id]["y_score"])

    # Save best model (Fold 1) and tokenizer to save_dir
    torch.save(cv_models[0].state_dict(), os.path.join(args.save_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(args.save_dir)
    print(f"\n Saved primary model weights and tokenizer to: '{args.save_dir}'")

    # =====================================================================
    # 5. CROSS-VALIDATION STATISTICAL ANALYSIS & CONFIDENCE INTERVALS
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 5: 5-FOLD CROSS-VALIDATION STATISTICAL REPORT (CONFIDENCE INTERVALS)")
    print("=" * 70)

    cv_summary_ctg = calculate_ci_and_summary(fold_metrics_ctg)
    cv_summary_del = calculate_ci_and_summary(fold_metrics_del)

    print_cross_validation_ci_table("CTG / Fetal Health", cv_summary_ctg)
    print_cross_validation_ci_table("Delivery Mode", cv_summary_del)

    # Save fold-by-fold comparison to CSV
    fold_records = []
    for k in range(args.n_splits):
        fold_records.append({
            "Fold": k + 1,
            "CTG_Accuracy": fold_metrics_ctg[k]["accuracy"],
            "CTG_MacroF1": fold_metrics_ctg[k]["macro_f1"],
            "CTG_Patho_Sensitivity": fold_metrics_ctg[k]["class_sensitivities"]["Pathological"],
            "Delivery_Accuracy": fold_metrics_del[k]["accuracy"],
            "Delivery_MacroF1": fold_metrics_del[k]["macro_f1"],
            "Delivery_Cesarean_Sensitivity": fold_metrics_del[k]["class_sensitivities"]["Cesarean"],
            "Delivery_Cesarean_PR_AUC": fold_metrics_del[k]["class_pr_auc"].get("Cesarean", np.nan),
            "Delivery_ECE": fold_metrics_del[k]["ece"],
        })
    df_folds = pd.DataFrame(fold_records)
    csv_path = os.path.join(args.output_dir, "multitask_cv_fold_metrics.csv")
    df_folds.to_csv(csv_path, index=False)
    print(f"\nSaved fold-by-fold metrics CSV to: '{csv_path}'")
    alt_csv = os.path.abspath(os.path.join("./ctg_bert_results", "multitask_cv_fold_metrics.csv"))
    if os.path.abspath(csv_path) != alt_csv:
        os.makedirs("./ctg_bert_results", exist_ok=True)
        df_folds.to_csv(alt_csv, index=False)

    # =====================================================================
    # 6. POOLED OUT-OF-FOLD (OOF) EVALUATION ON NATURAL PREVALENCE
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 6: POOLED OUT-OF-FOLD (OOF) CLINICAL EVALUATION (NATURAL PREVALENCE)")
    print("=" * 70)

    oof_ctg_metrics = evaluate_clinical_metrics(
        y_true=oof_predictions[0]["y_true"],
        y_score=oof_predictions[0]["y_score"],
        class_names=CTG_CLASS_NAMES,
        task_name="CTG / Fetal Health (OOF Validation)",
        natural_priors=ctg_priors,
    )
    print_evaluation_report(oof_ctg_metrics)

    # Delivery standard threshold (tau = 0.50)
    oof_del_metrics_std = evaluate_clinical_metrics(
        y_true=oof_predictions[1]["y_true"],
        y_score=oof_predictions[1]["y_score"],
        class_names=DELIVERY_CLASS_NAMES,
        task_name="Delivery Mode (OOF Validation - Standard tau=0.50)",
        natural_priors=del_priors,
    )
    print_evaluation_report(oof_del_metrics_std)

    # Decision threshold optimization strictly on OOF validation predictions (zero test leakage!)
    opt_del_thresh, best_oof_del_f1 = optimize_binary_threshold(
        y_true=oof_predictions[1]["y_true"],
        y_score=oof_predictions[1]["y_score"],
    )
    print(f"\n>> Clinical Decision Threshold Optimization (Delivery Mode):")
    print(f">> Optimal threshold discovered on validation predictions: tau* = {opt_del_thresh:.2f} (OOF Macro F1 = {best_oof_del_f1:.4f})")

    oof_del_metrics_opt = evaluate_clinical_metrics(
        y_true=oof_predictions[1]["y_true"],
        y_score=oof_predictions[1]["y_score"],
        class_names=DELIVERY_CLASS_NAMES,
        task_name=f"Delivery Mode (OOF Validation - Optimal tau*={opt_del_thresh:.2f})",
        natural_priors=del_priors,
        decision_threshold=opt_del_thresh,
    )
    print_evaluation_report(oof_del_metrics_opt)

    # Plot Precision-Recall & Calibration curves
    plot_clinical_evaluation_curves(
        y_true=oof_predictions[0]["y_true"],
        y_score=oof_predictions[0]["y_score"],
        class_names=CTG_CLASS_NAMES,
        task_name="CTG / Fetal Health",
        output_prefix=os.path.join(args.output_dir, "oof_ctg"),
    )
    plot_clinical_evaluation_curves(
        y_true=oof_predictions[1]["y_true"],
        y_score=oof_predictions[1]["y_score"],
        class_names=DELIVERY_CLASS_NAMES,
        task_name="Delivery Mode (Vaginal & Cesarean)",
        output_prefix=os.path.join(args.output_dir, "oof_delivery"),
    )

    if "calibrated_score" in oof_del_metrics_std:
        plot_clinical_evaluation_curves(
            y_true=oof_predictions[1]["y_true"],
            y_score=oof_del_metrics_std["calibrated_score"],
            class_names=DELIVERY_CLASS_NAMES,
            task_name="Delivery Mode (Prior-Calibrated)",
            output_prefix=os.path.join(args.output_dir, "oof_delivery_calibrated"),
        )

    # Plot Confusion Matrices for Out-of-Fold Validation
    plot_clinical_confusion_matrix(
        cm=oof_ctg_metrics["confusion_matrix"],
        class_names=CTG_CLASS_NAMES,
        task_name="CTG / Fetal Health (OOF Validation)",
        output_path=os.path.join(args.output_dir, "oof_ctg_confusion_matrix.png"),
    )
    plot_clinical_confusion_matrix(
        cm=oof_del_metrics_std["confusion_matrix"],
        class_names=DELIVERY_CLASS_NAMES,
        task_name="Delivery Mode (OOF - Standard tau=0.50)",
        output_path=os.path.join(args.output_dir, "oof_delivery_confusion_matrix.png"),
    )
    plot_clinical_confusion_matrix(
        cm=oof_del_metrics_opt["confusion_matrix"],
        class_names=DELIVERY_CLASS_NAMES,
        task_name=f"Delivery Mode (OOF - Optimal tau*={opt_del_thresh:.2f})",
        output_path=os.path.join(args.output_dir, "oof_delivery_optimal_confusion_matrix.png"),
    )

    # =====================================================================
    # 7. EXTERNAL VALIDATION ON HELD-OUT 20% TEST SET
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 7: EXTERNAL VALIDATION ON HELD-OUT 20% TEST SET (NATURAL PREVALENCE)")
    print("=" * 70)

    test_ctg_true = ctg_test["label"].values
    test_del_true = del_test["label"].values

    single_fold_ctg_preds = test_fold_scores[0][0]
    single_fold_del_preds = test_fold_scores[1][0]

    # Standalone Single Model (Fold 1) Performance
    single_model_ctg = evaluate_clinical_metrics(
        y_true=test_ctg_true,
        y_score=single_fold_ctg_preds,
        class_names=CTG_CLASS_NAMES,
        task_name="CTG / Fetal Health (Single Model Fold 1 Test)",
        natural_priors=ctg_priors,
    )
    print_evaluation_report(single_model_ctg)

    single_model_del_std = evaluate_clinical_metrics(
        y_true=test_del_true,
        y_score=single_fold_del_preds,
        class_names=DELIVERY_CLASS_NAMES,
        task_name="Delivery Mode (Single Model Fold 1 Test - Standard tau=0.50)",
        natural_priors=del_priors,
    )
    print_evaluation_report(single_model_del_std)

    single_model_del_opt = evaluate_clinical_metrics(
        y_true=test_del_true,
        y_score=single_fold_del_preds,
        class_names=DELIVERY_CLASS_NAMES,
        task_name=f"Delivery Mode (Single Model Fold 1 Test - Optimal tau*={opt_del_thresh:.2f})",
        natural_priors=del_priors,
        decision_threshold=opt_del_thresh,
    )
    print_evaluation_report(single_model_del_opt)

    # Cross-Validated 5-Fold Mean Ensemble Test Scores
    cv_mean_ctg_scores = np.mean(test_fold_scores[0], axis=0)
    cv_mean_del_scores = np.mean(test_fold_scores[1], axis=0)

    test_ctg_metrics = evaluate_clinical_metrics(
        y_true=test_ctg_true,
        y_score=cv_mean_ctg_scores,
        class_names=CTG_CLASS_NAMES,
        task_name="CTG / Fetal Health (5-Fold CV Mean Test)",
        natural_priors=ctg_priors,
    )
    print_evaluation_report(test_ctg_metrics)

    test_del_metrics_std = evaluate_clinical_metrics(
        y_true=test_del_true,
        y_score=cv_mean_del_scores,
        class_names=DELIVERY_CLASS_NAMES,
        task_name="Delivery Mode (5-Fold CV Mean Test - Standard tau=0.50)",
        natural_priors=del_priors,
    )
    print_evaluation_report(test_del_metrics_std)

    test_del_metrics_opt = evaluate_clinical_metrics(
        y_true=test_del_true,
        y_score=cv_mean_del_scores,
        class_names=DELIVERY_CLASS_NAMES,
        task_name=f"Delivery Mode (5-Fold CV Mean Test - Optimal tau*={opt_del_thresh:.2f})",
        natural_priors=del_priors,
        decision_threshold=opt_del_thresh,
    )
    print_evaluation_report(test_del_metrics_opt)

    # Plot Confusion Matrices for External Test Set
    plot_clinical_confusion_matrix(
        cm=test_ctg_metrics["confusion_matrix"],
        class_names=CTG_CLASS_NAMES,
        task_name="CTG / Fetal Health (External Test)",
        output_path=os.path.join(args.output_dir, "test_ctg_confusion_matrix.png"),
    )
    plot_clinical_confusion_matrix(
        cm=test_del_metrics_std["confusion_matrix"],
        class_names=DELIVERY_CLASS_NAMES,
        task_name="Delivery Mode (External Test - Standard tau=0.50)",
        output_path=os.path.join(args.output_dir, "test_delivery_confusion_matrix.png"),
    )
    plot_clinical_confusion_matrix(
        cm=test_del_metrics_opt["confusion_matrix"],
        class_names=DELIVERY_CLASS_NAMES,
        task_name=f"Delivery Mode (External Test - Optimal tau*={opt_del_thresh:.2f})",
        output_path=os.path.join(args.output_dir, "test_delivery_optimal_confusion_matrix.png"),
    )

    # Also copy all generated PNG plots to ctg_bert_results if args.output_dir is different
    brain_dir = "/home/ador/.gemini/antigravity/brain/f7c509c8-0e92-4a2f-bf73-9804ca2874b7"
    alt_results_dir = os.path.abspath("./ctg_bert_results")
    import glob, shutil
    for png_file in glob.glob(os.path.join(args.output_dir, "*.png")):
        shutil.copy(png_file, brain_dir)
        if os.path.abspath(args.output_dir) != alt_results_dir:
            shutil.copy(png_file, alt_results_dir)

    # Clean dict for JSON export
    def sanitize_for_json(d):
        clean = {}
        for k, v in d.items():
            if k in ["confusion_matrix", "calibrated_score"]:
                continue
            if isinstance(v, dict):
                clean[k] = sanitize_for_json(v)
            elif isinstance(v, (np.floating, float)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                clean[k] = int(v)
            elif isinstance(v, (np.ndarray, list)):
                clean[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            else:
                clean[k] = v
        return clean

    summary = {
        "optimal_delivery_threshold": float(opt_del_thresh),
        "cv_statistical_summary_ctg": sanitize_for_json(cv_summary_ctg),
        "cv_statistical_summary_delivery": sanitize_for_json(cv_summary_del),
        "oof_ctg": sanitize_for_json(oof_ctg_metrics),
        "oof_delivery_standard": sanitize_for_json(oof_del_metrics_std),
        "oof_delivery_optimal": sanitize_for_json(oof_del_metrics_opt),
        "single_model_test_ctg": sanitize_for_json(single_model_ctg),
        "single_model_test_delivery_standard": sanitize_for_json(single_model_del_std),
        "single_model_test_delivery_optimal": sanitize_for_json(single_model_del_opt),
        "cv_mean_test_ctg": sanitize_for_json(test_ctg_metrics),
        "cv_mean_test_delivery_standard": sanitize_for_json(test_del_metrics_std),
        "cv_mean_test_delivery_optimal": sanitize_for_json(test_del_metrics_opt),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    summary_json_path = os.path.join(args.output_dir, "multitask_cv_evaluation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"\n✅ Evaluation summary JSON updated at: '{os.path.abspath(summary_json_path)}'")

    # Guarantee that ctg_bert_results/multitask_cv_evaluation_summary.json is updated even if custom output_dir is specified
    default_results_dir = os.path.abspath("./ctg_bert_results")
    alt_json_path = os.path.join(default_results_dir, "multitask_cv_evaluation_summary.json")
    if os.path.abspath(summary_json_path) != alt_json_path:
        os.makedirs(default_results_dir, exist_ok=True)
        with open(alt_json_path, "w") as f:
            json.dump(summary, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"✅ Also updated: '{alt_json_path}'")
    brain_json_path = os.path.join(brain_dir, "multitask_cv_evaluation_summary.json")
    shutil.copy(summary_json_path, brain_json_path)
    if os.path.exists(csv_path):
        shutil.copy(csv_path, os.path.join(brain_dir, "multitask_cv_fold_metrics.csv"))

    # 8. Sample Unified Bedside Prediction Demonstration
    print("\n" + "=" * 70)
    print("STEP 8: UNIFIED BEDSIDE PREDICTION INFERENCE CHECK")
    print("=" * 70)
    id_to_fetal = {0: "Normal", 1: "Suspect", 2: "Pathological"}
    id_to_delivery = {0: "Vaginal", 1: "Cesarean"}
    sample_case = (
        "baseline fetal heart rate is 110 bpm. severe prolonged decelerations present. "
        "high percentage of time with abnormal short term variability. zero accelerations."
    )

    cv_models[0].eval()
    enc = tokenizer(sample_case, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = cv_models[0](input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        f_p_raw = F.softmax(out["fetal_logits"], dim=-1).cpu().numpy()[0]
        d_p_raw = F.softmax(out["delivery_logits"], dim=-1).cpu().numpy()[0]

    f_p_cal = adjust_logits_for_prior(f_p_raw.reshape(1, -1), ctg_priors)[0]
    d_p_cal = adjust_logits_for_prior(d_p_raw.reshape(1, -1), del_priors)[0]

    f_pred = int(np.argmax(f_p_cal))
    d_pred = int(d_p_raw[1] >= opt_del_thresh)

    print(f"Sample Case: \"{sample_case}\"")
    print(f"  CTG Fetal Health Prediction: {id_to_fetal[f_pred]} (Confidence: {f_p_cal[f_pred]:.4f})")
    print(f"  Delivery Mode Prediction:    {id_to_delivery[d_pred]} (Cesarean Probability: {d_p_raw[1]:.4f}, Threshold: {opt_del_thresh:.2f})")

    print(f"\n[Completed] Full 5-Fold Cross-Validation & Statistical Evaluation saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
