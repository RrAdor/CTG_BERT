#!/usr/bin/env python3
"""
Tokenizer Training & Masked Language Modeling (MLM) Pretraining for CTG-BERT.

Replicates and automates the exact steps from CTG_BERT.ipynb:
1. Trains a custom WordPiece tokenizer on the combined corpus of full CTG and Delivery datasets.
2. Saves the tokenizer to './my_perfect_tokenizer'.
3. Configures a custom BERT architecture (hidden_size=128, 4 layers, 4 heads, max_pos=128).
4. Prepares the tokenized MLM dataset (15% random masking).
5. Runs Masked Language Modeling (MLM) pretraining with HuggingFace Trainer.
6. Saves the pretrained base model to './CTG_BERT_BASE'.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


def find_file(possible_names, base_dir=Path(".")):
    """Locate the first existing file from candidate names."""
    for name in possible_names:
        p = base_dir / name
        if p.is_file():
            return p
    return None


def load_combined_corpus(
    ctg_path=None,
    delivery_path=None,
    delivery_test_size=0.2,
    seed=42,
    base_dir=Path("."),
):
    """
    Load text from both datasets:
    - Tokenizer training: uses full combined corpus (all CTG and all Delivery text).
    - MLM pre-training: uses all CTG text (remains unchanged) and ONLY the 80% training split for Delivery.
    """
    if ctg_path is None:
        ctg_path = find_file(["ctg_full_text_balanced.csv"], base_dir)
    else:
        ctg_path = Path(ctg_path)

    if delivery_path is None:
        delivery_path = find_file(["delivery.csv"], base_dir)
    else:
        delivery_path = Path(delivery_path)

    if not ctg_path or not ctg_path.is_file():
        raise FileNotFoundError(f"Could not find CTG dataset at {ctg_path}")
    if not delivery_path or not delivery_path.is_file():
        raise FileNotFoundError(f"Could not find Delivery dataset at {delivery_path}")

    print(f"Loading CTG data from:      {ctg_path}")
    print(f"Loading Delivery data from: {delivery_path}")

    ctg_df = pd.read_csv(ctg_path)
    deliv_df = pd.read_csv(delivery_path)

    # CTG texts remain exactly as they are now (all records)
    ctg_texts = ctg_df["clinical_text"].dropna().astype(str).str.lower().tolist()
    deliv_texts_full = deliv_df["caption"].dropna().astype(str).str.lower().tolist()

    # Stratified 80:20 split for Delivery dataset only
    stratify_col = deliv_df["delivery_type"] if "delivery_type" in deliv_df.columns else None
    deliv_train_df, deliv_test_df = train_test_split(
        deliv_df,
        test_size=delivery_test_size,
        random_state=seed,
        stratify=stratify_col,
    )
    deliv_train_texts = deliv_train_df["caption"].dropna().astype(str).str.lower().tolist()

    # Tokenizer gets full corpus to ensure complete vocabulary coverage
    tokenizer_texts = ctg_texts + deliv_texts_full

    # MLM Pre-training gets all CTG + only 80% training split of Delivery
    mlm_texts = ctg_texts + deliv_train_texts

    print(f"\n--- Dataset Loading & Pre-training Split Summary ---")
    print(f"  CTG samples (full / unchanged):     {len(ctg_texts):,}")
    print(f"  Delivery samples (full):            {len(deliv_texts_full):,}")
    print(f"    • Delivery 80% train split:       {len(deliv_train_texts):,} (used in MLM pretraining)")
    print(f"    • Delivery 20% test split:        {len(deliv_test_df):,} (held out, excluded from MLM)")
    print(f"  Tokenizer training corpus size:     {len(tokenizer_texts):,} samples")
    print(f"  MLM pretraining corpus size:        {len(mlm_texts):,} samples\n")

    return tokenizer_texts, mlm_texts


def train_wordpiece_tokenizer(
    texts,
    save_dir="./my_perfect_tokenizer",
    vocab_size=3000,
):
    """Train custom WordPiece tokenizer matching CTG_BERT.ipynb Cell 0."""
    print("\n" + "=" * 60)
    print("STEP 1: BUILDING & TRAINING WORDPIECE TOKENIZER")
    print("=" * 60)

    # Initialize WordPiece model with [UNK]
    raw_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # Normalizer: lowercase
    raw_tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase()])

    # Pre-tokenizer: whitespace splitting
    raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Special tokens matching BERT
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    print(f"Training WordPiece tokenizer on {len(texts):,} samples (vocab target: {vocab_size})...")
    raw_tokenizer.train_from_iterator(texts, trainer=trainer)

    # Wrap for HuggingFace Transformers compatibility
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # Save tokenizer
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    wrapped_tokenizer.save_pretrained(str(save_path))
    print(f"✅ Tokenizer saved to '{save_dir}'. Final vocabulary size: {wrapped_tokenizer.vocab_size}")

    # Sanity checks from notebook
    test_ctg = "Baseline fetal heart rate is 110 bpm. Severe prolonged decelerations present."
    test_delivery = "Presentation breech. First stage duration 240 minutes. Late decelerations observed."
    ctg_toks = wrapped_tokenizer.tokenize(test_ctg.lower())
    deliv_toks = wrapped_tokenizer.tokenize(test_delivery.lower())

    print("\nSanity Check:")
    print("  CTG Tokens:     ", ctg_toks)
    print("  Delivery Tokens:", deliv_toks)
    print("  CTG [UNK] count:     ", ctg_toks.count("[UNK]"))
    print("  Delivery [UNK] count:", deliv_toks.count("[UNK]"))

    return wrapped_tokenizer


def run_mlm_pretraining(
    texts,
    tokenizer,
    output_dir="./ctg_bert_training_shared",
    model_save_dir="./CTG_BERT_BASE",
    num_train_epochs=100,
    per_device_train_batch_size=16,
    learning_rate=1e-4,
    max_length=128,
    save_total_limit=2,
    seed=42,
):
    """Configure BERT architecture and train via Masked Language Modeling."""
    print("\n" + "=" * 60)
    print("STEP 2: CONFIGURING SHARED CTG-BERT ARCHITECTURE")
    print("=" * 60)

    set_seed(seed)

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=max_length,
    )
    model = BertForMaskedLM(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Shared CTG-BERT born! Trainable parameters: {num_params:,}")

    print("\n" + "=" * 60)
    print("STEP 3: PREPARING MLM DATASET & DATA COLLATOR")
    print("=" * 60)

    raw_dataset = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print("Tokenizing corpus...")
    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    print("\n" + "=" * 60)
    print("STEP 4: STARTING SHARED MLM PRE-TRAINING")
    print("=" * 60)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=50,
        learning_rate=learning_rate,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    print(f"Training for {num_train_epochs} epochs (batch size: {per_device_train_batch_size}, lr: {learning_rate})...")
    train_result = trainer.train()

    # Save final pretrained base model & tokenizer
    save_path = Path(model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    print("\n" + "=" * 60)
    print("PRETRAINING COMPLETE!")
    print("=" * 60)
    print(f"✅ Pretrained base model saved to: '{model_save_dir}'")
    print(f"   Final Training Loss: {train_result.training_loss:.4f}")
    print(f"   Total Train Runtime: {train_result.metrics.get('train_runtime', 0):.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Train WordPiece Tokenizer and run MLM Pretraining for CTG-BERT."
    )
    parser.add_argument("--ctg-path", type=str, default=None, help="Path to ctg_full_text_balanced.csv")
    parser.add_argument("--delivery-path", type=str, default=None, help="Path to delivery.csv")
    parser.add_argument("--tokenizer-dir", type=str, default="./my_perfect_tokenizer", help="Path to save tokenizer")
    parser.add_argument("--model-dir", type=str, default="./CTG_BERT_BASE", help="Path to save pretrained model")
    parser.add_argument("--checkpoint-dir", type=str, default="./ctg_bert_training_shared", help="Checkpoint directory")
    parser.add_argument("--vocab-size", type=int, default=3000, help="Vocabulary size (default: 3000)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--delivery-test-size", type=float, default=0.2, help="Delivery held-out test split ratio (default: 0.2 for 80%% train split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--skip-mlm", action="store_true", help="Only train tokenizer, skip MLM pretraining")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Max checkpoints to keep on disk (default: 2)")

    args = parser.parse_args()

    # 1. Load full text corpus for tokenizer and 80% split delivery + unchanged CTG for MLM
    tokenizer_texts, mlm_texts = load_combined_corpus(
        ctg_path=args.ctg_path,
        delivery_path=args.delivery_path,
        delivery_test_size=args.delivery_test_size,
        seed=args.seed,
    )

    # 2. Build & train WordPiece tokenizer
    tokenizer = train_wordpiece_tokenizer(
        texts=tokenizer_texts,
        save_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
    )

    # 3. Masked Language Modeling Pre-training
    if not args.skip_mlm:
        run_mlm_pretraining(
            texts=mlm_texts,
            tokenizer=tokenizer,
            output_dir=args.checkpoint_dir,
            model_save_dir=args.model_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
