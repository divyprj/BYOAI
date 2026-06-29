#!/usr/bin/env python3
"""
Fine-Tuning Script for BYOAI Intent Classification.

Loads preprocessed HuggingFace Datasets, fine-tunes DistilBERT for sequence
classification using the HuggingFace Trainer API, and persists the best model.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("byoai.fine_tune")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models", "best_model")
DEFAULT_MODEL_NAME = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute classification metrics for the Trainer.

    Computes accuracy, macro precision, macro recall, and macro F1.

    Args:
        eval_pred: ``EvalPrediction`` namedtuple with ``predictions`` and
            ``label_ids`` arrays.

    Returns:
        Dictionary of metric names to float values.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_label_mapping(processed_dir: str) -> Dict[str, Any]:
    """Load the label mapping produced by the preprocessing pipeline.

    Args:
        processed_dir: Directory containing ``label_mapping.json``.

    Returns:
        Dictionary with ``id2label`` and ``label2id`` mappings.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
    """
    mapping_path = Path(processed_dir) / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping not found at {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as fh:
        mapping = json.load(fh)

    logger.info("Loaded label mapping with %d classes", len(mapping["id2label"]))
    return mapping


# ---------------------------------------------------------------------------
# Fine-Tuning
# ---------------------------------------------------------------------------
def fine_tune(
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    warmup_steps: int = 50,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 3,
    fp16: bool = False,
) -> str:
    """Fine-tune DistilBERT on the preprocessed intent dataset.

    Args:
        processed_dir: Directory with preprocessed DatasetDict and label mapping.
        output_dir: Where to save the best model and tokenizer.
        model_name: Pretrained model identifier.
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate for AdamW.
        batch_size: Per-device batch size for train and eval.
        warmup_steps: Linear warmup steps.
        weight_decay: Weight decay coefficient.
        early_stopping_patience: Epochs to wait before early stopping.
        fp16: Whether to use mixed-precision training.

    Returns:
        Path to the saved model directory.
    """
    logger.info("=" * 60)
    logger.info("BYOAI Fine-Tuning Pipeline")
    logger.info("=" * 60)

    # 1. Load processed data
    dataset_path = os.path.join(processed_dir, "dataset")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. "
            "Run preprocess.py first."
        )
    logger.info("Loading processed dataset from %s", dataset_path)
    dataset_dict: DatasetDict = load_from_disk(dataset_path)
    logger.info("Dataset splits: %s", list(dataset_dict.keys()))

    # 2. Label mapping
    mapping = load_label_mapping(processed_dir)
    id2label: Dict[str, str] = mapping["id2label"]
    label2id: Dict[str, int] = mapping["label2id"]
    num_labels = len(id2label)
    logger.info("Number of labels: %d", num_labels)

    # 3. Load tokenizer and model
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id={k: int(v) for k, v in label2id.items()},
    )
    logger.info(
        "Model loaded - parameters: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        fp16=fp16,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )
    logger.info("TrainingArguments configured: epochs=%d, lr=%s, batch=%d", num_epochs, learning_rate, batch_size)

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # 6. Train
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # 7. Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(dataset_dict["test"])
    logger.info("Test metrics: %s", test_metrics)

    # 8. Save best model and tokenizer
    best_model_dir = os.path.join(output_dir, "final")
    Path(best_model_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info("Best model and tokenizer saved to %s", best_model_dir)

    # 9. Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    all_metrics = {
        "train": train_result.metrics,
        "test": test_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info("Training metrics saved to %s", metrics_path)

    logger.info("=" * 60)
    logger.info("Fine-tuning complete!")
    logger.info("=" * 60)
    return best_model_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed Namespace.
    """
    parser = argparse.ArgumentParser(
        description="BYOAI DistilBERT Fine-Tuning for Intent Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory with preprocessed datasets and label mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Pretrained HuggingFace model name.",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Per-device batch size."
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=50, help="Warmup steps."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay."
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed-precision (fp16) training.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        fine_tune(
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
            fp16=args.fp16,
        )
    except Exception as exc:
        logger.exception("Fine-tuning failed: %s", exc)
        sys.exit(1)
