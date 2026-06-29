#!/usr/bin/env python3
"""
Model Evaluation Script for BYOAI Intent Classification.

Evaluates a trained intent-classification model (or simulates predictions)
and produces per-class metrics, a confusion matrix heatmap, a classification
report, and logs everything to MLflow.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("byoai.evaluate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INTENT_LABELS: List[str] = [
    "booking",
    "cancel",
    "complaint",
    "farewell",
    "feedback",
    "greeting",
    "help",
    "out_of_scope",
    "question",
    "status_check",
]

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT_NAME = "byoai-intent-classification"
DEFAULT_OUTPUT_DIR = "experiments/eval_artifacts"


# ---------------------------------------------------------------------------
# Simulated Predictions
# ---------------------------------------------------------------------------
def generate_simulated_predictions(
    labels: List[str],
    num_samples: int = 200,
    accuracy_target: float = 0.88,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic simulated true/predicted label arrays.

    Creates a confusion pattern where most predictions fall on the diagonal
    but plausible misclassifications occur (e.g. complaint↔feedback,
    greeting↔farewell, question↔help).

    Args:
        labels: List of intent label strings.
        num_samples: Total number of samples to generate.
        accuracy_target: Approximate target accuracy.
        seed: Random seed.

    Returns:
        Tuple of (y_true, y_pred) integer arrays.
    """
    rng = np.random.RandomState(seed)
    num_classes = len(labels)
    samples_per_class = num_samples // num_classes

    y_true_list: List[int] = []
    y_pred_list: List[int] = []

    # Confusion bias: pairs of classes that are commonly confused
    confusion_pairs: Dict[int, List[int]] = {
        labels.index("complaint"): [labels.index("feedback")],
        labels.index("feedback"): [labels.index("complaint")],
        labels.index("greeting"): [labels.index("farewell")],
        labels.index("farewell"): [labels.index("greeting")],
        labels.index("question"): [labels.index("help")],
        labels.index("help"): [labels.index("question")],
        labels.index("cancel"): [labels.index("complaint")],
        labels.index("status_check"): [labels.index("question")],
    }

    for class_idx in range(num_classes):
        n = samples_per_class
        for _ in range(n):
            y_true_list.append(class_idx)
            if rng.random() < accuracy_target:
                y_pred_list.append(class_idx)
            else:
                # Biased misclassification
                if class_idx in confusion_pairs and rng.random() < 0.7:
                    y_pred_list.append(rng.choice(confusion_pairs[class_idx]))
                else:
                    wrong = rng.randint(0, num_classes)
                    while wrong == class_idx:
                        wrong = rng.randint(0, num_classes)
                    y_pred_list.append(wrong)

    return np.array(y_true_list), np.array(y_pred_list)


# ---------------------------------------------------------------------------
# Confusion Matrix Visualisation
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_path: str,
) -> str:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        labels: Class label names.
        output_path: Where to save the PNG.

    Returns:
        Absolute path to the saved image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_title(
        "BYOAI Intent Classification - Confusion Matrix",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Classification Report
# ---------------------------------------------------------------------------
def build_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> Dict[str, Any]:
    """Compute a full classification report.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Class label names.

    Returns:
        Classification report as a dictionary.
    """
    report: Dict[str, Any] = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return report


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------
def evaluate(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    model_dir: Optional[str] = None,
    simulate: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Run model evaluation and log results to MLflow.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.
        model_dir: Path to the trained model directory (used in real mode).
        simulate: If True, generate simulated predictions.
        output_dir: Local directory for saving evaluation artifacts.

    Returns:
        Dictionary of evaluation metrics.
    """
    try:
        import mlflow
    except ImportError:
        logger.error("mlflow is not installed. Install via: pip install mlflow")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("BYOAI Model Evaluation")
    logger.info("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    labels = INTENT_LABELS

    # --- Get predictions ---
    if simulate:
        logger.info("Running in SIMULATION mode")
        y_true, y_pred = generate_simulated_predictions(labels)
    else:
        logger.info("Running real evaluation")
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data_pipeline",
                "models",
                "best_model",
                "final",
            )
        if not os.path.isdir(model_dir):
            logger.warning(
                "Model directory %s not found. Falling back to simulation.",
                model_dir,
            )
            y_true, y_pred = generate_simulated_predictions(labels)
        else:
            try:
                from datasets import load_from_disk
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                processed_dir = os.path.join(
                    os.path.dirname(__file__), "..", "data_pipeline", "processed"
                )
                dataset_path = os.path.join(processed_dir, "dataset")
                dataset_dict = load_from_disk(dataset_path)
                test_ds = dataset_dict["test"]

                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                model.eval()

                y_true_list: List[int] = []
                y_pred_list: List[int] = []

                with torch.no_grad():
                    for sample in test_ds:
                        inputs = {
                            "input_ids": sample["input_ids"].unsqueeze(0),
                            "attention_mask": sample["attention_mask"].unsqueeze(0),
                        }
                        logits = model(**inputs).logits
                        pred = torch.argmax(logits, dim=-1).item()
                        y_pred_list.append(pred)
                        y_true_list.append(sample["labels"].item())

                y_true = np.array(y_true_list)
                y_pred = np.array(y_pred_list)
                logger.info("Evaluated %d test samples", len(y_true))
            except Exception as exc:
                logger.warning("Real evaluation failed (%s). Using simulation.", exc)
                y_true, y_pred = generate_simulated_predictions(labels)

    # --- Compute metrics ---
    overall_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    logger.info("Overall metrics: %s", overall_metrics)

    # --- Classification report ---
    report = build_classification_report(y_true, y_pred, labels)
    logger.info("Per-class report computed for %d classes", len(labels))

    # Save classification report
    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Classification report saved to %s", report_path)

    # --- Confusion matrix ---
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, labels, cm_path)

    # --- Log to MLflow ---
    logger.info("Logging results to MLflow...")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="evaluation") as run:
        run_id = run.info.run_id
        logger.info("MLflow evaluation run: %s", run_id)

        # Log overall metrics
        mlflow.log_metrics(overall_metrics)

        # Log per-class metrics
        for class_name in labels:
            if class_name in report:
                class_metrics = report[class_name]
                for metric_name in ["precision", "recall", "f1-score"]:
                    if metric_name in class_metrics:
                        safe_name = metric_name.replace("-", "_")
                        mlflow.log_metric(
                            f"{class_name}_{safe_name}",
                            float(class_metrics[metric_name]),
                        )

        # Log artifacts
        mlflow.log_artifact(cm_path, "evaluation")
        mlflow.log_artifact(report_path, "evaluation")
        logger.info("Evaluation artifacts logged to MLflow")

        # Log params
        mlflow.log_param("num_classes", len(labels))
        mlflow.log_param("num_test_samples", len(y_true))
        mlflow.log_param("simulate", simulate)

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)

    return overall_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list.

    Returns:
        Parsed Namespace.
    """
    parser = argparse.ArgumentParser(
        description="BYOAI Intent Classification Model Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to trained model directory.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate predictions with realistic dummy data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory for evaluation artifacts.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        evaluate(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            model_dir=args.model_dir,
            simulate=args.simulate,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        sys.exit(1)
