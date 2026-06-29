#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for BYOAI Intent Classification.

Loads raw intent data, cleans text, tokenizes with HuggingFace,
encodes labels, performs stratified splits, and saves processed datasets.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("byoai.preprocess")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 128
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "sample_data", "intents.json"
)
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean a single text string.

    Steps:
        1. Convert to lowercase.
        2. Remove special characters (keep alphanumeric, spaces, and basic
           punctuation: . , ? ! ' -).
        3. Normalise whitespace (collapse multiple spaces to one).
        4. Strip leading / trailing whitespace.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,?!'\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(data_path: str) -> pd.DataFrame:
    """Load intent data from a JSON file.

    The JSON file must contain an array of objects each having ``text`` and
    ``intent`` keys.

    Args:
        data_path: Path to the JSON file.

    Returns:
        A DataFrame with ``text`` and ``intent`` columns.

    Raises:
        FileNotFoundError: If *data_path* does not exist.
        ValueError: If required keys are missing from any record.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("Loading data from %s", data_path)
    with open(path, "r", encoding="utf-8") as fh:
        records: List[Dict[str, str]] = json.load(fh)

    # Validate
    for idx, record in enumerate(records):
        if "text" not in record or "intent" not in record:
            raise ValueError(
                f"Record {idx} missing 'text' or 'intent' key: {record}"
            )

    df = pd.DataFrame(records)
    logger.info("Loaded %d records with %d unique intents", len(df), df["intent"].nunique())
    return df


# ---------------------------------------------------------------------------
# Label Encoding
# ---------------------------------------------------------------------------
def encode_labels(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, LabelEncoder, Dict[int, str]]:
    """Encode intent labels as integers.

    Args:
        df: DataFrame with an ``intent`` column.

    Returns:
        Tuple of (updated DataFrame with ``label`` column, fitted
        LabelEncoder, id-to-label mapping dict).
    """
    le = LabelEncoder()
    df = df.copy()
    df["label"] = le.fit_transform(df["intent"])
    id2label: Dict[int, str] = {int(i): label for i, label in enumerate(le.classes_)}
    label2id: Dict[str, int] = {label: int(i) for i, label in enumerate(le.classes_)}
    logger.info("Label mapping: %s", id2label)
    return df, le, id2label


# ---------------------------------------------------------------------------
# Class Distribution
# ---------------------------------------------------------------------------
def report_class_distribution(df: pd.DataFrame, split_name: str = "full") -> None:
    """Log the class distribution for a DataFrame.

    Args:
        df: DataFrame with ``intent`` and ``label`` columns.
        split_name: Name of the split for logging purposes.
    """
    dist = df["intent"].value_counts().sort_index()
    total = len(df)
    logger.info("--- Class distribution (%s, n=%d) ---", split_name, total)
    for intent, count in dist.items():
        pct = count / total * 100
        logger.info("  %-20s %4d  (%5.1f%%)", intent, count, pct)


# ---------------------------------------------------------------------------
# Stratified Splitting
# ---------------------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train / validation / test with stratification.

    Args:
        df: Full DataFrame.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios do not approximately sum to 1.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    logger.info(
        "Splitting data: train=%.0f%% val=%.0f%% test=%.0f%%",
        train_ratio * 100,
        val_ratio * 100,
        test_ratio * 100,
    )

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["intent"],
        random_state=random_state,
    )

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["intent"],
        random_state=random_state,
    )

    logger.info("Split sizes — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize_data(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dataset:
    """Tokenize text data and return a HuggingFace Dataset.

    Args:
        df: DataFrame with ``text`` and ``label`` columns.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.

    Returns:
        HuggingFace Dataset with tokenized inputs and labels.
    """
    dataset = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def _tokenize(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(_tokenize, batched=True, desc="Tokenizing")
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


# ---------------------------------------------------------------------------
# Save Artifacts
# ---------------------------------------------------------------------------
def save_artifacts(
    dataset_dict: DatasetDict,
    id2label: Dict[int, str],
    output_dir: str,
) -> None:
    """Persist processed datasets and label mapping to disk.

    Args:
        dataset_dict: HuggingFace DatasetDict with train/val/test splits.
        id2label: Integer-to-label mapping.
        output_dir: Directory to save outputs into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_path = out / "dataset"
    dataset_dict.save_to_disk(str(dataset_path))
    logger.info("Saved DatasetDict to %s", dataset_path)

    label_map_path = out / "label_mapping.json"
    label2id = {v: k for k, v in id2label.items()}
    mapping = {"id2label": id2label, "label2id": label2id}
    with open(label_map_path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)
    logger.info("Saved label mapping to %s", label_map_path)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> DatasetDict:
    """Execute the full preprocessing pipeline.

    Args:
        data_path: Path to raw intents JSON file.
        output_dir: Directory for processed output.
        model_name: HuggingFace model / tokenizer name.
        max_length: Max token length for tokenizer.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        random_state: Random seed for reproducibility.

    Returns:
        The processed HuggingFace DatasetDict.
    """
    logger.info("=" * 60)
    logger.info("BYOAI Data Preprocessing Pipeline")
    logger.info("=" * 60)

    # 1. Load
    df = load_data(data_path)

    # 2. Clean
    logger.info("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)
    logger.info("Sample cleaned texts:\n%s", df["text"].head(5).to_string())

    # 3. Encode labels
    df, le, id2label = encode_labels(df)

    # 4. Distribution report (full)
    report_class_distribution(df, split_name="full")

    # 5. Stratified split
    train_df, val_df, test_df = stratified_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        report_class_distribution(split_df, split_name=name)

    # 6. Tokenize
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Tokenizing splits...")
    train_ds = tokenize_data(train_df, tokenizer, max_length)
    val_ds = tokenize_data(val_df, tokenizer, max_length)
    test_ds = tokenize_data(test_df, tokenizer, max_length)

    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    logger.info("DatasetDict: %s", dataset_dict)

    # 7. Save
    save_artifacts(dataset_dict, id2label, output_dir)

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("=" * 60)
    return dataset_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="BYOAI Intent Data Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to raw intents JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save processed data.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace tokenizer / model name.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training split ratio."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation split ratio."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test split ratio."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_length=args.max_length,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state,
        )
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
