"""
Intent classifier service for the BYOAI ML Service.

Wraps the HuggingFace zero-shot classification pipeline to classify
user messages into predefined intent categories. Applies a confidence
threshold to filter low-confidence predictions as out_of_scope.
"""

import logging
from typing import Any

from transformers import pipeline

logger = logging.getLogger(__name__)

# Predefined candidate labels for customer service intents
CANDIDATE_LABELS: list[str] = [
    "greeting",
    "farewell",
    "question",
    "complaint",
    "booking",
    "feedback",
    "help",
    "cancel",
    "status_check",
    "out_of_scope",
]


class IntentClassifier:
    """
    Zero-shot intent classifier using HuggingFace transformers.

    Uses the facebook/bart-large-mnli model (or a configured alternative)
    to classify input text against a set of candidate intent labels.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str = "cpu",
        confidence_threshold: float = 0.3,
    ) -> None:
        """
        Initialize the intent classifier.

        Args:
            model_name: HuggingFace model identifier for zero-shot classification.
            device: Device to run inference on ('cpu' or 'cuda').
            confidence_threshold: Minimum confidence to accept a classification.
                Below this threshold, the intent is set to 'out_of_scope'.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.candidate_labels = CANDIDATE_LABELS

        logger.info("Initializing zero-shot classification pipeline with model: %s", model_name)

        device_index = -1 if device == "cpu" else 0
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device_index,
        )

        logger.info("Pipeline initialized successfully on device: %s", device)

    def classify(self, text: str) -> dict[str, Any]:
        """
        Classify the intent of the given text.

        Args:
            text: The input text to classify.

        Returns:
            A dictionary containing:
                - intent (str): The top predicted intent label.
                - confidence (float): The confidence score of the top intent.
                - all_intents (list[dict]): All intents with their scores,
                  sorted by score descending.
        """
        result = self.pipeline(
            text,
            candidate_labels=self.candidate_labels,
            multi_label=False,
        )

        # Build sorted list of all intent scores
        all_intents = [
            {"intent": label, "score": round(score, 4)}
            for label, score in zip(result["labels"], result["scores"])
        ]

        top_intent = all_intents[0]["intent"]
        top_score = all_intents[0]["score"]

        # Apply confidence threshold
        if top_score < self.confidence_threshold:
            logger.info(
                "Low confidence (%.3f < %.3f) for text '%s...', defaulting to out_of_scope",
                top_score,
                self.confidence_threshold,
                text[:50],
            )
            top_intent = "out_of_scope"
            top_score = round(top_score, 4)

        logger.debug(
            "Classification result: intent=%s, confidence=%.4f, text='%s...'",
            top_intent,
            top_score,
            text[:50],
        )

        return {
            "intent": top_intent,
            "confidence": top_score,
            "all_intents": all_intents,
        }
