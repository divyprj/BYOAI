"""
Response generator service for the BYOAI ML Service.

Generates natural, conversational customer service responses based on
the classified intent. Uses template-based generation with randomization
for variety.
"""

import logging
import random

logger = logging.getLogger(__name__)

# Response templates organized by intent, each with 3-5 professional variations
RESPONSE_TEMPLATES: dict[str, list[str]] = {
    "greeting": [
        "Hello! Welcome to BYOAI support. How can I help you today?",
        "Hi there! Thanks for reaching out. What can I assist you with?",
        "Welcome! I'm here to help. What can I do for you today?",
        "Good to see you! How may I assist you today?",
        "Hello and welcome! I'd love to help - what's on your mind?",
    ],
    "farewell": [
        "Thank you for reaching out! Have a wonderful day.",
        "Goodbye! Don't hesitate to contact us if you need anything else.",
        "It was great helping you. Take care and have a great day!",
        "Thanks for chatting with us! We're always here if you need support.",
    ],
    "question": [
        "That's a great question! Let me look into that for you.",
        "I'd be happy to help answer that. Let me find the best information for you.",
        "Good question! Here's what I can tell you about that.",
        "Let me help clarify that for you. I'll pull up the relevant details.",
        "Excellent question! Allow me to provide you with a thorough answer.",
    ],
    "complaint": [
        "I'm sorry to hear about your experience. Let me help resolve this issue for you.",
        "I apologize for the inconvenience. Let's work together to find a solution.",
        "I understand your frustration, and I'm committed to making this right.",
        "Thank you for bringing this to our attention. We take this seriously and want to help.",
        "I sincerely apologize for the trouble. Let me escalate this and get it resolved quickly.",
    ],
    "booking": [
        "I'd be happy to help you with a booking. Could you provide more details?",
        "Let's get that booked for you! What dates and preferences do you have in mind?",
        "I can help you with your reservation. What specifics are you looking for?",
        "Sure, I'd love to assist with your booking. What details do you need?",
    ],
    "feedback": [
        "Thank you for your feedback! We truly value your input.",
        "We appreciate you sharing your thoughts. Your feedback helps us improve.",
        "Thanks for taking the time to provide feedback. We'll make sure it's heard.",
        "Your feedback is important to us. Thank you for helping us get better!",
        "We're grateful for your feedback! It helps shape our service.",
    ],
    "help": [
        "Of course! I'm here to help. What do you need assistance with?",
        "I'd be glad to assist you. Could you tell me more about what you need?",
        "Absolutely - let me help you with that. What's going on?",
        "You've come to the right place! How can I assist you today?",
    ],
    "cancel": [
        "I understand you'd like to cancel. Let me assist you with that process.",
        "I can help you with the cancellation. Could you provide your booking or order details?",
        "I'll help you process that cancellation right away. May I have your reference number?",
        "No problem - I can handle the cancellation for you. Let me just verify a few details.",
    ],
    "status_check": [
        "Let me check the status of that for you right away.",
        "I'll look up the latest status on your request. One moment please.",
        "Sure! Let me pull up the current status of your order or request.",
        "I'd be happy to check on that. Could you share your reference or order number?",
    ],
    "out_of_scope": [
        "I appreciate your message, but I'm not quite sure how to help with that. "
        "Could you rephrase or provide more details?",
        "That's a bit outside my area of expertise. Could you clarify what you need "
        "so I can direct you to the right resource?",
        "I want to make sure I help you correctly. Could you provide a bit more context "
        "about what you're looking for?",
        "I'm not entirely sure I understand your request. Could you elaborate so I can "
        "assist you better?",
    ],
}


class ResponseGenerator:
    """
    Generates contextual, conversational responses based on classified intent.

    Uses a template-based approach with random selection for natural variety.
    """

    def __init__(self, templates: dict[str, list[str]] | None = None) -> None:
        """
        Initialize the response generator.

        Args:
            templates: Optional custom response templates. If not provided,
                uses the default RESPONSE_TEMPLATES.
        """
        self.templates = templates or RESPONSE_TEMPLATES

    def generate(
        self,
        intent: str,
        confidence: float,
        original_text: str,
    ) -> str:
        """
        Generate a contextual response for the given intent.

        Args:
            intent: The classified intent label.
            confidence: The confidence score of the classification.
            original_text: The original user input text.

        Returns:
            A natural, conversational response string appropriate
            for the classified intent.
        """
        # Get templates for the intent, fall back to out_of_scope
        intent_templates = self.templates.get(
            intent, self.templates.get("out_of_scope", [])
        )

        if not intent_templates:
            logger.warning("No templates found for intent: %s", intent)
            return (
                "Thank you for your message. I'm here to help - "
                "could you provide a bit more detail?"
            )

        response = random.choice(intent_templates)

        logger.debug(
            "Generated response for intent=%s (confidence=%.3f): '%s...'",
            intent,
            confidence,
            response[:60],
        )

        return response
