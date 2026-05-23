"""
chatbot.py — Chatbot inference engine.
Loads trained DNN, predicts intent, routes to recommendation engine.
"""

import os
import sys
import random
import logging
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MODEL_FILE, WORDS_FILE, CLASSES_FILE,
    CONFIDENCE_THRESHOLD, ERROR_THRESHOLD, CHATBOT_NAME
)
from src.preprocessing import bag_of_words, load_artifacts
from src.utils import get_logger, load_intents, extract_entities
from src.recommendation_engine import RecommendationEngine

logger = get_logger("chatbot")


class MovieChatbot:
    """
    Closed-domain intent-classification chatbot for Bollywood movie recommendations.

    Pipeline:
    1. User query → clean_sentence → bag_of_words
    2. DNN model → softmax probabilities
    3. argmax → intent class (with confidence threshold)
    4. Entity extraction (genre, mood, year, actor, language)
    5. RecommendationEngine.dispatch(intent, entities) → movies + explanation
    6. Return (text_response, movies_list, explanation)
    """

    def __init__(self):
        self.model       = None
        self.words:    list[str] = []
        self.classes:  list[str] = []
        self.intents:  dict      = {}
        self.engine: RecommendationEngine = RecommendationEngine()
        self.conversation_history: list[dict] = []
        self._context: dict = {}  # Multi-turn context memory
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load keras model + vocabulary artifacts."""
        if not os.path.exists(MODEL_FILE):
            logger.error(
                "Model not found at %s — please run: python src/train.py", MODEL_FILE
            )
            raise FileNotFoundError(
                f"Trained model missing: {MODEL_FILE}\n"
                "Run:  python src/train.py"
            )

        import tensorflow as tf  # deferred import for startup speed
        self.model = tf.keras.models.load_model(MODEL_FILE)
        self.words, self.classes = load_artifacts()
        self.intents = load_intents()
        logger.info(
            "Model loaded — %d vocab words | %d intent classes",
            len(self.words), len(self.classes)
        )

    # ── Intent prediction ──────────────────────────────────────────────────────

    def predict_intent(self, sentence: str) -> list[dict]:
        """
        Return list of {intent, probability} sorted by probability desc.
        Applies ERROR_THRESHOLD to filter low-confidence classes.
        """
        bow = bag_of_words(sentence, self.words)
        bow = np.expand_dims(bow, axis=0)
        predictions = self.model.predict(bow, verbose=0)[0]

        results = [
            {"intent": self.classes[i], "probability": float(p)}
            for i, p in enumerate(predictions)
            if float(p) > ERROR_THRESHOLD
        ]
        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    def get_top_intent(self, sentence: str) -> tuple[str, float]:
        """Return (intent_tag, confidence) for the top prediction."""
        results = self.predict_intent(sentence)
        if not results:
            return "fallback", 0.0
        top = results[0]
        return top["intent"], top["probability"]

    # ── Response generation ────────────────────────────────────────────────────

    def _get_response(self, intent_tag: str) -> str:
        """Pick a random response string from the matched intent."""
        for intent in self.intents["intents"]:
            if intent["tag"] == intent_tag:
                return random.choice(intent["responses"])
        return "I'm not sure what you mean. Could you rephrase?"

    def _update_context(self, entities: dict) -> None:
        """Update multi-turn context with newly extracted entities (non-null only)."""
        for k, v in entities.items():
            if v is not None:
                self._context[k] = v

    def _merge_entities_with_context(self, entities: dict) -> dict:
        """Merge current-turn entities with remembered context."""
        merged = dict(self._context)
        for k, v in entities.items():
            if v is not None:
                merged[k] = v
        return merged

    # ── Main chat entry point ──────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        sidebar_filters: Optional[dict] = None,
        n_recommendations: int = 5,
    ) -> dict:
        """
        Process a user message and return a full response dict.

        Returns:
            {
                "intent":        str,
                "confidence":    float,
                "response":      str,      # text response
                "movies":        list,     # recommended movie dicts
                "explanation":   str,      # why these were recommended
                "entities":      dict,     # extracted entities
            }
        """
        # 1. Predict intent
        intent, confidence = self.get_top_intent(user_message)

        # Low-confidence → fallback
        if confidence < CONFIDENCE_THRESHOLD:
            intent = "fallback"

        # 2. Entity extraction
        entities  = extract_entities(user_message)
        self._update_context(entities)
        merged    = self._merge_entities_with_context(entities)

        # 3. Text response
        response = self._get_response(intent)

        # 4. Recommendation dispatch
        movies, explanation = self.engine.dispatch(
            intent=intent,
            entities=merged,
            sidebar_filters=sidebar_filters,
            n=n_recommendations,
        )

        # 5. Log conversation
        turn = {
            "user":        user_message,
            "intent":      intent,
            "confidence":  round(confidence, 4),
            "bot":         response,
            "movies":      [m["title"] for m in movies],
            "explanation": explanation,
        }
        self.conversation_history.append(turn)
        logger.info(
            "Turn — intent: %s (%.2f%%) | movies: %d | explanation: %s",
            intent, confidence * 100, len(movies), explanation
        )

        return {
            "intent":      intent,
            "confidence":  round(confidence, 4),
            "response":    response,
            "movies":      movies,
            "explanation": explanation,
            "entities":    merged,
        }

    def reset_context(self) -> None:
        """Clear multi-turn conversation context."""
        self._context = {}
        self.conversation_history = []
        logger.info("Conversation context reset.")

    def get_greeting(self) -> str:
        return self._get_response("greeting")

    def get_history(self) -> list[dict]:
        return self.conversation_history
