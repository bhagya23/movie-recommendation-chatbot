"""
chatbot.py — Chatbot orchestration (matches the academic notebook).

Pipeline:
    user text → intent classifier (BoW + DNN) → intent tag
              → control intents answer directly (greeting/goodbye/thanks/help)
              → otherwise filter the catalogue via INTENT_TAG_MAP and return top-k
"""

import os
import sys
import random
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TOP_RECOMMENDATIONS, DIRECT_RESPONSE_TAGS, CHATBOT_NAME
from src.predict import IntentPredictor
from src.recommender import RecommendationEngine
from src.utils import get_logger, load_intents

logger = get_logger("chatbot")


class MovieChatbot:
    """Closed-domain intent-classification chatbot for movie recommendations."""

    def __init__(self):
        self.predictor = IntentPredictor()
        self.engine = RecommendationEngine()
        intents_data = load_intents()
        self.intent_lookup: Dict[str, dict] = {
            intent["tag"]: intent for intent in intents_data["intents"]
        }
        logger.info("MovieChatbot ready — %d intents loaded.", len(self.intent_lookup))

    def _response_text(self, tag: str) -> str:
        """Pick a response for the tag from the intent corpus (with a sensible default)."""
        intent = self.intent_lookup.get(tag) or self.intent_lookup.get("recommend_general")
        if intent and intent.get("responses"):
            return random.choice(intent["responses"])
        return f"Here are some movie recommendations from {CHATBOT_NAME}!"

    def chat(self, user_text: str, n_recommendations: int = TOP_RECOMMENDATIONS) -> dict:
        """Process a user message and return intent, response text, and movies."""
        tag, confidence = self.predictor.predict(user_text)

        # Control intents respond directly, without recommendations
        if tag in DIRECT_RESPONSE_TAGS:
            response = self._response_text(tag)
            logger.info("Turn — intent: %s (%.2f%%) | direct response", tag, confidence * 100)
            return {
                "intent": tag,
                "confidence": confidence,
                "response": response,
                "explanation": "",
                "movies": [],
            }

        movies = self.engine.filter_for_tag(tag, k=n_recommendations)
        explanation = self.engine.explain(tag)
        response = self._response_text(tag)

        logger.info("Turn — intent: %s (%.2f%%) | movies: %d",
                    tag, confidence * 100, len(movies))
        return {
            "intent": tag,
            "confidence": confidence,
            "response": response,
            "explanation": explanation,
            "movies": movies,
        }


# ── Module-level convenience ─────────────────────────────────────────────────────

_bot = None


def get_chatbot() -> MovieChatbot:
    global _bot
    if _bot is None:
        _bot = MovieChatbot()
    return _bot


def get_chatbot_response(user_text: str, k: int = TOP_RECOMMENDATIONS) -> dict:
    return get_chatbot().chat(user_text, n_recommendations=k)


if __name__ == "__main__":
    bot = MovieChatbot()
    demo = [
        "Hi",
        "I want a happy Hindi movie",
        "Recommend a Korean thriller",
        "Show me top rated sci-fi films",
        "I need a movie for kids",
        "Suggest something romantic",
    ]
    for q in demo:
        result = bot.chat(q, n_recommendations=3)
        print(f"\nUser : {q}")
        print(f"Bot  : {result['response']}")
        print(f"Intent: {result['intent']} (confidence={result['confidence']:.2%})")
        for m in result["movies"]:
            print(f"  - {m['title']} ({m['year']}, {m['genre']}, IMDb {m['imdb']})")
