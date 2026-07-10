"""
predict.py — Model inference for intent prediction.

Pipeline:
Input → Text preprocessing → Tokenization → Lemmatization → 
Bag of Words → Neural Network → Softmax → Intent Prediction

Usage:
    from src.predict import IntentPredictor
    predictor = IntentPredictor()
    intent, confidence = predictor.predict("I want a comedy movie")
"""

import os
import sys
import logging
from typing import Tuple, List, Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MODEL_FILE, WORDS_FILE, CLASSES_FILE,
    CONFIDENCE_THRESHOLD, KEYWORD_TO_TAG,
)
from src.preprocessing import bag_of_words, load_artifacts
from src.utils import get_logger

logger = get_logger("predict")


class IntentPredictor:
    """
    Intent prediction using trained neural network.
    
    Pipeline:
    1. Load trained model and artifacts
    2. Preprocess input text
    3. Convert to Bag of Words
    4. Predict intent probabilities
    5. Return top intent with confidence
    """
    
    def __init__(self):
        """Load model and artifacts."""
        self.model = None
        self.words: List[str] = []
        self.classes: List[str] = []
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained Keras model and vocabulary artifacts."""
        if not os.path.exists(MODEL_FILE):
            logger.error("Model not found at %s — please run: python src/train.py", MODEL_FILE)
            raise FileNotFoundError(
                f"Trained model missing: {MODEL_FILE}\n"
                "Run: python src/train.py"
            )
        
        # Import TensorFlow only when needed (faster startup)
        import tensorflow as tf
        self.model = tf.keras.models.load_model(MODEL_FILE)
        self.words, self.classes = load_artifacts()
        
        logger.info(
            "Model loaded — vocab: %d words | classes: %d intents",
            len(self.words), len(self.classes)
        )
    
    def preprocess(self, sentence: str) -> np.ndarray:
        """
        Preprocess input sentence and convert to Bag of Words.
        
        Args:
            sentence: Raw input text
        
        Returns:
            Bag of Words numpy array
        """
        bow = bag_of_words(sentence, self.words)
        return bow
    
    def predict_proba(self, sentence: str) -> np.ndarray:
        """
        Predict intent probabilities for a sentence.
        
        Args:
            sentence: Raw input text
        
        Returns:
            Array of probabilities for each intent class
        """
        bow = self.preprocess(sentence)
        bow = np.expand_dims(bow, axis=0)
        probabilities = self.model.predict(bow, verbose=0)[0]
        return probabilities
    
    def predict(self, sentence: str) -> Tuple[str, float]:
        """
        Predict the top intent and confidence for a sentence.

        Mirrors the academic notebook: if the classifier confidence is below
        the threshold, fall back to keyword matching, then to recommend_general.

        Returns:
            Tuple of (intent_tag, confidence_score)
        """
        probabilities = self.predict_proba(sentence)
        top_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[top_idx])
        intent = self.classes[top_idx]

        if confidence >= CONFIDENCE_THRESHOLD:
            return intent, confidence

        # Low confidence → keyword-based fallback
        text_lower = sentence.lower()
        for keyword, mapped_tag in KEYWORD_TO_TAG.items():
            if keyword in text_lower:
                logger.info("Low confidence (%.2f%%); keyword '%s' → %s",
                            confidence * 100, keyword, mapped_tag)
                return mapped_tag, confidence

        logger.info("Low confidence (%.2f%%); defaulting to recommend_general",
                    confidence * 100)
        return "recommend_general", confidence
    
    def predict_top_k(self, sentence: str, k: int = 3) -> List[Dict[str, float]]:
        """
        Predict top K intents with their probabilities.
        
        Args:
            sentence: Raw input text
            k: Number of top intents to return
        
        Returns:
            List of {intent, probability} dicts sorted by probability
        """
        probabilities = self.predict_proba(sentence)
        top_k_indices = np.argsort(probabilities)[::-1][:k]
        return [
            {"intent": self.classes[idx], "probability": float(probabilities[idx])}
            for idx in top_k_indices
        ]
    
    def get_all_intents(self) -> List[str]:
        """Return list of all intent classes."""
        return self.classes.copy()
    
    def get_vocabulary_size(self) -> int:
        """Return vocabulary size."""
        return len(self.words)


# ── Convenience functions ─────────────────────────────────────────────────────

def predict_intent(sentence: str) -> Tuple[str, float]:
    """
    Convenience function for single prediction.
    
    Args:
        sentence: Raw input text
    
    Returns:
        Tuple of (intent_tag, confidence_score)
    """
    predictor = IntentPredictor()
    return predictor.predict(sentence)


def predict_top_intents(sentence: str, k: int = 3) -> List[Dict[str, float]]:
    """
    Convenience function for top K predictions.
    
    Args:
        sentence: Raw input text
        k: Number of top intents
    
    Returns:
        List of {intent, probability} dicts
    """
    predictor = IntentPredictor()
    return predictor.predict_top_k(sentence, k)


if __name__ == "__main__":
    # Test inference
    test_queries = [
        "I want a comedy movie",
        "Show me action films",
        "I'm feeling sad",
        "Recommend a romantic movie",
        "What are the latest movies?",
    ]
    
    predictor = IntentPredictor()
    
    print("=" * 60)
    print("Intent Prediction Test")
    print("=" * 60)
    
    for query in test_queries:
        intent, confidence = predictor.predict(query)
        print(f"\nQuery: '{query}'")
        print(f"Intent: {intent}")
        print(f"Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        top_k = predictor.predict_top_k(query, k=3)
        print("Top 3 predictions:")
        for i, pred in enumerate(top_k, 1):
            print(f"  {i}. {pred['intent']}: {pred['probability']:.2%}")
    
    print("\n" + "=" * 60)
