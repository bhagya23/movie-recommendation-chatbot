"""
test_integration.py — Integration testing for the Movie Recommendation Chatbot.

Tests:
1. Dataset loading
2. Preprocessing pipeline
3. Model training (quick test)
4. Intent prediction
5. Recommendation engine
6. End-to-end chatbot

Usage:
    python test_integration.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import get_logger

logger = get_logger("test")


def test_dataset_loading():
    """Test dataset loading and processing."""
    logger.info("=" * 60)
    logger.info("Test 1: Dataset Loading")
    logger.info("=" * 60)
    
    try:
        from src.dataset import load_processed_catalog
        movies = load_processed_catalog()
        
        assert len(movies) > 0, "No movies loaded"
        assert all('title' in m for m in movies), "Movies missing title field"
        assert all('genre' in m for m in movies), "Movies missing genre field"
        
        logger.info("✓ Dataset loading successful")
        logger.info("  Loaded %d movies", len(movies))
        return True
    except Exception as e:
        logger.error("✗ Dataset loading failed: %s", e)
        return False


def test_preprocessing():
    """Test preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("Test 2: Preprocessing Pipeline")
    logger.info("=" * 60)
    
    try:
        from src.preprocessing import preprocess_sentence, bag_of_words
        from src.utils import load_intents
        
        # Load intents to get vocabulary
        intents_data = load_intents()
        from src.preprocessing import build_vocabulary
        words, classes, documents = build_vocabulary(intents_data)
        
        # Test preprocessing
        test_sentence = "I want a comedy movie"
        processed = preprocess_sentence(test_sentence)
        
        assert len(processed) > 0, "Preprocessing returned empty list"
        
        # Test bag of words
        bow = bag_of_words(test_sentence, words)
        assert len(bow) == len(words), "BoW vector size mismatch"
        
        logger.info("✓ Preprocessing pipeline successful")
        logger.info("  Processed: %s", processed)
        logger.info("  Vocabulary size: %d", len(words))
        return True
    except Exception as e:
        logger.error("✗ Preprocessing failed: %s", e)
        return False


def test_model_loading():
    """Test model loading."""
    logger.info("=" * 60)
    logger.info("Test 3: Model Loading")
    logger.info("=" * 60)
    
    try:
        from src.config import MODEL_FILE
        
        if not os.path.exists(MODEL_FILE):
            logger.warning("⚠ Model file not found at %s", MODEL_FILE)
            logger.info("  Run: python src/train.py")
            return None
        
        from src.predict import IntentPredictor
        predictor = IntentPredictor()
        
        assert predictor.model is not None, "Model not loaded"
        assert len(predictor.classes) > 0, "No classes loaded"
        
        logger.info("✓ Model loading successful")
        logger.info("  Classes: %d", len(predictor.classes))
        logger.info("  Vocabulary: %d", len(predictor.words))
        return True
    except Exception as e:
        logger.error("✗ Model loading failed: %s", e)
        return False


def test_intent_prediction():
    """Test intent prediction."""
    logger.info("=" * 60)
    logger.info("Test 4: Intent Prediction")
    logger.info("=" * 60)
    
    try:
        from src.config import MODEL_FILE
        
        if not os.path.exists(MODEL_FILE):
            logger.warning("⚠ Model file not found, skipping prediction test")
            return None
        
        from src.predict import IntentPredictor
        predictor = IntentPredictor()
        
        test_queries = [
            "I want a comedy movie",
            "Show me action films",
            "I'm feeling sad",
        ]
        
        for query in test_queries:
            intent, confidence = predictor.predict(query)
            logger.info("  Query: '%s' → Intent: %s (%.2f%%)", query, intent, confidence * 100)
        
        logger.info("✓ Intent prediction successful")
        return True
    except Exception as e:
        logger.error("✗ Intent prediction failed: %s", e)
        return False


def test_recommendation_engine():
    """Test recommendation engine."""
    logger.info("=" * 60)
    logger.info("Test 5: Recommendation Engine")
    logger.info("=" * 60)
    
    try:
        from src.recommender import RecommendationEngine

        engine = RecommendationEngine()

        # Genre-tag filtering
        action = engine.filter_for_tag("genre_action", k=3)
        assert len(action) > 0, "No action movies returned"
        assert all(m["genre"] == "action" for m in action), "Non-action movie leaked in"

        # Language-tag filtering
        korean = engine.filter_for_tag("lang_korean", k=3)
        assert all(m["language"] == "korean" for m in korean), "Non-korean movie leaked in"

        # Top-rated + explanation
        movies, explanation = engine.recommend("top_rated", k=3)
        assert movies and explanation, "top_rated returned nothing"

        logger.info("✓ Recommendation engine successful")
        logger.info("  Total movies in catalog: %d", len(engine.movies))
        logger.info("  Sample recommendation: %s", movies[0]['title'] if movies else "None")
        return True
    except Exception as e:
        logger.error("✗ Recommendation engine failed: %s", e)
        return False


def test_end_to_end():
    """Test end-to-end chatbot pipeline."""
    logger.info("=" * 60)
    logger.info("Test 6: End-to-End Chatbot")
    logger.info("=" * 60)
    
    try:
        from src.config import MODEL_FILE
        
        if not os.path.exists(MODEL_FILE):
            logger.warning("⚠ Model file not found, skipping end-to-end test")
            return None
        
        from src.chatbot import MovieChatbot
        
        chatbot = MovieChatbot()
        
        test_query = "I want a comedy movie"
        result = chatbot.chat(test_query, n_recommendations=3)
        
        assert 'intent' in result, "Response missing intent"
        assert 'movies' in result, "Response missing movies"
        assert 'response' in result, "Response missing text response"
        
        logger.info("✓ End-to-end chatbot successful")
        logger.info("  Query: '%s'", test_query)
        logger.info("  Intent: %s", result['intent'])
        logger.info("  Movies recommended: %d", len(result['movies']))
        return True
    except Exception as e:
        logger.error("✗ End-to-end chatbot failed: %s", e)
        return False


def main():
    """Run all integration tests."""
    logger.info("\n" + "=" * 60)
    logger.info("MOVIE CHATBOT INTEGRATION TESTS")
    logger.info("=" * 60 + "\n")
    
    results = {
        'Dataset Loading': test_dataset_loading(),
        'Preprocessing': test_preprocessing(),
        'Model Loading': test_model_loading(),
        'Intent Prediction': test_intent_prediction(),
        'Recommendation Engine': test_recommendation_engine(),
        'End-to-End Chatbot': test_end_to_end(),
    }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        if result is True:
            logger.info("✓ %s: PASSED", test_name)
        elif result is False:
            logger.info("✗ %s: FAILED", test_name)
        else:
            logger.info("⚠ %s: SKIPPED", test_name)
    
    logger.info("\nTotal: %d passed, %d failed, %d skipped", passed, failed, skipped)
    logger.info("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
