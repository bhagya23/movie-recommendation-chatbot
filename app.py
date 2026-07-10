"""
app.py — Flask backend for Movie Recommendation Chatbot.

Routes:
- GET  /         : Serve chatbot UI
- POST /predict  : Process user query and return recommendations

Usage:
    python app.py
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (
    APP_TITLE, APP_SUBTITLE, CHATBOT_NAME,
    TOP_RECOMMENDATIONS, LOGS_DIR
)
from src.chatbot import MovieChatbot
from src.utils import get_logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logger = get_logger("app")

# Global chatbot instance (lazy-loaded)
chatbot: Optional[MovieChatbot] = None


def load_models():
    """Load the chatbot (intent classifier + recommendation engine)."""
    global chatbot
    if chatbot is None:
        try:
            chatbot = MovieChatbot()
            logger.info("MovieChatbot loaded successfully")
        except Exception as e:
            logger.error("Failed to load chatbot: %s", e)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the chatbot UI."""
    return render_template(
        'index.html',
        app_title=APP_TITLE,
        app_subtitle=APP_SUBTITLE,
        chatbot_name=CHATBOT_NAME
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    Process user query and return movie recommendations.
    
    Expected JSON payload:
    {
        "query": "I want a comedy movie",
        "n_recommendations": 5 (optional)
    }
    
    Returns JSON:
    {
        "intent": "comedy_movies",
        "confidence": 0.95,
        "response": "Here are some hilarious Bollywood comedies!",
        "movies": [...],
        "explanation": "Recommended based on genre: Comedy",
        "entities": {...}
    }
    """
    try:
        # Load models if not already loaded
        load_models()

        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        query = data.get('query', '').strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        n_recommendations = int(data.get('n_recommendations', TOP_RECOMMENDATIONS))
        logger.info("Processing query: '%s'", query)

        if chatbot is None:
            return jsonify({
                "error": "Model not trained. Please run: python src/train.py",
                "query": query,
                "fallback": True
            }), 503

        result = chatbot.chat(query, n_recommendations=n_recommendations)

        return jsonify({
            "query": query,
            "intent": result["intent"],
            "confidence": round(result["confidence"], 4),
            "response": result["response"],
            "movies": result["movies"],
            "explanation": result["explanation"],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error("Error processing prediction: %s", e, exc_info=True)
        return jsonify({
            "error": str(e),
            "query": data.get('query', '') if 'data' in locals() else ''
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    load_models()
    
    status = {
        "status": "healthy" if chatbot is not None else "degraded",
        "chatbot_loaded": chatbot is not None,
        "movies_loaded": len(chatbot.engine.movies) if chatbot is not None else 0,
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status)


@app.route('/intents', methods=['GET'])
def get_intents():
    """Return list of available intent classes."""
    load_models()
    if chatbot is None:
        return jsonify({"error": "Chatbot not loaded"}), 503
    intents = chatbot.predictor.get_all_intents()
    return jsonify({"intents": intents, "count": len(intents)})


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error("Internal server error: %s", error)
    return jsonify({"error": "Internal server error"}), 500


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting Flask server")
    logger.info("=" * 60)
    
    # Load models on startup
    load_models()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
