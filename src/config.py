"""
config.py — Central configuration for the Movie Recommendation Chatbot.
All path constants, model hyperparameters, and app settings live here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR   = os.path.join(OUTPUTS_DIR, "reports")

INTENTS_FILE  = os.path.join(DATA_DIR, "intents.json")
MOVIES_FILE   = os.path.join(DATA_DIR, "movies.json")
CSV_FILE      = os.path.join(DATA_DIR, "bollywood_movies.csv")

MODEL_FILE    = os.path.join(MODELS_DIR, "chatbot_model.keras")
WORDS_FILE    = os.path.join(MODELS_DIR, "words.npy")
CLASSES_FILE  = os.path.join(MODELS_DIR, "classes.npy")

TRAINING_CURVE_IMG  = os.path.join(OUTPUTS_DIR, "training_curve.png")
CONFUSION_MATRIX_IMG = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")

# ── Model Hyperparameters ──────────────────────────────────────────────────────
EPOCHS          = 200
BATCH_SIZE      = 5
LEARNING_RATE   = 0.001
DROPOUT_RATE    = 0.5
DENSE_1         = 128
DENSE_2         = 64
VALIDATION_SPLIT = 0.1

EARLY_STOPPING_PATIENCE = 30

# ── Inference ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
ERROR_THRESHOLD      = 0.25
TOP_RECOMMENDATIONS  = 5

# ── NLTK ───────────────────────────────────────────────────────────────────────
IGNORE_CHARS = ["?", "!", ".", ",", ";", ":", "'", '"', "-", "/", "\\"]

# ── Mood → Genre mapping ───────────────────────────────────────────────────────
MOOD_GENRE_MAP = {
    "funny":     ["Comedy"],
    "romantic":  ["Romance"],
    "emotional": ["Drama"],
    "suspense":  ["Thriller", "Mystery"],
    "energetic": ["Action", "Adventure"],
    "scary":     ["Horror"],
    "inspiring": ["Biography", "Drama"],
    "adventurous": ["Adventure", "Action"],
    "sad":       ["Drama", "Romance"],
    "happy":     ["Comedy", "Family"],
}

# ── Genre → Mood mapping ───────────────────────────────────────────────────────
GENRE_MOOD_MAP = {
    "comedy":    ["funny", "happy"],
    "romance":   ["romantic", "happy"],
    "drama":     ["emotional", "sad"],
    "thriller":  ["suspense"],
    "action":    ["energetic", "adventurous"],
    "horror":    ["scary"],
    "biography": ["inspiring", "emotional"],
    "adventure": ["adventurous", "energetic"],
    "family":    ["happy", "funny"],
    "mystery":   ["suspense"],
}

# ── App Settings ───────────────────────────────────────────────────────────────
APP_TITLE       = "🎬 Bollywood Movie Recommendation Chatbot"
APP_SUBTITLE    = "Your AI-powered Bollywood movie guide"
CHATBOT_NAME    = "CineBot"
MAX_HISTORY     = 20

for _dir in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
