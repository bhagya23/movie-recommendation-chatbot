"""
config.py — Central configuration for the Movie Recommendation Chatbot.
All path constants, model hyperparameters, and app settings live here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR  = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR   = os.path.join(OUTPUTS_DIR, "reports")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")

INTENTS_FILE         = os.path.join(DATA_DIR, "intents.json")
MOVIES_FILE          = os.path.join(PROCESSED_DATA_DIR, "movies.json")

# Kaggle dataset slugs (downloaded via kagglehub, matching the academic notebook)
BOLLYWOOD_DATASET_SLUG = "mitesh58/bollywood-movie-dataset"
TMDB_DATASET_SLUG      = "aayushsoni4/tmdb-5000-movie-dataset-with-ratings"
BOLLYWOOD_CSV_NAME     = "BollywoodMovieDetail.csv"
TMDB_CSV_NAME          = "tmdb_movie_dataset.csv"

MODEL_FILE           = os.path.join(MODELS_DIR, "movie_chatbot_model.keras")
WORDS_FILE           = os.path.join(MODELS_DIR, "movie_words.npy")
CLASSES_FILE         = os.path.join(MODELS_DIR, "movie_classes.npy")

TRAINING_CURVE_IMG  = os.path.join(OUTPUTS_DIR, "training_curve.png")
CONFUSION_MATRIX_IMG = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")

# ── Model Hyperparameters ──────────────────────────────────────────────────────
EPOCHS          = 200
BATCH_SIZE      = 8
LEARNING_RATE   = 0.001
DROPOUT_1       = 0.40
DROPOUT_2       = 0.30
DROPOUT_3       = 0.20
DENSE_1         = 256
DENSE_2         = 128
DENSE_3         = 64

# Stratified 70/15/15 train/val/test split (per the academic report)
TEST_SIZE       = 0.30   # 30% held out, then split 50/50 into val/test
SEED            = 42

EARLY_STOPPING_PATIENCE = 25
REDUCE_LR_FACTOR        = 0.5
REDUCE_LR_PATIENCE      = 8
REDUCE_LR_MIN_LR        = 1e-5

# ── Inference ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40
TOP_RECOMMENDATIONS  = 3

# ── Catalogue normalization (matches the academic notebook) ─────────────────────
GENRE_ALIAS = {
    "science fiction": "scifi", "sci fi": "scifi", "sci fi film": "scifi",
    "sci fi movie": "scifi", "rom com": "romance", "romcom": "romance",
    "romantic comedy": "romance", "animated": "animation", "cartoon": "animation",
    "family": "animation", "adventure": "action", "crime": "thriller",
    "mystery": "thriller", "biography": "drama", "documentary": "drama",
    "history": "drama", "historical": "drama",
}

LANGUAGE_ALIAS = {
    "en": "english", "eng": "english", "english": "english",
    "hi": "hindi", "hin": "hindi", "hindi": "hindi",
    "ko": "korean", "kor": "korean", "korean": "korean",
}

PRIMARY_GENRE_PRIORITY = [
    "horror", "thriller", "action", "comedy", "drama", "romance", "scifi", "animation",
]

GENRE_TO_MOODS = {
    "action":    ["excited", "adventurous"],
    "comedy":    ["happy", "funny"],
    "drama":     ["emotional", "thoughtful"],
    "thriller":  ["thrilled", "dark"],
    "horror":    ["scared", "dark"],
    "romance":   ["romantic", "happy"],
    "scifi":     ["thoughtful", "excited"],
    "animation": ["happy", "heartwarming"],
}

# ── Intent tag → movie filter rules (matches the academic notebook) ─────────────
INTENT_TAG_MAP = {
    "mood_happy":        {"moods": ["happy"],     "genres": []},
    "mood_sad":          {"moods": ["emotional"], "genres": ["drama", "comedy"]},
    "mood_romantic":     {"moods": ["romantic"],  "genres": ["romance"]},
    "mood_excited":      {"moods": ["excited"],   "genres": ["action"]},
    "mood_scared":       {"moods": ["scared"],    "genres": ["horror"]},
    "genre_action":      {"moods": [],            "genres": ["action"]},
    "genre_comedy":      {"moods": ["funny"],     "genres": ["comedy"]},
    "genre_drama":       {"moods": [],            "genres": ["drama"]},
    "genre_thriller":    {"moods": [],            "genres": ["thriller"]},
    "genre_scifi":       {"moods": [],            "genres": ["scifi"]},
    "genre_horror":      {"moods": ["scared"],    "genres": ["horror"]},
    "genre_animation":   {"moods": [],            "genres": ["animation"]},
    "lang_hindi":        {"moods": [], "genres": [], "lang": "hindi"},
    "lang_english":      {"moods": [], "genres": [], "lang": "english"},
    "lang_korean":       {"moods": [], "genres": [], "lang": "korean"},
    "age_teenager":      {"moods": [], "genres": [], "age": "teenager"},
    "age_kids":          {"moods": [], "genres": [], "age": "kids"},
    "age_adult":         {"moods": [], "genres": [], "age": "adult"},
    "top_rated":         {"moods": [], "genres": [], "min_imdb": 8.0},
    "recent_movies":     {"moods": [], "genres": [], "min_year": 2018},
    "recommend_general": {"moods": [], "genres": []},
}

# Keyword-based fallback when classifier confidence is low
KEYWORD_TO_TAG = {
    "hindi": "lang_hindi", "bollywood": "lang_hindi", "korean": "lang_korean",
    "english": "lang_english", "hollywood": "lang_english", "action": "genre_action",
    "comedy": "genre_comedy", "funny": "genre_comedy", "drama": "genre_drama",
    "thriller": "genre_thriller", "scifi": "genre_scifi", "sci-fi": "genre_scifi",
    "horror": "genre_horror", "animation": "genre_animation", "kids": "age_kids",
    "teen": "age_teenager", "teenager": "age_teenager", "adult": "age_adult",
    "recent": "recent_movies", "latest": "recent_movies", "top rated": "top_rated",
}

DIRECT_RESPONSE_TAGS = {"greeting", "goodbye", "thanks", "help"}

# ── App Settings ───────────────────────────────────────────────────────────────
APP_TITLE       = "🎬 Movie Recommendation Chatbot"
APP_SUBTITLE    = "Domain-Oriented NLP-Based Recommendation System"
CHATBOT_NAME    = "CineBot"
MAX_HISTORY     = 20

for _dir in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
