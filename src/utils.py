"""
utils.py — Shared utility helpers for the Movie Recommendation Chatbot.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INTENTS_FILE, MOVIES_FILE


# ── Logging ────────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


logger = get_logger("utils")


# ── JSON helpers ───────────────────────────────────────────────────────────────
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON → %s", path)


# ── Dataset loading ────────────────────────────────────────────────────────────
def load_intents() -> dict:
    return load_json(INTENTS_FILE)


def load_movies() -> list:
    return load_json(MOVIES_FILE)


# ── Presentation ───────────────────────────────────────────────────────────────
def format_movie_card(movie: dict) -> str:
    moods = ", ".join(movie.get("mood", [])[:3]) or "N/A"
    return (
        f"🎬 **{movie['title']}** ({movie['year']})\n"
        f"  ⭐ IMDb: {movie['imdb']}  |  🎭 Genre: {movie['genre']}  |  🌐 Language: {movie['language']}\n"
        f"  🎭 Mood: {moods}"
    )
