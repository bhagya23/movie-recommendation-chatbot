"""
utils.py — Shared utility helpers for the Movie Recommendation Chatbot.
"""

import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    INTENTS_FILE, MOVIES_FILE, CSV_FILE,
    MOOD_GENRE_MAP, GENRE_MOOD_MAP
)

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
def load_json(path: str) -> dict | list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def save_json(data: dict | list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON → %s", path)


# ── Dataset loading ────────────────────────────────────────────────────────────
def load_intents() -> dict:
    return load_json(INTENTS_FILE)


def load_movies() -> list:
    return load_json(MOVIES_FILE)


# ── CSV → movies.json conversion ───────────────────────────────────────────────
GENRE_NORMALIZATION = {
    "romantic": "Romance",
    "romance":  "Romance",
    "action":   "Action",
    "comedy":   "Comedy",
    "drama":    "Drama",
    "thriller": "Thriller",
    "horror":   "Horror",
    "sci-fi":   "Sci-Fi",
    "science fiction": "Sci-Fi",
    "adventure":"Adventure",
    "biography":"Biography",
    "family":   "Family",
    "mystery":  "Mystery",
    "musical":  "Musical",
    "historical":"Historical",
    "sport":    "Sports",
    "sports":   "Sports",
    "crime":    "Crime",
    "fantasy":  "Fantasy",
    "animation":"Animation",
}

LANGUAGE_NORMALIZATION = {
    "hindi": "Hindi",
    "english": "English",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
    "bengali": "Bengali",
    "marathi": "Marathi",
    "punjabi": "Punjabi",
}


def normalize_genre(genre_raw: str) -> str:
    if pd.isna(genre_raw):
        return "Unknown"
    normalized = GENRE_NORMALIZATION.get(genre_raw.strip().lower(), genre_raw.strip().title())
    return normalized


def normalize_language(lang_raw: str) -> str:
    if pd.isna(lang_raw):
        return "Hindi"
    return LANGUAGE_NORMALIZATION.get(lang_raw.strip().lower(), lang_raw.strip().title())


def genre_to_moods(genre: str) -> list:
    return GENRE_MOOD_MAP.get(genre.lower(), [])


def build_movies_json(csv_path: str = CSV_FILE, out_path: str = MOVIES_FILE) -> list:
    """Convert bollywood_movies.csv → movies.json with normalized fields."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Movie dataset CSV not found: {csv_path}") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Failed to parse movie CSV {csv_path}: {exc}") from exc
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["title"], inplace=True)

    # Flexible column detection
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ("title", "movie", "movie_name", "name", "film"):
            col_map["title"] = col
        elif cl in ("genre", "genres", "category"):
            col_map["genre"] = col
        elif cl in ("language", "lang"):
            col_map["language"] = col
        elif cl in ("year", "release_year", "release year"):
            col_map["year"] = col
        elif cl in ("imdb", "imdb_rating", "rating", "score"):
            col_map["imdb"] = col
        elif cl in ("actor", "actors", "cast", "stars", "lead_actor"):
            col_map["actors"] = col
        elif cl in ("family_safe", "family", "family_friendly"):
            col_map["family_safe"] = col
        elif cl in ("popularity", "votes", "vote_count"):
            col_map["popularity"] = col

    movies = []
    for _, row in df.iterrows():
        title = str(row.get(col_map.get("title", "title"), "")).strip()
        if not title or title.lower() == "nan":
            continue

        genre_raw  = str(row.get(col_map.get("genre",    "genre"), "Drama"))
        lang_raw   = str(row.get(col_map.get("language", "language"), "Hindi"))
        year_raw   = row.get(col_map.get("year", "year"), 2000)
        imdb_raw   = row.get(col_map.get("imdb", "imdb"), 5.0)
        actors_raw = str(row.get(col_map.get("actors", "actors"), ""))
        pop_raw    = row.get(col_map.get("popularity", "popularity"), 0)

        genre    = normalize_genre(genre_raw)
        language = normalize_language(lang_raw)

        try:
            year = int(float(str(year_raw)))
        except (ValueError, TypeError):
            year = 2000

        try:
            imdb = round(float(str(imdb_raw).replace(",", ".")), 1)
        except (ValueError, TypeError):
            imdb = 5.0

        actors = [a.strip() for a in re.split(r"[|,/]", actors_raw) if a.strip() and a.strip().lower() != "nan"]

        try:
            popularity = int(float(str(pop_raw))) if pop_raw and str(pop_raw).lower() != "nan" else 0
        except (ValueError, TypeError):
            popularity = 0

        family_safe_raw = row.get(col_map.get("family_safe", "family_safe"), None)
        if family_safe_raw is not None and str(family_safe_raw).lower() not in ("nan", ""):
            family_safe = str(family_safe_raw).lower() in ("true", "1", "yes", "y")
        else:
            family_safe = genre.lower() in ("family", "comedy", "animation")

        moods = genre_to_moods(genre)

        movies.append({
            "title":       title,
            "genre":       genre,
            "mood":        moods,
            "language":    language,
            "year":        year,
            "imdb":        imdb,
            "actors":      actors,
            "family_safe": family_safe,
            "popularity":  popularity,
        })

    save_json(movies, out_path)
    logger.info("Built movies.json with %d entries", len(movies))
    return movies


# ── Entity extraction from user query ─────────────────────────────────────────
YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")

MOOD_KEYWORDS = {
    "funny":      ["funny", "comedy", "laugh", "hilarious", "humour", "humor"],
    "romantic":   ["romantic", "romance", "love", "lovey", "couple", "valentines"],
    "emotional":  ["emotional", "cry", "sad", "touching", "drama", "tearjerker"],
    "suspense":   ["suspense", "thriller", "mystery", "suspenseful", "tense"],
    "energetic":  ["action", "fight", "energetic", "adrenaline", "adventure", "exciting"],
    "scary":      ["scary", "horror", "spooky", "frightening", "creepy", "ghost"],
    "inspiring":  ["inspiring", "motivational", "biography", "biopic", "real story"],
    "adventurous":["adventure", "adventurous", "quest", "journey", "exploration"],
    "happy":      ["happy", "cheerful", "joyful", "light-hearted"],
    "sad":        ["sad", "melancholy", "depressing", "heartbreaking"],
}

GENRE_KEYWORDS = {
    "Comedy":    ["comedy", "funny", "humor", "laugh"],
    "Action":    ["action", "fight", "war", "battle"],
    "Romance":   ["romance", "romantic", "love story", "love"],
    "Drama":     ["drama", "emotional", "serious"],
    "Thriller":  ["thriller", "suspense", "mystery"],
    "Horror":    ["horror", "scary", "spooky", "ghost"],
    "Family":    ["family", "kids", "children", "family-friendly"],
    "Biography": ["biopic", "biography", "real story", "based on true"],
    "Sci-Fi":    ["sci-fi", "science fiction", "space", "robot"],
    "Adventure": ["adventure", "journey", "quest"],
    "Crime":     ["crime", "heist", "gangster"],
    "Musical":   ["musical", "music", "dance", "singing"],
}

LANGUAGE_KEYWORDS = {
    "Hindi":     ["hindi", "bollywood", "hindustani"],
    "English":   ["english", "hollywood"],
    "Tamil":     ["tamil", "kollywood"],
    "Telugu":    ["telugu", "tollywood"],
    "Bengali":   ["bengali", "bangla"],
    "Kannada":   ["kannada"],
    "Malayalam": ["malayalam"],
    "Punjabi":   ["punjabi"],
    "Marathi":   ["marathi"],
}

KNOWN_ACTORS = [
    "shah rukh khan", "srk", "salman khan", "aamir khan", "amitabh bachchan",
    "deepika padukone", "priyanka chopra", "katrina kaif", "kareena kapoor",
    "hrithik roshan", "ranveer singh", "ranbir kapoor", "akshay kumar",
    "ajay devgn", "vidya balan", "taapsee pannu", "kangana ranaut",
    "varun dhawan", "tiger shroff", "shahid kapoor", "irrfan khan",
    "nawazuddin siddiqui", "ayushmann khurrana", "rajkummar rao",
    "sushant singh rajput", "anushka sharma", "alia bhatt", "sonam kapoor",
    "john abraham", "sunny deol", "bobby deol", "anil kapoor", "jackie shroff",
]


def extract_entities(query: str) -> dict:
    """Extract mood, genre, year, language, actor from a free-text query."""
    q = query.lower()
    entities = {"mood": None, "genre": None, "year": None, "language": None, "actor": None}

    # Year
    year_match = YEAR_PATTERN.search(q)
    if year_match:
        entities["year"] = int(year_match.group())

    # Mood
    for mood, kws in MOOD_KEYWORDS.items():
        if any(kw in q for kw in kws):
            entities["mood"] = mood
            break

    # Genre
    for genre, kws in GENRE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            entities["genre"] = genre
            break

    # Language
    for lang, kws in LANGUAGE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            entities["language"] = lang
            break

    # Actor
    for actor in KNOWN_ACTORS:
        if actor in q:
            entities["actor"] = actor.title()
            break

    return entities


# ── Misc ───────────────────────────────────────────────────────────────────────
def format_movie_card(movie: dict) -> str:
    actors = ", ".join(movie.get("actors", [])[:3]) or "N/A"
    return (
        f"🎬 **{movie['title']}** ({movie['year']})\n"
        f"  ⭐ IMDb: {movie['imdb']}  |  🎭 Genre: {movie['genre']}  |  🌐 Language: {movie['language']}\n"
        f"  👤 Cast: {actors}"
    )
