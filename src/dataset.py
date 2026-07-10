"""
dataset.py — Movie catalogue pipeline (matches the academic notebook).

Downloads two Kaggle datasets via kagglehub (with a local raw/ fallback),
standardizes each row into the chatbot schema, deduplicates, and saves a
processed catalogue to data/processed/movies.json.

Schema per movie:
    title, year, genre, language, mood, age_group, imdb, description, poster, source
"""

import os
import sys
import json
import re
import ast
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RAW_DATA_DIR, MOVIES_FILE,
    GENRE_ALIAS, LANGUAGE_ALIAS, PRIMARY_GENRE_PRIORITY, GENRE_TO_MOODS,
    BOLLYWOOD_DATASET_SLUG, TMDB_DATASET_SLUG,
    BOLLYWOOD_CSV_NAME, TMDB_CSV_NAME,
)
from src.utils import get_logger

logger = get_logger("dataset")


# ── Dataset acquisition ─────────────────────────────────────────────────────────

def download_dataset(slug: str) -> Optional[Path]:
    """Download a public Kaggle dataset via kagglehub; return local folder or None."""
    try:
        import kagglehub
        logger.info("Downloading %s via kagglehub ...", slug)
        return Path(kagglehub.dataset_download(slug))
    except Exception as exc:  # offline / no credentials / not installed
        logger.warning("kagglehub download failed for %s: %s", slug, exc)
        return None


def find_file(root: Path, target_name: str) -> Optional[Path]:
    """Find a file by exact name (case-insensitive) anywhere under root."""
    if root is None:
        return None
    target_lower = target_name.lower()
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() == target_lower:
            return path
    return None


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read a CSV with a couple of common encoding/parsing fallbacks."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
    except Exception:
        return pd.read_csv(path, low_memory=False)


def _resolve_csv(slug: str, csv_name: str, local_fallbacks: List[str]) -> Optional[Path]:
    """Locate a dataset CSV: prefer kagglehub download, fall back to data/raw/."""
    root = download_dataset(slug)
    found = find_file(root, csv_name) if root else None
    if found:
        return found
    # Local fallbacks in data/raw/
    for name in [csv_name, *local_fallbacks]:
        candidate = Path(RAW_DATA_DIR) / name
        if candidate.exists():
            logger.info("Using local dataset fallback: %s", candidate)
            return candidate
    return None


# ── Row-level parsing helpers (ported from the academic notebook) ────────────────

def first_present(row: pd.Series, candidates, default=""):
    for col in candidates:
        if col in row.index:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                return value
    return default


def normalize_token(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return GENRE_ALIAS.get(s, s)


def parse_tokens(value) -> List[str]:
    """Parse list-like genre/language fields into normalized tokens."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    items = []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        s = str(value).strip()
        if not s:
            return []
        parsed = None
        if s.startswith("[") or s.startswith("{"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(s)
                    break
                except Exception:
                    pass
        if parsed is not None:
            items = [parsed]
        else:
            names = re.findall(r"""['"]name['"]\s*:\s*['"]([^'"]+)['"]""", s)
            items = names if names else re.split(r"[|,;/]+", s)

    tokens: List[str] = []

    def collect(obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            for key in ("name", "genre", "title", "value"):
                if key in obj and obj[key]:
                    collect(obj[key])
                    return
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                collect(item)
        else:
            token = normalize_token(obj)
            if token:
                tokens.append(token)

    collect(items)

    seen, unique = set(), []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def choose_primary_genre(genres: List[str]) -> str:
    if not genres:
        return "drama"
    for preferred in PRIMARY_GENRE_PRIORITY:
        if preferred in genres:
            return preferred
    return genres[0]


def parse_year(value, default=2000) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    s = str(value).strip()
    if not s:
        return default
    if re.fullmatch(r"\d{4}", s):
        return int(s)
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return int(dt.year)
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else default


def parse_rating(value, default=7.0) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    s = str(value).strip().replace("%", "")
    if not s:
        return default
    try:
        rating = float(s)
    except Exception:
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        if not nums:
            return default
        rating = float(nums[0])
    if 10 < rating <= 100:
        rating /= 10.0
    return round(max(0.0, min(rating, 10.0)), 1)


def parse_poster(row: pd.Series) -> str:
    poster = first_present(
        row, ["poster", "Poster", "poster_url", "Poster URL", "poster_path", "Poster Path"], ""
    )
    if not poster:
        return ""
    poster = str(poster).strip()
    if poster.startswith("http"):
        return poster
    if poster.startswith("/"):
        return f"https://image.tmdb.org/t/p/w500{poster}"
    return poster


def infer_language(row: pd.Series, source: str) -> str:
    language = first_present(
        row, ["language", "Language", "original_language", "Original Language", "lang", "Lang"], ""
    )
    if language:
        s = str(language).strip().lower()
        if s in LANGUAGE_ALIAS:
            return LANGUAGE_ALIAS[s]
        if s in ("english", "hindi", "korean"):
            return s
    return "hindi" if source == "bollywood" else "english"


def infer_moods(genre: str, title: str = "", description: str = "") -> List[str]:
    moods = list(GENRE_TO_MOODS.get(genre, ["thoughtful"]))
    text = f"{title} {description}".lower()
    if any(w in text for w in ["funny", "comedy", "humor", "humour", "laugh"]):
        moods = sorted(set(moods + ["funny", "happy"]))
    if any(w in text for w in ["inspire", "motiv", "uplift", "success", "achievement"]):
        moods = sorted(set(moods + ["inspired"]))
    if any(w in text for w in ["heart", "feel good", "warming", "emotional"]):
        moods = sorted(set(moods + ["heartwarming", "emotional"]))
    return list(dict.fromkeys(moods))


def infer_age_groups(genre: str, row: pd.Series) -> List[str]:
    certificate = str(first_present(
        row, ["certificate", "Certification", "certification", "mpaa_rating",
              "rating_certification", "age_rating"], "")).lower()
    if any(x in certificate for x in ["g", "u", "all ages"]):
        return ["kids", "teenager", "adult"]
    if any(x in certificate for x in ["pg-13", "12a", "ua", "u/a", "13", "15", "16"]):
        return ["teenager", "adult"]
    if any(x in certificate for x in ["r", "18", "a", "adult"]):
        return ["adult"]
    if genre == "animation":
        return ["kids", "teenager", "adult"]
    if genre == "horror":
        return ["adult"]
    return ["teenager", "adult"]


def row_to_movie(row: pd.Series, source: str) -> Optional[dict]:
    title = str(first_present(
        row, ["title", "Title", "movie_title", "Movie Title", "name", "Name",
              "film_name", "Film Name"], "")).strip()
    if not title:
        return None

    year = parse_year(first_present(
        row, ["year", "Year", "release_year", "Release Year", "released_year",
              "release_date", "Release Date", "date"], 2000))
    genre_tokens = parse_tokens(first_present(
        row, ["genre", "Genre", "genres", "Genres", "category", "Category", "type", "Type"], ""))
    if not genre_tokens:
        genre_tokens = ["drama"]
    genre = choose_primary_genre(genre_tokens)

    language = infer_language(row, source)
    description = str(first_present(
        row, ["description", "Description", "overview", "Overview", "plot", "Plot",
              "story", "Synopsis", "summary", "Summary"], "")).strip()
    poster = parse_poster(row)
    imdb = parse_rating(first_present(
        row, ["imdb", "IMDB", "imdb_rating", "IMDb Rating", "vote_average",
              "rating", "Rating", "score", "Score"], 7.0))
    moods = infer_moods(genre, title=title, description=description)
    age_group = infer_age_groups(genre, row)

    if year < 1900 or year > 2100:
        year = 2000

    return {
        "title": title,
        "year": year,
        "genre": genre,
        "language": language,
        "mood": moods,
        "age_group": age_group,
        "imdb": imdb,
        "description": description,
        "poster": poster,
        "source": source,
    }


def dedupe_movies(records: List[dict]) -> List[dict]:
    seen, deduped = set(), []
    for movie in records:
        key = (movie["title"].strip().lower(), movie["year"], movie["language"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(movie)
    return deduped


# ── Catalogue build / load ───────────────────────────────────────────────────────

def build_processed_catalog(output_path: str = MOVIES_FILE) -> List[dict]:
    """Download, standardize, dedupe, and save the movie catalogue."""
    logger.info("=" * 60)
    logger.info("Building processed movie catalogue")
    logger.info("=" * 60)

    bollywood_csv = _resolve_csv(
        BOLLYWOOD_DATASET_SLUG, BOLLYWOOD_CSV_NAME, ["bollywood_movies.csv"])
    tmdb_csv = _resolve_csv(
        TMDB_DATASET_SLUG, TMDB_CSV_NAME, ["tmdb_movie_dataset.csv"])

    if bollywood_csv is None and tmdb_csv is None:
        raise FileNotFoundError(
            "No datasets available. Install kagglehub (with Kaggle credentials) "
            "or place the CSVs in data/raw/."
        )

    movies: List[dict] = []

    if bollywood_csv is not None:
        df = read_csv_flexible(bollywood_csv)
        logger.info("Bollywood rows: %d (%s)", len(df), bollywood_csv.name)
        for _, row in df.iterrows():
            rec = row_to_movie(row, source="bollywood")
            if rec:
                movies.append(rec)

    if tmdb_csv is not None:
        df = read_csv_flexible(tmdb_csv)
        logger.info("TMDB rows: %d (%s)", len(df), tmdb_csv.name)
        for _, row in df.iterrows():
            rec = row_to_movie(row, source="tmdb")
            if rec:
                movies.append(rec)

    movies = dedupe_movies(movies)
    movies.sort(key=lambda m: (m["year"], m["imdb"]), reverse=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("Catalogue build complete: %d movies -> %s", len(movies), output_path)
    logger.info("=" * 60)
    return movies


def load_processed_catalog(path: str = MOVIES_FILE) -> List[dict]:
    """Load the processed catalogue, building it if missing."""
    if not os.path.exists(path):
        logger.warning("Processed catalogue not found at %s — building it now.", path)
        return build_processed_catalog(path)
    with open(path, "r", encoding="utf-8") as f:
        movies = json.load(f)
    logger.info("Loaded processed catalogue: %d movies", len(movies))
    return movies


if __name__ == "__main__":
    built = build_processed_catalog()
    print(f"Built catalogue with {len(built)} movies")
