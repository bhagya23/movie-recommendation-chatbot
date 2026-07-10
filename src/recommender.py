"""
recommender.py — Intent-driven movie filtering (matches the academic notebook).

Given a predicted intent tag, the engine looks up the filtering rules in
INTENT_TAG_MAP (genre / mood / language / age group / min IMDb / min year),
filters the processed catalogue, and returns the top-k movies sorted by IMDb.
"""

import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MOVIES_FILE, TOP_RECOMMENDATIONS, INTENT_TAG_MAP
from src.dataset import load_processed_catalog
from src.utils import get_logger

logger = get_logger("recommender")


class RecommendationEngine:
    """Filter the movie catalogue according to a predicted intent tag."""

    def __init__(self):
        self.movies: List[dict] = load_processed_catalog()
        if not self.movies:
            raise ValueError("Processed catalogue is empty. Build it via src/dataset.py.")
        logger.info("Loaded %d movies into recommendation engine.", len(self.movies))

    # ── Core filtering ──────────────────────────────────────────────────────────

    def filter_for_tag(self, tag: str, k: int = TOP_RECOMMENDATIONS) -> List[dict]:
        """Filter movies using the intent tag's rules; sort by IMDb desc; return top-k."""
        cfg = INTENT_TAG_MAP.get(tag, {})
        results: List[dict] = []

        for movie in self.movies:
            ok = True
            if cfg.get("genres") and movie.get("genre") not in cfg["genres"]:
                ok = False
            if cfg.get("moods"):
                movie_moods = movie.get("mood", []) or []
                if not any(mood in movie_moods for mood in cfg["moods"]):
                    ok = False
            if cfg.get("lang") and movie.get("language") != cfg["lang"]:
                ok = False
            if cfg.get("age") and cfg["age"] not in (movie.get("age_group", []) or []):
                ok = False
            if cfg.get("min_imdb") and movie.get("imdb", 0) < cfg["min_imdb"]:
                ok = False
            if cfg.get("min_year") and movie.get("year", 0) < cfg["min_year"]:
                ok = False
            if ok:
                results.append(movie)

        if not results:
            results = list(self.movies)

        results = sorted(results, key=lambda m: m.get("imdb", 0), reverse=True)
        return results[:k]

    # ── Explanation ───────────────────────────────────────────────────────────

    @staticmethod
    def explain(tag: str) -> str:
        """Return a short human-readable reason for the recommendation."""
        cfg = INTENT_TAG_MAP.get(tag, {})
        parts: List[str] = []
        if cfg.get("genres"):
            parts.append("genre: " + ", ".join(cfg["genres"]))
        if cfg.get("moods"):
            parts.append("mood: " + ", ".join(cfg["moods"]))
        if cfg.get("lang"):
            parts.append(f"{cfg['lang']} language")
        if cfg.get("age"):
            parts.append(f"{cfg['age']} audience")
        if cfg.get("min_imdb"):
            parts.append(f"IMDb ≥ {cfg['min_imdb']}")
        if cfg.get("min_year"):
            parts.append(f"released {cfg['min_year']}+")
        if not parts:
            return "Here are some recommendations for you."
        return "Recommended based on " + "; ".join(parts) + "."

    def recommend(self, tag: str, k: int = TOP_RECOMMENDATIONS) -> Tuple[List[dict], str]:
        """Convenience: return (movies, explanation) for an intent tag."""
        return self.filter_for_tag(tag, k), self.explain(tag)


# ── Module-level convenience ─────────────────────────────────────────────────────

_engine: Optional[RecommendationEngine] = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine


def filter_movies_for_tag(tag: str, k: int = TOP_RECOMMENDATIONS) -> List[dict]:
    return get_engine().filter_for_tag(tag, k)


if __name__ == "__main__":
    engine = RecommendationEngine()
    for demo_tag in ["genre_action", "mood_romantic", "lang_korean", "top_rated", "recent_movies"]:
        movies, explanation = engine.recommend(demo_tag, k=3)
        print(f"\n[{demo_tag}] {explanation}")
        for m in movies:
            print(f"  - {m['title']} ({m['year']}, {m['genre']}, IMDb {m['imdb']})")
