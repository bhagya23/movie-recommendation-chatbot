"""
recommendation_engine.py — Multi-filter Bollywood movie recommendation engine.
Supports filtering by genre, mood, language, year, actor, IMDb rating.
Also implements TF-IDF + cosine similarity for content-based similarity.
"""

import os
import sys
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MOVIES_FILE, CSV_FILE, TOP_RECOMMENDATIONS, MOOD_GENRE_MAP
)
from src.utils import (
    get_logger, load_movies, build_movies_json,
    normalize_genre, normalize_language
)

logger = get_logger("recommendation_engine")


class RecommendationEngine:
    """
    Bollywood movie recommendation engine supporting:
    - Genre-based filtering
    - Mood-based filtering (via mood→genre map)
    - Language-based filtering
    - IMDb rating threshold
    - Year-based filtering
    - Actor-based filtering
    - Combined multi-filter
    - TF-IDF cosine similarity (content-based)
    """

    def __init__(self):
        self.movies: list[dict] = []
        self.df: pd.DataFrame = pd.DataFrame()
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._load_movies()
        self._build_tfidf_index()

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_movies(self) -> None:
        """Load movies from movies.json, auto-generate from CSV if missing."""
        if not os.path.exists(MOVIES_FILE):
            logger.info("movies.json not found, generating from CSV...")
            build_movies_json()

        self.movies = load_movies()
        if not self.movies:
            logger.warning("movies.json is empty — regenerating from CSV.")
            self.movies = build_movies_json()

        self.df = pd.DataFrame(self.movies)
        self.df["imdb"]   = pd.to_numeric(self.df["imdb"],   errors="coerce").fillna(5.0)
        self.df["year"]   = pd.to_numeric(self.df["year"],   errors="coerce").fillna(2000).astype(int)
        self.df["popularity"] = pd.to_numeric(
            self.df.get("popularity", pd.Series([0]*len(self.df))),
            errors="coerce"
        ).fillna(0).astype(int)
        logger.info("Loaded %d movies into recommendation engine.", len(self.df))

    # ── TF-IDF index ───────────────────────────────────────────────────────────

    def _build_tfidf_index(self) -> None:
        """Build TF-IDF matrix over combined movie metadata for similarity search."""
        if self.df.empty:
            return
        corpus = self.df.apply(self._movie_to_document, axis=1).tolist()
        self._tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        logger.info("TF-IDF index built: %s", self._tfidf_matrix.shape)

    @staticmethod
    def _movie_to_document(row) -> str:
        actors = " ".join(row.get("actors", []) or [])
        moods  = " ".join(row.get("mood",   []) or [])
        return (
            f"{row.get('title','')} {row.get('genre','')} {row.get('language','')} "
            f"{row.get('year','')} {actors} {moods}"
        ).lower()

    # ── Core filter helpers ────────────────────────────────────────────────────

    def _rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by IMDb desc → popularity desc → year desc."""
        return df.sort_values(
            ["imdb", "popularity", "year"],
            ascending=[False, False, False]
        )

    def _top_n(self, df: pd.DataFrame, n: int = TOP_RECOMMENDATIONS) -> list[dict]:
        ranked = self._rank(df).head(n)
        return ranked.to_dict(orient="records")

    # ── Public filter methods ──────────────────────────────────────────────────

    def recommend_by_genre(
        self, genre: str, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        genre_norm = normalize_genre(genre)
        mask = self.df["genre"].str.lower() == genre_norm.lower()
        result = self.df[mask]
        if result.empty:
            mask = self.df["genre"].str.lower().str.contains(genre.lower(), na=False)
            result = self.df[mask]
        return self._top_n(result, n)

    def recommend_by_mood(
        self, mood: str, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        target_genres = MOOD_GENRE_MAP.get(mood.lower(), [])
        if not target_genres:
            return []
        mask = self.df["genre"].str.lower().isin([g.lower() for g in target_genres])
        # Also check mood field if present
        mood_mask = self.df["mood"].apply(
            lambda m: isinstance(m, list) and mood.lower() in [x.lower() for x in m]
        )
        combined = self.df[mask | mood_mask]
        if combined.empty:
            combined = self.df[mask]
        return self._top_n(combined, n)

    def recommend_by_rating(
        self, min_rating: float = 7.0, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        result = self.df[self.df["imdb"] >= min_rating]
        return self._top_n(result, n)

    def recommend_by_language(
        self, language: str, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        lang_norm = normalize_language(language)
        mask = self.df["language"].str.lower() == lang_norm.lower()
        result = self.df[mask]
        return self._top_n(result, n)

    def recommend_by_year(
        self,
        year: Optional[int] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        df = self.df.copy()
        if year is not None:
            df = df[df["year"] == year]
        if year_from is not None:
            df = df[df["year"] >= year_from]
        if year_to is not None:
            df = df[df["year"] <= year_to]
        return self._top_n(df, n)

    def recommend_by_actor(
        self, actor: str, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        actor_lower = actor.lower()
        mask = self.df["actors"].apply(
            lambda acts: isinstance(acts, list) and
            any(actor_lower in a.lower() for a in acts)
        )
        result = self.df[mask]
        return self._top_n(result, n)

    def recommend_family_friendly(
        self, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        mask = self.df["family_safe"].astype(str).str.lower().isin(["true", "1", "yes"])
        result = self.df[mask]
        return self._top_n(result, n)

    def recommend_popular(
        self, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        result = self.df.sort_values("popularity", ascending=False)
        return self._top_n(result, n)

    def recommend_latest(
        self, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        result = self.df.sort_values("year", ascending=False)
        return self._top_n(result, n)

    def recommend_classics(
        self, before_year: int = 2000, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        result = self.df[self.df["year"] < before_year]
        return self._top_n(result, n)

    # ── Combined multi-filter ──────────────────────────────────────────────────

    def recommend_combined(
        self,
        genre: Optional[str]    = None,
        mood: Optional[str]     = None,
        language: Optional[str] = None,
        year: Optional[int]     = None,
        year_from: Optional[int]= None,
        year_to: Optional[int]  = None,
        min_imdb: float         = 0.0,
        actor: Optional[str]    = None,
        family_safe: Optional[bool] = None,
        n: int                  = TOP_RECOMMENDATIONS,
    ) -> list[dict]:
        """
        Multi-criteria filtering with graceful fallback:
        If strict filter yields <3 results, relax the least-important filters.
        """
        df = self.df.copy()

        # ── Genre filter ──────────────────────────────────────────────────────
        if genre:
            genre_norm = normalize_genre(genre)
            g_mask = df["genre"].str.lower() == genre_norm.lower()
            if g_mask.sum() == 0:
                g_mask = df["genre"].str.lower().str.contains(genre.lower(), na=False)
            df = df[g_mask]

        # ── Mood filter ───────────────────────────────────────────────────────
        # When genre is set: additionally filter by mood field on already-filtered df
        # When genre is NOT set: expand mood → target genres, then also check mood field
        if mood:
            mood_lower = mood.lower()
            mood_field_mask = df["mood"].apply(
                lambda m: isinstance(m, list) and mood_lower in [x.lower() for x in m]
            )
            if genre:
                # Genre already applied — narrow further by mood field if enough results
                if mood_field_mask.sum() >= 2:
                    df = df[mood_field_mask]
                # else: keep genre-only results (mood is incompatible with this genre)
            else:
                # No genre — use MOOD_GENRE_MAP expansion + mood field
                target_genres = MOOD_GENRE_MAP.get(mood_lower, [])
                genre_exp_mask = df["genre"].str.lower().isin(
                    [g.lower() for g in target_genres]
                )
                combined_mood_mask = genre_exp_mask | mood_field_mask
                if combined_mood_mask.sum() >= 2:
                    df = df[combined_mood_mask]

        # ── Language filter ───────────────────────────────────────────────────
        if language:
            lang_norm = normalize_language(language)
            df = df[df["language"].str.lower() == lang_norm.lower()]

        if actor:
            actor_lower = actor.lower()
            df = df[df["actors"].apply(
                lambda acts: isinstance(acts, list) and
                any(actor_lower in a.lower() for a in acts)
            )]

        if year:
            df = df[df["year"] == year]
        if year_from:
            df = df[df["year"] >= year_from]
        if year_to:
            df = df[df["year"] <= year_to]

        if min_imdb > 0:
            df = df[df["imdb"] >= min_imdb]

        if family_safe is not None:
            df = df[df["family_safe"].astype(str).str.lower().isin(
                ["true", "1", "yes"] if family_safe else ["false", "0", "no"]
            )]

        # ── Fallback ladder: relax filters one by one (most restrictive first) ──
        def _apply_genre_mood(base_df):
            """Re-apply genre + mood filters on a fresh base dataframe."""
            d = base_df.copy()
            if genre:
                gn = normalize_genre(genre)
                gm = d["genre"].str.lower() == gn.lower()
                if gm.sum() == 0:
                    gm = d["genre"].str.lower().str.contains(genre.lower(), na=False)
                d = d[gm]
            if mood:
                ml = mood.lower()
                mfm = d["mood"].apply(
                    lambda m: isinstance(m, list) and ml in [x.lower() for x in m]
                )
                if genre:
                    if mfm.sum() >= 2:
                        d = d[mfm]
                else:
                    tg = MOOD_GENRE_MAP.get(ml, [])
                    gem = d["genre"].str.lower().isin([g.lower() for g in tg])
                    cmm = gem | mfm
                    if cmm.sum() >= 2:
                        d = d[cmm]
            return d

        # Step 1: relax family_safe
        if len(df) < 3 and family_safe is not None:
            logger.debug("Relaxing family_safe filter (%d results) — keeping genre/mood/language.", len(df))
            df = _apply_genre_mood(self.df)
            if language:
                df = df[df["language"].str.lower() == normalize_language(language).lower()]
            if min_imdb > 0:
                df = df[df["imdb"] >= min_imdb]

        # Step 2: relax language — only when zero results remain
        if df.empty and language:
            logger.debug("No results with language=%s — relaxing language filter.", language)
            df = _apply_genre_mood(self.df)
            if min_imdb > 0:
                df = df[df["imdb"] >= min_imdb]

        # Step 3: relax IMDb threshold — ONLY when zero results remain
        # (respect explicit high ratings set by the user, e.g. IMDb >= 9.0)
        if df.empty and min_imdb > 0:
            logger.debug("No results at IMDb >= %.1f — relaxing threshold.", min_imdb)
            df = _apply_genre_mood(self.df)
            if language:
                df = df[df["language"].str.lower() == normalize_language(language).lower()]

        # Step 4: final fallback — top-rated overall
        if df.empty:
            logger.debug("No matches with any filters — returning top-rated overall.")
            df = self.df.copy()

        return self._top_n(df, n)

    # ── TF-IDF Cosine Similarity ───────────────────────────────────────────────

    def recommend_similar(
        self, movie_title: str, n: int = TOP_RECOMMENDATIONS
    ) -> list[dict]:
        """Find movies similar to a given title using cosine similarity on TF-IDF."""
        if self._tfidf_matrix is None:
            logger.warning("TF-IDF index not built.")
            return []

        title_lower = movie_title.lower()
        mask = self.df["title"].str.lower().str.contains(title_lower, na=False)
        if not mask.any():
            logger.warning("Movie '%s' not found in database.", movie_title)
            return self.recommend_popular(n)

        idx = mask.idxmax()
        query_vec = self._tfidf_matrix[idx]
        cosine_scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        cosine_scores[idx] = -1  # Exclude the query movie itself

        top_indices = cosine_scores.argsort()[::-1][:n]
        results = self.df.iloc[top_indices].to_dict(orient="records")
        for i, r in enumerate(results):
            r["similarity_score"] = round(float(cosine_scores[top_indices[i]]), 3)
        return results

    # ── Intent-based dispatcher ────────────────────────────────────────────────

    def dispatch(
        self,
        intent: str,
        entities: dict,
        sidebar_filters: Optional[dict] = None,
        n: int = TOP_RECOMMENDATIONS,
    ) -> tuple[list[dict], str]:
        """
        Route an intent + extracted entities to the correct recommendation method.

        Returns:
            (movies_list, explanation_string)
        """
        sf = sidebar_filters or {}
        genre       = entities.get("genre")    or sf.get("genre")
        mood        = entities.get("mood")     or sf.get("mood")
        language    = entities.get("language") or sf.get("language")
        year        = entities.get("year")     or sf.get("year")
        actor       = entities.get("actor")    or sf.get("actor")
        min_imdb    = float(sf.get("min_imdb", 0.0))
        family_safe = True if sf.get("family_safe") else None
        year_from   = sf.get("year_from")
        year_to     = sf.get("year_to")

        explanation = ""

        # Helper: build explanation string from active filters
        def _explain(base: str) -> str:
            parts = [base]
            if year:               parts.append(f"year {year}")
            elif year_from or year_to:
                yr = f"{year_from or ''}–{year_to or ''}"
                parts.append(f"years {yr}")
            if language:           parts.append(f"{language} language")
            if actor:              parts.append(f"starring {actor}")
            if min_imdb:           parts.append(f"IMDb ≥ {min_imdb}")
            if family_safe:        parts.append("family-friendly")
            return "Recommended " + ", ".join(parts) + "."

        # Helper: apply secondary entity filters on top of a primary genre/mood filter
        def _with_extras(primary_genre: Optional[str] = None,
                         primary_mood: Optional[str] = None) -> list[dict]:
            extras = any([year, year_from, year_to, language, actor,
                          min_imdb > 0, family_safe is not None])
            if extras or (primary_genre and mood) or (primary_mood and genre):
                return self.recommend_combined(
                    genre=primary_genre or genre,
                    mood=primary_mood or mood,
                    language=language, year=year,
                    year_from=year_from, year_to=year_to,
                    min_imdb=min_imdb, actor=actor,
                    family_safe=family_safe, n=n
                )
            if primary_genre:
                return self.recommend_by_genre(primary_genre, n)
            if primary_mood:
                return self.recommend_by_mood(primary_mood, n)
            return self.recommend_popular(n)

        # -- Intent routing --
        if intent == "comedy_movies":
            movies = _with_extras(primary_genre="Comedy")
            explanation = _explain("based on Comedy genre")

        elif intent == "action_movies":
            movies = _with_extras(primary_genre="Action")
            explanation = _explain("based on Action genre")

        elif intent == "romantic_movies":
            movies = _with_extras(primary_genre="Romance")
            explanation = _explain("based on Romance genre")

        elif intent == "thriller_movies":
            movies = _with_extras(primary_genre="Thriller")
            explanation = _explain("based on Thriller genre")

        elif intent == "emotional_movies":
            movies = _with_extras(primary_mood="emotional")
            explanation = _explain("for emotional/drama mood")

        elif intent == "horror_movies":
            movies = _with_extras(primary_genre="Horror")
            explanation = _explain("based on Horror genre")

        elif intent == "family_movies":
            if any([year, language, actor, min_imdb > 0]):
                movies = self.recommend_combined(
                    family_safe=True, language=language,
                    year=year, min_imdb=min_imdb, actor=actor, n=n
                )
            else:
                movies = self.recommend_family_friendly(n)
            explanation = _explain("family-friendly movies")

        elif intent == "high_rated_movies":
            rating = max(min_imdb, 7.5)
            movies = self.recommend_combined(
                genre=genre, language=language, year=year,
                min_imdb=rating, actor=actor, n=n
            )
            explanation = _explain(f"movies with IMDb ≥ {rating}")

        elif intent == "latest_movies":
            # If a specific year was mentioned, filter strictly to that year
            if year:
                movies = self.recommend_combined(
                    year=year, genre=genre, language=language,
                    min_imdb=min_imdb, actor=actor, n=n
                )
                explanation = _explain(f"latest movies from {year}")
            else:
                # No year specified — sort by year desc, apply other filters
                df = self.df.copy()
                if language:
                    df = df[df["language"].str.lower() == normalize_language(language).lower()]
                if genre:
                    df = df[df["genre"].str.lower() == normalize_genre(genre).lower()]
                if min_imdb > 0:
                    df = df[df["imdb"] >= min_imdb]
                movies = self._top_n(df.sort_values("year", ascending=False), n)
                explanation = _explain("latest Bollywood releases")

        elif intent == "old_classics":
            df = self.df[self.df["year"] < 2000].copy()
            if language:
                df = df[df["language"].str.lower() == normalize_language(language).lower()]
            if min_imdb > 0:
                df = df[df["imdb"] >= min_imdb]
            movies = self._top_n(df, n)
            explanation = _explain("classic Bollywood films (before 2000)")

        elif intent in ("hindi_movies", "south_movies"):
            lang = "Hindi" if intent == "hindi_movies" else (language or "Hindi")
            movies = self.recommend_combined(
                language=lang, genre=genre, year=year,
                min_imdb=min_imdb, actor=actor, n=n
            )
            explanation = _explain(f"{lang} language movies")

        elif intent == "actor_movies":
            if actor:
                movies = self.recommend_combined(
                    actor=actor, genre=genre, language=language,
                    year=year, min_imdb=min_imdb, n=n
                )
                explanation = _explain(f"movies featuring {actor}")
            else:
                movies = self.recommend_popular(n)
                explanation = "Recommended popular Bollywood movies."

        elif intent == "year_movies":
            if year:
                movies = self.recommend_combined(
                    year=year, genre=genre, language=language,
                    min_imdb=min_imdb, actor=actor, n=n
                )
                explanation = _explain(f"movies from {year}")
            else:
                movies = self.recommend_latest(n)
                explanation = "Recommended latest Bollywood releases."

        elif intent == "mood_movies":
            effective_mood = mood or "happy"
            movies = _with_extras(primary_mood=effective_mood)
            explanation = _explain(f"because you like {effective_mood} movies")

        elif intent == "popular_movies":
            movies = self.recommend_combined(
                genre=genre, language=language, year=year,
                min_imdb=min_imdb, actor=actor, n=n
            ) if any([genre, language, year, actor, min_imdb > 0]) else self.recommend_popular(n)
            explanation = _explain("most popular Bollywood movies")

        elif intent == "biography_movies":
            movies = _with_extras(primary_genre="Biography")
            explanation = _explain("based on Biography genre")

        elif intent == "award_movies":
            rating = max(min_imdb, 7.5)
            movies = self.recommend_combined(
                genre=genre, language=language, year=year,
                min_imdb=rating, actor=actor, n=n
            )
            explanation = _explain(f"critically acclaimed movies (IMDb ≥ {rating})")

        elif intent == "movie_like":
            title = entities.get("title", "")
            if title:
                movies = self.recommend_similar(title, n)
                explanation = f"Recommended movies similar to '{title}'."
            else:
                movies = self.recommend_popular(n)
                explanation = "Recommended popular movies."

        else:
            # Generic / multi-filter / recommendation / fallback
            if any([genre, mood, language, year, actor, min_imdb > 0]):
                movies = self.recommend_combined(
                    genre=genre, mood=mood, language=language,
                    year=year, min_imdb=min_imdb, actor=actor, n=n
                )
                parts = []
                if genre:    parts.append(f"genre: {genre}")
                if mood:     parts.append(f"mood: {mood}")
                if language: parts.append(f"language: {language}")
                if year:     parts.append(f"year: {year}")
                if actor:    parts.append(f"actor: {actor}")
                if min_imdb: parts.append(f"IMDb ≥ {min_imdb}")
                explanation = "Recommended based on " + ", ".join(parts) + "."
            else:
                movies = self.recommend_combined(
                    genre=genre, mood=mood, language=language,
                    year=year, year_from=year_from, year_to=year_to,
                    min_imdb=min_imdb, actor=actor,
                    family_safe=family_safe, n=n
                ) if any([genre, mood, language, year, year_from, year_to,
                          actor, min_imdb > 0, family_safe]) else self.recommend_popular(n)
                explanation = "Recommended popular Bollywood movies."

        if not movies:
            movies = self.recommend_popular(n)
            explanation = "No exact matches found — showing popular movies instead."

        return movies, explanation
