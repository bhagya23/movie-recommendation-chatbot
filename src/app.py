"""
app.py — Streamlit conversational UI for the Bollywood Movie Recommendation Chatbot.

Run with:
    streamlit run src/app.py
"""

import os
import sys
import time
import random
from html import escape as _esc

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    APP_TITLE, APP_SUBTITLE, CHATBOT_NAME, MAX_HISTORY,
    MODELS_DIR, MODEL_FILE
)

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="CineBot — Bollywood Movie Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0d0d0d; color: #e0e0e0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid #e94560;
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Title */
.main-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e94560, #f5a623);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    color: #888;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

/* Chat bubbles */
.user-bubble {
    background: linear-gradient(135deg, #e94560, #c0392b);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px 20%;
    max-width: 78%;
    float: right;
    clear: both;
    font-size: 0.97rem;
    box-shadow: 0 2px 8px rgba(233,69,96,0.4);
}
.bot-bubble {
    background: linear-gradient(135deg, #1e3a5f, #16213e);
    color: #e0e0e0;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 20% 6px 0;
    max-width: 78%;
    float: left;
    clear: both;
    font-size: 0.97rem;
    box-shadow: 0 2px 8px rgba(30,58,95,0.6);
    border-left: 3px solid #e94560;
}

/* Movie cards */
.movie-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #e94560;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    transition: transform 0.2s, box-shadow 0.2s;
}
.movie-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(233,69,96,0.35);
}
.movie-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f5a623;
    margin-bottom: 6px;
}
.movie-meta {
    font-size: 0.85rem;
    color: #aaa;
    margin: 3px 0;
}
.imdb-badge {
    background: #f5c518;
    color: #000;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
}
.genre-badge {
    background: #e94560;
    color: white;
    font-size: 0.78rem;
    padding: 2px 8px;
    border-radius: 10px;
    display: inline-block;
    margin: 2px;
}
.year-badge {
    background: #1e3a5f;
    color: #e0e0e0;
    font-size: 0.78rem;
    padding: 2px 8px;
    border-radius: 10px;
    display: inline-block;
    margin: 2px;
    border: 1px solid #3a7bd5;
}
.explanation-box {
    background: rgba(245,166,35,0.12);
    border-left: 3px solid #f5a623;
    padding: 8px 14px;
    border-radius: 6px;
    color: #f5a623;
    font-size: 0.87rem;
    margin: 8px 0;
}
.confidence-bar {
    font-size: 0.78rem;
    color: #666;
    margin-top: 4px;
}
.chat-container { padding: 10px 0; }
.clearfix::after { content: ""; display: table; clear: both; }

/* Input box */
.stTextInput > div > div > input {
    background: #1a1a2e !important;
    color: #e0e0e0 !important;
    border: 1px solid #e94560 !important;
    border-radius: 25px !important;
    padding: 12px 20px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c0392b);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 28px;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
def init_session():
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "movie_results" not in st.session_state:
        st.session_state.movie_results = []
    if "explanation" not in st.session_state:
        st.session_state.explanation = ""
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = ""
    if "last_confidence" not in st.session_state:
        st.session_state.last_confidence = 0.0
    if "last_entities" not in st.session_state:
        st.session_state.last_entities = {}
    if "last_sidebar_filters" not in st.session_state:
        st.session_state.last_sidebar_filters = {}


@st.cache_resource(show_spinner="Loading CineBot AI model...")
def load_chatbot():
    from src.chatbot import MovieChatbot
    return MovieChatbot()


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## 🎛️ Filter Movies")
        st.markdown("---")

        genre = st.selectbox(
            "🎭 Genre",
            ["Any", "Action", "Comedy", "Drama", "Romance", "Thriller",
             "Horror", "Biography", "Family", "Adventure", "Sci-Fi",
             "Mystery", "Crime", "Musical", "Historical"],
        )
        mood = st.selectbox(
            "😊 Mood",
            ["Any", "funny", "romantic", "emotional", "suspense",
             "energetic", "scary", "inspiring", "adventurous", "happy", "sad"],
        )
        language = st.selectbox(
            "🌐 Language",
            ["Any", "Hindi", "Tamil", "Telugu", "Bengali",
             "Kannada", "Malayalam", "Punjabi", "Marathi"],
        )
        min_imdb = st.slider("⭐ Min IMDb Rating", 0.0, 10.0, 0.0, 0.5)
        year_mode = st.radio(
            "📅 Year Filter",
            ["None", "Specific Year", "Year Range"],
            horizontal=True
        )

        year      = None
        year_from = None
        year_to   = None

        if year_mode == "Specific Year":
            year = st.number_input("Year", min_value=1950, max_value=2025, value=2023)
        elif year_mode == "Year Range":
            c1, c2 = st.columns(2)
            year_from = c1.number_input("From", min_value=1950, max_value=2025, value=2010)
            year_to   = c2.number_input("To",   min_value=1950, max_value=2025, value=2024)

        family_safe = st.checkbox("👨‍👩‍👧‍👦 Family-Friendly Only", value=False)

        st.markdown("---")
        st.markdown("### 🔍 Quick Searches")
        quick_btns = {
            "🎬 Latest Movies":    "Show me the latest Bollywood movies",
            "⭐ Top Rated":        "What are the highest rated Bollywood movies",
            "😂 Comedy":           "Suggest funny comedy movies",
            "💥 Action":           "Recommend action movies",
            "💕 Romance":          "Show romantic Bollywood movies",
            "👻 Horror":           "Scary horror movies",
            "👨‍👩‍👧‍👦 Family":       "Family friendly movies for kids",
            "🏆 Biopics":          "Best biography movies",
        }
        for label, query in quick_btns.items():
            if st.button(label, use_container_width=True, key=f"quick_{label}"):
                st.session_state["quick_query"] = query

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages      = []
            st.session_state.movie_results = []
            st.session_state.explanation   = ""
            if st.session_state.chatbot:
                st.session_state.chatbot.reset_context()
            st.rerun()

        st.markdown("---")
        st.markdown("**🤖 CineBot** v1.0")
        st.caption("Deep Learning + NLP Movie Chatbot")

    return {
        "genre":       None if genre == "Any" else genre,
        "mood":        None if mood  == "Any" else mood,
        "language":    None if language == "Any" else language,
        "min_imdb":    min_imdb,
        "year":        int(year) if year else None,
        "year_from":   int(year_from) if year_from else None,
        "year_to":     int(year_to)   if year_to   else None,
        "family_safe": family_safe if family_safe else None,
    }


# ── Chat rendering ─────────────────────────────────────────────────────────────
def render_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages[-MAX_HISTORY:]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">💬 {_esc(msg["content"])}</div>'
                '<div class="clearfix"></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-bubble">🎬 <b>{CHATBOT_NAME}:</b> {_esc(msg["content"])}</div>'
                '<div class="clearfix"></div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Movie card rendering ───────────────────────────────────────────────────────
def render_movie_card(movie: dict, rank: int) -> None:
    actors    = _esc(", ".join(movie.get("actors", [])[:3]) or "N/A")
    title     = _esc(str(movie.get("title", "Unknown Title")))
    genre     = _esc(str(movie.get("genre", "")))
    language  = _esc(str(movie.get("language", "Hindi")))
    year      = _esc(str(movie.get("year", "N/A")))
    imdb      = _esc(str(movie.get("imdb", "N/A")))

    # Build optional HTML snippets as plain variables (avoids f-string quote conflicts)
    family_badge = (
        '&nbsp;<span class="genre-badge" style="background:#27ae60;">&#x2705; Family-Safe</span>'
        if movie.get("family_safe") else ""
    )

    moods = movie.get("mood") or []
    mood_html = (
        f'<div class="movie-meta">&#x1F3AD; Mood: {_esc(", ".join(moods))}</div>'
        if moods else ""
    )

    sim_score = movie.get("similarity_score")
    sim_html = (
        f'<div class="movie-meta">&#x1F517; Similarity: {sim_score:.0%}</div>'
        if sim_score else ""
    )

    html = (
        f'<div class="movie-card">'
        f'<div class="movie-title">#{rank} {title}</div>'
        f'<div style="margin:6px 0;">'
        f'<span class="imdb-badge">&#x2B50; IMDb {imdb}</span>&nbsp;'
        f'<span class="genre-badge">{genre}</span>&nbsp;'
        f'<span class="year-badge">&#x1F4C5; {year}</span>'
        f'{family_badge}'
        f'</div>'
        f'<div class="movie-meta">&#x1F310; Language: <b>{language}</b></div>'
        f'<div class="movie-meta">&#x1F464; Cast: {actors}</div>'
        f'{mood_html}'
        f'{sim_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_movies_panel(movies: list[dict], explanation: str) -> None:
    if not movies:
        return

    if explanation:
        st.markdown(
            f'<div class="explanation-box">💡 {_esc(explanation)}</div>',
            unsafe_allow_html=True
        )

    st.markdown(f"### 🎬 Recommendations ({len(movies)} movies)")
    for i, movie in enumerate(movies, 1):
        render_movie_card(movie, i)


# ── Typing animation ───────────────────────────────────────────────────────────
def typing_animation(placeholder, text: str, delay: float = 0.015) -> None:
    """Simulate character-by-character typing."""
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(
            f'<div class="bot-bubble">🎬 <b>{CHATBOT_NAME}:</b> {_esc(displayed)}▋</div>'
            '<div class="clearfix"></div>',
            unsafe_allow_html=True
        )
        time.sleep(delay)
    placeholder.markdown(
        f'<div class="bot-bubble">🎬 <b>{CHATBOT_NAME}:</b> {_esc(displayed)}</div>'
        '<div class="clearfix"></div>',
        unsafe_allow_html=True
    )


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    init_session()
    sidebar_filters = render_sidebar()

    # Title
    st.markdown(f'<h1 class="main-title">{APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{APP_SUBTITLE}</p>',  unsafe_allow_html=True)

    # Model status
    model_not_trained = not os.path.exists(MODEL_FILE)
    if model_not_trained:
        st.warning(
            "⚠️ Model not trained yet. Please run: `python src/train.py`  \n"
            "Then refresh this page."
        )
        st.info("You can still explore the **recommendation engine** below while the model trains.")

    # Load chatbot (cached)
    if not model_not_trained:
        try:
            if st.session_state.chatbot is None:
                st.session_state.chatbot = load_chatbot()
                st.session_state.model_loaded = True
                if not st.session_state.messages:
                    greeting = st.session_state.chatbot.get_greeting()
                    st.session_state.messages.append({"role": "bot", "content": greeting})
        except Exception as e:
            st.error(f"Error loading chatbot: {e}")
            model_not_trained = True

    # Re-dispatch if sidebar filters changed after a prior chat (makes all sidebar
    # filters live-reactive — e.g. moving IMDb slider updates results immediately)
    if (
        st.session_state.last_intent
        and st.session_state.movie_results
        and not st.session_state.get("quick_query")
        and sidebar_filters != st.session_state.last_sidebar_filters
    ):
        st.session_state.last_sidebar_filters = sidebar_filters.copy()
        if st.session_state.chatbot:
            movies, explanation = st.session_state.chatbot.engine.dispatch(
                intent=st.session_state.last_intent,
                entities=st.session_state.last_entities,
                sidebar_filters=sidebar_filters,
            )
            st.session_state.movie_results = movies
            st.session_state.explanation   = explanation
        st.rerun()

    # Layout: chat left, movies right
    col_chat, col_movies = st.columns([3, 2], gap="large")

    with col_chat:
        st.markdown("### 💬 Chat")

        # Render existing messages
        render_chat()

        # Handle quick-search button queries — process directly, no form needed
        quick_query = st.session_state.pop("quick_query", None)
        if quick_query:
            _process_message(quick_query.strip(), sidebar_filters, model_not_trained)
            st.rerun()

        # Manual chat input area
        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([5, 1])
            user_input = c1.text_input(
                "Message",
                placeholder="Ask me about Bollywood movies...",
                label_visibility="collapsed",
                key="user_msg",
            )
            send_btn = c2.form_submit_button("Send ➤")

        if send_btn and user_input and user_input.strip():
            _process_message(user_input.strip(), sidebar_filters, model_not_trained)
            st.rerun()

        # Intent debug info
        if st.session_state.last_intent and st.session_state.model_loaded:
            st.markdown(
                f'<div class="confidence-bar">🔍 Intent: <b>{st.session_state.last_intent}</b> '
                f'| Confidence: <b>{st.session_state.last_confidence:.1%}</b></div>',
                unsafe_allow_html=True
            )

    with col_movies:
        st.markdown("### 🎥 Movie Recommendations")
        if st.session_state.movie_results:
            render_movies_panel(
                st.session_state.movie_results,
                st.session_state.explanation
            )
        else:
            _render_explore_section(sidebar_filters)


def _process_message(user_input: str, sidebar_filters: dict, model_not_trained: bool) -> None:
    """Process user message and update session state."""
    st.session_state.messages.append({"role": "user", "content": user_input})

    if model_not_trained or st.session_state.chatbot is None:
        # Fallback: use recommendation engine directly
        from src.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        from src.utils import extract_entities
        entities = extract_entities(user_input)
        movies, explanation = engine.dispatch("recommendation", entities, sidebar_filters)
        bot_response = f"I found {len(movies)} movies matching your request! (NLP model not trained yet)"
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        st.session_state.movie_results = movies
        st.session_state.explanation   = explanation
    else:
        result = st.session_state.chatbot.chat(
            user_message=user_input,
            sidebar_filters=sidebar_filters,
            n_recommendations=5,
        )
        st.session_state.messages.append({"role": "bot", "content": result["response"]})
        st.session_state.movie_results       = result["movies"]
        st.session_state.explanation         = result["explanation"]
        st.session_state.last_intent         = result["intent"]
        st.session_state.last_confidence     = result["confidence"]
        st.session_state.last_entities       = result.get("entities", {})


def _render_explore_section(sidebar_filters: dict) -> None:
    """Show explore cards when no conversation yet."""
    st.markdown("#### 🌟 Explore Bollywood")
    try:
        from src.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()

        # Apply sidebar filters if any active
        genre      = sidebar_filters.get("genre")
        mood       = sidebar_filters.get("mood")
        language   = sidebar_filters.get("language")
        year       = sidebar_filters.get("year")
        year_from  = sidebar_filters.get("year_from")
        year_to    = sidebar_filters.get("year_to")
        min_imdb   = float(sidebar_filters.get("min_imdb") or 0.0)
        family_safe = True if sidebar_filters.get("family_safe") else None

        any_active = any([genre, mood, language, year, year_from, year_to,
                          min_imdb > 0, family_safe is not None])
        if any_active:
            movies = engine.recommend_combined(
                genre=genre, mood=mood, language=language,
                year=year, year_from=year_from, year_to=year_to,
                min_imdb=min_imdb, family_safe=family_safe, n=5
            )
            parts = []
            if genre:       parts.append(genre)
            if mood:        parts.append(mood)
            if language:    parts.append(language)
            if year:        parts.append(str(year))
            if min_imdb:    parts.append(f"IMDb≥{min_imdb}")
            if family_safe: parts.append("family-friendly")
            explanation = "Filtered by: " + ", ".join(parts) + "." if parts else "Filtered by sidebar selections."
        else:
            movies = engine.recommend_popular(5)
            explanation = "Most popular Bollywood movies."

        render_movies_panel(movies, explanation)
    except Exception as e:
        st.info(f"Start chatting to get movie recommendations! 🎬\n\n_{e}_")


if __name__ == "__main__":
    main()
