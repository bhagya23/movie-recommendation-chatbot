# 🎬 Movie Recommendation Chatbot

A domain-oriented, NLP-based conversational movie recommender. A feed-forward
neural network classifies each user message into one of **25 intents** (mood,
genre, language, age group, or control intents), and a rule-based engine filters
a catalogue of **5,882 movies** to return the best matches.

> This project intentionally uses a **closed-domain intent classifier + rule-based
> recommender**. There is **no sentiment analysis** and no collaborative filtering —
> recommendations come from mapping the predicted intent to catalogue filters.

---

## ✨ Features

- **25-intent classifier** — greetings/goodbye/thanks/help, 5 moods, 8 genres,
  3 languages, 3 age groups, plus `top_rated`, `recent_movies`, `recommend_general`.
- **Custom NLP pipeline** — regex tokenizer + lightweight suffix-stripping
  lemmatizer + binary Bag-of-Words (no heavy NLTK/WordNet dependency).
- **Deep intent model** — `Dense(256) → BN → Dropout(0.4) → Dense(128) → BN →
  Dropout(0.3) → Dense(64) → Dropout(0.2) → Softmax(25)`.
- **Stratified 70/15/15 split** with `EarlyStopping` + `ReduceLROnPlateau`.
- **Confidence threshold (0.40)** with a keyword-matching fallback to
  `recommend_general`.
- **5,882-movie catalogue** merged from two public Kaggle datasets and
  auto-downloaded via `kagglehub`.
- **Flask web UI** with poster, description, genre, language, mood, and age-group
  chips.

---

## 🗂️ Project Structure

```
movie-chatbot/
├── app.py                     # Flask backend (serves UI + /predict API)
├── requirements.txt
├── data/
│   ├── intents.json           # 25-intent training corpus
│   ├── raw/                    # (optional) local dataset fallback CSVs
│   └── processed/
│       └── movies.json         # generated 5,882-movie catalogue
├── models/
│   ├── movie_chatbot_model.keras
│   ├── movie_words.npy
│   └── movie_classes.npy
├── outputs/                    # training_curve.png, confusion_matrix.png, reports/
├── src/
│   ├── config.py               # constants, intent→filter maps, hyperparameters
│   ├── preprocessing.py        # tokenizer + lemmatizer + Bag-of-Words
│   ├── dataset.py              # kagglehub download + catalogue builder
│   ├── train.py                # model training + evaluation
│   ├── predict.py              # intent inference + keyword fallback
│   ├── recommender.py          # INTENT_TAG_MAP filtering engine
│   ├── chatbot.py              # orchestration (intent → response + movies)
│   ├── evaluation.py           # metrics, confusion matrix, reports
│   └── utils.py                # logging + JSON helpers
├── templates/index.html        # chat UI
└── test_integration.py         # end-to-end checks
```

---

## 📚 Datasets

The catalogue is built from two public Kaggle datasets, downloaded automatically
via [`kagglehub`](https://pypi.org/project/kagglehub/):

| Dataset | Kaggle slug | Key file | Rows used |
|---|---|---|---|
| Bollywood Movie Dataset | `mitesh58/bollywood-movie-dataset` | `BollywoodMovieDetail.csv` | 1,284 |
| TMDB 5000 (with ratings) | `aayushsoni4/tmdb-5000-movie-dataset-with-ratings` | `tmdb_movie_dataset.csv` | 4,602 |

After row standardization and de-duplication (by title + year + language), the
final catalogue contains **5,882 movies**. Each record follows this schema:

```json
{
  "title": "Oldboy",
  "year": 2003,
  "genre": "thriller",
  "language": "korean",
  "mood": ["thrilled", "dark"],
  "age_group": ["adult"],
  "imdb": 8.0,
  "description": "...",
  "poster": "https://...",
  "source": "tmdb"
}
```

> `kagglehub` downloads public datasets anonymously. If it cannot reach Kaggle,
> the builder falls back to CSVs placed in `data/raw/`.

---

## 🚀 Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Build the catalogue (downloads datasets via kagglehub)

```powershell
python -m src.dataset
```

This produces `data/processed/movies.json` with 5,882 movies.

### 3. Train the intent model

```powershell
python -m src.train
```

Saves `models/movie_chatbot_model.keras` (+ `movie_words.npy`,
`movie_classes.npy`) and writes training curves, a confusion matrix, and a
classification report to `outputs/`.

### 4. Run the web app

```powershell
python app.py
```

Open <http://localhost:5000> and start chatting.

### 5. (Optional) Run the integration tests

```powershell
python test_integration.py
```

---

## 🧠 How It Works

1. **Preprocessing** (`src/preprocessing.py`) — the message is lowercased,
   tokenized with a regex, stripped of a small stopword set, and lemmatized by
   removing common suffixes. It is then encoded as a binary Bag-of-Words over the
   482-word vocabulary.
2. **Intent classification** (`src/predict.py`) — the BoW vector is fed to the
   trained network, which outputs a softmax over 25 intents. If the top
   confidence is below **0.40**, a keyword fallback maps obvious terms
   (e.g. "korean", "horror") to an intent, defaulting to `recommend_general`.
3. **Recommendation** (`src/recommender.py`) — the predicted intent is looked up
   in `INTENT_TAG_MAP` to obtain filter rules (genre / mood / language / age /
   min IMDb / min year). Matching movies are sorted by IMDb rating and the top-3
   are returned.
4. **Response** (`src/chatbot.py`) — control intents
   (`greeting`, `goodbye`, `thanks`, `help`) answer directly; all others return a
   response line plus the recommended movies.

---

## 📊 Model

| Setting | Value |
|---|---|
| Architecture | Dense 256 → 128 → 64 → Softmax(25), BatchNorm + Dropout |
| Optimizer | Adam (lr = 1e-3) |
| Loss | Categorical cross-entropy |
| Split | Stratified 70 / 15 / 15 (train / val / test) |
| Callbacks | EarlyStopping (patience 25) + ReduceLROnPlateau (factor 0.5, patience 8) |
| Vocabulary | 482 words |
| Intents | 25 classes |

Typical results: **~85% validation accuracy** and **~84% test accuracy**
(macro-F1 ≈ 0.83). Metrics and plots are saved to `outputs/`.

---

## 🎯 Intents

`greeting`, `goodbye`, `thanks`, `help`,
`mood_happy`, `mood_sad`, `mood_romantic`, `mood_excited`, `mood_scared`,
`genre_action`, `genre_comedy`, `genre_drama`, `genre_thriller`, `genre_scifi`,
`genre_horror`, `genre_animation`,
`lang_hindi`, `lang_english`, `lang_korean`,
`age_teenager`, `age_kids`, `age_adult`,
`top_rated`, `recent_movies`, `recommend_general`.

---

## 🔌 API

`POST /predict`

```json
{ "query": "Recommend a Korean thriller", "n_recommendations": 3 }
```

Response:

```json
{
  "intent": "lang_korean",
  "confidence": 0.9996,
  "response": "🇰🇷 Hallyu wave! Here are some incredible Korean films!",
  "explanation": "Recommended based on korean language.",
  "movies": [ { "title": "Oldboy", "year": 2003, "genre": "thriller", "imdb": 8.0, "...": "..." } ]
}
```

Other endpoints: `GET /health`, `GET /intents`.
