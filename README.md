# 🎬 Bollywood Movie Recommendation Chatbot

> A **production-quality, closed-domain conversational AI chatbot** for Bollywood movie recommendations powered by Deep Learning + NLP.

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  NLP PIPELINE                                           │
│  Tokenization → Lemmatization → Bag-of-Words           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  DEEP NEURAL NETWORK (TensorFlow/Keras)                 │
│  Input → Dense(128,relu) → Dropout(0.5)                │
│        → Dense(64,relu)  → Dropout(0.5)                │
│        → Dense(n_classes, softmax)                      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
              Intent Classification
                  (25 intents)
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  ENTITY EXTRACTION (Keyword Matching)                   │
│  genre | mood | year | language | actor                │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  RECOMMENDATION ENGINE                                  │
│  Multi-filter: genre + mood + year + IMDb + language   │
│  TF-IDF Cosine Similarity for "movies like X"          │
│  Sorted: IMDb → Popularity → Year                      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STREAMLIT UI                                           │
│  Dark theme | Movie cards | Chat history | Sidebar     │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
movie-chatbot/
│
├── data/
│   ├── bollywood_movies.csv     # Bollywood movie dataset (95+ movies)
│   ├── intents.json             # 25 intent classes with 15-20 patterns each
│   └── movies.json              # Auto-generated normalized movie database
│
├── models/
│   ├── chatbot_model.keras      # Trained DNN model
│   ├── words.npy                # Vocabulary (lemmatized)
│   └── classes.npy              # Intent class labels
│
├── notebooks/
│   └── experimentation.ipynb   # EDA + training experiments
│
├── src/
│   ├── config.py                # All constants & paths
│   ├── utils.py                 # Shared helpers + entity extraction
│   ├── preprocessing.py         # Tokenize → Lemmatize → BoW
│   ├── train.py                 # Training pipeline
│   ├── chatbot.py               # Inference engine
│   ├── recommendation_engine.py # Multi-filter + TF-IDF similarity
│   └── app.py                   # Streamlit UI
│
├── outputs/
│   ├── training_curve.png       # Loss + accuracy plots
│   ├── confusion_matrix.png     # Intent classification confusion matrix
│   └── reports/
│       └── classification_report.txt
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Setup & Installation

### 1. Clone / open the project
```bash
cd movie-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (auto-handled by preprocessing.py)
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🤖 Training the Model

```bash
python src/train.py
```

**What this does:**
1. Loads `data/intents.json`
2. Tokenizes + lemmatizes all patterns
3. Builds Bag-of-Words vocabulary
4. Trains DNN with EarlyStopping + ModelCheckpoint
5. Saves `models/chatbot_model.keras`, `words.npy`, `classes.npy`
6. Generates `outputs/training_curve.png` + `outputs/confusion_matrix.png`

---

## 💬 Running the Chatbot UI

```bash
streamlit run src/app.py
```

Opens at: **http://localhost:8501**

---

## 🧠 Model Architecture

| Layer | Config |
|-------|--------|
| Input | `vocab_size` features (BoW) |
| Dense 1 | 128 units, ReLU activation |
| Dropout 1 | 50% |
| Dense 2 | 64 units, ReLU activation |
| Dropout 2 | 50% |
| Output | `n_classes` units, Softmax |

**Optimizer:** Adam (lr=0.001)  
**Loss:** Categorical Crossentropy  
**Callbacks:** EarlyStopping (patience=20) + ModelCheckpoint

---

## 🎯 Supported Intents (25)

| Intent | Description |
|--------|-------------|
| `greeting` | Hello, Hi, Namaste |
| `goodbye` | Bye, Farewell |
| `thanks` | Thank you, Appreciate it |
| `help` | What can you do |
| `comedy_movies` | Funny movies |
| `action_movies` | Fight/action films |
| `romantic_movies` | Love story films |
| `thriller_movies` | Suspense/mystery |
| `emotional_movies` | Drama/tearjerker |
| `horror_movies` | Scary/ghost films |
| `family_movies` | Kids/family-friendly |
| `high_rated_movies` | IMDb 8+ films |
| `latest_movies` | New releases |
| `old_classics` | Pre-2000 films |
| `hindi_movies` | Hindi/Bollywood |
| `south_movies` | Tamil/Telugu films |
| `actor_movies` | By actor name |
| `year_movies` | By release year |
| `mood_movies` | By current mood |
| `recommendation` | General suggestion |
| `movie_like` | Similar to X |
| `popular_movies` | Trending/box office |
| `biography_movies` | Biopics |
| `award_movies` | Award-winning |
| `multi_filter` | Combined filters |
| `fallback` | Unrecognized input |

---

## 🔍 Entity Extraction

Automatically extracted from natural language:

| Entity | Example Input | Extracted |
|--------|--------------|-----------|
| Mood | "funny Hindi movies" | `funny` |
| Genre | "action films" | `Action` |
| Year | "movies from 2020" | `2020` |
| Language | "Tamil movies" | `Tamil` |
| Actor | "Shah Rukh Khan films" | `Shah Rukh Khan` |

---

## 📊 Recommendation Engine

### Filters
- `recommend_by_genre(genre)`
- `recommend_by_mood(mood)` — maps mood → genre(s)
- `recommend_by_rating(min_imdb)`
- `recommend_by_language(language)`
- `recommend_by_year(year, year_from, year_to)`
- `recommend_by_actor(actor)`
- `recommend_family_friendly()`
- `recommend_combined(...)` — multi-filter with graceful fallback

### Content-Based Similarity
- TF-IDF vectorization on movie metadata
- Cosine similarity for "movies like X"

### Ranking Priority
1. IMDb rating (descending)
2. Popularity / vote count (descending)
3. Release year (descending)

---

## 🎨 UI Features

- **Dark theme** with gradient design
- **Typing animation** for bot responses
- **Movie cards** with IMDb badge, genre badge, year badge
- **Sidebar filters**: Genre, Mood, Language, IMDb, Year
- **Quick search buttons** for common queries
- **Conversation memory** (multi-turn context)
- **Intent + confidence display**
- **Recommendation explanation** ("Recommended because you like...")

---

## 📈 Evaluation Outputs

After training:
- `outputs/training_curve.png` — train/val accuracy + loss
- `outputs/confusion_matrix.png` — 25×25 intent confusion matrix
- `outputs/reports/classification_report.txt` — precision, recall, F1

---

## 🔮 Future Improvements

- TMDB API integration for poster images
- Speech-to-text input
- Transformer-based intent classifier (BERT)
- User preference learning
- Collaborative filtering
- Streaming API for real-time recommendations
- Multi-language UI support

---

## 📋 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras |
| NLP | NLTK (tokenize, lemmatize) |
| Vectorization | Bag-of-Words + TF-IDF |
| Similarity | Scikit-learn cosine similarity |
| Data | Pandas + NumPy |
| UI | Streamlit |
| Visualization | Matplotlib + Seaborn |

---

## 📄 License

MIT License — Academic project use permitted.

---

*Built with ❤️ for Bollywood cinema lovers*
