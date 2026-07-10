"""
preprocessing.py — NLP preprocessing pipeline.
Tokenization → Lemmatization → Bag-of-Words vectorization.
"""

import os
import sys
import string
import logging

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IGNORE_CHARS, WORDS_FILE, CLASSES_FILE

logger = logging.getLogger("preprocessing")

# Download required NLTK data once
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_data() -> None:
    """Ensure required NLTK corpora are available, downloading any that are missing.

    Raises a clear ``RuntimeError`` if a package cannot be downloaded or located,
    rather than letting the failure surface later as a cryptic tokenizer/lemmatizer
    error deep inside the NLP pipeline.
    """
    for pkg, resource in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
            continue
        except LookupError:
            pass

        logger.info("NLTK resource '%s' missing — downloading...", pkg)
        if not nltk.download(pkg, quiet=True):
            raise RuntimeError(
                f"Failed to download required NLTK package '{pkg}'. "
                "Check your internet connection, or pre-install it with "
                f"`python -c \"import nltk; nltk.download('{pkg}')\"`."
            )

        try:
            nltk.data.find(resource)
        except LookupError as exc:
            raise RuntimeError(
                f"NLTK package '{pkg}' was downloaded but resource '{resource}' "
                "could not be located afterwards."
            ) from exc


ensure_nltk_data()

lemmatizer = WordNetLemmatizer()


# ── Core NLP functions ─────────────────────────────────────────────────────────

def tokenize(sentence: str) -> list[str]:
    """Tokenize a sentence into words using NLTK word_tokenize."""
    return nltk.word_tokenize(sentence)


def lemmatize(word: str) -> str:
    """Lemmatize a single word to its base/root form."""
    return lemmatizer.lemmatize(word.lower())


def clean_sentence(sentence: str) -> list[str]:
    """
    Full cleaning pipeline:
    1. Tokenize
    2. Lowercase
    3. Remove punctuation / ignore chars
    4. Lemmatize
    """
    tokens = tokenize(sentence)
    cleaned = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower not in IGNORE_CHARS and token_lower not in string.punctuation:
            cleaned.append(lemmatize(token_lower))
    return cleaned


def bag_of_words(sentence: str, words: list[str]) -> np.ndarray:
    """
    Convert a sentence into a Bag-of-Words numpy array.
    Each position = 1 if the corresponding word from vocabulary is present.

    Args:
        sentence: raw input sentence
        words:    vocabulary list (sorted, lemmatized)

    Returns:
        np.ndarray of shape (len(words),) with 0/1 values
    """
    sentence_words = clean_sentence(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for sw in sentence_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = 1.0
    return bag


# ── Vocabulary and classes builder ────────────────────────────────────────────

def build_vocabulary(intents_data: dict) -> tuple[list[str], list[str], list[tuple]]:
    """
    Build vocabulary (words), intent classes, and (bag, class_idx) training docs.

    Returns:
        words:    sorted unique lemmatized vocabulary
        classes:  sorted unique intent tag list
        documents: list of (pattern_word_list, tag) tuples
    """
    words = []
    classes = []
    documents = []

    for intent in intents_data["intents"]:
        tag = intent["tag"]
        if tag not in classes:
            classes.append(tag)

        for pattern in intent["patterns"]:
            word_list = tokenize(pattern)
            lemmatized = [lemmatize(w) for w in word_list
                          if w not in IGNORE_CHARS and w not in string.punctuation]
            words.extend(lemmatized)
            documents.append((lemmatized, tag))

    words = sorted(set(words))
    classes = sorted(set(classes))

    logger.info("Vocabulary size: %d words | %d intent classes | %d training docs",
                len(words), len(classes), len(documents))

    return words, classes, documents


def encode_training_data(
    documents: list[tuple],
    words: list[str],
    classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode (word_list, tag) documents into (X, y) arrays.

    X: BoW matrix  [n_samples, vocab_size]
    y: one-hot     [n_samples, n_classes]
    """
    X, y = [], []

    for word_list, tag in documents:
        # BoW vector
        bow = np.zeros(len(words), dtype=np.float32)
        for w in word_list:
            if w in words:
                bow[words.index(w)] = 1.0
        X.append(bow)

        # One-hot class label
        label = np.zeros(len(classes), dtype=np.float32)
        label[classes.index(tag)] = 1.0
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def save_artifacts(words: list[str], classes: list[str]) -> None:
    """Persist words and classes as .npy artifacts."""
    os.makedirs(os.path.dirname(WORDS_FILE), exist_ok=True)
    np.save(WORDS_FILE, np.array(words, dtype=object))
    np.save(CLASSES_FILE, np.array(classes, dtype=object))
    logger.info("Saved words.npy (%d) and classes.npy (%d)", len(words), len(classes))


def load_artifacts() -> tuple[list[str], list[str]]:
    """Load persisted vocabulary and classes."""
    words   = list(np.load(WORDS_FILE, allow_pickle=True))
    classes = list(np.load(CLASSES_FILE, allow_pickle=True))
    return words, classes
