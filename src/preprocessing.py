"""
preprocessing.py — NLP preprocessing pipeline (matches the academic notebook).

Pipeline:
- Tokenization via regex (lowercase alphabetic tokens)
- Lightweight suffix-stripping lemmatization
- Stopword removal (small IGNORE set)
- Binary Bag-of-Words encoding
- Vocabulary generation
"""

import os
import sys
import re
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import WORDS_FILE, CLASSES_FILE
from src.utils import get_logger

logger = get_logger("preprocess")

# Lightweight suffix-stripping lemmatizer used consistently for training/inference
SUFFIXES = ["ing", "tion", "ness", "ment", "able", "ful", "less", "es", "ed", "er", "s"]
IGNORE = {"a", "an", "the", "is", "i", "me", "my", "we", "our", "you", "it"}


# ── Lemmatization ──────────────────────────────────────────────────────────────

def lemmatise(word: str) -> str:
    """Strip a common English suffix if the remaining stem stays meaningful."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[: -len(suffix)]
    return word


# Backwards-compatible alias
def lemmatize(word: str) -> str:
    return lemmatise(word.lower())


# ── Tokenization ────────────────────────────────────────────────────────────────

def tokenise(text: str) -> List[str]:
    """Lowercase, extract alphabetic tokens, drop stopwords, and lemmatise."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [lemmatise(tok) for tok in tokens if tok not in IGNORE]


# Backwards-compatible aliases
def tokenize(sentence: str) -> List[str]:
    return tokenise(sentence)


def preprocess_sentence(sentence: str) -> List[str]:
    """Full preprocessing pipeline for a single sentence."""
    return tokenise(sentence)


# ── Bag of Words ───────────────────────────────────────────────────────────────

def bag_of_words(sentence: str, words: List[str]) -> np.ndarray:
    """
    Convert a sentence into a binary Bag-of-Words numpy array.
    Each position = 1 if the corresponding vocabulary word is present.
    """
    word_index = {w: i for i, w in enumerate(words)}
    bag = np.zeros(len(words), dtype=np.float32)
    for sw in preprocess_sentence(sentence):
        idx = word_index.get(sw)
        if idx is not None:
            bag[idx] = 1.0
    return bag


# ── Vocabulary generation ──────────────────────────────────────────────────────

def build_vocabulary(intents_data: dict) -> Tuple[List[str], List[str], List[Tuple]]:
    """
    Build vocabulary (words), intent classes, and training documents.
    
    Process:
    1. Extract all patterns from intents
    2. Preprocess each pattern
    3. Build unique vocabulary
    4. Extract unique intent classes
    5. Create (word_list, tag) document tuples
    
    Returns:
        words: sorted unique lemmatized vocabulary
        classes: sorted unique intent tag list
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
            word_list = tokenise(pattern)
            words.extend(word_list)
            documents.append((word_list, tag))

    # Remove duplicates and sort
    words = sorted(set(words))
    classes = sorted(set(classes))

    logger.info(
        "Vocabulary built: %d words | %d intent classes | %d training documents",
        len(words), len(classes), len(documents)
    )

    return words, classes, documents


# ── Training data encoding ─────────────────────────────────────────────────────

def encode_training_data (
    documents: List[Tuple],
    words: List[str],
    classes: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode (word_list, tag) documents into (X, y) arrays for training.
    
    Args:
        documents: list of (word_list, tag) tuples
        words: vocabulary list
        classes: intent class list
    
    Returns:
        X: BoW matrix [n_samples, vocab_size]
        y: One-hot encoded labels [n_samples, n_classes]
    """
    X, y = [], []

    for word_list, tag in documents:
        # Create BoW vector
        bow = np.zeros(len(words), dtype=np.float32)
        for w in word_list:
            if w in words:
                bow[words.index(w)] = 1.0
        X.append(bow)

        # Create one-hot encoded label
        label = np.zeros(len(classes), dtype=np.float32)
        label[classes.index(tag)] = 1.0
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Artifact persistence ───────────────────────────────────────────────────────

def save_artifacts(words: List[str], classes: List[str]) -> None:
    """Persist vocabulary and classes as .npy artifacts."""
    os.makedirs(os.path.dirname(WORDS_FILE), exist_ok=True)
    np.save(WORDS_FILE, np.array(words, dtype=object))
    np.save(CLASSES_FILE, np.array(classes, dtype=object))
    logger.info("Saved artifacts: words.npy (%d words), classes.npy (%d classes)", 
                len(words), len(classes))


def load_artifacts() -> Tuple[List[str], List[str]]:
    """Load persisted vocabulary and classes."""
    words = list(np.load(WORDS_FILE, allow_pickle=True))
    classes = list(np.load(CLASSES_FILE, allow_pickle=True))
    logger.info("Loaded artifacts: %d words, %d classes", len(words), len(classes))
    return words, classes
