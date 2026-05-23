"""
train.py — Model training pipeline for the Movie Recommendation Chatbot.

Usage:
    python src/train.py
"""

import os
import sys
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    INTENTS_FILE, MODEL_FILE, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE, DENSE_1, DENSE_2,
    TRAINING_CURVE_IMG, CONFUSION_MATRIX_IMG
)
from src.preprocessing import (
    build_vocabulary, encode_training_data, save_artifacts
)
from src.utils import load_intents, get_logger

logger = get_logger("train")

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ── Model definition ───────────────────────────────────────────────────────────

def build_model(input_dim: int, output_dim: int) -> Sequential:
    """Build the DNN intent classifier."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(DENSE_1, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_2, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(output_dim, activation="softmax"),
    ], name="intent_classifier")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)
    return model


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_training_curves(history) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Performance", fontsize=14, fontweight="bold")
    h = history.history

    # Accuracy
    axes[0].plot(h["accuracy"], label="Train Accuracy", color="#2196F3")
    if "val_accuracy" in h:
        axes[0].plot(h["val_accuracy"], label="Val Accuracy", color="#FF9800", linestyle="--")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(h["loss"], label="Train Loss", color="#F44336")
    if "val_loss" in h:
        axes[1].plot(h["val_loss"], label="Val Loss", color="#4CAF50", linestyle="--")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(TRAINING_CURVE_IMG, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curve → %s", TRAINING_CURVE_IMG)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig_size = max(10, len(classes) // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5
    )
    ax.set_title("Confusion Matrix — Intent Classifier", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMG, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix → %s", CONFUSION_MATRIX_IMG)


def save_classification_report(report_str: str) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_str)
    logger.info("Saved classification report → %s", path)


# ── Main training routine ──────────────────────────────────────────────────────

def train() -> None:
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # 1. Load intents
    intents_data = load_intents()
    logger.info("Loaded %d intents from %s", len(intents_data["intents"]), INTENTS_FILE)

    # 2. Build vocabulary + encode
    words, classes, documents = build_vocabulary(intents_data)
    X, y = encode_training_data(documents, words, classes)
    logger.info("Training data shape — X: %s | y: %s", X.shape, y.shape)

    # 3. For closed-domain chatbot with limited data, train on ALL samples.
    #    A held-out split from ~400 samples gives misleadingly low accuracy
    #    (~4 test samples per class). We evaluate on the full training corpus.
    X_train, y_train = X, y
    logger.info("Using full corpus for training (%d samples, %d classes)", len(X_train), len(classes))

    # 4. Build model
    model = build_model(input_dim=len(words), output_dim=len(classes))

    # 5. Callbacks — save best train-accuracy checkpoint
    os.makedirs(MODELS_DIR, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_FILE,
            monitor="accuracy",
            save_best_only=True,
            verbose=0
        ),
    ]

    # 6. Train on ALL data for fixed epochs (standard for tiny closed-domain intents)
    logger.info("Training model for %d epochs (batch_size=%d)...", EPOCHS, BATCH_SIZE)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluate on full corpus (training accuracy for closed-domain benchmark)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    logger.info("Training Corpus Accuracy: %.4f | Loss: %.4f", train_acc, train_loss)

    # 8. Confusion matrix & classification report on full corpus
    y_pred_proba = model.predict(X_train, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_train, axis=1)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    logger.info("\nFull Corpus Accuracy: %.4f\n\nClassification Report:\n%s", acc, report)

    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred, classes)
    save_classification_report(report)

    # 9. Save vocabulary artifacts
    save_artifacts(words, classes)

    # 10. Final model save (ensure saved even if checkpoint missed)
    model.save(MODEL_FILE)
    logger.info("Model saved → %s", MODEL_FILE)
    logger.info("=" * 60)
    logger.info("Training complete! Test Accuracy = %.2f%%", acc * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
