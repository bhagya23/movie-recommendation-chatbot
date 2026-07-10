"""
train.py — Model training pipeline for the Movie Recommendation Chatbot.

Architecture:
Input → Text preprocessing → Tokenization → Lemmatization → Bag of Words → 
Dense(256) → BatchNormalization → Dropout(0.40) → 
Dense(128) → BatchNormalization → Dropout(0.30) → 
Dense(64) → Dropout(0.20) → Softmax(25)

Usage:
    python src/train.py
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    INTENTS_FILE, MODEL_FILE, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, LOGS_DIR,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    DROPOUT_1, DROPOUT_2, DROPOUT_3,
    DENSE_1, DENSE_2, DENSE_3,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, REDUCE_LR_MIN_LR,
    TEST_SIZE, SEED,
    TRAINING_CURVE_IMG, CONFUSION_MATRIX_IMG
)
from src.preprocessing import (
    build_vocabulary, encode_training_data, save_artifacts
)
from src.utils import load_intents, get_logger

logger = get_logger("train")

# Reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── Model definition ───────────────────────────────────────────────────────────

def build_model(input_dim: int, output_dim: int) -> Sequential:
    """
    Build the DNN intent classifier with exact architecture:
    
    Input → Dense(256) → BatchNormalization → Dropout(0.40) → 
    Dense(128) → BatchNormalization → Dropout(0.30) → 
    Dense(64) → Dropout(0.20) → Softmax(output_dim)
    """
    model = Sequential([
        Input(shape=(input_dim,), name="input_layer"),
        
        Dense(DENSE_1, activation="relu", name="dense_256"),
        BatchNormalization(name="batch_norm_1"),
        Dropout(DROPOUT_1, name="dropout_1"),
        
        Dense(DENSE_2, activation="relu", name="dense_128"),
        BatchNormalization(name="batch_norm_2"),
        Dropout(DROPOUT_2, name="dropout_2"),
        
        Dense(DENSE_3, activation="relu", name="dense_64"),
        Dropout(DROPOUT_3, name="dropout_3"),
        
        Dense(output_dim, activation="softmax", name="output_softmax"),
    ], name="intent_classifier")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)
    return model


# ── Callbacks setup ───────────────────────────────────────────────────────────

def setup_callbacks() -> list:
    """Callbacks per the academic report: EarlyStopping + ReduceLROnPlateau (monitor val_loss)."""
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN_LR,
            verbose=1,
        ),
    ]
    logger.info("Callbacks configured: EarlyStopping(patience=%d), ReduceLROnPlateau(factor=%.2f, patience=%d)",
                EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE)
    return callbacks


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_training_curves(history) -> None:
    """Plot training and validation accuracy/loss curves."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Performance", fontsize=14, fontweight="bold")
    h = history.history

    # Accuracy
    axes[0].plot(h["accuracy"], label="Train Accuracy", color="#2196F3", linewidth=2)
    if "val_accuracy" in h:
        axes[0].plot(h["val_accuracy"], label="Val Accuracy", color="#FF9800", 
                    linestyle="--", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(h["loss"], label="Train Loss", color="#F44336", linewidth=2)
    if "val_loss" in h:
        axes[1].plot(h["val_loss"], label="Val Loss", color="#4CAF50", 
                    linestyle="--", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Loss", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(TRAINING_CURVE_IMG, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curve → %s", TRAINING_CURVE_IMG)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> None:
    """Plot confusion matrix heatmap."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig_size = max(10, len(classes) // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5, cbar_kws={"label": "Count"}
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
    """Save classification report to file."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_str)
    logger.info("Saved classification report → %s", path)


# ── Main training routine ──────────────────────────────────────────────────────

def train() -> None:
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # 1. Load intents
    intents_data = load_intents()
    logger.info("Loaded %d intents from %s", len(intents_data["intents"]), INTENTS_FILE)

    # 2. Build vocabulary + encode
    words, classes, documents = build_vocabulary(intents_data)
    X, y = encode_training_data(documents, words, classes)
    y_int = np.argmax(y, axis=1)
    logger.info("Encoded data — X: %s | y: %s", X.shape, y.shape)

    # 3. Stratified 70/15/15 train/val/test split (per the academic report)
    X_train, X_temp, y_train, y_temp, _, y_temp_int = train_test_split(
        X, y, y_int, test_size=TEST_SIZE, random_state=SEED, stratify=y_int
    )
    X_val, X_test, y_val, y_test, y_val_int, y_test_int = train_test_split(
        X_temp, y_temp, y_temp_int, test_size=0.50, random_state=SEED, stratify=y_temp_int
    )
    logger.info("Split — train: %d | val: %d | test: %d", len(X_train), len(X_val), len(X_test))

    # 4. Build model with exact architecture
    model = build_model(input_dim=len(words), output_dim=len(classes))

    # 5. Setup callbacks
    callbacks = setup_callbacks()

    # 6. Train with validation data
    logger.info("Training model for up to %d epochs (batch_size=%d)...", EPOCHS, BATCH_SIZE)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Validation-set evaluation
    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_acc = accuracy_score(y_val_int, val_pred)
    val_f1_macro = f1_score(y_val_int, val_pred, average="macro", zero_division=0)
    val_f1_weighted = f1_score(y_val_int, val_pred, average="weighted", zero_division=0)
    logger.info("Validation — Acc: %.4f | F1(macro): %.4f | F1(weighted): %.4f",
                val_acc, val_f1_macro, val_f1_weighted)

    # 8. Test-set evaluation (final, unseen data)
    test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
    test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_acc = accuracy_score(y_test_int, test_pred)
    test_f1_macro = f1_score(y_test_int, test_pred, average="macro", zero_division=0)
    test_f1_weighted = f1_score(y_test_int, test_pred, average="weighted", zero_division=0)
    logger.info("Test — Loss: %.4f | Acc: %.4f | F1(macro): %.4f | F1(weighted): %.4f",
                test_loss, test_acc, test_f1_macro, test_f1_weighted)

    # Classification report on the test set
    report = classification_report(y_test_int, test_pred, target_names=classes, zero_division=0)
    logger.info("\nTest Classification Report:\n%s", report)

    # 9. Generate plots (train/val curves; confusion matrix on test set)
    plot_training_curves(history)
    plot_confusion_matrix(y_test_int, test_pred, classes)
    save_classification_report(report)

    # 10. Save vocabulary artifacts + model
    save_artifacts(words, classes)
    model.save(MODEL_FILE)
    logger.info("Model saved → %s", MODEL_FILE)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Validation Accuracy: %.2f%% | Test Accuracy: %.2f%%",
                val_acc * 100, test_acc * 100)
    logger.info("Test F1 (Macro): %.2f%% | Test F1 (Weighted): %.2f%%",
                test_f1_macro * 100, test_f1_weighted * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
