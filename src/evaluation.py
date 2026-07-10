"""
evaluation.py — Model evaluation and metrics generation.

Generates:
- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro)
- F1 Score (Weighted)
- Confusion Matrix
- Training/Validation Accuracy Plots
- Training/Validation Loss Plots

Usage:
    from src.evaluation import evaluate_model
    metrics = evaluate_model(model, X_test, y_test, classes)
"""

import os
import sys
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUTS_DIR, REPORTS_DIR
from src.utils import get_logger

logger = get_logger("evaluation")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels (one-hot encoded or label encoded)
        y_pred: Predicted labels (one-hot encoded or label encoded)
        classes: List of class names
    
    Returns:
        Dictionary of metrics
    """
    # Convert one-hot to label indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    logger.info("Metrics calculated:")
    for key, value in metrics.items():
        logger.info("  %s: %.4f", key, value)
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    save_path: Optional[str] = None
) -> str:
    """
    Plot and save confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        save_path: Path to save the plot (optional)
    
    Returns:
        Path where the plot was saved
    """
    # Convert one-hot to label indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate figure size based on number of classes
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
    
    # Save plot
    if save_path is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info("Confusion matrix saved to: %s", save_path)
    return save_path


def plot_training_history(
    history,
    save_path: Optional[str] = None
) -> str:
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot (optional)
    
    Returns:
        Path where the plot was saved
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Performance", fontsize=14, fontweight="bold")
    h = history.history
    
    # Accuracy plot
    axes[0].plot(h["accuracy"], label="Train Accuracy", color="#2196F3", linewidth=2)
    if "val_accuracy" in h:
        axes[0].plot(h["val_accuracy"], label="Val Accuracy", color="#FF9800", 
                    linestyle="--", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
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
    
    # Save plot
    if save_path is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUTS_DIR, "training_curve.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info("Training curve saved to: %s", save_path)
    return save_path


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    save_path: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        save_path: Path to save the report (optional)
    
    Returns:
        Tuple of (report_string, save_path)
    """
    # Convert one-hot to label indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    
    # Save report
    if save_path is None:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        save_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info("Classification report saved to: %s", save_path)
    return report, save_path


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        classes: List of class names
        output_dir: Directory to save outputs (optional)
    
    Returns:
        Dictionary containing all metrics and file paths
    """
    logger.info("=" * 60)
    logger.info("Starting model evaluation")
    logger.info("=" * 60)
    
    # Set output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Convert y_test to label indices if one-hot encoded
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, classes)
    
    # Generate plots
    confusion_path = plot_confusion_matrix(y_true, y_pred, classes, output_dir)
    
    # Generate classification report
    report, report_path = generate_classification_report(y_true, y_pred, classes, output_dir)
    
    logger.info("\nClassification Report:\n%s", report)
    
    results = {
        'metrics': metrics,
        'confusion_matrix_path': confusion_path,
        'classification_report_path': report_path,
        'classification_report': report
    }
    
    logger.info("=" * 60)
    logger.info("Evaluation complete")
    logger.info("Accuracy: %.2f%%", metrics['accuracy'] * 100)
    logger.info("F1 Score (Macro): %.2f%%", metrics['f1_macro'] * 100)
    logger.info("F1 Score (Weighted): %.2f%%", metrics['f1_weighted'] * 100)
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    # Example usage
    import tensorflow as tf
    from src.config import MODEL_FILE, WORDS_FILE, CLASSES_FILE
    from src.preprocessing import load_artifacts
    from src.utils import load_intents
    
    logger.info("Loading model and artifacts...")
    model = tf.keras.models.load_model(MODEL_FILE)
    words, classes = load_artifacts()
    
    logger.info("Loading intents...")
    intents_data = load_intents()
    
    from src.preprocessing import build_vocabulary, encode_training_data
    words, classes, documents = build_vocabulary(intents_data)
    X, y = encode_training_data(documents, words, classes)
    
    logger.info("Evaluating model...")
    results = evaluate_model(model, X, y, classes)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Precision (Macro): {results['metrics']['precision_macro']:.4f}")
    print(f"Recall (Macro): {results['metrics']['recall_macro']:.4f}")
    print(f"F1 Score (Macro): {results['metrics']['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {results['metrics']['f1_weighted']:.4f}")
