"""Reproducible non-GUI run for lab 2.

This script trains the MLP from scratch with the same core code used by the GUI
and writes honest artifacts for the report: CSV dataset, history, metrics,
confusion matrix and illustrative plots.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab2_core import (
    CLASS_NAMES,
    DATASET_DIR,
    OUTPUT_DIR,
    TrainingConfig,
    export_pixels_csv,
    save_history,
    save_metrics,
    train_new_model,
)


def save_loss_plot(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.plot(epochs, [row["train_loss"] for row in history], color="#d64f8f", linewidth=2.2, label="train loss")
    ax.plot(epochs, [row["val_loss"] for row in history], color="#7457c8", linewidth=2.2, label="validation loss")
    ax.set_title("Изменение Cross-Entropy Loss при обучении MLP")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_accuracy_plot(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.plot(epochs, [row["train_accuracy"] for row in history], color="#db5c9f", linewidth=2.2, label="train accuracy")
    ax.plot(epochs, [row["val_accuracy"] for row in history], color="#41a8a8", linewidth=2.2, label="validation accuracy")
    ax.set_title("Accuracy на обучающей и валидационной выборках")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_confusion_matrix(metrics: dict, path: Path) -> None:
    matrix = np.asarray(metrics["validation"]["confusion_matrix"], dtype=int)
    labels = CLASS_NAMES
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=160)
    image = ax.imshow(matrix, cmap="RdPu")
    ax.set_title("Матрица ошибок на валидационной выборке")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > matrix.max() * 0.55 else "#3b2331"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_confusion_csv(metrics: dict, path: Path) -> None:
    matrix = metrics["validation"]["confusion_matrix"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["true/predicted", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name, *row])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(hidden_layers=(96, 48), epochs=50, learning_rate=0.05, batch_size=32, activation="relu", seed=34)
    export_pixels_csv(DATASET_DIR, OUTPUT_DIR / "dataset_pixels.csv")
    model, _dataset, history, metrics = train_new_model(config)
    model.save(OUTPUT_DIR / "model_mlp.npz")
    save_history(history, OUTPUT_DIR / "training_history.csv")
    save_metrics(metrics, OUTPUT_DIR / "metrics.json")
    save_loss_plot(history, OUTPUT_DIR / "loss_curve.png")
    save_accuracy_plot(history, OUTPUT_DIR / "accuracy_curve.png")
    save_confusion_matrix(metrics, OUTPUT_DIR / "confusion_matrix.png")
    save_confusion_csv(metrics, OUTPUT_DIR / "confusion_matrix.csv")
    print("Training complete")
    print(f"train_accuracy={metrics['train']['accuracy']:.4f}")
    print(f"val_accuracy={metrics['validation']['accuracy']:.4f}")
    print(f"val_loss={metrics['validation']['loss']:.4f}")
    print(f"artifacts={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
