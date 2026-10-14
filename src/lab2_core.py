"""Core math and data logic for laboratory work 2.

The module intentionally avoids ML frameworks: the multilayer perceptron,
softmax cross-entropy and backpropagation are implemented directly with NumPy.
The GUI imports this file and only handles visualization and user interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Callable, Iterable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "Chinese_Dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MODEL_PATH = OUTPUT_DIR / "model_mlp.npz"

CLASS_NAMES = ["shan", "shui", "huo", "mu", "ri", "yue", "ren", "da", "tian", "zhong"]
CLASS_CHARS = ["山", "水", "火", "木", "日", "月", "人", "大", "天", "中"]


@dataclass(frozen=True)
class TrainingConfig:
    hidden_layers: tuple[int, ...] = (96, 48)
    epochs: int = 50
    learning_rate: float = 0.05
    batch_size: int = 32
    activation: str = "relu"
    train_ratio: float = 0.8
    seed: int = 34


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    paths_train: list[str]
    paths_val: list[str]
    class_names: list[str]
    class_chars: list[str]


def _read_annotations(dataset_dir: Path) -> list[dict[str, str]]:
    annotations_path = dataset_dir / "annotations.csv"
    with annotations_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def image_to_vector(path: Path | str) -> np.ndarray:
    """Convert a 32x32 grayscale image to a normalized 1024-value vector.

    Dataset files contain dark strokes on a light background, so values are
    inverted: 1.0 means a stroke pixel, 0.0 means background.
    """

    # На вход сети подается не картинка как объект PIL, а числовой вектор.
    # 32x32 пикселя разворачиваются в 1024 признака.
    image = Image.open(path).convert("L").resize((32, 32))
    pixels = np.asarray(image, dtype=np.float32)
    # В датасете фон светлый, а штрих темный. Инверсия делает штрих близким
    # к 1.0, а фон близким к 0.0. Так сети проще выделять форму символа.
    return ((255.0 - pixels) / 255.0).reshape(-1)


def canvas_image_to_vector(image: Image.Image) -> np.ndarray:
    image = image.convert("L").resize((32, 32))
    pixels = np.asarray(image, dtype=np.float32)
    return ((255.0 - pixels) / 255.0).reshape(1, -1)


def load_dataset(dataset_dir: Path = DATASET_DIR, config: TrainingConfig | None = None) -> Dataset:
    config = config or TrainingConfig()
    rows = _read_annotations(dataset_dir)

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    rel_paths: list[str] = []
    for row in rows:
        rel_path = row["filename"]
        vectors.append(image_to_vector(dataset_dir / rel_path))
        labels.append(int(row["class_id"]))
        rel_paths.append(rel_path)

    x = np.vstack(vectors).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    paths = np.asarray(rel_paths, dtype=object)

    # Стратифицированное разбиение: каждый класс отдельно делится 80/20.
    # Это важно, чтобы валидационная выборка содержала все 10 классов.
    rng = np.random.default_rng(config.seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_id in sorted(set(labels)):
        class_indices = np.flatnonzero(y == class_id)
        rng.shuffle(class_indices)
        split_at = int(round(len(class_indices) * config.train_ratio))
        train_indices.extend(class_indices[:split_at].tolist())
        val_indices.extend(class_indices[split_at:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return Dataset(
        x_train=x[train_indices],
        y_train=y[train_indices],
        x_val=x[val_indices],
        y_val=y[val_indices],
        paths_train=paths[train_indices].tolist(),
        paths_val=paths[val_indices].tolist(),
        class_names=CLASS_NAMES.copy(),
        class_chars=CLASS_CHARS.copy(),
    )


def export_pixels_csv(dataset_dir: Path = DATASET_DIR, output_path: Path | None = None) -> Path:
    output_path = output_path or (OUTPUT_DIR / "dataset_pixels.csv")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_annotations(dataset_dir)
    header = [f"p{i}" for i in range(1024)] + ["class_id", "class_name", "character", "filename"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        for row in rows:
            vector = image_to_vector(dataset_dir / row["filename"])
            # Методичка рекомендует CSV-представление с бинарными пикселями:
            # 1 означает наличие штриха, 0 означает фон.
            binary = (vector > 0.05).astype(int)
            writer.writerow(binary.tolist() + [row["class_id"], row["class_name"], row["character"], row["filename"]])
    return output_path


def _one_hot(y: np.ndarray, class_count: int) -> np.ndarray:
    out = np.zeros((len(y), class_count), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    # Softmax переводит выходные значения сети в вероятности классов.
    # Вычитание max нужно для численной устойчивости exp().
    stable = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(stable)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _activation_pair(name: str) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    if name == "relu":
        return lambda z: np.maximum(0.0, z), lambda z: (z > 0.0).astype(np.float32)
    if name == "tanh":
        return np.tanh, lambda z: 1.0 - np.tanh(z) ** 2
    if name == "sigmoid":
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        return sig, lambda z: sig(z) * (1.0 - sig(z))
    raise ValueError(f"Unknown activation: {name}")


def cross_entropy(proba: np.ndarray, y: np.ndarray) -> float:
    # Cross-Entropy штрафует модель, если вероятность правильного класса мала.
    # Для каждого объекта берется вероятность его истинного класса.
    eps = 1e-12
    return float(-np.mean(np.log(proba[np.arange(len(y)), y] + eps)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    # Строка матрицы - истинный класс, столбец - предсказанный класс.
    # Диагональ показывает количество правильных классификаций.
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def metrics_from_predictions(y_true: np.ndarray, proba: np.ndarray, class_count: int) -> dict:
    y_pred = np.argmax(proba, axis=1)
    matrix = confusion_matrix(y_true, y_pred, class_count)
    total = max(1, len(y_true))
    accuracy = float(np.trace(matrix) / total)

    per_class = []
    precisions = []
    recalls = []
    f1_scores = []
    for idx in range(class_count):
        # TP - верно найденные объекты класса idx.
        # FP - объекты других классов, ошибочно отнесенные к idx.
        # FN - объекты класса idx, ошибочно отнесенные к другим классам.
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_class.append(
            {
                "class_id": idx,
                "class_name": CLASS_NAMES[idx],
                "character": CLASS_CHARS[idx],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(matrix[idx, :].sum()),
            }
        )

    return {
        "loss": cross_entropy(proba, y_true),
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1_scores)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


class MLPClassifier:
    """Multilayer perceptron with dense layers and softmax output."""

    def __init__(self, input_size: int, hidden_layers: Iterable[int], output_size: int, activation: str, seed: int = 42):
        self.input_size = input_size
        self.hidden_layers = tuple(int(v) for v in hidden_layers if int(v) > 0)
        self.output_size = output_size
        self.activation_name = activation
        self.activation, self.activation_grad = _activation_pair(activation)
        self.rng = np.random.default_rng(seed)
        layer_sizes = (input_size, *self.hidden_layers, output_size)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Инициализация весов небольшими случайными числами. Для ReLU
            # используется He-подобный масштаб sqrt(2/fan_in).
            limit = np.sqrt(2.0 / fan_in) if activation == "relu" else np.sqrt(1.0 / fan_in)
            self.weights.append(self.rng.normal(0.0, limit, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        # Прямое распространение: данные проходят через скрытые слои,
        # затем выходной слой softmax возвращает вероятности 10 классов.
        activations = [x]
        zs: list[np.ndarray] = []
        current = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = current @ weight + bias
            zs.append(z)
            current = self.activation(z)
            activations.append(current)
        # На последнем слое нелинейность ReLU не применяется: logits сразу
        # переводятся в вероятности функцией softmax.
        logits = current @ self.weights[-1] + self.biases[-1]
        zs.append(logits)
        proba = _softmax(logits)
        activations.append(proba)
        return activations, zs, proba

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[2]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def train(
        self,
        dataset: Dataset,
        config: TrainingConfig,
        progress_callback: Callable[[int, dict], None] | None = None,
    ) -> list[dict]:
        y_train_ohe = _one_hot(dataset.y_train, self.output_size)
        history: list[dict] = []
        rng = np.random.default_rng(config.seed)

        for epoch in range(1, config.epochs + 1):
            # В начале каждой эпохи перемешиваем обучающую выборку, чтобы
            # mini-batch не шел всегда в одном и том же порядке.
            order = rng.permutation(len(dataset.x_train))
            for start in range(0, len(order), config.batch_size):
                batch_idx = order[start : start + config.batch_size]
                xb = dataset.x_train[batch_idx]
                yb = y_train_ohe[batch_idx]

                activations, zs, proba = self.forward(xb)
                # Для softmax + cross-entropy производная по logits имеет
                # простой вид: predicted_probabilities - one_hot_labels.
                delta = (proba - yb) / len(xb)
                grad_w: list[np.ndarray] = [np.empty_like(w) for w in self.weights]
                grad_b: list[np.ndarray] = [np.empty_like(b) for b in self.biases]

                grad_w[-1] = activations[-2].T @ delta
                grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

                # Обратное распространение ошибки по скрытым слоям.
                # Ошибка проходит справа налево через веса следующего слоя
                # и умножается на производную выбранной функции активации.
                for layer in range(len(self.weights) - 2, -1, -1):
                    delta = (delta @ self.weights[layer + 1].T) * self.activation_grad(zs[layer])
                    grad_w[layer] = activations[layer].T @ delta
                    grad_b[layer] = np.sum(delta, axis=0, keepdims=True)

                for layer in range(len(self.weights)):
                    # Шаг градиентного спуска: веса и смещения сдвигаются
                    # в сторону уменьшения функции потерь.
                    self.weights[layer] -= config.learning_rate * grad_w[layer]
                    self.biases[layer] -= config.learning_rate * grad_b[layer]

            # После каждой эпохи считаются метрики и на train, и на validation.
            # Validation не участвует в обновлении весов: это честная проверка.
            train_metrics = metrics_from_predictions(dataset.y_train, self.predict_proba(dataset.x_train), self.output_size)
            val_metrics = metrics_from_predictions(dataset.y_val, self.predict_proba(dataset.x_val), self.output_size)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
            history.append(row)
            if progress_callback:
                progress_callback(epoch, row)
        return history

    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Сохраняются только веса, смещения и архитектура. Этого достаточно,
        # чтобы потом восстановить обученную сеть без повторного обучения.
        payload = {f"w{i}": w for i, w in enumerate(self.weights)}
        payload.update({f"b{i}": b for i, b in enumerate(self.biases)})
        payload["hidden_layers"] = np.asarray(self.hidden_layers, dtype=np.int64)
        payload["activation"] = np.asarray(self.activation_name)
        np.savez(path, **payload)
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "MLPClassifier":
        data = np.load(path, allow_pickle=True)
        hidden_layers = tuple(int(v) for v in data["hidden_layers"].tolist())
        activation = str(data["activation"].tolist())
        weights = [data[f"w{i}"] for i in range(len(hidden_layers) + 1)]
        biases = [data[f"b{i}"] for i in range(len(hidden_layers) + 1)]
        model = cls(weights[0].shape[0], hidden_layers, weights[-1].shape[1], activation)
        model.weights = [w.astype(np.float32) for w in weights]
        model.biases = [b.astype(np.float32) for b in biases]
        return model


def parse_hidden_layers(raw: str) -> tuple[int, ...]:
    values = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("At least one hidden layer is required for variant 2.")
    return tuple(values)


def train_new_model(config: TrainingConfig) -> tuple[MLPClassifier, Dataset, list[dict], dict]:
    dataset = load_dataset(config=config)
    model = MLPClassifier(dataset.x_train.shape[1], config.hidden_layers, len(dataset.class_names), config.activation, config.seed)
    history = model.train(dataset, config)
    final_metrics = {
        "config": {
            "hidden_layers": list(config.hidden_layers),
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "activation": config.activation,
            "train_ratio": config.train_ratio,
            "seed": config.seed,
        },
        "train": metrics_from_predictions(dataset.y_train, model.predict_proba(dataset.x_train), len(dataset.class_names)),
        "validation": metrics_from_predictions(dataset.y_val, model.predict_proba(dataset.x_val), len(dataset.class_names)),
    }
    return model, dataset, history, final_metrics


def save_history(history: list[dict], path: Path | None = None) -> Path:
    path = path or (OUTPUT_DIR / "training_history.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(history)
    return path


def save_metrics(metrics: dict, path: Path | None = None) -> Path:
    path = path or (OUTPUT_DIR / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
