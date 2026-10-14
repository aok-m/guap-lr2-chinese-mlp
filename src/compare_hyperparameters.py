"""Reproducible comparison of MLP hyperparameters for the lab report.

Only one factor is changed in each group. The dataset split, initialization
seed and all remaining parameters stay fixed, so the rows can be compared.
"""

from __future__ import annotations

import csv
import json

from lab2_core import OUTPUT_DIR, TrainingConfig, train_new_model


BASE = {
    "hidden_layers": (96, 48),
    "epochs": 50,
    "learning_rate": 0.05,
    "batch_size": 32,
    "activation": "relu",
    "seed": 34,
}


def run(group: str, value: str, **changes: object) -> dict:
    parameters = BASE | changes
    config = TrainingConfig(**parameters)
    _model, _dataset, history, metrics = train_new_model(config)
    validation = metrics["validation"]
    result = {
        "group": group,
        "value": value,
        "hidden_layers": ",".join(map(str, config.hidden_layers)),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "activation": config.activation,
        "seed": config.seed,
        "val_accuracy": validation["accuracy"],
        "val_precision": validation["precision"],
        "val_recall": validation["recall"],
        "val_f1": validation["f1"],
        "val_loss": validation["loss"],
        "train_accuracy": metrics["train"]["accuracy"],
    }
    print(
        f"{group:12} {value:10} "
        f"accuracy={result['val_accuracy']:.4f} loss={result['val_loss']:.4f}"
    )
    return result


def main() -> None:
    experiments = [
        ("activation", "relu", {}),
        ("activation", "tanh", {"activation": "tanh"}),
        ("activation", "sigmoid", {"activation": "sigmoid"}),
        ("architecture", "64", {"hidden_layers": (64,)}),
        ("architecture", "96", {"hidden_layers": (96,)}),
        ("architecture", "192", {"hidden_layers": (192,)}),
        ("architecture", "96,48", {"hidden_layers": (96, 48)}),
        ("architecture", "192,96", {"hidden_layers": (192, 96)}),
        ("learning_rate", "0.01", {"learning_rate": 0.01}),
        ("learning_rate", "0.03", {"learning_rate": 0.03}),
        ("learning_rate", "0.05", {}),
        ("learning_rate", "0.08", {"learning_rate": 0.08}),
        ("epochs", "20", {"epochs": 20}),
        ("epochs", "30", {"epochs": 30}),
        ("epochs", "50", {}),
        ("epochs", "80", {"epochs": 80}),
    ]
    rows = [run(group, value, **changes) for group, value, changes in experiments]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "hyperparameter_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "hyperparameter_comparison.json").write_text(
        json.dumps({"baseline": BASE, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
