"""Tkinter GUI for lab 2 MLP classifier.

Run from PyCharm or terminal:
    python gui_app.py
"""

from __future__ import annotations

import random
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageTk

from lab2_core import (
    CLASS_CHARS,
    CLASS_NAMES,
    DATASET_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    MLPClassifier,
    TrainingConfig,
    canvas_image_to_vector,
    export_pixels_csv,
    load_dataset,
    parse_hidden_layers,
    save_history,
    save_metrics,
    train_new_model,
)


PINK = "#f9dbe8"
PINK_LIGHT = "#fff7fb"
PINK_MID = "#f3c8dc"
ROSE = "#d94f93"
ROSE_DARK = "#b83a75"
INK = "#362436"
MUTED = "#7c6376"
LILAC = "#8d72c7"
TEAL = "#3aa6a6"
LINE = "#efaccb"

IMAGE_SIZE = 32
CANVAS_SIZE = 320
BRUSH_SIZE = 10


class SakuraMLPApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ЛР2 - MLP для рукописных китайских иероглифов")
        self.geometry("1220x780")
        self.minsize(1080, 700)
        self.configure(bg=PINK)

        self.model: MLPClassifier | None = None
        self.dataset = None
        self.history: list[dict] = []
        self.metrics: dict | None = None
        self.events: queue.Queue = queue.Queue()
        self.training_thread: threading.Thread | None = None

        self.draw_image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        self.draw_context = ImageDraw.Draw(self.draw_image)
        self.last_xy: tuple[int, int] | None = None
        self.loaded_preview: ImageTk.PhotoImage | None = None

        self._configure_style()
        self._build_layout()
        self._draw_grid()
        self.after(120, self._poll_events)

        if MODEL_PATH.exists():
            try:
                self.model = MLPClassifier.load(MODEL_PATH)
                self.status_var.set(f"Загружены сохраненные веса: {MODEL_PATH.name}")
            except Exception as exc:
                self.status_var.set(f"Не удалось загрузить веса: {exc}")

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=PINK, foreground=INK)
        style.configure("TFrame", background=PINK)
        style.configure("Card.TFrame", background=PINK_LIGHT, relief="flat")
        style.configure("TLabel", background=PINK, foreground=INK)
        style.configure("Card.TLabel", background=PINK_LIGHT, foreground=INK)
        style.configure("Muted.Card.TLabel", background=PINK_LIGHT, foreground=MUTED)
        style.configure("Title.TLabel", background=PINK, foreground=INK, font=("Segoe UI Semibold", 19))
        style.configure("Subtitle.TLabel", background=PINK, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Accent.TButton", background=ROSE, foreground="white", borderwidth=0, padding=(14, 8))
        style.map("Accent.TButton", background=[("active", ROSE_DARK), ("disabled", "#d9a7bf")])
        style.configure("Soft.TButton", background="#ffffff", foreground=INK, borderwidth=0, padding=(12, 7))
        style.map("Soft.TButton", background=[("active", "#fdebf3")])
        style.configure("TButton", padding=(12, 7), background="#ffffff")
        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure("TCombobox", fieldbackground="#ffffff")

    def _build_layout(self) -> None:
        header = ttk.Frame(self, padding=(20, 14, 20, 8))
        header.pack(fill="x")
        title_box = ttk.Frame(header)
        title_box.pack(side="left", fill="x", expand=True)
        ttk.Label(title_box, text="Многослойный перцептрон для китайских иероглифов", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_box, text="Обучение, проверка и распознавание рукописных символов", style="Subtitle.TLabel").pack(anchor="w")
        ttk.Label(header, text="✿  山 水 火 木 日 月 人 大 天 中", background=PINK, foreground=ROSE, font=("Segoe UI", 15)).pack(side="right")

        main = ttk.Frame(self, padding=(14, 6, 14, 12))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=0)
        main.rowconfigure(0, weight=1)

        self._build_controls(main)
        self._build_drawing_area(main)
        self._build_results(main)

        self.status_var = tk.StringVar(value="Готово. Можно обучить модель, загрузить веса или распознать изображение из датасета.")
        ttk.Label(self, textvariable=self.status_var, background=PINK_MID, foreground=INK, anchor="w", padding=(14, 6)).pack(fill="x", side="bottom")

    def _build_controls(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent, style="Card.TFrame", padding=16)
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        controls.columnconfigure(1, weight=1)

        self.epochs_var = tk.StringVar(value="50")
        self.lr_var = tk.StringVar(value="0.05")
        self.batch_var = tk.StringVar(value="32")
        self.hidden_var = tk.StringVar(value="96,48")
        self.activation_var = tk.StringVar(value="relu")
        self.seed_var = tk.StringVar(value="34")

        ttk.Label(controls, text="Обучение", style="Card.TLabel", font=("Segoe UI Semibold", 14)).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self._field(controls, "Эпохи", self.epochs_var, 1)
        self._field(controls, "Learning rate", self.lr_var, 2)
        self._field(controls, "Batch size", self.batch_var, 3)
        self._field(controls, "Seed", self.seed_var, 4)

        ttk.Label(controls, text="Активация", style="Card.TLabel").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Combobox(controls, textvariable=self.activation_var, values=["relu", "tanh", "sigmoid"], state="readonly", width=12).grid(row=5, column=1, sticky="ew", pady=6)

        ttk.Label(controls, text="Скрытые слои", style="Card.TLabel").grid(row=6, column=0, sticky="w", pady=(12, 4))
        ttk.Entry(controls, textvariable=self.hidden_var, width=18).grid(row=6, column=1, sticky="ew", pady=(12, 4))
        layer_buttons = ttk.Frame(controls, style="Card.TFrame")
        layer_buttons.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(2, 10))
        layer_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(layer_buttons, text="+ слой", style="Soft.TButton", command=self.add_hidden_layer).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(layer_buttons, text="- слой", style="Soft.TButton", command=self.remove_hidden_layer).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Button(controls, text="Обучить с нуля", style="Accent.TButton", command=self.start_training).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(16, 5))
        ttk.Button(controls, text="Графики", style="Soft.TButton", command=self.open_graph_window).grid(row=10, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="Сохранить веса", style="Soft.TButton", command=self.save_model).grid(row=11, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="Загрузить веса", style="Soft.TButton", command=self.load_model).grid(row=12, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(controls, text="CSV пикселей", style="Soft.TButton", command=self.export_csv).grid(row=13, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Separator(controls).grid(row=14, column=0, columnspan=2, sticky="ew", pady=16)
        self.metric_text = tk.Text(controls, width=35, height=14, bg="#fffafd", fg=INK, relief="flat", wrap="word", padx=8, pady=8)
        self.metric_text.grid(row=15, column=0, columnspan=2, sticky="nsew")
        controls.rowconfigure(15, weight=1)

    def _build_drawing_area(self, parent: ttk.Frame) -> None:
        draw_card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        draw_card.grid(row=0, column=1, sticky="nsew", padx=(0, 12))
        draw_card.columnconfigure(0, weight=1)
        draw_card.rowconfigure(2, weight=1)

        ttk.Label(draw_card, text="Рисование и распознавание", style="Card.TLabel", font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w")
        ttk.Label(draw_card, text="Нарисуйте символ или загрузите готовое изображение.", style="Muted.Card.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 10))

        canvas_wrap = ttk.Frame(draw_card, style="Card.TFrame")
        canvas_wrap.grid(row=2, column=0, sticky="n")
        self.canvas = tk.Canvas(
            canvas_wrap,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            cursor="crosshair",
            highlightthickness=2,
            highlightbackground=LINE,
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        buttons = ttk.Frame(draw_card, style="Card.TFrame")
        buttons.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(buttons, text="Распознать рисунок", style="Accent.TButton", command=self.recognize_canvas).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(buttons, text="Загрузить BMP", style="Soft.TButton", command=self.load_image_file).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(buttons, text="Очистить", style="Soft.TButton", command=self.clear_canvas).grid(row=0, column=2, sticky="ew", padx=(5, 0))

        bottom_buttons = ttk.Frame(draw_card, style="Card.TFrame")
        bottom_buttons.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        bottom_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(bottom_buttons, text="Случайное train-изображение", style="Soft.TButton", command=self.load_random_train_image).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(bottom_buttons, text="Показать графики", style="Soft.TButton", command=self.open_graph_window).grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _build_results(self, parent: ttk.Frame) -> None:
        result_card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        result_card.grid(row=0, column=2, sticky="ns")
        result_card.columnconfigure(0, weight=1)
        result_card.rowconfigure(3, weight=1)

        ttk.Label(result_card, text="Результат", style="Card.TLabel", font=("Segoe UI Semibold", 14)).grid(row=0, column=0, sticky="w")
        self.result_label = ttk.Label(
            result_card,
            text="Нарисуйте иероглиф или загрузите BMP из датасета.",
            style="Card.TLabel",
            font=("Segoe UI", 12),
            wraplength=330,
        )
        self.result_label.grid(row=1, column=0, sticky="ew", pady=(8, 10))

        self.preview_label = ttk.Label(result_card, style="Card.TLabel")
        self.preview_label.grid(row=2, column=0, sticky="n", pady=(0, 10))

        self.proba_text = tk.Text(result_card, width=38, height=22, bg="#fffafd", fg=INK, relief="flat", padx=8, pady=8)
        self.proba_text.grid(row=3, column=0, sticky="nsew")

    def _field(self, parent: ttk.Frame, label: str, var: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=var, width=14).grid(row=row, column=1, sticky="ew", pady=6)

    def add_hidden_layer(self) -> None:
        try:
            layers = list(parse_hidden_layers(self.hidden_var.get()))
        except Exception:
            layers = []
        layers.append(64)
        self.hidden_var.set(",".join(str(v) for v in layers))

    def remove_hidden_layer(self) -> None:
        try:
            layers = list(parse_hidden_layers(self.hidden_var.get()))
        except Exception:
            layers = []
        if len(layers) > 1:
            layers.pop()
        self.hidden_var.set(",".join(str(v) for v in layers))

    def _config_from_ui(self) -> TrainingConfig:
        return TrainingConfig(
            hidden_layers=parse_hidden_layers(self.hidden_var.get()),
            epochs=int(self.epochs_var.get()),
            learning_rate=float(self.lr_var.get().replace(",", ".")),
            batch_size=int(self.batch_var.get()),
            activation=self.activation_var.get(),
            seed=int(self.seed_var.get()),
        )

    def start_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Обучение", "Обучение уже выполняется.")
            return
        try:
            config = self._config_from_ui()
        except Exception as exc:
            messagebox.showerror("Параметры", str(exc))
            return
        self.history = []
        self.metrics = None
        self.metric_text.delete("1.0", "end")
        self.status_var.set("Обучение запущено...")
        self.training_thread = threading.Thread(target=self._train_worker, args=(config,), daemon=True)
        self.training_thread.start()

    def _train_worker(self, config: TrainingConfig) -> None:
        try:
            model, dataset, history, metrics = train_new_model(config)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save(MODEL_PATH)
            save_history(history)
            save_metrics(metrics)
            self.events.put(("done", model, dataset, history, metrics))
        except Exception as exc:
            self.events.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "done":
                    _, self.model, self.dataset, self.history, self.metrics = event
                    self.status_var.set("Обучение завершено, веса сохранены.")
                    self._show_metrics()
                elif event[0] == "error":
                    messagebox.showerror("Ошибка обучения", str(event[1]))
                    self.status_var.set("Ошибка обучения.")
        except queue.Empty:
            pass
        self.after(120, self._poll_events)

    def _show_metrics(self) -> None:
        self.metric_text.delete("1.0", "end")
        if not self.metrics:
            return
        val = self.metrics["validation"]
        train = self.metrics["train"]
        lines = [
            "Итоговые метрики",
            "",
            f"Train accuracy: {train['accuracy']:.4f}",
            f"Train precision: {train['precision']:.4f}",
            f"Train recall: {train['recall']:.4f}",
            f"Train loss: {train['loss']:.4f}",
            "",
            f"Validation accuracy: {val['accuracy']:.4f}",
            f"Validation precision: {val['precision']:.4f}",
            f"Validation recall: {val['recall']:.4f}",
            f"Validation loss: {val['loss']:.4f}",
            "",
            "По классам:",
        ]
        for item in val["per_class"]:
            lines.append(f"{item['class_name']} {item['character']}: P={item['precision']:.2f}, R={item['recall']:.2f}, F1={item['f1']:.2f}")
        self.metric_text.insert("1.0", "\n".join(lines))

    def _draw_grid(self) -> None:
        self.canvas.delete("grid")
        step = CANVAS_SIZE / IMAGE_SIZE
        for i in range(1, IMAGE_SIZE):
            p = i * step
            self.canvas.create_line(p, 0, p, CANVAS_SIZE, fill="#f1e7ee", tags="grid")
            self.canvas.create_line(0, p, CANVAS_SIZE, p, fill="#f1e7ee", tags="grid")

    def _start_draw(self, event) -> None:
        self.last_xy = (event.x, event.y)
        self._draw_point(event.x, event.y)

    def _draw(self, event) -> None:
        if self.last_xy is None:
            self.last_xy = (event.x, event.y)
        x0, y0 = self.last_xy
        x1, y1 = event.x, event.y
        self.canvas.create_line(x0, y0, x1, y1, fill="#1f1721", width=BRUSH_SIZE, capstyle=tk.ROUND, smooth=True, tags="stroke")

        scale = IMAGE_SIZE / CANVAS_SIZE
        self.draw_context.line((x0 * scale, y0 * scale, x1 * scale, y1 * scale), fill=0, width=max(1, BRUSH_SIZE // 10))
        self.last_xy = (x1, y1)

    def _draw_point(self, x: int, y: int) -> None:
        r = BRUSH_SIZE / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#1f1721", outline="#1f1721", tags="stroke")
        scale = IMAGE_SIZE / CANVAS_SIZE
        ix = x * scale
        iy = y * scale
        rr = max(0.45, BRUSH_SIZE * scale / 2)
        self.draw_context.ellipse((ix - rr, iy - rr, ix + rr, iy + rr), fill=0)

    def _stop_draw(self, _event) -> None:
        self.last_xy = None

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self.draw_image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        self.draw_context = ImageDraw.Draw(self.draw_image)
        self.loaded_preview = None
        self.preview_label.configure(image="")
        self.result_label.configure(text="Нарисуйте иероглиф или загрузите BMP из датасета.")
        self.proba_text.delete("1.0", "end")

    def recognize_canvas(self) -> None:
        self._recognize_image(self.draw_image, source="рисунок")

    def _recognize_image(self, image: Image.Image, source: str, true_label: str | None = None) -> None:
        if self.model is None:
            messagebox.showwarning("Модель", "Сначала обучите модель или загрузите сохраненные веса.")
            return
        proba = self.model.predict_proba(canvas_image_to_vector(image))[0]
        order = proba.argsort()[::-1]
        best = int(order[0])
        expected = f"\nИстинный класс: {true_label}" if true_label else ""
        self.result_label.configure(text=f"Источник: {source}\nРаспознано: {CLASS_NAMES[best]} {CLASS_CHARS[best]} ({proba[best]:.2%}){expected}")
        self.proba_text.delete("1.0", "end")
        self.proba_text.insert("1.0", "\n".join(f"{CLASS_NAMES[i]:>6} {CLASS_CHARS[i]}: {proba[i]:.2%}" for i in order))

    def load_image_file(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(DATASET_DIR),
            filetypes=[("Images", "*.bmp *.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_and_recognize_path(Path(path))

    def load_random_train_image(self) -> None:
        try:
            config = self._config_from_ui()
            self.dataset = load_dataset(config=config)
            rel_path = random.choice(self.dataset.paths_train)
            self._load_and_recognize_path(DATASET_DIR / rel_path)
        except Exception as exc:
            messagebox.showerror("Train image", str(exc))

    def _load_and_recognize_path(self, path: Path) -> None:
        image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        self.clear_canvas()
        self.draw_image = image.copy()
        self.draw_context = ImageDraw.Draw(self.draw_image)
        preview = image.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.NEAREST)
        self.loaded_preview = ImageTk.PhotoImage(preview)
        self.canvas.create_image(0, 0, image=self.loaded_preview, anchor="nw", tags="stroke")
        self._draw_grid()
        self.status_var.set(f"Загружено изображение: {path.name}")
        self._recognize_image(image, source=path.name, true_label=self._true_label_from_path(path))

    def _true_label_from_path(self, path: Path) -> str | None:
        for name, char in zip(CLASS_NAMES, CLASS_CHARS):
            if path.parent.name == name or path.name.startswith(name):
                return f"{name} {char}"
        return None

    def open_graph_window(self) -> None:
        if not self.history:
            messagebox.showinfo("Графики", "Сначала обучите модель: история обучения пока пустая.")
            return
        GraphWindow(self, self.history)

    def save_model(self) -> None:
        if self.model is None:
            messagebox.showwarning("Модель", "Нет обученной модели для сохранения.")
            return
        path = filedialog.asksaveasfilename(initialfile="model_mlp.npz", defaultextension=".npz", filetypes=[("NumPy weights", "*.npz")])
        if path:
            self.model.save(Path(path))
            self.status_var.set(f"Веса сохранены: {path}")

    def load_model(self) -> None:
        path = filedialog.askopenfilename(initialdir=str(OUTPUT_DIR), filetypes=[("NumPy weights", "*.npz")])
        if path:
            self.model = MLPClassifier.load(Path(path))
            self.status_var.set(f"Веса загружены: {path}")

    def export_csv(self) -> None:
        try:
            path = export_pixels_csv()
            size_mb = path.stat().st_size / (1024 * 1024)
        except Exception as exc:
            messagebox.showerror("CSV пикселей", f"Не удалось создать CSV:\n{exc}")
            self.status_var.set("Ошибка создания CSV с пикселями.")
            return
        self.status_var.set(f"CSV с бинарными пикселями создан: {path}")
        messagebox.showinfo(
            "CSV пикселей",
            f"Файл создан:\n{path}\n\nРазмер: {size_mb:.2f} МБ",
        )


class GraphWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, history: list[dict]) -> None:
        super().__init__(parent)
        self.title("Графики обучения MLP")
        self.geometry("980x720")
        self.configure(bg=PINK)
        self.history = history
        self.annotations = {}

        top = ttk.Frame(self, padding=(14, 12, 14, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Графики текущего обучения", style="Title.TLabel").pack(side="left")
        ttk.Button(top, text="Экспорт PNG", style="Accent.TButton", command=self.export_png).pack(side="right")

        body = ttk.Frame(self, style="Card.TFrame", padding=12)
        body.pack(fill="both", expand=True, padx=14, pady=10)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(9, 6.4), dpi=100, facecolor=PINK_LIGHT)
        self.ax_loss = self.figure.add_subplot(211)
        self.ax_acc = self.figure.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.figure, master=body)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._plot()

    def _plot(self) -> None:
        epochs = [row["epoch"] for row in self.history]
        for ax in (self.ax_loss, self.ax_acc):
            ax.clear()
            ax.set_facecolor(PINK_LIGHT)
            ax.grid(True, color="#f0c8da", alpha=0.7)

        self.ax_loss.plot(epochs, [row["train_loss"] for row in self.history], color=ROSE, linewidth=2.2, marker="o", markersize=3, label="train loss")
        self.ax_loss.plot(epochs, [row["val_loss"] for row in self.history], color=LILAC, linewidth=2.2, marker="o", markersize=3, label="validation loss")
        self.ax_acc.plot(epochs, [row["train_accuracy"] for row in self.history], color=ROSE, linewidth=2.2, marker="o", markersize=3, label="train accuracy")
        self.ax_acc.plot(epochs, [row["val_accuracy"] for row in self.history], color=TEAL, linewidth=2.2, marker="o", markersize=3, label="validation accuracy")

        self.ax_loss.set_title("Cross-Entropy Loss")
        self.ax_loss.set_xlabel("Эпоха")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend(loc="upper right")
        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("Эпоха")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.set_ylim(0, 1.02)
        self.ax_acc.legend(loc="lower right")

        for ax in (self.ax_loss, self.ax_acc):
            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(12, 12),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="#fffafd", ec=ROSE),
                arrowprops=dict(arrowstyle="->", color=ROSE),
            )
            annot.set_visible(False)
            self.annotations[ax] = annot

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_motion(self, event) -> None:
        changed = False
        for annot in self.annotations.values():
            if annot.get_visible():
                annot.set_visible(False)
                changed = True
        if event.inaxes in self.annotations and event.xdata is not None and event.ydata is not None:
            annot = self.annotations[event.inaxes]
            annot.xy = (event.xdata, event.ydata)
            annot.set_text(f"x={event.xdata:.2f}\ny={event.ydata:.4f}")
            annot.set_visible(True)
            changed = True
        if changed:
            self.canvas.draw_idle()

    def export_png(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            initialfile="training_graphs.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
        )
        if path:
            self.figure.savefig(path, dpi=160, facecolor=PINK_LIGHT)


if __name__ == "__main__":
    app = SakuraMLPApp()
    app.mainloop()
