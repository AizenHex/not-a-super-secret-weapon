"""Fitur extra E11: analisis kesalahan dari prediksi OOF + figur presentasi.

- confusion matrix (nilai ditulis di sel), pasangan kelas yang paling sering tertukar;
- baris dengan salah terbesar (indeks baris train asli → bisa dibuka di halaman Data);
- permutation importance (sklearn) dari model fold pertama di fold validasinya (bukan di data latihnya);
- figur PNG 16:9, huruf besar, tanpa judul di dalam gambar, colormap berurutan (bukan jet), label langsung
  (aturan figur `_memori/aturan.md`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import training

FIGSIZE = (16, 9)        # inci → 16:9
FONT = 18


def oof_labels(result: dict) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """(label asli, tebakan OOF, indeks baris train) untuk baris yang punya OOF."""
    rows = result["cv"]["baris_oof"]
    setelan, oof = result["setelan"], result["cv"]["oof"][rows]
    codes = training.decide(setelan["tugas"], oof, result["ambang"] or
                            training.default_thresholds(setelan["n_kelas"]), result["bobot_kelas"])
    classes = result["kelas"]
    return classes[result["y"][rows]], classes[np.clip(codes, 0, len(classes) - 1)], result["X"].index[rows]


def confusion(result: dict) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix

    truth, guess, _ = oof_labels(result)
    classes = result["kelas"]
    table = confusion_matrix(truth, guess, labels=classes)
    return pd.DataFrame(table, index=pd.Index(classes, name="label asli"), columns=pd.Index(classes, name="tebakan"))


def most_confused(table: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    pairs = table.stack().rename("baris").reset_index()
    pairs = pairs[pairs["label asli"] != pairs["tebakan"]].sort_values("baris", ascending=False).head(top)
    pairs["porsi dari kelas asli"] = (pairs["baris"] / pairs["label asli"].map(table.sum(axis=1))).round(3)
    return pairs.reset_index(drop=True)


def per_class(result: dict) -> pd.DataFrame:
    """Presisi/recall/F1 per kelas dari OOF (E12: skor per kelas sebelum/sesudah penanganan timpang)."""
    from sklearn.metrics import precision_recall_fscore_support

    truth, guess, _ = oof_labels(result)
    precision, recall, f1, support = precision_recall_fscore_support(truth, guess, labels=result["kelas"],
                                                                     zero_division=0)
    return pd.DataFrame({"kelas": result["kelas"], "presisi": precision.round(3), "recall": recall.round(3),
                         "f1": f1.round(3), "baris": support})


def worst_rows(result: dict, top: int = 50) -> pd.DataFrame:
    """Baris dengan salah terbesar: klasifikasi = 1 − peluang kelas asli; lainnya = |asli − tebakan|."""
    rows = result["cv"]["baris_oof"]
    oof, y = result["cv"]["oof"][rows], result["y"][rows]
    index = result["X"].index[rows]
    if result["setelan"]["tugas"] == "kelas":
        error = 1 - oof[np.arange(len(y)), y.astype(int)]
    else:
        error = np.abs(oof - y)
    frame = pd.DataFrame({"baris": index, "salah": np.round(error, 4)})
    if result["kelas"] is not None:
        truth, guess, _ = oof_labels(result)
        frame["label asli"], frame["tebakan"] = truth, guess
    return frame.sort_values("salah", ascending=False).head(top).reset_index(drop=True)


def importance(result: dict, repeats: int = 3, rows: int = 2000) -> pd.DataFrame:
    """Permutation importance model fold pertama, dinilai di fold validasinya (data yang tidak dilihat)."""
    from sklearn.inspection import permutation_importance

    pipe, val_rows = result["cv"]["model_fold"][0]
    X, y = result["X"].iloc[val_rows[:rows]], result["y"][val_rows[:rows]]
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "rmse": "neg_root_mean_squared_error",
               "mae": "neg_mean_absolute_error", "r2": "r2", "logloss": "neg_log_loss"}.get(
        result["setelan"]["metrik"], None)
    found = permutation_importance(pipe, X, y, n_repeats=repeats, random_state=0, scoring=scoring)
    return (pd.DataFrame({"kolom": X.columns, "importance": found.importances_mean.round(5),
                          "stddev": found.importances_std.round(5)})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": FONT})
    return plt, plt.subplots(figsize=FIGSIZE, dpi=100)


def figure_confusion(table: pd.DataFrame, path: str | Path) -> Path:
    plt, (fig, ax) = _figure()
    values = table.to_numpy()
    ax.imshow(values, cmap="Blues")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > values.max() / 2 else "black"
            ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center", color=color, fontsize=FONT)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("tebakan (OOF)")
    ax.set_ylabel("label asli")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_importance(table: pd.DataFrame, path: str | Path, top: int = 15) -> Path:
    plt, (fig, ax) = _figure()
    part = table.head(top).iloc[::-1]
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.barh(part["kolom"].astype(str).str.slice(0, 45), part["importance"], color="#3b6ea5", zorder=2)
    for y_pos, value in enumerate(part["importance"]):
        ax.text(value, y_pos, f" {value:.4f}", va="center", fontsize=FONT - 4)
    ax.set_xlabel("turunnya skor bila kolom diacak (permutation importance)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = ["confusion", "most_confused", "per_class", "worst_rows", "importance", "figure_confusion",
           "figure_importance"]
