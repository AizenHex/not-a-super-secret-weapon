"""AutoGluon sebagai pemeriksa: apakah langkah pembersihan membantu model, kolom mana yang penting,
dan probabilitas out-of-fold yang lebih kuat untuk cleanlab.

Perbandingan adil: train dibagi SEKALI jadi bagian latih (80%) dan bagian uji (20%, stratified).
Bagian uji diperlakukan seperti file test: ikut semua langkah tapi tidak pernah dibuang barisnya,
jadi data mentah dan data bersih dinilai di baris yang SAMA.
"""
from __future__ import annotations

import tempfile

import pandas as pd
from sklearn.model_selection import train_test_split

from core import steps as engine


def predictor(label: str, folder: str | None = None):
    from autogluon.tabular import TabularPredictor

    return TabularPredictor(label=label, path=folder or tempfile.mkdtemp(prefix="laya_ag_"), verbosity=0)


def split(train: pd.DataFrame, label: str, share: float = 0.2, seed: int = 0):
    data = train[train[label].notna()]
    stratify = data[label] if data[label].value_counts().min() >= 2 else None
    return train_test_split(data, test_size=share, random_state=seed, stratify=stratify)


def fit_and_score(fit_part: pd.DataFrame, hold_part: pd.DataFrame, label: str, time_limit: int,
                  presets: str) -> dict:
    model = predictor(label).fit(fit_part, time_limit=time_limit, presets=presets)
    scores = model.evaluate(hold_part, silent=True)
    board = model.leaderboard(hold_part, silent=True)[["model", "score_test", "score_val", "fit_time"]]
    return {"model": model, "skor": scores, "papan": board, "metrik": model.eval_metric.name}


def compare(train: pd.DataFrame, label: str, steps: list[dict], time_limit: int = 60,
            presets: str = "medium_quality") -> dict:
    """Skor AutoGluon di data mentah vs data setelah langkah, pada baris uji yang sama."""
    fit_part, hold_part = split(train, label)
    raw = fit_and_score(fit_part, hold_part, label, time_limit, presets)
    hold_features = hold_part.drop(columns=[label])
    result = engine.run(steps, label, fit_part, hold_features)
    if result["error"]:
        return {"mentah": raw, "galat": result["error"]}
    clean_hold = result["test"].assign(**{label: hold_part[label]})
    clean = fit_and_score(result["train"], clean_hold, label, time_limit, presets)
    return {"mentah": raw, "bersih": clean, "baris latih": (len(fit_part), len(result["train"])),
            "baris uji": len(hold_part), "galat": None}


def importance(model, hold_part: pd.DataFrame, rows: int = 2000) -> pd.DataFrame:
    """Permutation importance AutoGluon: turunnya skor bila isi kolom diacak."""
    table = model.feature_importance(hold_part, subsample_size=min(rows, len(hold_part)), silent=True)
    return table[["importance", "stddev", "p_value"]].rename_axis("kolom").reset_index()


def oof_probabilities(train: pd.DataFrame, label: str, time_limit: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Probabilitas out-of-fold (bagging 5 fold): tiap baris diprediksi model yang tidak melihatnya."""
    data = train[train[label].notna()]
    # fold dilatih berurutan di proses ini: multiprocessing spawn di Windows mengimpor ulang skrip utama
    model = predictor(label).fit(data, time_limit=time_limit, num_bag_folds=5, num_stack_levels=0,
                                 presets="medium_quality",
                                 ag_args_ensemble={"fold_fitting_strategy": "sequential_local"})
    return model.predict_proba_oof(), data[label]
