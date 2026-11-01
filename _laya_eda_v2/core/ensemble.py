"""Fitur extra E9: ensemble dari prediksi out-of-fold (OOF).

- **rata-rata berbobot**: bobot ≥ 0, jumlah 1, dicari di OOF (titik sudut = model tunggal + sampel Dirichlet acak
  dengan seed tetap);
- **stacking**: model meta (LogisticRegression / Ridge) dilatih di OOF; skornya diukur dengan cross_val_predict
  di OOF itu sendiri agar adil.
Ensemble hanya dipilih bila skornya LEBIH BAIK dari model tunggal terbaik (metrik E7). Masukan fungsi hanya OOF dan
label train: test tidak pernah dipakai untuk mencari bobot.
"""
from __future__ import annotations

import numpy as np

from core import training


def _combine(outputs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(w * o for w, o in zip(weights, outputs))


def weighted(oofs: dict[str, np.ndarray], y: np.ndarray, task: str, metric: str, thresholds=None,
             samples: int = 300, seed: int = 0) -> dict:
    names, outputs = list(oofs), list(oofs.values())
    k = len(names)
    rng = np.random.default_rng(seed)
    candidates = np.vstack([np.eye(k), rng.dirichlet(np.ones(k), samples)])
    cut = thresholds if thresholds is not None else training.default_thresholds(int(np.max(y)) + 1)
    scores = [training.score(metric, task, y, _combine(outputs, w), cut) for w in candidates]
    best = int(np.argmax(scores) if training.METRICS[metric]["higher"] else np.argmin(scores))
    singles = {name: scores[i] for i, name in enumerate(names)}
    return {"bobot": dict(zip(names, map(float, candidates[best]))), "skor": float(scores[best]), "tunggal": singles}


def stacking(oofs: dict[str, np.ndarray], y: np.ndarray, task: str, metric: str, thresholds=None) -> dict:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

    features = np.hstack([o.reshape(len(y), -1) for o in oofs.values()])
    if task == "kelas":
        meta = LogisticRegression(max_iter=2000)
        output = cross_val_predict(meta, features, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                   method="predict_proba")
    else:
        meta = Ridge(alpha=1.0)
        output = cross_val_predict(meta, features, y, cv=KFold(5, shuffle=True, random_state=0))
    cut = thresholds if thresholds is not None else training.default_thresholds(int(np.max(y)) + 1)
    return {"skor": training.score(metric, task, y, output, cut), "meta": meta.fit(features, y)}


def choose(oofs: dict[str, np.ndarray], y: np.ndarray, task: str, metric: str, thresholds=None) -> dict:
    """Bandingkan model tunggal terbaik, rata-rata berbobot, dan stacking; pilih yang terbaik (seri → tunggal)."""
    blend = weighted(oofs, y, task, metric, thresholds)
    single_name = max(blend["tunggal"], key=blend["tunggal"].get) if training.METRICS[metric]["higher"] \
        else min(blend["tunggal"], key=blend["tunggal"].get)
    single = blend["tunggal"][single_name]
    stack = stacking(oofs, y, task, metric, thresholds) if len(oofs) > 1 else {"skor": single}
    choice, best = single_name, single
    if training.better(metric, blend["skor"], best) and max(blend["bobot"].values()) < 1:
        choice, best = "rata-rata berbobot", blend["skor"]
    if training.better(metric, stack["skor"], best):
        choice, best = "stacking", stack["skor"]
    return {"pilihan": choice, "skor": best, "tunggal_terbaik": single_name, "skor_tunggal": single,
            "bobot": blend["bobot"], "skor_bobot": blend["skor"], "skor_stacking": stack["skor"],
            "skor_per_model": blend["tunggal"], "meta": stack.get("meta")}


def combine_outputs(choice: dict, outputs: dict[str, np.ndarray], task: str) -> np.ndarray:
    """Keluaran mentah (proba / nilai) tiap model di test → keluaran ensemble terpilih (bobot/meta dari OOF)."""
    if choice["pilihan"] == "rata-rata berbobot":
        return _combine([outputs[name] for name in choice["bobot"]], np.array(list(choice["bobot"].values())))
    if choice["pilihan"] == "stacking":
        features = np.hstack([outputs[name].reshape(len(next(iter(outputs.values()))), -1) for name in choice["bobot"]])
        return choice["meta"].predict_proba(features) if task == "kelas" else choice["meta"].predict(features)
    return outputs[choice["pilihan"]]


__all__ = ["weighted", "stacking", "choose", "combine_outputs"]
