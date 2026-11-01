"""Fitur extra E3, E5, E7, E8, E10, E12 (VISI.md bagian 0): validasi silang, latih cepat, metrik + label
berurutan, log percobaan, anggaran waktu, kelas timpang.

Alur satu percobaan (`experiment`): train mentah → split proyek (core/split) → blok EDA (langkah pandas, statistik
dari train) → baris berlabel → k-fold: pipeline (core/pipeline) di-fit ULANG per fold → prediksi out-of-fold (OOF)
→ skor per fold dengan metrik pilihan → ambang/bobot kelas dari OOF saja → model akhir di seluruh train →
skor di valid (bila ada split) dan prediksi test.

Semua acak memakai seed tetap (R14), jadi setelan yang sama memberi skor yang sama (buka ulang percobaan, E8).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import pipeline as pipes
from core import split as splitter
from core import steps as engine

ROOT = Path(__file__).resolve().parents[1]
# di luar repo (aturan repo: output ke ../../outputs/<kategori>/<nama>); uji memakai LAYA_OUTPUT
OUTPUT = Path(os.environ.get("LAYA_OUTPUT", ROOT.parents[1] / "outputs" / "laya_eda_v2"))

# ---------------------------------------------------------------- model (E5)

MODELS = {
    "histgb": dict(title="HistGradientBoosting (sklearn)", lib="sklearn",
                   imports=["from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor"],
                   kelas="HistGradientBoostingClassifier(max_iter={n}, learning_rate=0.08, random_state=0)",
                   angka="HistGradientBoostingRegressor(max_iter={n}, learning_rate=0.08, random_state=0)",
                   what="Gradient boosting bawaan sklearn, menerima kosong.", fits="Patokan tercepat, data ≤ 100 ribu baris.",
                   danger="Tanpa encoding kategori khusus: teks jadi kode angka (urutan abjad)."),
    "lightgbm": dict(title="LightGBM", lib="lightgbm", imports=["from lightgbm import LGBMClassifier, LGBMRegressor"],
                     kelas="LGBMClassifier(n_estimators={n}, learning_rate=0.05, random_state=0, n_jobs=2, verbose=-1)",
                     angka="LGBMRegressor(n_estimators={n}, learning_rate=0.05, random_state=0, n_jobs=2, verbose=-1)",
                     what="Gradient boosting berbasis histogram, cepat dan kuat di tabel.", fits="Pilihan pertama tabel.",
                     danger="Data kecil (< 1.000 baris) mudah overfit; turunkan num_leaves."),
    "xgboost": dict(title="XGBoost", lib="xgboost", imports=["from xgboost import XGBClassifier, XGBRegressor"],
                    kelas="XGBClassifier(n_estimators={n}, learning_rate=0.08, max_depth=6, tree_method=\"hist\", "
                          "random_state=0, n_jobs=2)",
                    angka="XGBRegressor(n_estimators={n}, learning_rate=0.08, max_depth=6, tree_method=\"hist\", "
                          "random_state=0, n_jobs=2)",
                    what="Gradient boosting dengan regularisasi.", fits="Pembanding LightGBM; sering bagus di ensemble.",
                    danger="Kelas harus lengkap di setiap fold latih (split kelompok/waktu bisa gagal)."),
    "catboost": dict(title="CatBoost", lib="catboost", imports=["from catboost import CatBoostClassifier, CatBoostRegressor"],
                     kelas="CatBoostClassifier(iterations={n}, learning_rate=0.08, random_seed=0, verbose=0, "
                           "thread_count=2, allow_writing_files=False)",
                     angka="CatBoostRegressor(iterations={n}, learning_rate=0.08, random_seed=0, verbose=0, "
                           "thread_count=2, allow_writing_files=False)",
                     what="Gradient boosting pohon simetris.", fits="Data dengan banyak kategori; stabil tanpa tuning.",
                     danger="Paling lambat dari keempatnya; di mode cepat iterasi dibatasi."),
    "autogluon": dict(title="AutoGluon TabularPredictor", lib="autogluon.tabular", imports=[], kelas="", angka="",
                      what="Ansambel banyak model otomatis (per fold, batas waktu).",
                      fits="Skor setinggi mungkin bila waktu cukup.",
                      danger="Lama dan berat (laptop panas); tidak bisa disimpan sebagai pipeline.joblib mandiri."),
}

# ---------------------------------------------------------------- mode waktu (E10)

MODES = {
    "cepat": dict(n=150, folds=3, budget=300, what="≤ 5 menit: 150 pohon, 3 fold, model berikut dilewati bila "
                                                   "perkiraan melewati anggaran."),
    "lengkap": dict(n=600, folds=5, budget=1800, what="600 pohon, 5 fold, anggaran 30 menit."),
}

# ---------------------------------------------------------------- kelas timpang (E12)

IMBALANCE = {
    "tanpa": dict(what="Model dilatih apa adanya.", fits="Kelas kurang lebih seimbang (rasio terkecil/terbesar > 0,3).",
                  danger="Kelas kecil jarang ditebak; macro-F1 rendah."),
    "bobot": dict(what="sample_weight = 'balanced' (kelas kecil berbobot besar), dihitung per fold latih.",
                  fits="Kelas timpang dan metrik memberi bobot sama per kelas (macro-F1).",
                  danger="Akurasi bisa turun: kelas besar lebih sering salah. Probabilitas tidak terkalibrasi."),
    "smote": dict(what="SMOTE (imbalanced-learn) membuat baris sintetis kelas kecil, HANYA di fold latih "
                       "(imblearn Pipeline: tidak dijalankan saat predict, jadi fold validasi dan test tidak disentuh).",
                  fits="Kelas sangat kecil dan fitur angka; dibandingkan dengan 'bobot' di papan skor.",
                  danger="Kosong diisi median dulu; baris sintetis bisa tidak masuk akal untuk kategori. "
                         "SMOTE sebelum split = bocor (di sini tidak terjadi)."),
    "ambang kelas": dict(what="Probabilitas tiap kelas dikali bobot yang dicari di prediksi OOF (bukan di test) "
                              "agar metrik pilihan maksimal.",
                         fits="Model sudah bagus tapi kelas kecil jarang terpilih.",
                         danger="Bobot dari OOF sedikit optimis; data kecil → bobot berisik."),
}

# ---------------------------------------------------------------- metrik (E7)

METRICS = {
    "accuracy": dict(kind="label", higher=True, title="akurasi"),
    "f1_macro": dict(kind="label", higher=True, title="macro-F1 (rata-rata F1 per kelas)"),
    "qwk": dict(kind="label", higher=True, title="QWK (quadratic weighted kappa, label berurutan)"),
    "logloss": dict(kind="proba", higher=False, title="log loss (probabilitas)"),
    "rmse": dict(kind="value", higher=False, title="RMSE"),
    "mae": dict(kind="value", higher=False, title="MAE"),
    "r2": dict(kind="value", higher=True, title="R²"),
}
TASKS = {"kelas": "klasifikasi", "berurutan": "label berurutan: regresi + ambang dari OOF", "angka": "regresi"}


def metric_value(metric: str, y: np.ndarray, label=None, proba=None, value=None) -> float:
    from sklearn import metrics as m

    if metric == "accuracy":
        return float(m.accuracy_score(y, label))
    if metric == "f1_macro":
        return float(m.f1_score(y, label, average="macro"))
    if metric == "qwk":
        return float(m.cohen_kappa_score(y, label, weights="quadratic"))
    if metric == "logloss":
        return float(m.log_loss(y, np.clip(proba, 1e-15, 1), labels=np.arange(proba.shape[1])))
    if metric == "rmse":
        return float(np.sqrt(m.mean_squared_error(y, value)))
    if metric == "mae":
        return float(m.mean_absolute_error(y, value))
    return float(m.r2_score(y, value))


def better(metric: str, a: float, b: float) -> bool:
    """a lebih baik dari b menurut arah metrik."""
    return a > b if METRICS[metric]["higher"] else a < b


def default_thresholds(k: int) -> np.ndarray:
    return np.arange(k - 1) + 0.5


def decide(task: str, output: np.ndarray, thresholds=None, weights=None) -> np.ndarray:
    """Keluaran model → kode kelas: argmax(proba × bobot) atau searchsorted(ambang, nilai)."""
    if task == "kelas":
        return np.argmax(output * (1 if weights is None else np.asarray(weights)), axis=1)
    return np.searchsorted(np.asarray(thresholds), output)


def score(metric: str, task: str, y: np.ndarray, output: np.ndarray, thresholds=None, weights=None) -> float:
    kind = METRICS[metric]["kind"]
    if kind == "proba":
        return metric_value(metric, y, proba=output)
    if kind == "value":
        return metric_value(metric, y, value=output)
    return metric_value(metric, y, label=decide(task, output, thresholds, weights))


def fit_thresholds(values: np.ndarray, y: np.ndarray, k: int, metric: str = "qwk", rounds: int = 3) -> list[float]:
    """Ambang pembulatan label berurutan, dicari HANYA di prediksi OOF (coordinate search per ambang)."""
    cuts = default_thresholds(k)
    best = score(metric, "berurutan", y, values, cuts)
    for _ in range(rounds):
        for i in range(len(cuts)):
            low = cuts[i - 1] if i else values.min() - 1
            high = cuts[i + 1] if i + 1 < len(cuts) else values.max() + 1
            for candidate in np.linspace(low, high, 42)[1:-1]:
                trial = cuts.copy()
                trial[i] = candidate
                value = score(metric, "berurutan", y, values, trial)
                if better(metric, value, best):
                    best, cuts = value, trial
    return [float(c) for c in cuts]


def fit_class_weights(proba: np.ndarray, y: np.ndarray, metric: str, rounds: int = 2) -> list[float]:
    """Bobot per kelas untuk argmax(proba × bobot), dicari HANYA di OOF (E12 'ambang kelas')."""
    weights = np.ones(proba.shape[1])
    best = score(metric, "kelas", y, proba, weights=weights)
    grid = np.geomspace(0.25, 6, 25)
    for _ in range(rounds):
        for k in range(len(weights)):
            for candidate in grid:
                trial = weights.copy()
                trial[k] = candidate
                value = score(metric, "kelas", y, proba, weights=trial)
                if better(metric, value, best):
                    best, weights = value, trial
    return [float(w) for w in weights]


def imbalance_report(y: pd.Series) -> dict:
    """Rasio kelas terkecil/terbesar (E12). Data riil: kelas 1 = 125 vs kelas 4 = 6.617 → 0,019."""
    counts = y.value_counts()
    ratio = float(counts.min() / counts.max()) if len(counts) else 1.0
    return {"jumlah": counts.sort_index(), "rasio": ratio, "timpang": ratio < 0.3,
            "terkecil": counts.idxmin(), "terbesar": counts.idxmax()}


# ---------------------------------------------------------------- validasi silang (E3)

FOLDS = {"stratified": "StratifiedKFold: proporsi kelas sama di tiap fold (klasifikasi).",
         "kfold": "KFold acak.", "group": "GroupKFold: satu kelompok selalu di satu fold.",
         "time": "TimeSeriesSplit: fold validasi selalu SESUDAH fold latih (urut kolom waktu)."}


def folds(kind: str, n_splits: int, seed: int = 0):
    from sklearn import model_selection as ms

    if kind == "stratified":
        return ms.StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    if kind == "group":
        return ms.GroupKFold(n_splits)
    if kind == "time":
        return ms.TimeSeriesSplit(n_splits)
    return ms.KFold(n_splits, shuffle=True, random_state=seed)


def model_code(name: str, task: str, n: int) -> str:
    return MODELS[name]["kelas" if task == "kelas" else "angka"].format(n=n)


def sampler_code(setelan: dict, y: np.ndarray) -> str | None:
    if setelan.get("timpang") != "smote":
        return None
    smallest = int(np.bincount(y.astype(int)).min()) if len(y) else 6
    return f"SMOTE(random_state=0, k_neighbors={max(1, min(5, smallest - 1))})"


def cross_validate(setelan: dict, X: pd.DataFrame, y: np.ndarray, groups=None, keep_models: bool = True) -> dict:
    """Pipeline di-fit ULANG di setiap fold (preprocessing tidak bocor ke fold validasi). OOF disimpan."""
    from sklearn.utils.class_weight import compute_sample_weight

    task, metric = setelan["tugas"], setelan["metrik"]
    k = int(setelan.get("n_kelas", 0))
    sampler = sampler_code(setelan, y)
    imports = MODELS[setelan["model"]]["imports"] + (["from imblearn.over_sampling import SMOTE"] if sampler else [])
    code = model_code(setelan["model"], task, int(setelan["n_estimators"]))
    splits = list(folds(setelan["fold"], int(setelan["n_fold"]), int(setelan.get("seed", 0))).split(X, y, groups))
    oof = np.full((len(X), k), np.nan) if task == "kelas" else np.full(len(X), np.nan)
    fold_scores, models, started = [], [], time.time()
    for fit_rows, val_rows in splits:
        pipe = pipes.build(setelan["langkah"], code, imports, sampler)
        params = {}
        if setelan.get("timpang") == "bobot" and task != "angka":
            params["model__sample_weight"] = compute_sample_weight("balanced", y[fit_rows])
        pipe.fit(X.iloc[fit_rows], y[fit_rows], **params)
        if task == "kelas":
            part = np.zeros((len(val_rows), k))
            part[:, pipe.classes_.astype(int)] = pipe.predict_proba(X.iloc[val_rows])
        else:
            part = pipe.predict(X.iloc[val_rows])
        oof[val_rows] = part
        fold_scores.append(score(metric, task, y[val_rows], part, default_thresholds(k)))
        if keep_models:
            models.append((pipe, val_rows))
    done = ~np.isnan(oof if oof.ndim == 1 else oof[:, 0])     # TimeSeriesSplit: blok pertama tanpa OOF
    return {"oof": oof, "baris_oof": done, "skor_fold": fold_scores, "skor_rata": float(np.mean(fold_scores)),
            "lama": time.time() - started, "model_fold": models, "kode_model": code, "imports": imports,
            "sampler": sampler}


# ---------------------------------------------------------------- satu percobaan

def prepare(setelan: dict, train: pd.DataFrame, test: pd.DataFrame | None) -> dict:
    """Split proyek → blok EDA (langkah pandas) → X, y berlabel. Langkah pipeline belum dijalankan."""
    label = setelan["label"]
    part, valid = splitter.apply(setelan.get("split"), label, train)
    eda = pipes.plan(setelan["langkah"])["eda"]
    result = engine.run(eda, label, part, engine.align(train, test, label), valid=valid)
    if result["error"]:
        raise RuntimeError(f"langkah {result['error']['langkah']} gagal: {result['error']['galat']}")
    data = result["train"][result["train"][label].notna()]
    columns = pipes.features(data, label, setelan.get("kecuali", []))
    if setelan["tugas"] == "angka":
        classes, y = None, data[label].to_numpy(dtype=float)
    else:
        classes = np.sort(data[label].unique())
        y = np.searchsorted(classes, data[label].to_numpy())
    valid = result["valid"][result["valid"][label].notna()] if label in result["valid"] else result["valid"].iloc[:0]
    return {"X": data[columns], "y": y, "kelas": classes, "fitur": columns, "valid": valid,
            "test": result["test"], "baris_tanpa_label": int(result["train"][label].isna().sum()),
            "grup": data[setelan["kolom_grup"]].astype(str).to_numpy() if setelan.get("kolom_grup") else None}


def experiment(setelan: dict, train: pd.DataFrame, test: pd.DataFrame | None = None, final: bool = True) -> dict:
    """Satu model: CV + ambang/bobot dari OOF + (opsional) model akhir di seluruh train, skor valid, prediksi test."""
    setelan = {**setelan, "langkah": engine.active(setelan["langkah"])}
    data = prepare(setelan, train, test)
    if setelan["tugas"] != "angka":
        setelan["n_kelas"] = len(data["kelas"])
    if setelan["model"] == "autogluon":
        return autogluon_experiment(setelan, data)
    X, y, task, metric = data["X"], data["y"], setelan["tugas"], setelan["metrik"]
    if setelan["fold"] == "time":
        order = np.argsort(X[setelan["kolom_waktu"]].to_numpy(), kind="stable")
        X, y = X.iloc[order], y[order]
    cv = cross_validate(setelan, X, y, data["grup"])
    rows = cv["baris_oof"]
    oof, y_oof = cv["oof"][rows], y[rows]
    thresholds = weights = None
    if task == "berurutan":
        thresholds = fit_thresholds(oof, y_oof, setelan["n_kelas"], metric if METRICS[metric]["kind"] == "label" else "qwk")
    if task == "kelas" and setelan.get("timpang") == "ambang kelas" and METRICS[metric]["kind"] == "label":
        weights = fit_class_weights(oof, y_oof, metric)
    result = {"setelan": setelan, "cv": cv, "y": y, "X": X, "kelas": data["kelas"], "fitur": data["fitur"],
              "ambang": thresholds, "bobot_kelas": weights,
              "skor_oof": score(metric, task, y_oof, oof, thresholds if thresholds else
                                default_thresholds(setelan.get("n_kelas", 0)), weights),
              "baris_tanpa_label": data["baris_tanpa_label"], "valid": data["valid"], "test": data["test"]}
    if final:
        result.update(final_model(result))
    return result


def final_model(result: dict) -> dict:
    """Pipeline akhir di SELURUH train berlabel → skor di valid (bila ada) dan prediksi test."""
    from sklearn.utils.class_weight import compute_sample_weight

    setelan, cv, X, y = result["setelan"], result["cv"], result["X"], result["y"]
    pipe = pipes.build(setelan["langkah"], cv["kode_model"], cv["imports"], cv["sampler"])
    params = {}
    if setelan.get("timpang") == "bobot" and setelan["tugas"] != "angka":
        params["model__sample_weight"] = compute_sample_weight("balanced", y)
    pipe.fit(X, y, **params)
    out = {"pipe": pipe, "skor_valid": None, "prediksi_test": None}
    valid, test = result["valid"], result["test"]
    if len(valid):
        y_valid = valid[setelan["label"]].to_numpy()
        if result["kelas"] is not None:
            y_valid = np.searchsorted(result["kelas"], y_valid)
        out["skor_valid"] = score(setelan["metrik"], setelan["tugas"], y_valid, raw_output(result, pipe, valid),
                                  result["ambang"] or default_thresholds(setelan.get("n_kelas", 0)), result["bobot_kelas"])
    if len(test):
        out["prediksi_test"] = predict(result, pipe, test)
    return out


def raw_output(result: dict, pipe, data: pd.DataFrame) -> np.ndarray:
    X = data.reindex(columns=result["fitur"])            # kolom hilang di test → kosong, urutan = train (R12)
    if result["setelan"]["tugas"] == "kelas":
        proba = np.zeros((len(X), result["setelan"]["n_kelas"]))
        proba[:, pipe.classes_.astype(int)] = pipe.predict_proba(X)
        return proba
    return pipe.predict(X)


def predict(result: dict, pipe, data: pd.DataFrame) -> np.ndarray:
    """Prediksi akhir dalam satuan label asli (kelas asli, bukan kode)."""
    output = raw_output(result, pipe, data)
    if result["setelan"]["tugas"] == "angka":
        return output
    codes = decide(result["setelan"]["tugas"], output, result["ambang"] or
                   default_thresholds(result["setelan"]["n_kelas"]), result["bobot_kelas"])
    return result["kelas"][np.clip(codes, 0, len(result["kelas"]) - 1)]


def autogluon_experiment(setelan: dict, data: dict) -> dict:
    """AutoGluon per fold (fold dan metrik sama dengan model lain). Tidak menghasilkan pipeline.joblib."""
    import importlib.util

    if importlib.util.find_spec("autogluon") is None:
        raise RuntimeError("autogluon.tabular belum terpasang (requirements.txt); model ini dilewati")
    from autogluon.tabular import TabularPredictor

    X, y, task = data["X"], data["y"], setelan["tugas"]
    k = setelan.get("n_kelas", 0)
    oof = np.full((len(X), k), np.nan) if task == "kelas" else np.full(len(X), np.nan)
    fold_scores, started = [], time.time()
    seconds = max(30, int(setelan.get("anggaran", 300)) // int(setelan["n_fold"]))
    for fit_rows, val_rows in folds(setelan["fold"], int(setelan["n_fold"]), int(setelan.get("seed", 0))).split(X, y, data["grup"]):
        frame = X.iloc[fit_rows].assign(_y=y[fit_rows])
        problem = "multiclass" if task == "kelas" else "regression"
        model = TabularPredictor("_y", problem_type=problem, verbosity=0).fit(frame, time_limit=seconds,
                                                                             presets="medium_quality")
        part = (model.predict_proba(X.iloc[val_rows]).reindex(columns=range(k), fill_value=0).to_numpy()
                if task == "kelas" else model.predict(X.iloc[val_rows]).to_numpy())
        oof[val_rows] = part
        fold_scores.append(score(setelan["metrik"], task, y[val_rows], part, default_thresholds(k)))
    cv = {"oof": oof, "baris_oof": ~np.isnan(oof if oof.ndim == 1 else oof[:, 0]), "skor_fold": fold_scores,
          "skor_rata": float(np.mean(fold_scores)), "lama": time.time() - started, "model_fold": []}
    rows = cv["baris_oof"]
    return {"setelan": setelan, "cv": cv, "y": y, "X": X, "kelas": data["kelas"], "fitur": data["fitur"],
            "ambang": None, "bobot_kelas": None, "pipe": None, "skor_valid": None, "prediksi_test": None,
            "skor_oof": score(setelan["metrik"], task, y[rows], oof[rows], default_thresholds(k)),
            "baris_tanpa_label": data["baris_tanpa_label"], "valid": data["valid"], "test": data["test"]}


# ---------------------------------------------------------------- papan skor + anggaran (E5, E10)

def estimate_seconds(setelan: dict, train: pd.DataFrame, models: list[str], sample: int = 1500) -> dict:
    """Perkiraan lama SEBELUM jalan: satu fit di ≤ 1.500 baris, diskalakan linear ke baris × (fold + 1)."""
    data = prepare({**setelan, "langkah": engine.active(setelan["langkah"])}, train, None)
    X, y = data["X"], data["y"]
    rows = np.random.default_rng(0).choice(len(X), min(sample, len(X)), replace=False)
    task, out = setelan["tugas"], {}
    for name in models:
        if name == "autogluon":
            out[name] = float(setelan.get("anggaran", 300))
            continue
        pipe = pipes.build(engine.active(setelan["langkah"]), model_code(name, task, int(setelan["n_estimators"])),
                           MODELS[name]["imports"])
        start = time.time()
        pipe.fit(X.iloc[rows], y[rows])
        out[name] = (time.time() - start) * len(X) / len(rows) * (int(setelan["n_fold"]) + 1)
    return out


def leaderboard(setelan: dict, train: pd.DataFrame, test: pd.DataFrame | None, models: list[str],
                budget: float | None = None, log_folder: Path | None = OUTPUT) -> dict:
    """Latih semua model di pipeline + fold + metrik yang sama. Model yang perkiraannya melewati sisa anggaran
    dilewati (E10). Setiap model yang selesai menambah satu baris log percobaan (E8)."""
    import importlib.util

    started, results, board = time.time(), {}, []
    for name in models:
        left = None if budget is None else budget - (time.time() - started)
        row = {"model": MODELS[name]["title"], "kunci": name}
        if importlib.util.find_spec(MODELS[name]["lib"].split(".")[0]) is None:
            board.append({**row, "status": f"dilewati: {MODELS[name]['lib']} belum terpasang"})
            continue
        if left is not None and results:
            spent = max(r["cv"]["lama"] for r in results.values())
            if spent * 1.3 > left:
                board.append({**row, "status": f"dilewati: anggaran waktu (sisa {left:.0f} s)"})
                continue
        try:
            result = experiment({**setelan, "model": name, "anggaran": left or setelan.get("anggaran", 300)}, train, test)
        except Exception as error:   # model gagal (mis. kelas hilang di fold): laporkan, lanjut model berikut
            board.append({**row, "status": f"galat: {type(error).__name__}: {error}"})
            continue
        results[name] = result
        board.append({**row, "status": "selesai", "skor CV": result["cv"]["skor_rata"], "skor OOF": result["skor_oof"],
                      "skor valid": result["skor_valid"], "lama (s)": round(result["cv"]["lama"], 1)})
        if log_folder is not None:
            log_experiment(log_folder, result)
    table = pd.DataFrame(board)
    if "skor CV" in table:
        table = table.sort_values("skor CV", ascending=not METRICS[setelan["metrik"]]["higher"], na_position="last")
    return {"papan": table.reset_index(drop=True), "hasil": results, "lama": time.time() - started}


# ---------------------------------------------------------------- log percobaan (E8)

def steps_hash(setelan: dict) -> str:
    """Hash langkah EDA aktif + split + label: langkah beda → hash beda."""
    key = json.dumps([engine.active(setelan["langkah"]), setelan.get("split"), setelan["label"]],
                     sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


SETTING_KEYS = ["label", "langkah", "split", "tugas", "metrik", "model", "n_estimators", "fold", "n_fold", "seed",
                "timpang", "kecuali", "kolom_grup", "kolom_waktu", "mode"]


def log_experiment(folder: Path, result: dict) -> dict:
    setelan = result["setelan"]
    row = {"waktu": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "hash_langkah": steps_hash(setelan),
           "model": setelan["model"], "metrik": setelan["metrik"],
           "parameter": json.dumps({"n_estimators": setelan["n_estimators"], "fold": setelan["fold"],
                                    "n_fold": setelan["n_fold"], "timpang": setelan.get("timpang", "tanpa")}),
           "skor_fold": json.dumps([round(s, 6) for s in result["cv"]["skor_fold"]]),
           "skor_rata": round(result["cv"]["skor_rata"], 6), "lama_detik": round(result["cv"]["lama"], 2),
           "setelan": json.dumps({k: setelan.get(k) for k in SETTING_KEYS}, ensure_ascii=False, default=str)}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "percobaan.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return row


def read_log(folder: Path = OUTPUT) -> pd.DataFrame:
    path = Path(folder) / "percobaan.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def reopen(row: pd.Series | dict) -> dict:
    """Setelan percobaan dari satu baris log → bisa dijalankan ulang dengan `experiment`."""
    return json.loads(row["setelan"])


__all__ = ["MODELS", "MODES", "IMBALANCE", "METRICS", "TASKS", "FOLDS", "metric_value", "score", "fit_thresholds",
           "fit_class_weights", "imbalance_report", "folds", "cross_validate", "prepare", "experiment", "predict",
           "estimate_seconds", "leaderboard", "steps_hash", "log_experiment", "read_log", "reopen", "OUTPUT"]
