"""Uji fitur extra E1–E12 (VISI.md bagian 0, kolom "Cara memastikan").

Jalankan dari folder _laya_eda_v2:  python -m tests.test_extra   (atau satu uji: python -m tests.test_extra test_e6_...)
Data: sintetis (tests/sintetis.py) kecuali uji bernama *_data_riil (dilewati bila data riil tidak ada).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tests import bantu, sintetis

os.environ.setdefault("LAYA_OUTPUT", tempfile.mkdtemp(prefix="laya_output_"))
warnings.filterwarnings("ignore", category=UserWarning)

from core import diagnose, ensemble, schema, steps, submission, training  # noqa: E402
from core import pipeline as pipes  # noqa: E402

LABEL = bantu.LABEL
SPLIT = {"method": "acak", "share": 0.2, "seed": 42, "stratify": True}


def data(n: int = 3000):
    train = sintetis.buat(n)
    return train, sintetis.buat_test(train)


def pipe_steps() -> list[dict]:
    """Blok EDA pandas (2 langkah) lalu 12 langkah yang punya transformer sklearn/feature_engine."""
    p = steps.new_project(LABEL)
    add = lambda op, **k: steps.add_step(p, op, k)  # noqa: E731 (uji, bukan kode hasil)
    add("tidy_text", columns=["Kota", "Pendidikan Orang Tua/Wali"])
    add("sentinel_to_nan", columns=["NJOP per Meter", "Luas Tanah Rumah (m2)"])
    add("impute", columns=["NJOP per Meter", "Luas Tanah Rumah (m2)"], strategy="median")
    add("scale", columns=["Luas Tanah Rumah (m2)"], method="standard")
    add("log_transform", columns=["NJOP per Meter"])
    add("ordinal", column="Pendapatan Keluarga", order=sintetis.INCOME)
    add("ordinal", column="Jalur Masuk", order=[])
    add("fill_constant", columns=["Status Kepesertaan Program Bantuan Pemerintah"], value="Tidak")
    add("outlier", column="Daya Listrik", method="manual", action="clip", low=None, high=10000, factor=0)
    add("rare_label", columns=["Pekerjaan Orang Tua/Wali"], tol=0.05)
    add("target_encode", columns=["Kota"], kind="kelas")
    add("onehot_sklearn", columns=["Jenis Kelamin", "Fakultas"], min_share=0.01)
    add("winsorize", columns=["Pengeluaran Bulanan Keluarga"], method="iqr", fold=3.0)
    add("power", columns=["Total Hutang"])
    return p["steps"]


def setelan(**extra) -> dict:
    base = dict(label=LABEL, langkah=pipe_steps(), split=SPLIT, tugas="kelas", metrik="f1_macro", model="lightgbm",
                n_estimators=60, fold="stratified", n_fold=3, seed=0, timpang="tanpa", kecuali=["id"])
    return {**base, **extra}


# ---------------------------------------------------------------- E1

def test_e1_pipeline_sama_dengan_kode_eda():
    train, test = data()
    train = train[train[LABEL].notna()].reset_index(drop=True)   # pipeline belajar dari baris berlabel saja
    chosen = pipe_steps()
    plan = pipes.plan(chosen)
    assert len(plan["eda"]) == 2 and len(plan["pipe"]) == 12, (len(plan["eda"]), len(plan["pipe"]))
    test = steps.align(train, test, LABEL)
    eda = steps.run(chosen, LABEL, train, test)                  # = blok kode EDA (exec, R5)
    pre = steps.run(plan["eda"], LABEL, train, test)
    assert eda["error"] is None and pre["error"] is None
    imports, code = training.MODELS["lightgbm"]["imports"], training.model_code("lightgbm", "kelas", 60)
    pipe = pipes.build(chosen, code, imports)
    X, y = pre["train"].drop(columns=[LABEL, "id"]), pre["train"][LABEL].astype(int).to_numpy()
    pipe.fit(X, y)
    order = list(pipe[:-2].get_feature_names_out())
    from_pipe = pipe[:-2].transform(pre["test"].drop(columns=["id"]))
    from_code = eda["test"].drop(columns=["id"])[order]
    numbers = [c for c in order if pd.api.types.is_numeric_dtype(from_code[c])]
    assert np.allclose(from_pipe[numbers].to_numpy(float), from_code[numbers].to_numpy(float), equal_nan=True)
    assert all(from_pipe[c].equals(from_code[c]) for c in order if c not in numbers)
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    same_model = Pipeline([("teks_sisa", clone(pipe.named_steps["teks_sisa"])), ("model", clone(pipe.named_steps["model"]))])
    same_model.fit(eda["train"].drop(columns=[LABEL, "id"])[order], y)
    assert (same_model.predict(from_code) == pipe.predict(pre["test"].drop(columns=["id"]))).all()
    code_text = pipes.pipeline_code(chosen, code, imports)
    import ast
    assert not [n for n in ast.walk(ast.parse(code_text)) if isinstance(n, (ast.FunctionDef, ast.Lambda, ast.ClassDef))]


# ---------------------------------------------------------------- E2

def test_e2_joblib_proses_baru(tmp_path):
    train, test = data()
    result = training.experiment(setelan(), train, test)
    path = tmp_path / "pipeline.joblib"
    meta = pipes.save(result["pipe"], path, submission.info_for(result, LABEL))
    import sklearn
    assert meta["sklearn"] == sklearn.__version__ and json.loads(Path(f"{path}.json").read_text())["sklearn"]
    X_test = result["test"].reindex(columns=result["fitur"])
    X_test.to_pickle(tmp_path / "x.pkl")
    child = ("import joblib, pandas as pd, numpy as np\n"
             f"pipe = joblib.load({str(path)!r})\n"
             f"np.save({str(tmp_path / 'p.npy')!r}, pipe.predict_proba(pd.read_pickle({str(tmp_path / 'x.pkl')!r})))\n")
    subprocess.run([sys.executable, "-c", child], check=True, cwd=bantu.ROOT)
    assert np.array_equal(np.load(tmp_path / "p.npy"), result["pipe"].predict_proba(X_test))


# ---------------------------------------------------------------- E3

def test_e3_cv_tanpa_bocor():
    """Data sengaja bocor: kode unik ±2 baris/kode, label acak. Target encoding di-fit di seluruh train lalu
    di-CV = skor palsu tinggi; di dalam pipeline (fit ulang per fold) = setara tebak acak."""
    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame({"kode": [f"k{i}" for i in rng.integers(0, n // 2, n)], "acak": rng.normal(size=n)})
    y = rng.integers(0, 2, n)
    base = dict(label="y", tugas="kelas", metrik="accuracy", model="histgb", n_estimators=50, fold="stratified",
                n_fold=5, seed=0, n_kelas=2, timpang="tanpa")
    inside = training.cross_validate({**base, "langkah": [{"op": "target_encode", "params": {"columns": ["kode"],
                                                                "kind": "angka"}, "on": True}]}, X, y)
    from sklearn.preprocessing import TargetEncoder
    leaky_encoder = TargetEncoder(target_type="continuous").set_output(transform="pandas").fit(X[["kode"]], y)
    leaky = X.assign(kode=leaky_encoder.transform(X[["kode"]])["kode"])   # fit lalu transform = bocor
    outside = training.cross_validate({**base, "langkah": []}, leaky, y)
    print("akurasi CV: pipeline", round(inside["skor_rata"], 3), "| bocor", round(outside["skor_rata"], 3))
    assert inside["skor_rata"] <= outside["skor_rata"] and outside["skor_rata"] > 0.7 and inside["skor_rata"] < 0.6
    assert inside["baris_oof"].all() and inside["oof"].shape == (n, 2)
    groups = X["kode"].to_numpy()
    for kind in ("kfold", "group", "time"):
        result = training.cross_validate({**base, "fold": kind, "langkah": []}, X[["acak"]], y, groups)
        assert len(result["skor_fold"]) == 5, kind
        if kind == "group":
            for pipe, val_rows in result["model_fold"]:
                assert not set(groups[val_rows]) & set(np.delete(groups, val_rows)), "kelompok terbelah"
    assert not result["baris_oof"].all(), "TimeSeriesSplit: blok pertama tidak punya OOF"


# ---------------------------------------------------------------- E4

def test_e4_skema_pandera():
    train, test = data()
    spec = schema.learn(train, LABEL, ["id"])
    messy = test.copy()
    messy.loc[messy.index[:7], "Pendapatan Keluarga"] = "Rp 999 juta"          # kategori baru
    messy["NJOP per Meter"] = messy["NJOP per Meter"].astype(object)
    messy.loc[messy.index[:4], "NJOP per Meter"] = "abc"                        # bukan angka
    messy.loc[messy.index[10:13], "Luas Tanah Rumah (m2)"] = 1e13              # di luar rentang train
    messy = messy.drop(columns=["Fakultas"])                                    # kolom hilang
    report = schema.validate(messy, spec, "lapor")
    print(report.to_string())
    by_column = report.groupby("kolom")["baris"].sum()
    assert by_column["Pendapatan Keluarga"] == 7 and "Rp 999 juta" in report["contoh nilai"].str.cat()
    assert by_column["NJOP per Meter"] >= 4 and by_column["Luas Tanah Rumah (m2)"] >= 3
    assert report["contoh nilai"].str.contains("Fakultas").any(), "kolom hilang dilaporkan"
    import pandera.pandas as pa
    try:
        schema.validate(messy, spec, "ketat")
        raise AssertionError("mode ketat harus berhenti")
    except pa.errors.SchemaErrors:
        pass
    result = training.experiment(setelan(), train, messy)                       # mode lapor: prediksi tetap ada
    assert len(result["prediksi_test"]) == len(messy)


# ---------------------------------------------------------------- E5 + E10

def test_e5_semua_baseline_masuk_papan():
    train, test = data()
    board = training.leaderboard(setelan(), train, test, list(training.MODELS), log_folder=None)
    table = board["papan"].set_index("kunci")
    print(table.to_string())
    for name in ("histgb", "lightgbm", "xgboost", "catboost"):
        assert table.loc[name, "status"] == "selesai", table.loc[name, "status"]
        assert np.isfinite(table.loc[name, "skor CV"])
    import importlib.util
    if importlib.util.find_spec("autogluon") is None:
        assert table.loc["autogluon", "status"].startswith("dilewati"), table.loc["autogluon", "status"]
        raise bantu.Lewati("4 model lulus; AutoGluon belum terpasang (autogluon.tabular 1.6.3) jadi baris AutoGluon "
                           "hanya 'dilewati', E5 belum lengkap")
    assert table.loc["autogluon", "status"] == "selesai"


def test_e5_e10_data_riil():
    bantu.perlu_data_riil()
    train = pd.read_csv(bantu.TRAIN_RIIL)
    start = time.time()
    board = training.leaderboard(setelan(n_estimators=training.MODES["cepat"]["n"], n_fold=3), train, None,
                                 list(training.MODELS), budget=training.MODES["cepat"]["budget"], log_folder=None)
    print(board["papan"].to_string())
    assert time.time() - start <= training.MODES["cepat"]["budget"] + 60


def test_e10_mode_cepat_dalam_batas():
    """Data sintetis seukuran data riil (30.380 baris): mode cepat ≤ 300 s; perkiraan tersedia sebelum jalan."""
    train = sintetis.buat(30380)
    mode = training.MODES["cepat"]
    chosen = setelan(n_estimators=mode["n"], n_fold=mode["folds"])
    guess = training.estimate_seconds(chosen, train, ["histgb", "lightgbm"])
    assert all(value > 0 for value in guess.values()), guess
    start = time.time()
    board = training.leaderboard(chosen, train, None, ["histgb", "lightgbm", "xgboost", "catboost"],
                                 budget=mode["budget"], log_folder=None)
    spent = time.time() - start
    print(board["papan"].to_string(), f"\nlama {spent:.0f} s, perkiraan {guess}")
    assert spent <= mode["budget"], spent


# ---------------------------------------------------------------- E6

def test_e6_template_mandiri_lolos_validator(tmp_path):
    paths = sintetis.tulis(tmp_path, 3000)
    train, test, sample = (pd.read_csv(paths[k]) for k in ("train", "test", "sample"))
    project = {"label": LABEL, "steps": pipe_steps(), "split": SPLIT}
    result = training.experiment(setelan(), train, test)
    info = submission.info_for(result, LABEL)
    pipes.save(result["pipe"], tmp_path / "pipeline.joblib", info)
    spec = submission.read_sample(sample)
    assert spec["id"] == "id" and spec["target"] == LABEL and spec["bulat"]
    code = submission.template(project, json.loads((tmp_path / "pipeline.joblib.json").read_text()), spec,
                               {"train": str(paths["train"]), "test": str(paths["test"]), "sample": str(paths["sample"]),
                                "pipeline": str(tmp_path / "pipeline.joblib")})
    import ast
    assert not [n for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.FunctionDef, ast.Lambda, ast.ClassDef))]
    (tmp_path / "buat_submission.py").write_text(code, encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    run = subprocess.run([sys.executable, "buat_submission.py"], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-3000:]
    print(run.stdout[-600:])
    made = pd.read_csv(tmp_path / "submission.csv")
    text = (tmp_path / "submission.csv").read_text()
    assert submission.validate(made, sample, result["kelas"], text) == []
    expected = pd.Series(result["prediksi_test"], index=test["id"]).reindex(sample["id"]).to_numpy()
    assert (made[LABEL].to_numpy() == expected).all(), "template = prediksi aplikasi"


def test_e6_validator_menolak_submission_rusak():
    train, test = data(500)
    sample = sintetis.sample_submission(test)
    good = sample.copy()
    good[LABEL] = 3
    assert submission.validate(good, sample, [0, 1, 2, 3]) == []
    broken = {
        "id hilang": good.iloc[1:],
        "urutan acak": good.sample(frac=1, random_state=0),
        "3.0": good.assign(**{LABEL: good[LABEL].astype(float)}),
        "NaN": good.assign(**{LABEL: good[LABEL].where(good.index > 0)}),
        "kelas asing": good.assign(**{LABEL: 9}),
    }
    for name, frame in broken.items():
        problems = submission.validate(frame, sample, [0, 1, 2, 3])
        print(name, "->", problems)
        assert problems, name
    assert submission.validate(good, sample, [0, 1, 2, 3], "id,Golongan UKT\n1,3.0\n"), "tulisan 3.0 di file"


def test_e6_data_riil():
    bantu.perlu_data_riil()
    raise bantu.Lewati("butuh test.csv + sample_submission.csv lomba 2024 (id,Golongan UKT) di samping data riil")


# ---------------------------------------------------------------- E7

def test_e7_label_berurutan_ambang_dari_oof():
    train, test = data()
    chosen = setelan(metrik="qwk", langkah=pipe_steps()[:2])
    ordinal = training.experiment({**chosen, "tugas": "berurutan"}, train, test)
    classes = training.experiment({**chosen, "tugas": "kelas"}, train, test)
    print("QWK valid: berurutan", round(ordinal["skor_valid"], 4), "| klasifikasi", round(classes["skor_valid"], 4))
    assert ordinal["skor_valid"] >= classes["skor_valid"]
    assert ordinal["skor_oof"] >= ordinal["cv"]["skor_rata"] - 0.02, "ambang tidak boleh memperburuk OOF"
    shuffled = test.sample(frac=1, random_state=3).reset_index(drop=True)
    shuffled["NJOP per Meter"] = shuffled["NJOP per Meter"].sample(frac=1, random_state=4).to_numpy()
    again = training.experiment({**chosen, "tugas": "berurutan"}, train, shuffled)
    assert again["ambang"] == ordinal["ambang"], "ambang hanya dari OOF train: test diubah, ambang sama"
    for metric in training.METRICS:
        task = "angka" if training.METRICS[metric]["kind"] == "value" else "kelas"
        result = training.cross_validate({**chosen, "tugas": task, "metrik": metric, "n_kelas": 8, "n_fold": 2},
                                         ordinal["X"], ordinal["y"].astype(float) if task == "angka" else ordinal["y"])
        assert np.isfinite(result["skor_rata"]), metric


# ---------------------------------------------------------------- E8

def test_e8_log_percobaan(tmp_path):
    train, test = data()
    chosen = setelan(langkah=pipe_steps()[:3], model="histgb")
    first = training.experiment(chosen, train, test, final=False)
    training.log_experiment(tmp_path, first)
    other = training.experiment({**chosen, "langkah": pipe_steps()[:4]}, train, test, final=False)
    training.log_experiment(tmp_path, other)
    log = training.read_log(tmp_path)
    assert len(log) == 2 and log["hash_langkah"].nunique() == 2, log
    again = training.experiment(training.reopen(log.iloc[0]), train, test, final=False)
    assert again["cv"]["skor_rata"] == log.iloc[0]["skor_rata"] or \
        abs(again["cv"]["skor_rata"] - log.iloc[0]["skor_rata"]) < 1e-6
    assert again["cv"]["skor_fold"] == first["cv"]["skor_fold"]


# ---------------------------------------------------------------- E9

def test_e9_ensemble_hanya_bila_lebih_baik():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    good = np.eye(3)[y] * 0.7 + 0.1
    bad = rng.dirichlet(np.ones(3), 600)
    choice = ensemble.choose({"bagus": good, "acak": bad}, y, "kelas", "accuracy")
    assert choice["pilihan"] == "bagus", choice
    weights = np.array(list(choice["bobot"].values()))
    assert (weights >= 0).all() and abs(weights.sum() - 1) < 1e-9
    noisy_a = np.eye(3)[np.where(rng.random(600) < 0.65, y, rng.integers(0, 3, 600))] * 0.6 + rng.dirichlet(np.ones(3), 600) * 0.4
    noisy_b = np.eye(3)[np.where(rng.random(600) < 0.65, y, rng.integers(0, 3, 600))] * 0.6 + rng.dirichlet(np.ones(3), 600) * 0.4
    mixed = ensemble.choose({"a": noisy_a, "b": noisy_b}, y, "kelas", "accuracy")
    print(mixed["pilihan"], mixed["skor"], mixed["skor_per_model"])
    assert mixed["pilihan"] in ("rata-rata berbobot", "stacking") and mixed["skor"] > max(mixed["skor_per_model"].values())
    import inspect
    assert list(inspect.signature(ensemble.choose).parameters)[:2] == ["oofs", "y"], "bobot hanya dari OOF + label train"


# ---------------------------------------------------------------- E11

def test_e11_figur_16_9_dan_baris_salah(tmp_path):
    train, test = data()
    result = training.experiment(setelan(), train, test)
    table = diagnose.confusion(result)
    assert table.to_numpy().sum() == result["cv"]["baris_oof"].sum()
    from PIL import Image
    for path in (diagnose.figure_confusion(table, tmp_path / "confusion.png"),
                 diagnose.figure_importance(diagnose.importance(result), tmp_path / "importance.png")):
        width, height = Image.open(path).size
        assert abs(width / height - 16 / 9) < 0.01, (path, width, height)
    worst = diagnose.worst_rows(result)
    assert set(worst["baris"]) <= set(train.index) and worst["salah"].is_monotonic_decreasing
    assert len(diagnose.most_confused(table)) > 0


# ---------------------------------------------------------------- E12

def test_e12_oversampling_hanya_di_fold_latih():
    from tests import pengintai

    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = np.where(rng.random(n) < 0.08, 1, 0)
    training.MODELS["pengintai"] = dict(training.MODELS["histgb"], imports=["from tests.pengintai import Pengintai"],
                                        kelas="Pengintai()")
    pengintai.CATATAN["fit"].clear(), pengintai.CATATAN["predict"].clear()
    try:
        result = training.cross_validate(dict(label="y", langkah=[], tugas="kelas", metrik="f1_macro", model="pengintai",
                                               n_estimators=1, fold="stratified", n_fold=4, seed=0, n_kelas=2,
                                               timpang="smote"), X, y)
    finally:
        del training.MODELS["pengintai"]
    for (pipe, val_rows), seen_fit, seen_predict in zip(result["model_fold"], pengintai.CATATAN["fit"],
                                                       pengintai.CATATAN["predict"]):
        fit_rows = np.setdiff1d(np.arange(n), val_rows)
        fit_set = {tuple(row) for row in seen_fit}
        assert len(seen_fit) > len(fit_rows), "SMOTE menambah baris di fold latih"
        assert {tuple(row) for row in X.to_numpy()[fit_rows]} <= fit_set
        assert not {tuple(row) for row in X.to_numpy()[val_rows]} & fit_set, "fold validasi tidak boleh ikut SMOTE"
        assert np.array_equal(seen_predict, X.to_numpy()[val_rows]), "predict = baris validasi asli saja"
    report = training.imbalance_report(sintetis.buat(3000)[LABEL])
    assert report["timpang"] and report["terkecil"] == 1
    for option in training.IMBALANCE.values():
        assert option["what"] and option["fits"] and option["danger"]


def test_e12_skor_per_kelas_sebelum_sesudah():
    train, test = data()
    before = diagnose.per_class(training.experiment(setelan(langkah=pipe_steps()[:3]), train, test, final=False))
    after = diagnose.per_class(training.experiment(setelan(langkah=pipe_steps()[:3], timpang="bobot"), train, test,
                                                   final=False))
    print(pd.concat({"tanpa": before.set_index("kelas"), "bobot": after.set_index("kelas")}, axis=1).to_string())
    assert len(before) == len(after) == 8


if __name__ == "__main__":
    bantu.main(globals())
