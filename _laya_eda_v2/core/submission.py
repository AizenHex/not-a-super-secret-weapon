"""Fitur extra E6: tab Submission. Baca `sample_submission.csv` → format (kolom id & target, urutan id, tipe
target), tulis TEMPLATE KODE lurus (tanpa def) yang: baca data → split + blok EDA yang sama dengan Kode EDA →
muat pipeline.joblib (E2) → cek skema test (E4, mode lapor) → prediksi → `pd.DataFrame()` baru, kolom ditambah
satu per satu mengikuti sample → urutan id = sample → tipe target = sample → validator → simpan.

`validate` (dipakai aplikasi dan uji) dan blok validator di template memeriksa hal yang sama: jumlah baris, nama &
urutan kolom, himpunan & urutan id, tanpa kosong, kelas ⊆ kelas train, target bilangan bulat bila sample bulat.
"""
from __future__ import annotations

import json

import pandas as pd

from core import pipeline as pipes
from core import schema as schemas
from core import steps as engine
from core.training import text_types


def read_sample(sample: pd.DataFrame) -> dict:
    """Kolom pertama = id; kolom kedua = target (kolom target lain hanya dilaporkan)."""
    target = sample.columns[1] if sample.shape[1] > 1 else None
    integer = bool(target is not None and pd.api.types.is_integer_dtype(sample[target]))
    return {"id": sample.columns[0], "target": target, "target_lain": list(sample.columns[2:]),
            "baris": len(sample), "bulat": integer, "kolom": list(sample.columns)}


def validate(submission: pd.DataFrame, sample: pd.DataFrame, classes=None, text: str | None = None) -> list[str]:
    """Daftar masalah (kosong = lolos). `text` = isi file CSV bila ingin memeriksa tulisan '3.0'."""
    spec, problems = read_sample(sample), []
    ident, target = spec["id"], spec["target"]
    if list(submission.columns) != spec["kolom"]:
        problems.append(f"kolom {list(submission.columns)} != sample {spec['kolom']}")
    if len(submission) != len(sample):
        problems.append(f"jumlah baris {len(submission)} != sample {len(sample)}")
    if ident in submission:
        if set(submission[ident]) != set(sample[ident]):
            problems.append(f"himpunan id beda: {len(set(sample[ident]) - set(submission[ident]))} id hilang, "
                            f"{len(set(submission[ident]) - set(sample[ident]))} id asing")
        elif not submission[ident].reset_index(drop=True).equals(sample[ident].reset_index(drop=True)):
            problems.append("urutan id beda dengan sample")
    if target in submission:
        values = submission[target]
        if values.isna().any():
            problems.append(f"{int(values.isna().sum())} target kosong (NaN)")
        if spec["bulat"] and not pd.api.types.is_integer_dtype(values):
            problems.append(f"target bertipe {values.dtype}, sample bilangan bulat (tulisan '3.0' ditolak)")
        if classes is not None and not set(values.dropna()) <= set(classes):
            problems.append(f"kelas di luar kelas train: {sorted(set(values.dropna()) - set(classes), key=str)[:5]}")
    if text is not None and spec["bulat"] and target is not None:
        cells = pd.Series(text.splitlines()[1:]).str.split(",").str[spec["kolom"].index(target)]
        if cells.str.contains(r"\.", regex=True).any():
            problems.append("file berisi target desimal (mis. '3.0')")
    return problems


VALIDATOR = '''# validator: format harus sama persis dengan sample_submission.csv
masalah = []
if list(submission.columns) != list(sample.columns):
    masalah.append("kolom beda dengan sample")
if len(submission) != len(sample):
    masalah.append("jumlah baris beda dengan sample")
if set(submission[ID]) != set(sample[ID]):
    masalah.append("himpunan id beda dengan sample")
elif not submission[ID].equals(sample[ID]):
    masalah.append("urutan id beda dengan sample")
if submission[TARGET].isna().any():
    masalah.append("ada target kosong")
if not set(submission[TARGET].dropna()) <= set(KELAS_TRAIN):
    masalah.append("ada kelas yang tidak ada di train")
if TARGET_BULAT and not pd.api.types.is_integer_dtype(submission[TARGET]):
    masalah.append("target bukan bilangan bulat")
if masalah:
    raise ValueError("submission TIDAK lolos: " + "; ".join(masalah))
print("validator: lolos")'''


def decision_code(info: dict) -> list[str]:
    if info["tugas"] == "kelas":
        lines = ["proba = pipe.predict_proba(X_test)"]
        if info.get("bobot_kelas"):
            lines += ['bobot = np.array(info["bobot_kelas"])[pipe.classes_.astype(int)]   # bobot kelas dari OOF (E12)',
                      "kode = pipe.classes_[np.argmax(proba * bobot, axis=1)].astype(int)"]
        else:
            lines.append("kode = pipe.classes_[np.argmax(proba, axis=1)].astype(int)")
        return lines + ['prediksi = np.array(info["kelas"])[kode]']
    if info["tugas"] == "berurutan":
        return ['kode = np.searchsorted(info["ambang"], pipe.predict(X_test))   # ambang dari OOF (E7)',
                'kode = np.clip(kode, 0, len(info["kelas"]) - 1)',
                'prediksi = np.array(info["kelas"])[kode]']
    return ["prediksi = pipe.predict(X_test)"]


def template(project: dict, info: dict, spec: dict, paths: dict) -> str:
    """Satu file lurus (blok `# %%`). `info` = isi pipeline.joblib.json; `spec` = read_sample(sample)."""
    eda = pipes.plan(info["langkah"])["eda"]
    p = {"label": project["label"], "steps": eda, "split": project.get("split")}
    blocks = engine.script_blocks(p, paths.get("train", "train.csv"), paths.get("test", "test.csv"))
    integer = "astype(\"int64\")" if spec["bulat"] else "to_numpy()"
    blocks.insert(2, "# %% id test disimpan SEBELUM langkah (baris test tidak pernah dibuang)\n"
                     f"SAMPLE_CSV = {paths.get('sample', 'sample_submission.csv')!r}\n"
                     f"ID, TARGET = {spec['id']!r}, {spec['target']!r}\n"
                     "sample = pd.read_csv(SAMPLE_CSV)\n"
                     "ID_TEST = test[ID].to_numpy()")
    blocks += [
        "# %% muat pipeline (E2): tanpa latih ulang\n"
        "import json\nfrom pathlib import Path\nimport joblib\nimport sklearn\n"
        f"PIPELINE = {paths.get('pipeline', 'pipeline.joblib')!r}\n"
        "pipe = joblib.load(PIPELINE)\n"
        'info = json.loads(Path(PIPELINE + ".json").read_text(encoding="utf-8"))\n'
        'print("sklearn saat simpan:", info["sklearn"], "| sekarang:", sklearn.__version__)\n'
        'X_test = test.reindex(columns=info["fitur"])   # kolom hilang -> kosong, urutan = train\n'
        'X_test = X_test.astype({kolom: object for kolom in info["kolom_teks"]})   # kolom teks tetap teks',
        "# %% cek skema test (pandera, mode lapor: prediksi tetap jalan)\n"
        + schemas.check_code(info.get("skema") or {}, "X_test"),
        "# %% prediksi -> submission (kolom ditambah satu per satu mengikuti sample)\n"
        + "\n".join(decision_code(info)) + "\n"
        "hasil = pd.Series(prediksi, index=ID_TEST)\n"
        "submission = pd.DataFrame()\n"
        "submission[ID] = sample[ID]\n"
        "submission[TARGET] = hasil.reindex(sample[ID]).to_numpy()   # urutan id = sample\n"
        "if submission[TARGET].notna().all():\n"
        f"    submission[TARGET] = submission[TARGET].{integer}\n"
        f"KELAS_TRAIN = {json.dumps(info.get('kelas'))}\n".replace("null", "None")
        + f"TARGET_BULAT = {spec['bulat']}\n"
        + (f"KELAS_TRAIN = list(submission[TARGET].dropna())   # regresi: tanpa cek kelas\n"
           if info.get("kelas") is None else "")
        + VALIDATOR + "\n"
        'submission.to_csv("submission.csv", index=False)\n'
        'print("submission.csv tersimpan:", submission.shape)',
    ]
    return "\n\n\n".join(blocks) + "\n"


def info_for(result: dict, project_label: str) -> dict:
    """Isi pipeline.joblib.json dari hasil training.experiment: semua yang dibutuhkan template."""
    classes = result["kelas"]
    setelan = result["setelan"]
    return {"label": project_label, "tugas": setelan["tugas"], "metrik": setelan["metrik"], "model": setelan["model"],
            "kelas": None if classes is None else classes.tolist(), "ambang": result["ambang"],
            "kolom_teks": list(text_types(result["X"])),
            "bobot_kelas": result["bobot_kelas"], "fitur": result["fitur"], "langkah": setelan["langkah"],
            "split": setelan.get("split"), "skema": schemas.learn(result["X"]),
            "skor_cv": result["cv"]["skor_rata"]}


__all__ = ["read_sample", "validate", "template", "info_for", "decision_code", "VALIDATOR"]
