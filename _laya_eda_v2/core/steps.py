"""Mesin langkah (ala Data Wrangler): daftar langkah → satu skrip pandas LURUS → dijalankan blok per blok.

Proyek = {"label": str | None, "steps": [{"op": nama, "params": {...}, "on": bool}, ...]}.
Aplikasi menjalankan blok yang sama persis dengan isi file .py yang diunduh, jadi angka di aplikasi
= angka dari kode.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json

import numpy as np
import pandas as pd

from core import split
from core.ops import OPS, summary


def new_project(label: str | None = None) -> dict:
    return {"label": label, "steps": [], "split": None}


def add_step(project: dict, op: str, params: dict) -> None:
    project["steps"].append({"op": op, "params": copy.deepcopy(params), "on": True})


def active(steps: list[dict]) -> list[dict]:
    return [step for step in steps if step.get("on", True)]


def step_code(step: dict) -> str:
    return OPS[step["op"]]["code"](step["params"])


def imports(steps: list[dict]) -> str:
    lines = ["import numpy as np", "import pandas as pd"]
    used = {step["op"] for step in active(steps)}
    if "skrub_encoder" in used:
        lines.append("import skrub")
    if "autogluon_features" in used:
        lines.append("from autogluon.features.generators import AutoMLPipelineFeatureGenerator")
    return "\n".join(lines)


def namespace(steps: list[dict], label: str | None, train: pd.DataFrame, test: pd.DataFrame | None,
              valid: pd.DataFrame | None = None) -> dict:
    """Pengganti blok 'baca data' + 'split' saat dijalankan di aplikasi: train/valid/test diberikan langsung."""
    ns = {}
    exec(imports(steps), ns)
    ns["LABEL"] = label
    ns["train"] = train.copy()
    ns["valid"] = valid.copy() if valid is not None else train.iloc[:0].copy()
    ns["test"] = (test.copy() if test is not None
                  else train.iloc[:0].drop(columns=[label] if label else [], errors="ignore").copy())
    return ns


def run(steps: list[dict], label: str | None, train: pd.DataFrame, test: pd.DataFrame | None,
        first: int = 1, valid: pd.DataFrame | None = None) -> dict:
    """Jalankan langkah aktif satu per satu. Berhenti di langkah pertama yang galat (galat tidak disembunyikan).

    Kembalikan train, valid, test, dan log per langkah: baris/kolom sebelum-sesudah, keluaran print, galat.
    `first` = nomor langkah pertama di log (untuk menjalankan lanjutan dari hasil langkah sebelumnya).
    `valid` = bagian valid hasil split (None = kosong); diperlakukan seperti test yang masih punya label.
    """
    ns = namespace(steps, label, train, test, valid)
    log = []
    for number, step in enumerate(steps, first):
        if not step.get("on", True):
            continue
        before = ns["train"].shape, list(ns["train"].columns), ns["test"].shape, len(ns["valid"])
        missing = ns["train"].isna().sum()
        output = io.StringIO()
        entry = {"langkah": number, "operasi": summary(step)}
        try:
            with contextlib.redirect_stdout(output):
                exec(step_code(step), ns)
        except Exception as error:   # kode langkah buatan pengguna: laporkan apa adanya, berhenti di sini
            entry.update(galat=f"{type(error).__name__}: {error}", keluaran=output.getvalue())
            log.append(entry)
            break
        after = ns["train"].columns
        gained = ns["train"].isna().sum() - missing.reindex(after, fill_value=0)   # sel kosong yang DIBUAT langkah ini
        entry.update({
            "baris train": f"{before[0][0]:,} → {ns['train'].shape[0]:,}",
            "kolom": f"{before[0][1]} → {ns['train'].shape[1]}",
            "baris valid": f"{before[3]:,} → {len(ns['valid']):,}",
            "baris test": f"{before[2][0]:,} → {ns['test'].shape[0]:,}",
            "kolom baru": [c for c in after if c not in before[1]],
            "kolom hilang": [c for c in before[1] if c not in after],
            "kosong baru": {c: int(n) for c, n in gained[gained > 0].sort_values(ascending=False).items()},
            "keluaran": output.getvalue().strip(),
            "galat": None,
        })
        log.append(entry)
    return {"train": ns["train"], "valid": ns["valid"], "test": ns["test"], "log": log,
            "error": next((e for e in log if e["galat"]), None)}


# Selaraskan test dengan train (R12), dijalankan di skrip (sesudah baca data) DAN di aplikasi (align):
# kolom hilang -> kosong, urutan kolom = train, kolom tambahan tetap di belakang, tipe mengikuti train
# ('1.500' teks di kolom angka -> 1500; kolom teks yang terbaca angka/kosong semua -> object).
ALIGN = (
    "KOLOM_FITUR = [kolom for kolom in train.columns if kolom != LABEL]\n"
    "tambahan = [kolom for kolom in test.columns if kolom not in KOLOM_FITUR]\n"
    "hilang = [kolom for kolom in KOLOM_FITUR if kolom not in test.columns]\n"
    "test = test.reindex(columns=KOLOM_FITUR + tambahan)\n"
    "for kolom in KOLOM_FITUR:\n"
    "    if pd.api.types.is_numeric_dtype(train[kolom]) and not pd.api.types.is_numeric_dtype(test[kolom]):\n"
    '        teks = test[kolom].astype("string").str.strip()\n'
    '        ribuan = teks.str.fullmatch(r"-?\\d{1,3}(?:\\.\\d{3})+(?:,\\d+)?").fillna(False).astype(bool)\n'
    '        teks = teks.where(~ribuan, teks.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))\n'
    '        test[kolom] = pd.to_numeric(teks, errors="coerce")\n'
    "    elif not pd.api.types.is_numeric_dtype(train[kolom]):\n"
    "        test[kolom] = test[kolom].astype(object)\n"
    'print("test: kolom hilang", len(hilang), "| kolom tambahan", len(tambahan))'
)


def align(train: pd.DataFrame, test: pd.DataFrame | None, label: str | None) -> pd.DataFrame | None:
    """Jalankan ALIGN apa adanya (satu sumber dengan skrip). None = tidak ada test."""
    if test is None:
        return None
    ns = {"pd": pd, "np": np, "train": train, "test": test.copy(), "LABEL": label}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(ALIGN, ns)
    return ns["test"]


def header(steps: list[dict]) -> str:
    """Blok pertama: docstring, impor, stdout aman cp1252 (R13), versi library (R14)."""
    uses_sklearn = any("sklearn" in step_code(step) for step in active(steps) if step["op"] in OPS)
    versions = 'print("versi: pandas", pd.__version__, "| numpy", np.__version__'
    versions += ', "| sklearn", sklearn.__version__)' if uses_sklearn else ")"
    return ('"""Pipeline EDA dari Laya EDA. Jalankan dari atas ke bawah; setiap `# %%` adalah satu langkah.\n\n'
            "Aturan: split dulu, lalu statistik (median, batas outlier, daftar kategori) dihitung dari train SAJA\n"
            "dan dipakai ke valid dan test; baris hanya dibuang dari train (valid dan test tidak pernah dibuang).\"\"\"\n"
            "import sys\n" + imports(steps) + ("\nimport sklearn" if uses_sklearn else "") + "\n"
            'sys.stdout.reconfigure(errors="replace")   # Windows cp1252: karakter lain jadi "?", skrip tidak berhenti\n'
            + versions)


def script_blocks(project: dict, train_path: str, test_path: str = "") -> list[str]:
    """Blok skrip sampai langkah terakhir: kepala → baca data + selaraskan test → split → langkah."""
    steps, label, config = project["steps"], project["label"], project.get("split")
    parts = [
        header(steps),
        "# %% baca data\n"
        f"TRAIN_CSV = {train_path!r}\n"
        f"TEST_CSV = {test_path!r}   # kosongkan bila tidak ada file test\n"
        f"LABEL = {label!r}\n"
        "train = pd.read_csv(TRAIN_CSV)\n"
        "test = pd.read_csv(TEST_CSV) if TEST_CSV else train.iloc[:0].drop(columns=[LABEL], errors=\"ignore\")\n"
        'print("train", train.shape, "test", test.shape)\n'
        "# selaraskan test dengan train: kolom hilang -> kosong, urutan & tipe = train (tahan test yang beda bentuk)\n"
        + ALIGN,
        f"# %% split data\n{split.split_code(config)}",
    ]
    for number, step in enumerate(active(steps), 1):
        parts.append(f"# %% langkah {number}: {summary(step)}\n{step_code(step)}")
    return parts


def build_script(project: dict, train_path: str, test_path: str = "") -> str:
    """Satu file .py tanpa fungsi. `# %%` = sel (VS Code / Jupyter). Urutan: baca → split → langkah → simpan."""
    config = project.get("split")
    parts = script_blocks(project, train_path, test_path)
    parts.append("# %% simpan\n"
                 'print("train", train.shape, "valid", valid.shape, "test", test.shape)\n'
                 'train.to_csv("train_bersih.csv", index=False)\n'
                 + ('valid.to_csv("valid_bersih.csv", index=False)\n' if config else "")
                 + 'test.to_csv("test_bersih.csv", index=False)')
    parts.append("# %% (opsional) latih model AutoGluon dengan data bersih\n"
                 "# from autogluon.tabular import TabularPredictor\n"
                 "# predictor = TabularPredictor(label=LABEL).fit(train, time_limit=600)\n"
                 "# print(predictor.leaderboard())")
    return "\n\n\n".join(parts) + "\n"


def to_notebook(script: str) -> str:
    cells = []
    for chunk in script.split("# %%"):
        if chunk.strip():
            source = ("# " + chunk.strip()) if cells else chunk.strip()
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": source.splitlines(keepends=True)})
    return json.dumps({"cells": cells, "metadata": {"language_info": {"name": "python"}},
                       "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1)


def diff(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Ringkasan sebelum → sesudah satu langkah: baris dibuang, kolom baru/hilang, sel berubah per kolom."""
    common_rows = before.index.intersection(after.index)
    common_cols = [c for c in before.columns if c in after.columns]
    old, new = before.loc[common_rows, common_cols], after.loc[common_rows, common_cols]
    changed = ~((old == new) | (old.isna() & new.isna()))
    per_column = changed.sum()
    per_column = per_column[per_column > 0]
    return {
        "baris dibuang": before.index.difference(after.index),
        "kolom baru": [c for c in after.columns if c not in before.columns],
        "kolom hilang": [c for c in before.columns if c not in after.columns],
        "sel berubah": per_column.sort_values(ascending=False),
        "contoh": changed.any(axis=1),
    }


def changed_examples(before: pd.DataFrame, after: pd.DataFrame, info: dict, limit: int = 30) -> pd.DataFrame:
    """Baris yang berubah: kolom 'x (sebelum)' dan 'x (sesudah)' berdampingan untuk kolom yang berubah."""
    rows = info["contoh"][info["contoh"]].index[:limit]
    columns = list(info["sel berubah"].index[:6])
    table = {}
    for column in columns:
        table[f"{column} (sebelum)"] = before.loc[rows, column]
        table[f"{column} (sesudah)"] = after.loc[rows, column]
    return pd.DataFrame(table, index=rows)


__all__ = ["new_project", "add_step", "run", "align", "header", "script_blocks", "build_script", "to_notebook",
           "diff", "changed_examples", "ALIGN"]
