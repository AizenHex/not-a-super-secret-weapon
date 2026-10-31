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

from core.ops import OPS, summary


def new_project(label: str | None = None) -> dict:
    return {"label": label, "steps": []}


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


def namespace(steps: list[dict], label: str | None, train: pd.DataFrame, test: pd.DataFrame | None) -> dict:
    """Pengganti blok 'baca data' saat dijalankan di aplikasi: train/test diberikan langsung."""
    ns = {}
    exec(imports(steps), ns)
    ns["LABEL"] = label
    ns["train"] = train.copy()
    ns["test"] = (test.copy() if test is not None
                  else train.iloc[:0].drop(columns=[label] if label else [], errors="ignore").copy())
    return ns


def run(steps: list[dict], label: str | None, train: pd.DataFrame, test: pd.DataFrame | None,
        first: int = 1) -> dict:
    """Jalankan langkah aktif satu per satu. Berhenti di langkah pertama yang galat (galat tidak disembunyikan).

    Kembalikan train, test, dan log per langkah: baris/kolom sebelum-sesudah, keluaran print, galat.
    `first` = nomor langkah pertama di log (untuk menjalankan lanjutan dari hasil langkah sebelumnya).
    """
    ns = namespace(steps, label, train, test)
    log = []
    for number, step in enumerate(steps, first):
        if not step.get("on", True):
            continue
        before = ns["train"].shape, list(ns["train"].columns), ns["test"].shape
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
        entry.update({
            "baris train": f"{before[0][0]:,} → {ns['train'].shape[0]:,}",
            "kolom": f"{before[0][1]} → {ns['train'].shape[1]}",
            "baris test": f"{before[2][0]:,} → {ns['test'].shape[0]:,}",
            "kolom baru": [c for c in after if c not in before[1]],
            "kolom hilang": [c for c in before[1] if c not in after],
            "keluaran": output.getvalue().strip(),
            "galat": None,
        })
        log.append(entry)
    return {"train": ns["train"], "test": ns["test"], "log": log,
            "error": next((e for e in log if e["galat"]), None)}


def build_script(project: dict, train_path: str, test_path: str = "") -> str:
    """Satu file .py tanpa fungsi. `# %%` = sel (VS Code / Jupyter)."""
    steps, label = project["steps"], project["label"]
    parts = [
        '"""Pipeline EDA dari Laya EDA. Jalankan dari atas ke bawah; setiap `# %%` adalah satu langkah.\n\n'
        "Aturan: statistik (median, batas outlier, daftar kategori) dihitung dari train lalu dipakai juga\n"
        'ke test; baris hanya dibuang dari train."""\n' + imports(steps),
        "# %% baca data\n"
        f"TRAIN_CSV = {train_path!r}\n"
        f"TEST_CSV = {test_path!r}   # kosongkan bila tidak ada file test\n"
        f"LABEL = {label!r}\n"
        "train = pd.read_csv(TRAIN_CSV)\n"
        "test = pd.read_csv(TEST_CSV) if TEST_CSV else train.iloc[:0].drop(columns=[LABEL], errors=\"ignore\")\n"
        'print("train", train.shape, "test", test.shape)',
    ]
    for number, step in enumerate(active(steps), 1):
        parts.append(f"# %% langkah {number}: {summary(step)}\n{step_code(step)}")
    parts.append("# %% simpan\n"
                 'print("train", train.shape, "test", test.shape)\n'
                 'train.to_csv("train_bersih.csv", index=False)\n'
                 'test.to_csv("test_bersih.csv", index=False)')
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


__all__ = ["new_project", "add_step", "run", "build_script", "to_notebook", "diff", "changed_examples"]
