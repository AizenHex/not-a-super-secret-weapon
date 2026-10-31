"""Jalankan kode pipeline yang dihasilkan codegen. Aplikasi memakai kode ini, bukan jalur lain (P2)."""
from __future__ import annotations

import types

import pandas as pd

from core.codegen import build_script


def load_script(script: str) -> types.ModuleType:
    module = types.ModuleType("laya_pipeline")
    exec(compile(script, "pipeline_eda.py", "exec"), module.__dict__)
    return module


def run_recipe(recipe: dict, train: pd.DataFrame, test: pd.DataFrame | None = None,
               paths: dict | None = None) -> tuple[dict, str]:
    """Bangun kode dari resep, jalankan, kembalikan (hasil run, kode)."""
    script = build_script(recipe, paths or {})
    return load_script(script).run(train, test), script
