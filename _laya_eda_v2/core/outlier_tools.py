"""Perbandingan metode outlier per kolom. Batas dihitung dengan MENJALANKAN kode langkah outlier (satu sumber)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.ops import outlier


def outlier_bounds(values: pd.Series, method: str, factor: float) -> tuple[float, float]:
    """Batas (bawah, atas) persis seperti di kode langkah outlier."""
    namespace = {"np": np, "pd": pd, "train": pd.DataFrame({"x": values}), "test": pd.DataFrame({"x": []})}
    code = outlier({"column": "x", "method": method, "factor": factor, "action": "nan"})
    exec(code.split("print(")[0], namespace)
    return float(namespace["bawah"]), float(namespace["atas"])


def is_outside(values: pd.Series, bounds: tuple[float, float], method: str) -> pd.Series:
    low, high = bounds
    outside = (values < low) | (values > high)
    return outside & (values > 0) if method.endswith("_log") else outside

FACTORS = {"iqr": 1.5, "iqr_log": 1.5, "mad": 3.5, "mad_log": 3.5}


def compare_methods(values: pd.Series, factors: dict[str, float] = FACTORS) -> pd.DataFrame:
    """Satu baris per metode: batas, jumlah di luar batas, dan peringatan bila metode tidak cocok."""
    values = values.dropna()
    rows = []
    for method, factor in factors.items():
        low, high = outlier_bounds(values, method, factor)
        outside = is_outside(values, (low, high), method)
        note = ""
        if np.isinf(low) and np.isinf(high):
            note = "sebaran nol (mayoritas nilai sama): metode ini tidak bisa membedakan"
        elif method.endswith("_log") and (values <= 0).any():
            note = f"{int((values <= 0).sum())} nilai <= 0 tidak dinilai (skala log)"
        elif outside.mean() > 0.10:
            note = "menandai > 10% data: kemungkinan distribusi miring, coba versi log"
        rows.append({"metode": method, "k": factor, "batas bawah": low, "batas atas": high,
                     "di luar batas": int(outside.sum()), "% data": round(outside.mean() * 100, 2),
                     "catatan": note})
    return pd.DataFrame(rows)


def outside_rows(df: pd.DataFrame, column: str, method: str, factor: float) -> pd.DataFrame:
    bounds = outlier_bounds(df[column], method, factor)
    return df[is_outside(df[column], bounds, method).fillna(False)]


def by_group(df: pd.DataFrame, column: str, group: str, method: str, factor: float) -> pd.DataFrame:
    """Batas per kelompok (mis. per Golongan UKT): nilai wajar di satu kelompok bisa aneh di lain."""
    rows = []
    for name, part in df.groupby(group, dropna=False):
        low, high = outlier_bounds(part[column], method, factor)
        outside = is_outside(part[column], (low, high), method)
        rows.append({group: name, "baris": len(part), "batas bawah": low, "batas atas": high,
                     "di luar batas": int(outside.sum())})
    return pd.DataFrame(rows)


def isolation_flags(df: pd.DataFrame, columns: list[str], contamination: float = 0.01) -> pd.Series:
    """IsolationForest multivariat (hanya menandai). Kosong diisi median sementara untuk skor saja."""
    from sklearn.ensemble import IsolationForest

    data = df[columns].apply(pd.to_numeric, errors="coerce")
    data = data.fillna(data.median())
    model = IsolationForest(contamination=contamination, random_state=0).fit(data)
    return pd.Series(model.predict(data) == -1, index=df.index, name="aneh_multivariat")
