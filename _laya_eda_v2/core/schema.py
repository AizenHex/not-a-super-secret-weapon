"""Fitur extra E4: validasi skema pandera. Skema dipelajari dari train (tipe, rentang, nilai sah), dicek ke test
sebelum `predict`. `lazy=True` = semua pelanggaran dilaporkan sekaligus.

Mode **lapor** (default): laporan dicetak, prediksi tetap jalan (R12: test berbeda bentuk tidak menghentikan
skrip). Mode **ketat**: berhenti dengan galat, hanya bila pengguna memilihnya.
Skema ditulis sebagai kode teks lurus (tanpa def) lalu dijalankan apa adanya: kode di template = skema di aplikasi.
"""
from __future__ import annotations

import pandas as pd

MAX_VALUES = 50          # kolom teks dengan ≤ 50 nilai di train → cek nilai sah (isin)


def learn(train: pd.DataFrame, label: str | None = None, exclude: list[str] = ()) -> dict:
    """{kolom: {"tipe": "angka" | "teks", "min", "max", "nilai"}} dari train saja."""
    spec = {}
    for column in train.columns:
        if column == label or column in exclude:
            continue
        values = train[column].dropna()
        if pd.api.types.is_numeric_dtype(train[column]):
            spec[column] = {"tipe": "angka", "min": float(values.min()) if len(values) else None,
                            "max": float(values.max()) if len(values) else None}
        else:
            unique = sorted(values.astype(str).unique())
            spec[column] = {"tipe": "teks", "nilai": unique if len(unique) <= MAX_VALUES else None}
    return spec


def schema_code(spec: dict) -> str:
    lines = ["import pandera.pandas as pa", "SKEMA = pa.DataFrameSchema({"]
    for column, rule in spec.items():
        if rule["tipe"] == "angka":
            check = (f", pa.Check.in_range({rule['min']!r}, {rule['max']!r})" if rule["min"] is not None else "")
            lines.append(f"    {column!r}: pa.Column(float{check}, nullable=True, coerce=True),")
        else:
            check = f", pa.Check.isin({rule['nilai']!r})" if rule["nilai"] is not None else ""
            lines.append(f"    {column!r}: pa.Column(str{check}, nullable=True, coerce=True),")
    lines.append("}, strict=False)   # kolom tambahan di test boleh; kolom hilang dilaporkan")
    return "\n".join(lines)


def check_code(spec: dict, frame: str = "test", strict: bool = False) -> str:
    """Blok lurus: validasi lazy; mode lapor mencetak ringkasan, mode ketat berhenti."""
    stop = "    raise" if strict else "    print(galat.failure_cases.groupby(['column', 'check']).size().to_string())"
    return (schema_code(spec) + "\n"
            f"try:\n    SKEMA.validate({frame}, lazy=True)\n    print('skema {frame}: lolos')\n"
            "except pa.errors.SchemaErrors as galat:\n"
            f"    print('skema {frame}:', len(galat.failure_cases), 'pelanggaran (mode {'ketat' if strict else 'lapor'})')\n"
            + stop)


def build(spec: dict):
    namespace: dict = {}
    exec(schema_code(spec), namespace)
    return namespace["SKEMA"]


def validate(data: pd.DataFrame, spec: dict, mode: str = "lapor") -> pd.DataFrame:
    """Laporan pelanggaran: kolom, pemeriksaan, jumlah baris, contoh nilai. Mode ketat → SchemaErrors dilempar."""
    import warnings

    import pandera.pandas as pa

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            build(spec).validate(data, lazy=True)
    except pa.errors.SchemaErrors as error:
        if mode == "ketat":
            raise
        cases = error.failure_cases
        rows = []
        for (column, check), group in cases.groupby(["column", "check"], dropna=False, sort=False):
            rows.append({"kolom": column, "pemeriksaan": str(check), "baris": int(group["index"].notna().sum()) or 1,
                         "contoh nilai": ", ".join(map(str, group["failure_case"].astype(str).unique()[:5]))})
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["kolom", "pemeriksaan", "baris", "contoh nilai"])


__all__ = ["learn", "schema_code", "check_code", "build", "validate"]
