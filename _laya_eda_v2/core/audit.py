"""Temuan data: profil kolom, tebakan tipe, penanda kosong palsu, nilai penjaga, kolom kembar.

Semua fungsi hanya MEMBACA data dan mengembalikan tabel temuan. Tidak ada yang mengubah data;
pengguna memilih temuan mana yang masuk resep.
"""
from __future__ import annotations

import pandas as pd

from core.ops import MISSING_MARKERS, SENTINEL_FLOOR

NUMBER_PATTERN = r"^\s*(?:rp\.?\s*)?-?[\d.,]+\s*$"


def tidy(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def guess_type(series: pd.Series) -> str:
    """angka / angka-teks (angka tersimpan sebagai teks) / kategori / isian bebas / id / kosong."""
    values = series.dropna()
    if values.empty:
        return "kosong"
    if pd.api.types.is_numeric_dtype(series):
        return "id" if series.is_unique and len(series) > 50 else "angka"
    text = tidy(series)
    if text.str.lower().str.match(NUMBER_PATTERN).mean() > 0.95:
        return "angka-teks"
    unique = text.nunique()
    if unique <= 30 or unique / len(text) < 0.002:
        return "kategori"
    return "isian bebas"


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        top = series.value_counts().head(3)
        rows.append({
            "kolom": column,
            "tipe tebakan": guess_type(series),
            "dtype pandas": str(series.dtype),
            "kosong": int(series.isna().sum()),
            "kosong %": round(series.isna().mean() * 100, 1),
            "unik": int(series.nunique()),
            "contoh tersering": ", ".join(f"{value!s} ({count})" for value, count in top.items()),
        })
    return pd.DataFrame(rows)


def fake_missing(df: pd.DataFrame, markers: list[str] = MISSING_MARKERS) -> pd.DataFrame:
    """Sel teks yang sebenarnya berarti 'kosong': '-', '?', 'null', ..."""
    rows = []
    for column in df.select_dtypes(exclude="number").columns:
        text = tidy(df[column]).str.lower()
        hits = text[text.isin(markers)].value_counts()
        for marker, count in hits.items():
            rows.append({"kolom": column, "penanda": repr(marker), "jumlah": int(count)})
    return pd.DataFrame(rows, columns=["kolom", "penanda", "jumlah"])


def sentinels(df: pd.DataFrame, floor: int = SENTINEL_FLOOR) -> pd.DataFrame:
    """Nilai penjaga: 2.147.483.647 (2^31 - 1) dan variannya, dipakai sistem sebagai 'tidak diisi'."""
    rows = []
    for column in df.select_dtypes(include="number").columns:
        big = df[column][df[column] >= floor]
        if len(big):
            rows.append({"kolom": column, "jumlah": len(big),
                         "nilai": ", ".join(f"{v:,.0f}" for v in sorted(big.unique())[:4]),
                         "maks wajar (di bawah batas)": df[column][df[column] < floor].max()})
    return pd.DataFrame(rows, columns=["kolom", "jumlah", "nilai", "maks wajar (di bawah batas)"])


def zero_share(df: pd.DataFrame, minimum: float = 0.02) -> pd.DataFrame:
    """Kolom angka dengan banyak nol: bisa nol sungguhan (tidak punya hutang) atau 'tidak diisi'."""
    share = (df.select_dtypes(include="number") == 0).mean()
    share = share[share >= minimum].sort_values(ascending=False)
    return pd.DataFrame({"kolom": share.index, "nol %": (share * 100).round(1).to_numpy()})


def single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Kolom dengan satu nilai + sel kosong: biasanya kosong = 'tidak' (checkbox)."""
    rows = []
    for column in df.columns:
        values = df[column].dropna().unique()
        if len(values) == 1 and df[column].isna().any():
            rows.append({"kolom": column, "satu-satunya nilai": values[0],
                         "kosong": int(df[column].isna().sum())})
    return pd.DataFrame(rows, columns=["kolom", "satu-satunya nilai", "kosong"])


def twin_columns(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Pasangan kolom yang isinya hampir sama (Cramér's V dari skrub.column_associations)."""
    import skrub

    data = df.loc[:, df.nunique() > 1]
    associations = skrub.column_associations(data)
    strong = associations[associations["cramer_v"] >= threshold]
    return strong[["left_column_name", "right_column_name", "cramer_v"]].rename(columns={
        "left_column_name": "kolom 1", "right_column_name": "kolom 2", "cramer_v": "Cramér's V"})


def row_findings(df: pd.DataFrame, label: str | None) -> dict:
    features = [column for column in df.columns if column != label]
    duplicated = df.duplicated(features, keep=False)
    conflict = 0
    if label and duplicated.any():
        conflict = int((df[duplicated].groupby(features, dropna=False)[label].nunique() > 1).sum())
    missing = df[features].isna().sum(axis=1)
    return {
        "duplikat persis": int(df.duplicated(features).sum()),
        "grup duplikat dengan label berbeda": conflict,
        "kosong per baris (maks)": int(missing.max()),
        "distribusi kosong per baris": missing.value_counts().sort_index().to_dict(),
    }


def label_issues(df: pd.DataFrame, label: str, folds: int = 5,
                 oof: pd.DataFrame | None = None) -> pd.DataFrame:
    """Label yang mungkin salah: prediksi out-of-fold + cleanlab (inti, tanpa paket tambahan Datalab).

    Tanpa `oof`: HistGradientBoosting 5-fold di kolom angka saja. Dengan `oof` (AutoGluon
    predict_proba_oof, kolom = kelas): semua kolom ikut. Baris berlabel kosong dilewati. Hasil
    diurutkan dari skor kualitas label terendah; label tidak pernah diubah.
    """
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    data = df[df[label].notna()]
    codes, classes = pd.factorize(data[label], sort=True)       # cleanlab butuh label 0..K-1
    if oof is not None:
        probabilities = oof.loc[data.index, list(classes)].to_numpy()
    else:
        features = data.drop(columns=[label]).select_dtypes("number")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        probabilities = cross_val_predict(HistGradientBoostingClassifier(random_state=0), features, codes,
                                          cv=splitter, method="predict_proba")
    issue = find_label_issues(codes, probabilities)
    result = data.assign(**{
        "skor label": get_label_quality_scores(codes, probabilities),
        "tebakan model": classes[probabilities.argmax(axis=1)],
        "yakin tebakan": probabilities.max(axis=1).round(3),
    })
    columns = [label, "tebakan model", "yakin tebakan", "skor label"]
    flagged = result[issue].sort_values("skor label")[columns + [c for c in data.columns if c != label]]
    flagged.attrs["akurasi model"] = float((probabilities.argmax(axis=1) == codes).mean())
    flagged.attrs["akurasi tebak terbanyak"] = float(pd.Series(codes).value_counts(normalize=True).iloc[0])
    return flagged


def check_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Cek aturan logika (pandas query), mis. '`Luas Bangunan` > `Luas Tanah`'. Kembalikan baris cocok."""
    return df[df.eval(rule).fillna(False).astype(bool)]
