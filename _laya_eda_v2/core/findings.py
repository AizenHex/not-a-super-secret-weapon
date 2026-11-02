"""Temuan UMUM (VISI.md G1): detektor berbasis profil kolom, bukan kamus domain.

`scan(data, label)` hanya MEMBACA data dan mengembalikan daftar temuan. Setiap temuan:
{"jenis", "kolom": [...], "judul", "alasan", "detail": DataFrame | None, "penting": 1 (tinggi) .. 3 (info),
 "pilihan": [{"judul", "op", "params"}, ...]}  ← pilihan pertama = usulan; semua parameter bisa diedit di UI (G2).
Tidak ada langkah yang ditambahkan otomatis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import audit

LONG_TEXT = 40              # rata-rata karakter: di atas ini = teks panjang
FEW_CATEGORIES = 15         # ≤ ini: one-hot masuk akal
SENTINEL_VALUES = {-1, -9, -99, -999, -9999, 99, 999, 9999, 99999, 999999, 2**31 - 1, 2**32 - 1}


def option(title: str, op: str | None, params: dict | None = None) -> dict:
    return {"judul": title, "op": op, "params": params or {}}


def finding(kind: str, columns: list, title: str, reason: str, choices: list[dict], detail=None,
            important: int = 2) -> dict:
    return {"jenis": kind, "kolom": list(columns), "judul": title, "alasan": reason, "detail": detail,
            "penting": important, "pilihan": choices}


def label_kind(data: pd.DataFrame, label: str | None) -> str:
    """'kelas' (teks / ≤ 20 nilai) atau 'angka' (regresi): menentukan jenis target encoding."""
    if not label or label not in data:
        return "kelas"
    y = data[label]
    return "kelas" if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20 else "angka"


def features(data: pd.DataFrame, label: str | None) -> list[str]:
    return [c for c in data.columns if c != label]


# ---------------------------------------------------------------- kosong

def missing(data: pd.DataFrame, label: str | None) -> list[dict]:
    share = data[features(data, label)].isna().mean()
    share = share[share > 0].sort_values(ascending=False)
    out = []
    mostly = list(share[share >= 0.6].index)
    if mostly:
        detail = pd.DataFrame({"kolom": mostly, "kosong %": (share[mostly] * 100).round(1).to_numpy()})
        out.append(finding("kosong", mostly, f"{len(mostly)} kolom ≥ 60% kosong",
                           "Kosong sebanyak ini jarang bisa diisi dengan wajar. Sering artinya 'tidak punya' "
                           "(mis. tidak ada kolam renang): pertahankan sebagai penanda, atau buang kolomnya.",
                           [option("Tambah kolom penanda kosong (0/1), kolom asli tetap", "missing_indicator",
                                   {"columns": mostly}),
                            option("Buang kolom", "drop_columns",
                                   {"columns": mostly, "reason": "kolom ≥ 60% kosong"})], detail, 1))
    rest = share[share < 0.6]
    numbers = [c for c in rest.index if pd.api.types.is_numeric_dtype(data[c])]
    texts = [c for c in rest.index if c not in numbers]
    if numbers:
        detail = pd.DataFrame({"kolom": numbers, "kosong %": (rest[numbers] * 100).round(2).to_numpy(),
                               "median": [data[c].median() for c in numbers]})
        out.append(finding("kosong", numbers, f"Angka kosong di {len(numbers)} kolom",
                           "Isi dengan median train (tahan outlier); kolom yang kosongnya ≥ 5% sebaiknya juga diberi "
                           "penanda karena 'kosong' bisa membawa informasi. Model pohon (LightGBM, HistGB) menerima "
                           "kosong apa adanya.",
                           [option("Isi median (statistik dari train)", "impute",
                                   {"columns": numbers, "strategy": "median"}),
                            option("Tambah penanda kosong dulu (kolom ≥ 5% kosong)", "missing_indicator",
                                   {"columns": [c for c in numbers if rest[c] >= 0.05] or numbers}),
                            option("Biarkan (model pohon menerima kosong)", None)], detail, 1))
    ticks = [c for c in texts if data[c].nunique() == 1]
    texts = [c for c in texts if c not in ticks]
    for column in ticks:
        only = data[column].dropna().iloc[0]
        out.append(finding("centang", [column], f"Kolom centang '{column}': hanya {only!r} + {int(data[column].isna().sum()):,} kosong",
                           "Satu nilai + kosong biasanya kotak centang: kosong = 'tidak'. Ubah nilai pengisi bila artinya lain.",
                           [option("Isi 'Tidak'", "fill_constant", {"columns": [column], "value": "Tidak"}),
                            option("Isi '(kosong)' sebagai kategori", "fill_constant", {"columns": [column], "value": "(kosong)"})],
                           None, 2))
    if texts:
        detail = pd.DataFrame({"kolom": texts, "kosong %": (rest[texts] * 100).round(2).to_numpy()})
        out.append(finding("kosong", texts, f"Teks/kategori kosong di {len(texts)} kolom",
                           "Kosong di kategori paling aman dijadikan kategori sendiri: tidak menebak isinya.",
                           [option("Isi '(kosong)' sebagai kategori", "fill_constant",
                                   {"columns": texts, "value": "(kosong)"}),
                            option("Isi modus (nilai tersering di train)", "impute",
                                   {"columns": texts, "strategy": "mode"})], detail, 2))
    return out


# ---------------------------------------------------------------- tipe & kardinalitas

def is_date_text(series: pd.Series) -> bool:
    values = series.dropna().astype(str)
    if values.empty or values.str.fullmatch(r"-?[\d.,\s]+").mean() > 0.5:      # angka bukan tanggal
        return False
    sample = values.sample(min(len(values), 300), random_state=0)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9 and sample.str.contains(r"\d{1,4}[-/.:]\d{1,2}", regex=True).mean() >= 0.9


def types(data: pd.DataFrame, label: str | None) -> list[dict]:
    out, n = [], len(data)
    constant, ids, dates, number_text, coded, long_few, long_many, few, many = ([] for _ in range(9))
    for column in features(data, label):
        series = data[column]
        values = series.dropna()
        unique = values.nunique()
        numeric = pd.api.types.is_numeric_dtype(series)
        if unique <= 1 or (len(values) and values.value_counts(normalize=True).iloc[0] >= 0.995):
            constant.append(column)
            continue
        if n > 50 and unique >= 0.95 * len(values) and (numeric and (values % 1 == 0).all() or not numeric) \
                and not (not numeric and is_date_text(series)):
            ids.append(column)
            continue
        if numeric:
            if (values % 1 == 0).all() and 2 < unique <= 10 and values.min() >= 0:
                coded.append(column)
            continue
        guess = audit.guess_type(series)
        if guess == "angka-teks":
            number_text.append(column)
        elif is_date_text(series):
            dates.append(column)
        elif values.astype(str).str.len().mean() > LONG_TEXT:
            (long_few if unique <= 30 else long_many).append(column)
        elif unique <= FEW_CATEGORIES:
            few.append(column)
        else:
            many.append(column)
    table = lambda cols: pd.DataFrame({"kolom": cols, "unik": [data[c].nunique() for c in cols],  # noqa: E731
                                       "contoh": [", ".join(map(str, data[c].dropna().unique()[:3])) for c in cols]})
    if constant:
        out.append(finding("konstan", constant, f"{len(constant)} kolom (hampir) satu nilai saja",
                           "Kolom yang ≥ 99,5% isinya sama tidak membantu model.",
                           [option("Buang kolom", "drop_columns", {"columns": constant, "reason": "kolom hampir konstan"})],
                           table(constant), 2))
    if ids:
        out.append(finding("id", ids, f"Kolom id / unik per baris: {', '.join(map(str, ids[:5]))}",
                           "Nilai unik di setiap baris tidak membawa pola; bila model menganggapnya penting, itu tanda "
                           "bocor urutan data. Simpan kolom id untuk submission (tab Latih: 'Bukan fitur').",
                           [option("Buang kolom", "drop_columns",
                                   {"columns": ids, "reason": "kolom id: nilai unik per baris"})], table(ids), 1))
    if number_text:
        out.append(finding("angka_teks", number_text, f"Angka tersimpan sebagai teks: {len(number_text)} kolom",
                           "Kolom angka yang terbaca teks (spasi, titik ribuan, 'Rp') tidak bisa dihitung.",
                           [option("Teks → angka (format Indonesia: titik = ribuan)", "to_number",
                                   {"columns": number_text, "indonesian": True}),
                            option("Teks → angka (format Inggris: koma = ribuan)", "to_number",
                                   {"columns": number_text, "indonesian": False})], table(number_text), 1))
    if dates:
        out.append(finding("tanggal", dates, f"Kolom tanggal/waktu tersimpan sebagai teks: {', '.join(dates[:4])}",
                           "Teks tanggal hampir unik per baris: model tidak bisa memakainya. Pecah jadi tahun, bulan, "
                           "hari, hari dalam minggu, jam.",
                           [option("Ambil fitur tanggal (kolom asli tetap)", "datetime_features",
                                   {"columns": dates, "parts": ["year", "month", "day", "dayofweek", "hour"]})],
                           table(dates), 1))
    if coded:
        out.append(finding("kategori_angka", coded, f"Kategori yang mungkin ditulis sebagai angka: {', '.join(coded[:6])}",
                           "Bilangan bulat dengan sedikit nilai bisa berupa KODE (jenis nyeri 0–3) atau BESARAN (jumlah "
                           "anak). Bila kode, model linear salah membaca jaraknya. Cek arti kolom dulu.",
                           [option("Jadikan kategori (teks)", "to_category", {"columns": coded}),
                            option("Biarkan sebagai angka", None)], table(coded), 3))
    if long_few:
        out.append(finding("teks_panjang", long_few, f"Teks panjang tapi nilainya sedikit: {', '.join(long_few[:4])}",
                           "Isinya kalimat, tetapi hanya beberapa kalimat berbeda: perlakukan sebagai kategori.",
                           [option("One-hot (sklearn)", "onehot_sklearn", {"columns": long_few, "min_share": 0.01}),
                            option("Biarkan (tab Latih memberi kode angka)", None)], table(long_few), 2))
    if long_many:
        out.append(finding("teks_panjang", long_many, f"Teks panjang/bebas: {', '.join(long_many[:4])}",
                           "Teks bebas tidak bisa langsung dipakai model. Ambil statistik sederhana (panjang, jumlah "
                           "kata), atau ubah jadi angka dengan encoder teks.",
                           [option("Statistik teks: panjang + jumlah kata (kolom asli tetap)", "text_stats",
                                   {"columns": long_many}),
                            option("skrub StringEncoder (n-gram → angka)", "skrub_encoder",
                                   {"column": long_many[0], "kind": "string", "n_components": 10})],
                           table(long_many), 2))
    if few:
        out.append(finding("kategori_sedikit", few, f"{len(few)} kolom kategori (≤ {FEW_CATEGORIES} nilai)",
                           "Model pohon di tab Latih sudah memberi kode angka otomatis. One-hot berguna untuk model "
                           "linear/KNN atau bila urutan kode abjad menyesatkan.",
                           [option("One-hot (sklearn, masuk Pipeline)", "onehot_sklearn",
                                   {"columns": few, "min_share": 0.01}),
                            option("Biarkan (kode angka otomatis di Latih)", None)], table(few), 3))
    if many:
        out.append(finding("kardinalitas_tinggi", many, f"{len(many)} kolom kategori dengan banyak nilai",
                           "One-hot akan membuat terlalu banyak kolom. Target encoding (cross-fitting) atau gabung "
                           "kategori langka lebih cocok.",
                           [option("Target encoding (sklearn TargetEncoder)", "target_encode",
                                   {"columns": many, "kind": label_kind(data, label)}),
                            option("Kategori langka → 'Lainnya' (porsi < 1%)", "rare_label",
                                   {"columns": many, "tol": 0.01})], table(many), 2))
    return out


# ---------------------------------------------------------------- nilai penjaga, kebocoran, label

def sentinels(data: pd.DataFrame, label: str | None) -> list[dict]:
    """Nilai ekstrem yang BERULANG (2147483647, 9999, -1) = kode 'tidak diisi', bukan ukuran."""
    rows = []
    for column in features(data, label):
        values = data[column].dropna()
        if not pd.api.types.is_numeric_dtype(values) or values.nunique() < 10:
            continue
        counts = values.value_counts()
        for value in (values.max(), values.min()):
            count = int(counts.get(value, 0))
            rest = values[values != value]
            if count < max(3, 0.001 * len(values)) or rest.empty:
                continue
            spread = rest.quantile(0.99) - rest.quantile(0.01)
            far = value > rest.quantile(0.99) + 10 * spread if value == values.max() else \
                value < rest.quantile(0.01) - 10 * spread
            if value in SENTINEL_VALUES and (far or value < 0 <= rest.min()) or (far and count >= 0.002 * len(values)):
                rows.append({"kolom": column, "nilai": value, "baris": count,
                             "maks wajar (di bawah batas)": rest.max()})
    if not rows:
        return []
    detail = pd.DataFrame(rows)
    return [finding("penjaga", list(detail["kolom"].unique()), f"Nilai penjaga (kode 'tidak diisi') di {detail['kolom'].nunique()} kolom",
                    "Nilai ekstrem yang sama berulang (mis. 2147483647, 9999, -1) biasanya kode sistem, bukan ukuran.",
                    [option("Jadikan kosong (nilai persis ini)", "values_to_nan",
                            {"columns": list(detail["kolom"].unique()),
                             "values": sorted({float(v) for v in detail["nilai"]})})], detail, 1)]


def leakage(data: pd.DataFrame, label: str | None, rows: int = 3000) -> list[dict]:
    """Satu kolom yang menebak label hampir sempurna = curiga bocor (atau label turunan kolom itu)."""
    if not label or label not in data or data[label].notna().sum() < 30:
        return []
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    sample = data[data[label].notna()]
    sample = sample.sample(min(rows, len(sample)), random_state=0)
    y = sample[label]
    classes = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20
    found = []
    for column in features(sample, label):
        x = sample[column]
        if x.nunique() > 0.5 * len(x) or x.nunique() < 2:
            continue
        codes = (x.astype("category").cat.codes if not pd.api.types.is_numeric_dtype(x) else x.fillna(-1))
        X = codes.to_numpy().reshape(-1, 1)
        model = DecisionTreeClassifier(max_depth=4, random_state=0) if classes else \
            DecisionTreeRegressor(max_depth=6, random_state=0)
        score = cross_val_score(model, X, y.astype(str) if classes else y, cv=3).mean()
        baseline = y.value_counts(normalize=True).iloc[0] if classes else 0
        if score >= 0.98 and score - baseline >= 0.1:
            found.append({"kolom": column, "skor 1 kolom": round(float(score), 4), "tebak terbanyak": round(float(baseline), 4)})
    if not found:
        return []
    detail = pd.DataFrame(found)
    return [finding("bocor", list(detail["kolom"]), f"Curiga bocor: {', '.join(map(str, detail['kolom'][:4]))} menebak label "
                    "hampir sempurna sendirian",
                    "Satu kolom saja memberi skor ≥ 0,98 (pohon keputusan, validasi silang 3 fold). Biasanya kolom itu "
                    "dihitung DARI label atau ditulis sesudah label diketahui; di test tidak akan seberguna ini. "
                    "Bisa juga memang aturan sungguhan: pastikan kolom itu tersedia di test.",
                    [option("Buang kolom yang bocor", "drop_columns",
                            {"columns": list(detail["kolom"]), "reason": "curiga bocor: menebak label sendirian"}),
                     option("Biarkan (kolom ini sah dan ada di test)", None)], detail, 1)]


def label_checks(data: pd.DataFrame, label: str | None) -> list[dict]:
    if not label or label not in data:
        return [finding("label", [], "Belum memilih kolom label", "Pilih label di sidebar Data agar temuan label, "
                        "kebocoran, dan tab Latih bisa dipakai.", [option("—", None)], None, 2)]
    out, y = [], data[label]
    if y.isna().any():
        out.append(finding("label", [label], f"{int(y.isna().sum()):,} baris train tanpa label",
                           "Baris tanpa label tidak bisa dipakai melatih model (tab Latih melewatinya otomatis).",
                           [option("Buang baris tanpa label (train)", "drop_rows_query",
                                   {"rule": f"`{label}`.isna()", "reason": "label kosong"}),
                            option("Biarkan", None)], None, 2))
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20:
        counts = y.value_counts()
        if len(counts) > 1 and counts.min() / counts.max() < 0.2:
            out.append(finding("timpang", [label], f"Label timpang: kelas {counts.idxmin()!r} {counts.min():,} baris vs "
                               f"{counts.idxmax()!r} {counts.max():,}",
                               "Pakai metrik yang adil per kelas (macro-F1) dan pilihan 'kelas timpang' di tab Latih "
                               "(bobot / SMOTE di dalam fold / ambang kelas).", [option("Lihat tab Latih", None)],
                               counts.rename_axis("kelas").rename("baris").reset_index(), 2))
    return out


def rows(data: pd.DataFrame, label: str | None) -> list[dict]:
    out = []
    info = audit.row_findings(data, label)
    if info["duplikat persis"]:
        out.append(finding("duplikat", [], f"{info['duplikat persis']:,} baris duplikat persis di train",
                           f"Grup duplikat dengan label berbeda: {info['grup duplikat dengan label berbeda']}.",
                           [option("Buang duplikat (train saja)", "drop_duplicates",
                                   {"reason": "baris train kembar persis"}), option("Biarkan", None)], None, 2))
    heavy = max(5, data.shape[1] // 2)
    if info["kosong per baris (maks)"] >= heavy:
        spread = info["distribusi kosong per baris"]
        out.append(finding("baris_kosong", [], f"Ada baris dengan {info['kosong per baris (maks)']} kolom kosong",
                           "Baris yang sebagian besar kosong sering rusak. Dibuang dari train saja.",
                           [option(f"Buang baris train dengan > {heavy} kolom kosong", "drop_missing_rows",
                                   {"max_missing": heavy})],
                           pd.DataFrame({"kosong per baris": list(spread), "jumlah baris": list(spread.values())}), 3))
    return out


def text_mess(data: pd.DataFrame, label: str | None, markers: list[str]) -> list[dict]:
    out = []
    fake = audit.fake_missing(data, markers)
    messy = audit.messy_text(data)
    plain = fake[~fake["mungkin kategori"].astype(bool)] if len(fake) else fake
    columns = sorted((set(plain["kolom"]) if len(plain) else set()) | (set(messy["kolom"]) if len(messy) else set()))
    columns = [c for c in columns if c != label]
    if columns:
        out.append(finding("teks_kotor", columns, f"Spasi berlebih / penanda kosong palsu di {len(columns)} kolom",
                           "Spasi di depan/belakang membuat 'A' ≠ 'A '; penanda seperti '?', '-', 'null' sebenarnya "
                           "kosong. Penanda yang sering (≥ 5%) bisa berarti jawaban 'tidak ada': cek dulu.",
                           [option("Rapikan teks + penanda kosong → kosong", "tidy_text",
                                   {"columns": columns, "markers": markers})],
                           pd.concat([messy, plain], ignore_index=True) if len(messy) or len(plain) else None, 1))
    spelling = audit.spelling_findings(data[[c for c in data.columns if c != label]])
    if len(spelling):
        out.append(finding("ejaan", list(spelling["kolom"]), f"Ejaan berbeda untuk nilai yang sama di {len(spelling)} kolom",
                           "Nilai yang hanya beda huruf besar/spasi/tanda baca dihitung sebagai kategori berbeda.",
                           [option("Satukan ejaan (aturan kunci)", "spelling",
                                   {"groups": dict(zip(spelling["kolom"], spelling["groups"])),
                                    "examples": dict(zip(spelling["kolom"], spelling["examples"]))})],
                           spelling[["kolom", "nilai unik", "sesudah", "contoh"]], 2))
    return out


def scan(data: pd.DataFrame, label: str | None, markers: list[str] | None = None,
         reader_notes: list[dict] | None = None) -> list[dict]:
    """Semua temuan, urut penting (1 = tinggi). `reader_notes` = temuan saat membaca file (core/reader.py)."""
    from core.ops import MISSING_MARKERS

    markers = markers or MISSING_MARKERS
    found = list(reader_notes or [])
    for detector in (label_checks, leakage, sentinels, missing, types, rows):
        found += detector(data, label)
    found += text_mess(data, label, markers)
    return sorted(found, key=lambda item: item["penting"])


__all__ = ["scan", "missing", "types", "sentinels", "leakage", "label_checks", "rows", "text_mess", "is_date_text"]
