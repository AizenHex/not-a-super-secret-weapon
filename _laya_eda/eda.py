"""EDA tabel: profil kolom, hubungan kolom dengan label, korelasi, data kosong, waktu.
Semua fungsi murni (tanpa streamlit): masuk DataFrame, keluar DataFrame / angka."""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import helpers
from outliers import decimal_text

CATEGORY_LIMIT = 50       # kolom dengan nilai unik <= ini dianggap kolom kategori
SMALL_NUMERIC_UNIQUE = 10  # kolom angka dengan nilai unik <= ini juga dianggap kategori
REGRESSION_UNIQUE = 10    # label angka dengan nilai unik > ini = regresi (bukan kelas)
RARE_SHARE = 0.01         # kategori di bawah 1% = langka
MIXED_MIN = 0.5           # kolom teks yang >= 50% terbaca angka tetapi tidak 100% = campur
SKEW_LIMIT = 1.0          # |skew| di atas ini = miring
REDUNDANT_LIMIT = 0.9     # |rho| di atas ini = kolom redundan
ALPHA = 0.05              # p-value di bawah ini = signifikan
CORRELATION_LIMITS = [0.1, 0.3, 0.5]    # ambang efek kecil / sedang / besar (Cohen) untuk V dan rho
EPSILON_LIMITS = [0.01, 0.06, 0.14]     # ambang efek kecil / sedang / besar untuk epsilon kuadrat
EFFECT_NAMES = ["diabaikan", "kecil", "sedang", "besar"]


# ---------------- profil per kolom ----------------

def is_category(series):
    unique = series.nunique()
    if pd.api.types.is_numeric_dtype(series):
        return unique <= SMALL_NUMERIC_UNIQUE
    return unique <= CATEGORY_LIMIT


def numeric_note(values):
    # (skew, saran) untuk kolom angka
    clean = values.dropna()
    if len(clean) < 3 or clean.nunique() < 2:
        return np.nan, ""
    skew = clean.skew()
    if skew > SKEW_LIMIT and clean.min() > 0:
        return skew, "miring ke kanan: coba skala log"
    if abs(skew) > SKEW_LIMIT:
        return skew, "miring: periksa ekor sebaran"
    return skew, ""


def mixed_note(series, indonesian):
    # kolom teks yang sebagian besar angka tetapi ada isian yang bukan angka
    present = series.dropna()
    parsed = helpers.to_number(present, indonesian)
    share = parsed.notna().mean()
    if share < MIXED_MIN or share == 1:
        return np.nan, ""
    bad = present[parsed.isna()].astype(str)
    example = bad.iloc[0]
    return share, f"angka bercampur teks ({100 * share:.0f}% terbaca angka; contoh '{example}')"


def add_type_notes(series, indonesian, line, notes):
    # skew (kolom angka) atau campuran angka-teks (kolom teks)
    note = ""
    if pd.api.types.is_numeric_dtype(series):
        line["skew"], note = numeric_note(series)
    elif series.dtype == object and series.notna().any():
        _, note = mixed_note(series, indonesian)
    if note != "":
        notes.append(note)


def add_rare_notes(series, counts, line, notes):
    if not is_category(series):
        return
    line["kategori_langka"] = int((counts < RARE_SHARE).sum())
    if line["kategori_langka"] > 0:
        notes.append(f"{line['kategori_langka']} kategori langka (< {int(100 * RARE_SHARE)}%)")


def profile_column(series, indonesian):
    # satu baris profil: tipe, kosong, unik, terbanyak, skew, kategori langka, saran
    counts = series.value_counts(normalize=True)
    line = {"tipe": str(series.dtype), "kosong_persen": round(100 * series.isna().mean(), 1),
            "unik": series.nunique(), "terbanyak": "", "skew": np.nan, "kategori_langka": 0}
    if len(counts) > 0:
        top_value = str(counts.index[0])[:40]
        line["terbanyak"] = f"{top_value} ({100 * counts.iloc[0]:.0f}%)"
    notes = []
    add_type_notes(series, indonesian, line, notes)
    add_rare_notes(series, counts, line, notes)
    line["saran"] = "; ".join(notes)
    return line


def profile_columns(data, indonesian):
    lines = []
    for column in data.columns:
        line = profile_column(data[column], indonesian)
        line["kolom"] = column
        lines.append(line)
    table = pd.DataFrame(lines)
    order = ["kolom", "tipe", "kosong_persen", "unik", "terbanyak", "skew", "kategori_langka",
             "saran"]
    return table[order]


# ---------------- hubungan kolom dengan label ----------------

def is_regression_label(label):
    return pd.api.types.is_numeric_dtype(label) and label.nunique() > REGRESSION_UNIQUE


def effect_name(value, limits):
    name = EFFECT_NAMES[0]
    for number, limit in enumerate(limits):
        if abs(value) >= limit:
            name = EFFECT_NAMES[number + 1]
    return name


def chi_square_test(feature, label):
    # (chi2, p, Cramer's V) atau None kalau tabel silang kurang dari 2 baris / kolom
    table = pd.crosstab(feature, label)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, p_value, _, _ = scipy_stats.chi2_contingency(table)
    total = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (total * (min(table.shape) - 1)))
    return chi2, p_value, cramers_v


def kruskal_test(values, groups):
    # (H, p, epsilon kuadrat) atau None; values dibandingkan antar kelompok groups
    frame = pd.DataFrame({"nilai": values, "kelompok": groups}).dropna()
    if frame["nilai"].nunique() < 2 or frame["kelompok"].nunique() < 2:
        return None
    parts = []
    for _, part in frame.groupby("kelompok"):
        parts.append(part["nilai"].to_numpy())
    h_value, p_value = scipy_stats.kruskal(*parts)
    epsilon2 = (h_value - len(parts) + 1) / (len(frame) - len(parts))
    return h_value, p_value, max(epsilon2, 0.0)


def spearman_test(values, other):
    frame = pd.DataFrame({"a": values, "b": other}).dropna()
    if len(frame) < 3 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return None
    rho, p_value = scipy_stats.spearmanr(frame["a"], frame["b"])
    return rho, p_value, abs(rho)


def p_text(p_value):
    if p_value < 0.001:
        return "p < 0,001"
    return "p = " + decimal_text(p_value)


def interpretation(p_value, effect_label):
    if p_value < ALPHA:
        text = "signifikan"
    else:
        text = "tidak signifikan"
    text = text + f" ({p_text(p_value)}), efek {effect_label}"
    if p_value < ALPHA and effect_label == EFFECT_NAMES[0]:
        text = text + ": hubungan nyata secara statistik tetapi terlalu lemah untuk berguna"
    return text


def related(interpretation_text):
    # True bila tafsir berbunyi "signifikan" dan efeknya bukan "diabaikan"
    text = interpretation_text
    return "tidak signifikan" not in text and "efek diabaikan" not in text


def relation_line(column, kind, test_name, effect_symbol, result, limits):
    statistic, p_value, effect = result
    effect_label = effect_name(effect, limits)
    return {"kolom": column, "jenis": kind, "uji": test_name, "statistik": round(statistic, 3),
            "p_value": p_value, "ukuran_efek": round(effect, 3), "nama_efek": effect_symbol,
            "efek": effect_label, "tafsir": interpretation(p_value, effect_label)}


def relation_with_feature(column, feature, label, is_number, regression):
    # pilih uji yang cocok: jenis kolom (angka / kategori) x jenis label (kelas / angka)
    if is_number and regression:
        result = spearman_test(feature, label)
        return result, ("angka", "Spearman", "rho", CORRELATION_LIMITS)
    if is_number:
        return kruskal_test(feature, label), ("angka", "Kruskal-Wallis", "epsilon2", EPSILON_LIMITS)
    if regression:
        return kruskal_test(label, feature), ("kategori", "Kruskal-Wallis", "epsilon2",
                                              EPSILON_LIMITS)
    return chi_square_test(feature, label), ("kategori", "chi-square", "Cramer V",
                                             CORRELATION_LIMITS)


def feature_columns(data, label_column, numbers, stats):
    # {nama: (nilai, apakah angka)}: kolom angka terpilih, panjang teks, dan kolom kategori
    features = {}
    for column in numbers.columns:
        features[column] = (numbers[column], True)
    features["teks_panjang"] = (stats["panjang"], True)
    features["teks_jumlah_kata"] = (stats["jumlah_kata"], True)
    for column in data.columns:
        if column == label_column or column in features:
            continue
        if is_category(data[column]):
            features[column] = (data[column], False)
    return features


def label_relations(data, label_column, numbers, stats):
    label = data[label_column]
    regression = is_regression_label(label)
    features = feature_columns(data, label_column, numbers, stats)
    lines = []
    for column in features:
        feature, is_number = features[column]
        result, info = relation_with_feature(column, feature, label, is_number, regression)
        if result is not None:
            lines.append(relation_line(column, info[0], info[1], info[2], result, info[3]))
    table = pd.DataFrame(lines)
    if len(table) > 0:
        table = table.sort_values("ukuran_efek", ascending=False)
    return table


# ---------------- korelasi antar kolom angka ----------------

def spearman_matrix(numbers):
    usable = []
    for column in numbers.columns:
        if numbers[column].nunique() > 1:
            usable.append(column)
    return numbers[usable].corr(method="spearman")


def redundant_pairs(matrix):
    lines = []
    names = list(matrix.columns)
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            rho = matrix.iloc[first, second]
            if abs(rho) > REDUNDANT_LIMIT:
                lines.append({"kolom_a": names[first], "kolom_b": names[second],
                              "rho": round(rho, 3)})
    return pd.DataFrame(lines, columns=["kolom_a", "kolom_b", "rho"])


def correlation_long(matrix):
    lines = []
    for first in matrix.columns:
        for second in matrix.columns:
            lines.append({"kolom_a": first, "kolom_b": second, "rho": matrix.at[first, second]})
    return pd.DataFrame(lines)


# ---------------- pola data kosong ----------------

def missing_vs_label(missing, label):
    # apakah baris kosong punya sebaran label yang berbeda dari baris terisi?
    result = chi_square_test(missing, label)
    if result is None:
        return {}
    _, p_value, cramers_v = result
    shares = pd.crosstab(missing, label, normalize="index")
    top_empty = shares.loc[True]
    effect_label = effect_name(cramers_v, CORRELATION_LIMITS)
    return {"p_value": p_value, "cramers_v": round(cramers_v, 3),
            "label_terbanyak_saat_kosong": f"{top_empty.idxmax()} ({100 * top_empty.max():.0f}%)",
            "tafsir": interpretation(p_value, effect_label)}


def missing_table(data, label_column):
    lines = []
    for column in data.columns:
        missing = data[column].isna()
        if missing.sum() == 0:
            continue
        line = {"kolom": column, "kosong": int(missing.sum()),
                "persen_kosong": round(100 * missing.mean(), 2)}
        if label_column is not None and column != label_column:
            line.update(missing_vs_label(missing, data[label_column]))
        lines.append(line)
    return pd.DataFrame(lines)


def missing_together(data):
    # pasangan kolom yang kosong pada baris yang sama (korelasi penanda kosong > 0,95)
    flags = data.isna()
    usable = []
    for column in flags.columns:
        if 0 < flags[column].sum() < len(flags):
            usable.append(column)
    lines = []
    corr = flags[usable].astype(int).corr()
    for first in range(len(usable)):
        for second in range(first + 1, len(usable)):
            if corr.iloc[first, second] > 0.95:
                lines.append({"kolom_a": usable[first], "kolom_b": usable[second],
                              "korelasi_kosong": round(corr.iloc[first, second], 3)})
    return pd.DataFrame(lines, columns=["kolom_a", "kolom_b", "korelasi_kosong"])


# ---------------- waktu ----------------

def parse_dates(series):
    return pd.to_datetime(series, errors="coerce", format="mixed")


def monthly_counts(dates):
    months = dates.dropna().dt.to_period("M")
    counts = months.value_counts().sort_index()
    counts.index = counts.index.astype(str)
    return counts


def label_by_month(dates, label):
    frame = pd.DataFrame({"bulan": dates.dt.to_period("M").astype(str), "label": label})
    frame = frame[dates.notna() & label.notna()]
    shares = pd.crosstab(frame["bulan"], frame["label"], normalize="index")
    return (100 * shares).round(1)


def drift_test(dates, label):
    # bagi baris berdasarkan urutan waktu: separuh awal vs separuh akhir; sebaran label berubah?
    frame = pd.DataFrame({"tanggal": dates, "label": label}).dropna()
    frame = frame.sort_values("tanggal")
    half = len(frame) // 2
    frame["bagian"] = "awal"
    frame.iloc[half:, frame.columns.get_loc("bagian")] = "akhir"
    result = chi_square_test(frame["bagian"], frame["label"])
    if result is None:
        return None
    _, p_value, cramers_v = result
    shares = pd.crosstab(frame["label"], frame["bagian"], normalize="columns")
    change = (shares["akhir"] - shares["awal"]) * 100
    biggest = change.abs().idxmax()
    return {"p_value": p_value, "cramers_v": cramers_v, "label": biggest,
            "perubahan_poin": change[biggest], "awal": shares.loc[biggest, "awal"] * 100,
            "akhir": shares.loc[biggest, "akhir"] * 100}
