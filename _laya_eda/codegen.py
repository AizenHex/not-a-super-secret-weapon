"""Pembuat kode EDA otomatis: dari pilihan dan temuan di Laya EDA, tulis notebook bergaya
ATURAN_KODE (sel # %%) yang bisa langsung ditempel ke notebook lomba. Hanya pandas + numpy:
tanpa Laya, tanpa helpers.
Baris yang dibuang dipilih lewat ATURAN pandas (loop per kolom), bukan daftar nomor baris;
hanya keputusan manual dari tab Telusuri yang ditulis per baris (bersama catatannya).

config (dict):
    csv_path, test_csv (atau None), columns, label_column (atau None),
    number_columns, indonesian_format, use_log, ratio (pasangan [atas, bawah] atau None),
    manual_drop (dict nomor baris -> catatan),
    new_labels (dict nomor baris -> label baru), drop_rules (list, kunci RULES),
    outlier_columns (list), outlier_category (atau None), features (list, kunci FEATURES),
    category_columns (list, untuk frekuensi), date_columns (list, untuk tanggal)
"""
import datetime
import pprint
import textwrap

import nbformat

# aturan buang yang bisa dipilih di tab Kode EDA
RULES = {
    "duplikat": "duplikat persis (isi kolom teks sama)",
    "teks_kosong": "semua kolom teks kosong",
    "tidak_terbaca": "angka tidak terbaca (mis. 'hubungi wa')",
    "negatif": "angka negatif atau nol",
    "outlier": "outlier per kategori (IQR)",
}

# fitur yang bisa ditulis ulang sebagai kode pandas
FEATURES = {
    "gaya": "statistik gaya teks (panjang, kata, kapital, !, ?, angka)",
    "negasi": "penanda negasi (tidak, bukan, gak, ...)",
    "panjang_kolom": "panjang teks tiap kolom",
    "kosong": "penanda nilai kosong tiap kolom",
    "log_angka": "log kolom angka",
    "rasio_median": "angka ÷ median kategorinya",
    "frekuensi": "frequency encoding kolom kategori",
    "tanggal": "bagian tanggal (tahun, bulan, hari, jam)",
}


def literal(value):
    # tulis nilai Python sebagai teks kode yang valid (dict/list dirapikan, lebar ≤ 90)
    return pprint.pformat(value, width=90, sort_dicts=False)


def markdown(text):
    return ("markdown", text)


def code(text):
    return ("code", text)


def uses_outlier(config):
    if config["outlier_category"] is None:
        return False
    return "outlier" in config["drop_rules"] and len(config["outlier_columns"]) > 0


def feature_category(config):
    # 'angka ÷ median kategori' butuh kategori yang ada juga di test (bukan label)
    category = config["outlier_category"]
    if category is None or category == config["label_column"]:
        return None
    return category


def usable_features(config):
    # buang pilihan fitur yang bahannya tidak ada
    chosen = []
    for name in config["features"]:
        if name in ["log_angka", "rasio_median"] and len(config["number_columns"]) == 0:
            continue
        if name == "rasio_median" and feature_category(config) is None:
            continue
        if name == "frekuensi" and len(config["category_columns"]) == 0:
            continue
        if name == "tanggal" and len(config["date_columns"]) == 0:
            continue
        chosen.append(name)
    return chosen


def intro_cells(config):
    today = datetime.date.today().isoformat()
    steps = ["baca data"]
    if len(config["new_labels"]) > 0:
        steps.append("ganti label")
    if len(config["number_columns"]) > 0:
        steps.append("ubah kolom angka")
    steps.append("tandai lalu buang baris menurut aturan")
    if len(usable_features(config)) > 0:
        steps.append("tambah fitur")
    steps.append("simpan")
    title = (f"# EDA dari Laya EDA ({today})\n"
             f"Dibuat otomatis dari temuan di `_laya_eda`. Langkah: {', '.join(steps)}.\n\n"
             "Jalankan dari atas. Ubah nilai di sel **Konfigurasi** bila path atau kolom beda.")
    imports = "import numpy as np\nimport pandas as pd"
    return [markdown(title), code(imports)]


def outlier_names(config):
    names = []
    for number in config["outlier_columns"]:
        if number in config["number_columns"]:
            names.append(f"{number}_angka")
        else:
            names.append(number)
    return names


def config_cells(config):
    features = usable_features(config)
    lines = [f'TRAIN_CSV = "{config["csv_path"]}"   # GANTI: path relatif dari folder notebook']
    if config["test_csv"] is not None:
        lines.append(f'TEST_CSV = "{config["test_csv"]}"')
    lines.append(f"TEXT_COLUMNS = {literal(config['columns'])}")
    if config["label_column"] is not None:
        lines.append(f'LABEL = "{config["label_column"]}"')
    if len(config["number_columns"]) > 0:
        lines.append(f"NUMBER_COLUMNS = {literal(config['number_columns'])}")
        lines.append(f"INDONESIAN_FORMAT = {config['indonesian_format']}   # titik = ribuan")
    if uses_outlier(config) or "rasio_median" in features:
        lines.append(f'CATEGORY = "{config["outlier_category"]}"   # kategori pembanding')
    if uses_outlier(config):
        lines.append(f"OUTLIER_COLUMNS = {literal(outlier_names(config))}")
    if "frekuensi" in features:
        lines.append(f"CATEGORY_COLUMNS = {literal(config['category_columns'])}")
    if "tanggal" in features:
        lines.append(f"DATE_COLUMNS = {literal(config['date_columns'])}")
    lines.append('OUTPUT_TRAIN = "train_bersih.csv"')
    if config["test_csv"] is not None:
        lines.append('OUTPUT_TEST = "test_bersih.csv"')
    return [markdown("## Konfigurasi"), code("\n".join(lines))]


def read_cells(config):
    text = "train = pd.read_csv(TRAIN_CSV)\nprint(\"train:\", train.shape)"
    if config["test_csv"] is not None:
        text = text + "\ntest = pd.read_csv(TEST_CSV)\nprint(\"test:\", test.shape)"
    text = text + "\ntrain.head()"
    return [markdown("## Baca data"), code(text)]


def label_cells(config):
    if len(config["new_labels"]) == 0 or config["label_column"] is None:
        return []
    explain = ("## Ganti label (keputusan manual)\n"
               "Label yang kamu putuskan salah di tab *Telusuri + catatan*. Nomor baris = urutan "
               "baris di CSV asli, jadi sel ini dijalankan sebelum ada baris yang dibuang.")
    lines = [f"NEW_LABELS = {literal(config['new_labels'])}",
             "",
             "for row in NEW_LABELS:",
             "    print(row, train.loc[row, LABEL], \"->\", NEW_LABELS[row])",
             "    train.loc[row, LABEL] = NEW_LABELS[row]"]
    return [markdown(explain), code("\n".join(lines))]


def number_cells(config):
    if len(config["number_columns"]) == 0:
        return []
    explain = ("## Ubah kolom angka\n"
               "Teks seperti `Rp 1.250.000` atau `1.500.000/bulan` diubah jadi angka. Yang tidak "
               "berisi angka (`hubungi wa`) menjadi kosong (NaN).")
    text = '''def to_number(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    text = series.astype(str).str.lower()
    text = text.str.replace("rp", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    if INDONESIAN_FORMAT:
        text = text.str.replace(".", "", regex=False)    # titik = pemisah ribuan
        text = text.str.replace(",", ".", regex=False)   # koma = desimal
    else:
        text = text.str.replace(",", "", regex=False)
    number_text = text.str.extract(r"^(-?\\d+(?:\\.\\d+)?)")[0]
    return pd.to_numeric(number_text, errors="coerce")


# coba: sekitar 1250000 dan NaN
print(to_number(pd.Series(["Rp 1.250.000", "hubungi wa"])).tolist())

for column in NUMBER_COLUMNS:
    train[f"{column}_angka"] = to_number(train[column])
    print(column, "tidak terbaca:", train[f"{column}_angka"].isna().sum())'''
    if config["test_csv"] is not None:
        text = text + "\n    test[f\"{column}_angka\"] = to_number(test[column])"
    ratio = config["ratio"]
    if ratio is not None:
        name = f"{ratio[0]}_per_{ratio[1]}"
        tables = ["train"]
        if config["test_csv"] is not None:
            tables.append("test")
        text = text + "\n"
        for table in tables:
            top = f'{table}["{ratio[0]}_angka"]'
            text = text + f'\nbottom = {table}["{ratio[1]}_angka"]'
            text = text + f'\n{table}["{name}"] = {top} / bottom.where(bottom > 0)'
    return [markdown(explain), code(text)]


def basic_rule_lines(config):
    lines = []
    if len(config["manual_drop"]) > 0:
        lines = lines + ["# keputusan 'buang' dari tab Telusuri: nomor baris CSV asli -> catatan",
                         f"MANUAL_DROP = {literal(config['manual_drop'])}",
                         "manual = train.index.isin(list(MANUAL_DROP))",
                         "print(\"keputusan manual:\", manual.sum())",
                         "drop = drop | manual", ""]
    if "duplikat" in config["drop_rules"]:
        lines = lines + ["duplicate = train.duplicated(subset=TEXT_COLUMNS, keep=\"first\")",
                         "print(\"duplikat persis:\", duplicate.sum())",
                         "drop = drop | duplicate", ""]
    if "teks_kosong" in config["drop_rules"]:
        lines = lines + ["empty = pd.Series(True, index=train.index)",
                         "for column in TEXT_COLUMNS:",
                         "    text = train[column].fillna(\"\").astype(str)",
                         "    empty = empty & (text.str.strip() == \"\")",
                         "print(\"teks kosong:\", empty.sum())",
                         "drop = drop | empty", ""]
    return lines


def number_rule_lines(config):
    lines = []
    if len(config["number_columns"]) == 0:
        return lines
    if "tidak_terbaca" in config["drop_rules"]:
        lines = lines + ["for column in NUMBER_COLUMNS:",
                         "    unreadable = train[f\"{column}_angka\"].isna()",
                         "    print(column, \"tidak terbaca:\", unreadable.sum())",
                         "    drop = drop | unreadable", ""]
    if "negatif" in config["drop_rules"]:
        lines = lines + ["for column in NUMBER_COLUMNS:",
                         "    not_positive = train[f\"{column}_angka\"] <= 0",
                         "    print(column, \"negatif / nol:\", not_positive.sum())",
                         "    drop = drop | not_positive", ""]
    return lines


def rule_cells(config):
    explain = ("## Tandai baris yang dibuang\n"
               "Tiap aturan menghasilkan kolom benar/salah dari pandas, lalu digabung ke `drop` "
               "dengan `|` (atau). Baris belum dibuang di sini, jadi jumlah tiap aturan bisa "
               "dibandingkan dulu. Hapus blok aturan yang tidak kamu setujui.")
    lines = ["drop = pd.Series(False, index=train.index)", ""]
    lines = lines + basic_rule_lines(config) + number_rule_lines(config)
    lines.append("print(\"sementara ditandai:\", drop.sum())")
    return [markdown(explain), code("\n".join(lines))]


def outlier_cells(config):
    if not uses_outlier(config):
        return []
    scale = ""
    if config["use_log"]:
        scale = ", pada skala log10 karena angka seperti harga menyebar secara perkalian"
    explain = ("## Outlier per kategori\n"
               "Batas wajar dihitung **di dalam tiap kategori** (Q1 − 1,5×IQR sampai "
               f"Q3 + 1,5×IQR{scale}). Nilai yang wajar secara keseluruhan bisa aneh untuk "
               "jenisnya sendiri. Kolom `outlier_<kolom>` disimpan supaya bisa dicek.")
    lines = ["for column in OUTLIER_COLUMNS:", "    values = train[column]"]
    if config["use_log"]:
        lines.append("    values = np.log10(values.where(values > 0))")
    lines = lines + ["    grouped = values.groupby(train[CATEGORY])",
                     "    q1 = grouped.transform(\"quantile\", 0.25)",
                     "    q3 = grouped.transform(\"quantile\", 0.75)",
                     "    spread = 1.5 * (q3 - q1)",
                     "    outlier = (values < q1 - spread) | (values > q3 + spread)",
                     "    train[f\"outlier_{column}\"] = outlier",
                     "    print(column, \"outlier:\", outlier.sum())",
                     "    drop = drop | outlier",
                     "",
                     "print(\"sementara ditandai:\", drop.sum())"]
    return [markdown(explain), code("\n".join(lines))]


def drop_cells(config):
    look = "print(\"dibuang:\", drop.sum(), \"dari\", len(train), \"baris\")\ntrain[drop].head(10)"
    finish = "train = train[~drop]\nprint(\"train sekarang:\", train.shape)"
    return [markdown("## Buang baris yang ditandai\nLihat dulu contohnya, baru buang."),
            code(look), code(finish)]


def text_feature_lines(features):
    lines = []
    if "gaya" in features or "negasi" in features:
        lines = lines + ["    texts = table[TEXT_COLUMNS[0]].fillna(\"\").astype(str)",
                         "    for column in TEXT_COLUMNS[1:]:",
                         "        texts = texts + \" \" + table[column].fillna(\"\").astype(str)"]
    if "gaya" in features:
        lines = lines + ["    table[\"panjang\"] = texts.str.len()",
                         "    table[\"jumlah_kata\"] = texts.str.split().str.len()",
                         "    for name in STYLE_PATTERNS:",
                         "        table[name] = texts.str.count(STYLE_PATTERNS[name])",
                         "    table[\"rasio_kapital\"] = table[\"kapital\"] / "
                         "table[\"panjang\"].clip(lower=1)"]
    if "negasi" in features:
        lines = lines + ["    lower = texts.str.lower()",
                         "    table[\"jumlah_negasi\"] = lower.str.count(NEGATION)"]
    if "panjang_kolom" in features:
        lines = lines + ["    for column in TEXT_COLUMNS:",
                         "        values = table[column].fillna(\"\").astype(str)",
                         "        table[f\"panjang_{column}\"] = values.str.len()"]
    if "kosong" in features:
        lines = lines + ["    for column in TEXT_COLUMNS:",
                         "        table[f\"kosong_{column}\"] = table[column].isna().astype(int)"]
    return lines


def table_feature_lines(features):
    lines = []
    if "log_angka" in features:
        lines = lines + ["    for column in NUMBER_COLUMNS:",
                         "        values = table[f\"{column}_angka\"].clip(lower=0)",
                         "        table[f\"log_{column}\"] = np.log1p(values)"]
    if "rasio_median" in features:
        lines = lines + ["    for column in NUMBER_COLUMNS:",
                         "        medians = train.groupby(CATEGORY)[f\"{column}_angka\"].median()",
                         "        category_median = table[CATEGORY].map(medians)",
                         "        table[f\"rasio_{column}\"] = table[f\"{column}_angka\"] / "
                         "category_median"]
    if "frekuensi" in features:
        lines = lines + ["    for column in CATEGORY_COLUMNS:",
                         "        shares = train[column].value_counts(normalize=True)",
                         "        table[f\"frek_{column}\"] = table[column].map(shares).fillna(0)"]
    if "tanggal" in features:
        lines = lines + ["    for column in DATE_COLUMNS:",
                         "        dates = pd.to_datetime(table[column], errors=\"coerce\")",
                         "        table[f\"{column}_tahun\"] = dates.dt.year",
                         "        table[f\"{column}_bulan\"] = dates.dt.month",
                         "        table[f\"{column}_hari\"] = dates.dt.dayofweek",
                         "        table[f\"{column}_jam\"] = dates.dt.hour"]
    return lines


def feature_cells(config):
    features = usable_features(config)
    if len(features) == 0:
        return []
    explain = ("## Tambah fitur\n"
               f"Fitur yang dipilih: {', '.join(features)}. Dihitung dengan cara yang sama untuk "
               "train dan test; statistik (median, frekuensi) diambil dari train saja supaya "
               "test tidak bocor.")
    lines = []
    if "gaya" in features:
        lines.append("STYLE_PATTERNS = {\"kapital\": r\"[A-Z]\", \"tanda_seru\": \"!\", "
                     "\"tanda_tanya\": r\"\\?\", \"angka\": r\"\\d+\"}")
    if "negasi" in features:
        lines.append("NEGATION = r\"\\b(tidak|bukan|gak|nggak|ga|tak|jangan)\\b\"")
    tables = "[train]"
    if config["test_csv"] is not None:
        tables = "[train, test]"    # fitur dihitung dengan cara yang sama di train dan test
    lines.append(f"for table in {tables}:")
    lines = lines + text_feature_lines(features) + table_feature_lines(features)
    lines.append("print(\"kolom train sekarang:\", train.shape[1])")
    lines.append("train.head()")
    return [markdown(explain), code("\n".join(lines))]


def save_cells(config):
    text = 'train.to_csv(OUTPUT_TRAIN, index=False)\nprint("disimpan:", OUTPUT_TRAIN, train.shape)'
    if config["test_csv"] is not None:
        text = text + '\ntest.to_csv(OUTPUT_TEST, index=False)\nprint("disimpan:", OUTPUT_TEST)'
    return [markdown("## Simpan"), code(text)]


def build_cells(config):
    cells = intro_cells(config)
    cells = cells + config_cells(config)
    cells = cells + read_cells(config)
    cells = cells + label_cells(config)
    cells = cells + number_cells(config)
    cells = cells + rule_cells(config)
    cells = cells + outlier_cells(config)
    cells = cells + drop_cells(config)
    cells = cells + feature_cells(config)
    cells = cells + save_cells(config)
    return cells


def to_script(cells):
    # format # %% yang dipakai seluruh toolkit (_alat/nb/*.py)
    parts = []
    for kind, text in cells:
        if kind == "markdown":
            commented = []
            for line in text.splitlines():
                pieces = textwrap.wrap(line, width=96)    # baris markdown panjang dipotong
                if len(pieces) == 0:
                    pieces = [""]
                for piece in pieces:
                    commented.append(("# " + piece).rstrip())
            parts.append("# %% [markdown]\n" + "\n".join(commented))
        else:
            parts.append("# %%\n" + text)
    return "\n\n".join(parts) + "\n"


def to_notebook(cells):
    notebook = nbformat.v4.new_notebook()
    for kind, text in cells:
        if kind == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            notebook.cells.append(nbformat.v4.new_code_cell(text))
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                       "name": "python3"}
    return nbformat.writes(notebook)
