"""Pembuat Kode EDA: langkah yang dipilih di tab Kode EDA ditulis jadi notebook bergaya
ATURAN_KODE (sel # %%) yang bisa langsung ditempel. Hanya pandas + numpy: tanpa Laya, tanpa
helpers. Nama kolom asli tidak pernah diubah (kolom angka diisi angka di kolom yang sama).

Langkah yang mengubah isi data dijalankan untuk train DAN test dengan nilai dari train (kolom
angka, penanda kosong, isi angka kosong, fitur, buang kolom), jadi bisa dipakai lagi saat
inferensi. Ganti label dan buang baris hanya untuk train.
Aplikasi menjalankan sel yang sama (app_script + exec) untuk membuat train_siap, jadi isi file
di tab Ekspor = hasil kode ini.

config (dict):
    csv_path, test_csv (atau None), columns, label_column (atau None),
    number_columns, indonesian_format, use_log, ratio (pasangan [atas, bawah] atau None),
    manual_drop (dict nomor baris -> catatan), new_labels (dict nomor baris -> label baru),
    drop_rules (list, kunci RULES), query_rules (list aturan pandas query),
    outlier_columns (list), category (atau None), fill (dict kolom -> cara isi),
    features (list, kunci FEATURES), number_features (list), category_columns (list),
    date_columns (list), drop_columns (list), split_lines (list baris kode atau None),
    encoding (dict kolom -> cara, kunci ENCODE_FUNCTIONS), encode_drop (bool),
    label_is_number (bool), minilm_path
"""
import datetime
import pprint
import textwrap

import nbformat

# aturan buang baris (train saja) yang bisa dipilih di tab Kode EDA
RULES = {
    "duplikat": "duplikat persis (isi kolom teks sama)",
    "teks_kosong": "semua kolom teks kosong",
    "tidak_terbaca": "angka kosong / tidak terbaca (mis. 'hubungi wa')",
    "negatif": "angka negatif atau nol",
    "outlier": "outlier per kategori (IQR)",
}

# fitur yang bisa ditulis ulang sebagai kode pandas (dihitung sama untuk train dan test)
FEATURES = {
    "gaya": "statistik gaya teks (panjang, kata, kapital, !, ?, angka)",
    "negasi": "penanda negasi (tidak, bukan, gak, ...)",
    "panjang_kolom": "panjang teks tiap kolom",
    "kosong": "penanda nilai kosong tiap kolom (sebelum diisi)",
    "log_angka": "log kolom angka",
    "rasio_median": "angka ÷ median kategorinya",
    "frekuensi": "frequency encoding kolom kategori",
    "tanggal": "bagian tanggal (tahun, bulan, hari, jam)",
}

# encoding kolom kategori: cara -> fungsi yang ditulis ke notebook (hanya yang dipakai)
ENCODE_FUNCTIONS = {
    "ordinal": '''def encode_ordinal(column, tables):
    # kode 0, 1, 2, ... urut dari nilai tersering di train; nilai yang tidak ada di train = -1
    counts = train[column].value_counts()
    codes = pd.Series(np.arange(len(counts)), index=counts.index)
    for table in tables:
        table[f"kode_{column}"] = table[column].map(codes).fillna(-1).astype(int)''',
    "one_hot": '''def encode_one_hot(column, tables):
    # satu kolom 0/1 per nilai tersering di train (maks ONE_HOT_MAX); nilai lain = semua 0
    for value in train[column].value_counts().index[:ONE_HOT_MAX]:
        for table in tables:
            table[f"oh_{column}_{value}"] = (table[column] == value).astype(int)''',
    "target": '''def target_shares(column):
    # rata-rata label (angka) atau porsi tiap kelas per nilai kategori, dari train
    if LABEL_IS_NUMBER:
        shares = train.groupby(column)[LABEL].mean().to_frame("rata2")
        prior = pd.Series({"rata2": train[LABEL].mean()})
        return shares, prior
    shares = pd.crosstab(train[column], train[LABEL], normalize="index")
    return shares, train[LABEL].value_counts(normalize=True)


def encode_target(column, tables):
    # nilai jarang ditarik ke rata-rata umum: bobot = jumlah / (jumlah + TARGET_SMOOTHING)
    shares, prior = target_shares(column)
    counts = train[column].value_counts()
    weight = counts / (counts + TARGET_SMOOTHING)
    for name in shares.columns:
        smooth = shares[name] * weight + prior[name] * (1 - weight)
        for table in tables:
            table[f"target_{column}_{name}"] = table[column].map(smooth).fillna(prior[name])''',
    "makna": '''def meaning_vectors(values):
    # vektor makna MiniLM (panjang 1) untuk tiap nilai kategori sebagai teks
    texts = []
    for value in values:
        texts.append(str(value))
    if len(texts) == 0:
        return np.zeros((0, MINILM.get_sentence_embedding_dimension()))
    return MINILM.encode(texts, batch_size=256, normalize_embeddings=True)


def encode_meaning(column, tables):
    # sumbu PCA dari nilai di train; nilai baru di test tetap dapat angka yang masuk akal
    # karena maknanya dibaca MiniLM, bukan dihafal. Kosong = 0 (makna rata-rata)
    known = train[column].dropna().unique()
    vectors = meaning_vectors(known)
    center = vectors.mean(axis=0)
    axes = np.linalg.svd(vectors - center, full_matrices=False)[2]
    axes = axes[:max(1, min(MEANING_SIZE, len(known) - 1))]
    for table in tables:
        values = table[column].dropna().unique()
        points = (meaning_vectors(values) - center) @ axes.T
        for number in range(len(axes)):
            mapping = pd.Series(points[:, number], index=values)
            table[f"makna_{column}_{number + 1}"] = table[column].map(mapping).fillna(0)''',
}
ENCODE_NAMES = {"ordinal": "encode_ordinal", "one_hot": "encode_one_hot",
                "target": "encode_target", "makna": "encode_meaning"}
ENCODE_SETTINGS = {"one_hot": "ONE_HOT_MAX = 30   # nilai tersering yang dapat kolom sendiri",
                   "target": "TARGET_SMOOTHING = 10   # makin besar, nilai jarang makin ditarik",
                   "makna": "MEANING_SIZE = 8   # banyak angka makna per kolom"}


def literal(value):
    # tulis nilai Python sebagai teks kode yang valid (dict/list dirapikan, lebar ≤ 90)
    return pprint.pformat(value, width=90, sort_dicts=False)


def markdown(text):
    return ("markdown", text, None)


def code(text, role=None):
    # role "read" / "model" / "split" / "save": tidak dijalankan aplikasi (aplikasi memberi
    # train/test/MINILM sendiri dan menulis file di tab Ekspor)
    return ("code", text, role)


def has_test(config):
    return config["test_csv"] is not None


def table_list(config):
    if has_test(config):
        return "[train, test]"    # dihitung dengan cara yang sama di train dan test
    return "[train]"


def text_columns(config):
    # kolom teks tanpa label (label tidak ada di test)
    columns = []
    for column in config["columns"]:
        if column != config["label_column"]:
            columns.append(column)
    return columns


def uses_outlier(config):
    if config["category"] is None:
        return False
    return "outlier" in config["drop_rules"] and len(config["outlier_columns"]) > 0


def feature_category(config):
    # 'angka ÷ median kategori' butuh kategori yang ada juga di test (bukan label)
    category = config["category"]
    if category is None or category == config["label_column"]:
        return None
    return category


def usable_features(config):
    # buang pilihan fitur yang bahannya tidak ada
    chosen = []
    for name in config["features"]:
        if name in ["log_angka", "rasio_median"] and len(config["number_features"]) == 0:
            continue
        if name == "rasio_median" and feature_category(config) is None:
            continue
        if name == "frekuensi" and len(config["category_columns"]) == 0:
            continue
        if name == "tanggal" and len(config["date_columns"]) == 0:
            continue
        chosen.append(name)
    return chosen


def step_names(config):
    steps = ["baca data"]
    if len(config["new_labels"]) > 0:
        steps.append("ganti label")
    if len(config["number_columns"]) > 0:
        steps.append("ubah kolom angka (nama tetap)")
    steps.append("buang baris (train)")
    if len(config["fill"]) > 0:
        steps.append("isi angka kosong")
    if len(usable_features(config)) > 0:
        steps.append("tambah fitur")
    if len(config["encoding"]) > 0:
        steps.append("encoding kategori")
    if len(config["drop_columns"]) > 0:
        steps.append("buang kolom")
    if config["split_lines"] is not None:
        steps.append("split validasi")
    return steps + ["simpan"]


def intro_cells(config):
    today = datetime.date.today().isoformat()
    title = (f"# Kode EDA dari Laya EDA ({today})\n"
             f"Langkah: {', '.join(step_names(config))}.\n\n"
             "Sel bertanda **(train + test)** wajib dijalankan juga untuk data test saat "
             "inferensi, dengan nilai yang dihitung dari train. Sel **(train)** hanya untuk "
             "train. Nama kolom asli tidak diubah. Ubah path di sel **Konfigurasi** bila beda.")
    return [markdown(title), code("import numpy as np\nimport pandas as pd")]


def column_config_lines(config):
    features = usable_features(config)
    lines = [f"TEXT_COLUMNS = {literal(text_columns(config))}"]
    if config["label_column"] is not None:
        lines.append(f'LABEL = "{config["label_column"]}"')
    if len(config["number_columns"]) > 0:
        lines.append(f"NUMBER_COLUMNS = {literal(config['number_columns'])}")
        lines.append(f"INDONESIAN_FORMAT = {config['indonesian_format']}   # titik = ribuan")
    if uses_outlier(config) or "rasio_median" in features:
        lines.append(f'CATEGORY = "{config["category"]}"   # kategori pembanding')
    if uses_outlier(config):
        lines.append(f"OUTLIER_COLUMNS = {literal(config['outlier_columns'])}")
    if len(config["query_rules"]) > 0:
        lines.append(f"DROP_QUERIES = {literal(config['query_rules'])}")
    return lines


def table_config_lines(config):
    features = usable_features(config)
    lines = []
    if len(config["fill"]) > 0:
        lines.append(f"FILL = {literal(config['fill'])}   # kolom -> cara isi")
    if "log_angka" in features or "rasio_median" in features:
        lines.append(f"NUMBER_FEATURES = {literal(config['number_features'])}")
    if "frekuensi" in features:
        lines.append(f"CATEGORY_COLUMNS = {literal(config['category_columns'])}")
    if "tanggal" in features:
        lines.append(f"DATE_COLUMNS = {literal(config['date_columns'])}")
    if len(config["encoding"]) > 0:
        lines.append(f"ENCODING = {literal(config['encoding'])}   # kolom -> cara encoding")
    if "makna" in config["encoding"].values():
        lines.append(f'MINILM_PATH = "{config["minilm_path"]}"   # folder model MiniLM')
    if len(config["drop_columns"]) > 0:
        lines.append(f"DROP_COLUMNS = {literal(config['drop_columns'])}")
    return lines


def config_cells(config):
    lines = [f'TRAIN_CSV = "{config["csv_path"]}"   # GANTI: path relatif dari folder notebook']
    if has_test(config):
        lines.append(f'TEST_CSV = "{config["test_csv"]}"')
    lines = lines + column_config_lines(config) + table_config_lines(config)
    lines.append('OUTPUT_TRAIN = "train_siap.csv"')
    if has_test(config):
        lines.append('OUTPUT_TEST = "test_siap.csv"')
    return [markdown("## Konfigurasi"), code("\n".join(lines))]


def read_cells(config):
    text = "train = pd.read_csv(TRAIN_CSV)\nprint(\"train:\", train.shape)"
    if has_test(config):
        text = text + "\ntest = pd.read_csv(TEST_CSV)\nprint(\"test:\", test.shape)"
    return [markdown("## Baca data"), code(text + "\ntrain.head()", "read")]


def label_cells(config):
    if len(config["new_labels"]) == 0 or config["label_column"] is None:
        return []
    explain = ("## Ganti label (train)\n"
               "Label yang kamu putuskan salah di tab *Telusuri + catatan*. Nomor baris = urutan "
               "baris di CSV asli, jadi sel ini dijalankan sebelum ada baris yang dibuang.")
    lines = [f"NEW_LABELS = {literal(config['new_labels'])}",
             "",
             "for row in NEW_LABELS:",
             "    print(row, train.loc[row, LABEL], \"->\", NEW_LABELS[row])",
             "    train.loc[row, LABEL] = NEW_LABELS[row]"]
    return [markdown(explain), code("\n".join(lines))]


NUMBER_FUNCTION = '''def to_number(series):
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
    train[column] = to_number(train[column])    # nama kolom tetap, isinya jadi angka
    print(column, "kosong / tidak terbaca:", train[column].isna().sum())'''


def ratio_lines(config):
    top, bottom = config["ratio"]
    name = f"{top}_per_{bottom}"
    lines = [f"for table in {table_list(config)}:",
             f'    bottom = table["{bottom}"]',
             f'    table["{name}"] = table["{top}"] / bottom.where(bottom > 0)']
    return lines


def number_cells(config):
    if len(config["number_columns"]) == 0:
        return []
    explain = ("## Ubah kolom angka (train + test)\n"
               "Teks seperti `Rp 1.250.000` atau `1.500.000/bulan` diubah jadi angka **di kolom "
               "yang sama** (nama kolom tidak berubah). Yang tidak berisi angka (`hubungi wa`) "
               "menjadi kosong (NaN).")
    lines = [NUMBER_FUNCTION]
    if has_test(config):
        lines = lines + ["", "for column in NUMBER_COLUMNS:",
                         "    if column in test.columns:    # kolom label tidak ada di test",
                         "        test[column] = to_number(test[column])"]
    if config["ratio"] is not None:
        lines = lines + [""] + ratio_lines(config)
    return [markdown(explain), code("\n".join(lines))]


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
                         "    unreadable = train[column].isna()",
                         "    print(column, \"kosong / tidak terbaca:\", unreadable.sum())",
                         "    drop = drop | unreadable", ""]
    if "negatif" in config["drop_rules"]:
        lines = lines + ["for column in NUMBER_COLUMNS:",
                         "    not_positive = train[column] <= 0",
                         "    print(column, \"negatif / nol:\", not_positive.sum())",
                         "    drop = drop | not_positive", ""]
    return lines


def query_rule_lines(config):
    if len(config["query_rules"]) == 0:
        return []
    return ["# aturan buang kustom (pandas query)",
            "for rule in DROP_QUERIES:",
            "    matched = train.index.isin(train.query(rule, engine=\"python\").index)",
            "    print(rule, \":\", matched.sum())",
            "    drop = drop | matched", ""]


def rule_cells(config):
    explain = ("## Tandai baris yang dibuang (train)\n"
               "Tiap aturan menghasilkan kolom benar/salah dari pandas, lalu digabung ke `drop` "
               "dengan `|` (atau). Baris belum dibuang di sini, jadi jumlah tiap aturan bisa "
               "dibandingkan dulu. Hapus blok aturan yang tidak kamu setujui. Test tidak "
               "pernah dibuang barisnya.")
    lines = ["drop = pd.Series(False, index=train.index)", ""]
    lines = lines + basic_rule_lines(config) + number_rule_lines(config)
    lines = lines + query_rule_lines(config)
    lines.append("print(\"sementara ditandai:\", drop.sum())")
    return [markdown(explain), code("\n".join(lines))]


def outlier_cells(config):
    if not uses_outlier(config):
        return []
    scale = ""
    if config["use_log"]:
        scale = ", pada skala log10 karena angka seperti harga menyebar secara perkalian"
    explain = ("## Outlier per kategori (train)\n"
               "Batas wajar dihitung **di dalam tiap kategori** (Q1 − 1,5×IQR sampai "
               f"Q3 + 1,5×IQR{scale}). Nilai yang wajar secara keseluruhan bisa aneh untuk "
               "jenisnya sendiri.")
    lines = ["for column in OUTLIER_COLUMNS:", "    values = train[column]"]
    if config["use_log"]:
        lines.append("    values = np.log10(values.where(values > 0))")
    lines = lines + ["    grouped = values.groupby(train[CATEGORY])",
                     "    q1 = grouped.transform(\"quantile\", 0.25)",
                     "    q3 = grouped.transform(\"quantile\", 0.75)",
                     "    spread = 1.5 * (q3 - q1)",
                     "    outlier = (values < q1 - spread) | (values > q3 + spread)",
                     "    print(column, \"outlier:\", outlier.sum())",
                     "    drop = drop | outlier",
                     "",
                     "print(\"sementara ditandai:\", drop.sum())"]
    return [markdown(explain), code("\n".join(lines))]


def drop_cells(config):
    look = "print(\"dibuang:\", drop.sum(), \"dari\", len(train), \"baris\")\ntrain[drop].head(10)"
    finish = "train = train[~drop].copy()\nprint(\"train sekarang:\", train.shape)"
    return [markdown("## Buang baris yang ditandai (train)\nLihat dulu contohnya, baru buang."),
            code(look), code(finish)]


def missing_cells(config):
    # penanda kosong dihitung SEBELUM angka kosong diisi
    if "kosong" not in usable_features(config):
        return []
    explain = ("## Penanda nilai kosong (train + test)\n"
               "Dibuat sebelum angka kosong diisi, supaya model tetap tahu baris mana yang "
               "aslinya kosong.")
    columns = text_columns(config)
    for column in config["fill"]:
        if column not in columns:
            columns.append(column)
    lines = [f"MISSING_COLUMNS = {literal(columns)}",
             f"for table in {table_list(config)}:",
             "    for column in MISSING_COLUMNS:",
             "        table[f\"kosong_{column}\"] = table[column].isna().astype(int)"]
    return [markdown(explain), code("\n".join(lines))]


FILL_FUNCTION = '''def fill_value(values, method):
    if values.notna().sum() == 0:
        return np.nan
    if method == "median":
        return values.median()
    if method == "mean":
        return values.mean()
    if method == "modus":
        return values.mode().iloc[0]
    return 0


FILL_VALUES = {}
for column in FILL:
    FILL_VALUES[column] = float(fill_value(train[column], FILL[column]))
print("nilai isi (dari train):", FILL_VALUES)
train = train.fillna(FILL_VALUES)'''


def fill_cells(config):
    if len(config["fill"]) == 0:
        return []
    explain = ("## Isi angka kosong (train + test)\n"
               "Nilai isi dihitung dari train (sesudah baris dibuang), lalu dipakai **sama** "
               "untuk test. Simpan `FILL_VALUES` untuk inferensi; jangan hitung ulang dari test.")
    text = FILL_FUNCTION
    if has_test(config):
        text = text + "\ntest = test.fillna(FILL_VALUES)"
    return [markdown(explain), code(text)]


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
    return lines


def table_feature_lines(features):
    lines = []
    if "log_angka" in features:
        lines = lines + ["    for column in NUMBER_FEATURES:",
                         "        table[f\"log_{column}\"] = np.log1p(table[column].clip(lower=0))"]
    if "rasio_median" in features:
        lines = lines + ["    for column in NUMBER_FEATURES:",
                         "        medians = train.groupby(CATEGORY)[column].median()",
                         "        category_median = table[CATEGORY].map(medians)",
                         "        table[f\"rasio_{column}\"] = table[column] / category_median"]
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
    features = []
    for name in usable_features(config):
        if name != "kosong":    # penanda kosong sudah dibuat sebelum angka diisi
            features.append(name)
    if len(features) == 0:
        return []
    titles = []
    for name in features:
        titles.append(FEATURES[name])
    explain = ("## Tambah fitur (train + test)\n"
               f"Fitur: {'; '.join(titles)}. Kolom baru ditambahkan, kolom asli tetap. "
               "Statistik (median, frekuensi) diambil dari train saja supaya test tidak bocor.")
    lines = []
    if "gaya" in features:
        lines.append("STYLE_PATTERNS = {\"kapital\": r\"[A-Z]\", \"tanda_seru\": \"!\", "
                     "\"tanda_tanya\": r\"\\?\", \"angka\": r\"\\d+\"}")
    if "negasi" in features:
        lines.append("NEGATION = r\"\\b(tidak|bukan|gak|nggak|ga|tak|jangan)\\b\"")
    lines.append(f"for table in {table_list(config)}:")
    lines = lines + text_feature_lines(features) + table_feature_lines(features)
    lines.append("print(\"kolom train sekarang:\", train.shape[1])")
    return [markdown(explain), code("\n".join(lines))]


def encoding_methods(config):
    methods = []
    for name in ENCODE_FUNCTIONS:
        if name in config["encoding"].values():
            methods.append(name)
    return methods


def encoding_setup_lines(config, methods):
    lines = []
    if "target" in methods:
        lines.append(f"LABEL_IS_NUMBER = {config['label_is_number']}   # False = per kelas")
    for name in methods:
        if name in ENCODE_SETTINGS:
            lines.append(ENCODE_SETTINGS[name])
    return lines


def encoding_drop_lines(config):
    # kolom asli dibuang sesudah jadi angka (kecuali yang memang dibuang di langkah berikutnya)
    dropped = []
    for column in config["encoding"]:
        if column not in config["drop_columns"]:
            dropped.append(column)
    if not config["encode_drop"] or len(dropped) == 0:
        return []
    lines = [f"ENCODED_DROP = {literal(dropped)}   # kolom asli, sudah jadi angka",
             "train = train.drop(columns=ENCODED_DROP)"]
    if has_test(config):
        lines.append("test = test.drop(columns=ENCODED_DROP)")
    return lines


def encoding_loop_lines(config, methods):
    lines = ["ENCODERS = {"]
    for name in methods:
        lines.append(f'    "{name}": {ENCODE_NAMES[name]},')
    lines = lines + ["}", "for column in ENCODING:",
                     f"    ENCODERS[ENCODING[column]](column, {table_list(config)})"]
    lines = lines + encoding_drop_lines(config)
    return lines + ["print(\"kolom train sesudah encoding:\", train.shape[1])"]


def encoding_cells(config):
    methods = encoding_methods(config)
    if len(methods) == 0:
        return []
    explain = ("## Encoding kolom kategori (train + test)\n"
               "Urutan kode, rata-rata label, dan sumbu makna dihitung dari train lalu dipakai "
               "sama untuk test. *Target encoding* memakai label train sendiri, jadi skor "
               "validasi bisa terlalu bagus; *kedekatan makna* (MiniLM) memberi angka mirip "
               "untuk nilai yang artinya mirip, termasuk nilai baru di test.")
    cells = [markdown(explain)]
    if "makna" in methods:
        cells.append(code("from sentence_transformers import SentenceTransformer\n"
                          "MINILM = SentenceTransformer(MINILM_PATH, device=\"cpu\")", "model"))
    parts = ["\n".join(encoding_setup_lines(config, methods))]
    for name in methods:
        parts.append(ENCODE_FUNCTIONS[name])
    parts.append("\n".join(encoding_loop_lines(config, methods)))
    cells.append(code("\n\n\n".join(parts)))
    return cells


def drop_column_cells(config):
    if len(config["drop_columns"]) == 0:
        return []
    text = "train = train.drop(columns=DROP_COLUMNS)"
    if has_test(config):
        text = text + "\ntest = test.drop(columns=DROP_COLUMNS, errors=\"ignore\")"
    text = text + "\nprint(\"train:\", train.shape)"
    return [markdown("## Buang kolom (train + test)"), code(text)]


def split_cells(config):
    if config["split_lines"] is None:
        return []
    explain = ("## Split validasi\nSama dengan pilihan di tab *Split data*. Latih di "
               "`train_part`, ukur di `valid_part`; model akhir boleh dilatih ulang di `train`.")
    return [markdown(explain), code("\n".join(config["split_lines"]), "split")]


def save_cells(config):
    text = ('train.to_csv(OUTPUT_TRAIN, index_label="row")    # row = nomor baris CSV asli\n'
            'print("disimpan:", OUTPUT_TRAIN, train.shape)')
    if has_test(config):
        text = text + '\ntest.to_csv(OUTPUT_TEST, index=False)\nprint("disimpan:", OUTPUT_TEST)'
    if config["split_lines"] is not None:
        text = text + ('\ntrain_part.to_csv("train_split.csv", index_label="row")'
                       '\nvalid_part.to_csv("valid_split.csv", index_label="row")')
    return [markdown("## Simpan"), code(text, "save")]


def build_cells(config):
    cells = intro_cells(config) + config_cells(config) + read_cells(config)
    cells = cells + label_cells(config) + number_cells(config)
    cells = cells + rule_cells(config) + outlier_cells(config) + drop_cells(config)
    cells = cells + missing_cells(config) + fill_cells(config) + feature_cells(config)
    cells = cells + encoding_cells(config) + drop_column_cells(config)
    cells = cells + split_cells(config) + save_cells(config)
    return cells


def app_script(cells):
    # sel yang dijalankan aplikasi: semua sel kode kecuali baca, model (aplikasi memberi
    # MINILM sendiri), split, dan simpan
    parts = []
    for kind, text, role in cells:
        if kind == "code" and role is None:
            parts.append(text)
    return "\n\n".join(parts) + "\n"


def to_script(cells):
    # format # %% yang dipakai seluruh toolkit (_alat/nb/*.py)
    parts = []
    for kind, text, role in cells:
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
    for kind, text, role in cells:
        if kind == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            notebook.cells.append(nbformat.v4.new_code_cell(text))
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                       "name": "python3"}
    return nbformat.writes(notebook)
