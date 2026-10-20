"""Logika tab 'Fitur -> label' dan 'Train vs test' (tanpa streamlit).
Semua fungsi murni pandas / sklearn, jadi bisa dicoba sendiri di notebook."""
import os
import re
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats as scipy_stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder


SEED = 42
FOLDS = 5
FLIP_SEED = 2024            # seed untuk mengacak label saat mengukur pencari label salah
ALAY_CSV = "../../data_latihan/preprocessing/kamus_alay_salsabila.csv"
NEGATION_WORDS = ["tidak", "bukan", "gak", "nggak", "ga", "tak", "jangan"]
EMOJI_PATTERN = "[\U0001F300-\U0001FAFF☀-➿]"
DATE_WORDS = ["tanggal", "tgl", "date", "waktu", "time"]
LOGISTIC = {"C": 3, "solver": "saga", "max_iter": 500, "tol": 0.001}    # cepat untuk TF-IDF
BETTER_BY = 0.01             # selisih skor minimal supaya disebut "bantu" / "merugikan"
LEAK_SCORE = 0.95            # fitur tunggal dengan skor segini = curiga bocor
NYASAR_RECALL = 0.86         # angka uji sebelumnya: metode "nyasar" MiniLM (tab Outlier)
NYASAR_PRECISION = 0.177

# nama grup -> (jenis, judul pendek). Urutan = urutan tampil di tab.
GROUPS = {
    "kata": ("teks", "TF-IDF kata 1-2 gram (baseline, selalu ikut)"),
    "char": ("teks", "TF-IDF karakter 2-5 gram (char_wb)"),
    "gaya": ("teks", "statistik gaya (panjang, kapital, !, ?, emoji, huruf diulang ...)"),
    "negasi": ("teks", "penanda negasi (tidak, bukan, gak, nggak, ga, tak, jangan)"),
    "alay": ("teks", "rasio kata alay (kamus Salsabila)"),
    "minilm": ("teks", "vektor makna MiniLM (hanya baris yang punya vektor)"),
    "laya": ("teks", "lensa Laya (nilai + one-hot jawaban)"),
    "kelompok": ("teks", "kelompok MiniLM (one-hot)"),
    "log_angka": ("tabel", "log transform kolom angka"),
    "bin_angka": ("tabel", "binning kuantil kolom angka (10 kotak)"),
    "rasio_median": ("tabel", "angka dibagi median kategorinya (kategori = dimensi pilihan)"),
    "frekuensi": ("tabel", "frequency encoding kolom kategori"),
    "target_enc": ("tabel", "target encoding kolom kategori (out-of-fold, tanpa bocor)"),
    "kosong": ("tabel", "penanda nilai kosong tiap kolom"),
    "regex": ("tabel", "ekstraksi regex (angka terbesar di teks, kata kunci seperti /bulan)"),
    "panjang_kolom": ("tabel", "panjang teks tiap kolom"),
    "tanggal": ("tabel", "bagian tanggal (tahun, bulan, hari, jam)"),
}
SPARSE_GROUPS = ["kata", "char"]


# ================= dasar: jenis tugas, baris, statistik =================

def task_of(label):
    # "kelas" = label kategori (F1 macro), "angka" = label angka (MAE, R2)
    if not pd.api.types.is_numeric_dtype(label):
        return "kelas"
    if label.nunique() <= 20:
        return "kelas"
    return "angka"


def signature(ctx):
    # tanda pengenal data: dipakai st.cache_data supaya hitungan diulang bila data berubah
    csv_path = ctx["csv_path"]
    modified = 0
    if os.path.exists(csv_path):
        modified = os.path.getmtime(csv_path)
    vector_shape = (0, 0)
    if ctx["vectors"] is not None:
        vector_shape = ctx["vectors"].shape
    lens_counts = []
    for name in ctx["lens_done"]:
        lens_counts.append((name, int(ctx["table"][f"{name}__nilai"].notna().sum())))
    return (csv_path, modified, len(ctx["data"]), tuple(ctx["columns"]), ctx["label_column"],
            vector_shape, tuple(lens_counts), tuple(ctx["numbers"].columns))


def style_features(texts):
    # statistik gaya penulisan, tidak memakai label, jadi aman dihitung di semua baris
    frame = pd.DataFrame(index=texts.index)
    lengths = texts.str.len().clip(lower=1)
    frame["panjang"] = texts.str.len()
    frame["jumlah_kata"] = texts.str.split().str.len()
    frame["rasio_kapital"] = texts.str.count(r"[A-Z]") / lengths
    frame["tanda_seru"] = texts.str.count("!")
    frame["tanda_tanya"] = texts.str.count(r"\?")
    frame["jumlah_angka"] = texts.str.count(r"\d+")
    frame["url"] = texts.str.count(r"https?://|www\.|\[URL\]")
    frame["mention"] = texts.str.count(r"@\w+|\[USERNAME\]")
    frame["hashtag"] = texts.str.count(r"#\w+")
    frame["emoji"] = texts.str.count(EMOJI_PATTERN)
    frame["kata_diulang"] = texts.str.count(r"(?i)\b(\w+)\s+\1\b")
    frame["huruf_diulang"] = texts.str.count(r"(\w)\1{2,}")
    return frame


def negation_features(texts):
    frame = pd.DataFrame(index=texts.index)
    pattern = r"\b(" + "|".join(NEGATION_WORDS) + r")\b"
    lower = texts.str.lower()
    frame["jumlah_negasi"] = lower.str.count(pattern)
    frame["ada_negasi"] = (frame["jumlah_negasi"] > 0).astype(int)
    return frame


def load_alay_words():
    # kata alay = bentuk "slang" di kamus yang berbeda dari bentuk bakunya
    if not os.path.exists(ALAY_CSV):
        return set()
    dictionary = pd.read_csv(ALAY_CSV)
    words = set()
    for slang, standard in zip(dictionary["slang"], dictionary["baku"]):
        if slang != standard:
            words.add(str(slang).lower())
    return words


def alay_features(texts, alay_words):
    counts = []
    ratios = []
    for text in texts:
        words = re.findall(r"[a-z]+", text.lower())
        found = 0
        for word in words:
            if word in alay_words:
                found = found + 1
        counts.append(found)
        ratios.append(found / max(len(words), 1))
    return pd.DataFrame({"jumlah_alay": counts, "rasio_alay": ratios}, index=texts.index)


# ================= fitur tabel (angka, kategori, kolom) =================

def log_features(numbers):
    frame = pd.DataFrame(index=numbers.index)
    for column in numbers.columns:
        values = numbers[column]
        frame[f"log_{column}"] = np.sign(values) * np.log1p(values.abs())
    return frame


def bin_features(numbers, n_bins=10):
    frame = pd.DataFrame(index=numbers.index)
    for column in numbers.columns:
        codes = pd.qcut(numbers[column], q=n_bins, labels=False, duplicates="drop")
        dummies = pd.get_dummies(codes, prefix=f"bin_{column}", dtype=float)
        for name in dummies.columns:
            frame[name] = dummies[name]
    return frame


def ratio_features(numbers, category):
    # angka dibagi median kategorinya: harga 1 miliar untuk 'kost' jauh lebih janggal dari 'rumah'
    frame = pd.DataFrame(index=numbers.index)
    for column in numbers.columns:
        medians = numbers[column].groupby(category).transform("median")
        ratio = numbers[column] / medians.where(medians > 0)
        frame[f"rasio_{column}"] = ratio
        frame[f"log_rasio_{column}"] = np.log(ratio.where(ratio > 0))
    return frame


def frequency_features(data, category_columns):
    frame = pd.DataFrame(index=data.index)
    for column in category_columns:
        shares = data[column].value_counts(normalize=True)
        frame[f"frek_{column}"] = data[column].map(shares)
    return frame


def missing_features(data, columns):
    # kosong = NaN, teks kosong, atau '-'
    frame = pd.DataFrame(index=data.index)
    for column in columns:
        empty = data[column].isna()
        if data[column].dtype == object:
            text = data[column].astype(str).str.strip()
            empty = empty | text.isin(["", "-", "nan"])
        if empty.sum() > 0:
            frame[f"kosong_{column}"] = empty.astype(int)
    return frame


def regex_features(texts, keywords):
    frame = pd.DataFrame(index=texts.index)
    largest = []
    for text in texts:
        biggest = 0.0
        for found in re.findall(r"\d[\d.]*", text):
            digits = found.replace(".", "")
            if len(digits) <= 15:
                biggest = max(biggest, float(digits))
        largest.append(np.log1p(biggest))
    frame["angka_terbesar_log"] = largest
    lower = texts.str.lower()
    for keyword in keywords:
        frame[f"kata_{keyword}"] = lower.str.count(re.escape(keyword))
    return frame


def length_features(data, columns):
    frame = pd.DataFrame(index=data.index)
    for column in columns:
        values = data[column].fillna("")
        values = values.astype(str)
        frame[f"panjang_{column}"] = values.str.len()
    return frame


def date_features(data, date_columns):
    frame = pd.DataFrame(index=data.index)
    for column in date_columns:
        dates = pd.to_datetime(data[column], errors="coerce")
        frame[f"{column}_tahun"] = dates.dt.year
        frame[f"{column}_bulan"] = dates.dt.month
        frame[f"{column}_hari"] = dates.dt.day
        frame[f"{column}_hari_minggu"] = dates.dt.dayofweek
        frame[f"{column}_jam"] = dates.dt.hour
    return frame


# ================= pembangun grup (ctx, options) -> kolom padat =================
# Hasil kosong (0 kolom) = grup belum bisa dipakai. Baris yang tidak punya nilai dibuang.

def usable_numbers(ctx):
    # kolom angka tanpa kolom label (dan turunannya) supaya tidak bocor
    label = ctx["label_column"]
    keep = []
    for column in ctx["numbers"].columns:
        if column == label:
            continue
        if str(column).startswith(f"{label}_per_") or str(column).endswith(f"_per_{label}"):
            continue
        keep.append(column)
    return ctx["numbers"][keep]


def usable_lenses(ctx):
    # lensa yang melihat kolom label tidak boleh jadi fitur (bocor)
    names = []
    for name in ctx["lens_done"]:
        if ctx["lenses"][name].get("with_label"):
            continue
        names.append(name)
    return names


def category_candidates(ctx):
    # kolom teks yang isinya sedikit variasi (2-50 nilai) = kategori
    columns = []
    for column in ctx["columns"]:
        if column == ctx["label_column"]:
            continue
        values = ctx["data"][column]
        if values.dtype == object and 2 <= values.nunique() <= 50:
            columns.append(column)
    return columns


def date_candidates(ctx):
    columns = []
    for column in ctx["data"].columns:
        for word in DATE_WORDS:
            if word in str(column).lower():
                columns.append(column)
                break
    return columns


def dimension_candidates(ctx):
    names = []
    for name in ctx["dimensions"]:
        if name != ctx["label_column"]:
            names.append(name)
    return names


def build_gaya(ctx, options):
    return style_features(ctx["texts"])


def build_negasi(ctx, options):
    return negation_features(ctx["texts"])


def build_alay(ctx, options):
    alay_words = load_alay_words()
    if len(alay_words) == 0:
        return pd.DataFrame()
    return alay_features(ctx["texts"], alay_words)


def build_minilm(ctx, options):
    if ctx["vectors"] is None:
        return pd.DataFrame()
    names = []
    for i in range(ctx["vectors"].shape[1]):
        names.append(f"minilm_{i}")
    return pd.DataFrame(ctx["vectors"], index=ctx["vector_rows"], columns=names)


def build_laya(ctx, options):
    names = usable_lenses(ctx)
    frame = pd.DataFrame(index=ctx["table"].index)
    ready = pd.Series(True, index=ctx["table"].index)
    for name in names:
        frame[f"laya_{name}__nilai"] = ctx["table"][f"{name}__nilai"]
        dummies = pd.get_dummies(ctx["table"][name], prefix=f"laya_{name}", dtype=float)
        for column in dummies.columns:
            frame[column] = dummies[column]
        ready = ready & ctx["table"][f"{name}__nilai"].notna()
    if len(names) == 0:
        return pd.DataFrame()
    return frame[ready]


def build_kelompok(ctx, options):
    if "kelompok" not in ctx["table"].columns:
        return pd.DataFrame()
    groups = ctx["table"]["kelompok"].dropna()
    return pd.get_dummies(groups.astype(int), prefix="kelompok", dtype=float)


def build_log_angka(ctx, options):
    return log_features(usable_numbers(ctx))


def build_bin_angka(ctx, options):
    return bin_features(usable_numbers(ctx))


def build_rasio_median(ctx, options):
    dimension = options["dimension"]
    if dimension is None or usable_numbers(ctx).shape[1] == 0:
        return pd.DataFrame()
    category = ctx["table"][dimension]
    frame = ratio_features(usable_numbers(ctx), category)
    return frame[category.notna()]


def build_frekuensi(ctx, options):
    return frequency_features(ctx["data"], options["category_columns"])


def build_kosong(ctx, options):
    return missing_features(ctx["data"], ctx["columns"])


def build_regex(ctx, options):
    return regex_features(ctx["texts"], options["keywords"])


def build_panjang_kolom(ctx, options):
    return length_features(ctx["data"], ctx["columns"])


def build_tanggal(ctx, options):
    return date_features(ctx["data"], options["date_columns"])


BUILDERS = {
    "gaya": build_gaya, "negasi": build_negasi, "alay": build_alay, "minilm": build_minilm,
    "laya": build_laya, "kelompok": build_kelompok, "log_angka": build_log_angka,
    "bin_angka": build_bin_angka, "rasio_median": build_rasio_median,
    "frekuensi": build_frekuensi, "kosong": build_kosong, "regex": build_regex,
    "panjang_kolom": build_panjang_kolom, "tanggal": build_tanggal,
}


def group_availability(ctx, options):
    # grup -> (jumlah kolom, jumlah baris yang punya nilai), 0 kolom = belum bisa dipakai
    result = {}
    result["kata"] = (1, len(ctx["data"]))
    result["char"] = (1, len(ctx["data"]))
    for name in BUILDERS:
        frame = BUILDERS[name](ctx, options)
        result[name] = (frame.shape[1], frame.shape[0])
    result["target_enc"] = (len(options["category_columns"]), len(ctx["data"]))
    return result


# ================= model: pipeline sklearn =================

def text_parts(spec):
    parts = []
    if spec["kata"]:
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        parts.append(("kata", word, "teks"))
    if spec["char"]:
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                               sublinear_tf=True, max_features=100000)
        parts.append(("char", char, "teks"))
    return parts


def dense_parts(spec, linear, n_classes):
    # model linear: nilai padat distandarkan, tiap grup diperkecil agar normnya ~1 seperti TF-IDF
    parts = []
    weights = {}
    for name in spec["dense"]:
        columns = spec["dense"][name]
        if linear:
            steps = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
            parts.append((name, make_pipeline(*steps), columns))
            weights[name] = 1 / np.sqrt(len(columns))
        else:
            parts.append((name, "passthrough", columns))
    if len(spec["te"]) > 0:
        encoder = TargetEncoder(random_state=SEED)
        weights["te"] = 1 / np.sqrt(len(spec["te"]) * max(n_classes, 1))
        parts.append(("te", make_pipeline(encoder, StandardScaler()), spec["te"]))
    return parts, weights


def make_estimator(task, linear, class_weight):
    if task == "kelas" and linear:
        return LogisticRegression(class_weight=class_weight, **LOGISTIC)
    if task == "angka" and linear:
        return Ridge(alpha=1.0)
    settings = {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 15,
                "min_child_samples": 10, "verbose": -1, "random_state": SEED, "n_jobs": 4}
    if task == "kelas":
        return LGBMClassifier(class_weight=class_weight, **settings)
    return LGBMRegressor(**settings)


def make_model(spec, task, n_classes=2, class_weight=None):
    # satu spec -> satu pipeline. Ada TF-IDF = model linear, hanya angka padat = LightGBM
    linear = spec["kata"] or spec["char"]
    parts = text_parts(spec)
    extra, weights = dense_parts(spec, linear, n_classes)
    transformer = ColumnTransformer(parts + extra, transformer_weights=weights)
    return Pipeline([("fitur", transformer), ("model", make_estimator(task, linear, class_weight))])


def run_cv(model, X, y, task):
    # skor cross-validation: kelas -> F1 macro (StratifiedKFold), angka -> R2 (KFold)
    started = time.time()
    if task == "kelas":
        splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
        scoring = {"skor": "f1_macro", "kedua": "accuracy"}
    else:
        splitter = KFold(FOLDS, shuffle=True, random_state=SEED)
        scoring = {"skor": "r2", "kedua": "neg_mean_absolute_error"}
    result = cross_validate(model, X, y, cv=splitter, scoring=scoring)
    return {"skor": result["test_skor"].mean(), "std": result["test_skor"].std(),
            "kedua": result["test_kedua"].mean(), "detik": time.time() - started,
            "baris": len(y)}


def cv_scores(X, y, spec, task, class_weight=None):
    model = make_model(spec, task, y.nunique(), class_weight)
    return run_cv(model, X, y, task)


# ================= merakit X dari grup yang dipilih =================

def pick_rows(candidates, max_rows, rows):
    if rows is not None:
        return pd.Index(rows)
    if len(candidates) <= max_rows:
        return candidates.sort_values()
    chosen = np.random.RandomState(SEED).choice(candidates, size=max_rows, replace=False)
    return pd.Index(np.sort(chosen))


def drop_rare_classes(rows, y):
    # CV 5 lipatan butuh minimal 5 contoh per kelas; kelas yang lebih kecil tidak ikut diuji
    if task_of(y) != "kelas":
        return rows
    counts = y[rows].value_counts()
    enough = counts[counts >= FOLDS].index
    return rows[y[rows].isin(enough)]


def assemble(ctx, groups, options, max_rows, rows=None):
    # hasil: X (kolom 'teks' + kolom padat), y, spec. Baris = baris berlabel yang punya semua grup
    y = ctx["data"][ctx["label_column"]]
    candidates = y.dropna().index
    frames = {}
    for name in groups:
        if name in BUILDERS:
            frames[name] = BUILDERS[name](ctx, options)
            if frames[name].shape[1] > 0:
                candidates = candidates.intersection(frames[name].index)
    rows = pick_rows(candidates, max_rows, rows)
    rows = drop_rare_classes(rows, y)
    X = pd.DataFrame({"teks": ctx["texts"][rows]})
    spec = {"kata": "kata" in groups, "char": "char" in groups, "dense": {}, "te": []}
    for name in frames:
        if frames[name].shape[1] > 0:
            X = pd.concat([X, frames[name].loc[rows]], axis=1)
            spec["dense"][name] = list(frames[name].columns)
    if "target_enc" in groups:
        spec["te"] = add_category_columns(X, ctx, options, rows)
    return X, y[rows], spec


def add_category_columns(X, ctx, options, rows):
    # kolom kategori mentah untuk target encoding (nilainya diubah di dalam pipeline, per lipatan)
    names = []
    for column in options["category_columns"]:
        values = ctx["data"].loc[rows, column].fillna("(kosong)")
        X[f"kat_{column}"] = values.astype(str)
        names.append(f"kat_{column}")
    return names


def group_details(ctx, name, options):
    # nama kolom + parameter satu kelompok, untuk fitur_terpilih.json
    fixed = {"kata": {"tfidf": "kata", "ngram": [1, 2], "min_df": 2, "sublinear_tf": True},
             "char": {"tfidf": "char_wb", "ngram": [2, 5], "min_df": 3, "sublinear_tf": True},
             "target_enc": {"kolom": options["category_columns"], "cv": "out-of-fold"}}
    if name in fixed:
        return fixed[name]
    columns = list(BUILDERS[name](ctx, options).columns)
    return {"jumlah_kolom": len(columns), "kolom": columns[:30]}


def decide(difference, noise):
    # selisih harus cukup besar DAN melebihi derau antar lipatan, baru dianggap nyata
    if difference >= BETTER_BY and difference > noise:
        return "bantu"
    if difference <= -BETTER_BY and -difference > noise:
        return "merugikan"
    return "tidak beda"


def result_line(name, with_group, base):
    difference = with_group["skor"] - base["skor"]
    return {"grup": name, "baris": with_group["baris"], "skor_baseline": base["skor"],
            "skor_dengan_grup": with_group["skor"], "selisih": difference,
            "std_fold": with_group["std"], "keputusan": decide(difference, with_group["std"]),
            "detik": with_group["detik"] + base["detik"]}


def score_group(ctx, groups, options, max_rows, scorer=cv_scores):
    # baseline (TF-IDF kata) vs baseline + grup, pada baris yang sama persis
    task = task_of(ctx["data"][ctx["label_column"]])
    X, y, spec = assemble(ctx, ["kata"] + groups, options, max_rows)
    with_group = scorer(X, y, spec, task)
    X0, y0, spec0 = assemble(ctx, ["kata"], options, max_rows, rows=X.index)
    base = scorer(X0, y0, spec0, task)
    return result_line("+".join(groups), with_group, base)


def ablation_table(ctx, options, max_rows, scorer=cv_scores):
    # apakah Laya menambah informasi? TF-IDF saja vs TF-IDF + Laya vs Laya saja
    task = task_of(ctx["data"][ctx["label_column"]])
    X, y, spec = assemble(ctx, ["kata", "laya"], options, max_rows)
    X0, y0, spec0 = assemble(ctx, ["kata"], options, max_rows, rows=X.index)
    X1, y1, spec1 = assemble(ctx, ["laya"], options, max_rows, rows=X.index)
    chance = chance_score(y, task)
    chance["model"] = "tebakan kelas terbanyak"
    lines = [chance]
    runs = [("TF-IDF saja", X0, y0, spec0), ("TF-IDF + lensa Laya", X, y, spec),
            ("lensa Laya saja", X1, y1, spec1)]
    for name, X_run, y_run, spec_run in runs:
        result = scorer(X_run, y_run, spec_run, task)
        result["model"] = name
        lines.append(result)
    return pd.DataFrame(lines)


def chance_score(y, task):
    if task == "kelas":
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = DummyRegressor()
    return run_cv(model, np.zeros((len(y), 1)), y, task)


# ================= importance =================

def top_words(texts, y, task, n_words=15):
    # kata penentu: koefisien model linear pada TF-IDF kata, hasil: kelas -> (positif, negatif)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    if task == "angka":
        model = Ridge(alpha=1.0).fit(matrix, y)
        return {"nilai label": split_words(model.coef_, words, n_words)}
    model = LogisticRegression(**LOGISTIC).fit(matrix, y)
    result = {}
    for position, label in enumerate(model.classes_):
        coefficients = model.coef_[position]
        if len(model.classes_) == 2 and position == 0:
            coefficients = -model.coef_[0]
        result[str(label)] = split_words(coefficients, words, n_words)
    return result


def split_words(coefficients, words, n_words):
    order = np.argsort(coefficients)
    positive = pd.DataFrame({"kata": words[order[::-1][:n_words]]})
    positive["bobot"] = np.sort(coefficients)[::-1][:n_words].round(2)
    negative = pd.DataFrame({"kata": words[order[:n_words]]})
    negative["bobot"] = np.sort(coefficients)[:n_words].round(2)
    return positive, negative


def dense_only(X, spec):
    # hanya kolom padat (tanpa teks, tanpa vektor MiniLM yang terlalu banyak kolom)
    names = []
    owner = {}
    for group in spec["dense"]:
        if group == "minilm":
            continue
        for column in spec["dense"][group]:
            names.append(column)
            owner[column] = group
    return X[names].astype(float), owner


def permutation_table(X, y, spec, task):
    # Breiman 2001: acak satu kolom di data uji, turunnya skor = pentingnya kolom itu
    dense, owner = dense_only(X, spec)
    only_dense = {"kata": False, "char": False, "dense": {"padat": list(dense.columns)}, "te": []}
    stratify = None
    scoring = "r2"
    if task == "kelas":
        stratify = y
        scoring = "f1_macro"
    X_train, X_test, y_train, y_test = train_test_split(dense, y, test_size=0.25,
                                                        random_state=SEED, stratify=stratify)
    model = make_model(only_dense, task, y.nunique()).fit(X_train, y_train)
    result = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=5,
                                    random_state=SEED)
    table = pd.DataFrame({"fitur": dense.columns, "turun_skor": result.importances_mean.round(4),
                          "std": result.importances_std.round(4)})
    table["grup"] = table["fitur"].map(owner)
    return table.sort_values("turun_skor", ascending=False)


def mutual_info_table(X, y, spec, task):
    # informasi bersama tiap fitur dengan label (0 = tak berhubungan), menangkap hubungan non-linear
    dense, owner = dense_only(X, spec)
    filled = dense.fillna(dense.median()).fillna(0)
    if task == "kelas":
        values = mutual_info_classif(filled, y, random_state=SEED)
    else:
        values = mutual_info_regression(filled, y, random_state=SEED)
    table = pd.DataFrame({"fitur": dense.columns, "mutual_info": np.round(values, 4)})
    table["grup"] = table["fitur"].map(owner)
    return table.sort_values("mutual_info", ascending=False)


# ================= pencari label salah (confident learning sederhana) =================

def out_of_fold_proba(X, y, spec):
    # peluang tiap kelas untuk tiap baris dari model yang TIDAK pernah melihat baris itu
    model = make_model(spec, "kelas", y.nunique())
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    proba = cross_val_predict(model, X, y, cv=splitter, method="predict_proba")
    return np.array(sorted(y.unique())), proba


def class_thresholds(proba, given):
    # Northcutt dkk. 2021: ambang kelas = rata-rata peluang kelas itu pada baris berlabel kelas itu
    thresholds = np.zeros(proba.shape[1])
    for position in range(proba.shape[1]):
        thresholds[position] = proba[given == position, position].mean()
    return thresholds


def label_issues(proba, classes, labels):
    # tandai: peluang label sendiri < ambangnya DAN ada kelas lain di atas ambangnya
    positions = pd.Series(range(len(classes)), index=classes)
    given = positions.loc[labels.to_numpy()].to_numpy()
    rows = np.arange(len(labels))
    thresholds = class_thresholds(proba, given)
    p_given = proba[rows, given]
    above = proba >= thresholds
    above[rows, given] = False
    flagged = above.any(axis=1) & (p_given < thresholds[given])
    best = np.where(above, proba, -1).argmax(axis=1)
    table = pd.DataFrame(index=labels.index)
    table["label_diberikan"] = labels.to_numpy()
    table["label_saran"] = classes[best]
    table["peluang_label"] = p_given
    table["peluang_saran"] = proba[rows, best]
    table["selisih"] = table["peluang_saran"] - table["peluang_label"]
    table["alasan"] = issue_reasons(table, thresholds, best)
    return table[flagged].sort_values("selisih", ascending=False)


def issue_reasons(table, thresholds, best):
    reasons = []
    for i in range(len(table)):
        row = table.iloc[i]
        reasons.append(f"model memberi label '{row['label_diberikan']}' "
                       f"{row['peluang_label']:.0%}, '{row['label_saran']}' "
                       f"{row['peluang_saran']:.0%} (ambang '{row['label_saran']}' "
                       f"{thresholds[best[i]]:.0%})")
    return reasons


def flip_labels(labels, fraction, seed):
    # acak sebagian label ke kelas lain (untuk mengukur pencari label salah)
    rng = np.random.RandomState(seed)
    classes = sorted(labels.unique())
    noisy = labels.copy()
    flipped = pd.Series(False, index=labels.index)
    chosen = rng.choice(len(labels), size=int(fraction * len(labels)), replace=False)
    for position in chosen:
        others = []
        for label in classes:
            if label != labels.iloc[position]:
                others.append(label)
        noisy.iloc[position] = others[rng.randint(len(others))]
        flipped.iloc[position] = True
    return noisy, flipped


def precision_recall(flagged, flipped):
    hit = int((flagged & flipped).sum())
    precision = hit / max(int(flagged.sum()), 1)
    recall = hit / max(int(flipped.sum()), 1)
    return {"ditandai": int(flagged.sum()), "tepat": hit, "presisi": precision,
            "recall": recall, "dasar_acak": flipped.mean()}


def precision_at_top(issues, flipped):
    # presisi bila hanya mengambil K teratas, K = jumlah label yang memang diacak
    top = issues.head(int(flipped.sum())).index
    return float(flipped[top].mean())


def nyasar_flags(ctx, noisy):
    # metode "nyasar" (tab Outlier): teks lebih dekat ke pusat kelas lain daripada kelasnya sendiri
    if ctx["vectors"] is None:
        return None
    keep = np.isin(ctx["vector_rows"], noisy.index)
    vectors = ctx["vectors"][keep]
    rows = ctx["vector_rows"][keep]
    labels = noisy[rows].to_numpy()
    names = sorted(pd.unique(labels))
    centers = []
    for name in names:
        center = vectors[labels == name].mean(axis=0)
        centers.append(center / np.linalg.norm(center))
    nearest = np.array(names, dtype=object)[(vectors @ np.array(centers).T).argmax(axis=1)]
    return pd.Series(nearest != labels, index=rows)


# ================= ketidakseimbangan kelas =================

def class_balance(y):
    counts = y.value_counts()
    table = pd.DataFrame({"jumlah": counts})
    table["persen"] = (100 * counts / counts.sum()).round(1)
    return table, counts.max() / counts.min()


def imbalance_advice(ratio):
    if ratio < 1.5:
        return "seimbang: akurasi biasa masih wajar, tetap pakai stratify supaya adil."
    if ratio < 3:
        return "agak tidak seimbang: pakai F1 macro, stratify, dan coba class_weight='balanced'."
    return "tidak seimbang: wajib F1 macro, stratify, class_weight='balanced' atau bobot sendiri."


def imbalance_experiment(X, y, spec):
    plain = cv_scores(X, y, spec, "kelas")
    weighted = cv_scores(X, y, spec, "kelas", class_weight="balanced")
    return plain, weighted


# ================= cek fitur bocor =================

def score_numeric_column(values, y, task):
    X = pd.DataFrame({"nilai": values})
    spec = {"kata": False, "char": False, "dense": {"padat": ["nilai"]}, "te": []}
    return cv_scores(X, y, spec, task)


def score_text_column(values, y, task):
    vectorizer = TfidfVectorizer(token_pattern=r"\w+", min_df=1)
    estimator = LogisticRegression(**LOGISTIC)
    if task == "angka":
        estimator = Ridge(alpha=1.0)
    model = make_pipeline(vectorizer, estimator)
    return run_cv(model, values.fillna("").astype(str), y, task)


def leak_candidates(ctx):
    # semua kolom data asli (termasuk id) + nomor baris + nilai lensa Laya (juga yang melihat label)
    columns = {}
    for column in ctx["data"].columns:
        if column != ctx["label_column"]:
            columns[column] = ctx["data"][column]
    columns["(nomor baris)"] = pd.Series(ctx["data"].index, index=ctx["data"].index)
    for name in ctx["lens_done"]:
        note = ""
        if ctx["lenses"][name].get("with_label"):
            note = " (Laya melihat label)"
        columns[f"laya {name}{note}"] = ctx["table"][f"{name}__nilai"]
    return columns


def single_feature_scores(ctx, max_rows):
    # tiap fitur sendirian -> skor CV. Skor mendekati sempurna = curiga bocor
    y = ctx["data"][ctx["label_column"]]
    task = task_of(y)
    lines = []
    candidates = leak_candidates(ctx)
    for name in candidates:
        rows = y.dropna().index.intersection(candidates[name].dropna().index)
        rows = pick_rows(rows, max_rows, None)
        rows = drop_rare_classes(rows, y)
        if len(rows) < 30:
            continue
        result = score_column(candidates[name][rows], y[rows], task)
        lines.append({"fitur": name, "baris": len(rows), "skor": result["skor"],
                      "std": result["std"]})
    table = pd.DataFrame(lines).sort_values("skor", ascending=False)
    table["status"] = table["skor"].map(leak_status)
    return table


def score_column(values, y, task):
    if pd.api.types.is_numeric_dtype(values):
        return score_numeric_column(values, y, task)
    return score_text_column(values, y, task)


def leak_status(score):
    if score >= LEAK_SCORE:
        return "CURIGA BOCOR"
    if score >= 0.8:
        return "sangat kuat, periksa"
    return "wajar"


# ================= TRAIN vs TEST =================

def percent_histogram(train, test, bins=30):
    # sebaran (persen) di sumbu yang sama
    both = pd.concat([train, test]).dropna()
    edges = np.histogram_bin_edges(both, bins=bins)
    train_counts, _ = np.histogram(train.dropna(), edges)
    test_counts, _ = np.histogram(test.dropna(), edges)
    chart = pd.DataFrame(index=edges[:-1].round(1))
    chart["train (%)"] = 100 * train_counts / max(train_counts.sum(), 1)
    chart["test (%)"] = 100 * test_counts / max(test_counts.sum(), 1)
    return chart


def ks_line(name, train, test):
    # uji Kolmogorov-Smirnov dua sampel: D = jarak terbesar antar dua kurva kumulatif
    train = train.dropna()
    test = test.dropna()
    result = scipy_stats.ks_2samp(train, test)
    verdict = "mirip"
    if result.pvalue < 0.01 and result.statistic >= 0.1:
        verdict = "BEDA"
    elif result.pvalue < 0.05:
        verdict = "sedikit beda"
    return {"fitur": name, "median_train": train.median(), "median_test": test.median(),
            "D": round(result.statistic, 3), "p": round(result.pvalue, 4), "kesimpulan": verdict}


def psi_value(train_share, test_share):
    # Population Stability Index: <0.1 stabil, 0.1-0.25 bergeser, >0.25 beda besar
    train_share = np.clip(train_share, 1e-4, None)
    test_share = np.clip(test_share, 1e-4, None)
    return float(np.sum((test_share - train_share) * np.log(test_share / train_share)))


def category_shift(train, test):
    counts = pd.DataFrame({"train": train.fillna("(kosong)").value_counts(),
                           "test": test.fillna("(kosong)").value_counts()}).fillna(0)
    shares = counts / counts.sum()
    chi2, p_value, _, _ = scipy_stats.chi2_contingency(counts.T.to_numpy())
    cramer = np.sqrt(chi2 / counts.to_numpy().sum())
    psi = psi_value(shares["train"].to_numpy(), shares["test"].to_numpy())
    return counts, {"chi2": round(chi2, 2), "p": round(p_value, 4),
                    "cramer_v": round(cramer, 3), "psi": round(psi, 3)}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def count_new_words(known, test_texts):
    new_words = {}
    total = 0
    rows_with_new = 0
    for text in test_texts:
        found_new = False
        for word in tokenize(text):
            total = total + 1
            if word not in known:
                found_new = True
                new_words[word] = new_words.get(word, 0) + 1
        if found_new:
            rows_with_new = rows_with_new + 1
    return new_words, total, rows_with_new


def vocabulary_gap(train_texts, test_texts, n_words=30):
    # kata di test yang tidak pernah muncul di train (out-of-vocabulary)
    known = set()
    for text in train_texts:
        known.update(tokenize(text))
    new_words, total, rows_with_new = count_new_words(known, test_texts)
    unseen = sum(new_words.values())
    counts = pd.Series(new_words, dtype=float)
    top = counts.sort_values(ascending=False).head(n_words)
    summary = {"kata_test": total, "kata_baru": unseen, "persen_kata_baru": 100 * unseen / total,
               "persen_baris_ada_kata_baru": 100 * rows_with_new / len(test_texts),
               "jenis_kata_baru": len(new_words)}
    return summary, top.rename("jumlah_di_test").to_frame()


def normalize_text(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def build_lookup(train_texts, train_labels):
    # teks -> daftar label di train, persis dan setelah dinormalisasi
    exact = {}
    simple = {}
    for row, text in train_texts.items():
        label = ""
        if train_labels is not None:
            label = str(train_labels[row])
        exact.setdefault(text, []).append(label)
        simple.setdefault(normalize_text(text), []).append(label)
    return exact, simple


def find_overlaps(train_texts, train_labels, test_texts):
    # teks test yang sama dengan teks train, persis atau setelah huruf kecil dan tanpa tanda baca
    exact, simple = build_lookup(train_texts, train_labels)
    lines = []
    for row, text in test_texts.items():
        kind = "persis"
        labels = exact.get(text)
        if labels is None:
            kind = "mirip (beda huruf besar/tanda baca)"
            labels = simple.get(normalize_text(text))
        if labels is None or normalize_text(text) == "":
            continue
        common = pd.Series(labels).value_counts()
        lines.append({"baris_test": row, "jenis": kind, "label_di_train": common.index[0],
                      "jumlah_di_train": len(labels), "label_bentrok": len(common) > 1,
                      "teks": text})
    return pd.DataFrame(lines)


def adversarial_validation(train_texts, test_texts, max_rows=5000):
    # model membedakan train (0) dan test (1). AUC ~0.5 = tak bisa dibedakan = mirip
    train_part = train_texts.sample(n=min(max_rows, len(train_texts)), random_state=SEED)
    test_part = test_texts.sample(n=min(max_rows, len(test_texts)), random_state=SEED)
    texts = pd.concat([train_part, test_part]).to_numpy()
    target = np.array([0] * len(train_part) + [1] * len(test_part))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    classifier = LogisticRegression(C=1, max_iter=1000, class_weight="balanced")
    model = make_pipeline(vectorizer, classifier)
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    scores = cross_validate(model, texts, target, cv=splitter, scoring="roc_auc")
    model.fit(texts, target)
    words = model.named_steps["tfidfvectorizer"].get_feature_names_out()
    coefficients = model.named_steps["logisticregression"].coef_[0]
    train_side, test_side = split_words(coefficients, words, 15)
    return {"auc": scores["test_score"].mean(), "std": scores["test_score"].std(),
            "n_train": len(train_part), "n_test": len(test_part)}, test_side, train_side


def adversarial_auc(frame, target, name):
    # AUC model LightGBM yang membedakan train (0) dan test (1) dari kolom padat saja
    spec = {"kata": False, "char": False, "dense": {name: list(frame.columns)}, "te": []}
    model = make_model(spec, "kelas", 2, "balanced")
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    scores = cross_validate(model, frame, target, cv=splitter, scoring="roc_auc")
    return scores["test_score"].mean(), scores["test_score"].std()


def adversarial_style(train_texts, test_texts, max_rows=5000):
    # sama, tetapi hanya dari statistik gaya (panjang, kapital, tanda baca, ...)
    train_part = train_texts.sample(n=min(max_rows, len(train_texts)), random_state=SEED)
    test_part = test_texts.sample(n=min(max_rows, len(test_texts)), random_state=SEED)
    frame = pd.concat([style_features(train_part), style_features(test_part)])
    target = np.array([0] * len(train_part) + [1] * len(test_part))
    return adversarial_auc(frame, target, "gaya")


def adversarial_numbers(train_numbers, test_numbers, max_rows=5000):
    # sama, tetapi dari kolom angka saja: menangkap geseran angka yang tak terlihat di teks
    train_part = train_numbers.sample(n=min(max_rows, len(train_numbers)), random_state=SEED)
    test_part = test_numbers.sample(n=min(max_rows, len(test_numbers)), random_state=SEED)
    frame = pd.concat([train_part, test_part])
    target = np.array([0] * len(train_part) + [1] * len(test_part))
    return adversarial_auc(frame, target, "angka")


def auc_verdict(auc):
    if auc < 0.45:
        return ("di bawah 0,5: biasanya karena baris kembar di train dan test "
                "(lihat bagian Kebocoran)")
    if auc < 0.55:
        return "mirip: model hampir tidak bisa membedakan train dan test"
    if auc < 0.65:
        return "sedikit beda"
    if auc < 0.8:
        return "cukup beda: waspadai skor validasi yang terlalu optimis"
    return "sangat beda: validasi dari train tidak mewakili test"
