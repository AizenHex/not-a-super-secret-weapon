"""Mengukur klaim di METODE.md (tanpa Streamlit). Jalankan dari folder _laya_eda:
    python _uji/ukur_metode.py
Memakai GPU sebentar: Laya 500 baris saja (bagian 3). Bagian lain hanya CPU."""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))
sys.path.insert(0, os.getcwd())

import eda
import eda_teks
import helpers
import outliers

SEED = 42
EMOT_CSV = "../../data_latihan/text/classification/emot_train.csv"
ROW_COLUMNS = ["judul", "deskripsi", "harga", "alamat", "luas_m2"]
PROBLEMS = ["harga_mustahil", "harga_teks", "luas_negatif", "alamat_kabur", "deskripsi_kosong",
            "salah_jenis"]


def title(text):
    print()
    print("=" * 8, text)


# ---------- 1. outlier harga: per kategori vs tanpa kategori ----------
def measure_price_outliers(properti):
    title("1. outlier harga skala log (jenis_asli sebagai kategori)")
    price = helpers.to_number(properti["harga"], True)
    impossible = properti["masalah"] == "harga_mustahil"
    clean = properti["masalah"].isna()
    checked = outliers.compare_outliers(price, properti["jenis_asli"], True)
    in_group = pd.Series(False, index=properti.index)
    in_group[checked.index] = checked["outlier"]
    overall = outliers.iqr_outliers(price, True).fillna(False)
    print("harga mustahil:", int(impossible.sum()), "| baris normal:", int(clean.sum()))
    print("per kategori : tertangkap", int((in_group & impossible).sum()),
          "| baris normal ikut", int((in_group & clean).sum()))
    print("tanpa kategori: tertangkap", int((overall & impossible).sum()),
          "| baris normal ikut", int((overall & clean).sum()))


# ---------- 2. cakupan alasan outlier ----------
def fake_context(properti):
    # ctx secukupnya untuk outliers.collect_reasons (tanpa Laya dan tanpa vektor)
    numbers = pd.DataFrame(index=properti.index)
    numbers["harga"] = helpers.to_number(properti["harga"], True)
    numbers["luas_m2"] = helpers.to_number(properti["luas_m2"], True)
    area = numbers["luas_m2"].where(numbers["luas_m2"] > 0)
    numbers["harga_per_luas_m2"] = numbers["harga"] / area
    return {"data": properti, "texts": helpers.row_texts(properti, ROW_COLUMNS),
            "label_column": "jenis_asli", "use_log": True, "lens_done": [], "lenses": {},
            "doubt_limit": 0.6, "distances": None, "reason_dim": "jenis_asli",
            "numbers": numbers, "table": properti.copy()}


def measure_reasons(properti):
    title("2. cakupan alasan (angka + duplikat saja, tanpa Laya dan tanpa vektor teks)")
    ctx = fake_context(properti)
    start = time.time()
    parts = outliers.collect_reasons(ctx)
    reasons = outliers.join_reasons(parts, properti.index)
    print("waktu kumpulkan alasan:", round(time.time() - start, 2), "dtk untuk", len(properti),
          "baris")
    has_reason = reasons != ""
    for problem in PROBLEMS:
        mask = properti["masalah"] == problem
        print(f"{problem:16s} {int(mask.sum()):3d} baris | punya alasan: "
              f"{100 * has_reason[mask].mean():5.1f}%")
    clean = properti["masalah"].isna()
    print(f"baris normal {int(clean.sum())} | punya alasan: {100 * has_reason[clean].mean():.1f}%")


# ---------- 3. pilihan kedua Laya dan kata 'ragu' ----------
def run_laya_sample(properti):
    import laya
    from lenses import PRESETS
    judge = laya.load("../../pretrained/laya", subfolder="multilingual", device="cuda")
    sample = properti.sample(n=500, random_state=SEED).index
    start = time.time()
    answers = helpers.answer_chunk(judge, properti, sample, ROW_COLUMNS, None, "jenis_properti",
                                   PRESETS["jenis_properti"])
    seconds = time.time() - start
    print("waktu:", round(seconds, 1), "dtk ->", round(500 / seconds), "baris/dtk")
    return answers.set_index("row")


def measure_laya_second(properti):
    title("3. Laya jenis_properti: pilihan pertama / kedua, 500 baris (GPU)")
    answers = run_laya_sample(properti)
    truth = properti["jenis_asli"].reindex(answers.index)
    valid = properti["masalah"].reindex(answers.index) != "salah_jenis"
    first_ok = (answers["answer"] == truth)[valid]
    second_ok = (answers["second"] == truth)[valid]
    wrong = ~first_ok
    doubt = answers["confidence"][valid] < 0.6
    print("baris dengan label benar:", int(valid.sum()))
    print("akurasi pilihan pertama:", round(100 * first_ok.mean(), 1), "%")
    both = first_ok | second_ok
    print("benar di pilihan pertama ATAU kedua:", round(100 * both.mean(), 1), "%")
    print("jawaban salah:", int(wrong.sum()), "| dari itu benar di pilihan kedua:",
          int((wrong & second_ok).sum()))
    print("ragu (yakin < 0,6):", int(doubt.sum()), "| akurasi saat ragu:",
          round(100 * first_ok[doubt].mean(), 1), "% | akurasi saat yakin:",
          round(100 * first_ok[~doubt].mean(), 1), "%")
    print("salah yang tertangkap 'ragu':", int((wrong & doubt).sum()), "dari", int(wrong.sum()))


# ---------- 4. konflik label ----------
def measure_conflicts(emot):
    title("4. konflik label")
    texts = helpers.row_texts(emot, ["text"])
    base = eda_teks.label_conflicts(texts, emot["label"], helpers.simple_keys(texts))
    print("emot_train asli: kelompok bentrok", base["kelompok"].nunique(), "| baris", len(base))
    copies = emot.sample(n=100, random_state=SEED).copy()
    labels = sorted(emot["label"].unique())
    new_labels = []
    for label in copies["label"]:
        new_labels.append(labels[(labels.index(label) + 1) % len(labels)])
    copies["label"] = new_labels
    mixed = pd.concat([emot, copies], ignore_index=True)
    texts = helpers.row_texts(mixed, ["text"])
    found = eda_teks.label_conflicts(texts, mixed["label"], helpers.simple_keys(texts))
    print("setelah menyisipkan 100 salinan berlabel beda: kelompok bentrok",
          found["kelompok"].nunique(), "(harapan 100 + bawaan", base["kelompok"].nunique(), ")")


# ---------- 5. nilai tidak konsisten ----------
def make_variant(value, kind, number):
    if kind == "huruf":
        options = [value.upper(), value.lower(), value + " ", " " + value, value + "."]
        return options[number % 5]
    position = 1 + number % (len(value) - 2)
    return value[:position] + value[position + 1:]


def inject_variants(cities, kind):
    column = cities.copy()
    chosen = column.sample(frac=0.1, random_state=SEED).index
    for number, row in enumerate(chosen):
        column[row] = make_variant(column[row], kind, number)
    return column, chosen


def measure_typo_length():
    for word in ["Depok", "Bogor", "Medan", "Malang", "Bandung", "Tangerang", "Yogyakarta"]:
        typo = word[:2] + word[3:]
        series = pd.Series([word] * 50 + [typo] * 3)
        found = eda_teks.inconsistent_values(series)
        print(f"typo '{typo}' (kata {len(word)} huruf) ditemukan: {len(found) == 1}")


def measure_inconsistent(properti):
    title("5. nilai tidak konsisten (kota diambil dari alamat)")
    address = properti["alamat"].dropna()
    cities = address.str.split(", ").str[-1]
    for kind in ["huruf", "salah_ketik"]:
        column, chosen = inject_variants(cities, kind)
        mapping = eda_teks.correction_map(eda_teks.inconsistent_values(column))
        correct = 0
        for row in chosen:
            if mapping.get(column[row]) == cities[row]:
                correct = correct + 1
        wrong_merge = 0
        for value in set(cities):
            if value in mapping:
                wrong_merge = wrong_merge + 1
        print(f"{kind:12s} disisipkan {len(chosen)} baris | diganti ke kota yang benar: {correct}"
              f" | kota bersih yang ikut diganti (salah): {wrong_merge}")
    measure_typo_length()


# ---------- 6. kalibrasi uji ----------
def share_significant(make_column, test, label, trials, generator):
    hits = 0
    for _ in range(trials):
        result = test(make_column(generator), label)
        if result[1] < 0.05:
            hits = hits + 1
    return 100 * hits / trials


def random_category(generator):
    return pd.Series(generator.choice(["a", "b", "c", "d"], 3521))


def random_number(generator):
    return pd.Series(generator.normal(size=3521))


def measure_drift_false_alarm(emot, generator, trials):
    dates = pd.Series(pd.date_range("2024-01-01", periods=len(emot), freq="3h"))
    hits = 0
    for _ in range(trials):
        shuffled = dates.sample(frac=1, random_state=generator.randint(1000000))
        drift = eda.drift_test(shuffled.reset_index(drop=True), emot["label"])
        if drift["p_value"] < 0.05:
            hits = hits + 1
    print(f"pergeseran label, tanggal diacak: p < 0,05 pada {100 * hits / trials:.1f}% dari "
          f"{trials} percobaan")


def measure_tests(emot):
    title("6. uji hubungan & pergeseran")
    texts = helpers.row_texts(emot, ["text"])
    generator = np.random.RandomState(SEED)
    label = emot["label"]
    percent = share_significant(random_category, eda.chi_square_test, label, 300, generator)
    print(f"kolom kategori acak: p < 0,05 pada {percent:.1f}% dari 300 percobaan (harapan ~5%)")
    percent = share_significant(random_number, eda.kruskal_test, label, 300, generator)
    print(f"kolom angka acak: p < 0,05 pada {percent:.1f}% dari 300 percobaan (harapan ~5%)")
    numbers = pd.DataFrame({"id": emot["id"]})
    table = eda.label_relations(emot, "label", numbers, helpers.text_stats(texts))
    print(table[["kolom", "uji", "statistik", "p_value", "ukuran_efek", "efek"]].to_string())
    half = label.where(generator.rand(len(emot)) < 0.5, "x")
    result = eda.chi_square_test(half, label)
    print("kolom separuh salinan label: Cramer V =", round(result[2], 3), "p =", result[1])
    measure_drift_false_alarm(emot, generator, 300)


# ---------- 7. peta makna dan pemisahan kelas ----------
def measure_map(emot):
    title("7. silhouette dan t-SNE (vektor emot_train yang sudah ada)")
    vectors = np.load("hasil/emot_train/vektor.npy")
    rows = pd.read_csv("hasil/emot_train/kelompok.csv")["row"].to_numpy()
    positions = eda_teks.sample_positions(len(vectors), 2000, SEED)
    sample = vectors[positions]
    labels = emot["label"][rows[positions]].to_numpy()
    table, overall = eda_teks.separation_table(sample, labels)
    print("silhouette (2000 sampel):", round(overall, 3))
    print(table.to_string(index=False))
    start = time.time()
    _, explained = eda_teks.tsne_2d(sample, SEED)
    print("t-SNE 2000 titik di CPU:", round(time.time() - start, 1), "dtk | ragam PCA-50:",
          round(100 * explained), "%")
    shuffled = pd.Series(labels).sample(frac=1, random_state=SEED)
    print("silhouette dengan label diacak penuh:",
          round(eda_teks.separation_table(sample, shuffled.to_numpy())[1], 3))


# ---------- 8. teks: n-gram, alay, bahasa ----------
def measure_text(emot):
    title("8. kosakata, alay, campuran bahasa (emot_train)")
    texts = helpers.row_texts(emot, ["text"])
    print(eda_teks.vocabulary_stats(texts))
    ratios = eda_teks.word_ratios(texts, eda_teks.load_slang())
    print("rata-rata rasio alay:", round(100 * ratios["rasio_alay"].mean(), 1), "% | teks campur "
          "bahasa:", round(100 * ratios["campuran_bahasa"].mean(), 1), "%")
    start = time.time()
    ngrams = eda_teks.class_ngrams(texts, emot["label"])
    print("waktu n-gram:", round(time.time() - start, 2), "dtk")
    top = ngrams.groupby("kelas").head(3)
    print(top.to_string(index=False))


if __name__ == "__main__":
    properti = pd.read_csv("contoh/properti.csv")
    emot = pd.read_csv(EMOT_CSV)
    measure_price_outliers(properti)
    measure_reasons(properti)
    measure_laya_second(properti)
    measure_conflicts(emot)
    measure_inconsistent(properti)
    measure_tests(emot)
    measure_map(emot)
    measure_text(emot)
