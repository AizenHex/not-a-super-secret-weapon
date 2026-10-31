"""Tab 'Train vs test': apakah data test mirip data train? Sebaran, kosakata, kebocoran,
adversarial validation, dan (opsional) sebaran jawaban lensa Laya. Dipakai lewat show(ctx).
Logika hitungan ada di features.py, file ini hanya menggambar tampilan."""
import io
import os

import pandas as pd
import streamlit as st

import features
import helpers
import kamus

DEFAULT_TEST = "../../data_latihan/text/classification/emot_test.csv"
LAYA_FOLDER = "../../pretrained/laya"
MAX_LAYA_ROWS = 300

EXPLAIN = {
    "sebaran": {
        "apa": "Membandingkan sebaran panjang teks, kolom angka, dan kolom kategori antara "
               "train dan test.",
        "cara": "Angka (panjang teks, jumlah kata, kolom angka): uji Kolmogorov-Smirnov dua "
                "sampel, D = jarak terbesar antara dua kurva kumulatif (0 = sama, 1 = beda "
                "total). Kategori: uji chi-square, ukuran efek Cramér's V, dan PSI = Σ (p_test "
                "- p_train) × ln(p_test / p_train), PSI < 0,1 stabil, 0,1-0,25 bergeser, "
                "> 0,25 beda besar.",
        "kenapa": "Model belajar dari train, bila test berbeda (lebih panjang, kategori baru, "
                  "angka lain), skor validasi menjadi terlalu optimis. Tahu lebih awal berarti "
                  "bisa memilih fitur yang tahan geser.",
        "batasan": "Dengan baris banyak, uji KS menganggap beda kecil pun 'nyata' (p kecil), "
                   "nilai D dan grafik lebih penting daripada p. PSI sensitif pada kategori "
                   "langka. Uji ini per kolom, tidak melihat kombinasi kolom.",
    },
    "kosakata": {
        "apa": "Seberapa banyak kata di test yang belum pernah dilihat model saat training.",
        "cara": "Kata = potongan huruf/angka kecil. Kamus train = semua kata di train. Kata "
                "test yang tidak ada di kamus = OOV (out-of-vocabulary). Dilaporkan % kata, "
                "% baris test yang punya minimal satu kata baru, dan kata baru terbanyak.",
        "kenapa": "Model berbasis kata tidak bisa memakai kata yang tidak dikenal. OOV tinggi "
                  "= pertimbangkan TF-IDF karakter, MiniLM, atau menambah data.",
        "batasan": "Tanpa stemming: 'makan' dan 'makanan' dianggap kata beda. Nama/angka "
                   "yang unik wajar menjadi OOV.",
    },
    "kebocoran": {
        "apa": "Teks test yang sama dengan teks train, beserta label di train.",
        "cara": "Cocokkan teks persis, lalu setelah huruf kecil dan tanda baca dibuang. "
                "Hasilnya memuat label di train (kelas terbanyak) dan apakah labelnya bentrok.",
        "kenapa": "Teks kembar bisa dijawab langsung dari train (menghemat model), tetapi bila "
                  "labelnya bentrok itu tanda label berisik. Juga memperingatkan bila validasi "
                  "internal terlalu mudah karena duplikat.",
        "batasan": "Hanya mendeteksi kesamaan (hampir) persis, bukan kalimat yang artinya sama "
                   "tapi kata-katanya beda.",
    },
    "adversarial": {
        "apa": "Melatih model untuk menebak 'baris ini dari train atau test?'.",
        "cara": "Teks train diberi label 0 dan test label 1, lalu TF-IDF + LogisticRegression "
                "dinilai dengan cross-validation 5 lipatan memakai AUC. AUC ≈ 0,5 = tidak "
                "bisa dibedakan (mirip), AUC tinggi = mudah dibedakan (beda). Koefisien "
                "terbesar menunjukkan kata yang paling membedakan. Dilengkapi AUC dari "
                "statistik gaya saja.",
        "kenapa": "Satu angka yang merangkum seluruh perbedaan, termasuk yang tidak terpikir "
                  "saat memeriksa kolom satu per satu. Kata pembeda langsung menunjuk sumber "
                  "geseran.",
        "batasan": "Test kecil membuat AUC berisik (lihat std antar lipatan). Model hanya "
                   "melihat teks (dan gaya), bukan kolom angka/kategori lain.",
    },
    "laya": {
        "apa": "Menjalankan lensa Laya yang sama pada sampel test lalu membandingkan sebaran "
               "jawabannya dengan train.",
        "cara": "Maksimal 300 baris test, dengan tombol. Jawaban tiap lensa dihitung persennya "
                "di train dan di test, lalu dibandingkan dengan chi-square, Cramér's V, dan PSI.",
        "kenapa": "Laya membaca makna: bila sebaran emosi/topik bergeser, test memang berbeda "
                  "isi walau panjang dan kosakatanya mirip.",
        "batasan": "Sampel test kecil sehingga persen kasar. CPU lambat, GPU dapat membuat "
                   "laptop mati bila dipakai berat, jadi default CPU.",
    },
}


# ================= bantu =================

def explain(key, jury):
    text = EXPLAIN[key]
    with st.expander("ℹ️ Cara kerja & kenapa"):
        st.markdown(f"**Apa**  \n{text['apa']}")
        st.markdown(f"**Cara (rumus / langkah)**  \n{text['cara']}")
        st.markdown(f"**Kenapa dipilih**  \n{text['kenapa']}")
        st.markdown(f"**Batasan**  \n{text['batasan']}")
        if jury is None:
            jury = "(kalimat berangka muncul setelah hitungannya jalan)"
        st.markdown(f"**Kalimat untuk juri**  \n> {jury}")


def add_finding(ctx, section, finding, reason, advice, importance):
    kept = []
    for item in ctx["findings"]:
        if item["bagian"] != section:
            kept.append(item)
    kept.append({"bagian": section, "temuan": finding, "alasan": reason, "saran": advice,
                 "penting": importance})
    ctx["findings"][:] = kept


def clicked(key, label):
    # tombol yang 'diingat': hasil tetap tampil walau widget lain diklik
    if st.button(label, key=f"test_tombol_{key}"):
        st.session_state[f"test_flag_{key}"] = True
    return st.session_state.get(f"test_flag_{key}", False)


def forget_old_results(signature):
    if st.session_state.get("test_signature") == signature:
        return
    st.session_state["test_signature"] = signature
    for key in list(st.session_state.keys()):
        if key.startswith("test_flag_"):
            del st.session_state[key]


@st.cache_data(show_spinner=False)
def read_csv_file(path, modified):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_csv_bytes(content):
    return pd.read_csv(io.BytesIO(content))


@st.cache_resource
def load_laya(device):
    import laya
    return laya.load(LAYA_FOLDER, subfolder="multilingual", device=device)


def read_test(path):
    # CSV test dari unggahan atau path, None bila file tidak ada
    uploaded = st.file_uploader("... atau unggah CSV test", type="csv", key="test_unggah")
    if uploaded is not None:
        return read_csv_bytes(uploaded.getvalue()), uploaded.name
    if os.path.exists(path):
        return read_csv_file(path, os.path.getmtime(path)), path
    st.error(f"File tidak ditemukan: {path}")
    return None, path


def load_test(ctx):
    path = st.text_input("File CSV test (path relatif ke folder _laya_eda)", DEFAULT_TEST,
                         key="test_path")
    test, name = read_test(path)
    if test is None:
        return None, name
    missing = []
    for column in ctx["columns"]:
        if column not in test.columns:
            missing.append(column)
    if len(missing) > 0:
        st.error(f"Kolom ini ada di train tetapi tidak ada di test: {missing}")
        return None, name
    return test.reset_index(drop=True), name


# ================= 1. sebaran =================

def verdict_psi(psi):
    if psi < 0.1:
        return "stabil"
    if psi < 0.25:
        return "bergeser"
    return "BEDA BESAR"


def test_numbers(ctx, test):
    # kolom angka test diuraikan dengan aturan yang sama seperti train
    indonesian = st.session_state.get("test_format_indonesia", True)
    numbers = pd.DataFrame(index=test.index)
    for column in ctx["numbers"].columns:
        if column in test.columns:
            numbers[column] = helpers.to_number(test[column], indonesian)
    return numbers


def show_ks_table(ctx, test, test_stats):
    st.checkbox("Kolom angka test berformat Indonesia (titik = ribuan)", value=True,
                key="test_format_indonesia")
    lines = [features.ks_line("panjang teks", ctx["stats"]["panjang"], test_stats["panjang"]),
             features.ks_line("jumlah kata", ctx["stats"]["jumlah_kata"],
                              test_stats["jumlah_kata"])]
    numbers = test_numbers(ctx, test)
    for column in numbers.columns:
        lines.append(features.ks_line(column, ctx["numbers"][column], numbers[column]))
    table = pd.DataFrame(lines)
    kamus.show_table(table, hide_index=True)
    return table


def show_length_charts(ctx, test_stats):
    left, right = st.columns(2)
    length = features.percent_histogram(ctx["stats"]["panjang"], test_stats["panjang"])
    left.write("Panjang teks (huruf), % baris")
    left.line_chart(length)
    words = features.percent_histogram(ctx["stats"]["jumlah_kata"], test_stats["jumlah_kata"])
    right.write("Jumlah kata, % baris")
    right.line_chart(words)


def show_category_shift(ctx, test):
    lines = []
    for column in features.category_candidates(ctx):
        counts, result = features.category_shift(ctx["data"][column], test[column])
        result["kolom"] = column
        result["kesimpulan"] = verdict_psi(result["psi"])
        lines.append(result)
        with st.expander(f"Kolom kategori {column}: PSI {result['psi']}"):
            st.bar_chart(100 * counts / counts.sum())
    if len(lines) == 0:
        st.caption("Tidak ada kolom kategori (2-50 nilai) di kolom yang dibaca.")
        return None
    table = pd.DataFrame(lines)[["kolom", "chi2", "p", "cramer_v", "psi", "kesimpulan"]]
    kamus.show_table(table, hide_index=True)
    return table


def distribution_sentence(ks_table, category_table):
    text = ks_table.iloc[0]
    different = ks_table[ks_table["kesimpulan"] == "BEDA"]
    sentence = (f"Panjang teks train vs test: median {text['median_train']:.0f} vs "
                f"{text['median_test']:.0f}, KS D = {text['D']} (p = {text['p']}), "
                f"{len(different)} dari {len(ks_table)} ukuran numerik berbeda nyata.")
    if category_table is not None:
        worst = category_table.sort_values("psi", ascending=False).iloc[0]
        sentence = (f"{sentence} Kolom kategori paling bergeser: {worst['kolom']} "
                    f"(PSI {worst['psi']}, {worst['kesimpulan']}).")
    return sentence


def section_distribution(ctx, test, test_texts):
    st.markdown("### 1. Sebaran train vs test")
    test_stats = helpers.text_stats(test_texts)
    show_length_charts(ctx, test_stats)
    ks_table = show_ks_table(ctx, test, test_stats)
    category_table = show_category_shift(ctx, test)
    sentence = distribution_sentence(ks_table, category_table)
    st.success(sentence)
    add_finding(ctx, "Sebaran train vs test", sentence, "Uji KS, chi-square, PSI.",
                "Bila ada kolom BEDA, jangan percaya skor validasi train begitu saja.", 0.6)
    explain("sebaran", sentence)


# ================= 2. kosakata =================

def section_vocabulary(ctx, test_texts):
    st.markdown("### 2. Kosakata: kata test yang tidak ada di train")
    summary, new_words = features.vocabulary_gap(ctx["texts"], test_texts)
    first, second, third = st.columns(3)
    first.metric("kata test yang baru", f"{summary['persen_kata_baru']:.1f}%")
    second.metric("baris test dengan kata baru", f"{summary['persen_baris_ada_kata_baru']:.1f}%")
    third.metric("jenis kata baru", summary["jenis_kata_baru"])
    kamus.show_table(new_words)
    sentence = (f"{summary['persen_kata_baru']:.1f}% kata di test tidak pernah muncul di train "
                f"({summary['kata_baru']} dari {summary['kata_test']} kata), "
                f"{summary['persen_baris_ada_kata_baru']:.0f}% baris test memuat minimal satu "
                "kata baru.")
    st.success(sentence)
    add_finding(ctx, "Kosakata test", sentence, "Kata test di luar kamus train (OOV).",
                "OOV tinggi: pakai TF-IDF karakter atau MiniLM agar tahan kata baru.", 0.5)
    explain("kosakata", sentence)


# ================= 3. kebocoran =================

def section_overlap(ctx, test_texts):
    st.markdown("### 3. Kebocoran: teks test yang sama dengan train")
    labels = None
    if ctx["label_column"] is not None:
        labels = ctx["data"][ctx["label_column"]]
    overlaps = features.find_overlaps(ctx["texts"], labels, test_texts)
    percent = 100 * len(overlaps) / len(test_texts)
    sentence = (f"{len(overlaps)} dari {len(test_texts)} teks test ({percent:.1f}%) "
                "sudah ada di train.")
    if len(overlaps) > 0:
        exact = int((overlaps["jenis"] == "persis").sum())
        conflict = int(overlaps["label_bentrok"].sum())
        sentence = (f"{sentence} {exact} persis sama, {conflict} punya label train yang bentrok "
                    "(teks sama, label beda).")
        kamus.show_table(overlaps.head(300), hide_index=True)
    st.success(sentence)
    add_finding(ctx, "Kebocoran train-test", sentence, "Cocokkan teks persis / ternormalisasi.",
                "Jawab teks kembar dari train atau hapus dari validasi.",
                0.5 + min(percent, 50) / 100)
    explain("kebocoran", sentence)


# ================= 4. adversarial validation =================

@st.cache_data(show_spinner=False)
def cached_adversarial(train_texts, test_texts, train_numbers, test_numbers):
    text_result = features.adversarial_validation(train_texts, test_texts)
    style_result = features.adversarial_style(train_texts, test_texts)
    number_result = None
    if train_numbers.shape[1] > 0:
        number_result = features.adversarial_numbers(train_numbers, test_numbers)
    return text_result, style_result, number_result


def show_adversarial_metrics(summary, style_result, number_result):
    columns = st.columns(3)
    columns[0].metric("AUC dari teks (TF-IDF)", f"{summary['auc']:.3f}", f"± {summary['std']:.3f}")
    columns[1].metric("AUC dari statistik gaya", f"{style_result[0]:.3f}",
                      f"± {style_result[1]:.3f}")
    if number_result is not None:
        columns[2].metric("AUC dari kolom angka", f"{number_result[0]:.3f}",
                          f"± {number_result[1]:.3f}")


def adversarial_sentence(summary, number_result):
    sentence = (f"Model yang membedakan train dari test mencapai AUC {summary['auc']:.2f} "
                f"(± {summary['std']:.2f}, 0,5 = tak bisa dibedakan) pada {summary['n_train']} "
                f"train + {summary['n_test']} test: {features.auc_verdict(summary['auc'])}.")
    if number_result is not None:
        sentence = f"{sentence} Dari kolom angka saja AUC {number_result[0]:.2f}."
    return sentence


def show_adversarial(ctx, result):
    (summary, test_side, train_side), style_result, number_result = result
    show_adversarial_metrics(summary, style_result, number_result)
    left, right = st.columns(2)
    left.write("**Kata paling khas TEST** (beda dari train)")
    left.dataframe(test_side, hide_index=True)
    right.write("**Kata paling khas TRAIN**")
    right.dataframe(train_side, hide_index=True)
    sentence = adversarial_sentence(summary, number_result)
    st.success(sentence)
    worst = summary["auc"]
    if number_result is not None:
        worst = max(worst, number_result[0])
    add_finding(ctx, "Adversarial validation", sentence, "AUC CV model train-vs-test.",
                "AUC tinggi: validasi harus meniru test (mis. split berdasarkan sumber/waktu).",
                0.4 + 0.6 * max(worst - 0.5, 0) * 2)
    return sentence


def section_adversarial(ctx, test, test_texts):
    st.markdown("### 4. Adversarial validation")
    sentence = None
    if clicked("adversarial", "Jalankan adversarial validation"):
        numbers = test_numbers(ctx, test)
        result = cached_adversarial(ctx["texts"], test_texts, ctx["numbers"][numbers.columns],
                                    numbers)
        sentence = show_adversarial(ctx, result)
    explain("adversarial", sentence)


# ================= 5. lensa Laya pada sampel test =================

def laya_lenses(ctx):
    names = []
    for name in ctx["lens_done"]:
        if not ctx["lenses"][name].get("with_label"):
            names.append(name)
    return names


def run_laya_on_test(ctx, test, folder, names, n_rows, use_gpu):
    device = "cpu"
    if use_gpu:
        device = "cuda"
    judge = load_laya(device)
    order = test.sample(frac=1, random_state=features.SEED).index[:n_rows]
    for name in names:
        done = helpers.done_rows(folder, name)
        todo = []
        for row in order:
            if row not in done:
                todo.append(row)
        if len(todo) > 0:
            bar = st.progress(0.0, text=f"lensa {name} ...")
            helpers.run_lens(judge, test, todo, ctx["columns"], None, name,
                             ctx["lenses"][name], folder, bar)


def compare_lens(ctx, name, answers):
    train = ctx["table"][name].dropna()
    counts, result = features.category_shift(train, answers)
    percent = (100 * counts / counts.sum()).round(1)
    st.write(f"**{name}**: PSI {result['psi']} ({verdict_psi(result['psi'])}), "
             f"chi-square p = {result['p']}, Cramér's V = {result['cramer_v']}")
    st.bar_chart(percent)
    return result["psi"]


def compare_all_lenses(ctx, results, names):
    # gambar perbandingan tiap lensa, kembalikan lensa yang paling bergeser
    worst = None
    worst_psi = -1
    for name in names:
        if name not in results.columns:
            continue
        psi = compare_lens(ctx, name, results[name].dropna())
        if psi > worst_psi:
            worst = name
            worst_psi = psi
    return worst, worst_psi


def show_laya_comparison(ctx, test, folder, names):
    results = helpers.lens_results(folder, ctx["lenses"], test.index)
    worst, worst_psi = compare_all_lenses(ctx, results, names)
    if worst is None:
        return None
    count = int(results[worst].notna().sum())
    sentence = (f"Pada {count} baris test yang dibaca Laya, sebaran jawaban lensa '{worst}' "
                f"paling bergeser dari train (PSI {worst_psi:.2f}, {verdict_psi(worst_psi)}).")
    if count < 50:
        sentence = f"{sentence} Contoh baru {count} baris, terlalu kecil untuk disimpulkan."
    st.success(sentence)
    add_finding(ctx, "Lensa Laya train vs test", sentence, "PSI jawaban lensa pada sampel test.",
                "Bila bergeser, isi test memang berbeda dari train.", 0.4)
    return sentence


def section_laya(ctx, test):
    st.markdown("### 5. (Opsional) Lensa Laya pada sampel test")
    names = laya_lenses(ctx)
    sentence = None
    if len(names) == 0:
        st.info("Belum ada lensa Laya yang punya hasil di data train (dan tidak melihat label).")
    else:
        folder = f"{ctx['folder']}/test"
        os.makedirs(folder, exist_ok=True)
        n_rows = st.number_input("Jumlah baris test untuk Laya (maks 300)", 10, MAX_LAYA_ROWS,
                                 50, key="test_laya_baris")
        use_gpu = st.checkbox("Pakai GPU (peringatan: GPU berat dapat mematikan laptop, "
                              "default CPU, lebih lambat)", key="test_laya_gpu")
        if clicked("laya", f"Jalankan Laya pada {int(n_rows)} baris test"):
            run_laya_on_test(ctx, test, folder, names, int(n_rows), use_gpu)
        sentence = show_laya_comparison(ctx, test, folder, names)
    explain("laya", sentence)


# ================= pintu masuk =================

def show(ctx):
    st.subheader("Train vs test")
    test, name = load_test(ctx)
    if test is None:
        return
    forget_old_results((name, len(test), tuple(ctx["columns"]), len(ctx["data"])))
    test_texts = helpers.row_texts(test, ctx["columns"])
    st.caption(f"Train: {len(ctx['data'])} baris · test: {len(test)} baris · kolom dibaca: "
               f"{', '.join(ctx['columns'])}")
    section_distribution(ctx, test, test_texts)
    section_vocabulary(ctx, test_texts)
    section_overlap(ctx, test_texts)
    section_adversarial(ctx, test, test_texts)
    section_laya(ctx, test)
