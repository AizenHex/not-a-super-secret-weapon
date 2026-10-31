"""Tab 'Fitur -> label': fitur mana yang membantu menebak label, kata penentu, label yang
mungkin salah, ketidakseimbangan kelas, dan fitur bocor. Dipakai app.py lewat show(ctx).
Logika hitungan ada di features.py, file ini hanya menggambar tampilan."""
import json
import os

import pandas as pd
import streamlit as st

import features
import kamus

BETTER = features.BETTER_BY

# Penjelasan tiap bagian. "Kalimat untuk juri" dibuat dari angka hasil hitungan.
EXPLAIN = {
    "lab": {
        "apa": "Mencoba kelompok fitur satu per satu dan mengukur apakah skor model naik.",
        "cara": "1. Baseline = TF-IDF kata (1-2 gram) + LogisticRegression.  \n"
                "2. Tiap kelompok: baseline + kelompok itu, dinilai cross-validation 5 lipatan "
                "(label kategori: StratifiedKFold, skor F1 macro, label angka: KFold, skor R²) "
                "pada baris yang SAMA persis dengan baseline-nya.  \n"
                "3. selisih = skor dengan kelompok - skor baseline. **bantu** bila selisih ≥ "
                "+0,01 DAN lebih besar dari std antar lipatan (supaya bukan derau), "
                "**merugikan** bila sebaliknya, selain itu **tidak beda**.  \n"
                "4. Semua yang belajar dari data (TF-IDF, penskala, target encoding) hanya "
                "di-fit pada lipatan latih, jadi tidak bocor. Target encoding memakai "
                "TargetEncoder sklearn (dihitung out-of-fold). Nilai padat distandarkan dan "
                "diperkecil per kelompok agar normnya ≈ 1 seperti TF-IDF.",
        "kenapa": "Fitur yang terasa berguna sering tidak menaikkan skor. Angka CV pada baris "
                  "yang sama memberi keputusan yang bisa dipertanggungjawabkan, dan kelompok "
                  "yang tidak membantu dibuang supaya model lebih sederhana.",
        "batasan": "Selisih yang lebih kecil dari 'std antar lipatan' bisa kebetulan. Kelompok "
                   "diuji satu per satu, jadi kerja sama antar kelompok baru terlihat di baris "
                   "SEMUA. Laya/MiniLM hanya ada di sebagian baris (lihat kolom baris). Hasil "
                   "berlaku untuk model linear + TF-IDF, model lain bisa berbeda.",
    },
    "ablation": {
        "apa": "Menjawab 'apakah lensa Laya menambah informasi di luar kata-kata teks?'",
        "cara": "Pada baris yang sudah dibaca Laya, tiga model diadu dengan CV yang sama: "
                "TF-IDF saja, TF-IDF + lensa Laya, dan lensa Laya saja. Pembanding bawah = "
                "tebakan kelas terbanyak. Lensa yang melihat kolom label tidak dipakai (bocor).",
        "kenapa": "Laya mahal (GPU/waktu). Kalau skor tidak naik, lensanya cukup dipakai untuk "
                  "membaca data, tidak perlu jadi fitur model.",
        "batasan": "Baris yang dibaca Laya biasanya sedikit, sehingga skor berisik. Bandingkan "
                   "selisih dengan 'std antar lipatan'. Model 'Laya saja' memakai LightGBM.",
    },
    "kata": {
        "apa": "Kata (dan pasangan kata) yang paling mendorong model memilih tiap kelas.",
        "cara": "TF-IDF kata 1-2 gram, lalu LogisticRegression. Koefisien positif untuk kelas "
                "itu = kata mendorong ke kelas itu, negatif = menjauhkan. Tabel menampilkan 15 "
                "teratas masing-masing.",
        "kenapa": "Kata penentu mudah dijelaskan ke juri dan mengungkap kata-kata jebakan "
                  "(misalnya nama akun) yang sebenarnya bukan isi teks.",
        "batasan": "Koefisien dipengaruhi kata yang berkorelasi, kata langka bisa berbobot besar "
                   "(min_df=2 menahan sebagian). Ini penjelasan model, bukan sebab-akibat.",
    },
    "permutation": {
        "apa": "Seberapa turun skor model bila satu kolom fitur diacak.",
        "cara": "Breiman (2001): LightGBM dilatih pada 75% baris, di 25% sisanya satu kolom "
                "diacak 5 kali dan dihitung turunnya skor (F1 macro / R²). Turun besar = "
                "kolom penting. Kolom MiniLM tidak ikut (384 kolom, tak bermakna sendiri-sendiri).",
        "kenapa": "Berlaku untuk model apa pun dan dihitung pada data yang belum dilihat model, "
                  "jadi lebih jujur daripada importance bawaan pohon.",
        "batasan": "Kolom yang saling berkorelasi saling menutupi (diacak satu, yang lain "
                   "menggantikan). Nilai negatif kecil = derau.",
    },
    "mi": {
        "apa": "Seberapa banyak tiap fitur memberi tahu tentang label (mutual information).",
        "cara": "mutual_info_classif / mutual_info_regression sklearn pada kolom padat, 0 = "
                "tidak berhubungan, makin besar makin berhubungan, termasuk hubungan "
                "tidak lurus.",
        "kenapa": "Cepat, tidak butuh model, dan menangkap hubungan non-linear yang lolos dari "
                  "korelasi biasa. Pelengkap permutation importance.",
        "batasan": "Tiap fitur dilihat sendiri (tanpa interaksi). Taksiran MI berisik pada baris "
                   "sedikit. Nilainya tidak punya satuan baku, hanya untuk membandingkan.",
    },
    "label_salah": {
        "apa": "Daftar baris yang labelnya kemungkinan salah (confident learning sederhana).",
        "cara": "1. Peluang tiap kelas per baris dari model yang tidak pernah melihat baris itu "
                "(cross_val_predict, predict_proba).  \n"
                "2. Ambang kelas = rata-rata peluang kelas itu pada baris yang berlabel kelas "
                "itu (Northcutt dkk. 2021).  \n"
                "3. Ditandai bila peluang label yang diberikan di bawah ambangnya DAN ada kelas "
                "lain di atas ambangnya sendiri.  \n"
                "4. Diurutkan menurut peluang kelas saran dikurangi peluang label sendiri.",
        "kenapa": "Label salah menurunkan skor dan membuat evaluasi menyesatkan. Ambang per kelas "
                  "membuat kelas yang sulit tidak ditandai berlebihan.",
        "batasan": "Model lemah menandai banyak baris yang sebenarnya benar (presisi rendah). "
                   "Teks yang memang ambigu ikut terjaring. Tandanya untuk DIPERIKSA manual, "
                   "bukan dibuang otomatis.",
    },
    "imbalance": {
        "apa": "Jumlah baris per kelas dan akibatnya pada cara menilai model.",
        "cara": "Hitung baris per kelas, rasio terbesar/terkecil, lalu ukur F1 macro baseline "
                "dengan dan tanpa class_weight='balanced' (CV yang sama).",
        "kenapa": "Pada kelas timpang, akurasi bisa tinggi walau kelas kecil tak pernah tertebak. "
                  "F1 macro memberi bobot sama pada tiap kelas, stratify menjaga proporsi di "
                  "tiap lipatan, class_weight membuat kesalahan di kelas kecil lebih mahal.",
        "batasan": "Ambang rasio (1,5 dan 3) hanya patokan. Efek class_weight diukur pada model "
                   "baseline saja.",
    },
    "bocor": {
        "apa": "Mencari fitur yang sendirian hampir sempurna menebak label (tanda kebocoran).",
        "cara": "Tiap kolom data asli (termasuk id), nomor baris, dan nilai lensa Laya dipakai "
                "SENDIRIAN untuk menebak label dengan CV 5 lipatan (kolom angka: LightGBM, "
                "kolom teks: TF-IDF + LogisticRegression). Skor ≥ 0,95 = CURIGA BOCOR.",
        "kenapa": "Kolom seperti id, urutan baris, atau 'jawaban' yang menyelinap ke fitur "
                  "memberi skor tinggi di validasi tetapi gagal di data test.",
        "batasan": "Fitur yang kuat bisa saja sah (misalnya judul iklan memang menyebut jenis "
                   "properti). Yang perlu dipastikan: apakah kolom itu ADA dan SAMA artinya "
                   "saat data test.",
    },
}


# ================= bantu: ingatan hasil, penjelasan, temuan =================

def forget_old_results(ctx):
    # data berganti -> hasil lama dilupakan dan pilihan fitur kembali ke bawaan
    signature = features.signature(ctx)
    if st.session_state.get("fitur_signature") == signature:
        return
    st.session_state["fitur_signature"] = signature
    st.session_state["fitur_version"] = st.session_state.get("fitur_version", 0) + 1
    for key in list(st.session_state.keys()):
        if key.startswith("fitur_flag_"):
            del st.session_state[key]


def clicked(key, label):
    # tombol yang 'diingat': hasil tetap tampil walau widget lain diklik
    if st.button(label, key=f"fitur_tombol_{key}"):
        st.session_state[f"fitur_flag_{key}"] = True
    return st.session_state.get(f"fitur_flag_{key}", False)


def explain(key, jury):
    text = EXPLAIN[key]
    with st.expander("ℹ️ Cara kerja & kenapa"):
        st.markdown(f"**Apa**  \n{text['apa']}")
        st.markdown(f"**Cara (rumus / langkah)**  \n{text['cara']}")
        st.markdown(f"**Kenapa dipilih**  \n{text['kenapa']}")
        st.markdown(f"**Batasan**  \n{text['batasan']}")
        if jury is None:
            jury = "(jalankan hitungannya dulu, kalimat berangka muncul di sini)"
        st.markdown(f"**Kalimat untuk juri**  \n> {jury}")


def add_finding(ctx, section, finding, reason, advice, importance):
    kept = []
    for item in ctx["findings"]:
        if item["bagian"] != section:
            kept.append(item)
    kept.append({"bagian": section, "temuan": finding, "alasan": reason, "saran": advice,
                 "penting": importance})
    ctx["findings"][:] = kept


def metric_name(task):
    if task == "kelas":
        return "F1 macro"
    return "R²"


def label_task(ctx):
    return features.task_of(ctx["data"][ctx["label_column"]])


def widget_key(name):
    return f"fitur_{name}_{st.session_state.get('fitur_version', 0)}"


# ================= hitungan berat (di-cache) =================

@st.cache_data(show_spinner=False)
def cached_availability(_ctx, signature, options):
    return features.group_availability(_ctx, options)


@st.cache_data(show_spinner=False)
def cached_cv(X, y, spec, task):
    return features.cv_scores(X, y, spec, task)


@st.cache_data(show_spinner=False)
def cached_line(_ctx, signature, groups, options, max_rows):
    return features.score_group(_ctx, list(groups), options, max_rows, cached_cv)


@st.cache_data(show_spinner=False)
def cached_ablation(_ctx, signature, options, max_rows):
    return features.ablation_table(_ctx, options, max_rows)


@st.cache_data(show_spinner=False)
def cached_words(_ctx, signature, max_rows, options):
    X, y, _ = features.assemble(_ctx, ["kata"], options, max_rows)
    return features.top_words(X["teks"], y, label_task(_ctx))


@st.cache_data(show_spinner=False)
def cached_dense_importance(_ctx, signature, groups, options, max_rows):
    X, y, spec = features.assemble(_ctx, list(groups), options, max_rows)
    task = label_task(_ctx)
    permutation = features.permutation_table(X, y, spec, task)
    mutual = features.mutual_info_table(X, y, spec, task)
    return permutation, mutual, len(X)


@st.cache_data(show_spinner=False)
def cached_issues(_ctx, signature, groups, options, max_rows):
    X, y, spec = features.assemble(_ctx, list(groups), options, max_rows)
    classes, proba = features.out_of_fold_proba(X, y, spec)
    return features.label_issues(proba, classes, y), len(y)


@st.cache_data(show_spinner=False)
def cached_noise_test(_ctx, signature, groups, options, max_rows):
    X, y, spec = features.assemble(_ctx, list(groups), options, max_rows)
    noisy, flipped = features.flip_labels(y, 0.10, features.FLIP_SEED)
    classes, proba = features.out_of_fold_proba(X, noisy, spec)
    issues = features.label_issues(proba, classes, noisy)
    flagged = pd.Series(False, index=y.index)
    flagged[issues.index] = True
    mine = features.precision_recall(flagged, flipped)
    mine["presisi_top_k"] = features.precision_at_top(issues, flipped)
    nearest = features.nyasar_flags(_ctx, noisy)
    other = None
    same = None
    if nearest is not None:
        other = features.precision_recall(nearest, flipped[nearest.index])
        same = features.precision_recall(flagged[nearest.index], flipped[nearest.index])
    return mine, other, same, int(flipped.sum()), len(y)


@st.cache_data(show_spinner=False)
def cached_imbalance(_ctx, signature, options, max_rows):
    X, y, spec = features.assemble(_ctx, ["kata"], options, max_rows)
    return features.imbalance_experiment(X, y, spec)


@st.cache_data(show_spinner=False)
def cached_leak(_ctx, signature, max_rows):
    return features.single_feature_scores(_ctx, max_rows)


# ================= pengaturan umum =================

def parse_keywords(keyword_text):
    keywords = []
    for word in keyword_text.split(","):
        if word.strip() != "":
            keywords.append(word.strip().lower())
    return keywords


def pick_options(ctx):
    # pengaturan fitur tabel: dimensi pembanding, kolom kategori, kata kunci, kolom tanggal
    with st.expander("Pengaturan fitur tabel (dimensi, kolom kategori, kata kunci, tanggal)"):
        names = ["(tidak ada)"] + features.dimension_candidates(ctx)
        dimension = st.selectbox("Dimensi untuk 'angka ÷ median kategorinya'", names,
                                 key=widget_key("dimensi"))
        category_columns = st.multiselect("Kolom kategori (frequency & target encoding)",
                                          list(ctx["columns"]),
                                          default=features.category_candidates(ctx),
                                          key=widget_key("kolom_kategori"))
        keyword_text = st.text_input("Kata kunci untuk ekstraksi regex (pisah koma)",
                                     "/bulan,nego,hubungi", key=widget_key("kata_kunci"))
        dates = st.multiselect("Kolom tanggal", list(ctx["data"].columns),
                               default=features.date_candidates(ctx),
                               key=widget_key("opsi_tanggal"))
    chosen_dimension = None
    if dimension != "(tidak ada)":
        chosen_dimension = dimension
    return {"dimension": chosen_dimension, "category_columns": category_columns,
            "keywords": parse_keywords(keyword_text), "date_columns": dates}


def pick_groups(availability):
    chosen = []
    for name in features.GROUPS:
        if name == "kata":
            continue
        columns, rows = availability[name]
        usable = columns > 0
        label = f"{features.GROUPS[name][1]} · {columns} kolom · {rows} baris"
        key = widget_key("grup_" + name)
        checked = st.checkbox(label, value=usable, disabled=not usable, key=key)
        if checked and usable:
            chosen.append(name)
    return chosen


def joint_groups(ctx, chosen, availability):
    # kelompok yang hanya punya sebagian baris (Laya, MiniLM) menyusutkan gabungan, jadi opsional
    full = int(ctx["data"][ctx["label_column"]].notna().sum())
    include = st.checkbox("Sertakan kelompok yang hanya punya sebagian baris (mis. Laya, MiniLM) "
                          "dalam gabungan SEMUA dan importance (barisnya jadi sedikit)",
                          key=widget_key("sertakan_sebagian"))
    kept = []
    for name in chosen:
        if include or availability[name][1] >= 0.9 * full:
            kept.append(name)
    return kept


def dense_groups(chosen):
    names = []
    for name in chosen:
        if name in ["char", "minilm", "target_enc"]:
            continue
        names.append(name)
    return tuple(names)


# ================= 1. lab feature engineering =================

def save_selected(ctx, options, table, task):
    # simpan kelompok yang lolos (selisih ≥ 0,01) untuk dipakai pembuat kode EDA
    selected = []
    for _, row in table.iterrows():
        if row["keputusan"] == "bantu" and "+" not in str(row["grup_kode"]):
            details = features.group_details(ctx, row["grup_kode"], options)
            selected.append({"grup": row["grup_kode"], "judul": row["grup fitur"],
                             "selisih": round(float(row["selisih"]), 4), "detail": details})
    content = {"tugas": task, "metrik": metric_name(task), "label": ctx["label_column"],
               "fitur_terpilih": selected, "parameter": options}
    os.makedirs(ctx["folder"], exist_ok=True)
    path = f"{ctx['folder']}/fitur_terpilih.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, default=str)
    return path, len(selected)


def lab_table(lines):
    table = pd.DataFrame(lines)
    table["grup_kode"] = table["grup"]
    titles = []
    for name in table["grup"]:
        if "+" in name:
            titles.append("SEMUA yang dicentang (bersama)")
        else:
            titles.append(features.GROUPS[name][1])
    table["grup fitur"] = titles
    shown = table[["grup fitur", "baris", "skor_baseline", "skor_dengan_grup", "selisih",
                   "keputusan", "std_fold", "detik", "grup_kode"]]
    return shown.round(4)


def lab_sentence(table, task):
    singles = table[~table["grup_kode"].str.contains(r"\+")]
    helpful = singles[singles["keputusan"] == "bantu"]
    helped = len(helpful)
    best = singles.sort_values("selisih", ascending=False).iloc[0]
    if helped > 0:
        best = helpful.sort_values("selisih", ascending=False).iloc[0]
    start = (f"Kami menguji {len(singles)} kelompok fitur dengan cross-validation 5 lipatan "
             f"({metric_name(task)}), baseline TF-IDF kata. ")
    if helped == 0:
        return (f"{start}Tidak ada kelompok yang menaikkan skor ≥ {BETTER}: selisih terbesar "
                f"hanya {best['selisih']:+.3f} ('{best['grup_kode']}', {best['baris']} baris), "
                "jadi baseline sudah cukup.")
    return (f"{start}Yang paling membantu: '{best['grup_kode']}' ({best['selisih']:+.3f} dari "
            f"baseline {best['skor_baseline']:.3f} pada {best['baris']} baris). "
            f"{helped} kelompok membantu dan {len(singles) - helped} tidak.")


def run_lab(ctx, options, chosen, joint, max_rows):
    signature = features.signature(ctx)
    lines = []
    bar = st.progress(0.0, text="menghitung ...")
    for number, name in enumerate(chosen):
        lines.append(cached_line(ctx, signature, (name,), options, max_rows))
        bar.progress((number + 1) / (len(chosen) + 1), text=f"selesai: {name}")
    if len(joint) >= 2:
        lines.append(cached_line(ctx, signature, tuple(joint), options, max_rows))
    bar.empty()
    return lines


def show_lab_result(ctx, options, lines):
    task = label_task(ctx)
    table = lab_table(lines)
    kamus.show_table(table.drop(columns=["grup_kode"]), hide_index=True)
    st.bar_chart(table, x="grup fitur", y="selisih", horizontal=True)
    st.caption(f"skor = {metric_name(task)} (makin besar makin baik). Baseline = TF-IDF kata. "
               "Tiap kelompok dibandingkan dengan baseline pada baris yang sama.")
    sentence = lab_sentence(table, task)
    path, count = save_selected(ctx, options, table, task)
    st.success(f"{sentence}  \nFitur yang lolos ({count} kelompok) disimpan di `{path}`.")
    add_finding(ctx, "Fitur → label", sentence,
                f"{count} kelompok fitur menaikkan {metric_name(task)} ≥ {BETTER}.",
                f"Pakai kelompok yang 'bantu' (lihat {path}).", 0.7)
    return sentence


def section_lab(ctx, options, chosen, joint, max_rows):
    st.markdown("### 1. Lab feature engineering")
    st.write("Centang calon fitur, lalu jalankan. Setiap kelompok diuji dengan cross-validation "
             "dan diputuskan **bantu / tidak beda / merugikan** dibanding baseline.")
    sentence = None
    if len(chosen) == 0:
        st.info("Belum ada kelompok fitur yang dicentang.")
    elif clicked("lab", "Jalankan uji fitur"):
        lines = run_lab(ctx, options, chosen, joint, max_rows)
        sentence = show_lab_result(ctx, options, lines)
    explain("lab", sentence)


# ================= 2. ablation Laya =================

def ablation_sentence(table, task):
    scores = {}
    for _, row in table.iterrows():
        scores[row["model"]] = row
    both = scores["TF-IDF + lensa Laya"]
    only_text = scores["TF-IDF saja"]
    gain = both["skor"] - only_text["skor"]
    verdict = "TIDAK menambah informasi yang jelas"
    if gain >= BETTER and gain > both["std"]:
        verdict = "MENAMBAH informasi"
    return (f"Pada {int(only_text['baris'])} baris yang dibaca Laya, TF-IDF saja "
            f"{only_text['skor']:.3f}, TF-IDF + Laya {both['skor']:.3f}, Laya saja "
            f"{scores['lensa Laya saja']['skor']:.3f} ({metric_name(task)}): Laya {verdict} "
            f"(selisih {gain:+.3f}, std antar lipatan {both['std']:.3f}).")


def show_ablation(ctx, table):
    task = label_task(ctx)
    shown = table[["model", "baris", "skor", "std", "kedua", "detik"]].round(4)
    shown = shown.rename(columns={"std": "std antar lipatan", "kedua": "akurasi / MAE"})
    kamus.show_table(shown, hide_index=True)
    sentence = ablation_sentence(table, task)
    st.success(sentence)
    add_finding(ctx, "Ablation Laya", sentence, "TF-IDF vs TF-IDF + Laya vs Laya saja (CV).",
                "Pakai lensa sebagai fitur hanya bila selisihnya jelas di atas derau.", 0.6)
    return sentence


def section_ablation(ctx, options, availability, max_rows):
    st.markdown("### 2. Ablation Laya: apakah Laya menambah informasi?")
    columns, rows = availability["laya"]
    sentence = None
    if columns == 0 or rows < 40:
        st.info("Butuh hasil lensa Laya minimal 40 baris (jalankan lensa dulu di tab Lensa Laya).")
    else:
        st.caption(f"{rows} baris sudah punya jawaban Laya. Lensa yang melihat label dilewati.")
        if clicked("ablation", "Jalankan ablation Laya"):
            table = cached_ablation(ctx, features.signature(ctx), options, max_rows)
            sentence = show_ablation(ctx, table)
    explain("ablation", sentence)


# ================= 3. importance =================

def show_class_words(words):
    summary = []
    for label in words:
        positive = words[label][0]
        summary.append({"kelas": label, "kata penentu": ", ".join(positive["kata"].head(8))})
    kamus.show_table(pd.DataFrame(summary), hide_index=True)
    chosen = st.selectbox("Lihat kelas", list(words), key=widget_key("kelas_kata"))
    left, right = st.columns(2)
    left.write(f"**Mendorong ke '{chosen}'** (koefisien positif)")
    left.dataframe(words[chosen][0], hide_index=True)
    right.write(f"**Menjauhkan dari '{chosen}'** (koefisien negatif)")
    right.dataframe(words[chosen][1], hide_index=True)
    return summary


def section_words(ctx, options, max_rows):
    st.markdown("#### 3a. Kata penentu tiap kelas")
    sentence = None
    if clicked("kata", "Hitung kata penentu"):
        words = cached_words(ctx, features.signature(ctx), max_rows, options)
        summary = show_class_words(words)
        first = summary[0]
        sentence = (f"Model TF-IDF + regresi logistik mengenali kelas '{first['kelas']}' dari "
                    f"kata: {first['kata penentu']}, tiap kelas punya kata penentunya sendiri.")
        st.success(sentence)
        add_finding(ctx, "Kata penentu", sentence, "Koefisien regresi logistik pada TF-IDF.",
                    "Cek apakah kata penentu benar-benar isi teks, bukan nama akun.", 0.4)
    explain("kata", sentence)


def importance_sentences(permutation, mutual, rows):
    top = permutation.head(3)
    names = ", ".join(top["fitur"])
    useless = int((permutation["turun_skor"] <= 0).sum())
    first_mi = mutual.iloc[0]
    permutation_text = (f"Pada {rows} baris, fitur terpenting menurut permutation importance: "
                        f"{names} (skor turun {top['turun_skor'].iloc[0]:.3f} bila yang pertama "
                        f"diacak), {useless} dari {len(permutation)} fitur tidak berpengaruh.")
    mutual_text = (f"Fitur paling berhubungan dengan label menurut mutual information adalah "
                   f"'{first_mi['fitur']}' (MI {first_mi['mutual_info']:.3f}).")
    return permutation_text, mutual_text


def show_dense_importance(ctx, permutation, mutual, rows):
    permutation_text, mutual_text = importance_sentences(permutation, mutual, rows)
    st.write("**Permutation importance** (turunnya skor bila kolom diacak)")
    st.bar_chart(permutation.head(25), x="fitur", y="turun_skor", horizontal=True)
    kamus.show_table(permutation.head(40), hide_index=True)
    st.success(permutation_text)
    st.write("**Mutual information** (kolom padat vs label)")
    st.bar_chart(mutual.head(25), x="fitur", y="mutual_info", horizontal=True)
    kamus.show_table(mutual.head(40), hide_index=True)
    st.success(mutual_text)
    add_finding(ctx, "Permutation importance", permutation_text,
                "Skor model turun saat kolom diacak.", "Pertahankan fitur teratas.", 0.5)
    add_finding(ctx, "Mutual information", mutual_text, "MI fitur padat vs label.",
                "Bandingkan dengan permutation importance.", 0.3)
    return permutation_text, mutual_text


def section_dense_importance(ctx, options, joint, max_rows):
    st.markdown("#### 3b. Permutation importance dan 3c. Mutual information")
    groups = dense_groups(joint)
    sentences = [None, None]
    if len(groups) == 0:
        st.info("Centang minimal satu kelompok fitur padat (selain MiniLM) di bagian 1.")
    elif clicked("padat", "Hitung importance fitur padat"):
        signature = features.signature(ctx)
        permutation, mutual, rows = cached_dense_importance(ctx, signature, groups, options,
                                                            max_rows)
        st.caption(f"{rows} baris dipakai (baris yang punya semua kelompok yang dicentang).")
        sentences = show_dense_importance(ctx, permutation, mutual, rows)
    explain("permutation", sentences[0])
    explain("mi", sentences[1])


def section_importance(ctx, options, joint, max_rows):
    st.markdown("### 3. Importance")
    section_words(ctx, options, max_rows)
    section_dense_importance(ctx, options, joint, max_rows)


# ================= 4. pencari label salah =================

def issue_groups(availability):
    choices = {"TF-IDF kata": ["kata"], "TF-IDF kata + karakter": ["kata", "char"]}
    if availability["minilm"][0] > 0:
        choices["TF-IDF kata + MiniLM"] = ["kata", "minilm"]
    name = st.radio("Model pencari", list(choices), horizontal=True, key=widget_key("model_label"))
    return tuple(choices[name])


def show_issues(ctx, issues, rows):
    percent = 100 * len(issues) / rows
    sentence = (f"Dari {rows} baris, {len(issues)} ({percent:.1f}%) ditandai labelnya mungkin "
                "salah (peluang label aslinya di bawah ambang kelasnya, sedangkan kelas lain di "
                "atas ambangnya). Periksa yang teratas dulu.")
    st.success(sentence)
    shown = issues.copy()
    shown.insert(0, "teks", ctx["texts"][issues.index])
    kamus.show_table(shown.head(300).round(3))
    add_finding(ctx, "Label salah", sentence, "Confident learning sederhana (Northcutt dkk. 2021).",
                "Periksa manual baris teratas, jangan dibuang otomatis.", 0.8)
    return sentence


def noise_line(name, result):
    return {"metode": name, "ditandai": result["ditandai"], "presisi": result["presisi"],
            "recall": result["recall"]}


def noise_table(mine, other, same, rows):
    random_line = {"metode": "tebakan acak (dasar)", "ditandai": mine["ditandai"],
                   "presisi": mine["dasar_acak"], "recall": mine["ditandai"] / rows}
    lines = [random_line, noise_line("pencari ini (confident learning)", mine)]
    if other is not None:
        lines.append(noise_line("pencari ini, hanya baris yang punya vektor MiniLM", same))
        lines.append(noise_line("'nyasar' MiniLM (dihitung langsung, baris yang sama)", other))
    lines.append({"metode": "'nyasar' MiniLM (uji sebelumnya, seluruh emot_train)",
                  "ditandai": None, "presisi": features.NYASAR_PRECISION,
                  "recall": features.NYASAR_RECALL})
    return pd.DataFrame(lines)


def show_noise(ctx, result):
    mine, other, same, flipped, rows = result
    kamus.show_table(noise_table(mine, other, same, rows).round(3), hide_index=True)
    sentence = (f"Saat {flipped} dari {rows} label (10%) sengaja diacak, pencari ini menemukan "
                f"{mine['recall']:.0%} di antaranya (recall) dengan presisi {mine['presisi']:.0%} "
                f"(tebakan acak: 10%), jika hanya {flipped} teratas diambil, presisinya "
                f"{mine['presisi_top_k']:.0%}.")
    st.success(sentence)
    add_finding(ctx, "Ukuran pencari label salah", sentence,
                "Label diacak 10% (seed tetap) lalu diukur presisi & recall.",
                "Pakai urutan teratas untuk pemeriksaan manual.", 0.7)
    return sentence


def section_label_issues(ctx, options, availability, max_rows):
    st.markdown("### 4. Pencari label salah")
    sentence = None
    if label_task(ctx) != "kelas":
        st.info("Hanya untuk label kategori.")
        return
    groups = issue_groups(availability)
    signature = features.signature(ctx)
    if clicked("issues", "Cari label yang mungkin salah"):
        issues, rows = cached_issues(ctx, signature, groups, options, max_rows)
        sentence = show_issues(ctx, issues, rows)
    st.write("**Ukur sungguhan:** acak 10% label, lalu lihat berapa yang ketemu.")
    if clicked("noise", "Ukur dengan 10% label diacak"):
        sentence = show_noise(ctx, cached_noise_test(ctx, signature, groups, options, max_rows))
    explain("label_salah", sentence)


# ================= 5. ketidakseimbangan kelas =================

def show_imbalance(ctx):
    y = ctx["data"][ctx["label_column"]].dropna()
    table, ratio = features.class_balance(y)
    left, right = st.columns(2)
    left.dataframe(table)
    right.bar_chart(table["jumlah"], horizontal=True)
    advice = features.imbalance_advice(ratio)
    biggest = table.index[0]
    smallest = table.index[-1]
    sentence = (f"Kelas terbesar '{biggest}' ({table['jumlah'].iloc[0]} baris) berbanding "
                f"'{smallest}' ({table['jumlah'].iloc[-1]} baris) = {ratio:.1f}×: {advice}")
    st.info(sentence)
    add_finding(ctx, "Ketidakseimbangan kelas", sentence,
                f"Rasio kelas terbesar/terkecil {ratio:.1f}.",
                "Nilai dengan F1 macro, gunakan stratify, coba class_weight='balanced'.",
                min(0.8, 0.2 + 0.1 * ratio))
    return sentence


def section_imbalance(ctx, options, max_rows):
    st.markdown("### 5. Ketidakseimbangan kelas")
    if label_task(ctx) != "kelas":
        st.info("Hanya untuk label kategori.")
        return
    sentence = show_imbalance(ctx)
    if clicked("imbalance", "Ukur efek class_weight"):
        plain, weighted = cached_imbalance(ctx, features.signature(ctx), options, max_rows)
        gain = weighted["skor"] - plain["skor"]
        sentence = (f"{sentence} Ukuran: F1 macro {plain['skor']:.3f} tanpa class_weight, "
                    f"{weighted['skor']:.3f} dengan class_weight='balanced' ({gain:+.3f}).")
        st.success(f"F1 macro {plain['skor']:.3f} → {weighted['skor']:.3f} ({gain:+.3f}) "
                   "dengan class_weight='balanced'.")
    explain("imbalance", sentence)


# ================= 6. cek fitur bocor =================

def show_leak(ctx, table):
    kamus.show_table(table.round(4), hide_index=True)
    suspects = table[table["status"] == "CURIGA BOCOR"]
    best = table.iloc[0]
    if len(suspects) > 0:
        names = ", ".join(suspects["fitur"])
        sentence = (f"PERINGATAN: {len(suspects)} fitur sendirian memberi skor ≥ "
                    f"{features.LEAK_SCORE} ({names}), periksa apakah ada dan sama artinya "
                    "saat data test.")
        st.error(sentence)
        importance = 1.0
    else:
        sentence = (f"Tidak ada fitur tunggal yang hampir sempurna, yang tertinggi "
                    f"'{best['fitur']}' dengan skor {best['skor']:.3f}.")
        st.success(sentence)
        importance = 0.3
    add_finding(ctx, "Fitur bocor", sentence, "Skor CV tiap fitur sendirian.",
                "Buang atau perbaiki fitur yang curiga bocor sebelum melatih.", importance)
    return sentence


def section_leak(ctx, max_rows):
    st.markdown("### 6. Cek fitur bocor")
    sentence = None
    if clicked("bocor", "Cek fitur bocor"):
        table = cached_leak(ctx, features.signature(ctx), min(max_rows, 3000))
        sentence = show_leak(ctx, table)
    explain("bocor", sentence)


# ================= pintu masuk =================

def show(ctx):
    st.subheader("Fitur → label")
    if ctx["label_column"] is None:
        st.info("Pilih kolom label di samping dulu: tab ini mengukur fitur terhadap label.")
        return
    forget_old_results(ctx)
    rows_total = len(ctx["data"])
    max_rows = st.slider("Maksimal baris yang dipakai (supaya cepat di CPU)", 200,
                         max(rows_total, 201), min(5000, max(rows_total, 201)), 100,
                         key=widget_key("maks_baris"))
    options = pick_options(ctx)
    availability = cached_availability(ctx, features.signature(ctx), options)
    chosen = pick_groups(availability)
    joint = joint_groups(ctx, chosen, availability)
    section_lab(ctx, options, chosen, joint, max_rows)
    section_ablation(ctx, options, availability, max_rows)
    section_importance(ctx, options, joint, max_rows)
    section_label_issues(ctx, options, availability, max_rows)
    section_imbalance(ctx, options, max_rows)
    section_leak(ctx, max_rows)
