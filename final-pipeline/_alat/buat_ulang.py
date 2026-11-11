"""Bangun ulang notebook final-pipeline dari notebook teruji di repo (contoh/text).

Yang diubah dari aslinya HANYA:
  1. path "../../../" -> "../../" (folder ini ada di root repo; pretrained, outputs di luar repo),
  2. sel Konfigurasi diawali blok NILAI SOAL (SOAL di bawah; tiap notebook mandiri, tanpa konfig.py)
     dan diakhiri jalur final-pipeline (path dari HASIL); baris asli bernama sama dibuang,
  3. sel demo dibuang: "Kotori data demo", "Cek dengan kunci jawaban", cek kotoran demo.
Pakai (dari folder final-pipeline):  python _alat/buat_ulang.py
PERINGATAN: menimpa notebook 1_..8_ di folder ini (ubahan tangan di notebook hilang).
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
SOURCE = os.path.join(REPO, "contoh", "text")
DATA = os.path.join(REPO, "..", "data_latihan")

# Nilai soal, disalin ke awal sel Konfigurasi TIAP notebook (hanya variabel yang dipakai notebook itu).
SOAL = r"""# ============ NILAI SOAL: isi SAMA di semua notebook 1_..7_ (tidak ada konfig.py) ============
# Sesudah mengubah: Restart Kernel + Run All. Nilai awal = data demo SmSA (sentimen, 3 kelas).
# ---- File soal (garis miring /, bukan \) ----
TRAIN_MENTAH = "demo/smsa_train.csv"          # GANTI: mis. "C:/GEMASTIK/soal/train.csv"
TEST_MENTAH = "demo/smsa_test.csv"            # GANTI: mis. "C:/GEMASTIK/soal/test.csv"
SAMPLE_PANITIA = None                         # GANTI: "C:/GEMASTIK/soal/sample_submission.csv"
#                                               None = sample demo dibuat sendiri (format BELUM dicek)
# ---- Kolom ----
ID_COL = "id"                                 # GANTI: kolom id
TEXT_COLS = ["text"]                          # GANTI: kolom teks. 4_finetune hanya memakai yang pertama
TARGET = "label"                              # GANTI: kolom kelas
SUBMIT_COL = TARGET                           # GANTI bila kolom prediksi di sample bernama lain
LABEL_COLS = None                             # MULTILABEL: daftar kolom label 0/1, mis. ["a", "b"].
#                                               None = single label. Bila diisi: pakai 3m, 4m, 5m
GROUP_COL = None                              # kolom penulis/sumber yang berulang (GroupKFold), None = tidak
DATE_COL = None                               # kolom tanggal (tren di EDA dan topik), None = tidak
LANGUAGE = "id"                               # "id" = stopword Indonesia, "en" = Inggris
# ---- Hasil ----
HASIL = "../../outputs/final"                 # GANTI NAMA bila ganti dataset (mis. "../../outputs/lomba"),
#                                               supaya OOF demo tidak tercampur dengan OOF soal
TFIDF_BERAT = False                           # True = 3_baseline memakai kolom <teks>_tfidf (bersih berat).
#                                               Nyalakan hanya bila tabel "Ukur" di 2_praproses naik
TOPIK_BERAT = True                            # True = kata topik 6_topik dari kolom <teks>_tfidf (slang -> baku,
#                                               stopword, stemming). Embedding tetap teks ringan. False = teks ringan
SCORING = "f1_macro"                          # metrik sel ukur 2_praproses

# Coba jalur multilabel di data demo CASA (ulasan mobil, 6 label): buka # di empat baris ini
# TRAIN_MENTAH = "demo/casa_train.csv"
# TEST_MENTAH = "demo/casa_test.csv"
# LABEL_COLS = ["fuel", "machine", "others", "part", "price", "service"]
# HASIL = "../../outputs/final_multilabel"
# Multilabel: EDA, praproses, dan "topik x kelas" butuh SATU kolom kelas. Bila TARGET bukan salah satu
# LABEL_COLS, dipakai label pertama (ganti TARGET ke label yang paling penting).
if LABEL_COLS is not None and TARGET not in LABEL_COLS:
    TARGET = LABEL_COLS[0]
    SUBMIT_COL = TARGET
# ====================================================================================================
"""
SOAL_NAMES = set(re.findall(r"^([A-Z_]+) =", SOAL, flags=re.M))
ALWAYS = {"TARGET", "SUBMIT_COL", "LABEL_COLS"}   # dipakai baris "if LABEL_COLS ..." di akhir blok
HEAD = "\n\n# ---- jalur final-pipeline: dihitung dari HASIL di atas, tidak perlu diubah ----\n"
# sebutan konfig.py lama di teks notebook -> sel Konfigurasi
WORDING = [("di `konfig.py`", "di sel Konfigurasi"), ("di konfig.py", "di sel Konfigurasi"),
           ("(konfig.py)", "(sel Konfigurasi)")]
RAW = 'TRAIN_CSV = TRAIN_MENTAH\nTEST_CSV = TEST_MENTAH\n'
CLEAN = 'TRAIN_CSV = f"{HASIL}/praproses/train_bersih.csv"\n'
CLEAN += 'TEST_CSV = f"{HASIL}/praproses/test_bersih.csv"\n'
OOF = 'OOF_FOLDER = f"{HASIL}/oof"\n'
SAMPLE = 'SAMPLE_CSV = f"{OUTPUT_FOLDER}/sample_submission_demo.csv"\n'
SAMPLE += 'if SAMPLE_PANITIA is not None:\n    SAMPLE_CSV = SAMPLE_PANITIA\n'


def out(name):
    return f'OUTPUT_FOLDER = f"{{HASIL}}/{name}"\n'


TFIDF = ('if TFIDF_BERAT:\n    heavy_cols = []\n    for column in TEXT_COLS:\n'
         '        heavy_cols.append(column + "_tfidf")\n    TEXT_COLS = heavy_cols\n')
# kata topik dari kolom bersih BERAT 2_praproses (slang -> baku, stopword, stemming); embedding tetap ringan
TOPIK = ('TOPIC_TEXT_COLS = TEXT_COLS\nif TOPIK_BERAT:\n    TOPIC_TEXT_COLS = []\n    for column in TEXT_COLS:\n'
         '        TOPIC_TEXT_COLS.append(column + "_tfidf")\n'
         'EXTRA_STOPWORDS += ["urltoken", "mentiontoken", "angkatoken", "emoji", "banget", "enggak"]'
         '   # token buatan + KEEP_WORDS 2_praproses\n')
BERTOPIC = ('DATA_CSV = f"{HASIL}/praproses/train_bersih.csv"\nTEXT_COL = TEXT_COLS[0]\n'
            'LABEL_COL = TARGET\nEMBEDDING_MODEL = "../../pretrained/minilm"\n'
            'N_TOPICS = None           # None = otomatis (HDBSCAN), atau angka = target topik + 1\n'
            + out("bertopic") + 'import os\nos.makedirs(OUTPUT_FOLDER, exist_ok=True)\n')

MULTI = ('if LABEL_COLS is None:\n'
         '    raise ValueError("Isi LABEL_COLS di sel Konfigurasi (kolom label 0/1) untuk multilabel")\n'
         + CLEAN + out("finetune_multilabel") + 'OOF_FOLDER = f"{HASIL}/oof_multilabel"\n'
         + SAMPLE + 'TEXT_COL = TEXT_COLS[0]\nNOTEBOOK = "4m_finetune_multilabel"\n')

# nama baru -> (notebook asal, ekor sel Konfigurasi)
NOTEBOOKS = {
    "1_eda": ("eda_lengkap", RAW + out("eda")),
    "2_praproses": ("praproses_lengkap", RAW + out("praproses") + 'DICTIONARY_FOLDER = "kamus"\n'),
    "3_baseline": ("baseline_oof", CLEAN + out("baseline") + OOF + SAMPLE + TFIDF
                   + 'SLANG_CSV = "kamus/kamus_alay_salsabila.csv"\n'),
    "4_finetune": ("finetune_oof", CLEAN + out("finetune") + OOF + SAMPLE + 'TEXT_A = TEXT_COLS[0]\n'),
    "5_ensemble": ("ensemble_oof", CLEAN + out("ensemble") + OOF + SAMPLE),
    "6_topik": ("topik_lengkap", CLEAN + out("topik") + TOPIK),
    "6b_topik_bertopic": ("topik_bertopic", BERTOPIC),
    "7_laporan_model": ("laporan_model", CLEAN + out("laporan") + OOF),
    "4m_finetune_multilabel": ("finetune_multilabel", MULTI),
}

# (notebook, teks lama, teks baru): wajib ketemu tepat sekali, kalau tidak skrip berhenti
REPLACEMENTS = [
    ("6_topik", "Tautan, sebutan akun, angka, dan stopword dibuang; teks asli tetap\ndipakai untuk embedding.",
     "Tautan, sebutan akun, angka, dan stopword dibuang; teks asli tetap\ndipakai untuk embedding. "
     "`TOPIK_BERAT = True` (konfig.py): kata diambil dari kolom `<teks>_tfidf`\n"
     "2_praproses (slang -> baku, stemming), jadi \"gak\"/\"tidak\" dan \"makanannya\"/\"makan\" menyatu."),
    ("6_topik", "clean_text = raw_text.map(clean_for_topics)",
     "topic_text = data[TOPIC_TEXT_COLS[0]].fillna(\"\").astype(str)   # TOPIK_BERAT di konfig.py\n"
     "for column in TOPIC_TEXT_COLS[1:]:\n"
     "    topic_text = topic_text + \" \" + data[column].fillna(\"\").astype(str)\n"
     "clean_text = topic_text.map(clean_for_topics)\n"
     "print(\"kolom kata topik:\", TOPIC_TEXT_COLS)"),
    ("4_finetune", "../../outputs/text/ensemble_oof/pseudo_label.csv",
     "../../outputs/final/ensemble/pseudo_label.csv"),
    ("6b_topik_bertopic", 'word_counter = CountVectorizer(stop_words="english", min_df=2)',
     "from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory\n"
     "stop_words = None\n"
     'if LANGUAGE == "id":\n'
     "    stop_words = StopWordRemoverFactory().get_stop_words()\n"
     'if LANGUAGE == "en":\n'
     '    stop_words = "english"\n'
     "word_counter = CountVectorizer(stop_words=stop_words, min_df=2)"),
    ("6b_topik_bertopic", 'result.to_csv("topik_dokumen.csv", index=False)',
     'result.to_csv(f"{OUTPUT_FOLDER}/topik_dokumen.csv", index=False)   # per baris: JANGAN dibawa'),
    ("6b_topik_bertopic", 'topic_info.to_csv("info_topik.csv", index=False)',
     'topic_info.drop(columns=["Representative_Docs"]).to_csv(f"{OUTPUT_FOLDER}/info_topik.csv",\n'
     "                                                        index=False)   # tanpa teks asli"),
]
TOPIC_LABEL_MD = """## Topik x label (hanya soal MULTILABEL, `LABEL_COLS` di konfig.py)
Tiap sel = porsi teks di satu topik yang punya label itu (satu teks boleh punya beberapa label,
jadi baris TIDAK berjumlah 100%). **Lift** = porsi itu dibagi porsi label di seluruh train."""
TOPIC_LABEL_CODE = """if LABEL_COLS is not None:
    multi = data[(data["asal"] == "train") & (data["topik"] >= 0)]
    label_share = multi.groupby("topik")[LABEL_COLS].mean()
    label_base = multi[LABEL_COLS].mean()
    label_lift = label_share / label_base.clip(lower=1e-9)
    figure, axis = plt.subplots(figsize=(1.1 * len(LABEL_COLS) + 4, 0.5 * len(label_share) + 2))
    axis.imshow(label_share.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(LABEL_COLS)), LABEL_COLS, rotation=30, ha="right")
    axis.set_yticks(range(len(label_share)), np.array(names)[label_share.index], fontsize=8)
    for row in range(len(label_share)):
        for column in range(len(LABEL_COLS)):
            value = label_share.iloc[row, column]
            axis.text(column, row, f"{100 * value:.0f}%\\nx{label_lift.iloc[row, column]:.1f}",
                      ha="center", va="center", fontsize=8, color=["black", "white"][int(value > 0.6)])
    axis.set_title("Porsi teks berlabel tiap topik (angka kedua = lift)")
    table = label_share.round(3).reset_index()
    table["topik"] = np.array(names)[label_share.index]
    save_figure(figure, "06b_topik_x_label", table)
    for name in LABEL_COLS:
        top_topic = label_lift[name].idxmax()
        add_fact("Topik x label", f"Label {name} (porsi umum {pct(label_base[name])}) paling kuat di "
                 f"{names[top_topic]}: {pct(label_share.loc[top_topic, name])} dari topik itu "
                 f"(lift {label_lift.loc[top_topic, name]:.1f}).")
print("LABEL_COLS:", LABEL_COLS)"""
# sel sisipan: (notebook, judul sel sesudahnya, markdown, kode)
EDA_LABEL_MD = """## 7f. Multilabel (hanya bila `LABEL_COLS` diisi di sel Konfigurasi)
Bagian lain EDA hanya melihat `TARGET` (satu label). Di sini semua label: porsi tiap label, jumlah
label per teks, label yang sering muncul bersama, panjang teks per label, dan teks kembar dengan
kombinasi label berbeda. Label sangat jarang (< 1%) = ambang per label di 3m/4m akan goyah."""
EDA_LABEL_CODE = """if LABEL_COLS is not None:
    labels = train[LABEL_COLS]
    odd = []
    for column in LABEL_COLS:
        if not labels[column].dropna().isin([0, 1]).all():
            odd.append(column)
    if odd:
        raise ValueError(f"kolom label harus berisi 0/1: {odd}. Pecah dulu di 2_praproses (lihat petunjuk)")
    labels = labels.fillna(0).astype(int)
    per_text = labels.sum(axis=1)
    label_share = labels.mean().sort_values(ascending=False)
    together = labels.T.dot(labels).div(labels.sum().clip(lower=1), axis=0)   # baris a, kolom b: P(b | a)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    (100 * label_share).plot.bar(ax=axes[0], color=PALETTE[0], rot=30)
    axes[0].set_ylabel("% teks berlabel")
    per_text.value_counts().sort_index().plot.bar(ax=axes[1], color=PALETTE[1], rot=0)
    axes[1].set_xlabel("jumlah label per teks")
    axes[2].imshow(together.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_xticks(range(len(LABEL_COLS)), LABEL_COLS, rotation=30, ha="right")
    axes[2].set_yticks(range(len(LABEL_COLS)), LABEL_COLS)
    axes[2].grid(False)
    for row in range(len(LABEL_COLS)):
        for column in range(len(LABEL_COLS)):
            value = together.iloc[row, column]
            axes[2].text(column, row, f"{100 * value:.0f}", ha="center", va="center", fontsize=8,
                         color=["black", "white"][int(value > 0.6)])
    axes[2].set_title("P(kolom | baris) %")
    show(figure, "07f_multilabel")
    rare = label_share[label_share < 0.01].index.tolist()
    note(f"multilabel {len(LABEL_COLS)} label: terbanyak {label_share.index[0]} "
         f"({100 * label_share.iloc[0]:.1f}%), tersedikit {label_share.index[-1]} "
         f"({100 * label_share.iloc[-1]:.1f}%), label < 1%: {rare}")
    note(f"label per teks: rata-rata {per_text.mean():.2f}, tanpa label {share(per_text == 0)}%, "
         f">= 2 label {share(per_text >= 2)}%")
    pairs = together.where(~np.eye(len(LABEL_COLS), dtype=bool)).stack().sort_values(ascending=False)
    if len(pairs):
        note(f"pasangan paling sering bersama: bila {pairs.index[0][0]} maka {pairs.index[0][1]} "
             f"{100 * pairs.iloc[0]:.0f}%")
    length_rows = []
    for column in LABEL_COLS:
        length_rows.append({"label": column, "teks": int(labels[column].sum()),
                            "median kata bila 1": train_words[labels[column] == 1].median(),
                            "median kata bila 0": train_words[labels[column] == 0].median()})
    length_table = pd.DataFrame(length_rows)
    length_table.to_csv(f"{OUTPUT_FOLDER}/07f_panjang_per_label.csv", index=False)
    combo = labels.astype(str).agg("".join, axis=1)
    combos_per_text = combo[filled].groupby(train_key[filled]).transform("nunique")
    note(f"teks kembar dengan kombinasi label berbeda: {int((combos_per_text > 1).sum())} baris")
    print(length_table)
print("LABEL_COLS:", LABEL_COLS)"""
INSERTS = [("6_topik", "## Topik train vs test", TOPIC_LABEL_MD, TOPIC_LABEL_CODE),
           ("1_eda", "## 8. Kekuatan tiap kolom", EDA_LABEL_MD, EDA_LABEL_CODE)]
# notebook yang ditulis sendiri (bukan salinan repo): sumber .py bersel di _alat/
OWN = ["3m_baseline_multilabel", "5m_ensemble_multilabel"]
DROP_HEADINGS = ["## Kotori data demo", "## Cek dengan kunci jawaban"]
DROP_CODE = ["# HAPUS saat lomba: cek bahwa semua kotoran demo hilang"]


def keep_cells(cells):
    kept, skip_next = [], False
    for cell in cells:
        source = "".join(cell["source"])
        if skip_next:
            skip_next = False
            continue
        if cell["cell_type"] == "markdown" and source.startswith(tuple(DROP_HEADINGS)):
            skip_next = True                      # sel kode sesudah judul ikut dibuang
            continue
        if cell["cell_type"] == "code" and source.startswith(tuple(DROP_CODE)):
            continue
        kept.append(cell)
    return kept


def config_index(cells):
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("## Konfigurasi"):
            return index + 1
    raise SystemExit("sel '## Konfigurasi' tidak ketemu")


def standalone(cells, own, tail):
    """Sel Konfigurasi mandiri: blok SOAL (variabel yang dipakai notebook) + baris asli sisanya + tail."""
    kept = []
    replaced = SOAL_NAMES | set(re.findall(r"^([A-Z_]+) =", tail, flags=re.M))   # nilai yang ditimpa
    for line in own.rstrip("\n").split("\n"):
        match = re.match(r"([A-Z_]+)\s*=", line)
        if match and match.group(1) in replaced:
            if line.count("[") != line.count("]") or line.count("(") != line.count(")"):
                raise SystemExit(f"baris konfigurasi terpotong, tidak bisa dibuang: {line}")
            continue
        kept.append(line)
    others = tail
    for cell in cells:
        if cell["source"] is not own:
            others += "\n" + "".join(cell["source"])
    for line in list(kept):                        # nilai demo yang tidak dipakai lagi (mis. DATA_FOLDER)
        match = re.match(r"([A-Z_]+)\s*=", line)
        rest_text = "\n".join(other for other in kept if other is not line)
        if match and not re.search(rf"\b{match.group(1)}\b", rest_text + others):
            kept.remove(line)
    rest = "\n".join(kept).strip("\n")
    used = rest + others
    block, owner = [], None
    for line in SOAL.rstrip("\n").split("\n"):
        match = re.match(r"([A-Z_]+) =", line)
        if match:
            owner = match.group(1)
        elif not line.startswith("#" + " " * 10):
            owner = None                           # baris lanjutan komentar ikut variabel di atasnya
        if owner is None or owner in ALWAYS or re.search(rf"\b{owner}\b", used):
            block.append(line)
    for cell in cells:
        for old, new in WORDING:
            cell["source"] = "".join(cell["source"]).replace(old, new)
    return "\n".join(block) + "\n\n" + rest + HEAD + tail.rstrip("\n")


def build(new_name, old_name, tail):
    with open(os.path.join(SOURCE, old_name + ".ipynb"), encoding="utf-8") as file:
        notebook = json.load(file)
    cells = keep_cells(notebook["cells"])
    for cell in cells:
        cell["source"] = "".join(cell["source"]).replace("../../../", "../../")
        cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs")
        if cell["cell_type"] != "code":
            cell.pop("outputs", None)
    for name, old, new in REPLACEMENTS:
        if name == new_name:
            hits = [cell for cell in cells if old in cell["source"]]
            if len(hits) != 1:
                raise SystemExit(f"{new_name}: '{old[:40]}' ketemu {len(hits)} kali, harus 1")
            hits[0]["source"] = hits[0]["source"].replace(old, new)
    for name, before, markdown, code in INSERTS:
        if name == new_name:
            spots = [i for i, cell in enumerate(cells) if cell["source"].startswith(before)]
            if len(spots) != 1:
                raise SystemExit(f"{new_name}: judul '{before}' ketemu {len(spots)} kali")
            extra = [{"cell_type": "markdown", "metadata": {}, "source": markdown},
                     {"cell_type": "code", "metadata": {}, "source": code, "outputs": [],
                      "execution_count": None}]
            cells[spots[0]:spots[0]] = extra
    config = cells[config_index(cells)]
    config["source"] = standalone(cells, config["source"], tail)
    notebook["cells"] = cells
    with open(os.path.join(ROOT, new_name + ".ipynb"), "w", encoding="utf-8") as file:
        json.dump(notebook, file, ensure_ascii=False, indent=1)
    print(f"{new_name}.ipynb  <- contoh/text/{old_name}.ipynb  ({len(cells)} sel)")


def copy_data():
    """Kamus alay + data demo ikut di folder ini, supaya tidak bergantung pada data_latihan."""
    pairs = [("preprocessing/kamus_alay_salsabila.csv", "kamus"), ("preprocessing/kamus_slang.csv", "kamus"),
             ("text/sentiment/smsa_train.csv", "demo"), ("text/sentiment/smsa_test.csv", "demo"),
             ("text/multilabel/casa_train.csv", "demo"), ("text/multilabel/casa_test.csv", "demo")]
    for relative, folder in pairs:
        os.makedirs(os.path.join(ROOT, folder), exist_ok=True)
        target = os.path.join(ROOT, folder, os.path.basename(relative))
        if not os.path.exists(target):
            shutil.copy2(os.path.join(DATA, relative), target)
            print("disalin:", os.path.relpath(target, ROOT))


for new_name, (old_name, tail) in NOTEBOOKS.items():
    build(new_name, old_name, tail)
for name in OWN:
    subprocess.run([sys.executable, os.path.join(REPO, "_alat", "py2nb.py"),
                    os.path.join(HERE, name + ".py"), os.path.join(ROOT, name + ".ipynb")], check=True)
    path = os.path.join(ROOT, name + ".ipynb")
    with open(path, encoding="utf-8") as file:
        notebook = json.load(file)
    configs = [cell for cell in notebook["cells"] if 'exec(open("konfig.py"' in "".join(cell["source"])]
    if len(configs) != 1:
        raise SystemExit(f"{name}: sel exec(konfig.py) ketemu {len(configs)} kali, harus 1")
    lines = "".join(configs[0]["source"]).split("\n")
    cut = [i for i, line in enumerate(lines) if line.startswith("# ---- final-pipeline")][0]
    configs[0]["source"] = "\n".join(lines[:cut])
    configs[0]["source"] = standalone(notebook["cells"], configs[0]["source"],
                                      "\n".join(lines[cut + 2:]))   # buang komentar + baris exec
    with open(path, "w", encoding="utf-8") as file:
        json.dump(notebook, file, ensure_ascii=False, indent=1)
    print(f"{name}.ipynb  <- _alat/{name}.py")
copy_data()
