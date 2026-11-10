"""Bangun ulang notebook final-pipeline dari notebook teruji di repo (contoh/text).

Yang diubah dari aslinya HANYA:
  1. path "../../../" -> "../../" (folder ini ada di root repo; pretrained, outputs di luar repo),
  2. sel Konfigurasi ditambah ekor yang membaca konfig.py (satu tempat untuk nilai soal),
  3. sel demo dibuang: "Kotori data demo", "Cek dengan kunci jawaban", cek kotoran demo.
Pakai (dari folder final-pipeline):  python _alat/buat_ulang.py
PERINGATAN: menimpa notebook 1_..8_ di folder ini (ubahan tangan di notebook hilang).
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
SOURCE = os.path.join(REPO, "contoh", "text")
DATA = os.path.join(REPO, "..", "data_latihan")

HEAD = '\n\n# ---- final-pipeline: nilai soal dibaca dari konfig.py (ubah DI SANA, bukan di sini) ----\n'
HEAD += 'exec(open("konfig.py", encoding="utf-8").read())\n'
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
BERTOPIC = ('DATA_CSV = f"{HASIL}/praproses/train_bersih.csv"\nTEXT_COL = TEXT_COLS[0]\n'
            'LABEL_COL = TARGET\nEMBEDDING_MODEL = "../../pretrained/minilm"\n'
            'N_TOPICS = None           # None = otomatis (HDBSCAN), atau angka = target topik + 1\n'
            + out("bertopic") + 'import os\nos.makedirs(OUTPUT_FOLDER, exist_ok=True)\n')

MULTI = ('if LABEL_COLS is None:\n'
         '    raise ValueError("Isi LABEL_COLS di konfig.py (kolom label 0/1) untuk multilabel")\n'
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
    "6_topik": ("topik_lengkap", CLEAN + out("topik")),
    "6b_topik_bertopic": ("topik_bertopic", BERTOPIC),
    "7_laporan_model": ("laporan_model", CLEAN + out("laporan") + OOF),
    "4m_finetune_multilabel": ("finetune_multilabel", MULTI),
}

# (notebook, teks lama, teks baru): wajib ketemu tepat sekali, kalau tidak skrip berhenti
REPLACEMENTS = [
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
INSERTS = [("6_topik", "## Topik train vs test", TOPIC_LABEL_MD, TOPIC_LABEL_CODE)]
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
    config["source"] = config["source"].rstrip("\n") + HEAD + tail.rstrip("\n")
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
    print(f"{name}.ipynb  <- _alat/{name}.py")
copy_data()
