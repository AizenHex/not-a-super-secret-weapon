# %% [markdown]
# # Teks — Anotasi Sendiri (aturan kata kunci + tombol ipywidgets + cari sampel ragu)
# **Kapan dipakai:** ribuan teks **belum berlabel** dan kita harus melabeli sendiri dengan cepat.
#
# **Langkah:**
# 1. Label awal dari kata kunci (`RULES`).
# 2. Kita melabeli sampel lewat tombol, tiap klik disimpan ke CSV.
# 3. Latih TF-IDF + LogisticRegression di teks yang sudah berlabel.
# 4. Model menunjuk teks yang paling ragu → label ulang bagian itu saja (kembali ke langkah 2).
#
# Saat *Run All* tidak ada klik = tidak ada label manual, jadi notebook tetap jalan sampai akhir.
#
# **Data demo:** SmSA (sentimen). Kolom `label` asli **tidak dipakai** sampai sel terakhir.

# %%
import os

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text/sentiment"   # GANTI saat lomba
TEXTS_CSV = f"{DATA_FOLDER}/smsa_train.csv"
MANUAL_CSV = "label_manual.csv"     # hasil klik tombol, ditambah terus tiap klik

TEXT_COL = "text"
ID_COL = "id"
CLASSES = ["positive", "negative", "neutral"]     # satu tombol per kelas
N_QUEUE = 10        # jumlah teks yang dilabeli lewat tombol per putaran
N_UNSURE = 10       # jumlah teks paling ragu yang ditampilkan
SEED = 42

# Kata kunci per kelas (kata yang hampir selalu muncul di satu kelas).
RULES = {
    "positive": ["enak", "bagus", "mantap", "puas", "nyaman", "recommended", "suka", "murah",
                 "ramah", "lezat", "terbaik", "senang", "hebat", "keren", "bersih", "nikmat"],
    "negative": ["jelek", "kecewa", "mahal", "lambat", "kotor", "buruk", "tidak enak", "kurang",
                 "bodoh", "korupsi", "benci", "sampah", "parah", "bohong", "gagal", "payah"],
    "neutral": ["pilkada", "pilgub", "cagub", "jabar", "jatim", "resmi", "ktp", "transaksi",
                "menteri", "koalisi", "golkar", "demokrat", "kpk", "tokopedia", "bukalapak"],
}

# %% [markdown]
# ## Baca data (tanpa memakai label)

# %%
texts = pd.read_csv(TEXTS_CSV)
print(texts.shape)
print(texts[TEXT_COL].isna().sum(), "teks kosong")
texts.head()

# %% [markdown]
# ## Label awal dari aturan kata kunci
# Kelas dengan kata kunci terbanyak menang. Seri atau tanpa kata kunci = belum berlabel.

# %%
def rule_label(text):
    text = text.lower()
    counts = {}
    for name in RULES:
        counts[name] = 0
        for keyword in RULES[name]:
            if keyword in text:
                counts[name] += 1
    highest = max(counts.values())
    winners = []
    for name in counts:
        if counts[name] == highest:
            winners.append(name)
    if highest == 0 or len(winners) > 1:
        return None
    return winners[0]


print(rule_label("makanannya enak dan pelayannya ramah"))

texts["label_aturan"] = texts[TEXT_COL].apply(rule_label)
print("berlabel oleh aturan:", texts["label_aturan"].notna().sum(), "dari", len(texts))
print(texts["label_aturan"].value_counts())

# %% [markdown]
# ## Tombol pelabelan (ipywidgets)
# Klik tombol kelas untuk teks yang tampil. Tiap klik langsung ditulis ke `label_manual.csv`.
# Belum ada klik = sel ini tidak mengubah apa-apa.

# %%
queue = texts[texts["label_aturan"].isna()].sample(N_QUEUE, random_state=SEED)
# queue = unsure   # putaran berikutnya: pakai daftar paling ragu dari sel di bawah
position = 0
text_box = widgets.HTML(value=queue.iloc[0][TEXT_COL])
counter = widgets.Label(value=f"0/{len(queue)}")


def on_click(button):
    global position
    row = queue.iloc[position]
    clicked = pd.DataFrame({"id": [row[ID_COL]], "label": [button.description]})
    clicked.to_csv(MANUAL_CSV, mode="a", header=not os.path.exists(MANUAL_CSV), index=False)
    position = position + 1
    counter.value = f"{position}/{len(queue)}"
    text_box.value = "selesai, lanjut ke sel berikutnya"
    if position < len(queue):
        text_box.value = queue.iloc[position][TEXT_COL]


buttons = []
for name in CLASSES:
    button = widgets.Button(description=name)
    button.on_click(on_click)
    buttons.append(button)
display(widgets.VBox([counter, text_box, widgets.HBox(buttons)]))

# %% [markdown]
# ## Gabung label: label manual menimpa label aturan

# %%
manual = {}
if os.path.exists(MANUAL_CSV):
    manual_table = pd.read_csv(MANUAL_CSV)
    manual = dict(zip(manual_table["id"], manual_table["label"]))
texts["label_kita"] = texts[ID_COL].map(manual)
texts["label_kita"] = texts["label_kita"].fillna(texts["label_aturan"])
print("label manual:", len(manual), "| berlabel total:", texts["label_kita"].notna().sum())

# %% [markdown]
# ## Model: latih di teks yang sudah berlabel
# Pilih SATU model: hapus `#` di baris yang ingin dipakai.

# %%
labeled = texts[texts["label_kita"].notna()]
X_train, X_val, y_train, y_val = train_test_split(
    labeled[TEXT_COL], labeled["label_kita"], test_size=0.2, stratify=labeled["label_kita"],
    random_state=SEED)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
model = LogisticRegression(C=5, max_iter=2000)
# model = LogisticRegression(C=5, max_iter=2000, class_weight="balanced")   # kelas timpang
pipe = make_pipeline(vectorizer, model)

pipe.fit(X_train, y_train)
val_predictions = pipe.predict(X_val)
print("F1 macro (terhadap label kita):", round(f1_score(y_val, val_predictions,
                                                         average="macro"), 4))
pipe.fit(labeled[TEXT_COL], labeled["label_kita"])

# %% [markdown]
# ## Teks paling ragu → label ulang
# Ragu = selisih peluang dua kelas teratas paling kecil. Masukkan daftar ini ke sel tombol
# (`queue = unsure`), klik, lalu jalankan ulang sel Gabung dan Model.

# %%
unlabeled = texts[texts["label_kita"].isna()].copy()
proba = pipe.predict_proba(unlabeled[TEXT_COL])
proba.sort(axis=1)
unlabeled["selisih"] = proba[:, -1] - proba[:, -2]
unsure = unlabeled.sort_values("selisih").head(N_UNSURE)
for text, gap in zip(unsure[TEXT_COL], unsure["selisih"]):
    print(round(gap, 3), "|", text[:90])

# %% [markdown]
# ## Simpan hasil label
# Semua teks mendapat label: label kita bila ada, selain itu tebakan model.

# %%
texts["label_akhir"] = texts["label_kita"]
not_labeled = texts["label_akhir"].isna()
texts.loc[not_labeled, "label_akhir"] = pipe.predict(texts.loc[not_labeled, TEXT_COL])

result = pd.DataFrame()
result["id"] = texts[ID_COL]
result["label"] = texts["label_akhir"]
result.to_csv("hasil_label.csv", index=False)
result.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: kolom `label` asli hanya ada di data latihan.

# %%
labeled = texts[texts["label_kita"].notna()]
print("akurasi label kita vs asli   :", round(accuracy_score(labeled["label"],
                                                             labeled["label_kita"]), 4))
print("akurasi model di teks sisanya:", round(accuracy_score(
    texts.loc[not_labeled, "label"], texts.loc[not_labeled, "label_akhir"]), 4))
print("akurasi model di teks ragu   :", round(accuracy_score(
    unsure["label"], pipe.predict(unsure[TEXT_COL])), 4))
# simulasi manusia menjawab benar untuk daftar ragu, lalu jalankan ulang sel Gabung dan Model:
# pd.DataFrame({"id": unsure[ID_COL], "label": unsure["label"]}).to_csv(MANUAL_CSV, index=False)
