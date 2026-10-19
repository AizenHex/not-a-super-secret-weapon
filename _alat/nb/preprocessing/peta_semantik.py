# %% [markdown]
# # Preprocessing — Teks bebas → kelompok makna (Laya + embedding sebagai dua juri)
# **Kapan dipakai:** isian bebas yang **beda kata tapi sama makna**, misalnya pekerjaan `ojol`,
# `ojek online`, `sopir`. Regex dan fuzzy tidak bisa menyamakan ini karena hurufnya berbeda.
#
# **Langkah:** dua juri memilih satu kelompok untuk tiap **nilai unik**:
# - **Laya** (`pretrained/laya`, multilingual): model keputusan "pilih satu opsi".
# - **MiniLM** (`pretrained/minilm`): kemiripan embedding dengan deskripsi kelompok.
#
# Kalau kedua juri **setuju**, usulan diterima. Kalau **tidak setuju**, nilai itu diperiksa manual
# dan hasilnya ditulis di `CORRECTIONS`.
#
# Hasil di data demo (36 nilai unik, 8 kelompok):
# - Laya benar 22 dari 36, MiniLM juga 22 dari 36 (61%). Tidak cukup bagus untuk dipakai sendirian.
# - Kedua juri setuju di 19 nilai, 18 di antaranya benar. Sisanya 17 nilai diperiksa manual.
# - Yang setuju tapi salah: `notaris`. Jadi daftar "setuju" tetap dibaca sekilas.
#
# Laya hanya dijalankan pada nilai unik (sekitar 0,5 detik per nilai di GPU), bukan semua baris.
#
# **Data demo:** `pekerjaan_ayah` di `ukt_SINTETIS` (sama dengan `tabular/survei_kotor`).

# %%
import laya
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular_extra/survei_kotor"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/ukt_SINTETIS_train.csv"
TEST_CSV = f"{DATA_FOLDER}/ukt_SINTETIS_test.csv"
LAYA_DIR = "Z:/GEMASTIK-DATABASE/pretrained/laya"
EMBED_DIR = "Z:/GEMASTIK-DATABASE/pretrained/minilm"
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

COLUMN = "pekerjaan_ayah"
PREFIX = "Pekerjaan ayah: "        # konteks yang ditempel di depan tiap nilai untuk Laya
QUESTION = "Pekerjaan ini termasuk kelompok mana?"
# Kelompok tujuan + deskripsi singkat. Deskripsi yang jelas = usulan yang lebih tepat.
GROUPS = {
    "tidak_bekerja": "tidak bekerja, belum bekerja, menganggur, kosong",
    "buruh": "buruh, kuli, pekerja kasar harian",
    "petani": "petani, pekebun, nelayan, peternak",
    "informal": "pedagang kecil, warung, ojek, ojol, usaha kecil",
    "karyawan": "karyawan swasta, sopir, satpam, guru honorer",
    "pns_rendah": "PNS golongan rendah (I, II), honorer pemerintah, perangkat desa",
    "pns": "PNS golongan III ke atas, TNI, Polri, guru PNS, pensiunan PNS",
    "profesional": "dokter, dosen, notaris, direktur, pengusaha besar",
}
# Diisi SETELAH membaca tabel "tidak setuju" (nilai unik -> kelompok)
CORRECTIONS = {"-": "tidak_bekerja", "tni": "pns", "polri": "pns", "pensiunan pns": "pns",
               "pns gol iii": "pns", "pns gol ii": "pns_rendah", "honorer pemda": "pns_rendah",
               "perangkat desa": "pns_rendah", "guru honorer": "karyawan", "satpam": "karyawan",
               "sopir": "karyawan", "wiraswasta warung": "informal", "ojol": "informal",
               "dokter": "profesional", "dosen": "profesional", "direktur": "profesional",
               "pengusaha": "profesional", "notaris": "profesional"}

# %% [markdown]
# ## Baca data → nilai unik (kosong jadi "-", spasi dibuang, huruf kecil)

# %%
def clean_text(values):
    values = values.fillna("-")
    values = values.str.strip()
    return values.str.lower()


train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
all_values = clean_text(pd.concat([train[COLUMN], test[COLUMN]]))
counts = all_values.value_counts()

draft = pd.DataFrame()
draft["value"] = counts.index
draft["count"] = counts.to_numpy()
print(len(draft), "nilai unik dari", len(all_values), "baris")
draft.head()

# %% [markdown]
# ## Juri 1: Laya

# %%
judge = laya.load(LAYA_DIR, subfolder="multilingual", device=DEVICE)
question = {"group": {"type": "choice", "instructions": QUESTION, "criteria": GROUPS}}
prompts = []
for value in draft["value"]:
    prompts.append(PREFIX + value)
answers = judge.predict_batch(prompts, question)

laya_choices = []
laya_confidence = []
for answer in answers:
    group_answer = answer["answers"]["group"]
    laya_choices.append(group_answer["choice"])
    laya_confidence.append(max(group_answer["probabilities"].values()))
draft["laya"] = laya_choices
draft["laya_confidence"] = laya_confidence
draft.head()

# %% [markdown]
# ## Juri 2: embedding MiniLM (kemiripan kosinus dengan deskripsi kelompok)

# %%
encoder = SentenceTransformer(EMBED_DIR, device=DEVICE)
group_names = list(GROUPS.keys())
group_texts = []
for name, description in GROUPS.items():
    group_texts.append(name + ": " + description)
value_vectors = encoder.encode(list(draft["value"]), normalize_embeddings=True)
group_vectors = encoder.encode(group_texts, normalize_embeddings=True)
similarity = value_vectors @ group_vectors.T     # baris = nilai, kolom = kelompok

embedding_choices = []
for best_index in similarity.argmax(axis=1):
    embedding_choices.append(group_names[best_index])
draft["embedding"] = embedding_choices
draft["embedding_similarity"] = similarity.max(axis=1).round(2)
draft["agree"] = draft["laya"] == draft["embedding"]
draft.to_csv(f"peta_{COLUMN}_DRAF.csv", index=False)
print("setuju:", draft["agree"].sum(), "| perlu dicek:", (~draft["agree"]).sum())

# %% [markdown]
# ## Periksa
# Tabel pertama (tidak setuju) perlu diputuskan manual: tulis hasilnya di `CORRECTIONS`.
# Tabel kedua (setuju) cukup dibaca sekilas. Yang salah juga ditulis di `CORRECTIONS`.

# %%
draft[~draft["agree"]]

# %%
draft[draft["agree"]].sort_values("laya_confidence")

# %% [markdown]
# ## Terapkan peta

# %%
mapping = {}
for value, choice, agree in zip(draft["value"], draft["laya"], draft["agree"]):
    if agree:
        mapping[value] = choice
mapping.update(CORRECTIONS)

train_groups = clean_text(train[COLUMN]).map(mapping)
test_groups = clean_text(test[COLUMN]).map(mapping)
train_missing = train_groups.isna().mean()
test_missing = test_groups.isna().mean()
print("train tanpa kelompok:", round(train_missing, 3))
print("test tanpa kelompok:", round(test_missing, 3))
train_groups.value_counts(dropna=False)

# %% [markdown]
# ## Simpan peta
# Pakai sebagai kolom kategori baru, misalnya di `tabular/survei_kotor`:
# `train["kelompok_kerja"] = train_groups`, lalu tambahkan `"kelompok_kerja"` ke `CAT_COLS`.

# %%
mapping_table = pd.DataFrame()
mapping_table["nilai"] = list(mapping.keys())
mapping_table["kelompok"] = list(mapping.values())
mapping_table.to_csv(f"peta_{COLUMN}.csv", index=False)
mapping_table.head()
