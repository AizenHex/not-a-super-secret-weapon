# %% [markdown]
# # Vision — Image Captioning (BLIP): gambar → kalimat → fitur teks
# **Kapan dipakai:**
# - memahami isi ribuan gambar dengan cepat (EDA);
# - caption sebagai **fitur teks** untuk klasifikasi / multimodal;
# - bahan laporan/presentasi.
#
# **Bentuk data yang didukung** (pilih satu sel di bagian Baca data):
# 1. CSV `id,label` + gambar dalam satu folder (aktif)
# 2. folder per kelas: `train/<kelas>/*.jpg`
# 3. label tertulis di nama file: `Abyssinian_12.jpg` → `Abyssinian`
#
# **Langkah:** BLIP membuat kalimat deskripsi (bahasa Inggris) → klasifikasi dari caption:
# kata kunci (tanpa latih) dan TF-IDF + LogisticRegression (dilatih pada caption).
#
# **Data demo:** `pets` (10 ras kucing/anjing). Caption umum sering hanya menyebut "a cat" /
# "a dog"; dengan awalan "this pet is a" BLIP lebih sering menyebut ras (F1 TF-IDF 0.31 → 0.65).
# Tetap jauh di bawah `klasifikasi_embedding` (0.98), jadi caption = fitur tambahan / EDA.
# Bobot: `pretrained/blip_caption` (BLIP base, COCO).

# %%
import glob
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from transformers import BlipForConditionalGeneration, BlipProcessor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/pets"    # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"              # tidak ada? pakai CSV contoh jawaban
TRAIN_IMAGE_FOLDER = f"{DATA_FOLDER}/train"
TEST_IMAGE_FOLDER = f"{DATA_FOLDER}/test"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
BLIP_FOLDER = "../../pretrained/blip_caption"
OUTPUT_FOLDER = "../../outputs/vision/captioning"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"           # kolom nama file gambar (juga kolom id di submission)
TARGET = "label"        # kolom kelas
IMAGE_EXTENSION = ""    # isi ".jpg" bila kolom id TIDAK memuat ekstensi (misalnya "img_01")
BATCH_SIZE = 8          # kecil supaya GPU laptop tidak terlalu panas (num_beams=3 = 3× beban)
# kata di caption → kelas. Tambah sinonim setelah melihat contoh caption.
KEYWORDS = {"Abyssinian": ["abyssinian", "abyss"], "Bengal": ["bengal", "leopard"],
            "Persian": ["persian", "fluffy"], "Siamese": ["siamese", "siam", "himalayan"],
            "Egyptian_Mau": ["mau", "egyptian", "tabby"], "beagle": ["beagle"], "pug": ["pug"],
            "shiba_inu": ["shiba", "shi", "fox", "akita"],
            "yorkshire_terrier": ["yorkshire", "york", "yorkie"],
            "staffordshire_bull_terrier": ["pit", "bull", "staffordshire", "pitbull"]}
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data — bentuk 1: CSV `id,label` + folder gambar (aktif)

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train["path"] = TRAIN_IMAGE_FOLDER + "/" + train[ID_COL].astype(str) + IMAGE_EXTENSION
test["path"] = TEST_IMAGE_FOLDER + "/" + test[ID_COL].astype(str) + IMAGE_EXTENSION
print("train:", train.shape, "| test:", test.shape)
print("contoh path:", train["path"].iloc[0])

# %% [markdown]
# ## (Alternatif) bentuk 2: folder per kelas `train/<kelas>/*.jpg`

# %%
# rows = []
# for class_name in sorted(os.listdir(TRAIN_IMAGE_FOLDER)):
#     for path in sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/{class_name}/*")):
#         rows.append([os.path.basename(path), class_name, path])
# train = pd.DataFrame(rows, columns=[ID_COL, TARGET, "path"])

# %% [markdown]
# ## (Alternatif) bentuk 3: label di nama file `Abyssinian_12.jpg`

# %%
# train = pd.DataFrame()
# train["path"] = sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/*"))
# train[ID_COL] = train["path"].apply(os.path.basename)
# train[TARGET] = train[ID_COL].str.rsplit("_", n=1).str[0]

# %% [markdown]
# ## Model BLIP + buat caption
# Awalan memancing isi caption. Awalan umum ("a photo of") hampir selalu menghasilkan "a cat" /
# "a dog"; awalan "this pet is a" + `num_beams=3` lebih sering menyebut nama ras. GANTI sesuai
# domain soal (mis. "this food is a"). `no_repeat_ngram_size=2` mencegah "shi shi shi ...".
# Pilih SATU awalan: hapus `#` di baris yang ingin dipakai.

# %%
processor = BlipProcessor.from_pretrained(BLIP_FOLDER)
blip = BlipForConditionalGeneration.from_pretrained(BLIP_FOLDER)
blip = blip.to(DEVICE)
blip.eval()
PREFIX = "this pet is a"   # caption dipaksa mulai dengan awalan ini
# PREFIX = "a photo of"    # awalan umum
# PREFIX = None            # caption bebas


def make_captions(paths):
    captions = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = []
        for path in paths[start:start + BATCH_SIZE]:
            images.append(Image.open(path).convert("RGB"))
        prefixes = None
        if PREFIX is not None:
            prefixes = [PREFIX] * len(images)
        encoded = processor(images=images, text=prefixes, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = blip.generate(**encoded, max_new_tokens=20, num_beams=3,
                                   no_repeat_ngram_size=2)
        captions.extend(processor.batch_decode(output, skip_special_tokens=True))
    return captions


train["caption"] = make_captions(list(train["path"]))
test["caption"] = make_captions(list(test["path"]))
pd.set_option("display.max_colwidth", 80)
train[[TARGET, "caption"]].head(10)

# %% [markdown]
# ## Kata yang paling sering muncul di caption tiap kelas
# Bahan untuk mengisi `KEYWORDS`: kata khas suatu kelas yang jarang muncul di kelas lain.

# %%
for class_name in sorted(train[TARGET].unique()):
    words = train.loc[train[TARGET] == class_name, "caption"].str.split().explode()
    common = words[~words.isin(["a", "of", "photo", "the", "with", "on", "in", "and", "this",
                                "pet", "is"])]
    print(f"{class_name:28s}", list(common.value_counts().head(6).index))

# %% [markdown]
# ## Klasifikasi cara 1: kata kunci (tanpa latih)
# Kelas pertama yang kata kuncinya (atau bentuk jamak +s) muncul di caption.

# %%
def keyword_class(caption):
    words = caption.lower().split()
    for name in KEYWORDS:
        for keyword in KEYWORDS[name]:
            if keyword in words or keyword + "s" in words:
                return name
    return "tidak_tahu"


print(keyword_class("a photo of two pugs on the grass"))
test["kelas_kata_kunci"] = test["caption"].apply(keyword_class)
print("tidak tahu:", round((test["kelas_kata_kunci"] == "tidak_tahu").mean(), 3))

# %% [markdown]
# ## Klasifikasi cara 2: TF-IDF + LogisticRegression pada caption

# %%
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                     LogisticRegression(max_iter=2000, class_weight="balanced"))
pipe.fit(train["caption"], train[TARGET])
predictions = pipe.predict(test["caption"])

# %% [markdown]
# ## Hasil: caption + prediksi kelas per gambar
# Untuk gabung dengan fitur gambar lain, lihat `multimodal/gambar_tabular`.

# %%
submission = pd.DataFrame()
submission[ID_COL] = test[ID_COL]
submission["caption"] = test["caption"]
submission[TARGET] = predictions
submission.to_csv(f"{OUTPUT_FOLDER}/submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
print("urutan id sama:", (test_key[ID_COL].to_numpy() == submission[ID_COL].to_numpy()).all())
keyword_f1 = f1_score(test_key[TARGET], test["kelas_kata_kunci"], average="macro")
print("kata kunci  F1 macro:", round(keyword_f1, 4))
print("TF-IDF + LR F1 macro:", round(f1_score(test_key[TARGET], predictions, average="macro"), 4))
print(classification_report(test_key[TARGET], predictions, zero_division=0))

# %% [markdown]
# ## Masalah umum
# - **Caption terlalu umum ("a cat"):** coba awalan lain, atau tanyakan langsung dengan VQA
#   (`vqa`): "what breed is this?". Awalan "a photo of a breed of" justru memicu pengulangan.
# - **Caption lambat:** turunkan `BATCH_SIZE` bila GPU penuh; di CPU caption ribuan gambar bisa
#   memakan puluhan menit, jadi simpan hasil caption ke CSV dan jangan diulang.
# - **Kata kunci salah tangkap** ("bull" di "bulldog"): kata dicocokkan per kata utuh, tetapi
#   tetap cek contoh caption kelas yang skornya rendah.
