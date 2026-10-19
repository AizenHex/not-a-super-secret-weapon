# %% [markdown]
# # Vision — Image Captioning (BLIP): gambar → kalimat → fitur teks
# **Kapan dipakai:**
# - memahami isi ribuan gambar dengan cepat (EDA);
# - caption sebagai **fitur teks** untuk klasifikasi / multimodal;
# - bahan laporan/presentasi.
#
# **Langkah:** BLIP membuat kalimat deskripsi (bahasa Inggris) → klasifikasi dari caption:
# kata kunci (tanpa latih) dan TF-IDF + LogisticRegression (dilatih pada caption).
#
# **Data demo:** CIFAR-10 kecil (32×32, sengaja sulit). Bobot: `pretrained/blip_caption`
# (BLIP base, COCO).

# %%
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
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
BLIP_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/blip_caption"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
PER_CLASS = 100       # caption ±0.05 detik per gambar di GPU, train cukup sebagian
BATCH_SIZE = 8         # kecil supaya GPU tidak terlalu panas
SEED = 42
# kata di caption → kelas. Tambah sinonim bila perlu.
KEYWORDS = {"airplane": ["plane", "airplane", "jet", "aircraft"], "automobile": ["car"],
            "bird": ["bird", "parrot", "owl", "duck"], "cat": ["cat", "kitten"],
            "deer": ["deer", "antelope", "elk"], "dog": ["dog", "puppy"],
            "frog": ["frog", "toad", "lizard"], "horse": ["horse", "pony"],
            "ship": ["ship", "boat"], "truck": ["truck", "bus", "van"]}
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
train = train.groupby(TARGET).sample(PER_CLASS, random_state=SEED)   # HAPUS bila waktu cukup
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## Model BLIP + buat caption
# Pilih SATU awalan: hapus `#` di baris yang ingin dipakai.

# %%
processor = BlipProcessor.from_pretrained(BLIP_FOLDER)
blip = BlipForConditionalGeneration.from_pretrained(BLIP_FOLDER)
blip = blip.to(DEVICE)
blip.eval()
PREFIX = "a photo of"     # caption dipaksa mulai dengan awalan ini
# PREFIX = None           # caption bebas


def make_captions(paths):
    captions = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = []
        for path in paths[start:start + BATCH_SIZE]:
            images.append(Image.open(f"{DATA_FOLDER}/{path}").convert("RGB"))
        prefixes = None
        if PREFIX is not None:
            prefixes = [PREFIX] * len(images)
        encoded = processor(images=images, text=prefixes, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = blip.generate(**encoded, max_new_tokens=20)   # num_beams=3: lebih bagus, berat
        captions.extend(processor.batch_decode(output, skip_special_tokens=True))
    return captions


train["caption"] = make_captions(list(train[PATH_COL]))
test["caption"] = make_captions(list(test[PATH_COL]))
pd.set_option("display.max_colwidth", 80)
train[[TARGET, "caption"]].head(10)

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


print(keyword_class("a photo of two dogs on the grass"))
test["kelas_kata_kunci"] = test["caption"].apply(keyword_class)
print("tidak tahu:", round((test["kelas_kata_kunci"] == "tidak_tahu").mean(), 3))

# %% [markdown]
# ## Klasifikasi cara 2: TF-IDF + LogisticRegression pada caption

# %%
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000))
pipe.fit(train["caption"], train[TARGET])
predictions = pipe.predict(test["caption"])

# %% [markdown]
# ## Hasil: caption + prediksi kelas per gambar
# Untuk gabung dengan fitur gambar lain, lihat `multimodal/gambar_tabular`.

# %%
submission = pd.DataFrame()
submission["id"] = test["id"]
submission["caption"] = test["caption"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)[TARGET]
keyword_f1 = f1_score(test_key, test["kelas_kata_kunci"], average="macro")
print("kata kunci  F1 macro:", round(keyword_f1, 4))
print("TF-IDF + LR F1 macro:", round(f1_score(test_key, predictions, average="macro"), 4))
print(classification_report(test_key, predictions))
