# %% [markdown]
# # Vision — OCR Transformer (TrOCR) untuk Tulisan Tangan dan Cetak, dibanding EasyOCR
# **Kapan dipakai:** gambar berisi **satu baris** teks (potongan formulir, tulisan tangan) dan
# EasyOCR banyak salah baca.
#
# **Langkah:** TrOCR (encoder gambar + decoder teks) membaca tiap gambar → bandingkan dengan
# EasyOCR memakai CER dan WER (makin kecil makin baik).
#
# **Data demo SINTETIS:** `ocr/tangan` (60 baris gaya tulisan tangan) dan `ocr/baris` (240 baris
# cetak dengan gangguan). Bobot: `trocr_tangan` (tulisan tangan) atau `trocr_cetak` (cetak).

# %%
import cv2
import easyocr
import jiwer
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/ocr"   # GANTI saat lomba
HANDWRITTEN_CSV = f"{DATA_FOLDER}/tangan.csv"
PRINTED_CSV = f"{DATA_FOLDER}/baris.csv"
EASYOCR_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/easyocr"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TEXT_COL = "teks"     # teks asli, hanya ada di data latihan
BATCH_SIZE = 16
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
handwritten = pd.read_csv(HANDWRITTEN_CSV)
printed = pd.read_csv(PRINTED_CSV)
print("tangan:", handwritten.shape, "| cetak:", printed.shape)
handwritten.head()

# %% [markdown]
# ## Lihat contoh gambar

# %%
example_paths = list(handwritten[PATH_COL][:2]) + list(printed[PATH_COL][:2])
fig, axes = plt.subplots(4, 1, figsize=(10, 5))
for ax, path in zip(axes, example_paths):
    ax.imshow(Image.open(f"{DATA_FOLDER}/{path}"), cmap="gray")
    ax.set_title(path)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Model TrOCR
# Pilih SATU bobot: hapus `#` di baris yang ingin dipakai.

# %%
TROCR_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/trocr_tangan"
# TROCR_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/trocr_cetak"

processor = TrOCRProcessor.from_pretrained(TROCR_FOLDER)
trocr = VisionEncoderDecoderModel.from_pretrained(TROCR_FOLDER)
trocr = trocr.to(DEVICE)
trocr.eval()


def read_trocr(paths):
    texts = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = []
        for path in paths[start:start + BATCH_SIZE]:
            images.append(Image.open(f"{DATA_FOLDER}/{path}").convert("RGB"))
        pixels = processor(images=images, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            output = trocr.generate(pixels, max_new_tokens=48, num_beams=3)
        for text in processor.batch_decode(output, skip_special_tokens=True):
            # TrOCR dilatih pada kalimat (IAM) → sering menambah " ." di akhir, dibuang di sini
            text = text.removesuffix(" .")
            texts.append(text.strip())
    return texts


print(read_trocr(list(handwritten[PATH_COL][:2])))

# %% [markdown]
# ## Pembanding: EasyOCR

# %%
reader = easyocr.Reader(["en", "id"], gpu=torch.cuda.is_available(),
                        model_storage_directory=EASYOCR_FOLDER,
                        user_network_directory=EASYOCR_FOLDER,
                        download_enabled=False, verbose=False)


def read_easyocr(paths):
    texts = []
    for path in paths:
        pieces = reader.readtext(cv2.imread(f"{DATA_FOLDER}/{path}"), detail=0)
        texts.append(" ".join(pieces))
    return texts


print(read_easyocr(list(handwritten[PATH_COL][:2])))

# %% [markdown]
# ## Evaluasi CER/WER per jenis tulisan (memakai kunci, hanya untuk latihan)
# Di data demo: TrOCR menang CER pada teks cetak bergangguan, EasyOCR menang pada "tulisan
# tangan" sintetis (masih berupa font). Pada tulisan tangan asli TrOCR biasanya unggul.

# %%
def tidy(text):
    return " ".join(str(text).lower().split())


for name, table in [("tangan", handwritten), ("cetak", printed)]:
    table["trocr"] = read_trocr(list(table[PATH_COL]))
    table["easyocr"] = read_easyocr(list(table[PATH_COL]))
    truth = list(table[TEXT_COL].apply(tidy))
    for column in ["trocr", "easyocr"]:
        guess = list(table[column].apply(tidy))
        print(name, column, "CER", round(jiwer.cer(truth, guess), 4),
              "| WER", round(jiwer.wer(truth, guess), 4))

# %% [markdown]
# ## Contoh berdampingan

# %%
pd.set_option("display.max_colwidth", 60)
handwritten[[TEXT_COL, "trocr", "easyocr"]].head(10)

# %% [markdown]
# ## Hasil
# Saat lomba tidak ada kolom `teks` asli. Simpan hasil OCR sesuai format soal.

# %%
submission = pd.DataFrame()
submission["path"] = handwritten[PATH_COL]
submission["teks"] = handwritten["trocr"]
submission.to_csv("submission.csv", index=False)
submission.head()
