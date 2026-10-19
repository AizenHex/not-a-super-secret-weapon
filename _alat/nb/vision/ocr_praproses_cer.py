# %% [markdown]
# # Vision — OCR: Praproses Gambar + Evaluasi CER/WER (EasyOCR lokal)
# **Kapan dipakai:** gambar teks miring, buram, berbintik, atau kontras rendah, dan hasil OCR
# mentah banyak salah.
#
# **Langkah:** perbesar → haluskan bintik → hitam-putih (threshold) → EasyOCR → bandingkan
# CER/WER gambar mentah vs hasil praproses.
# CER = huruf salah / jumlah huruf, WER = kata salah / jumlah kata (makin kecil makin baik).
#
# **Data demo:** 240 baris teks SINTETIS (font acak). Kolom `gangguan` 0 / 0.4 / 0.8 = bersih /
# sedang / berat. Kunci jawaban = kolom `teks`.

# %%
import cv2
import easyocr
import jiwer
import matplotlib.pyplot as plt
import pandas as pd
import torch

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/ocr"   # GANTI saat lomba
LINES_CSV = f"{DATA_FOLDER}/baris.csv"
EASYOCR_FOLDER = "../../pretrained/easyocr"

LANGUAGES = ["en", "id"]
PATH_COL = "path"         # path gambar, relatif terhadap DATA_FOLDER
TEXT_COL = "teks"         # teks asli, hanya ada di data latihan
NOISE_COL = "gangguan"    # hanya ada di data demo

# %% [markdown]
# ## Baca data

# %%
lines = pd.read_csv(LINES_CSV)
print(lines.shape)
print(lines[NOISE_COL].value_counts())
lines.head()

# %% [markdown]
# ## Contoh tiap tingkat gangguan

# %%
noise_levels = sorted(lines[NOISE_COL].unique())
fig, axes = plt.subplots(len(noise_levels), 1, figsize=(10, 4))
for ax, level in zip(axes, noise_levels):
    example = lines[lines[NOISE_COL] == level].iloc[0]
    ax.imshow(cv2.imread(f"{DATA_FOLDER}/{example[PATH_COL]}"), cmap="gray")
    ax.set_title(f"gangguan {level}: {example[TEXT_COL]}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Praproses
# Perbesar 2x → haluskan bintik → ubah jadi hitam-putih. Pilih SATU baris `result`: hapus `#`
# di baris yang ingin dipakai.

# %%
def preprocess(image):
    bigger = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    smooth = cv2.medianBlur(bigger, 3)
    threshold, result = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # result = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 31, 15)
    # result = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(smooth)
    # result = smooth
    return result


example_image = cv2.imread(f"{DATA_FOLDER}/{lines.loc[2, PATH_COL]}", cv2.IMREAD_GRAYSCALE)
fig, axes = plt.subplots(2, 1, figsize=(10, 3))
axes[0].imshow(example_image, cmap="gray")
axes[1].imshow(preprocess(example_image), cmap="gray")
plt.show()

# %% [markdown]
# ## OCR: gambar mentah vs gambar hasil praproses
# Pilih SATU cara baca: hapus `#` di baris yang ingin dipakai.

# %%
reader = easyocr.Reader(LANGUAGES, gpu=torch.cuda.is_available(),
                        model_storage_directory=EASYOCR_FOLDER,
                        user_network_directory=EASYOCR_FOLDER,
                        download_enabled=False, verbose=False)


def read_text(image):
    pieces = reader.readtext(image, detail=0)
    # pieces = reader.readtext(image, detail=0, decoder="beamsearch")
    # pieces = reader.readtext(image, detail=0, contrast_ths=0.3, adjust_contrast=0.7)
    return " ".join(pieces)


raw_texts = []
clean_texts = []
for path in lines[PATH_COL]:
    image = cv2.imread(f"{DATA_FOLDER}/{path}", cv2.IMREAD_GRAYSCALE)
    raw_texts.append(read_text(image))
    clean_texts.append(read_text(preprocess(image)))
lines["ocr_mentah"] = raw_texts
lines["ocr_praproses"] = clean_texts

# %% [markdown]
# ## Evaluasi CER/WER (memakai kunci, hanya untuk latihan)
# Teks disamakan dulu: huruf kecil dan spasi tunggal.

# %%
def tidy(text):
    return " ".join(str(text).lower().split())


truth = list(lines[TEXT_COL].apply(tidy))
for column in ["ocr_mentah", "ocr_praproses"]:
    guess = list(lines[column].apply(tidy))
    print(column, "CER", round(jiwer.cer(truth, guess), 4),
          "| WER", round(jiwer.wer(truth, guess), 4))

for level in noise_levels:
    part = lines[lines[NOISE_COL] == level]
    part_truth = list(part[TEXT_COL].apply(tidy))
    raw_cer = jiwer.cer(part_truth, list(part["ocr_mentah"].apply(tidy)))
    clean_cer = jiwer.cer(part_truth, list(part["ocr_praproses"].apply(tidy)))
    print("gangguan", level, "CER mentah", round(raw_cer, 4), "| praproses", round(clean_cer, 4))

# %% [markdown]
# ## Kesalahan terbesar (bahan analisis presentasi)

# %%
line_cer = []
for true_text, guess_text in zip(truth, lines["ocr_praproses"]):
    line_cer.append(jiwer.cer(true_text, tidy(guess_text)))
lines["cer"] = line_cer
worst = lines.sort_values("cer", ascending=False)
worst[[TEXT_COL, "ocr_praproses", NOISE_COL, "cer"]].head(10)

# %% [markdown]
# ## Hasil
# Saat lomba tidak ada kolom `teks` asli. Simpan hasil OCR sesuai format soal.

# %%
submission = pd.DataFrame()
submission["path"] = lines[PATH_COL]
submission["teks"] = lines["ocr_praproses"]
submission.to_csv("submission.csv", index=False)
submission.head()
