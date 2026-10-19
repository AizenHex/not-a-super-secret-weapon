# %% [markdown]
# # Vision — OCR: Ambil Field dari Foto Struk (toko, tanggal, total)
# **Kapan dipakai:** ada foto dokumen (struk, kwitansi, faktur, KTP) dan diminta mengambil
# isian tertentu, misalnya nama toko, tanggal, dan total belanja.
#
# **Langkah:**
# 1. OCR (EasyOCR) membaca teks beserta kotak posisinya.
# 2. Kotak-kotak disusun menjadi baris teks (atas ke bawah, kiri ke kanan).
# 3. Regex mengambil field yang dicari dari baris-baris itu.
# 4. Gambar bisa disiapkan dulu (**preprocessing**: abu-abu, diperbesar, hitam-putih) sebelum OCR.
#
# **Data demo:** 80 foto struk SINTETIS, kunci jawaban di kolom `toko`, `tanggal`, `total`.

# %%
import re

import cv2
import easyocr
import matplotlib.pyplot as plt
import pandas as pd
import torch

# %% [markdown]
# ## Konfigurasi
# Saat lomba: ganti `DATA_FOLDER`, nama file CSV, dan pola regex sesuai dokumen soal.

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/ocr"            # GANTI saat lomba
LABEL_CSV = f"{DATA_FOLDER}/struk.csv"       # kolom: path gambar + kunci jawaban tiap field
OCR_MODEL_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/easyocr"            # bobot EasyOCR offline

LANGUAGES = ["en", "id"]     # bahasa teks di dokumen

# Pola regex. Coba dulu di https://regex101.com kalau ingin mengubah.
DATE_PATTERN = r"\d{2}[-/.]\d{2}[-/.]\d{4}"          # contoh cocok: 24-03-2025, 24/03/2025
# DATE_PATTERN = r"\d{1,2} (Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|" \
#                r"Oktober|November|Desember) \d{4}"  # contoh cocok: 24 Maret 2025
# OCR sering salah baca: T jadi I ("IOIAL"), O jadi 0, dan ada spasi di angka ("84 . 000").
# [TI] = huruf T atau I. [O0] = huruf O atau angka 0. Angka boleh berisi titik, koma, spasi.
TOTAL_PATTERN = r"[TI][O0][TI]AL\D*(\d[\d., ]*)"

# %% [markdown]
# ## Baca data
# Setiap baris = satu foto + jawaban benar.

# %%
data = pd.read_csv(LABEL_CSV)
print("jumlah foto:", len(data))
data.head()

# %% [markdown]
# ## Langkah 1 — OCR satu foto: teks + kotak posisi
# `reader.readtext(gambar)` menghasilkan daftar `(kotak, teks, keyakinan)`.
# `kotak` = 4 titik sudut: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah.

# %%
reader = easyocr.Reader(LANGUAGES, gpu=torch.cuda.is_available(),
                        model_storage_directory=OCR_MODEL_FOLDER,
                        user_network_directory=OCR_MODEL_FOLDER,
                        download_enabled=False, verbose=False)

example_image = cv2.imread(f"{DATA_FOLDER}/{data['path'].iloc[0]}")
example_result = reader.readtext(example_image)

for box, text, confidence in example_result[:5]:
    print("kiri-atas:", box[0], "| teks:", text, "| yakin:", round(confidence, 2))

for box, text, confidence in example_result:
    top_left = (int(box[0][0]), int(box[0][1]))
    bottom_right = (int(box[2][0]), int(box[2][1]))
    cv2.rectangle(example_image, top_left, bottom_right, (0, 0, 255), 2)
plt.figure(figsize=(5, 6))
plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))   # OpenCV memakai urutan warna BGR
plt.axis("off")
plt.show()

# %% [markdown]
# ## Langkah 2a — kelompokkan kotak per baris
# Dua kotak dianggap **satu baris** kalau posisi tengah-y keduanya berselisih kurang dari
# setengah tinggi kotak. Aturan ini tetap benar walaupun gambar diperbesar.
# `boxes.sort()` mengurutkan dari elemen pertama (tengah-y), jadi hasilnya urut atas → bawah.

# %%
def group_lines(ocr_result):
    boxes = []
    for box, text, confidence in ocr_result:
        top = box[0][1]
        bottom = box[2][1]
        boxes.append([(top + bottom) / 2, box[0][0], bottom - top, text])
    boxes.sort()

    lines = []
    previous_y = -1000
    for center_y, left_x, height, text in boxes:
        if center_y - previous_y > height / 2:
            lines.append([])
        lines[-1].append([left_x, text])
        previous_y = center_y
    return lines


# coba pada foto contoh
example_groups = group_lines(example_result)
print("jumlah baris:", len(example_groups))
print("baris pertama:", example_groups[0])

# %% [markdown]
# ## Langkah 2b — gabungkan isi tiap baris menjadi satu kalimat
# Kotak dalam satu baris diurutkan dari kiri ke kanan, lalu teksnya digabung dengan spasi.

# %%
def lines_to_text(lines):
    line_texts = []
    for line in lines:
        line.sort()                      # urut kiri → kanan (dari left_x)
        words = []
        for left_x, text in line:
            words.append(text)
        line_texts.append(" ".join(words))
    return line_texts


# coba pada foto contoh
example_lines = lines_to_text(example_groups)
for line in example_lines:
    print(line)

# %% [markdown]
# ## Langkah 3 — ambil field dengan regex
# - **toko** = baris pertama struk.
# - **tanggal** = teks pertama yang cocok dengan `DATE_PATTERN`, pemisah diseragamkan jadi `-`.
# - **total** = angka setelah kata TOTAL, titik/koma/spasi dibuang (`84 . 000` → `84000`).
# Kalau tidak ketemu: tanggal = "" dan total = -1.

# %%
def extract_fields(line_texts):
    full_text = "\n".join(line_texts)
    fields = {"toko": "", "tanggal": "", "total": -1}
    if len(line_texts) > 0:
        fields["toko"] = line_texts[0]

    date_match = re.search(DATE_PATTERN, full_text)
    if date_match is not None:
        date = date_match.group(0)
        date = date.replace("/", "-")
        date = date.replace(".", "-")
        fields["tanggal"] = date

    total_match = re.search(TOTAL_PATTERN, full_text, flags=re.IGNORECASE)
    if total_match is not None:
        digits = re.sub(r"\D", "", total_match.group(1))    # \D = semua yang bukan angka
        if digits != "":
            fields["total"] = int(digits)
    return fields


# coba pada foto contoh
print(extract_fields(example_lines))

# %% [markdown]
# ## Langkah 4a — cara dasar preprocessing gambar
# - **original**: gambar apa adanya.
# - **gray**: abu-abu, warna dibuang.
# - **enlarge_2x**: diperbesar 2 kali, membantu kalau huruf kecil.
# - **black_white**: hitam-putih dengan ambang otomatis (Otsu), membantu kalau latar kotor.
# Mau menambah cara? Buat fungsi baru, lalu tambah satu baris `# prepare = ...` di bawah.

# %%
def original(image):
    return image


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enlarge_2x(image):
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def black_white(image):
    image = gray(image)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image


# coba pada foto contoh: ukuran (tinggi, lebar, warna) sebelum dan sesudah
example_image = cv2.imread(f"{DATA_FOLDER}/{data['path'].iloc[0]}")
print("original   :", original(example_image).shape)
print("gray       :", gray(example_image).shape, "(warna dibuang)")
print("enlarge_2x :", enlarge_2x(example_image).shape)
print("black_white:", black_white(example_image).shape)

# %% [markdown]
# ## Langkah 4b — cara gabungan + pilih SATU preprocessing
# Cara gabungan memakai cara dasar di atas secara berurutan.
# Pilih SATU: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
def gray_enlarge_2x(image):
    image = gray(image)
    return enlarge_2x(image)


def black_white_enlarge_2x(image):
    image = enlarge_2x(image)
    return black_white(image)


prepare = original                  # gambar apa adanya
# prepare = gray                    # abu-abu
# prepare = enlarge_2x              # diperbesar 2x, untuk huruf kecil
# prepare = gray_enlarge_2x         # abu-abu lalu diperbesar 2x
# prepare = black_white             # hitam-putih, untuk latar kotor
# prepare = black_white_enlarge_2x  # diperbesar 2x lalu hitam-putih

# %% [markdown]
# ## Lihat hasil preprocessing pada foto contoh
# Kiri: foto asli. Kanan: setelah `prepare`.

# %%
example_image = cv2.imread(f"{DATA_FOLDER}/{data['path'].iloc[0]}")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
axes[0].set_title("asli")
axes[0].axis("off")
axes[1].imshow(prepare(example_image), cmap="gray")
axes[1].set_title(prepare.__name__)          # __name__ = nama fungsi yang dipilih
axes[1].axis("off")
plt.show()

# %% [markdown]
# ## Langkah 5 — baca semua foto dan cek ketepatan
# Untuk tiap foto: baca → preprocessing → OCR → susun baris → ambil field.
# Ketepatan dihitung dengan kunci jawaban (hanya ada di data demo / data train lomba).

# %%
def read_receipts(table):
    rows = []
    for path in table["path"]:
        image = cv2.imread(f"{DATA_FOLDER}/{path}")
        image = prepare(image)
        ocr_result = reader.readtext(image)
        line_texts = lines_to_text(group_lines(ocr_result))
        rows.append(extract_fields(line_texts))
    return pd.DataFrame(rows)


guess = read_receipts(data)

# (A == B) menghasilkan True/False per foto. .mean() = persen yang True.
store_correct = (guess["toko"].str.upper() == data["toko"].str.upper()).mean()
date_correct = (guess["tanggal"] == data["tanggal"]).mean()
total_correct = (guess["total"] == data["total"]).mean()
print("toko tepat   :", round(store_correct, 3))
print("tanggal tepat:", round(date_correct, 3))
print("total tepat  :", round(total_correct, 3))

# %% [markdown]
# ## Foto yang salah
# Lihat baris yang salah untuk tahu apakah perlu mengubah regex atau preprocessing.

# %%
wrong_store = guess["toko"].str.upper() != data["toko"].str.upper()
wrong_date = guess["tanggal"] != data["tanggal"]
wrong_total = guess["total"] != data["total"]
wrong = wrong_store | wrong_date | wrong_total          # | = "atau"
print("foto yang salah minimal satu field:", wrong.sum())

wrong_table = pd.DataFrame()
wrong_table["path"] = data["path"][wrong]
wrong_table["toko_benar"] = data["toko"][wrong]
wrong_table["toko_tebakan"] = guess["toko"][wrong]
wrong_table["tanggal_benar"] = data["tanggal"][wrong]
wrong_table["tanggal_tebakan"] = guess["tanggal"][wrong]
wrong_table["total_benar"] = data["total"][wrong]
wrong_table["total_tebakan"] = guess["total"][wrong]
wrong_table.head(10)

# %% [markdown]
# ## Submission
# Nama kolom disalin dari format soal. Saat lomba, ganti `data` di bawah dengan tabel daftar
# foto test, lalu jalankan `read_receipts` untuk tabel itu.
# File `submission.csv` disimpan di folder yang sama dengan notebook ini.

# %%
predictions = guess

submission = pd.DataFrame()
submission["path"] = data["path"]
submission["toko"] = predictions["toko"]
submission["tanggal"] = predictions["tanggal"]
submission["total"] = predictions["total"]
submission.to_csv("submission.csv", index=False)
submission.head()
