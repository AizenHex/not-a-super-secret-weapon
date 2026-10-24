# %% [markdown]
# # Dokumen — Ekstraksi PDF (teks, isian key-value, tabel, QR; PDF digital dan hasil scan)
# **Kapan dipakai:** data soal berupa file PDF (laporan, faktur, formulir) dan isinya harus jadi
# tabel.
#
# **Langkah:**
# 1. Cek PDF **digital** (teks bisa disalin) atau **scan** (hanya gambar).
# 2. Digital → teks langsung (`pymupdf`), tabel bergaris (`pdfplumber`).
# 3. Scan → render jadi gambar → luruskan (deskew) → OCR (EasyOCR).
# 4. Isian "Kunci: nilai" dan pola (NIK, telepon, tanggal, rupiah) diambil dengan regex.
# 5. QR code dibaca dengan OpenCV.
#
# **Data demo:** PDF laporan SINTETIS (`data_latihan/dokumen/buat_pdf_demo.py`).

# %%
import os
import re

import cv2
import easyocr
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import torch

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/dokumen"   # GANTI saat lomba
DIGITAL_PDF = f"{DATA_FOLDER}/laporan_digital.pdf"
SCAN_PDF = f"{DATA_FOLDER}/laporan_scan.pdf"
EASYOCR_FOLDER = "../../pretrained/easyocr"

DPI = 200             # resolusi render PDF → gambar (lebih tinggi = OCR lebih teliti, lebih lambat)
MIN_CHARACTERS = 20   # halaman dengan teks lebih sedikit dari ini dianggap hasil scan
NUMBER_COLUMNS = ["Jumlah", "Harga Satuan", "Subtotal"]   # kolom tabel berisi angka

# pola yang dicari di teks (tambah/ubah sesuai soal)
PATTERNS = {
    "nik": r"\b\d{16}\b",
    "telepon": r"\b08\d{1,2}[- ]?\d{3,4}[- ]?\d{3,4}\b",
    "tanggal": r"\b\d{2}/\d{2}/\d{4}\b",
    "rupiah": r"Rp\s?[\d.]+",
}

# %% [markdown]
# ## 1. PDF digital atau scan?

# %%
for path in [DIGITAL_PDF, SCAN_PDF]:
    document = fitz.open(path)
    n_characters = len(document[0].get_text().strip())
    kind = "scan"
    if n_characters >= MIN_CHARACTERS:
        kind = "digital"
    print(os.path.basename(path), "| halaman:", len(document),
          "| karakter teks:", n_characters, "->", kind)

# %% [markdown]
# ## 2. Teks PDF digital (pymupdf)

# %%
document = fitz.open(DIGITAL_PDF)
digital_text = ""
for page in document:
    digital_text = digital_text + page.get_text() + "\n"
print(digital_text)

# %% [markdown]
# ## 3. Isian "Kunci: nilai" (key-value / formulir) dan pola regex

# %%
fields = {}
for line in digital_text.splitlines():
    match = re.match(r"^\s*([A-Za-z ]+?)\s*:\s*(.+)$", line)
    if match:
        fields[match.group(1)] = match.group(2).strip()
print(pd.Series(fields))

found = {}
for name in PATTERNS:
    found[name] = re.findall(PATTERNS[name], digital_text)
print(found)

# %% [markdown]
# ## 4. Tabel PDF digital (pdfplumber, tabel bergaris)
# Tabel tanpa garis: coba `page.extract_tables({"vertical_strategy": "text",
# "horizontal_strategy": "text"})`.

# %%
with pdfplumber.open(DIGITAL_PDF) as pdf:
    tables = pdf.pages[0].extract_tables()
print("jumlah tabel:", len(tables))
table = pd.DataFrame(tables[0][1:], columns=tables[0][0])   # baris pertama = nama kolom

# angka Indonesia "1.280.000" → 1280000
for column in NUMBER_COLUMNS:
    table[column] = table[column].str.replace(".", "", regex=False)
    table[column] = table[column].astype(int)
check = table["Jumlah"] * table["Harga Satuan"] == table["Subtotal"]
print("cek: jumlah x harga = subtotal?", check.all())
table

# %% [markdown]
# ## 5. Render halaman jadi gambar + baca QR code

# %%
def render_page(path, page_number):
    # PDF → gambar abu-abu numpy (dipakai untuk QR dan OCR)
    page = fitz.open(path)[page_number]
    pixmap = page.get_pixmap(dpi=DPI)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8)
    array = array.reshape(pixmap.height, pixmap.width, pixmap.n)
    return cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)


digital_image = render_page(DIGITAL_PDF, 0)
qr_detector = cv2.QRCodeDetector()
qr_text, _, _ = qr_detector.detectAndDecode(digital_image)
print("isi QR:", qr_text)

# %% [markdown]
# ## 6. PDF scan: luruskan (deskew)
# Sudut miring dicari dari kotak terkecil yang membungkus semua piksel tulisan (`minAreaRect`).

# %%
scan_image = render_page(SCAN_PDF, 0)
binary = cv2.threshold(scan_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# buang bintik noise supaya sudut dihitung dari tulisan saja
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
points = np.column_stack(np.where(binary > 0)).astype(np.float32)   # (baris, kolom)
angle = cv2.minAreaRect(points[:, ::-1])[-1]                       # butuh (kolom, baris)
if angle > 45:
    angle = angle - 90
print("sudut miring terdeteksi:", round(angle, 2), "derajat")

height, width = scan_image.shape
rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
straight = cv2.warpAffine(scan_image, rotation, (width, height), borderValue=255)

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].imshow(scan_image, cmap="gray")
axes[0].set_title("scan asli")
axes[1].imshow(straight, cmap="gray")
axes[1].set_title("setelah deskew")
plt.savefig("deskew.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. OCR halaman scan + regex yang sama

# %%
reader = easyocr.Reader(["en", "id"], gpu=torch.cuda.is_available(),
                        model_storage_directory=EASYOCR_FOLDER,
                        user_network_directory=EASYOCR_FOLDER,
                        download_enabled=False, verbose=False)
ocr_lines = reader.readtext(straight, detail=0, paragraph=False)
scan_text = "\n".join(ocr_lines)
print(scan_text[:600])

found_scan = {}
for name in PATTERNS:
    found_scan[name] = re.findall(PATTERNS[name], scan_text)
print(found_scan)

# %% [markdown]
# ## Simpan hasil ekstraksi

# %%
result = pd.DataFrame()
result["kunci"] = list(fields.keys())
result["nilai"] = list(fields.values())
result.to_csv("isian.csv", index=False)
table.to_csv("tabel.csv", index=False)
result
