# %% [markdown]
# # OCR potongan teks (pretrained, offline) → kolom `text`, `conf`, `rotation`
#
# **Butuh:** tabel `crops` dengan kolom `fixed_path` dari `potong_perbaiki.ipynb` (sel 5
# "Terapkan ke SEMUA"). Kernel sempat mati? `crops = pd.read_csv("outputs/crops.csv")`.
#
# **Menghasilkan:** kolom `crops["text"]`, `["conf"]`, `["rotation"]`, lalu `["clean"]`, `["lot"]`,
# `["fab"]`; semuanya disimpan ke `outputs/crops.csv`.
#
# Engine: EasyOCR (pretrained, bisa banyak baris sekaligus). Soal tidak memberi label teks, jadi OCR
# tidak bisa dilatih dari soal: andalkan perbaikan gambar + pilih rotasi + rapikan teks.
# Melatih OCR sendiri dari teks sintetis: `ocr_cnn_transformer.ipynb`.
#
# ## Cara menempel ke notebook-mu
# Salin **berurutan**, di bawah sel "Terapkan ke SEMUA" dari `potong_perbaiki`:
#
# | urutan | sel | wajib? | gunanya |
# |---|---|---|---|
# | 1 | Muat EasyOCR | wajib | membuat `reader` |
# | 2 | Fungsi baca | wajib | `read_easyocr(gambar)` → (teks, skor) |
# | 3 | Pilih rotasi | wajib | teks terbalik 180° dibaca di dua arah, pilih yang terbaca |
# | 4 | Baca SEMUA + simpan | **wajib** | tanpa sel ini, belum ada teks |
# | 5 | Rapikan + ambil field | sesuai soal | LOT / FAB / EXP dengan regex |
# | 6 | Metrik | opsional | hanya bila punya teks benar |

# %% [markdown]
# ## 1. Muat EasyOCR
# Ubah: `EASYOCR_FOLDER` (berisi craft_mlt_25k.pth + latin_g2.pth), `LANGUAGES`, `ALLOWLIST`
# (karakter yang mungkin muncul; `None` = semua).

# %%
import easyocr

EASYOCR_FOLDER = "pretrained/easyocr"   # UBAH
LANGUAGES = ["fr", "en"]                # UBAH: "fr"/"id"/"de" dll. memakai latin_g2.pth
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:/.-, "   # UBAH

reader = easyocr.Reader(LANGUAGES, model_storage_directory=EASYOCR_FOLDER,
                        download_enabled=False, gpu=True, verbose=False)
print("EasyOCR siap")

# %% [markdown]
# ## 2. Fungsi baca satu gambar → (teks, skor 0–1)
# **Butuh:** `reader`, `crops`. EasyOCR mengembalikan kata-kata acak urutannya; `join_lines`
# menyusunnya baris atas → bawah, kiri → kanan.

# %%
def line_text(words):
    # words: [(x_kiri, teks)] → satu baris, kiri ke kanan
    ordered = []
    for _, text in sorted(words):
        ordered.append(text)
    return " ".join(ordered)


def join_lines(items):
    # items: [(y_tengah, x_kiri, tinggi, teks)] → baris atas ke bawah, digabung spasi
    texts, current, last_y = [], [], 0
    for y_center, x_left, height, text in sorted(items):
        if current and y_center - last_y > height / 2:   # turun ke baris baru
            texts.append(line_text(current))
            current = []
        current.append((x_left, text))
        last_y = y_center
    texts.append(line_text(current))
    return " ".join(texts).strip()


print(join_lines([(10, 50, 8, "B"), (10, 5, 8, "A"), (30, 5, 8, "C")]))   # → "A B C"

# %%
import numpy as np
from PIL import Image


def read_easyocr(image):
    items, scores = [], [0.0]
    for corners, text, score in reader.readtext(image, allowlist=ALLOWLIST):
        corners = np.array(corners)
        items.append((corners[:, 1].mean(), corners[:, 0].min(), np.ptp(corners[:, 1]), text))
        scores.append(score)
    return join_lines(items), float(np.mean(scores[1:] or scores))


# coba pada satu potongan yang sudah diperbaiki
example_fixed = np.array(Image.open(crops["fixed_path"].iloc[0]))
print(read_easyocr(example_fixed))

# %% [markdown]
# ## 3. Pilih rotasi dengan skor tertinggi (teks terbalik/menyamping)
# **Butuh:** sel 2. `ROTATIONS`: 0 = asli, 1 = 90°, 2 = 180°, 3 = 270° (berlawanan jarum jam).
# Potongan sudah didatarkan di `potong_perbaiki` → cukup `[0, 2]` (tegak atau terbalik).
# Peringkat = skor × akar panjang teks: supaya bacaan pendek yang "yakin" tidak menang dari bacaan
# lengkap.

# %%
import numpy as np

ROTATIONS = [0, 2]   # UBAH: [0, 1, 2, 3] bila potongan belum didatarkan


def read_best_rotation(image):
    best = ("", 0.0, 0, -1.0)
    for times in ROTATIONS:
        text, score = read_easyocr(np.ascontiguousarray(np.rot90(image, times)))
        rank = score * np.sqrt(len(text.replace(" ", "")))
        if rank > best[3]:
            best = (text, score, times, rank)
    return best[:3]   # (teks, skor, rotasi)


print(read_best_rotation(example_fixed))

# %% [markdown]
# ## 4. Baca SEMUA potongan + simpan
# **Butuh:** `crops["fixed_path"]`, sel 3. **Menghasilkan:** `crops["text"]`, `["conf"]`,
# `["rotation"]`, `outputs/crops.csv`. Ratusan potongan = beberapa menit (GPU).

# %%
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

texts, scores, rotations = [], [], []
for fixed_path in tqdm(crops["fixed_path"]):
    image = np.array(Image.open(fixed_path))
    text, score, times = read_best_rotation(image)
    texts.append(text)
    scores.append(score)
    rotations.append(times)
crops["text"] = texts
crops["conf"] = scores
crops["rotation"] = rotations
crops.to_csv("outputs/crops.csv", index=False)
print(crops[["fixed_path", "text", "conf", "rotation"]].head(10).to_string())
print("teks kosong:", (crops["text"] == "").sum())

# %% [markdown]
# ## 5. Rapikan teks + ambil field dengan regex
# **Butuh:** `crops["text"]`. Ubah: karakter yang boleh di `clean_text`, nama field di baris
# `crops["..."] = ...`. `fix_digits` hanya untuk bagian yang pasti angka (O→0, I/L→1, S→5, B→8).

# %%
import re


def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9:/.\- ]", "", text)   # UBAH: karakter yang boleh
    return re.sub(r"\s+", " ", text).strip()


def extract_field(text, name, pattern=r"[A-Z0-9/.\-]+"):
    # extract_field("LOT: AB12 FAB: 03/2019", "FAB") → "03/2019"
    match = re.search(rf"{name}\s*[:.,]?\s*({pattern})", text)
    if match:
        return match.group(1)
    return ""


def fix_digits(text):
    return text.translate(str.maketrans("OILSB", "01158"))


print(extract_field(clean_text("Lot: ab12  fab: O3/2019"), "FAB"))   # → O3/2019
crops["clean"] = crops["text"].apply(clean_text)
crops["lot"] = crops["clean"].apply(extract_field, args=("LOT",))         # UBAH: nama field
crops["fab"] = crops["clean"].apply(extract_field, args=("FAB",))
crops["fab"] = crops["fab"].apply(fix_digits)
crops.to_csv("outputs/crops.csv", index=False)
print(crops[["clean", "lot", "fab"]].head(10).to_string())

# %% [markdown]
# ## 6. (Opsional) Metrik OCR, bila punya teks benar
# Mis. ketik tangan 30 potongan untuk validasi. CER = jumlah edit karakter / panjang teks benar
# (0 = sempurna). Exact match = persis sama.
# Pemakaian: `cer_score(crops["clean"][:30], teks_benar_30)`.

# %%
def edit_distance(source, target):
    # Levenshtein: sisip/hapus/ganti satu karakter = 1 langkah
    previous = list(range(len(target) + 1))
    for i, char_source in enumerate(source, 1):
        current = [i]
        for column, char_target in enumerate(target, 1):
            substitute = previous[column - 1] + (char_source != char_target)
            current.append(min(previous[column] + 1, current[column - 1] + 1, substitute))
        previous = current
    return previous[-1]


def cer_score(predictions, truths):
    errors, length = 0, 0
    for prediction, truth in zip(predictions, truths):
        errors += edit_distance(str(prediction), str(truth))
        length += len(str(truth))
    return errors / max(length, 1)


print(edit_distance("LOT:AB12", "LOT:A812"), cer_score(["LOT:AB12"], ["LOT:A812"]))
