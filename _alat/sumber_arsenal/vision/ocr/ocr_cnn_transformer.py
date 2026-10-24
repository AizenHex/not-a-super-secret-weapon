# %% [markdown]
# # OCR buatan sendiri: CNN + Transformer + CTC (dilatih dari teks sintetis)
# Untuk soal tanpa label teks: render teks acak dengan font → gambar baris berlabel gratis.
# Model membaca SATU baris (tinggi 32, lebar 256). Potongan banyak baris → `split_lines` dulu.
# Bila nanti punya tabel (crop_path, text) asli, fine-tune dengan `LineTableDataset`.
#
# **Butuh:** font .ttf (bawaan Windows cukup). Untuk sel "Baca potongan asli": `crops` dengan
# `fixed_path` dari `potong_perbaiki` (dan `rotation` dari `ocr.ipynb` bila ada).
#
# **Menghasilkan:** model OCR di memori (`model`), kolom `crops["text_cnn"]` di `outputs/crops.csv`.
#
# ## Cara menempel ke notebook-mu
# Notebook ini berdiri sendiri: jalankan **semua sel berurutan dari atas** di notebook baru, atau
# salin semua sel ke bawah sel OCR milikmu. Sel bertanda (Opsional) boleh dilewati. Latih ±menit–jam
# tergantung `EPOCHS` × `SAMPLES_PER_EPOCH`: coba `EPOCHS = 1` dulu untuk cek jalan.

# %% [markdown]
# ## Alfabet + generator teks sintetis
# Ubah: `CHARS` (semua karakter yang mungkin), `FONT_PATHS`, `random_text` (tiru pola teks soal).

# %%
import glob
import random
import string

CHARS = string.ascii_uppercase + string.digits + ":/.-, "   # UBAH: indeks 0 dipakai "blank" CTC
FONT_PATHS = []
for name in ["arial", "arialbd", "ARIALN", "cour", "courbd", "consola", "calibri", "tahoma",
             "verdana", "times", "OCRAEXT"]:   # UBAH: di Linux cari di /usr/share/fonts
    FONT_PATHS = FONT_PATHS + glob.glob(f"C:/Windows/Fonts/{name}.ttf")
MONTHS = ["JAN", "FEV", "MAR", "AVR", "MAI", "JUN", "JUL", "AOU", "SEP", "OCT", "NOV", "DEC"]


def random_code(low=3, high=9):
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choices(alphabet, k=random.randint(low, high)))


def random_text():
    # UBAH: tiru teks soal. Di sini: "LOT:AB123", "FAB:03/2019", "EXP: JAN 2022", acak bebas
    month, year = f"{random.randint(1, 12):02d}", str(random.randint(2015, 2026))
    choices = [f"{random.choice(['LOT', 'LOT N', 'N LOT'])}:{random_code()}",
               f"{random.choice(['FAB', 'EXP', 'PER', 'DE'])}:{month}/{year}",
               f"{random.choice(['FAB', 'EXP', 'PER'])}: {random.choice(MONTHS)} {year}",
               random_code(2, 6) + " " + random_code(2, 8)]
    return random.choice(choices)


print("font:", len(FONT_PATHS), "|", random_text(), "|", random_text())

# %% [markdown]
# ## Gambar → tensor baris (tinggi 32, lebar 256, rasio dijaga, sisa kanan diisi latar)

# %%
import random

import numpy as np
import torch
from PIL import Image

IMAGE_HEIGHT = 32    # UBAH
IMAGE_WIDTH = 256    # UBAH: teks panjang → 384/512


def to_line_tensor(image):
    image = image.convert("L")
    width = max(1, min(IMAGE_WIDTH, round(image.width * IMAGE_HEIGHT / image.height)))
    image = image.resize((width, IMAGE_HEIGHT))
    array = np.asarray(image, dtype=np.float32) / 255
    canvas = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), np.median(array), dtype=np.float32)
    canvas[:, :width] = array
    return torch.tensor(canvas).unsqueeze(0)   # 1 × tinggi × lebar


print(to_line_tensor(Image.new("L", (500, 40), 255)).shape)

# %%
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def render_text(text):
    # teks → gambar mirip foto: font/ukuran acak, latar abu, miring sedikit, blur
    font = ImageFont.truetype(random.choice(FONT_PATHS), random.randint(20, 32))
    left, top, right, bottom = font.getbbox(text)
    background = random.randint(150, 255)
    image = Image.new("L", (right - left + 20, bottom - top + 14), background)
    ImageDraw.Draw(image).text((10 - left, 7 - top), text, font=font, fill=random.randint(0, 90))
    image = image.rotate(random.uniform(-2, 2), expand=True, fillcolor=background)
    return image.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.0)))


render_text("LOT:AB123 FAB:03/2019").save("contoh_sintetis.png")
print(to_line_tensor(render_text("LOT:AB123")).shape)

# %% [markdown]
# ## Dataset: sintetis (tak terbatas) dan tabel asli (crop_path, text) + `collate` untuk CTC

# %%
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def encode(text):
    numbers = []
    for char in text.upper():
        if char in CHARS:
            numbers.append(CHARS.index(char) + 1)   # 0 = blank
    return torch.tensor(numbers, dtype=torch.long)


class SyntheticTextDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        text = random_text()
        image = to_line_tensor(render_text(text))
        image = (image + torch.randn_like(image) * 0.05).clamp(0, 1)   # derau sensor
        return image, encode(text)

# %%
import torch
from torch.utils.data import Dataset


class LineTableDataset(Dataset):
    # tabel asli: kolom crop_path (gambar SATU baris) dan text
    def __init__(self, table):
        self.paths = list(table["crop_path"])
        self.texts = list(table["text"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return to_line_tensor(Image.open(self.paths[index])), encode(self.texts[index])


def collate(batch):
    # CTC butuh: gambar ditumpuk, semua label disambung, panjang tiap label
    images, labels, lengths = [], [], []
    for image, label in batch:
        images.append(image)
        labels.append(label)
        lengths.append(len(label))
    return torch.stack(images), torch.cat(labels), torch.tensor(lengths)

# %% [markdown]
# ## Model: CNN (tinggi 32 → 1, lebar 256 → 64 langkah) → Transformer encoder → kelas per langkah
# Ubah: `D_MODEL`, `LAYERS`. Alternatif klasik: `CrnnModel` (BiLSTM) di sel berikutnya.

# %%
from torch import nn

D_MODEL = 256   # UBAH
LAYERS = 3      # UBAH


def conv_block(channels_in, channels_out, pool):
    return nn.Sequential(nn.Conv2d(channels_in, channels_out, 3, padding=1),
                         nn.BatchNorm2d(channels_out), nn.ReLU(), nn.MaxPool2d(pool))


def make_cnn():
    # 1×32×256 → 64×16×128 → 128×8×64 → 256×4×64 → D_MODEL×1×64
    return nn.Sequential(conv_block(1, 64, (2, 2)), conv_block(64, 128, (2, 2)),
                         conv_block(128, 256, (2, 1)), conv_block(256, D_MODEL, (4, 1)))


print(make_cnn())

# %%
import torch
from torch import nn


class OcrModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cnn = make_cnn()
        self.position = nn.Parameter(torch.zeros(1, IMAGE_WIDTH // 4, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, nhead=4, dim_feedforward=D_MODEL * 2,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Linear(D_MODEL, n_classes)

    def forward(self, images):
        features = self.cnn(images).squeeze(2)          # batch × fitur × langkah
        features = features.permute(0, 2, 1)             # batch × langkah × fitur
        features = self.encoder(features + self.position)
        return self.head(features)                                 # batch × langkah × kelas

# %%
import torch
from torch import nn


class CrnnModel(nn.Module):
    # alternatif: CNN yang sama + BiLSTM 2 lapis (CRNN, baseline OCR klasik)
    def __init__(self, n_classes):
        super().__init__()
        self.cnn = make_cnn()
        self.rnn = nn.LSTM(D_MODEL, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(256, n_classes)

    def forward(self, images):
        features = self.cnn(images).squeeze(2)
        features, _ = self.rnn(features.permute(0, 2, 1))
        return self.head(features)


print(OcrModel(len(CHARS) + 1)(torch.zeros(2, 1, IMAGE_HEIGHT, IMAGE_WIDTH)).shape)

# %% [markdown]
# ## Latih dengan CTC loss + decode greedy + CER
# Ubah: `MODEL_CLASS`, `EPOCHS`, `SAMPLES_PER_EPOCH`, `BATCH` (kecil = GPU aman).

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader

MODEL_CLASS = OcrModel        # UBAH: OcrModel / CrnnModel
EPOCHS = 10                   # UBAH
SAMPLES_PER_EPOCH = 20000     # UBAH
BATCH = 32                    # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = MODEL_CLASS(len(CHARS) + 1).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
train_loader = DataLoader(SyntheticTextDataset(SAMPLES_PER_EPOCH), batch_size=BATCH,
                          collate_fn=collate)
val_loader = DataLoader(SyntheticTextDataset(1000), batch_size=BATCH, collate_fn=collate)


def decode(logits):
    # greedy: ambil kelas terbaik tiap langkah, gabung yang berulang, buang blank
    texts = []
    for sequence in logits.argmax(-1).tolist():
        chars, previous = [], 0
        for number in sequence:
            if number != previous and number != 0:
                chars.append(CHARS[number - 1])
            previous = number
        texts.append("".join(chars))
    return texts

# %%
import torch


def train_one_epoch(loader):
    model.train()
    for images, targets, lengths in loader:
        log_probs = model(images.to(DEVICE)).log_softmax(-1).permute(1, 0, 2)   # langkah × batch
        steps = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
        loss = loss_fn(log_probs, targets, steps, lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return loss.item()


def evaluate(loader):
    # → (CER, exact match); edit_distance dari ocr.py
    model.eval()
    errors, length, exact, total = 0, 0, 0, 0
    with torch.no_grad():
        for images, targets, lengths in loader:
            predictions = decode(model(images.to(DEVICE)))
            for prediction, truth in zip(predictions, decode_targets(targets, lengths)):
                errors += edit_distance(prediction, truth)
                length += len(truth)
                exact += int(prediction == truth)
                total += 1
    return errors / max(length, 1), exact / total

# %%
def decode_targets(targets, lengths):
    texts, start = [], 0
    for length in lengths.tolist():
        chars = []
        for number in targets[start:start + length].tolist():
            chars.append(CHARS[number - 1])
        texts.append("".join(chars))
        start += length
    return texts


def edit_distance(source, target):
    previous = list(range(len(target) + 1))
    for i, char_source in enumerate(source, 1):
        current = [i]
        for column, char_target in enumerate(target, 1):
            substitute = previous[column - 1] + (char_source != char_target)
            current.append(min(previous[column] + 1, current[column - 1] + 1, substitute))
        previous = current
    return previous[-1]


for epoch in range(EPOCHS):
    loss = train_one_epoch(train_loader)
    cer, exact = evaluate(val_loader)
    print(f"epoch {epoch + 1} | loss {loss:.3f} | CER val {cer:.3f} | exact {exact:.3f}")

# %% [markdown]
# ## Potongan banyak baris → baris-baris (profil horizontal piksel gelap)
# Masuk gambar grayscale mendatar (hasil `fix`). Ubah `min_height` bila baris tipis terlewat.

# %%
import cv2
import numpy as np


def split_lines(gray, min_height=8, margin=3):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = mask.sum(axis=1) / 255
    is_text = ink > 0.02 * gray.shape[1]   # baris piksel yang berisi tinta
    lines, start = [], None
    for row, has_ink in enumerate(list(is_text) + [False]):
        if has_ink and start is None:
            start = row
        if not has_ink and start is not None:
            if row - start >= min_height:
                lines.append(gray[max(0, start - margin):row + margin])
            start = None
    return lines


print("baris di contoh:", len(split_lines(np.array(render_text("LOT:AB1")))))

# %% [markdown]
# ## Baca potongan asli → `crops["text_cnn"]`

# %%
import numpy as np
import torch
from PIL import Image


def read_lines(gray):
    lines = split_lines(gray)
    if not lines:
        return ""
    batch = []
    for line in lines:
        batch.append(to_line_tensor(Image.fromarray(line)))
    model.eval()
    with torch.no_grad():
        return " ".join(decode(model(torch.stack(batch).to(DEVICE))))


ROTATION_COLUMN = "rotation"   # UBAH: putaran tegak dari ocr.py / orientasi_cnn.py ("" = tidak)

texts = []
for row in crops.itertuples():
    gray = np.array(Image.open(row.fixed_path))
    if ROTATION_COLUMN:
        gray = np.ascontiguousarray(np.rot90(gray, getattr(row, ROTATION_COLUMN)))
    texts.append(read_lines(gray))
crops["text_cnn"] = texts
crops.to_csv("outputs/crops.csv", index=False)
print(crops[["text_cnn"]].head(10).to_string())

# %% [markdown]
# ## (Opsional) Fine-tune dengan data asli: tabel `line_table` (crop_path = gambar satu baris, text)
# Campur dengan sintetis supaya tidak lupa. Ubah: `FINE_TUNE_EPOCHS`.

# %%
from torch.utils.data import ConcatDataset, DataLoader

FINE_TUNE_EPOCHS = 5   # UBAH

mixed = ConcatDataset([LineTableDataset(line_table), SyntheticTextDataset(len(line_table))])
fine_tune_loader = DataLoader(mixed, batch_size=BATCH, shuffle=True, collate_fn=collate)
for epoch in range(FINE_TUNE_EPOCHS):
    print(f"fine-tune {epoch + 1} | loss {train_one_epoch(fine_tune_loader):.3f}")
