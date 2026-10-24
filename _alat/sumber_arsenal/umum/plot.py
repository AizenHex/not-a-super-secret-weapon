# %% [markdown]
# # Grafik cepat dengan matplotlib: sebaran, hitungan, hubungan, waktu, korelasi, gambar
#
# **Butuh:** tabel pandas `table` (atau array numpy). **Menghasilkan:** grafik di layar + file
# `.png` di `outputs/plots/` (berguna untuk laporan/presentasi).
#
# ## Cara menempel ke notebook-mu
# Tiap bagian berdiri sendiri. Ganti nama kolom di baris `# UBAH`. Sel `import` cukup sekali.
#
# | ingin melihat | bagian |
# |---|---|
# | sebaran tiap kolom angka (histogram) | 1 |
# | jumlah per kategori / per label (cek data timpang) | 2 |
# | hubungan dua kolom angka, diwarnai kelas | 3 |
# | nilai dari waktu ke waktu (per hari/minggu/bulan) | 4 |
# | sebaran angka per kelas (boxplot) | 5 |
# | korelasi antar kolom (heatmap) | 6 |
# | confusion matrix (tebakan vs jawaban) | 7 |
# | contoh gambar per kelas (grid) | 8 |
# | kurva loss/akurasi saat training | 9 |
#
# Pola dasar matplotlib (dipakai di semua bagian):
# `fig, axes = plt.subplots(baris, kolom, figsize=(lebar, tinggi))` → gambar di `axes[i]` →
# `fig.tight_layout()` → `fig.savefig(...)` → `plt.show()`.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import os

import numpy as np
import pandas as pd
from PIL import Image

rng = np.random.default_rng(0)
table = pd.DataFrame({
    "age": rng.integers(18, 60, 200),
    "income": rng.normal(5_000_000, 1_500_000, 200),
    "city": rng.choice(["jakarta", "bandung", "surabaya", "medan"], 200, p=[0.5, 0.2, 0.2, 0.1]),
    "label": rng.choice(["ya", "tidak"], 200, p=[0.3, 0.7]),
    "date": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 120, 200), unit="D"),
})
table["spend"] = table["income"] * 0.3 + rng.normal(0, 300_000, 200)

os.makedirs("data/images", exist_ok=True)
image_rows = []
for i in range(12):
    pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    path = f"data/images/img_{i}.png"
    Image.fromarray(pixels).save(path)
    image_rows.append({"path": path, "label": ["kucing", "anjing", "burung"][i % 3]})
images = pd.DataFrame(image_rows)
print(table.head())

# %%
import os

import matplotlib.pyplot as plt

PLOT_FOLDER = "outputs/plots"   # UBAH
os.makedirs(PLOT_FOLDER, exist_ok=True)
plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## 1. Histogram tiap kolom angka
# `bins` = jumlah batang. Ekor panjang ke kanan (uang, jumlah) → coba `np.log1p(kolom)`.

# %%
import matplotlib.pyplot as plt
import numpy as np

NUMBER_COLUMNS = ["age", "income", "spend"]   # UBAH
N_COLUMNS_PER_ROW = 3                         # UBAH

n_rows = int(np.ceil(len(NUMBER_COLUMNS) / N_COLUMNS_PER_ROW))
fig, axes = plt.subplots(n_rows, N_COLUMNS_PER_ROW, figsize=(4 * N_COLUMNS_PER_ROW, 3 * n_rows))
axes = np.array(axes).reshape(-1)   # selalu 1 dimensi, berapa pun barisnya
for i in range(len(axes)):
    if i >= len(NUMBER_COLUMNS):
        axes[i].axis("off")   # kotak sisa dikosongkan
        continue
    column = NUMBER_COLUMNS[i]
    axes[i].hist(table[column].dropna(), bins=30)
    axes[i].set_title(column)
fig.tight_layout()
fig.savefig(f"{PLOT_FOLDER}/histogram.png")
plt.show()

# %% [markdown]
# ## 2. Jumlah per kategori / per label
# Untuk label: kalau satu kelas jauh lebih banyak (timpang), akurasi menyesatkan → pakai F1.

# %%
import matplotlib.pyplot as plt

COUNT_COLUMN = "city"   # UBAH: kolom kategori atau label

counts = table[COUNT_COLUMN].value_counts()
print(counts)
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(counts.index.astype(str), counts.values)
for i in range(len(counts)):
    ax.text(i, counts.values[i], str(counts.values[i]), ha="center", va="bottom")   # angka di atas
ax.set_title(f"jumlah per {COUNT_COLUMN}")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(f"{PLOT_FOLDER}/count_{COUNT_COLUMN}.png")
plt.show()

# %% [markdown]
# ## 3. Hubungan dua kolom angka (scatter), diwarnai per kelas
# `alpha` < 1 membuat titik yang menumpuk terlihat lebih gelap.

# %%
import matplotlib.pyplot as plt

X_COLUMN = "income"   # UBAH
Y_COLUMN = "spend"    # UBAH
COLOR_COLUMN = "label"   # UBAH: kolom kelas, atau None untuk satu warna

fig, ax = plt.subplots(figsize=(6, 4))
if COLOR_COLUMN is None:
    ax.scatter(table[X_COLUMN], table[Y_COLUMN], s=10, alpha=0.6)
else:
    for value in table[COLOR_COLUMN].unique():
        part = table[table[COLOR_COLUMN] == value]
        ax.scatter(part[X_COLUMN], part[Y_COLUMN], s=10, alpha=0.6, label=str(value))
    ax.legend()
ax.set_xlabel(X_COLUMN)
ax.set_ylabel(Y_COLUMN)
fig.tight_layout()
fig.savefig(f"{PLOT_FOLDER}/scatter_{X_COLUMN}_{Y_COLUMN}.png")
plt.show()

# %% [markdown]
# ## 4. Nilai dari waktu ke waktu
# `resample` butuh kolom tanggal bertipe datetime (lihat `tabel_bersih.ipynb` bagian 5).
# Periode: `"D"` harian, `"W"` mingguan, `"MS"` bulanan. Fungsi: `"sum"`, `"mean"`, `"count"`.

# %%
import matplotlib.pyplot as plt

DATE_COLUMN = "date"     # UBAH
VALUE_COLUMN = "spend"   # UBAH
PERIOD = "W"             # UBAH

ordered = table.sort_values(DATE_COLUMN)
per_period = ordered.resample(PERIOD, on=DATE_COLUMN)[VALUE_COLUMN].sum()   # UBAH: sum/mean/count
print(per_period.head())
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(per_period.index, per_period.values, marker="o")
ax.set_title(f"{VALUE_COLUMN} per {PERIOD}")
fig.autofmt_xdate()   # label tanggal dimiringkan supaya tidak bertumpuk
fig.tight_layout()
fig.savefig(f"{PLOT_FOLDER}/time_{VALUE_COLUMN}.png")
plt.show()

# %% [markdown]
# ## 5. Sebaran angka per kelas (boxplot)
# Kotak = 50% data di tengah, garis tengah = median, titik = outlier.
# Kotak antar kelas sangat beda → kolom ini kemungkinan fitur yang berguna.

# %%
import matplotlib.pyplot as plt

VALUE_COLUMN = "income"   # UBAH
GROUP_COLUMN = "city"     # UBAH

groups = sorted(table[GROUP_COLUMN].unique())
values_per_group = []
for group in groups:
    values_per_group.append(table.loc[table[GROUP_COLUMN] == group, VALUE_COLUMN].dropna())
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot(values_per_group)
ax.set_xticks(range(1, len(groups) + 1))
ax.set_xticklabels(groups)
ax.set_title(f"{VALUE_COLUMN} per {GROUP_COLUMN}")
fig.tight_layout()
fig.savefig(f"{PLOT_FOLDER}/box_{VALUE_COLUMN}_{GROUP_COLUMN}.png")
plt.show()

# %% [markdown]
# ## 6. Heatmap korelasi
# Nilai -1..1. Dekat 1 / -1 = dua kolom bergerak bersama (salah satunya mungkin bisa dibuang).

# %%
import matplotlib.pyplot as plt

numeric = table.select_dtypes("number")
correlation = numeric.corr()
names = correlation.columns.tolist()
fig, ax = plt.subplots(figsize=(1 + len(names), 1 + len(names)))
picture = ax.imshow(correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
for row in range(len(names)):
    for col in range(len(names)):
        ax.text(col, row, f"{correlation.values[row, col]:.2f}", ha="center", va="center")
fig.colorbar(picture)
fig.tight_layout()
fig.savefig(f"{PLOT_FOLDER}/correlation.png")
plt.show()

# %% [markdown]
# ## 7. Confusion matrix (tanpa sklearn)
# **Butuh:** dua list/kolom sepanjang sama: `y_true` (jawaban) dan `y_pred` (tebakan model).
# Baris = jawaban, kolom = tebakan. Diagonal = benar. Angka besar di luar diagonal = kelas yang
# sering tertukar.

# %%
import matplotlib.pyplot as plt
import numpy as np

y_true = table["label"].tolist()                                     # UBAH
y_pred = np.random.default_rng(1).choice(["ya", "tidak"], len(table))   # UBAH: tebakan model

classes = sorted(set(y_true) | set(y_pred))
position = {}
for i in range(len(classes)):
    position[classes[i]] = i
matrix = np.zeros((len(classes), len(classes)), dtype=int)
for answer, guess in zip(y_true, y_pred):
    matrix[position[answer], position[guess]] += 1
fig, ax = plt.subplots(figsize=(1.5 + len(classes), 1.5 + len(classes)))
ax.imshow(matrix, cmap="Blues")
for row in range(len(classes)):
    for col in range(len(classes)):
        ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes)
ax.set_xlabel("tebakan")
ax.set_ylabel("jawaban")
fig.tight_layout()
fig.savefig(f"{PLOT_FOLDER}/confusion.png")
plt.show()

# %% [markdown]
# ## 8. Contoh gambar per kelas (grid)
# **Butuh:** tabel `images` dengan kolom `path` dan `label` (dari `baca_folder.ipynb` bagian 1).
# Satu baris grid = satu kelas.

# %%
import matplotlib.pyplot as plt
from PIL import Image

PER_CLASS = 4   # UBAH: jumlah contoh per kelas

classes = sorted(images["label"].unique())
fig, axes = plt.subplots(len(classes), PER_CLASS, figsize=(2 * PER_CLASS, 2 * len(classes)),
                         squeeze=False)   # squeeze=False: axes selalu 2 dimensi
for row in range(len(classes)):
    examples = images[images["label"] == classes[row]].head(PER_CLASS)
    paths = examples["path"].tolist()
    for col in range(PER_CLASS):
        axes[row, col].axis("off")
        if col < len(paths):
            axes[row, col].imshow(Image.open(paths[col]))
            axes[row, col].set_title(classes[row], fontsize=9)
fig.tight_layout()
fig.savefig(f"{PLOT_FOLDER}/image_grid.png")
plt.show()

# %% [markdown]
# ## 9. Kurva loss / akurasi saat training
# **Butuh:** list angka per epoch, mis. di loop training tambahkan
# `history["train_loss"].append(train_loss)`. Garis val naik saat train turun = overfitting.

# %%
import matplotlib.pyplot as plt

history = {"train_loss": [0.9, 0.6, 0.45, 0.35, 0.28, 0.22],   # UBAH: isi dari loop training
           "val_loss": [0.95, 0.7, 0.55, 0.5, 0.52, 0.56]}

fig, ax = plt.subplots(figsize=(6, 3))
for name, values in history.items():
    ax.plot(range(1, len(values) + 1), values, marker="o", label=name)
ax.set_xlabel("epoch")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_FOLDER}/history.png")
plt.show()
