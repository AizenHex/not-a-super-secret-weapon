# %% [markdown]
# # Multi-label gambar: satu gambar bisa punya BANYAK label sekaligus
#
# **Butuh:** `train` dengan kolom `path` + satu kolom 0/1 per label, `test` dengan `path`.
# **Menghasilkan:** probabilitas per label, ambang per label, `submission.csv` (2 bentuk).
#
# Bedanya dengan klasifikasi biasa: tiap label = pertanyaan ya/tidak sendiri ("ada kucing?",
# "ada mobil?"). Jadi dilatih satu model ya/tidak per label, dan ambang 0,5 belum tentu terbaik.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | label dalam bentuk teks `"cat dog"` → kolom 0/1 | kalau labelnya teks |
# | 2 | konfigurasi + `embed_images` | **ya** |
# | 3 | vektor SEMUA train + test | **ya** |
# | 4 | model per label + validasi + ambang terbaik per label | **ya** |
# | 5 | latih di semua train → prediksi test pakai ambang | **ya** |
# | 6 | `submission.csv`: kolom 0/1 per label, atau teks label dipisah spasi | **ya** |
#
# Fine-tune CNN untuk multi-label (`finetune.ipynb` dengan 3 perubahan): kepala
# `nn.Linear(..., len(LABELS))`, loss `nn.BCEWithLogitsLoss()`, target `float` 0/1 per label,
# dan `torch.sigmoid(logits)` menggantikan `torch.softmax`.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Mosaik 2×2 gambar CIFAR: tiap gambar berisi 2–4 benda dari 10 kelas.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/vision/multilabel/train.csv")
train["path"] = f"{DEMO_FOLDER}/vision/multilabel/" + train["path"]
test = pd.read_csv(f"{DEMO_FOLDER}/vision/multilabel/test.csv")
test["path"] = f"{DEMO_FOLDER}/vision/multilabel/" + test["path"]
print(train.head(3))

# %% [markdown]
# ## 1. Label teks → kolom 0/1 (lewati kalau sudah kolom 0/1)
# Contoh kolom `tags` berisi `"cat dog"` atau `"cat,dog"`. `str.get_dummies(sep=...)` membuat
# satu kolom per label.

# %%
example = pd.DataFrame({"tags": ["cat dog", "ship", "cat ship truck"]})
dummies = example["tags"].str.get_dummies(sep=" ")    # UBAH: sep="," kalau dipisah koma
print(pd.concat([example, dummies], axis=1))

# %% [markdown]
# ## 2. Konfigurasi + `embed_images` (sama dengan `vision/klasifikasi/embedding.ipynb`)

# %%
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

PRETRAINED_FOLDER = "pretrained"   # UBAH
LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
          "truck"]                 # UBAH: nama kolom 0/1
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
backbone = models.convnext_tiny(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/convnext_tiny/model.pth"))
backbone.classifier[2] = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()
embed_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# %% [markdown]
# Fungsi `embed_images`: daftar path → array (jumlah gambar, 768).

# %%
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm


def embed_images(paths):

    vectors = []
    for start in tqdm(range(0, len(paths), BATCH_SIZE)):
        batch = []
        for path in paths[start:start + BATCH_SIZE]:
            image = ImageOps.exif_transpose(Image.open(path))
            batch.append(embed_transform(image.convert("RGB")))
        with torch.no_grad():
            output = backbone(torch.stack(batch).to(DEVICE))
        vectors.append(output.cpu().numpy())
    return np.concatenate(vectors)


print(embed_images(train["path"].tolist()[:2]).shape)

# %% [markdown]
# ## 3. Vektor SEMUA train + test → `X`, `X_test`, `y` (matriks 0/1: gambar × label)

# %%
X = embed_images(train["path"].tolist())
X_test = embed_images(test["path"].tolist())
y = train[LABELS].to_numpy()
print(X.shape, y.shape, "| rata-rata label per gambar:", y.sum(axis=1).mean())
print(train[LABELS].sum().sort_values())   # label yang sangat jarang → F1-nya akan rendah

# %% [markdown]
# ## 4. Model per label + validasi + ambang terbaik per label → `thresholds`
# `MultiOutputClassifier` = satu model ya/tidak per kolom label. Ambang dicari di val: coba
# 0,05..0,95, ambil yang F1 label itu paling tinggi.

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = MultiOutputClassifier(make_pipeline(StandardScaler(),
                                            LogisticRegression(C=0.1, max_iter=3000)))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
val_probabilities = np.zeros(y_val.shape)
for i in range(len(LABELS)):
    val_probabilities[:, i] = model.estimators_[i].predict_proba(X_val)[:, 1]
print(val_probabilities[:3].round(2))

# %% [markdown]
# Cari ambang terbaik per label di val → `thresholds`, lalu bandingkan dengan ambang 0,5.

# %%
import numpy as np
from sklearn.metrics import f1_score

thresholds = np.zeros(len(LABELS))
for i in range(len(LABELS)):
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.05):
        score = f1_score(y_val[:, i], val_probabilities[:, i] >= threshold, zero_division=0)
        if score > best_f1:
            best_f1, thresholds[i] = score, threshold
print("ambang:", dict(zip(LABELS, thresholds.round(2).tolist())))
for name, cut in [("0.5", 0.5), ("per label", thresholds)]:
    predicted = val_probabilities >= cut
    print(f"ambang {name}: F1 micro {f1_score(y_val, predicted, average='micro'):.4f}"
          f" | F1 macro {f1_score(y_val, predicted, average='macro'):.4f}")

# %% [markdown]
# ## 5. Latih di SEMUA train → prediksi test → `test_predicted` (0/1)
# Gambar tanpa label sama sekali padahal soal bilang minimal satu? Paksa label dengan
# probabilitas tertinggi menjadi 1 (baris bertanda `# opsional`).

# %%
import numpy as np

model.fit(X, y)
test_probabilities = np.zeros((len(X_test), len(LABELS)))
for i in range(len(LABELS)):
    test_probabilities[:, i] = model.estimators_[i].predict_proba(X_test)[:, 1]
test_predicted = (test_probabilities >= thresholds).astype(int)
empty_rows = test_predicted.sum(axis=1) == 0
test_predicted[empty_rows, test_probabilities[empty_rows].argmax(axis=1)] = 1   # opsional
print("baris kosong yang dipaksa:", empty_rows.sum())

# %% [markdown]
# ## 6. `submission.csv`
# Bentuk A: satu kolom 0/1 per label. Bentuk B: satu kolom teks, label dipisah spasi.
# Lihat `sample_submission.csv` soal, pakai salah satu.

# %%
import pandas as pd

submission = pd.DataFrame()                 # bentuk A
submission["id"] = test["id"]               # UBAH
for i in range(len(LABELS)):
    submission[LABELS[i]] = test_predicted[:, i]
submission.to_csv("submission.csv", index=False)
print(submission.head(3))

label_texts = []                            # bentuk B
for row in test_predicted:
    names = []
    for i in range(len(LABELS)):
        if row[i] == 1:
            names.append(LABELS[i])
    label_texts.append(" ".join(names))
submission = pd.DataFrame()
submission["id"] = test["id"]               # UBAH
submission["labels"] = label_texts          # UBAH
submission.to_csv("submission.csv", index=False)   # hapus salah satu bentuk

print(submission.head(3))
