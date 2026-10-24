# %% [markdown]
# # Vektor (gambar/teks) + kolom tabel → satu model: gabung awal, gabung akhir, stacking
#
# **Butuh:** `train_vectors`, `test_vectors` (dari `jembatan/gambar_ke_vektor` atau
# `jembatan/teks_ke_vektor`, urutan baris = urutan tabel) + kolom tabel biasa.
# **Menghasilkan:** tabel skor CV tiap cara gabung, `submission.csv`.
#
# | cara | ide | kapan |
# |---|---|---|
# | tabel saja / vektor saja | baseline masing-masing | **selalu** (tahu sumbangan tiap sumber) |
# | gabung awal (PCA + LightGBM) | vektor diringkas jadi 16–64 kolom, ditempel ke tabel | cepat |
# | gabung akhir (rata-rata proba) | proba model vektor & model tabel dirata-rata | sederhana |
# | stacking | proba OOF model vektor jadi fitur model tabel | biasanya terbaik |
#
# Vektor 512–768 angka langsung ke LightGBM = lambat & mudah overfit. Model LINEAR
# (LogisticRegression/Ridge) paling cocok untuk vektor mentah; LightGBM untuk kolom tabel.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | vektor (hitung cepat resnet18, ATAU `np.load`) | **ya** |
# | 2 | konfigurasi kolom tabel + fold + `oof_and_test` | **ya** |
# | 3 | baseline tabel saja, vektor saja | **ya** |
# | 4 | gabung awal: PCA vektor + kolom tabel | coba |
# | 5 | gabung akhir: rata-rata proba | **ya** |
# | 6 | stacking: proba OOF vektor → fitur tabel | disarankan |
# | 7 | pilih → `submission.csv` | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Foto kecil (CIFAR-10) + data tabel buatan: berat, panjang, kecepatan, warna dominan.
# Tabel saja sulit membedakan kucing vs anjing; gambar saja sulit di foto 32×32 piksel.

# %%
import os

import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

FOLDER = f"{DEMO_FOLDER}/multimodal/gambar_tabular"
train = pd.read_csv(f"{FOLDER}/train.csv")
test = pd.read_csv(f"{FOLDER}/test.csv")
test_labels = pd.read_csv(f"{FOLDER}/test_labels.csv")
for table in [train, test]:
    table["path"] = f"{DEMO_FOLDER}/multimodal/" + table["path"]   # path di csv relatif
    table["path"] = table["path"].apply(os.path.normpath)
print(train.shape, test.shape, "| file ada:", os.path.exists(train["path"].iloc[0]))

# %% [markdown]
# ## 1. Vektor gambar: resnet18 pretrained (cepat). Sudah punya `.npy`? Pakai baris `np.load`.

# %%
import torch
from torch import nn
from torchvision import models

PRETRAINED_FOLDER = "pretrained"   # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
backbone = models.resnet18(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/resnet18/model.pth",
                                    map_location="cpu"))
backbone.fc = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()
print(DEVICE, "| resnet18 siap (vektor 512 angka)")

# %% [markdown]
# `embed(paths)` → array (jumlah gambar, 512). Lengkapnya: `jembatan/gambar_ke_vektor`.

# %%
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

prepare = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])])


def embed(paths, batch_size=128):
    vectors = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start + batch_size]:
            batch.append(prepare(Image.open(path).convert("RGB")))
        with torch.no_grad():
            vectors.append(backbone(torch.stack(batch).to(DEVICE)).cpu().numpy())
    return np.concatenate(vectors)


train_vectors = embed(train["path"].tolist())     # atau: np.load("outputs/train_vectors.npy")
test_vectors = embed(test["path"].tolist())       # atau: np.load("outputs/test_vectors.npy")
print(train_vectors.shape, test_vectors.shape)


# %% [markdown]
# ## 2. Konfigurasi kolom tabel + fold + `oof_and_test(make_model, X, X_test)`
# `take(X, index)`: ambil baris dari numpy array ATAU tabel pandas (berdasarkan posisi).

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TARGET = "label"                                             # UBAH
NUMERIC = ["berat_kg", "panjang_m", "kecepatan_kmj"]         # UBAH
CATEGORICAL = ["warna_dominan"]                              # UBAH
CLASSES = sorted(train[TARGET].unique())
y = train[TARGET].map(dict(zip(CLASSES, range(len(CLASSES))))).to_numpy()
tabular_train = train[NUMERIC].copy()
tabular_test = test[NUMERIC].copy()
for column in CATEGORICAL:
    values = sorted(train[column].astype(str).unique())
    tabular_train[column] = pd.Categorical(train[column].astype(str), categories=values)
    tabular_test[column] = pd.Categorical(test[column].astype(str), categories=values)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tabular_train, y))
print(tabular_train.dtypes.to_dict())

# %% [markdown]
# `take` + `oof_and_test` → proba OOF dan rata-rata proba test semua fold.

# %%
import numpy as np
import pandas as pd


def take(X, index):
    if isinstance(X, pd.DataFrame):
        return X.iloc[index]
    return X[index]


def oof_and_test(make_model, X, X_test):
    oof = np.zeros((len(y), len(CLASSES)))
    test_proba = np.zeros((len(X_test), len(CLASSES)))
    for index_fit, index_val in folds:
        model = make_model().fit(take(X, index_fit), y[index_fit])
        oof[index_val] = model.predict_proba(take(X, index_val))
        test_proba += model.predict_proba(X_test) / len(folds)
    return oof, test_proba


print(take(tabular_train, [0, 1]))

# %% [markdown]
# ## 3. Baseline: tabel saja (LightGBM) dan vektor saja (LogisticRegression)

# %%
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_lightgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1)


def make_linear():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))


oofs, tests = {}, {}
oofs["tabel"], tests["tabel"] = oof_and_test(make_lightgbm, tabular_train, tabular_test)
oofs["vektor"], tests["vektor"] = oof_and_test(make_linear, train_vectors, test_vectors)
for name in oofs:
    print(f"{name:8s} akurasi OOF: {accuracy_score(y, oofs[name].argmax(axis=1)):.4f}")

# %% [markdown]
# ## 4. Gabung awal: vektor diringkas PCA (`N_COMPONENTS` kolom) + kolom tabel → LightGBM
# PCA di-fit di train saja (test hanya transform).

# %%
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

N_COMPONENTS = 32   # UBAH: 16–64
pca = PCA(n_components=N_COMPONENTS, random_state=42).fit(train_vectors)
names = []
for i in range(N_COMPONENTS):
    names.append(f"pc_{i}")
early_train = pd.concat([tabular_train, pd.DataFrame(pca.transform(train_vectors),
                                                     columns=names)], axis=1)
early_test = pd.concat([tabular_test, pd.DataFrame(pca.transform(test_vectors),
                                                   columns=names)], axis=1)
oofs["gabung_awal"], tests["gabung_awal"] = oof_and_test(make_lightgbm, early_train,
                                                          early_test)
print("gabung_awal akurasi OOF:", round(accuracy_score(y, oofs["gabung_awal"].argmax(axis=1)), 4))

# %% [markdown]
# ## 5. Gabung akhir: rata-rata proba model tabel & model vektor (bobot dicari di OOF)

# %%
import numpy as np
from sklearn.metrics import accuracy_score

best_weight, best_score = 0.5, 0.0
for weight in np.arange(0, 1.01, 0.1):
    mixed = weight * oofs["tabel"] + (1 - weight) * oofs["vektor"]
    score = accuracy_score(y, mixed.argmax(axis=1))
    if score > best_score:
        best_weight, best_score = weight, score
oofs["gabung_akhir"] = best_weight * oofs["tabel"] + (1 - best_weight) * oofs["vektor"]
tests["gabung_akhir"] = best_weight * tests["tabel"] + (1 - best_weight) * tests["vektor"]
print(f"bobot tabel {best_weight:.1f} | gabung_akhir akurasi OOF: {best_score:.4f}")

# %% [markdown]
# ## 6. Stacking: proba OOF model vektor (10 kolom) jadi fitur tambahan model tabel
# OOF dipakai untuk train (tidak bocor), proba test dari rata-rata fold dipakai untuk test.

# %%
import pandas as pd
from sklearn.metrics import accuracy_score

proba_names = []
for name in CLASSES:
    proba_names.append(f"proba_vektor_{name}")
stack_train = pd.concat([tabular_train, pd.DataFrame(oofs["vektor"], columns=proba_names)],
                        axis=1)
stack_test = pd.concat([tabular_test, pd.DataFrame(tests["vektor"], columns=proba_names)],
                       axis=1)
oofs["stacking"], tests["stacking"] = oof_and_test(make_lightgbm, stack_train, stack_test)
print("stacking akurasi OOF:", round(accuracy_score(y, oofs["stacking"].argmax(axis=1)), 4))

# %% [markdown]
# ## 7. Pilih cara terbaik (akurasi OOF) → `submission.csv`

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

board = {}
for name in oofs:
    board[name] = round(accuracy_score(y, oofs[name].argmax(axis=1)), 4)
board = pd.Series(board).sort_values(ascending=False)
print(board)
CHOSEN = board.index[0]   # UBAH
submission = pd.DataFrame()
submission["id"] = test["id"]                                                  # UBAH
submission["label"] = np.array(CLASSES)[tests[CHOSEN].argmax(axis=1)]          # UBAH
submission.to_csv("submission.csv", index=False)
print("dipilih:", CHOSEN)

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
from sklearn.metrics import accuracy_score

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
for name in tests:
    guess = np.array(CLASSES)[tests[name].argmax(axis=1)]
    print(f"{name:12s} akurasi test: {accuracy_score(test_labels['label'], guess):.4f}")
