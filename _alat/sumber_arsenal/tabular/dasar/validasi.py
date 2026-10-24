# %% [markdown]
# # Validasi tabel: skor lokal yang bisa dipercaya, kebocoran, beda train vs test
#
# **Butuh:** `train` (fitur + target), `test`. **Menghasilkan:** kolom `fold` di train
# (dipakai SEMUA notebook tabular lain), daftar kolom bocor, daftar kolom yang bergeser.
#
# Dua fondasi menurut para Kaggle Grandmaster (playbook NVIDIA 2025): eksperimen cepat dan
# **validasi lokal yang benar**. Skor CV yang salah = memilih model/fitur yang salah.
#
# | situasi data | pembagian fold | sel |
# |---|---|---|
# | baris saling bebas, target angka | `KFold` | 1 |
# | klasifikasi | `StratifiedKFold` (proporsi kelas sama tiap fold) | 1 |
# | satu orang/toko/pasien punya banyak baris | `GroupKFold` (grup utuh di satu fold) | 1–2 |
# | ada waktu dan test = masa depan | potong per waktu / `TimeSeriesSplit` | 1, 5 |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | `make_folds(table, kind, ...)` → kolom `fold` | **ya** |
# | 2 | bukti: fold acak vs per grup (skor terlalu optimis) | baca |
# | 3 | `leak_scan`: kolom yang sendirian menebak target hampir sempurna | **ya** |
# | 4 | adversarial validation: train vs test bisa dibedakan? kolom mana? | **ya** |
# | 5 | data berwaktu: fold acak vs potong waktu | kalau ada waktu |
# | 6 | stabilitas: CV diulang beberapa seed | disarankan |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# `mpg`: mobil + pabrikan (grup). `titanic`: ada kolom bocor. `fraud`: transaksi berwaktu.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

cars = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/grouped/mpg_train.csv")
titanic = pd.read_csv(f"{DEMO_FOLDER}/tabular/classification/titanic_train.csv")
fraud = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/fraud_transaksi/transaksi_sparkov.csv")
print(cars.shape, titanic.shape, fraud.shape)
print(cars["maker"].value_counts().head())

# %% [markdown]
# ## 1. `make_folds(table, kind, target, group, n_splits)` → kolom `fold` (0..n_splits-1)
# Simpan kolom `fold` sekali, lalu SEMUA model memakai fold yang sama → skor bisa
# dibandingkan dan prediksi out-of-fold (OOF) bisa digabung (ensembling).

# %%
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def make_folds(table, kind, target=None, group=None, n_splits=5, seed=42):
    folds = np.zeros(len(table), dtype=int)
    if kind == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        parts = splitter.split(table, table[target])
    elif kind == "group":
        splitter = GroupKFold(n_splits=n_splits)
        parts = splitter.split(table, groups=table[group])
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        parts = splitter.split(table)
    for fold, (_, index_val) in enumerate(parts):
        folds[index_val] = fold
    return folds


cars["fold"] = make_folds(cars, "group", group="maker")
print(cars.groupby("fold")["maker"].nunique())   # tiap pabrikan hanya di satu fold

# %% [markdown]
# ## 2. Bukti: fold acak vs fold per grup
# Soal: menebak mpg mobil dari pabrikan yang **belum pernah dilihat**. Fold acak membiarkan
# mobil kembar (pabrikan & nama sama) ada di train dan val → skor terlalu bagus.

# %%
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]


def cv_score(table, features, target, fold_column):
    errors = []
    for fold in sorted(table[fold_column].unique()):
        part_train = table[table[fold_column] != fold]
        part_val = table[table[fold_column] == fold]
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1)
        model.fit(part_train[features], part_train[target])
        errors.append(mean_absolute_error(part_val[target], model.predict(part_val[features])))
    return round(float(np.mean(errors)), 3)


cars["fold_random"] = make_folds(cars, "kfold")
print("MAE fold acak    :", cv_score(cars, FEATURES, "mpg", "fold_random"))
print("MAE fold per grup:", cv_score(cars, FEATURES, "mpg", "fold"), "← lebih jujur")

# %% [markdown]
# ## 3. `leak_scan(table, target)` → kolom yang sendirian menebak target hampir sempurna
# Tiap kolom dicoba SENDIRIAN dengan pohon kecil (cross-validation). Skor ≈ 1,0 = curiga bocor
# (kolom dibuat SETELAH kejadian, salinan target, atau id yang berurutan dengan target).
# Klasifikasi biner: AUC. Kolom teks diubah ke nomor kategori dulu.

# %%
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def leak_scan(table, target, skip=()):
    rows = []
    for column in table.columns:
        if column == target or column in skip:
            continue
        values = table[[column]].copy()
        if values[column].dtype == object or values[column].dtype == bool:
            values[column] = values[column].astype("category").cat.codes
        values = values.fillna(-999)
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
        auc = cross_val_score(tree, values, table[target], cv=5, scoring="roc_auc").mean()
        rows.append({"column": column, "auc_sendiri": round(auc, 4)})
    return pd.DataFrame(rows).sort_values("auc_sendiri", ascending=False)


print(leak_scan(titanic, "survived", skip=["id"]).head(8))   # `alive` = salinan target!

# %% [markdown]
# ## 4. Adversarial validation: bisakah model membedakan train dari test?
# Gabung train (label 0) + test (label 1), latih klasifikasi. AUC ≈ 0,5 = mirip (bagus).
# AUC > 0,7 = bergeser → kolom penting teratas = penyebabnya: buang, ubah (mis. tanggal →
# jam/hari), atau pilih val yang mirip test (baris train dengan proba "test" tertinggi).

# %%
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

fraud["waktu_transaksi"] = pd.to_datetime(fraud["waktu_transaksi"])
fraud["unix_time"] = fraud["waktu_transaksi"].astype("int64") // 10**9
fraud["hour"] = fraud["waktu_transaksi"].dt.hour
fraud["category_code"] = fraud["category"].astype("category").cat.codes
cut = fraud["waktu_transaksi"].quantile(0.7)             # demo: test = 30% waktu terakhir
fraud_train = fraud[fraud["waktu_transaksi"] < cut]
fraud_test = fraud[fraud["waktu_transaksi"] >= cut]
CHECK = ["amt", "city_pop", "lat", "long", "hour", "category_code", "unix_time"]   # UBAH


def adversarial_auc(train_part, test_part, columns):
    both = pd.concat([train_part[columns], test_part[columns]], ignore_index=True)
    is_test = np.r_[np.zeros(len(train_part)), np.ones(len(test_part))]
    model = LGBMClassifier(n_estimators=200, importance_type="gain", verbose=-1)
    proba = cross_val_predict(model, both, is_test, cv=5, method="predict_proba")[:, 1]
    model.fit(both, is_test)
    importance = pd.Series(model.feature_importances_, index=columns)
    return roc_auc_score(is_test, proba), importance.sort_values(ascending=False)


auc, importance = adversarial_auc(fraud_train, fraud_test, CHECK)
print("AUC train vs test:", round(auc, 4))
print((importance / importance.sum()).round(3).head(4))   # porsi gain: penyebab di atas

# %% [markdown]
# Buang kolom penyebab (di sini `unix_time`: test selalu lebih baru) lalu cek ulang.

# %%
CHECK_WITHOUT_TIME = ["amt", "city_pop", "lat", "long", "hour", "category_code"]   # UBAH
auc, importance = adversarial_auc(fraud_train, fraud_test, CHECK_WITHOUT_TIME)
print("AUC tanpa unix_time:", round(auc, 4), "← mendekati 0,5 = aman")

# %% [markdown]
# ## 5. Data berwaktu: fold acak vs potong waktu
# Kalau test = masa depan, val juga harus "masa depan" dari train. Fold acak membiarkan model
# melihat pola dari hari yang sama (bocor) → skor CV lebih tinggi dari skor test sebenarnya.

# %%
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES_FRAUD = ["amt", "city_pop", "hour", "category_code", "unix_time"]   # UBAH
TARGET_FRAUD = "is_fraud"
early, late = train_test_split(fraud_train, test_size=0.25, shuffle=False)   # potong waktu
random_a, random_b = train_test_split(fraud_train, test_size=0.25, random_state=42)
for name, (part_fit, part_val) in {"acak": (random_a, random_b), "waktu": (early, late)}.items():
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1)
    model.fit(part_fit[FEATURES_FRAUD], part_fit[TARGET_FRAUD])
    auc = roc_auc_score(part_val[TARGET_FRAUD], model.predict_proba(part_val[FEATURES_FRAUD])[:, 1])
    print(f"val {name:5s}: AUC {auc:.4f}")
model.fit(fraud_train[FEATURES_FRAUD], fraud_train[TARGET_FRAUD])
test_auc = roc_auc_score(fraud_test[TARGET_FRAUD],
                         model.predict_proba(fraud_test[FEATURES_FRAUD])[:, 1])
print("test sebenarnya: AUC", round(test_auc, 4), "→ pilih val yang paling dekat angka ini")

# %% [markdown]
# Banyak potongan waktu (`TimeSeriesSplit`): train selalu SEBELUM val, val bergeser maju.
# Data harus sudah diurutkan menurut waktu.

# %%
from sklearn.model_selection import TimeSeriesSplit

ordered = fraud_train.sort_values("waktu_transaksi").reset_index(drop=True)
for fold, (index_fit, index_val) in enumerate(TimeSeriesSplit(n_splits=4).split(ordered)):
    first = ordered["waktu_transaksi"].iloc[index_val[0]].date()
    last = ordered["waktu_transaksi"].iloc[index_val[-1]].date()
    print(f"fold {fold}: train {len(index_fit):6d} baris | val {first} s.d. {last}")

# %% [markdown]
# ## 6. Stabilitas: ulangi CV dengan beberapa seed
# Selisih dua model lebih kecil dari simpangan baku antar seed = belum tentu benar-benar lebih
# baik. Data kecil (< 2000 baris) paling butuh ini.

# %%
import numpy as np

scores = []
for seed in [0, 1, 2]:   # UBAH
    cars["fold_seed"] = make_folds(cars, "kfold", seed=seed)
    scores.append(cv_score(cars, FEATURES, "mpg", "fold_seed"))
print("MAE per seed:", scores, "| simpangan baku:", round(float(np.std(scores)), 3))
