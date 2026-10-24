# %% [markdown]
# # Tabular — Feature Engineering (Klasifikasi)
# **Kapan dipakai:** model sudah jalan, dan kita ingin menambah kolom baru (feature) yang
# membantu model, misalnya ukuran keluarga = jumlah saudara + jumlah orang tua + 1.
#
# **Langkah:** tulis feature baru di fungsi `add_features`, lalu bandingkan skor CV sebelum
# dan sesudah. Tambah/hapus baris di fungsi itu untuk mencoba feature lain.
#
# **Data demo:** Titanic (target `survived`).

# %%
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/tabular/classification"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/titanic_train.csv"
TEST_CSV = f"{DATA_FOLDER}/titanic_test.csv"

TARGET = "survived"
NUM_COLS = ["pclass", "age", "sibsp", "parch", "fare"]
CAT_COLS = ["sex", "embarked", "who", "deck"]
AGE_BINS = [0, 12, 18, 35, 60, 100]      # batas kelompok umur
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## Feature baru
# - `family_size`, `is_alone`, `fare_per_person`: hitungan dari kolom yang ada.
# - `fare_vs_class`: harga tiket dibanding rata-rata harga di kelas yang sama.
#   Rata-rata kelas dihitung dari TRAIN saja, lalu dipakai juga untuk test.
# - `age_missing`: 1 kalau umur kosong. Kosong pun bisa jadi informasi.
# - `age_group`: umur dikelompokkan (anak, remaja, dewasa, ...).
# - `sex_pclass`: gabungan dua kategori menjadi satu kategori baru.

# %%
def add_features(table, fare_mean_per_class):
    table = table.copy()
    table["family_size"] = table["sibsp"] + table["parch"] + 1
    table["is_alone"] = (table["family_size"] == 1).astype(int)
    table["fare_per_person"] = table["fare"] / table["family_size"]
    table["fare_vs_class"] = table["fare"] / table["pclass"].map(fare_mean_per_class)
    table["age_missing"] = table["age"].isna().astype(int)
    table["age_group"] = pd.cut(table["age"], AGE_BINS).astype(str)
    table["sex_pclass"] = table["sex"] + "_" + table["pclass"].astype(str)
    return table


fare_mean_per_class = train.groupby("pclass")["fare"].mean()
train_fe = add_features(train, fare_mean_per_class)
test_fe = add_features(test, fare_mean_per_class)

NEW_NUM_COLS = NUM_COLS + ["family_size", "is_alone", "fare_per_person", "fare_vs_class",
                           "age_missing"]
NEW_CAT_COLS = CAT_COLS + ["age_group", "sex_pclass"]
train_fe[NEW_NUM_COLS + NEW_CAT_COLS].head()

# %% [markdown]
# ## Feature tambahan lain
# Hapus `#` pada satu blok untuk mencoba, lalu bandingkan skor CV di bawah.

# %%
# Frequency encoding: berapa kali kategori muncul di train
# deck_count = train["deck"].value_counts()
# train_fe["deck_count"] = train_fe["deck"].map(deck_count).fillna(0)
# test_fe["deck_count"] = test_fe["deck"].map(deck_count).fillna(0)
# NEW_NUM_COLS = NEW_NUM_COLS + ["deck_count"]

# Capping: potong nilai ekstrem ke persentil 1% dan 99% train
# fare_low = train["fare"].quantile(0.01)
# fare_high = train["fare"].quantile(0.99)
# train_fe["fare"] = train_fe["fare"].clip(fare_low, fare_high)
# test_fe["fare"] = test_fe["fare"].clip(fare_low, fare_high)

# Buang kolom angka yang korelasinya > 0.95 dengan kolom sebelumnya
# correlation = train_fe[NEW_NUM_COLS].corr().abs()
# dropped = []
# for i in range(len(NEW_NUM_COLS)):
#     for previous in range(i):
#         if correlation.iloc[i, previous] > 0.95:
#             dropped.append(NEW_NUM_COLS[i])
# print("dibuang:", dropped)
# kept = []
# for col in NEW_NUM_COLS:
#     if col not in dropped:
#         kept.append(col)
# NEW_NUM_COLS = kept

# %% [markdown]
# ## Model
# Pilih SATU `model` dan SATU `encoder` (cara mengubah kategori menjadi angka):
# hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.
# - OneHot: satu kolom 0/1 per kategori.
# - Target: rata-rata target per kategori (dihitung dengan CV agar tidak bocor).
# - Ordinal: satu angka per kategori (cocok untuk model pohon).

# %%
model = LogisticRegression(C=1.0, max_iter=2000)
# model = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED, verbose=-1)
# model = RandomForestClassifier(n_estimators=500, random_state=SEED)

encoder = OneHotEncoder(handle_unknown="ignore")
# encoder = TargetEncoder(random_state=SEED)
# encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

num_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"), encoder)

# %% [markdown]
# ## Evaluasi: CV sebelum vs sesudah feature baru
# F1 macro dihitung di 5 fold. Kalau "sesudah" lebih besar, feature baru membantu.

# %%
preprocess_before = ColumnTransformer([
    ("num", num_steps, NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
])
preprocess_after = ColumnTransformer([
    ("num", num_steps, NEW_NUM_COLS),
    ("cat", cat_steps, NEW_CAT_COLS),
])
pipe_before = make_pipeline(preprocess_before, model)
pipe_after = make_pipeline(preprocess_after, model)

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
y = train[TARGET]
scores_before = cross_val_score(pipe_before, train, y, cv=cv, scoring="f1_macro")
scores_after = cross_val_score(pipe_after, train_fe, y, cv=cv, scoring="f1_macro")
print("F1 sebelum:", round(scores_before.mean(), 4), "±", round(scores_before.std(), 4))
print("F1 sesudah:", round(scores_after.mean(), 4), "±", round(scores_after.std(), 4))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe_after.fit(train_fe, y)
predictions = pipe_after.predict(test_fe)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["survived"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
