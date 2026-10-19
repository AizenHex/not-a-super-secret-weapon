# %% [markdown]
# # Recsys — Prediksi Rating (pengguna X akan memberi item Y rating berapa?)
# **Kapan dipakai:** ada tabel (pengguna, item, rating) dan diminta menebak rating untuk
# pasangan pengguna–item di file test. Metrik biasanya RMSE atau MAE.
#
# **Idenya:** paket `surprise` mempelajari "selera" tiap pengguna dan "sifat" tiap item dari
# rating yang ada (SVD = faktor tersembunyi, KNN = pengguna/item yang mirip).
#
# **Langkah:**
# 1. Split menurut waktu: rating terbaru jadi validasi.
# 2. Latih model, evaluasi RMSE/MAE dibanding tebakan rata-rata.
# 3. Latih ulang di semua data, tebak rating test.
#
# **Data demo:** MovieLens 100k (`ratings.csv`: user, item, rating 1–5, timestamp).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from surprise import NMF, SVD, BaselineOnly, Dataset, KNNBaseline, Reader, SVDpp

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/recsys/ml-100k"   # GANTI saat lomba
RATINGS_CSV = f"{DATA_FOLDER}/ratings.csv"

USER_COL = "user"
ITEM_COL = "item"
RATING_COL = "rating"
TIME_COL = "timestamp"
MIN_RATING = 1
MAX_RATING = 5
TRAIN_FRACTION = 0.8      # porsi rating tertua untuk latih
SEED = 42

# %% [markdown]
# ## Baca data

# %%
ratings = pd.read_csv(RATINGS_CSV)
print(ratings.shape)
print("user:", ratings[USER_COL].nunique(), "| item:", ratings[ITEM_COL].nunique())
rating_counts = ratings[RATING_COL].value_counts()
rating_counts.sort_index().plot.bar()
plt.title("Jumlah tiap nilai rating")
plt.show()

# %% [markdown]
# ## Langkah 1 — split menurut waktu
# Rating terbaru jadi validasi, supaya model tidak "mengintip" masa depan.

# %%
ratings = ratings.sort_values(TIME_COL)
cut = int(len(ratings) * TRAIN_FRACTION)
train = ratings.iloc[:cut]
valid = ratings.iloc[cut:]
print("latih:", len(train), "| validasi:", len(valid))
print("validasi dengan user baru:", (~valid[USER_COL].isin(train[USER_COL])).sum())

reader = Reader(rating_scale=(MIN_RATING, MAX_RATING))
train_data = Dataset.load_from_df(train[[USER_COL, ITEM_COL, RATING_COL]], reader)

# %% [markdown]
# ## Langkah 2 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.05, random_state=SEED)
# model = SVDpp(n_factors=20, n_epochs=20, random_state=SEED)    # lebih teliti, lebih lambat
# model = KNNBaseline(k=40, sim_options={"name": "pearson_baseline", "user_based": False})
# model = NMF(n_factors=30, n_epochs=50, random_state=SEED)
# model = BaselineOnly()                                         # rata-rata + bias, cepat

model.fit(train_data.build_full_trainset())

pred_valid = []
for user, item in zip(valid[USER_COL], valid[ITEM_COL]):
    pred_valid.append(model.predict(user, item).est)
pred_valid = np.clip(pred_valid, MIN_RATING, MAX_RATING)
average_guess = np.full(len(valid), train[RATING_COL].mean())
print("RMSE model    :", round(root_mean_squared_error(valid[RATING_COL], pred_valid), 4))
print("MAE model     :", round(mean_absolute_error(valid[RATING_COL], pred_valid), 4))
print("RMSE rata-rata:", round(root_mean_squared_error(valid[RATING_COL], average_guess), 4))

# %% [markdown]
# ## Langkah 3 — latih ulang di semua data + submission
# Demo: `test` = baris validasi tanpa kolom rating. Saat lomba: `test = pd.read_csv(...)`
# dari file test soal.

# %%
test = valid.drop(columns=RATING_COL)
all_data = Dataset.load_from_df(ratings[[USER_COL, ITEM_COL, RATING_COL]], reader)
model.fit(all_data.build_full_trainset())

predictions = []
for user, item in zip(test[USER_COL], test[ITEM_COL]):
    predictions.append(model.predict(user, item).est)
predictions = np.clip(predictions, MIN_RATING, MAX_RATING)

submission = pd.DataFrame()
submission["user"] = test[USER_COL]
submission["item"] = test[ITEM_COL]
submission["rating"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
