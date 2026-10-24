# %% [markdown]
# # Menjelaskan model tabel: fitur penting, arah pengaruh, alasan satu prediksi
#
# **Butuh:** model terlatih (LightGBM/XGBoost/CatBoost/sklearn), `X_val` + `y_val`.
# **Menghasilkan:** tabel kepentingan fitur (4 cara), plot SHAP, PDP, penjelasan per baris
# (bahan slide/laporan: "kenapa model memutuskan begini").
#
# | cara | menjawab | catatan |
# |---|---|---|
# | gain / split bawaan | fitur yang sering dipakai pohon | cepat, bias ke fitur banyak nilai |
# | permutation importance | skor turun berapa kalau fitur diacak | jujur, dihitung di VAL |
# | SHAP (Lundberg & Lee 2017) | sumbangan tiap fitur ke tiap prediksi | arah + besar + per baris |
# | PDP / ICE | tebakan berubah bagaimana saat fitur naik | bentuk hubungan |
#
# Hati-hati: dua fitur yang berkorelasi kuat "berbagi" kepentingan → keduanya tampak lemah
# pada permutation importance. Penjelasan = perilaku MODEL, bukan sebab-akibat di dunia nyata.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | latih model + bagi val | (pakai model sendiri) |
# | 2 | importance bawaan (gain, split) | cepat |
# | 3 | permutation importance di val | **ya** |
# | 4 | SHAP: beeswarm (ringkasan) + tabel rata-rata \|SHAP\| | **ya** |
# | 5 | SHAP: ketergantungan satu fitur | disarankan |
# | 6 | SHAP: penjelasan satu baris (waterfall) + teks | untuk presentasi |
# | 7 | PDP + ICE | disarankan |
# | 8 | tabel gabungan 4 cara | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Harga median rumah per blok di California (dalam 100 ribu USD).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

data = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/geo/california_train.csv")
print(data.shape)
print(data.head(3))

# %% [markdown]
# ## 1. Latih model + sisihkan val (penjelasan dihitung di data yang TIDAK dilatih)

# %%
import lightgbm as lgb
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"   # UBAH
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
            "Latitude", "Longitude"]   # UBAH
X_fit, X_val, y_fit, y_val = train_test_split(data[FEATURES], data[TARGET], test_size=0.25,
                                              random_state=42)
model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, verbose=-1,
                          random_state=42)
model.fit(X_fit, y_fit)
print("R² val:", round(model.score(X_val, y_val), 4))

# %% [markdown]
# ## 2. Importance bawaan: `split` (berapa kali dipakai) vs `gain` (total perbaikan)

# %%
import pandas as pd

booster = model.booster_
importance = pd.DataFrame(index=FEATURES)
importance["split"] = booster.feature_importance(importance_type="split")
importance["gain"] = booster.feature_importance(importance_type="gain")
for column in ["split", "gain"]:
    importance[column] = importance[column] / importance[column].sum()
print(importance.sort_values("gain", ascending=False).round(3))

# %% [markdown]
# ## 3. Permutation importance di val: acak satu kolom → skor turun berapa?
# `n_repeats` diulang supaya ada simpangan baku. Turun ≤ 0 → fitur tidak berguna (atau
# informasinya sudah ada di fitur lain yang berkorelasi).

# %%
import pandas as pd
from sklearn.inspection import permutation_importance

SCORING = "r2"   # UBAH: "neg_root_mean_squared_error", "roc_auc", "f1_macro", ...
permutation = permutation_importance(model, X_val, y_val, scoring=SCORING, n_repeats=5,
                                     random_state=42, n_jobs=-1)
importance["permutation"] = permutation.importances_mean
importance["permutation_std"] = permutation.importances_std
print(importance[["permutation", "permutation_std"]].sort_values("permutation",
                                                                  ascending=False).round(4))

# %% [markdown]
# ## 4. SHAP: beeswarm = ringkasan semua baris val
# Tiap titik = satu baris. Posisi kanan = menaikkan tebakan; warna = nilai fitur (merah tinggi).
# Contoh baca: "MedInc tinggi (merah) → di kanan → pendapatan tinggi menaikkan harga".
# Klasifikasi: SHAP dalam skala log-odds; multikelas → satu set SHAP per kelas.

# %%
import matplotlib.pyplot as plt
import numpy as np
import shap

explainer = shap.TreeExplainer(model)
explanation = explainer(X_val)
shap.plots.beeswarm(explanation, max_display=10, show=False)
plt.tight_layout()
plt.show()
importance["mean_abs_shap"] = np.abs(explanation.values).mean(axis=0)
print("tebakan dasar (rata-rata):", round(float(explanation.base_values[0]), 3))

# %% [markdown]
# ## 5. SHAP satu fitur: nilai fitur (x) vs sumbangannya (y), diwarnai fitur lain
# Menunjukkan bentuk hubungan + interaksi (warna berbeda pola = ada interaksi).

# %%
import matplotlib.pyplot as plt
import shap

FEATURE_TO_SHOW = "MedInc"   # UBAH
shap.plots.scatter(explanation[:, FEATURE_TO_SHOW], color=explanation, show=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# Fitur lokasi: peta sumbangan SHAP gabungan Latitude + Longitude (merah = lokasi mahal).

# %%
import matplotlib.pyplot as plt

location_effect = explanation[:, "Latitude"].values + explanation[:, "Longitude"].values
fig, ax = plt.subplots(figsize=(6, 5))
points = ax.scatter(X_val["Longitude"], X_val["Latitude"], c=location_effect, cmap="coolwarm",
                    s=6)
plt.colorbar(points, label="sumbangan lokasi ke tebakan")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()

# %% [markdown]
# ## 6. Penjelasan satu baris: waterfall + kalimat siap pakai di laporan
# `ROW` = posisi baris di `X_val` (mis. tebakan paling salah, atau contoh menarik).

# %%
import matplotlib.pyplot as plt
import numpy as np
import shap

ROW = 0   # UBAH
shap.plots.waterfall(explanation[ROW], max_display=8, show=False)
plt.tight_layout()
plt.show()
contribution = explanation.values[ROW]
order = np.argsort(-np.abs(contribution))[:3]
print(f"Tebakan {model.predict(X_val.iloc[[ROW]])[0]:.2f} "
      f"(rata-rata {explanation.base_values[ROW]:.2f}). Alasan utama:")
for i in order:
    direction = "menurunkan"
    if contribution[i] > 0:
        direction = "menaikkan"
    print(f"- {FEATURES[i]} = {X_val.iloc[ROW, i]:.2f} {direction} {abs(contribution[i]):.2f}")

# %% [markdown]
# ## 7. PDP + ICE: tebakan rata-rata (garis tebal) dan per baris (garis tipis) saat fitur diubah
# Garis tipis yang menyebar beda arah = efek fitur bergantung fitur lain (interaksi).

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

PDP_FEATURES = ["MedInc", "AveOccup", "HouseAge"]   # UBAH
fig, ax = plt.subplots(1, len(PDP_FEATURES), figsize=(13, 3.5))
PartialDependenceDisplay.from_estimator(model, X_val.sample(400, random_state=42),
                                        PDP_FEATURES, kind="both", ax=ax,
                                        ice_lines_kw={"alpha": 0.1})
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Tabel gabungan: peringkat fitur menurut 4 cara
# Fitur yang rendah di SEMUA cara → aman dibuang (cek ulang skor CV setelah dibuang).
# Contoh di sini: Latitude/Longitude rendah di gain tapi tertinggi di permutation → keduanya
# bekerja BERSAMA (lokasi); gain membagi jasanya ke banyak pemisahan kecil.

# %%
import pandas as pd

ranks = pd.DataFrame(index=FEATURES)
for column in ["split", "gain", "permutation", "mean_abs_shap"]:
    ranks[column] = importance[column].rank(ascending=False).astype(int)
ranks["rata_rata_peringkat"] = ranks.mean(axis=1)
print(ranks.sort_values("rata_rata_peringkat").to_string())
