# %% [markdown]
# # Visualisasi — Evaluasi Regresi
# **Kapan dipakai:** "seberapa dekat tebakan dengan nilai asli? di mana model sering salah?
# apakah kesalahannya acak atau berpola?"
#
# **Isi:** perbandingan model vs baseline, prediksi vs aktual, residual vs prediksi, residual per
# fitur dan per grup, distribusi error + QQ plot, error per kuantil, peta error, plotly interaktif.
# Semua gambar disimpan di folder notebook.
#
# **Data demo:** California housing (harga rumah, ada lat/lon). Model dilatih cepat di sel Model,
# fokus notebook ini pada grafiknya.

# %%
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/tabular_extra/geo"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/california_train.csv"

ID_COL = "id"
TARGET = "MedHouseVal"     # nilai median rumah (satuan 100 ribu USD)
CHECK_FEATURE = "MedInc"   # fitur yang residualnya ingin diperiksa
GROUP_FEATURE = "HouseAge" # fitur yang dibagi 4 kelompok untuk residual per grup
SEED = 42

# %% [markdown]
# ## Gaya grafik (sama di semua notebook visualisasi)
# Font besar supaya terbaca di slide, palet tetap, garis tepi atas/kanan dimatikan.

# %%
PALETTE = ["#2E6FB7", "#F28E2B", "#3BA272", "#D1495B", "#7B5EA7", "#9AA0A6"]
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = PALETTE
plt.rcParams.update({
    "figure.figsize": (8, 5), "figure.dpi": 100, "savefig.dpi": 200,
    "font.size": 14, "axes.titlesize": 18, "axes.titleweight": "bold", "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})
sns.set_palette(PALETTE)

# %% [markdown]
# ## Data dan model (dilatih cepat, bukan fokus notebook ini)
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris LightGBM.
# Grafik di bawah memakai `predictions` dari model ini.

# %%
train = pd.read_csv(TRAIN_CSV)
X = train.drop(columns=[ID_COL, TARGET])
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=SEED)

model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)
# model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED)
# model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)
predictions = model.predict(X_val)
residuals = y_val - predictions   # residual = aktual - prediksi
print("MAE:", round(mean_absolute_error(y_val, predictions), 3))
print("RMSE:", round(root_mean_squared_error(y_val, predictions), 3))
print("R2:", round(r2_score(y_val, predictions), 3))

# %% [markdown]
# ## 1. Perbandingan model vs baseline
# **Kapan dipakai:** slide pertama hasil, menunjukkan model benar-benar lebih baik dari tebakan
# paling sederhana. **Cara baca:** makin pendek bar makin baik. Bar abu = selalu menebak rata-rata.

# %%
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
linear = make_pipeline(StandardScaler(), LinearRegression())
linear.fit(X_train, y_train)

mae = pd.Series(dtype=float)
mae["rata-rata (baseline)"] = mean_absolute_error(y_val, baseline.predict(X_val))
mae["regresi linear"] = mean_absolute_error(y_val, linear.predict(X_val))
mae["model"] = mean_absolute_error(y_val, predictions)
mae.plot.barh(color=[GREY, ORANGE, BLUE])
for i, value in enumerate(mae):
    plt.text(value, i, f" {value:.2f}", va="center")
plt.title("Kesalahan rata-rata (MAE) per model")
plt.xlabel("MAE (satuan 100 ribu USD, makin kecil makin baik)")
plt.savefig("evalreg_perbandingan_model.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Prediksi vs aktual
# **Kapan dipakai:** gambaran paling cepat tentang kualitas model regresi.
# **Cara baca:** titik di garis diagonal = tebakan sempurna. Titik menyebar melebar ke kanan =
# error makin besar untuk nilai tinggi. Pita datar di atas = nilai target terpotong (clipping).

# %%
plt.scatter(y_val, predictions, alpha=0.3, s=15, color=BLUE)
line_range = [0, y_val.max()]
plt.plot(line_range, line_range, color=RED, linewidth=2, linestyle="--", label="sempurna")
plt.title(f"Prediksi vs aktual (R2 = {r2_score(y_val, predictions):.2f})")
plt.xlabel("nilai aktual")
plt.ylabel("nilai prediksi")
plt.legend()
# Alternatif data besar: plt.hexbin(y_val, predictions, gridsize=40, bins="log", cmap="Blues")
plt.savefig("evalreg_prediksi_vs_aktual.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Residual vs prediksi
# **Kapan dipakai:** mendeteksi pola kesalahan. Residual yang baik tampak seperti awan acak
# di sekitar nol. **Cara baca:** bentuk corong = error membesar di nilai besar (coba log-target),
# lengkungan = model belum menangkap hubungan non-linear.

# %%
sns.regplot(x=predictions, y=residuals, lowess=True,
            scatter_kws={"alpha": 0.25, "s": 15, "color": BLUE},
            line_kws={"color": ORANGE, "linewidth": 3})
plt.axhline(0, color="black", linewidth=1)
plt.title("Residual vs prediksi")
plt.xlabel("nilai prediksi")
plt.ylabel("residual (aktual - prediksi)")
plt.savefig("evalreg_residual_vs_prediksi.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Residual per fitur
# **Kapan dipakai:** mencari fitur yang menyimpan informasi belum terpakai atau butuh fitur baru.
# **Cara baca:** garis oranye = rata-rata residual per rentang fitur, harus datar di nol. Bagian
# yang menyimpang = model bias di rentang itu.

# %%
feature_bins = pd.qcut(X_val[CHECK_FEATURE], 15)
mean_residual = residuals.groupby(feature_bins, observed=True).mean()
bin_middle = X_val[CHECK_FEATURE].groupby(feature_bins, observed=True).mean()
plt.scatter(X_val[CHECK_FEATURE], residuals, alpha=0.2, s=12, color=GREY)
plt.plot(bin_middle, mean_residual, color=ORANGE, linewidth=3, marker="o")
plt.axhline(0, color="black", linewidth=1)
plt.title(f"Residual menurut {CHECK_FEATURE}")
plt.xlabel(CHECK_FEATURE)
plt.ylabel("residual")
plt.savefig("evalreg_residual_per_fitur.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Residual per grup
# **Kapan dipakai:** memeriksa keadilan/konsistensi model antar kelompok (wilayah, jenis, usia).
# **Cara baca:** semua kotak sebaiknya berpusat di nol dengan tinggi mirip. Grup yang bergeser
# atau lebih lebar = model lemah di grup itu.

# %%
groups = pd.qcut(X_val[GROUP_FEATURE], 4, labels=["baru", "agak baru", "agak tua", "tua"])
sns.boxplot(x=groups, y=residuals, color=BLUE, fliersize=2)
plt.axhline(0, color="black", linewidth=1)
plt.title("Residual per kelompok umur rumah")
plt.xlabel("umur rumah (kuartil)")
plt.ylabel("residual")
plt.savefig("evalreg_residual_per_grup.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Distribusi error + QQ plot
# **Kapan dipakai:** memeriksa apakah error simetris dan tak berekor panjang (penting untuk
# interval kepercayaan). **Cara baca:** kiri = histogram residual (puncak di nol = baik).
# Kanan = QQ plot: titik di garis lurus = error mendekati normal, ujung melengkung = ekor tebal.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(residuals, bins=50, kde=True, color=BLUE, ax=axes[0])
axes[0].axvline(0, color="black", linewidth=1)
axes[0].set_title("Distribusi residual")
axes[0].set_xlabel("residual")
stats.probplot(residuals, dist="norm", plot=axes[1])
qq_lines = axes[1].get_lines()
qq_lines[0].set_color(BLUE)
qq_lines[1].set_color(RED)
axes[1].set_title("QQ plot residual")
plt.savefig("evalreg_distribusi_error.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Error per kuantil nilai aktual
# **Kapan dipakai:** menjawab "model akurat di rentang harga mana?" (murah/menengah/mahal).
# **Cara baca:** tinggi bar = MAE di kuantil itu. Bar yang melonjak di kuantil teratas = model
# kesulitan memprediksi nilai ekstrem.

# %%
quantile_names = ["20% termurah", "20-40%", "40-60%", "60-80%", "20% termahal"]
quantiles = pd.qcut(y_val, 5, labels=quantile_names)
absolute_residuals = residuals.abs()
mae_per_quantile = absolute_residuals.groupby(quantiles, observed=True).mean()
mae_per_quantile.plot.bar(color=BLUE, rot=15)
plt.axhline(mae["model"], color=RED, linestyle="--", label="MAE keseluruhan")
plt.title("Kesalahan per kuantil nilai aktual")
plt.xlabel("")
plt.ylabel("MAE")
plt.legend()
plt.savefig("evalreg_error_per_kuantil.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Peta error (data punya lat/lon)
# **Kapan dipakai:** data lokasi, mencari wilayah yang salah diprediksi (fitur lokasi kurang).
# **Cara baca:** merah = model menebak terlalu rendah, biru = terlalu tinggi, putih = pas.
# Gerombol warna sejenis = error berkorelasi spasial.

# %%
plt.figure(figsize=(7, 6))
color_limit = absolute_residuals.quantile(0.95)
points = plt.scatter(X_val["Longitude"], X_val["Latitude"], c=residuals, cmap="RdBu_r", s=12,
                     vmin=-color_limit, vmax=color_limit)
plt.colorbar(points, label="residual (aktual - prediksi)")
plt.title("Peta residual California")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig("evalreg_peta_error.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Prediksi vs aktual interaktif (plotly)
# **Kapan dipakai:** menelusuri titik yang meleset jauh. **Cara baca:** arahkan kursor ke titik
# untuk melihat nilai dan fiturnya, zoom untuk memeriksa area padat.

# %%
plot_table = X_val.copy()
plot_table["aktual"] = y_val
plot_table["prediksi"] = predictions
plot_table["selisih"] = residuals
fig = px.scatter(plot_table, x="aktual", y="prediksi", color="selisih", opacity=0.6,
                 color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                 hover_data=["MedInc", "HouseAge", "Latitude", "Longitude"],
                 title="Prediksi vs aktual (interaktif)")
fig.write_html("evalreg_interaktif.html", include_plotlyjs=True)
fig.show()
