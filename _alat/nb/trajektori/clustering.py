# %% [markdown]
# # Trajektori — Tempat yang Sering Dikunjungi (DBSCAN jarak haversine)
# **Kapan dipakai:** ada jejak GPS satu orang/kendaraan dan pertanyaan "di mana saja dia sering
# berada" (rumah, kantor, halte, pangkalan).
#
# **Idenya:** ambil titik saat **diam** (kecepatan rendah), lalu DBSCAN mengelompokkan titik yang
# berdekatan. Jarak = haversine (jarak di permukaan bumi). Untuk DBSCAN haversine, koordinat
# harus dalam **radian** dan `eps` = jarak km / 6371. Titik noise (label -1) = bukan tempat.
#
# **Langkah:**
# 1. Hitung kecepatan antar titik, saring titik diam.
# 2. Pilih `EPS_KM` dari grafik jarak tetangga, jalankan DBSCAN.
# 3. Ringkas tiap tempat (pusat, jumlah titik, jumlah hari), simpan, gambar.
#
# **Data demo:** satu pengguna GeoLife (Beijing, 2008–2009), 1 titik per 10 detik.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/trajektori/geolife_mini"   # GANTI saat lomba
POINTS_CSV = f"{DATA_FOLDER}/titik_user_000.csv"

TIME_COL = "waktu"
LAT_COL = "lat"
LON_COL = "lon"
EARTH_RADIUS_KM = 6371
EPS_KM = 0.1              # jarak antar titik yang masih satu tempat (100 m)
MIN_SAMPLES = 30          # minimal titik per tempat (30 titik x 10 detik = 5 menit)
SECONDS_PER_POINT = 10    # jarak waktu antar titik, untuk menghitung lama di tempat
STILL_SPEED = 1.0         # titik dengan kecepatan di bawah ini (m/s) dianggap diam
ZOOM = 0.15               # setengah lebar peta (derajat) di sekitar tempat terpadat

# %% [markdown]
# ## Baca data

# %%
points = pd.read_csv(POINTS_CSV, parse_dates=[TIME_COL])
points = points.sort_values(TIME_COL)
print(points.shape)
print("dari", points[TIME_COL].min(), "sampai", points[TIME_COL].max())
points.head()

# %% [markdown]
# ## Langkah 1 — kecepatan antar titik, saring titik diam
# Rumus haversine: jarak permukaan bumi antara dua titik (lat, lon dalam derajat).

# %%
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    inner = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(inner))


# coba: Monas ke Bundaran HI, sekitar 2 km
print(round(haversine_km(-6.1754, 106.8272, -6.1950, 106.8231), 2), "km")

distance_m = 1000 * haversine_km(points[LAT_COL].shift(1), points[LON_COL].shift(1),
                                 points[LAT_COL], points[LON_COL])
seconds = points[TIME_COL].diff().dt.total_seconds()
points["speed"] = distance_m / seconds
still = points[points["speed"] < STILL_SPEED].copy()
print("titik diam:", len(still), "dari", len(points))

# %% [markdown]
# ## Langkah 2a — patokan memilih `EPS_KM`
# Jarak tiap titik ke tetangga ke-`MIN_SAMPLES`, diurutkan. Lutut kurva = eps yang masuk akal.

# %%
coordinates = np.radians(still[[LAT_COL, LON_COL]].to_numpy())
neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES, metric="haversine")
neighbors.fit(coordinates)
distances, _ = neighbors.kneighbors(coordinates)
farthest_km = distances[:, -1] * EARTH_RADIUS_KM
plt.plot(np.sort(farthest_km))
plt.axhline(EPS_KM, color="red", linestyle="--")
plt.ylim(0, 1)
plt.ylabel(f"jarak ke tetangga ke-{MIN_SAMPLES} (km)")
plt.show()

# %% [markdown]
# ## Langkah 2b — DBSCAN

# %%
model = DBSCAN(eps=EPS_KM / EARTH_RADIUS_KM, min_samples=MIN_SAMPLES, metric="haversine",
               algorithm="ball_tree")
still["place"] = model.fit_predict(coordinates)
print("jumlah tempat:", still["place"].max() + 1)
print("titik noise:", (still["place"] == -1).sum())

# %% [markdown]
# ## Langkah 3 — ringkasan per tempat

# %%
still["day"] = still[TIME_COL].dt.date
still["hour"] = still[TIME_COL].dt.hour
places = still[still["place"] >= 0]
per_place = places.groupby("place")

summary = pd.DataFrame()
summary["n_points"] = per_place[LAT_COL].count()
summary["center_lat"] = per_place[LAT_COL].mean()
summary["center_lon"] = per_place[LON_COL].mean()
summary["n_days"] = per_place["day"].nunique()
summary["median_hour"] = per_place["hour"].median()
summary["total_minutes"] = summary["n_points"] * SECONDS_PER_POINT / 60
summary = summary.sort_values("n_points", ascending=False)

submission = pd.DataFrame()
submission["place"] = summary.index
submission["center_lat"] = summary["center_lat"].to_numpy()
submission["center_lon"] = summary["center_lon"].to_numpy()
submission["n_days"] = summary["n_days"].to_numpy()
submission["total_minutes"] = summary["total_minutes"].to_numpy()
submission.to_csv("submission.csv", index=False)
submission.head(10)

# %% [markdown]
# ## Gambar: semua titik (abu-abu) dan tempat sering dikunjungi (berwarna)
# Peta difokuskan di sekitar tempat terpadat (perjalanan jauh membuat peta terlalu lebar).

# %%
busiest = summary.iloc[0]
plt.figure(figsize=(8, 7))
plt.scatter(points[LON_COL], points[LAT_COL], s=1, color="lightgray")
plt.scatter(places[LON_COL], places[LAT_COL], s=4, c=places["place"], cmap="tab20")
top_places = summary.head(5)
for number, lat, lon in zip(top_places.index, top_places["center_lat"], top_places["center_lon"]):
    plt.annotate(f"#{number}", (lon, lat), fontsize=12, color="red")
plt.xlim(busiest["center_lon"] - ZOOM, busiest["center_lon"] + ZOOM)
plt.ylim(busiest["center_lat"] - ZOOM, busiest["center_lat"] + ZOOM)
plt.xlabel("lon")
plt.ylabel("lat")
plt.show()
