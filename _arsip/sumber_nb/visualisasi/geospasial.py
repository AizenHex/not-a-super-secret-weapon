# %% [markdown]
# # Visualisasi — Geospasial (peta titik, kepadatan, hotspot, choropleth)
# **Kapan dipakai:** "di mana kejadian terkonsentrasi? provinsi mana yang paling banyak? di mana
# hotspot-nya?"
#
# Semua peta digambar **tanpa geopandas/folium dan tanpa tile internet**: batas provinsi dari
# GeoJSON lokal digambar dengan matplotlib (PolyCollection) dan plotly.
#
# **Isi:** titik kota berlabel, sebaran titik, hexbin kepadatan, grid persegi, KDE hotspot
# (haversine), choropleth provinsi statis dan interaktif. Semua gambar disimpan di folder notebook.
#
# **Data demo:** gempa M>=4.5 Indonesia 2015-2025 (USGS), 34 kota, batas provinsi
# (geoBoundaries). Sumber dan lisensi: `data_latihan/SOURCES_geo.md`.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import shapely
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import PowerNorm
from shapely.geometry import shape
from sklearn.neighbors import KernelDensity

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan"   # GANTI saat lomba
PROVINCE_GEOJSON = f"{DATA_FOLDER}/geo/indonesia_provinsi.geojson"
EARTHQUAKE_CSV = f"{DATA_FOLDER}/geo/gempa_indonesia_2015_2025.csv"
CITY_CSV = f"{DATA_FOLDER}/graph/kota_indonesia.csv"

PROVINCE_KEY = "shapeName"      # nama provinsi di properties GeoJSON
LON_LIMITS = (94, 142)          # kotak peta Indonesia
LAT_LIMITS = (-11.5, 6.5)
BANDWIDTH_KM = 100              # lebar KDE hotspot
EARTH_RADIUS_KM = 6371
LABEL_MIN = 400                 # provinsi dengan gempa >= sekian diberi label di peta

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
# ## Baca data dan batas provinsi
# Tiap provinsi dibaca sebagai poligon shapely. `outlines` = daftar koordinat untuk
# PolyCollection, `outline_owner` = nama provinsi tiap poligon (satu provinsi bisa banyak pulau =
# banyak poligon).

# %%
with open(PROVINCE_GEOJSON, encoding="utf-8") as file:
    geojson = json.load(file)
province_names = []
province_shapes = []
for feature in geojson["features"]:
    province_names.append(feature["properties"][PROVINCE_KEY])
    province_shapes.append(shape(feature["geometry"]))

outlines = []
outline_owner = []
for name, geometry in zip(province_names, province_shapes):
    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    else:
        polygons = list(geometry.geoms)   # MultiPolygon = provinsi dengan banyak pulau
    for polygon in polygons:
        outlines.append(np.array(polygon.exterior.coords))
        outline_owner.append(name)

quakes = pd.read_csv(EARTHQUAKE_CSV)
cities = pd.read_csv(CITY_CSV)
print(len(province_names), "provinsi |", len(outlines), "poligon |", quakes.shape, cities.shape)
quakes.head(3)

# %% [markdown]
# ## 1. Titik lokasi berlabel di atas batas provinsi
# **Kapan dipakai:** data berupa daftar lokasi (kota, cabang, sensor); cek posisi sudah benar.
# **Cara baca:** garis abu = batas provinsi, titik = lokasi. Titik yang jatuh di laut atau di
# negara lain = lat/lon tertukar atau salah tanda (cek kolom lat dan lon).

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.add_collection(PolyCollection(outlines, facecolor="#F1F3F4", edgecolor=GREY, linewidth=0.6))
ax.scatter(cities["lon"], cities["lat"], color=RED, s=25, zorder=3)
for _, row in cities.iterrows():
    ax.text(row["lon"] + 0.3, row["lat"] + 0.2, row["kota"], fontsize=8)
ax.set_xlim(LON_LIMITS)
ax.set_ylim(LAT_LIMITS)
ax.set_aspect("equal")
ax.set_title("34 kota di Indonesia")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
plt.savefig("geo_titik_kota.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Sebaran titik (ukuran dan warna = atribut)
# **Kapan dipakai:** titik kejadian dengan atribut (magnitudo, harga, jumlah).
# **Cara baca:** ukuran lingkaran = magnitudo, warna = kedalaman (gelap = dangkal). Alur titik
# yang memanjang = patahan/zona subduksi. Gunakan alpha kecil karena banyak titik menumpuk.

# %%
point_sizes = (quakes["mag"] - 4) ** 3 * 8     # magnitudo besar = lingkaran jauh lebih besar
fig, ax = plt.subplots(figsize=(12, 5))
ax.add_collection(PolyCollection(outlines, facecolor="#F1F3F4", edgecolor=GREY, linewidth=0.5))
points = ax.scatter(quakes["longitude"], quakes["latitude"], s=point_sizes,
                    c=quakes["depth"], cmap="viridis_r", alpha=0.4, vmax=300)
plt.colorbar(points, label="kedalaman (km)", shrink=0.8)
ax.set_xlim(LON_LIMITS)
ax.set_ylim(LAT_LIMITS)
ax.set_aspect("equal")
ax.set_title("Gempa M>=4.5, 2015-2025")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
plt.savefig("geo_sebaran_titik.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Hexbin kepadatan
# **Kapan dipakai:** ribuan titik menumpuk; ingin tahu daerah terpadat tanpa garis batas kaku.
# **Cara baca:** tiap segi enam diwarnai jumlah titik (skala log); terang = padat.

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.add_collection(PolyCollection(outlines, facecolor="none", edgecolor=GREY, linewidth=0.5))
density = ax.hexbin(quakes["longitude"], quakes["latitude"], gridsize=70, bins="log",
                    mincnt=1, cmap="YlOrRd", extent=LON_LIMITS + LAT_LIMITS)
plt.colorbar(density, label="jumlah gempa (skala log)", shrink=0.8)
ax.set_xlim(LON_LIMITS)
ax.set_ylim(LAT_LIMITS)
ax.set_aspect("equal")
ax.set_title("Kepadatan gempa (hexbin)")
# Alternatif seaborn: sns.histplot(data=quakes, x="longitude", y="latitude", bins=80)
plt.savefig("geo_hexbin.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Grid persegi (sel 1 derajat)
# **Kapan dipakai:** ingin tabel per sel grid (untuk fitur "jumlah kejadian di sekitar") atau
# peta sederhana yang mudah dijelaskan. **Cara baca:** tiap kotak 1x1 derajat (sekitar 111 km);
# kotak kosong tidak digambar.

# %%
lon_edges = np.arange(LON_LIMITS[0], LON_LIMITS[1] + 1, 1)
lat_edges = np.arange(LAT_LIMITS[0], LAT_LIMITS[1] + 1, 1)
fig, ax = plt.subplots(figsize=(12, 5))
ax.add_collection(PolyCollection(outlines, facecolor="none", edgecolor=GREY, linewidth=0.5))
grid_result = ax.hist2d(quakes["longitude"], quakes["latitude"], bins=[lon_edges, lat_edges],
                        cmin=1, cmap="Blues", norm=PowerNorm(0.4))
plt.colorbar(grid_result[3], label="jumlah gempa per sel", shrink=0.8)
ax.set_xlim(LON_LIMITS)
ax.set_ylim(LAT_LIMITS)
ax.set_aspect("equal")
ax.set_title("Jumlah gempa per sel 1 derajat")
# Sel sebagai kolom tabel: quakes["cell_lon"] = quakes["longitude"] // 1
plt.savefig("geo_grid_persegi.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. KDE hotspot (jarak haversine)
# **Kapan dipakai:** menandai hotspot sebagai permukaan halus; jarak dihitung di bola bumi
# (haversine), bukan jarak lurus pada lat/lon. **Cara baca:** kontur makin gelap = kepadatan
# makin tinggi. `BANDWIDTH_KM` kecil = hotspot tajam dan lokal, besar = halus dan melebar.

# %%
coordinates = np.radians(quakes[["latitude", "longitude"]].to_numpy())   # haversine butuh radian
kde = KernelDensity(bandwidth=BANDWIDTH_KM / EARTH_RADIUS_KM, metric="haversine", rtol=1e-3)
kde.fit(coordinates)

lon_values = np.linspace(LON_LIMITS[0], LON_LIMITS[1], 160)
lat_values = np.linspace(LAT_LIMITS[0], LAT_LIMITS[1], 90)
lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
grid_points = np.radians(np.column_stack((lat_grid.ravel(), lon_grid.ravel())))
log_density = kde.score_samples(grid_points)
density = np.exp(log_density)
density = density.reshape(lat_grid.shape)

fig, ax = plt.subplots(figsize=(12, 5))
filled = ax.contourf(lon_grid, lat_grid, density, levels=12, cmap="Reds")
ax.add_collection(PolyCollection(outlines, facecolor="none", edgecolor=GREY, linewidth=0.5))
plt.colorbar(filled, label="kepadatan relatif", shrink=0.8)
ax.set_aspect("equal")
ax.set_title(f"Hotspot gempa (KDE haversine, {BANDWIDTH_KM} km)")
plt.savefig("geo_kde_hotspot.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Hitung jumlah per provinsi (bahan choropleth)
# Titik masuk provinsi mana ditentukan `shapely.contains_xy` (point-in-polygon, tanpa geopandas).

# %%
counts = []
for geometry in province_shapes:
    inside = shapely.contains_xy(geometry, quakes["longitude"], quakes["latitude"])
    counts.append(int(inside.sum()))
per_province = pd.DataFrame()
per_province["province"] = province_names
per_province["quake_count"] = counts
print("titik di dalam provinsi:", sum(counts), "dari", len(quakes), "(sisanya laut/negara lain)")
ranking = per_province.sort_values("quake_count", ascending=False)
ranking.head(5)

# %% [markdown]
# ## 6. Choropleth provinsi (statis)
# **Kapan dipakai:** membandingkan wilayah administratif (jumlah, rata-rata, persentase).
# **Cara baca:** warna poligon = nilai provinsi; abu = nol. Pakai nilai relatif (per kapita atau
# per luas) bila ukuran wilayah sangat berbeda; angka mutlak condong menonjolkan provinsi besar.

# %%
count_of = {}
for name, count in zip(province_names, counts):
    count_of[name] = count
color_scale = PowerNorm(0.4, vmin=0, vmax=max(counts))
colors = []
for name in outline_owner:
    if count_of[name] > 0:
        colors.append(plt.cm.OrRd(color_scale(count_of[name])))
    else:
        colors.append("#E8EAED")   # abu = tidak ada kejadian
fig, ax = plt.subplots(figsize=(12, 5))
ax.add_collection(PolyCollection(outlines, facecolors=colors, edgecolor="white", linewidth=0.5))
for name, geometry in zip(province_names, province_shapes):
    if count_of[name] >= LABEL_MIN:
        center = geometry.representative_point()   # titik yang pasti di dalam provinsi
        ax.text(center.x, center.y, f"{name}\n{count_of[name]}", ha="center", fontsize=8)
legend_scale = ScalarMappable(norm=color_scale, cmap="OrRd")
plt.colorbar(legend_scale, ax=ax, label="jumlah gempa", shrink=0.8)
ax.set_xlim(LON_LIMITS)
ax.set_ylim(LAT_LIMITS)
ax.set_aspect("equal")
ax.set_title("Jumlah gempa per provinsi (peta choropleth)")
plt.savefig("geo_choropleth_statis.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Choropleth interaktif (plotly, tanpa tile internet)
# **Kapan dipakai:** lampiran HTML untuk juri; hover nama dan nilai provinsi.
# **Cara baca:** arahkan kursor ke provinsi; seret untuk geser, scroll untuk zoom. Latar `white-bg`
# sengaja tanpa peta dasar supaya jalan offline.

# %%
fig = px.choropleth_map(per_province, geojson=geojson, locations="province",
                        featureidkey=f"properties.{PROVINCE_KEY}", color="quake_count",
                        color_continuous_scale="OrRd", map_style="white-bg", zoom=3.2,
                        center={"lat": -2.5, "lon": 118}, opacity=0.9, height=500,
                        title="Jumlah gempa per provinsi (interaktif)")
fig.write_html("geo_choropleth_interaktif.html", include_plotlyjs=True)
fig.show()
