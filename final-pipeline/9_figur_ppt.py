"""Figur gaya paper untuk PPT, dibuat HANYA dari CSV hasil 1_eda, 5_ensemble, 6_topik, 7_laporan_model.

Pakai:  python 9_figur_ppt.py                       -> membaca HASIL di konfig.py
        python 9_figur_ppt.py ../../outputs/bawa_pulang   -> membaca folder lain (mis. di laptop)
Hasil:  final-pipeline/figur_ppt/*.png (tempel ke PPT) dan *.svg (seret ke Figma, teks tetap bisa diedit).
Figur yang CSV-nya tidak ada dilewati. Tidak ada angka yang diketik tangan di file ini.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, ScalarFormatter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import konfig

# ---------------- Konfigurasi ----------------
SUMBER = konfig.HASIL
if len(sys.argv) > 1:
    SUMBER = sys.argv[1]
KELUAR = "figur_ppt"          # di dalam repo (final-pipeline/figur_ppt), tidak ikut git
UKURAN = (13, 5.4)            # UBAH: inci, sama dengan lebar slide (lihat _memori/aturan.md bagian Figur)
JUMLAH_KATA = 8               # UBAH: kata per topik di figur kata teratas
BATAS_MIRIP = 0.10            # UBAH: garis antar topik digambar bila kemiripan >= ini
PASANGAN_MAKS = 6             # UBAH: pasangan kelas tertukar yang ditampilkan

TINTA, TINTA2, REDUP, GARIS = "#0b0b0b", "#52514e", "#898781", "#d5d4cc"
# urutan tetap, lolos cek buta warna (validate_palette.js); lebih dari 8 seri = abu-abu
PALET = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BIRU, JINGGA, HIJAU = PALET[0], PALET[1], PALET[2]
BENTUK = ["o", "s", "D", "^", "v", "P", "X", "*"]
PETA_BIRU = LinearSegmentedColormap.from_list("biru", ["#ffffff", "#cde2fb", "#2a78d6", "#0d366b"])

for berkas in sorted(os.listdir("_alat/fonts")):
    if berkas.endswith(".ttf"):
        font_manager.fontManager.addfont("_alat/fonts/" + berkas)
plt.rcParams.update({
    "font.family": ["Be Vietnam Pro", "DejaVu Sans"], "font.size": 15, "svg.fonttype": "none",
    "text.color": TINTA, "axes.labelcolor": TINTA2, "axes.labelsize": 15,
    "axes.edgecolor": GARIS, "axes.linewidth": 1.0, "axes.spines.top": False,
    "axes.spines.right": False, "axes.axisbelow": True, "axes.grid": True,
    "grid.color": GARIS, "grid.linewidth": 0.8, "grid.linestyle": (0, (4, 4)),
    "xtick.color": TINTA2, "ytick.color": TINTA2, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
    "xtick.labelsize": 14, "ytick.labelsize": 15, "legend.frameon": False, "legend.fontsize": 14,
    "lines.linewidth": 2.4, "lines.markersize": 10,
})


# ---------------- Alat bersama ----------------
def baca(sub, nama):
    path = os.path.join(SUMBER, sub, nama)
    if not os.path.exists(path):
        print("  tidak ada:", path)
        return None
    return pd.read_csv(path)


def koma(x, desimal=3):
    return f"{x:.{desimal}f}".replace(".", ",")


def persen(x):
    return koma(100 * x, 1) + "%"


def kanvas(kolom, lebar=None, baris=1):
    fig, axes = plt.subplots(baris, kolom, figsize=UKURAN, width_ratios=lebar, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.3, h_pad=0.22, wspace=0.05, hspace=0.06)
    return fig, axes


def panel(ax, huruf, judul):
    ax.set_title(f"({huruf})  {judul}", loc="left", fontsize=18, fontweight="bold", pad=14)


def angka_sumbu(x, posisi):
    return f"{x:g}".replace(".", ",")


def simpan(fig, nama):
    os.makedirs(KELUAR, exist_ok=True)
    for ax in fig.axes:                      # angka sumbu ikut koma desimal seperti label
        for sumbu in [ax.xaxis, ax.yaxis]:
            if isinstance(sumbu.get_major_formatter(), ScalarFormatter):
                sumbu.set_major_formatter(FuncFormatter(angka_sumbu))
    fig.savefig(f"{KELUAR}/{nama}.png", dpi=200)
    fig.savefig(f"{KELUAR}/{nama}.svg")
    plt.close(fig)
    print("  simpan", nama)


def warna_sorot(n, sorot):
    warna = []
    for i in range(n):
        if i == sorot:
            warna.append(BIRU)
        else:
            warna.append(REDUP)
    return warna


def titik_h(ax, nama, nilai, warna, desimal=3):
    """Dot plot mendatar: sumbu boleh tidak mulai dari 0 karena yang dibaca posisi, bukan panjang."""
    nilai = np.asarray(nilai, dtype=float)
    y = np.arange(len(nilai))
    ax.scatter(nilai, y, s=190, c=warna, zorder=3)
    for i in y:
        ax.annotate(koma(nilai[i], desimal), (nilai[i], i), xytext=(13, 0),
                    textcoords="offset points", va="center", fontsize=14, color=TINTA2)
    ax.set_yticks(y, list(nama))
    rentang = max(nilai.max() - nilai.min(), 1e-6)
    ax.set_xlim(nilai.min() - 0.15 * rentang, nilai.max() + 0.4 * rentang)
    ax.set_ylim(-0.6, len(nilai) - 0.4)
    ax.grid(axis="x", visible=False)


def peta_panas(ax, nilai, teks, label_x):
    """Heatmap satu warna, nilai ditulis di tiap sel, sel dipisah celah putih."""
    baris, kolom = nilai.shape
    ax.imshow(nilai, cmap=PETA_BIRU, vmin=0, vmax=1, aspect="auto")
    ukuran = max(9, 17 - max(baris, kolom) // 2)
    for i in range(baris):
        for j in range(kolom):
            warna = TINTA
            if nilai[i, j] > 0.55:
                warna = "white"
            ax.text(j, i, teks[i][j], ha="center", va="center", fontsize=ukuran, color=warna)
    ax.set_xticks(np.arange(kolom), label_x)
    ax.set_xticks(np.arange(kolom + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(baris + 1) - 0.5, minor=True)
    ax.grid(visible=False)
    ax.grid(which="minor", color="white", linewidth=3, linestyle="-")
    ax.spines[:].set_visible(False)


def kode(nama):
    return str(nama).split(" ")[0]


def warna_topik():
    ukuran = baca("topik", "02_ukuran_topik.csv")
    warna = {}
    if ukuran is None:
        return warna
    for i in range(len(ukuran)):
        warna[ukuran["nama"].iloc[i]] = PALET[i] if i < len(PALET) else REDUP
    return warna


# ---------------- What We See ----------------
def fig_data():
    kelas = baca("laporan", "03_laporan_kelas.csv")
    kolom = baca("eda", "kekuatan_kolom.csv")
    if kelas is None or kolom is None:
        return
    fig, axes = kanvas(2, [1, 1.25])
    panel(axes[0], "a", "Jumlah teks per kelas")
    kelas = kelas.sort_values("jumlah")
    y = np.arange(len(kelas))
    axes[0].barh(y, kelas["jumlah"], color=BIRU, height=0.55)
    total = kelas["jumlah"].sum()
    for i in y:
        n = kelas["jumlah"].iloc[i]
        axes[0].annotate(f"{n}  ({persen(n / total)})", (n, i), xytext=(8, 0),
                         textcoords="offset points", va="center", fontsize=14, color=TINTA2)
    axes[0].set_yticks(y, kelas["kelas"])
    axes[0].margins(x=0.38)
    axes[0].grid(axis="y", visible=False)
    panel(axes[1], "b", "Skor tiap kolom sederhana")
    kolom = kolom.sort_values("skor")
    acak = kolom.loc[kolom["kolom"] == "kolom acak", "skor"]
    warna = []
    for skor in kolom["skor"]:
        warna.append(BIRU if len(acak) > 0 and skor > acak.iloc[0] else REDUP)
    titik_h(axes[1], kolom["kolom"], kolom["skor"], warna)
    if len(acak) > 0:
        axes[1].axvline(acak.iloc[0], color=TINTA2, linewidth=1.4, linestyle=(0, (2, 3)), zorder=1)
    axes[1].set_xlabel("skor (garis = kolom acak)")
    simpan(fig, "01_data")


# ---------------- What We Found: klasifikasi ----------------
def fig_model():
    coba = baca("laporan", "01_percobaan.csv")
    gabung = baca("ensemble", "ablation.csv")
    if coba is None:
        return
    kolom = 2 if gabung is not None else 1
    fig, axes = kanvas(kolom, [1.15, 1][:kolom])
    axes = np.atleast_1d(axes)
    coba = coba.sort_values("skor_cv")
    panel(axes[0], "a", "Skor CV tiap model")
    warna = []
    for skor in coba["skor_cv"]:                 # skor seri = sama-sama disorot
        warna.append(BIRU if skor >= coba["skor_cv"].max() else REDUP)
    titik_h(axes[0], coba["model"], coba["skor_cv"], warna, 4)
    axes[0].set_xlabel(str(coba["metrik"].iloc[0]) + " (OOF)")
    if gabung is not None:
        gabung = gabung[~gabung["langkah"].str.startswith("model ")].iloc[::-1]
        nama = gabung["langkah"].str.replace(" (jujur)", "", regex=False)
        sorot = -1
        for i in range(len(nama)):
            if nama.iloc[i].startswith("dipilih"):
                sorot = i
        panel(axes[1], "b", "Cara gabung model")
        titik_h(axes[1], nama, gabung["skor"], warna_sorot(len(nama), sorot), 4)
        axes[1].set_xlabel(str(coba["metrik"].iloc[0]) + " (OOF)")
    simpan(fig, "02_model")


def fig_kelas():
    cm = baca("laporan", "04_confusion_matrix.csv")
    lap = baca("laporan", "03_laporan_kelas.csv")
    if cm is None or lap is None:
        return
    fig, axes = kanvas(2, [1, 1.05])
    panel(axes[0], "a", "Confusion matrix (OOF)")
    nama = list(cm.iloc[:, 0])
    jumlah = cm.iloc[:, 1:].to_numpy(dtype=float)
    porsi = jumlah / jumlah.sum(axis=1, keepdims=True)
    teks = []
    for i in range(len(nama)):
        baris = []
        for j in range(len(nama)):
            isi = str(int(jumlah[i, j]))
            if len(nama) <= 8:
                isi = isi + "\n" + persen(porsi[i, j])
            baris.append(isi)
        teks.append(baris)
    peta_panas(axes[0], porsi, teks, nama)
    axes[0].set_yticks(np.arange(len(nama)), nama)
    axes[0].set_xlabel("prediksi")
    axes[0].set_ylabel("sebenarnya")
    panel(axes[1], "b", "Precision, recall, F1 per kelas")
    lap = lap.sort_values("f1")
    y = np.arange(len(lap))
    metrik = ["precision", "recall", "f1"]
    axes[1].hlines(y, lap[metrik].min(axis=1), lap[metrik].max(axis=1), color=GARIS, linewidth=5)
    for k in range(3):
        axes[1].scatter(lap[metrik[k]], y, s=170, color=PALET[k], marker=BENTUK[k], zorder=3,
                        label=metrik[k], edgecolors="white", linewidths=1.5)
    label = []
    for i in y:
        label.append(f"{lap['kelas'].iloc[i]}\nn = {lap['jumlah'].iloc[i]}")
    axes[1].set_yticks(y, label)
    axes[1].set_ylim(-0.6, len(lap) - 0.4)
    axes[1].grid(axis="y", visible=False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, handletextpad=0.2)
    simpan(fig, "03_kelas")


def fig_salah():
    tukar = baca("laporan", "05_pasangan_tertukar.csv")
    panjang = baca("laporan", "06_akurasi_panjang_teks.csv")
    if tukar is None or panjang is None:
        return
    fig, axes = kanvas(2, [1.15, 1])
    panel(axes[0], "a", "Pasangan kelas yang paling sering tertukar")
    tukar = tukar.head(PASANGAN_MAKS).iloc[::-1]
    y = np.arange(len(tukar))
    axes[0].barh(y, tukar["a_jadi_b"], color=BIRU, height=0.55, label="kiri diprediksi kanan")
    axes[0].barh(y, tukar["b_jadi_a"], left=tukar["a_jadi_b"], color=JINGGA, height=0.55,
                 label="kanan diprediksi kiri", edgecolor="white", linewidth=2)
    label = []
    for i in y:
        a, b = tukar["a_jadi_b"].iloc[i], tukar["b_jadi_a"].iloc[i]
        label.append(f"{tukar['kelas_a'].iloc[i]}\nvs {tukar['kelas_b'].iloc[i]}")
        axes[0].text(a / 2, i, str(a), ha="center", va="center", color="white", fontweight="bold")
        axes[0].text(a + b / 2, i, str(b), ha="center", va="center", color="white", fontweight="bold")
        axes[0].annotate(str(a + b), (a + b, i), xytext=(8, 0), textcoords="offset points",
                         va="center", color=TINTA2)
    axes[0].set_yticks(y, label)
    axes[0].set_xlabel("jumlah baris salah")
    axes[0].grid(axis="y", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    panel(axes[1], "b", "Skor menurut panjang teks")
    x = np.arange(len(panjang))
    seri = ["akurasi", "f1_makro"]
    for k in range(2):
        axes[1].plot(x, panjang[seri[k]], color=PALET[k], marker=BENTUK[k], label=seri[k],
                     markeredgecolor="white", markeredgewidth=1.5)
        akhir = panjang[seri[k]].iloc[-1]
        axes[1].annotate(koma(akhir), (x[-1], akhir), xytext=(12, 0), textcoords="offset points",
                         va="center", fontsize=14, color=TINTA2)
    label = []
    for i in x:
        label.append(f"{panjang['kata_min'].iloc[i]}-{panjang['kata_maks'].iloc[i]}")
    axes[1].set_xticks(x, label)
    axes[1].set_xlabel("jumlah kata (per kuartil)")
    axes[1].margins(x=0.16, y=0.15)
    axes[1].grid(axis="x", visible=False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    simpan(fig, "04_salah")


def fig_keandalan():
    andal = baca("laporan", "07_kurva_keandalan.csv")
    if andal is None:
        return
    fig, axes = kanvas(2, [1, 1.1])
    panel(axes[0], "a", "Keyakinan model vs akurasi")
    axes[0].plot([0, 1], [0, 1], color=REDUP, linewidth=1.4, linestyle=(0, (2, 3)), label="kalibrasi sempurna")
    axes[0].fill_between(andal["keyakinan"], andal["keyakinan"], andal["akurasi"], color=JINGGA, alpha=0.18)
    axes[0].plot(andal["keyakinan"], andal["akurasi"], color=BIRU, marker="o", label="model",
                 markeredgecolor="white", markeredgewidth=1.5)
    ece = (andal["jumlah"] * (andal["akurasi"] - andal["keyakinan"]).abs()).sum() / andal["jumlah"].sum()
    axes[0].text(0.97, 0.06, "ECE " + koma(ece), transform=axes[0].transAxes, ha="right",
                 fontsize=18, fontweight="bold")
    bawah = min(andal["keyakinan"].min(), andal["akurasi"].min()) - 0.05
    axes[0].set_xlim(bawah, 1.02)
    axes[0].set_ylim(bawah, 1.02)
    axes[0].set_xlabel("keyakinan (probabilitas kelas terpilih)")
    axes[0].set_ylabel("akurasi")
    axes[0].legend(loc="upper left")
    panel(axes[1], "b", "Jumlah prediksi per rentang keyakinan")
    x = np.arange(len(andal))
    axes[1].bar(x, andal["jumlah"], color=BIRU, width=0.62)
    total = andal["jumlah"].sum()
    for i in x:
        n = andal["jumlah"].iloc[i]
        axes[1].annotate(f"{n}\n{persen(n / total)}", (i, n), xytext=(0, 6), textcoords="offset points",
                         ha="center", va="bottom", fontsize=13, color=TINTA2)
    axes[1].set_xticks(x, andal["bin"].str.replace(".", ",", regex=False))
    axes[1].set_xlabel("keyakinan")
    axes[1].margins(y=0.2)
    axes[1].grid(axis="x", visible=False)
    simpan(fig, "05_keandalan")


def fig_sebaran():
    sebar = baca("laporan", "10_sebaran_prediksi_test.csv")
    if sebar is None:
        return
    fig, ax = kanvas(1)
    panel(ax, "a", "Porsi kelas: label train vs prediksi")
    seri = list(sebar.columns[1:])
    x = np.arange(len(sebar))
    lebar = 0.8 / len(seri)
    for k in range(len(seri)):
        posisi = x + (k - (len(seri) - 1) / 2) * lebar
        ax.bar(posisi, sebar[seri[k]], width=lebar - 0.03, color=PALET[k],
               label=seri[k].replace("porsi_", "").replace("_", " "))
        if len(sebar) <= 6:
            for i in x:
                ax.annotate(persen(sebar[seri[k]].iloc[i]), (posisi[i], sebar[seri[k]].iloc[i]),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=13, color=TINTA2)
    ax.set_xticks(x, sebar["kelas"])
    ax.set_ylabel("porsi baris")
    ax.margins(y=0.15)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(seri))
    simpan(fig, "06_sebaran_test")


# ---------------- What We Found: topic modeling ----------------
def fig_topik_pilih():
    pilih = baca("topik", "01_pilih_metode_dan_jumlah_topik.csv")
    info = baca("topik", "info_topik.csv")
    if pilih is None:
        return
    fig, axes = kanvas(3)
    ukur = ["npmi", "keragaman", "kestabilan"]
    judul = ["Koherensi (NPMI)", "Keragaman topik", "Kestabilan"]
    metode = list(pilih["metode"].unique())
    # shortcut: metode terpilih = NPMI tertinggi pada K yang dipakai info_topik.csv; ganti bila notebook memilih lain
    dipakai = None
    if info is not None:
        calon = pilih[pilih["jumlah_topik"] == len(info)]
        if len(calon) > 0:
            dipakai = calon.sort_values("npmi").iloc[-1]
    for p in range(3):
        panel(axes[p], "abc"[p], judul[p])
        for k in range(len(metode)):
            bagian = pilih[pilih["metode"] == metode[k]]
            axes[p].plot(bagian["jumlah_topik"], bagian[ukur[p]], color=PALET[k % 8], marker=BENTUK[k % 8],
                         label=metode[k], markeredgecolor="white", markeredgewidth=1.5)
        if dipakai is not None:
            axes[p].scatter(dipakai["jumlah_topik"], dipakai[ukur[p]], s=520, facecolors="none",
                            edgecolors=TINTA, linewidths=2.2, zorder=4,
                            label=f"dipakai: {dipakai['metode']}, {dipakai['jumlah_topik']} topik")
        axes[p].set_xticks(sorted(pilih["jumlah_topik"].unique()))
        axes[p].set_xlabel("jumlah topik")
        axes[p].grid(axis="x", visible=False)
    gagang, nama = axes[0].get_legend_handles_labels()
    fig.legend(gagang, nama, loc="outside lower center", ncol=len(nama), markerscale=0.6)
    simpan(fig, "07_topik_pilih")


def fig_topik_kata():
    kata = baca("topik", "03_kata_teratas.csv")
    ukuran = baca("topik", "02_ukuran_topik.csv")
    if kata is None or ukuran is None:
        return
    ukuran = ukuran.head(8)          # paling banyak 8 topik per figur (batas palet)
    warna = warna_topik()
    n = len(ukuran)
    baris = 1 if n <= 4 else 2
    kolom = math.ceil(n / baris)
    fig, axes = kanvas(kolom, None, baris)
    axes = np.atleast_1d(axes).ravel()
    for a in range(len(axes)):
        axes[a].set_axis_off()
        if a >= n:
            continue
        nama = ukuran["nama"].iloc[a]
        bagian = kata[kata["topik"] == nama].sort_values("urutan").head(JUMLAH_KATA)
        y = np.arange(len(bagian))
        axes[a].barh(y, bagian["bobot"], color=warna[nama], alpha=0.32, height=0.82)
        axes[a].barh(y, bagian["bobot"].max() * 0.012, color=warna[nama], height=0.82)
        for i in y:
            axes[a].text(bagian["bobot"].max() * 0.04, i, bagian["kata"].iloc[i], va="center",
                         fontsize=15 if baris == 1 else 13)
        axes[a].set_ylim(len(bagian) - 0.4, -0.6)
        axes[a].set_title(f"{kode(nama)}  ·  {persen(ukuran['porsi'].iloc[a])} teks", loc="left",
                          fontsize=17, fontweight="bold", pad=8)
    simpan(fig, "08_topik_kata")


def fig_topik_kelas():
    silang = baca("topik", "06_topik_x_kelas.csv")
    ukuran = baca("topik", "02_ukuran_topik.csv")
    if silang is None or ukuran is None:
        return
    warna = warna_topik()
    ukuran = ukuran.sort_values("porsi", ascending=False)
    silang = silang.set_index("topik").reindex(ukuran["nama"])
    y = np.arange(len(ukuran))
    fig, axes = kanvas(2, [1, 1.15])
    panel(axes[0], "a", "Porsi teks per topik")
    daftar = []
    for nama in ukuran["nama"]:
        daftar.append(warna[nama])
    axes[0].barh(y, ukuran["porsi"], color=daftar, height=0.6)
    for i in y:
        axes[0].annotate(persen(ukuran["porsi"].iloc[i]), (ukuran["porsi"].iloc[i], i), xytext=(8, 0),
                         textcoords="offset points", va="center", fontsize=14, color=TINTA2)
    axes[0].set_yticks(y, ukuran["nama"])
    axes[0].set_ylim(len(y) - 0.5, -0.5)
    axes[0].margins(x=0.22)
    axes[0].set_xticks([])
    axes[0].grid(visible=False)
    panel(axes[1], "b", "Isi kelas di tiap topik")
    nilai = silang.to_numpy(dtype=float)
    teks = []
    for i in range(nilai.shape[0]):
        baris = []
        for j in range(nilai.shape[1]):
            baris.append(koma(100 * nilai[i, j], 0) + "%")
        teks.append(baris)
    peta_panas(axes[1], nilai, teks, list(silang.columns))
    kode_topik = []
    for nama in ukuran["nama"]:
        kode_topik.append(kode(nama))
    axes[1].set_yticks(y, kode_topik)
    simpan(fig, "09_topik_kelas")


def geser_label(peta, nama):
    """Label gelembung yang hampir berimpit digeser menjauhi tetangga terdekatnya."""
    lebar = max(peta["x"].max() - peta["x"].min(), 1e-9)
    tinggi = max(peta["y"].max() - peta["y"].min(), 1e-9)
    dx = (peta.loc[nama, "x"] - peta["x"]) / lebar
    dy = (peta.loc[nama, "y"] - peta["y"]) / tinggi
    jarak = np.sqrt(dx ** 2 + dy ** 2)
    jarak[nama] = np.inf
    dekat = jarak.idxmin()
    if jarak[dekat] > 0.15:
        return 0.0, 0.0
    arah = max(jarak[dekat], 1e-9)
    return 0.09 * lebar * dx[dekat] / arah, 0.09 * tinggi * dy[dekat] / arah


def fig_topik_peta():
    peta = baca("topik", "09_peta_dokumen.csv")
    mirip = baca("topik", "05_kemiripan_topik.csv")
    banding = baca("topik", "07_topik_train_vs_test.csv")
    if peta is None or banding is None:
        return
    warna = warna_topik()
    fig, axes = kanvas(2, [1, 1.2])
    panel(axes[0], "a", "Peta topik (ukuran = jumlah teks)")
    peta = peta.set_index("topik")
    if mirip is not None:
        for _, baris in mirip[mirip["kemiripan"] >= BATAS_MIRIP].iterrows():
            a, b = peta.loc[baris["topik_a"]], peta.loc[baris["topik_b"]]
            axes[0].plot([a["x"], b["x"]], [a["y"], b["y"]], color=REDUP, linewidth=30 * baris["kemiripan"],
                         alpha=0.45, solid_capstyle="round", zorder=1)
    for nama in peta.index:
        axes[0].scatter(peta.loc[nama, "x"], peta.loc[nama, "y"], s=500 + 5500 * peta.loc[nama, "titik"]
                        / peta["titik"].max(), color=warna[nama], edgecolors="white", linewidths=2.5,
                        alpha=0.9, zorder=2)
        gx, gy = geser_label(peta, nama)
        tulisan = axes[0].text(peta.loc[nama, "x"] + gx, peta.loc[nama, "y"] + gy, kode(nama), ha="center",
                               va="center", fontsize=16, fontweight="bold", zorder=3)
        tulisan.set_path_effects([patheffects.withStroke(linewidth=3.5, foreground="white")])
    axes[0].margins(0.28)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(visible=False)
    axes[0].spines[:].set_visible(True)
    if mirip is not None:
        axes[0].set_xlabel(f"garis = kemiripan kata >= {koma(BATAS_MIRIP, 2)}")
    panel(axes[1], "b", "Porsi topik di train dan test")
    banding = banding.sort_values("train")
    y = np.arange(len(banding))
    axes[1].hlines(y, banding[["train", "test"]].min(axis=1), banding[["train", "test"]].max(axis=1),
                   color=GARIS, linewidth=5)
    for i in y:
        c = warna[banding["topik"].iloc[i]]
        axes[1].scatter(banding["train"].iloc[i], i, s=190, color=c, marker="o", zorder=3)
        axes[1].scatter(banding["test"].iloc[i], i, s=190, color="white", edgecolors=c, linewidths=2.6,
                        marker="s", zorder=3)
    axes[1].scatter([], [], s=150, color=TINTA2, marker="o", label="train")
    axes[1].scatter([], [], s=150, color="white", edgecolors=TINTA2, linewidths=2.6, marker="s", label="test")
    axes[1].set_yticks(y, banding["topik"])
    axes[1].set_ylim(-0.6, len(y) - 0.4)
    axes[1].set_xlabel("porsi teks")
    axes[1].grid(axis="y", visible=False)
    axes[1].legend(loc="lower right")
    simpan(fig, "10_topik_peta")


print("Membaca", os.path.abspath(SUMBER))
fig_data()
fig_model()
fig_kelas()
fig_salah()
fig_keandalan()
fig_sebaran()
fig_topik_pilih()
fig_topik_kata()
fig_topik_kelas()
fig_topik_peta()
print("Selesai:", os.path.abspath(KELUAR))
