"""Figur gaya paper penelitian untuk slide (What We See / What We Did / What We Found).

Pakai:  python 9b_slide.py            -> membaca folder HASIL di bawah (butuh teks + OOF, jadi di PC lomba)
        python 9b_slide.py <folder>   -> folder hasil lain
Hasil:  final-pipeline/slide/*.png + *.svg dan slide/pratinjau.html.
Semua angka dan teks diambil dari hasil pipeline; tidak ada yang diketik tangan. Tidak ada model yang dilatih ulang
(UMAP dan bootstrap hanya untuk menggambar).
"""
import html
import json
import os
import re
import sys
import textwrap
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------- Konfigurasi (samakan dengan sel Konfigurasi notebook) ----------------
HASIL = "../../outputs/final"   # GANTI
ID_COL = "id"                    # GANTI
TEXT_COLS = ["text"]             # GANTI
TARGET = "label"                 # GANTI
SUMBER = sys.argv[1] if len(sys.argv) > 1 else HASIL
KELUAR = "slide"
PRETRAINED = "../../pretrained"     # folder model lokal (hanya tokenizer yang dibaca)
CONTOH_ID = None                    # UBAH: id baris latih untuk contoh berjalan di skema pipeline (None = otomatis)
BOOTSTRAP = 1000                    # UBAH: jumlah resampling untuk sebaran F1
LEBAR, TINGGI = 160, 90             # satuan kanvas; 12 satuan = 1 inci -> 13,33 x 7,5 inci (16:9)

TINTA, ABU, REDUP, GARIS, LATAR = "#141414", "#555555", "#8c8c8c", "#d9d9d9", "#f4f4f1"
SOROT = "#fff1b8"
W_KELAS = {"negative": "#c8453b", "negatif": "#c8453b", "neutral": "#7c8794", "netral": "#7c8794",
           "positive": "#1f8a70", "positif": "#1f8a70"}
W_LAIN = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#7f7f7f", "#8c564b"]
W_MODEL = ["#27408b", "#b07a1e", "#6a3d9a", "#2b8a8a"]   # urut dari F1 tertinggi
PETA_NILA = LinearSegmentedColormap.from_list("nila", ["#ffffff", "#c9d3ee", "#27408b"])

for berkas in sorted(os.listdir("_alat/fonts")):
    if berkas.endswith(".ttf"):
        font_manager.fontManager.addfont("_alat/fonts/" + berkas)
SANS = ["IBM Plex Sans", "Be Vietnam Pro", "DejaVu Sans"]
MONO = ["IBM Plex Mono", "DejaVu Sans Mono"]
plt.rcParams.update({
    "font.family": SANS, "font.size": 11.5, "svg.fonttype": "none", "text.color": TINTA,
    "axes.labelcolor": ABU, "axes.labelsize": 11.5, "axes.edgecolor": TINTA, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "xtick.color": ABU,
    "ytick.color": ABU, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "xtick.major.size": 3,
    "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
})
PUTIH = [patheffects.withStroke(linewidth=3, foreground="white")]
FIGUR = []


# ---------------- Alat ----------------
def baca(sub, nama):
    path = os.path.join(SUMBER, sub, nama)
    if not os.path.exists(path):
        print("  tidak ada:", path)
        return None
    return pd.read_csv(path)


def koma(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def persen(x, d=1):
    return koma(100 * x, d) + "%"


def ribuan(n):
    return f"{int(n):,}".replace(",", ".")


def kanvas():
    fig = plt.figure(figsize=(LEBAR / 12, TINGGI / 12))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, LEBAR)
    ax.set_ylim(0, TINGGI)
    ax.set_axis_off()
    return fig, ax


def sumbu(fig, x, y, w, h):
    return fig.add_axes([x / LEBAR, y / TINGGI, w / LEBAR, h / TINGGI])


def huruf(ax, x, y, h, judul):
    """Huruf panel gaya jurnal: huruf kecil tebal + judul pendek."""
    ax.text(x, y, h, fontsize=16, fontweight="bold", va="baseline")
    ax.text(x + 3.2, y, judul, fontsize=12.5, fontweight="semibold", va="baseline")


def simpan(fig, bagian, nama, ket):
    os.makedirs(KELUAR, exist_ok=True)
    fig.savefig(f"{KELUAR}/{nama}.png", dpi=200)
    fig.savefig(f"{KELUAR}/{nama}.svg")
    plt.close(fig)
    FIGUR.append((bagian, nama, ket))
    print("  simpan", nama)


def warna_kelas(kelas):
    return {k: W_KELAS.get(str(k).lower(), W_LAIN[i % len(W_LAIN)]) for i, k in enumerate(kelas)}


def rapikan(teks):
    return re.sub(r"\s+", " ", str(teks)).strip()


def bungkus(teks, lebar, baris):
    potong = textwrap.wrap(rapikan(teks), lebar)
    if len(potong) > baris:
        potong = potong[:baris]
        potong[-1] = potong[-1][:lebar - 2].rstrip() + " …"
    return potong


def kotak(ax, x, y, w, h, fc="white", ec=TINTA, lw=0.9, r=0.8, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def panah(ax, *titik, warna=ABU, lw=1.1):
    """Garis patah dengan kepala panah di titik terakhir."""
    xs, ys = zip(*titik)
    if len(titik) > 2:
        ax.plot(xs[:-1], ys[:-1], color=warna, lw=lw, solid_capstyle="butt", zorder=1)
    ax.annotate("", titik[-1], titik[-2], arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.5",
                                                          color=warna, lw=lw, shrinkA=0, shrinkB=0), zorder=1)


def nama_pendek(topik):
    """'T0 makanan / makan / bandung' -> ('T0', 'makanan')."""
    kode, _, sisa = str(topik).partition(" ")
    return kode, sisa.split(" / ")[0]


def f1_makro(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    bawah = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(bawah > 0, 2 * np.diag(cm) / np.maximum(bawah, 1), 0)))


# ---------------- Data ----------------
def catatan(model):
    coba = baca("oof", "percobaan.csv")
    if coba is None or model not in set(coba["model"]):
        return ""
    return str(coba[coba["model"] == model]["catatan"].iloc[-1])


def muat():
    """Teks latih + probabilitas OOF tiap model (urutan kelas dicek ke akurasi tercatat) + topik per teks."""
    path = os.path.join(SUMBER, "praproses", "train_bersih.csv")
    skor = baca("laporan", "02_skor_oof.csv")
    if not os.path.exists(path) or skor is None:
        print("  tidak ada teks latih / skor OOF: figur per teks dilewati")
        return None
    df = pd.read_csv(path)
    kol = TEXT_COLS[0]
    df["_teks"] = df[kol].fillna("").astype(str).map(rapikan)
    df["_bersih"] = df[kol + "_tfidf"].fillna("").astype(str) if kol + "_tfidf" in df else df["_teks"].str.lower()
    df["_kata"] = df["_teks"].str.split().str.len()
    kelas = sorted(df[TARGET].unique())
    y = df[TARGET].map({k: i for i, k in enumerate(kelas)}).to_numpy()
    prob = {}
    for m in skor.sort_values("f1_makro", ascending=False)["model"]:
        p_path = os.path.join(SUMBER, "oof", f"oof_{m}.npy")
        if not os.path.exists(p_path):
            continue
        p = np.load(p_path)
        if p.shape != (len(df), len(kelas)):
            continue
        tercatat = skor.loc[skor["model"] == m, "akurasi"].iloc[0]
        if abs((p.argmax(1) == y).mean() - tercatat) > 1e-3:
            print(f"  OOF {m}: akurasi tidak cocok dengan 02_skor_oof, dilewati")
            continue
        prob[m] = p
    if not prob:
        return None
    topik = baca("topik", "topik_dokumen.csv")
    if topik is not None:
        topik = topik[topik["asal"] == "train"].drop_duplicates(ID_COL).set_index(ID_COL)
        df["_topik"] = df[ID_COL].map(topik["nama_topik"])
        df["_kuat"] = df[ID_COL].map(topik["kuat_topik"])
    return {"df": df, "kelas": kelas, "y": y, "prob": prob, "model": list(prob), "wk": warna_kelas(kelas)}


def pilih_contoh(D):
    """Contoh berjalan: pendek, model terbaik benar, model kedua salah (selisih keyakinan terbesar)."""
    df, y = D["df"], D["y"]
    if CONTOH_ID is not None and (df[ID_COL] == CONTOH_ID).any():
        return int(np.where(df[ID_COL] == CONTOH_ID)[0][0])
    p1 = D["prob"][D["model"][0]]
    benar1 = p1[np.arange(len(y)), y]
    calon = df["_kata"].between(7, 12) & (df["_teks"].str.len() <= 78) & (p1.argmax(1) == y)
    nilai = benar1.copy()
    if len(D["model"]) > 1:
        p2 = D["prob"][D["model"][1]]
        calon &= p2.argmax(1) != y
        nilai = benar1 - p2[np.arange(len(y)), y]
    if "_kuat" in df:
        calon &= df["_kuat"].fillna(0) >= 0.5
    if not calon.any():
        calon = df["_kata"].between(5, 14)
    return int(np.where(calon, nilai, -9).argmax())


def token_bert(teks, model):
    nama = model.replace("ft_", "")
    path = os.path.join(PRETRAINED, nama)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        return ["[CLS]"] + tok.tokenize(teks) + ["[SEP]"], "wordpiece"
    except Exception as e:  # tokenizer tidak ada: tampilkan kata, dan sebut "kata"
        print("  tokenizer tidak terbaca:", e)
        return teks.split(), "kata"


# ================= WHAT WE DID: skema pipeline =================
def keping(ax, x, y, lebar, daftar, baris, fc, ec, khusus=()):
    """Deretan keping token (mono), dibungkus dalam `lebar`."""
    cx, cy, n = x, y, 1
    for t in daftar:
        w = 0.82 * len(t) + 1.5
        if cx + w > x + lebar:
            cx, cy, n = x, cy - 4.1, n + 1
            if n > baris:
                ax.text(x + lebar, cy + 4.1 + 1.5, "…", fontsize=10, ha="right", va="center", color=ABU)
                return
        gelap = t in khusus
        kotak(ax, cx, cy, w, 3.2, fc=TINTA if gelap else fc, ec=TINTA if gelap else ec, lw=0.7, r=0.5)
        ax.text(cx + w / 2, cy + 1.55, t, fontsize=8.2, family=MONO, ha="center", va="center",
                color="white" if gelap else TINTA, zorder=3)
        cx += w + 0.7


def batang_prob(ax, x, y, w, kelas, p, wk, benar):
    """Batang p(kelas) satu teks; kelas label asli diberi tanda."""
    for j, k in enumerate(kelas):
        yy = y - j * 3.6
        ax.add_patch(Rectangle((x, yy), w, 2.4, fc=LATAR, ec="none", zorder=2))
        ax.add_patch(Rectangle((x, yy), w * p[j], 2.4, fc=wk[k], ec="none", zorder=3))
        ax.text(x - 0.8, yy + 1.2, str(k)[:3], fontsize=8.5, ha="right", va="center", color=wk[k],
                fontweight="semibold")
        ax.text(x + w + 0.8, yy + 1.2, koma(p[j], 2), fontsize=8.5, family=MONO, va="center",
                fontweight="semibold" if j == int(np.argmax(p)) else "normal")
        if j == benar:
            ax.text(x + w + 6.2, yy + 1.2, "◂ label", fontsize=8, va="center", color=ABU)


def fig_pipeline(D):
    df, kelas, y, wk = D["df"], D["kelas"], D["y"], D["wk"]
    i = pilih_contoh(D)
    skor = baca("laporan", "02_skor_oof.csv").set_index("model")
    abl = baca("ensemble", "ablation.csv")
    kirim = baca("ensemble", "submission.csv")
    info = baca("topik", "02_ukuran_topik.csv")
    fig, ax = kanvas()
    m1 = D["model"][0]
    m2 = D["model"][1] if len(D["model"]) > 1 else None
    c1 = catatan(m1)
    atur = dict(re.findall(r"(?<![/\w])(epoch|lr|len) (\S+)", c1))
    lipat = re.search(r"fold \[([^\]]*)\]", c1)
    n_fold = len(lipat.group(1).split(",")) if lipat else None

    # kepala kolom
    kolom = [(2, "DATA"), (33, "PRAPROSES"), (62, "REPRESENTASI"), (99, "MODEL"), (121, "KEYAKINAN CONTOH"),
             (143, "PILIH · KIRIM")]
    for x, t in kolom:
        ax.text(x, 86.3, t, fontsize=9.5, fontweight="semibold", color=ABU, va="center")
    ax.plot([2, 158], [84.2, 84.2], color=TINTA, lw=1.0)
    ax.plot([2, 158], [27.5, 27.5], color=GARIS, lw=0.9, ls=(0, (3, 3)))
    ax.text(2, 80.6, "klasifikasi (dengan label)", fontsize=10, style="italic", color=ABU, va="center")
    ax.text(2, 24.2, "topic modeling (tanpa label)", fontsize=10, style="italic", color=ABU, va="center")

    # --- DATA: potongan tabel asli, baris contoh disorot
    baris = sorted([i] + [int(np.where(y == j)[0][0]) for j in range(len(kelas)) if j != y[i]][:2])
    ty = 72
    ax.text(2.6, ty + 3.4, f"{ID_COL:<6}{TEXT_COLS[0]:<17}{TARGET}", fontsize=8.6, family=MONO,
            color=ABU, va="center")
    ax.plot([2, 30], [ty + 1.6, ty + 1.6], color=TINTA, lw=0.7)
    for n, r in enumerate(baris):
        yy = ty - 1 - n * 4.4
        if r == i:
            ax.add_patch(Rectangle((2, yy - 2.0), 28, 4.2, fc=SOROT, ec="none", zorder=1))
        t = df["_teks"].iloc[r]
        ax.text(2.6, yy, f"{str(df[ID_COL].iloc[r]):<6}{(t[:14] + '…'):<17}", fontsize=8.6, family=MONO,
                va="center", zorder=3)
        ax.text(23.4, yy, str(df[TARGET].iloc[r])[:8], fontsize=8.6, family=MONO, va="center",
                color=wk[df[TARGET].iloc[r]], fontweight="semibold", zorder=3)
    ybawah = ty - 1 - len(baris) * 4.4
    ax.text(2.6, ybawah, "…", fontsize=9, family=MONO, color=ABU, va="center")
    ax.plot([2, 30], [ybawah - 2, ybawah - 2], color=TINTA, lw=0.7)
    n_uji = len(kirim) if kirim is not None else None
    ax.text(2, ybawah - 5, f"{ribuan(len(df))} baris latih" + (f" · {ribuan(n_uji)} baris uji" if n_uji else ""),
            fontsize=10, fontweight="semibold", va="center")
    ax.text(2, ybawah - 8.4, f"dipakai: {TEXT_COLS[0]} → {TARGET}", fontsize=9.5, color=ABU,
            va="center")
    y_contoh = ty - 1 - baris.index(i) * 4.4

    # --- PRAPROSES: teks contoh apa adanya dan versi bersih
    def kartu_teks(ytop, judul, isi):
        pot = bungkus(isi, 27, 4)
        h = 3.4 * len(pot) + 2.2
        ax.text(33, ytop + 1.6, judul, fontsize=9, color=ABU, va="center")
        kotak(ax, 33, ytop - h, 25, h, fc=SOROT, ec=TINTA, lw=0.8)
        for n, b in enumerate(pot):
            ax.text(34, ytop - 2.7 - n * 3.4, b, fontsize=8.2, family=MONO, va="center", zorder=3)
        return ytop - h / 2, ytop - h

    ya, _ = kartu_teks(74, TEXT_COLS[0] + "  (apa adanya)", df["_teks"].iloc[i])
    yb, yb_bawah = kartu_teks(49, TEXT_COLS[0] + "_tfidf  (dibersihkan)", df["_bersih"].iloc[i])
    panah(ax, (30, y_contoh), (31.5, y_contoh), (31.5, ya), (33, ya))
    panah(ax, (31.5, y_contoh), (31.5, yb), (33, yb))

    # --- REPRESENTASI
    tok, jenis = token_bert(df["_teks"].iloc[i], m1)
    ax.text(62, 77.6, f"{len(tok)} {jenis}" + (f"  ·  max_length {atur['len']}" if "len" in atur else ""),
            fontsize=9, color=ABU, va="center")
    keping(ax, 62, 71.6, 33, tok, 4, "#e9edf8", W_MODEL[0], khusus=("[CLS]", "[SEP]"))
    kata = df["_bersih"].iloc[i].split()
    ax.text(62, 52.6, f"{len(kata)} kata → bobot TF-IDF (kata + frasa)", fontsize=9, color=ABU, va="center")
    keping(ax, 62, 46.6, 33, kata, 3, "#f6ecd9", W_MODEL[1])
    panah(ax, (58, ya), (61.3, ya))
    panah(ax, (58, yb), (61.3, yb))

    # --- MODEL: tumpukan layer encoder, dan pemisah linear
    lapis = None
    try:
        with open(os.path.join(PRETRAINED, m1.replace("ft_", ""), "config.json")) as f:
            lapis = json.load(f).get("num_hidden_layers")
    except Exception:
        pass
    kotak(ax, 99, 58, 17, 20, fc="white", ec=W_MODEL[0], lw=1.2)
    for n in range(lapis or 6):
        hh = 13.0 / (lapis or 6)
        ax.add_patch(Rectangle((101, 59.6 + n * hh), 13, hh * 0.62, fc=W_MODEL[0], alpha=0.22 + 0.5 * n / (lapis or 6),
                               ec="none", zorder=3))
    ax.text(107.5, 75.4, m1.replace("ft_", "").replace("indobert", "IndoBERT"), fontsize=10.5, fontweight="bold",
            ha="center", va="center", color=W_MODEL[0], zorder=4)
    ax.text(107.5, 55.6, (f"{lapis} layer · " if lapis else "") + "fine-tune", fontsize=9, ha="center", color=ABU)
    if atur:
        ax.text(107.5, 52.4, " · ".join(f"{k} {v}" for k, v in atur.items() if k != "len"), fontsize=8.6,
                family=MONO, ha="center", color=ABU)
    panah(ax, (95.5, 68), (98.4, 68))
    if m2:
        kotak(ax, 99, 34, 17, 13, fc="white", ec=W_MODEL[1], lw=1.2)
        rng = np.random.default_rng(0)   # hiasan skematis: titik di dua sisi garis pemisah, bukan data
        for sisi, w in [(-1, REDUP), (1, W_MODEL[1])]:
            px = rng.uniform(101, 114, 7)
            py = 34 + 6.5 + (px - 107.5) * 0.45 + sisi * rng.uniform(1.2, 3.6, 7)
            ok = (py > 35) & (py < 43.4)
            ax.scatter(px[ok], py[ok], s=7, color=w, zorder=3)
        ax.plot([101, 114], [34 + 6.5 - 2.9, 34 + 6.5 + 2.9], color=W_MODEL[1], lw=1.2, zorder=3)
        ax.text(107.5, 45.2, "LinearSVC", fontsize=10.5, fontweight="bold", ha="center", va="center",
                color=W_MODEL[1], zorder=4)
        ax.text(107.5, 31.6, m2, fontsize=8.6, family=MONO, ha="center", color=ABU)
        panah(ax, (95.5, 41), (98.4, 41))
    # bingkai validasi silang
    kotak(ax, 97.4, 29.4, 42.2, 53.2, fc="none", ec=ABU, lw=0.8, ls=(0, (4, 3)), r=1.2, z=1)
    if n_fold:
        for n in range(n_fold):
            ax.add_patch(Rectangle((99 + n * 2.2, 79.6), 1.8, 1.8, fc=TINTA if n == 0 else "white", ec=TINTA, lw=0.7,
                                   zorder=3))
        ax.text(99 + n_fold * 2.2 + 0.8, 80.5, f"{n_fold} fold · prediksi OOF",
                fontsize=8.2, va="center", color=ABU)

    # --- KEYAKINAN contoh (OOF asli)
    batang_prob(ax, 124.5, 71.4, 7, kelas, D["prob"][m1][i], wk, y[i])
    ax.text(121, 59.6, "F1 makro OOF", fontsize=8.6, color=ABU)
    ax.text(121, 55.4, koma(skor.loc[m1, "f1_makro"], 4), fontsize=14, fontweight="bold", color=W_MODEL[0])
    panah(ax, (116, 68), (119.4, 68))
    if m2:
        batang_prob(ax, 124.5, 44.4, 7, kelas, D["prob"][m2][i], wk, y[i])
        ax.text(121, 34.2, "F1 makro OOF", fontsize=8.6, color=ABU)
        ax.text(121, 30.4, koma(skor.loc[m2, "f1_makro"], 4), fontsize=14, fontweight="bold", color=W_MODEL[1])
        panah(ax, (116, 41), (119.4, 41))

    # --- PILIH · KIRIM: cara gabung dari ablation.csv
    yy = 78
    if abl is not None:
        cara = abl[abl["langkah"].str.startswith("cara")].copy()
        dipilih = abl[abl["langkah"].str.startswith("dipilih")]["langkah"]
        nama_pilih = dipilih.iloc[0].split()[1] if len(dipilih) else ""
        ax.text(143, yy, "cara gabung · F1 OOF", fontsize=9, color=ABU, va="center")
        for _, r in cara.iterrows():
            yy -= 4.2
            nm = r["langkah"].split()[1]
            pilih = nm == nama_pilih
            if pilih:
                ax.add_patch(Rectangle((142.4, yy - 1.9), 16, 3.9, fc=SOROT, ec="none", zorder=1))
            ax.text(143, yy, ("● " if pilih else "○ ") + nm, fontsize=9.5, va="center",
                    fontweight="semibold" if pilih else "normal", zorder=3)
            ax.text(158, yy, koma(r["skor"], 4), fontsize=9, family=MONO, va="center", ha="right", zorder=3)
        ax.text(143, yy - 4, f"dipakai: {nama_pilih}", fontsize=9, color=ABU, va="center")
        panah(ax, (139.6, 62), (142, 62))
    if kirim is not None:
        ty2 = 46
        panah(ax, (150, yy - 6.4), (150, ty2 + 6.4))
        ax.text(143, ty2 + 4.2, "submission.csv", fontsize=9.5, fontweight="semibold", va="center")
        ax.text(143.6, ty2, f"{kirim.columns[0]:<8}{kirim.columns[1]}", fontsize=8.6, family=MONO, color=ABU,
                va="center")
        ax.plot([143, 158], [ty2 - 1.8, ty2 - 1.8], color=TINTA, lw=0.7)
        for n in range(min(3, len(kirim))):
            ax.text(143.6, ty2 - 4 - n * 3.6, f"{str(kirim.iloc[n, 0]):<8}", fontsize=8.6, family=MONO, va="center")
            ax.text(143.6 + 8 * 0.86, ty2 - 4 - n * 3.6, str(kirim.iloc[n, 1]), fontsize=8.6, family=MONO,
                    va="center", color=wk.get(kirim.iloc[n, 1], TINTA), fontweight="semibold")
        ax.plot([143, 158], [ty2 - 14.4, ty2 - 14.4], color=TINTA, lw=0.7)
        ax.text(143, ty2 - 16.8, f"{ribuan(len(kirim))} baris", fontsize=9, color=ABU, va="center")

    # --- jalur topik
    if info is not None:
        dok = baca("topik", "topik_dokumen.csv")
        pilih = baca("topik", "01_pilih_metode_dan_jumlah_topik.csv")
        k = len(info)
        metode = ""
        if pilih is not None:
            c = pilih[pilih["jumlah_topik"] == k].sort_values("npmi")
            metode = c["metode"].iloc[-1].upper() if len(c) else ""
        n_dok = len(dok) if dok is not None else len(df)
        panah(ax, (45.5, yb_bawah), (45.5, 20.6))
        kotak(ax, 33, 9, 25, 11, fc="white", ec=TINTA, lw=0.8)
        ax.text(45.5, 16.6, f"{ribuan(n_dok)} teks", fontsize=11, fontweight="bold", ha="center", va="center")
        ax.text(45.5, 12.4, "latih + uji, tanpa label", fontsize=9, ha="center", va="center", color=ABU)
        rng = np.random.default_rng(1)   # matriks skematis, bukan data
        for a in range(7):
            for b in range(16):
                ax.add_patch(Rectangle((62 + b * 1.5, 9 + a * 1.55), 1.25, 1.3,
                                       fc=TINTA if rng.random() < 0.13 else LATAR, ec="none"))
        ax.text(62, 21.6, "matriks dokumen × kata (TF-IDF)", fontsize=9, color=ABU, va="center")
        panah(ax, (58, 14.5), (61.3, 14.5))
        ax.text(89.5, 14.5, "≈", fontsize=15, ha="center", va="center")
        for a in range(7):
            for b in range(2):
                ax.add_patch(Rectangle((92.5 + b * 1.5, 9 + a * 1.55), 1.25, 1.3, fc=W_LAIN[b], alpha=0.75,
                                       ec="none"))
        ax.text(96.6, 14.5, "×", fontsize=11, ha="center", va="center")
        for a in range(2):
            for b in range(12):
                ax.add_patch(Rectangle((98.4 + b * 1.5, 13 + a * 1.55), 1.25, 1.3, fc=W_LAIN[1 - a], alpha=0.75,
                                       ec="none"))
        ax.text(92.5, 21.6, f"{metode}: dokumen × topik, topik × kata", fontsize=9, color=ABU, va="center")
        ax.text(92.5, 6.4, f"K = {k}, dipilih dengan NPMI {koma(info['npmi'].mean())}", fontsize=9.5,
                fontweight="semibold", va="center")
        panah(ax, (117, 14.5), (120.4, 14.5))
        tinggi = min(3.5, 20 / k)
        for n, r in enumerate(info.itertuples()):
            yy = 22.2 - n * tinggi
            kode, _ = nama_pendek(r.nama)
            milik = "_topik" in df and df["_topik"].iloc[i] == r.nama
            if milik:
                ax.add_patch(Rectangle((121, yy - tinggi / 2 + 0.1), 37, tinggi - 0.2, fc=SOROT, ec="none", zorder=1))
            ax.scatter(122.6, yy, s=26, color=W_LAIN[n % len(W_LAIN)], zorder=3)
            ax.text(124.4, yy, kode, fontsize=9, fontweight="bold", va="center", zorder=3)
            ax.text(128, yy, ", ".join(str(r.kata_teratas).split(", ")[:3]), fontsize=9, va="center", zorder=3)
            ax.text(158, yy, ("contoh ini · " + koma(df["_kuat"].iloc[i], 2)) if milik else persen(r.porsi, 0),
                    fontsize=8.6, family=MONO, va="center", ha="right", color=TINTA if milik else ABU, zorder=3)
    simpan(fig, "did", "did_1_pipeline",
           "Alur sistem dengan satu teks latih asli yang diikuti dari tabel sampai keluaran. Atas: dua model "
           "klasifikasi dinilai dengan prediksi OOF; batang menunjukkan keyakinan tiap model pada contoh itu. "
           "Bawah: topic modeling pada semua teks tanpa label.")


# ================= WHAT WE SEE =================
def kata_beda(df, a, b):
    """Log-odds berprior (Monroe dkk. 2008) kelas a vs b; z > 0 khas a."""
    hit = {}
    for k in (a, b):
        hit[k] = Counter(w for t in df.loc[df[TARGET] == k, "_bersih"] for w in t.split())
    semua = Counter(w for t in df["_bersih"] for w in t.split())
    total = sum(semua.values())
    na, nb = sum(hit[a].values()), sum(hit[b].values())
    a0 = 500.0
    baris = []
    for w, f in semua.items():
        fa, fb = hit[a][w], hit[b][w]
        if fa + fb < 5 or len(w) < 2:
            continue
        al = a0 * f / total
        d = np.log((fa + al) / (na + a0 - fa - al)) - np.log((fb + al) / (nb + a0 - fb - al))
        baris.append((w, fa + fb, d / np.sqrt(1 / (fa + al) + 1 / (fb + al))))
    return pd.DataFrame(baris, columns=["kata", "n", "z"])


def renggang(nilai, jarak):
    """Geser posisi label agar berjarak minimal `jarak` (urutan dipertahankan)."""
    urut = np.argsort(nilai)
    hasil = np.array(nilai, dtype=float)
    for a, b in zip(urut[:-1], urut[1:]):
        if hasil[b] - hasil[a] < jarak:
            hasil[b] = hasil[a] + jarak
    return hasil


def fig_data(D):
    df, kelas, y, wk = D["df"], D["kelas"], D["y"], D["wk"]
    fig, ax = kanvas()
    # a: kata pembeda dua kelas terbesar
    besar = sorted(df[TARGET].value_counts().index[:2])
    ka, kb = besar[-1], besar[0]
    kb_df = kata_beda(df, ka, kb)
    huruf(ax, 2, 84.5, "a", f"Kata yang membedakan {ka} dan {kb}")
    a = sumbu(fig, 9, 9, 66, 70)
    a.scatter(kb_df["n"], kb_df["z"], s=5, color=GARIS, lw=0, zorder=1)
    a.axhline(0, color=TINTA, lw=0.8)
    lim = kb_df["z"].abs().max() * 1.3
    kolom_x = kb_df["n"].max() * 2.2   # label kata dijajarkan dalam satu kolom di kanan titik
    for kls, tanda in [(ka, 1), (kb, -1)]:
        atas = kb_df[kb_df["z"] * tanda > 0].assign(m=lambda d: d["z"].abs()).nlargest(12, "m")
        a.scatter(atas["n"], atas["z"], s=16, color=wk[kls], lw=0, zorder=3)
        pos = renggang(atas["z"].to_numpy() * tanda, lim / 13) * tanda
        for (w, n, z), yy in zip(atas[["kata", "n", "z"]].itertuples(index=False), pos):
            a.annotate(w, (n, z), xytext=(kolom_x, yy), fontsize=10, color=wk[kls], va="center",
                       fontweight="semibold", path_effects=PUTIH,
                       arrowprops=dict(arrowstyle="-", color=wk[kls], lw=0.4, shrinkA=0, shrinkB=1))
    a.set_xscale("log")
    a.set_xlim(4, kb_df["n"].max() * 14)
    a.set_ylim(-lim * 1.25, lim * 1.25)
    a.set_xlabel(f"berapa kali kata muncul (skala log; {ribuan(len(kb_df))} kata yang muncul ≥ 5 kali)")
    a.set_ylabel("skor z log-odds")
    a.text(0.015, 0.985, f"↑ lebih khas {ka}", transform=a.transAxes, ha="left", va="top", color=wk[ka],
           fontsize=11, fontweight="bold")
    a.text(0.015, 0.015, f"↓ lebih khas {kb}", transform=a.transAxes, ha="left", va="bottom", color=wk[kb],
           fontsize=11, fontweight="bold")

    # b: komposisi kelas, label latih vs prediksi uji
    sebar = baca("laporan", "10_sebaran_prediksi_test.csv")
    huruf(ax, 86, 84.5, "b", "Komposisi kelas")
    x0, w = 104, 52
    porsi_l = df[TARGET].value_counts(normalize=True).reindex(kelas).to_numpy()
    tepi_l = np.concatenate([[0], np.cumsum(porsi_l)])
    for j, k in enumerate(kelas):
        ax.add_patch(Rectangle((x0 + w * tepi_l[j], 73), w * porsi_l[j] - 0.3, 5, fc=wk[k], ec="none"))
        ax.text(x0 + w * (tepi_l[j] + porsi_l[j] / 2), 75.5, f"{k} {persen(porsi_l[j], 0)}", fontsize=9.5,
                color="white", ha="center", va="center", fontweight="semibold")
    ax.text(x0 - 1.2, 75.5, f"label latih\nn = {ribuan(len(df))}", fontsize=9.5, ha="right", va="center",
            linespacing=1.15)
    if sebar is not None:
        porsi_u = sebar.set_index("kelas")["porsi_prediksi_test"].reindex(kelas).to_numpy()
        tepi_u = np.concatenate([[0], np.cumsum(porsi_u)])
        for j, k in enumerate(kelas):
            ax.add_patch(Polygon([(x0 + w * tepi_l[j], 73), (x0 + w * tepi_l[j + 1] - 0.3, 73),
                                  (x0 + w * tepi_u[j + 1] - 0.3, 67), (x0 + w * tepi_u[j], 67)], fc=wk[k],
                                 alpha=0.18, ec="none"))
            ax.add_patch(Rectangle((x0 + w * tepi_u[j], 62), w * porsi_u[j] - 0.3, 5, fc=wk[k], ec="none"))
            if porsi_u[j] > 0.07:
                ax.text(x0 + w * (tepi_u[j] + porsi_u[j] / 2), 64.5, persen(porsi_u[j], 0), fontsize=9.5,
                        color="white", ha="center", va="center", fontweight="semibold")
        ax.text(x0 - 1.2, 64.5, "prediksi uji", fontsize=9.5, ha="right", va="center")

    # c: panjang teks per kelas (sebaran + titik tiap teks)
    huruf(ax, 86, 54.5, "c", "Panjang teks per kelas")
    c = sumbu(fig, 104, 29, 52, 22)
    batas = np.percentile(df["_kata"], 99)
    rng = np.random.default_rng(0)
    from scipy.stats import gaussian_kde
    grid = np.linspace(0, batas, 200)
    for j, k in enumerate(kelas):
        v = df.loc[df[TARGET] == k, "_kata"].to_numpy()
        v = v[v <= batas]
        d = gaussian_kde(v)(grid)
        dasar = len(kelas) - 1 - j
        c.fill_between(grid, dasar + 0.08, dasar + 0.08 + 0.78 * d / d.max(), color=wk[k], alpha=0.8, lw=0)
        ambil = rng.choice(v, min(len(v), 500), replace=False)
        c.scatter(ambil + rng.uniform(-0.4, 0.4, len(ambil)), dasar - rng.uniform(0.02, 0.22, len(ambil)), s=1.2,
                  color=wk[k], alpha=0.45, lw=0)
        med = np.median(v)
        c.plot([med, med], [dasar - 0.24, dasar + 0.86], color=TINTA, lw=1)
        c.text(med + 0.8, dasar + 0.62, f"median {koma(med, 0)}", fontsize=9, va="center", path_effects=PUTIH)
    c.set_yticks(range(len(kelas)), kelas[::-1], fontsize=10.5)
    for t, k in zip(c.get_yticklabels(), kelas[::-1]):
        t.set_color(wk[k])
        t.set_fontweight("semibold")
    c.tick_params(axis="y", length=0)
    c.spines["left"].set_visible(False)
    c.set_ylim(-0.3, len(kelas))
    c.set_xlim(0, batas)
    c.set_xlabel("jumlah kata per teks")

    # d: label menurut urutan baris + skor satu kolom
    kol = baca("eda", "kekuatan_kolom.csv")
    huruf(ax, 86, 19.5, "d", "Urutan baris membocorkan label")
    d = sumbu(fig, 104, 12.5, 52, 3.6)
    d.imshow(y[None, :], aspect="auto", cmap=ListedColormap([wk[k] for k in kelas]), interpolation="nearest",
             extent=(1, len(y), 0, 1))
    d.set_yticks([])
    d.set_xticks([1, len(y)], ["baris 1", f"baris {ribuan(len(y))}"])
    d.tick_params(length=0)
    d.get_xticklabels()[0].set_ha("left")
    d.get_xticklabels()[1].set_ha("right")
    for s in d.spines.values():
        s.set_visible(False)
    ax.text(102.8, 14.3, "label", fontsize=9.5, ha="right", va="center")
    if kol is not None:
        kol = kol.set_index("kolom")["skor"]
        bocor = [f"{n} {koma(kol[n])}" for n in kol.index if kol[n] >= 0.95]
        acak = [f"{n} {koma(kol[n])}" for n in kol.index if n == "kolom acak"]
        ax.text(104, 5.6, "skor menebak label dari satu kolom", fontsize=9.5, color=ABU, va="center")
        ax.text(104, 2.4, " · ".join(bocor), fontsize=10, fontweight="bold", va="center", color="#c8453b")
        if acak:
            ax.text(156, 2.4, acak[0], fontsize=10, va="center", ha="right", color=ABU)
    simpan(fig, "see", "see_1_data",
           "Data latih. (a) Tiap titik satu kata; makin jauh dari garis tengah makin khas untuk satu kelas "
           "(log-odds berprior). (b) Porsi kelas pada label latih dan pada prediksi data uji. (c) Sebaran panjang "
           "teks; titik = teks. (d) Label disusun berblok menurut urutan baris, sehingga id dan nomor baris bisa "
           "menebak label; keduanya tidak dipakai sebagai fitur.")


def peta_dokumen(df):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    import umap
    X = TfidfVectorizer(min_df=5, max_df=0.4, sublinear_tf=True).fit_transform(df["_bersih"])
    Z = TruncatedSVD(50, random_state=0).fit_transform(X)
    Z = Z[:, 1:] / np.maximum(np.linalg.norm(Z[:, 1:], axis=1, keepdims=True), 1e-9)
    return umap.UMAP(n_neighbors=40, min_dist=0.6, spread=1.5, random_state=0).fit_transform(Z)


def fig_peta(D):
    df, kelas, y, wk = D["df"], D["kelas"], D["y"], D["wk"]
    try:
        xy = peta_dokumen(df)
    except Exception as e:
        print("  peta dokumen dilewati:", e)
        return
    fig, ax = kanvas()
    p1 = D["prob"][D["model"][0]]
    salah = p1.argmax(1) != y
    ada_topik = "_topik" in df
    topik = sorted(t for t in df["_topik"].dropna().unique() if not str(t).startswith("tanpa")) if ada_topik else []
    batas = np.percentile(xy, [0.5, 99.5], axis=0)   # buang titik terpencil dari bingkai
    pusat = {t: np.median(xy[(df["_topik"] == t).to_numpy()], axis=0) for t in topik}
    if topik:
        geser = renggang([pusat[t][1] for t in topik], (batas[1, 1] - batas[0, 1]) * 0.05)
        pusat = {t: (pusat[t][0], g) for t, g in zip(topik, geser)}
    wt = {t: W_LAIN[i % len(W_LAIN)] for i, t in enumerate(topik)}
    panel = [("a", "Diwarnai menurut label"), ("b", "Diwarnai menurut topik (NMF)"),
             ("c", f"Salah prediksi · {D['model'][0]}")]
    for n, (h, judul) in enumerate(panel):
        x0 = 3 + n * 52.5
        huruf(ax, x0, 84.5, h, judul)
        a = sumbu(fig, x0, 5, 50, 76)
        a.set_axis_off()
        a.set_xlim(batas[0, 0], batas[1, 0])
        a.set_ylim(batas[0, 1], batas[1, 1])
        if n == 0:
            a.scatter(xy[:, 0], xy[:, 1], s=3.2, c=[wk[k] for k in df[TARGET]], alpha=0.55, lw=0)
            for j, k in enumerate(kelas):
                ax.text(x0 + 1 + j * 14, 3, f"● {k}", color=wk[k], fontsize=10.5, fontweight="semibold", va="center")
        elif n == 1 and ada_topik:
            a.scatter(xy[:, 0], xy[:, 1], s=3.2, c=[wt.get(t, GARIS) for t in df["_topik"]], alpha=0.55, lw=0)
            for t in topik:
                m = (df["_topik"] == t).to_numpy()
                kode, kata = nama_pendek(t)
                a.text(pusat[t][0], pusat[t][1], f"{kode} {kata}", fontsize=10.5, fontweight="bold",
                       ha="center", va="center", path_effects=PUTIH)
        else:
            a.scatter(xy[~salah, 0], xy[~salah, 1], s=3.2, color=GARIS, alpha=0.6, lw=0)
            a.scatter(xy[salah, 0], xy[salah, 1], s=9, c=[wk[kelas[j]] for j in y[salah]], lw=0.3, edgecolors=TINTA)
            for t in topik:
                m = (df["_topik"] == t).to_numpy()
                kode, _ = nama_pendek(t)
                a.text(pusat[t][0], pusat[t][1], f"{kode}: {persen(salah[m].mean(), 0)} salah",
                       fontsize=10, fontweight="bold", ha="center", va="center", path_effects=PUTIH)
            ax.text(x0 + 1, 3, f"{ribuan(salah.sum())} dari {ribuan(len(y))} teks salah; warna = label asli",
                    fontsize=10, color=ABU, va="center")
    simpan(fig, "see", "see_2_peta",
           "Peta dokumen: tiap titik satu teks latih, teks dengan kata serupa berdekatan (TF-IDF → SVD → UMAP, hanya "
           "untuk gambar). (a) label, (b) topik hasil NMF, (c) teks yang salah diprediksi model terbaik dan porsi "
           "salah per topik.")


# ================= WHAT WE FOUND =================
def fig_model(D):
    kelas, y, wk = D["kelas"], D["y"], D["wk"]
    K, n = len(kelas), len(y)
    pred = {m: p.argmax(1) for m, p in D["prob"].items()}
    nama = list(pred)
    warna = {m: W_MODEL[i % len(W_MODEL)] for i, m in enumerate(nama)}
    rata = None
    if len(nama) > 1:
        rata = "rata-rata " + str(len(nama)) + " model"
        pred[rata] = np.mean([D["prob"][m] for m in nama], axis=0).argmax(1)
        warna[rata] = W_MODEL[2]
    rng = np.random.default_rng(0)
    boot = {m: np.empty(BOOTSTRAP) for m in pred}
    for b in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        for m in pred:
            boot[m][b] = f1_makro(y[idx], pred[m][idx], K)
    fig, ax = kanvas()
    from scipy.stats import gaussian_kde

    # a: sebaran F1 bootstrap
    huruf(ax, 2, 84.5, "a", f"F1 makro OOF dan ketidakpastiannya ({ribuan(BOOTSTRAP)} bootstrap)")
    a = sumbu(fig, 24, 53, 60, 27)
    urut = list(pred)[::-1]
    for j, m in enumerate(urut):
        g = np.linspace(boot[m].min(), boot[m].max(), 120)
        d = gaussian_kde(boot[m])(g)
        a.fill_between(g, j, j + 0.8 * d / d.max(), color=warna[m], alpha=0.75, lw=0)
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        asli = f1_makro(y, pred[m], K)
        a.plot([lo, hi], [j - 0.1, j - 0.1], color=TINTA, lw=1.4)
        a.scatter(asli, j - 0.1, s=22, color=TINTA, zorder=3)
        a.text(hi + 0.002, j + 0.25, f"{koma(asli, 4)}  [{koma(lo)}–{koma(hi)}]", fontsize=10, va="center",
               family=MONO)
    a.set_yticks(np.arange(len(urut)) + 0.2, urut, fontsize=10.5)
    for t, m in zip(a.get_yticklabels(), urut):
        t.set_color(warna[m])
        t.set_fontweight("semibold")
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    a.set_ylim(-0.4, len(urut))
    a.set_xlim(right=a.get_xlim()[1] + 0.035)
    a.set_xlabel("F1 makro (garis = selang 95%)")
    a.xaxis.set_major_formatter(lambda v, p: koma(v, 2))

    # b: selisih berpasangan rata-rata vs tunggal terbaik
    if rata:
        beda = boot[rata] - boot[nama[0]]
        huruf(ax, 96, 84.5, "b", "Apakah menggabung model menolong?")
        b = sumbu(fig, 103, 53, 52, 19)
        b.hist(beda, bins=40, color=W_MODEL[2], alpha=0.75, lw=0)
        b.axvline(0, color=TINTA, lw=1)
        lo, hi = np.percentile(beda, [2.5, 97.5])
        b.set_yticks([])
        b.spines["left"].set_visible(False)
        b.set_xlabel(f"ΔF1 = {rata} − {nama[0]}")
        b.xaxis.set_major_formatter(lambda v, p: ("+" if v > 0 else "") + koma(v, 3))
        ax.text(103, 78.5, f"Δ = {koma(f1_makro(y, pred[rata], K) - f1_makro(y, pred[nama[0]], K), 4)}   "
                f"selang 95% [{koma(lo, 4)}, {koma(hi, 4)}]", fontsize=10, family=MONO, va="center")
        ax.text(103, 75, f"Δ > 0 di {persen((beda > 0).mean(), 0)} resampling: selang memuat nol",
                fontsize=10, family=MONO, va="center")

    # c: F1 per kelas, tiap model
    huruf(ax, 2, 43.5, "c", "F1 per kelas")
    c = sumbu(fig, 14, 10, 34, 27)
    f1k = {}
    for m in nama:
        cm = np.bincount(y * K + pred[m], minlength=K * K).reshape(K, K)
        f1k[m] = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
    for j, k in enumerate(kelas):
        nilai = [f1k[m][j] for m in nama]
        c.plot([min(nilai), max(nilai)], [j, j], color=GARIS, lw=2.5, zorder=1)
        for m in nama:
            c.scatter(f1k[m][j], j, s=70, color=warna[m], zorder=3)
        c.text(max(nilai) + 0.006, j, koma(max(nilai)), fontsize=10, va="center", family=MONO)
        c.text(min(nilai) - 0.006, j, koma(min(nilai)), fontsize=10, va="center", ha="right", family=MONO)
    c.set_yticks(range(K), kelas, fontsize=10.5)
    for t, k in zip(c.get_yticklabels(), kelas):
        t.set_color(wk[k])
        t.set_fontweight("semibold")
    c.set_ylim(K - 0.5, -0.5)
    c.tick_params(axis="y", length=0)
    c.spines["left"].set_visible(False)
    kiri, kanan = c.get_xlim()
    c.set_xlim(kiri - 0.03, kanan + 0.03)
    c.xaxis.set_major_formatter(lambda v, p: koma(v, 2))
    c.set_xlabel("F1")
    for j, m in enumerate(nama):
        ax.text(14 + j * 17, 39.4, "● " + m, color=warna[m], fontsize=9.5, fontweight="semibold", va="center")

    # d: confusion matrix model terbaik
    huruf(ax, 60, 43.5, "d", f"Tertukar ke mana · {nama[0]}")
    d = sumbu(fig, 74, 10, 28, 29)
    cm = np.bincount(y * K + pred[nama[0]], minlength=K * K).reshape(K, K)
    porsi = cm / cm.sum(1, keepdims=True)
    d.imshow(porsi, cmap=PETA_NILA, vmin=0, vmax=1, aspect="auto")
    for i in range(K):
        for j in range(K):
            d.text(j, i, f"{ribuan(cm[i, j])}\n{persen(porsi[i, j])}", ha="center", va="center", fontsize=9.5,
                   color="white" if porsi[i, j] > 0.6 else TINTA, linespacing=1.3)
    d.set_xticks(range(K), kelas)
    d.set_yticks(range(K), kelas)
    d.tick_params(length=0)
    d.set_xlabel("prediksi")
    d.set_ylabel("label")
    for s in d.spines.values():
        s.set_visible(False)

    # e: keandalan keyakinan
    huruf(ax, 110, 43.5, "e", "Keyakinan vs kenyataan")
    e = sumbu(fig, 121, 10, 34, 29)
    e.plot([1 / K, 1], [1 / K, 1], color=REDUP, lw=0.9, ls=(0, (3, 3)))
    tepi = np.linspace(1 / K, 1, 9)
    for j, m in enumerate(nama):
        yakin = D["prob"][m].max(1)
        betul = pred[m] == y
        kotak_ke = np.clip(np.digitize(yakin, tepi) - 1, 0, len(tepi) - 2)
        xs, ys, ns = [], [], []
        for q in range(len(tepi) - 1):
            mm = kotak_ke == q
            if mm.sum() >= 10:
                xs.append(yakin[mm].mean())
                ys.append(betul[mm].mean())
                ns.append(mm.sum())
        ece = sum(nn * abs(a_ - b_) for a_, b_, nn in zip(xs, ys, ns)) / n
        e.plot(xs, ys, color=warna[m], lw=1.6, zorder=2)
        e.scatter(xs, ys, s=12 + 160 * np.array(ns) / n, color=warna[m], zorder=3, lw=0.6, edgecolors="white")
        e.text(0.03, 0.97 - j * 0.1, f"● {m}  ECE {koma(ece)}", transform=e.transAxes, color=warna[m],
               fontsize=9.5, fontweight="semibold", va="top")
    e.set_xlabel("keyakinan (lingkaran besar = banyak teks)")
    e.set_ylabel("porsi benar")
    e.set_xlim(1 / K - 0.03, 1.03)
    e.set_ylim(1 / K - 0.1, 1.03)
    for sb in (e.xaxis, e.yaxis):
        sb.set_major_formatter(lambda v, p: koma(v, 1))
    simpan(fig, "found", "found_1_model",
           "Hasil klasifikasi dari prediksi OOF. (a) F1 makro tiap model dengan sebaran bootstrap dan selang 95%. "
           "(b) Selisih berpasangan antara rata-rata model dan model tunggal terbaik. (c) F1 per kelas. (d) "
           "Confusion matrix model terbaik, persen per baris. (e) Diagram keandalan: di garis putus-putus keyakinan "
           "sama dengan porsi benar.")


def fig_contoh(D):
    df, kelas, y, wk = D["df"], D["kelas"], D["y"], D["wk"]
    if len(D["model"]) < 2:
        return
    m1, m2 = D["model"][:2]
    p1, p2 = D["prob"][m1], D["prob"][m2]
    r = np.arange(len(y))
    t1, t2 = p1.argmax(1), p2.argmax(1)
    pendek = (df["_teks"].str.len() <= 190).to_numpy()

    def ambil(mask, nilai, n=2):
        idx = np.where(mask & pendek)[0]
        return list(idx[np.argsort(-nilai[idx])][:n])

    kelompok = [
        (f"Kedua model sepakat melawan label ({ribuan(((t1 != y) & (t1 == t2)).sum())} teks): kandidat label keliru,"
         " belum dicek manusia", ambil((t1 != y) & (t1 == t2), p1.max(1) + p2.max(1), 3)),
        (f"Hanya {m1} yang salah ({ribuan(((t1 != y) & (t2 == y)).sum())} teks)",
         ambil((t1 != y) & (t2 == y), p1.max(1))),
        (f"{m1} benar, {m2} salah ({ribuan(((t1 == y) & (t2 != y)).sum())} teks): yang didapat dari fine-tune",
         ambil((t1 == y) & (t2 != y), p1[r, y] - p2[r, y])),
    ]
    fig, ax = kanvas()
    X_T, X_L, X_1, X_2 = 3, 99, 113, 137
    ax.plot([2, 158], [87.5, 87.5], color=TINTA, lw=1.3)
    for x, t in [(X_T, "teks"), (X_L, "label"), (X_1, f"p(kelas) · {m1}"), (X_2, f"p(kelas) · {m2}")]:
        ax.text(x, 84.8, t, fontsize=10.5, fontweight="semibold", va="center")
    ax.plot([2, 158], [82.4, 82.4], color=TINTA, lw=0.8)
    yy = 82.4
    tampil = kelas[:3]   # shortcut: hanya 3 kelas pertama digambar per baris; lebih dari itu perlu baris lebih tinggi
    for judul, baris in kelompok:
        if not baris:
            continue
        yy -= 4.4
        ax.text(X_T, yy + 1.2, judul, fontsize=10, style="italic", color=ABU, va="center")
        for i in baris:
            for n, b in enumerate(bungkus(df["_teks"].iloc[i], 100, 2)):
                ax.text(X_T, yy - 3.0 - n * 3.6, b, fontsize=9.2, family=MONO, va="center")
            ax.text(X_L, yy - 3.0, kelas[y[i]], fontsize=10.5, fontweight="bold", color=wk[kelas[y[i]]],
                    va="center")
            for x, p in [(X_1, p1[i]), (X_2, p2[i])]:
                for j, k in enumerate(tampil):
                    by = yy - 1.8 - j * 2.6
                    ax.add_patch(Rectangle((x + 4, by - 0.9), 11, 1.8, fc=LATAR, ec="none"))
                    ax.add_patch(Rectangle((x + 4, by - 0.9), 11 * p[j], 1.8, fc=wk[k], ec="none"))
                    ax.text(x + 3.2, by, str(k)[:3], fontsize=8, ha="right", va="center", color=wk[k])
                    if j == int(p.argmax()):
                        ax.text(x + 15.8, by, koma(p[j], 2) + (" ✓" if j == y[i] else " ✗"), fontsize=8.6,
                                family=MONO, va="center", fontweight="semibold",
                                color=TINTA if j == y[i] else "#c8453b")
            yy -= 9.6
            ax.plot([2, 158], [yy + 0.6, yy + 0.6], color=GARIS, lw=0.6)
    ax.plot([2, 158], [yy + 0.6, yy + 0.6], color=TINTA, lw=1.3)
    simpan(fig, "found", "found_2_contoh",
           "Contoh teks asli dari data latih, dipilih otomatis (keyakinan tertinggi di tiap kelompok). Batang = "
           "keyakinan OOF tiap model per kelas; ✓ prediksi sama dengan label, ✗ berbeda.")


def fig_topik():
    pilih = baca("topik", "01_pilih_metode_dan_jumlah_topik.csv")
    info = baca("topik", "02_ukuran_topik.csv")
    kata = baca("topik", "03_kata_teratas.csv")
    silang = baca("topik", "06_topik_x_kelas.csv")
    geser = baca("topik", "07_topik_train_vs_test.csv")
    if info is None or kata is None:
        return
    topik = list(info["nama"])
    wt = {t: W_LAIN[i % len(W_LAIN)] for i, t in enumerate(topik)}
    fig, ax = kanvas()
    # a: pemilihan metode dan K
    if pilih is not None:
        huruf(ax, 2, 84.5, "a", "Memilih metode dan jumlah topik")
        a = sumbu(fig, 10, 52, 38, 28)
        for m in pilih["metode"].unique():
            b = pilih[pilih["metode"] == m]
            dipakai = (b["jumlah_topik"] == len(info)).any() and b["npmi"].max() == pilih["npmi"].max()
            a.plot(b["jumlah_topik"], b["npmi"], color=TINTA if dipakai else REDUP, lw=2 if dipakai else 1.2,
                   marker="o", markersize=5 if dipakai else 3.5)
            a.text(b["jumlah_topik"].iloc[-1] + 0.4, b["npmi"].iloc[-1], m, fontsize=10.5, va="center",
                   fontweight="bold" if dipakai else "normal", color=TINTA if dipakai else ABU)
            if dipakai:
                d = b[b["jumlah_topik"] == len(info)].iloc[0]
                a.scatter(d["jumlah_topik"], d["npmi"], s=190, facecolors=SOROT, edgecolors=TINTA, lw=1.4, zorder=4)
                a.annotate(f"K = {len(info)}\nNPMI {koma(d['npmi'])}", (d["jumlah_topik"], d["npmi"]),
                           xytext=(14, 4), textcoords="offset points", fontsize=10, fontweight="semibold",
                           va="center")
        a.axhline(0, color=GARIS, lw=0.8)
        a.set_xticks(sorted(pilih["jumlah_topik"].unique()))
        a.set_xlim(right=pilih["jumlah_topik"].max() + 3.2)
        a.set_xlabel("jumlah topik K")
        a.set_ylabel("NPMI (koherensi)")
        a.yaxis.set_major_formatter(lambda v, p: koma(v, 1))
    # b: topik x kelas, c: porsi latih vs uji (baris sama)
    if silang is not None:
        silang = silang.set_index("topik").reindex(topik)
        kelas = list(silang.columns)
        wk = warna_kelas(sorted(kelas))
        urut = list(silang[kelas[0]].sort_values(ascending=False).index)
        huruf(ax, 62, 84.5, "b", "Isi kelas tiap topik")
        y0, hh = 52, 28 / len(urut)
        for n, t in enumerate(urut):
            yy = y0 + 28 - (n + 0.5) * hh
            kode, kt = nama_pendek(t)
            ax.scatter(64, yy, s=30, color=wt[t])
            ax.text(66, yy, f"{kode} {kt}", fontsize=10.5, fontweight="semibold", va="center")
            kiri = 82
            for k in kelas:
                w = 30 * silang.loc[t, k]
                ax.add_patch(Rectangle((kiri, yy - hh * 0.36), max(w - 0.25, 0), hh * 0.72, fc=wk[k], ec="none"))
                if silang.loc[t, k] >= 0.14:
                    ax.text(kiri + w / 2, yy, persen(silang.loc[t, k], 0), fontsize=9, color="white", ha="center",
                            va="center", fontweight="semibold")
                kiri += w
        for j, k in enumerate(kelas):
            ax.text(82 + j * 10.5, 49.6, "■ " + k, color=wk[k], fontsize=9.5, fontweight="semibold", va="center")
        if geser is not None:
            geser = geser.set_index("topik").reindex(urut)
            huruf(ax, 120, 84.5, "c", "Porsi topik: latih vs uji")
            maks = float(geser[["train", "test"]].max().max())
            x0, w = 122, 22
            for n, t in enumerate(urut):
                yy = y0 + 28 - (n + 0.5) * hh
                xa, xb = x0 + w * geser.loc[t, "train"] / maks, x0 + w * geser.loc[t, "test"] / maks
                ax.plot([x0, x0 + w], [yy, yy], color=GARIS, lw=0.6)
                ax.plot([xa, xb], [yy, yy], color=wt[t], lw=2.4)
                ax.scatter(xa, yy, s=44, facecolors="white", edgecolors=wt[t], lw=1.6, zorder=3)
                ax.scatter(xb, yy, s=44, color=wt[t], zorder=3)
                ax.text(x0 + w + 2, yy, f"{persen(geser.loc[t, 'train'], 0)} → {persen(geser.loc[t, 'test'], 0)}",
                        fontsize=9, family=MONO, va="center")
            ax.text(x0, 49.6, "○ latih   ● uji", fontsize=9.5, color=ABU, va="center")
    # d: kata teratas tiap topik
    huruf(ax, 2, 42.5, "d", "Kata berbobot tertinggi tiap topik")
    lebar = 156 / len(topik)
    for n, t in enumerate(topik):
        k = kata[kata["topik"] == t].sort_values("urutan").head(8)
        x0 = 2 + n * lebar
        kode, _ = nama_pendek(t)
        r = info[info["nama"] == t].iloc[0]
        ax.text(x0, 37.4, kode, fontsize=12, fontweight="bold", color=wt[t], va="center")
        ax.text(x0 + 4, 37.4, f"{persen(r['porsi'], 0)} teks · NPMI {koma(r['npmi'], 2)}", fontsize=9, color=ABU,
                va="center")
        ax.plot([x0, x0 + lebar - 3], [35.2, 35.2], color=wt[t], lw=1.6)
        for j, (w, b) in enumerate(zip(k["kata"], k["bobot"])):
            yy = 32.4 - j * 3.7
            ax.add_patch(Rectangle((x0, yy - 1.45), (lebar - 3) * b / k["bobot"].max(), 2.9, fc=wt[t], alpha=0.22,
                                   ec="none"))
            ax.text(x0 + 0.7, yy, w, fontsize=9.8, va="center")
    simpan(fig, "found", "found_3_topik",
           "Topic modeling. (a) NPMI tiap metode dan jumlah topik; lingkaran = yang dipakai. (b) Porsi kelas di "
           "dalam tiap topik. (c) Porsi topik di data latih dan data uji. (d) Delapan kata berbobot tertinggi tiap "
           "topik; panjang batang = bobot relatif.")


# ---------------- Pratinjau ----------------
def pratinjau():
    judul = {"see": "What We See", "did": "What We Did", "found": "What We Found"}
    isi, no = [], 0
    for b in ("see", "did", "found"):
        for bag, nama, ket in FIGUR:
            if bag != b:
                continue
            no += 1
            isi.append(f'<figure><div class="b">{judul[b]}</div><img src="{nama}.png" alt="{nama}">'
                       f'<figcaption><b>Gambar {no}.</b> {html.escape(ket)} <span>{nama}.png · .svg</span>'
                       f'</figcaption></figure>')
    with open(f"{KELUAR}/pratinjau.html", "w", encoding="utf-8") as f:
        f.write("""<!doctype html><html lang="id"><meta charset="utf-8"><title>Figur Slide</title><style>
body{margin:0;background:#fff;color:#141414;font:15px/1.5 'IBM Plex Sans',system-ui,sans-serif}
figure{max-width:1280px;margin:36px auto 56px;padding:0 16px}img{width:100%;display:block;border:1px solid #e3e3e3}
.b{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#777;margin-bottom:8px}
figcaption{margin-top:10px;max-width:960px}span{color:#999;font-size:12px;margin-left:6px}</style>
""" + "\n".join(isi) + "</html>")
    print("  pratinjau", f"{KELUAR}/pratinjau.html")


print("Membaca", os.path.abspath(SUMBER))
D = muat()
if D is not None:
    fig_data(D)
    fig_peta(D)
    fig_pipeline(D)
    fig_model(D)
    fig_contoh(D)
fig_topik()
pratinjau()
print("Selesai:", os.path.abspath(KELUAR))
