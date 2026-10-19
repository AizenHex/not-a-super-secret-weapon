# Membuat semua gambar TikZ (fig/*.tex) dan angka hasil hitung (angka.tex) untuk ringkas.tex.
# Data mainan dibuat dengan seed tetap, jadi hasilnya selalu sama.
# Jalankan: python buat_gambar.py  lalu  pdflatex ringkas.tex (dua kali, untuk daftar isi)
import numpy as np

FIG_FOLDER = "fig"
AR = 1.6            # lebar panel 2D = 1.6, tinggi = 1
C0 = "biru2"        # warna kelas 0
C1 = "merah3"       # warna kelas 1
macros = []


def koma(nilai, desimal):
    return f"{nilai:.{desimal}f}".replace(".", "{,}")


def angka(nama, nilai, desimal=2):
    macros.append(f"\\newcommand{{\\{nama}}}{{{koma(nilai, desimal)}}}")


def simpan(nama, isi):
    with open(f"{FIG_FOLDER}/{nama}.tex", "w", encoding="utf-8") as f:
        f.write(isi)


def num(x):
    return f"{x:.4f}"


# ---------------------------------------------------------------- data
def moons(n, seed, noise):
    rng = np.random.default_rng(seed)
    kelas = np.arange(n) % 2
    t = rng.uniform(0, np.pi, n)
    x = np.where(kelas == 1, 1 - np.cos(t), np.cos(t)) + rng.normal(0, noise, n)
    y = np.where(kelas == 1, 0.5 - np.sin(t), np.sin(t)) + rng.normal(0, noise, n)
    u = (x + 1.4) / 3.8 * AR
    v = (y + 0.9) / 2.3
    u = np.clip(u, 0.02, AR - 0.02)  # titik tetap di dalam bingkai gambar
    v = np.clip(v, 0.02, 0.98)
    return np.column_stack([u, v]), kelas


X_train, y_train = moons(160, 7, 0.35)
X_test, y_test = moons(800, 99, 0.35)


# ---------------------------------------------------------------- gambar bantu
def titik(X, y, r="1.7pt"):
    baris = []
    for i in range(len(X)):
        warna = C1 if y[i] == 1 else C0
        baris.append(f"\\fill[{warna},draw=white,line width=0.25pt] ({num(X[i,0])},{num(X[i,1])}) circle ({r});")
    return "\n".join(baris)


def warna_peluang(p):
    yakin = abs(p - 0.5) * 2
    warna = C1 if p >= 0.5 else C0
    return f"{warna}!{round(6 + 26 * yakin)}"


def arsir(prob, kolom=48, baris=30):
    hasil = []
    lebar = AR / kolom
    tinggi = 1 / baris
    for i in range(kolom):
        for j in range(baris):
            u = (i + 0.5) * lebar
            v = (j + 0.5) * tinggi
            p = prob(u, v)
            hasil.append(f"\\fill[{warna_peluang(p)}] ({num(i*lebar)},{num(j*tinggi)}) rectangle ++({num(lebar+0.002)},{num(tinggi+0.002)});")
    return "\n".join(hasil)


def bingkai(judul):
    return (f"\\draw[black!35] (0,0) rectangle ({AR},1);\n"
            f"\\node[font=\\scriptsize\\bfseries,anchor=south] at ({AR/2},1.01) {{{judul}}};")


def panel(isi, geser):
    return f"\\begin{{scope}}[shift={{({geser},0)}}]\n{isi}\n\\end{{scope}}"


def gambar2d(isi, satuan="4.3cm"):
    return f"\\begin{{tikzpicture}}[x={satuan},y={satuan}]\n{isi}\n\\end{{tikzpicture}}\n"


# ---------------------------------------------------------------- decision tree
def build_tree(X, y, depth, max_depth, pilih_fitur):
    p = y.mean()
    if depth >= max_depth or p == 0 or p == 1 or len(y) < 2:
        return {"leaf": True, "p": p}
    terbaik = None
    for f in pilih_fitur():
        urut = np.argsort(X[:, f])
        nilai = X[urut, f]
        kelas = y[urut]
        kiri1 = np.cumsum(kelas)[:-1]
        n_kiri = np.arange(1, len(y))
        n_kanan = len(y) - n_kiri
        kanan1 = kelas.sum() - kiri1
        pk = kiri1 / n_kiri
        pn = kanan1 / n_kanan
        gini = n_kiri * 2 * pk * (1 - pk) + n_kanan * 2 * pn * (1 - pn)
        gini[nilai[1:] == nilai[:-1]] = np.inf
        i = np.argmin(gini)
        if gini[i] < np.inf and (terbaik is None or gini[i] < terbaik[0]):
            terbaik = (gini[i], f, (nilai[i] + nilai[i + 1]) / 2)
    if terbaik is None:
        return {"leaf": True, "p": p}
    _, f, t = terbaik
    kiri = X[:, f] < t
    return {"leaf": False, "f": f, "t": t,
            "l": build_tree(X[kiri], y[kiri], depth + 1, max_depth, pilih_fitur),
            "r": build_tree(X[~kiri], y[~kiri], depth + 1, max_depth, pilih_fitur)}


def pred_tree(node, u, v):
    while not node["leaf"]:
        nilai = u if node["f"] == 0 else v
        node = node["l"] if nilai < node["t"] else node["r"]
    return node["p"]


def akurasi(prob, X, y):
    benar = 0
    for i in range(len(X)):
        tebak = 1 if prob(X[i, 0], X[i, 1]) >= 0.5 else 0
        benar += tebak == y[i]
    return benar / len(X)


def daun(node):
    if node["leaf"]:
        return 1
    return daun(node["l"]) + daun(node["r"])


def kotak_tree(node, u0, u1, v0, v1, out, garis):
    if node["leaf"]:
        out.append(f"\\fill[{warna_peluang(node['p'])}] ({num(u0)},{num(v0)}) rectangle ({num(u1)},{num(v1)});")
        return
    t = node["t"]
    if node["f"] == 0:
        garis.append(f"\\draw[black!45,line width=0.3pt] ({num(t)},{num(v0)})--({num(t)},{num(v1)});")
        kotak_tree(node["l"], u0, t, v0, v1, out, garis)
        kotak_tree(node["r"], t, u1, v0, v1, out, garis)
    else:
        garis.append(f"\\draw[black!45,line width=0.3pt] ({num(u0)},{num(t)})--({num(u1)},{num(t)});")
        kotak_tree(node["l"], u0, u1, v0, t, out, garis)
        kotak_tree(node["r"], u0, u1, t, v1, out, garis)


def dua_fitur():
    return [0, 1]


isi = []
for k, d in enumerate([3, 12]):
    pohon = build_tree(X_train, y_train, 0, d, dua_fitur)
    out = []
    garis = []
    kotak_tree(pohon, 0, AR, 0, 1, out, garis)
    prob = lambda u, v, pohon=pohon: pred_tree(pohon, u, v)
    latih = akurasi(prob, X_train, y_train) * 100
    val = akurasi(prob, X_test, y_test) * 100
    judul = f"max\\_depth = {d}: {daun(pohon)} daun, latih {latih:.0f}\\,\\%, validasi {val:.0f}\\,\\%"
    isi.append(panel("\n".join(out + garis) + "\n" + titik(X_train, y_train) + "\n" + bingkai(judul), k * 1.75))
simpan("tree", gambar2d("\n".join(isi)))

latih_list = []
val_list = []
for d in range(1, 13):
    pohon = build_tree(X_train, y_train, 0, d, dua_fitur)
    prob = lambda u, v, pohon=pohon: pred_tree(pohon, u, v)
    latih_list.append(f"({d},{akurasi(prob, X_train, y_train)*100:.1f})")
    val_list.append(f"({d},{akurasi(prob, X_test, y_test)*100:.1f})")
simpan("treekurva", f"""\\begin{{tikzpicture}}
\\begin{{axis}}[width=8.4cm,height=4.6cm,xmin=1,xmax=12,ymin=70,ymax=101,xlabel={{max\\_depth}},ylabel={{akurasi (\\%)}},
  grid=both,grid style={{black!8}},tick label style={{font=\\scriptsize}},label style={{font=\\scriptsize}},
  legend style={{font=\\tiny,at={{(0.98,0.04)}},anchor=south east,draw=black!30}},xtick={{1,2,...,12}}]
\\addplot[merah1,thick,mark=*,mark size=1.2pt] coordinates {{{' '.join(latih_list)}}};
\\addplot[biru1,thick,mark=square*,mark size=1.2pt] coordinates {{{' '.join(val_list)}}};
\\legend{{latih (dihafal),validasi (data baru)}}
\\end{{axis}}
\\end{{tikzpicture}}
""")

# ---------------------------------------------------------------- random forest
rng = np.random.default_rng(3)


def satu_fitur_acak():
    return [int(rng.integers(0, 2))]


hutan = []
for k in range(50):
    ambil = rng.integers(0, len(X_train), len(X_train))
    hutan.append(build_tree(X_train[ambil], y_train[ambil], 0, 12, satu_fitur_acak))

isi = []
for k, n in enumerate([1, 50]):
    def prob(u, v, n=n):
        total = 0
        for pohon in hutan[:n]:
            total += pred_tree(pohon, u, v)
        return total / n
    val = akurasi(prob, X_test, y_test) * 100
    angka("rfval" + ("satu" if n == 1 else "banyak"), val, 1)
    judul = f"{n} pohon: validasi {val:.0f}\\,\\%"
    isi.append(panel(arsir(prob) + "\n" + titik(X_train, y_train) + "\n" + bingkai(judul), k * 1.75))
simpan("forest", gambar2d("\n".join(isi)))

# ---------------------------------------------------------------- gradient boosting 1D
rng = np.random.default_rng(11)
gx = rng.uniform(0, 10, 70)
gy = np.sin(gx) * 1.6 + 0.25 * gx + rng.normal(0, 0.45, 70)
tx = rng.uniform(0, 10, 300)
ty = np.sin(tx) * 1.6 + 0.25 * tx + rng.normal(0, 0.45, 300)


def fit_reg(x, r, depth):
    if depth == 0 or len(x) < 6:
        return {"leaf": True, "v": r.mean()}
    urut = np.argsort(x)
    xs = x[urut]
    rs = r[urut]
    terbaik = None
    for k in range(3, len(xs) - 2):
        sse = ((rs[:k] - rs[:k].mean()) ** 2).sum() + ((rs[k:] - rs[k:].mean()) ** 2).sum()
        if terbaik is None or sse < terbaik[0]:
            terbaik = (sse, (xs[k - 1] + xs[k]) / 2)
    t = terbaik[1]
    kiri = x < t
    return {"leaf": False, "t": t, "l": fit_reg(x[kiri], r[kiri], depth - 1),
            "r": fit_reg(x[~kiri], r[~kiri], depth - 1)}


def pred_reg(node, x):
    hasil = np.zeros(len(x))
    for i in range(len(x)):
        n = node
        while not n["leaf"]:
            n = n["l"] if x[i] < n["t"] else n["r"]
        hasil[i] = n["v"]
    return hasil


def latih_boost(lr, ronde):
    awal = gy.mean()
    F = np.full(len(gx), awal)
    pohon = []
    for m in range(ronde + 1):
        t = fit_reg(gx, gy - F, 2)
        pohon.append(t)
        F = F + lr * pred_reg(t, gx)
    return awal, pohon


def model_boost(awal, pohon, lr, k, x):
    hasil = np.full(len(x), awal)
    for m in range(k):
        hasil = hasil + lr * pred_reg(pohon[m], x)
    return hasil


LR = 0.3
awal, pohon = latih_boost(LR, 300)
grid_x = np.linspace(0, 10, 301)


def koordinat(xs, ys):
    return " ".join(f"({x:.3f},{y:.3f})" for x, y in zip(xs, ys))


atas = []
bawah = []
for k in [1, 5, 50]:
    fx = model_boost(awal, pohon, LR, k, gx)
    kurva = model_boost(awal, pohon, LR, k, grid_x)
    sisa = "\n".join(f"\\draw[merah1!70,line width=0.35pt] (axis cs:{gx[i]:.3f},{gy[i]:.3f})--(axis cs:{gx[i]:.3f},{fx[i]:.3f});" for i in range(len(gx)))
    rmse = np.sqrt(((gy - fx) ** 2).mean())
    atas.append(f"""\\nextgroupplot[title={{{k} pohon, RMSE latih {koma(rmse, 2)}}}]
{sisa}
\\addplot[only marks,mark=*,mark size=0.9pt,black!55] coordinates {{{koordinat(gx, gy)}}};
\\addplot[biru1,very thick] coordinates {{{koordinat(grid_x, kurva)}}};""".replace(".", ".", 1))
    berikut = pred_reg(pohon[k], grid_x)
    bawah.append(f"""\\nextgroupplot[ymin=-3,ymax=3,height=3.1cm,ytick={{-2,0,2}}]
\\addplot[only marks,mark=*,mark size=0.9pt,merah1] coordinates {{{koordinat(gx, gy - fx)}}};
\\addplot[merah3,very thick] coordinates {{{koordinat(grid_x, berikut)}}};""")
simpan("boost", f"""\\begin{{tikzpicture}}
\\begin{{groupplot}}[group style={{group size=3 by 2,horizontal sep=0.55cm,vertical sep=0.45cm,
  xticklabels at=edge bottom,yticklabels at=edge left}},
  width=6.2cm,height=4.4cm,xmin=0,xmax=10,ymin=-2.6,ymax=4.8,grid=both,grid style={{black!8}},
  tick label style={{font=\\tiny}},title style={{font=\\scriptsize\\bfseries,yshift=-4pt}}]
{chr(10).join(atas)}
{chr(10).join(bawah)}
\\end{{groupplot}}
\\end{{tikzpicture}}
""")

kurva_rmse = []
for lr, warna, gaya in [(0.3, "merah1", ""), (0.05, "biru1", "")]:
    awal_lr, pohon_lr = latih_boost(lr, 300)
    titik_latih = []
    titik_val = []
    F_latih = np.full(len(gx), awal_lr)
    F_val = np.full(len(tx), awal_lr)
    for m in range(301):
        if m % 5 == 0:
            titik_latih.append(f"({m},{np.sqrt(((gy - F_latih) ** 2).mean()):.4f})")
            titik_val.append(f"({m},{np.sqrt(((ty - F_val) ** 2).mean()):.4f})")
        if m < 300:
            F_latih = F_latih + lr * pred_reg(pohon_lr[m], gx)
            F_val = F_val + lr * pred_reg(pohon_lr[m], tx)
    kurva_rmse.append(f"\\addplot[{warna},thick,densely dashed] coordinates {{{' '.join(titik_latih)}}};")
    kurva_rmse.append(f"\\addplot[{warna},thick] coordinates {{{' '.join(titik_val)}}};")
simpan("boostkurva", f"""\\begin{{tikzpicture}}
\\begin{{axis}}[width=8.6cm,height=4.8cm,xmin=0,xmax=300,ymin=0.2,ymax=1.6,xlabel={{jumlah pohon (n\\_estimators)}},ylabel={{RMSE}},
  grid=both,grid style={{black!8}},tick label style={{font=\\scriptsize}},label style={{font=\\scriptsize}},
  legend style={{font=\\tiny,at={{(0.98,0.98)}},anchor=north east,draw=black!30}},legend columns=2]
{chr(10).join(kurva_rmse)}
\\legend{{lr 0{{,}}3 latih,lr 0{{,}}3 validasi,lr 0{{,}}05 latih,lr 0{{,}}05 validasi}}
\\end{{axis}}
\\end{{tikzpicture}}
""")

# ---------------------------------------------------------------- regresi logistik
rng = np.random.default_rng(5)
kelas_lr = np.arange(120) % 2
X_lr = np.column_stack([np.where(kelas_lr == 1, 0.95, 0.62) + rng.normal(0, 0.17, 120),
                        np.where(kelas_lr == 1, 0.62, 0.38) + rng.normal(0, 0.13, 120)])
Z_lr = np.column_stack([(X_lr[:, 0] - 0.8) * 4, (X_lr[:, 1] - 0.5) * 4])
w = np.array([-0.6, 1.4])
b = 0.8
for langkah in range(600):
    p = 1 / (1 + np.exp(-(Z_lr @ w + b)))
    w = w - 0.4 * Z_lr.T @ (p - kelas_lr) / 120
    b = b - 0.4 * (p - kelas_lr).mean()


def prob_lr(u, v):
    return 1 / (1 + np.exp(-(w[0] * (u - 0.8) * 4 + w[1] * (v - 0.5) * 4 + b)))


def v_batas(u):
    return 0.5 - (w[0] * (u - 0.8) * 4 + b) / (w[1] * 4)


angka("lrwa", w[0])
angka("lrwb", w[1])
angka("lrc", b)
angka("lracc", akurasi(prob_lr, X_lr, kelas_lr) * 100, 1)
kiri = panel(arsir(prob_lr) + "\n\\begin{scope}\\clip (0,0) rectangle (1.6,1);\n"
             f"\\draw[black,thick] (0,{num(v_batas(0))})--({AR},{num(v_batas(AR))});\n\\end{{scope}}\n"
             + titik(X_lr, kelas_lr) + "\n" + bingkai("batas keputusan: skor = 0 (peluang 0,5)"), 0)
skor = Z_lr @ w + b
peluang = 1 / (1 + np.exp(-skor))
gs = np.linspace(-6, 6, 121)
titik_sig = []
for i in range(120):
    warna = C1 if kelas_lr[i] == 1 else C0
    s = float(np.clip(skor[i], -6, 6))
    titik_sig.append(f"\\fill[{warna},draw=white,line width=0.2pt] (axis cs:{s:.3f},{peluang[i]:.3f}) circle (1.1pt);")
simpan("logreg", gambar2d(kiri) + f"""\\hspace{{4mm}}\\begin{{tikzpicture}}
\\begin{{axis}}[width=5.6cm,height=4.9cm,xmin=-6,xmax=6,ymin=-0.03,ymax=1.03,xlabel={{skor $z$}},ylabel={{peluang kelas 1}},
  grid=both,grid style={{black!8}},tick label style={{font=\\scriptsize}},label style={{font=\\scriptsize}},
  title={{sigmoid: skor $\\to$ peluang}},title style={{font=\\scriptsize\\bfseries}},ytick={{0,0.5,1}},yticklabels={{0,0{{,}}5,1}}]
\\addplot[biru1,very thick] coordinates {{{koordinat(gs, 1/(1+np.exp(-gs)))}}};
{chr(10).join(titik_sig)}
\\end{{axis}}
\\end{{tikzpicture}}
""")

# ---------------------------------------------------------------- KNN
X_knn, y_knn = moons(120, 21, 0.2)
tanya = np.array([0.82, 0.5])


def prob_knn(u, v, k):
    jarak = np.hypot(X_knn[:, 0] - u, X_knn[:, 1] - v)
    dekat = np.argsort(jarak)[:k]
    return y_knn[dekat].mean()


isi = []
for j, k in enumerate([1, 15]):
    jarak = np.hypot(X_knn[:, 0] - tanya[0], X_knn[:, 1] - tanya[1])
    dekat = np.argsort(jarak)[:k]
    garis = []
    for i in dekat:
        garis.append(f"\\draw[black!50,line width=0.3pt] ({tanya[0]},{tanya[1]})--({num(X_knn[i,0])},{num(X_knn[i,1])});")
    lingkar = f"\\draw[black!50,densely dashed] ({tanya[0]},{tanya[1]}) circle ({num(jarak[dekat[-1]] + 0.012)});"
    suara1 = int(y_knn[dekat].sum())
    tebak = 1 if suara1 * 2 > k else 0
    bintang = (f"\\node[star,star points=5,star point ratio=2.3,fill={C1 if tebak else C0},draw=black,"
               f"inner sep=1.4pt,line width=0.4pt] at ({tanya[0]},{tanya[1]}) {{}};")
    judul = f"k = {k}: suara {k - suara1} lawan {suara1} $\\Rightarrow$ kelas {tebak}"
    isi.append(panel(arsir(lambda u, v, k=k: prob_knn(u, v, k), 40, 25) + "\n" + "\n".join(garis) + "\n" + lingkar
                     + "\n" + titik(X_knn, y_knn) + "\n" + bintang + "\n" + bingkai(judul), j * 1.75))
simpan("knn", gambar2d("\n".join(isi)))

# ---------------------------------------------------------------- KMeans
rng = np.random.default_rng(17)
pusat_asli = np.array([[0.35, 0.7], [0.45, 0.28], [1.1, 0.72], [1.2, 0.3]])
grup = np.arange(200) % 4
X_km = pusat_asli[grup] + rng.normal(0, 1, (200, 2)) * np.array([0.1, 0.085])
WARNA_K = [C0, C1, "hijau2", "ungu2"]
rng_init = np.random.default_rng(2)
pusat = X_km[rng_init.choice(200, 4, replace=False)].copy()


def panel_km(pusat, label, garis_ke_pusat, judul):
    out = []
    if garis_ke_pusat:
        for i in range(len(X_km)):
            c = pusat[label[i]]
            out.append(f"\\draw[{WARNA_K[label[i]]}!45,line width=0.2pt] ({num(X_km[i,0])},{num(X_km[i,1])})--({num(c[0])},{num(c[1])});")
    for i in range(len(X_km)):
        warna = "black!35" if label is None else WARNA_K[label[i]]
        out.append(f"\\fill[{warna},draw=white,line width=0.2pt] ({num(X_km[i,0])},{num(X_km[i,1])}) circle (1.6pt);")
    for j in range(len(pusat)):
        out.append(f"\\node[cross out,draw=white,line width=2.6pt,minimum size=7pt] at ({num(pusat[j,0])},{num(pusat[j,1])}) {{}};")
        out.append(f"\\node[cross out,draw={WARNA_K[j]},line width=1.3pt,minimum size=7pt] at ({num(pusat[j,0])},{num(pusat[j,1])}) {{}};")
    out.append(bingkai(judul))
    return "\n".join(out)


def tugaskan(pusat):
    jarak = ((X_km[:, None, :] - pusat[None, :, :]) ** 2).sum(axis=2)
    return jarak.argmin(axis=1)


def inersia(pusat, label):
    return ((X_km - pusat[label]) ** 2).sum()


label1 = tugaskan(pusat)
pusat_baru = pusat.copy()
for j in range(4):
    pusat_baru[j] = X_km[label1 == j].mean(axis=0)
langkah_km = 1
pusat_akhir = pusat_baru.copy()
label_akhir = tugaskan(pusat_akhir)
while True:
    pusat_lagi = pusat_akhir.copy()
    for j in range(4):
        pusat_lagi[j] = X_km[label_akhir == j].mean(axis=0)
    label_lagi = tugaskan(pusat_lagi)
    langkah_km += 1
    pusat_akhir = pusat_lagi
    if (label_lagi == label_akhir).all():
        break
    label_akhir = label_lagi
angka("kmiter", langkah_km, 0)
panel_a = panel_km(pusat, None, False, "0. pusat acak")
panel_b = panel_km(pusat, label1, True, "1. titik ikut pusat terdekat")
panel_c = panel_km(pusat_baru, label1, False, "2. pusat pindah ke tengah")
panel_d = panel_km(pusat_akhir, label_akhir, False, f"selesai (iterasi {langkah_km}), inertia {inersia(pusat_akhir, label_akhir):.2f}".replace(".", "{,}"))
simpan("kmeans", gambar2d(panel(panel_a, 0) + "\n" + panel(panel_b, 1.75) + "\n"
                          + "\\begin{scope}[shift={(0,-1.2)}]\n" + panel(panel_c, 0) + "\n" + panel(panel_d, 1.75) + "\n\\end{scope}"))

# ---------------------------------------------------------------- PCA
rng = np.random.default_rng(23)
sudut = np.deg2rad(28)
a = rng.normal(0, 0.26, 150)
bb = rng.normal(0, 0.06, 150)
X_pca = np.column_stack([0.8 + a * np.cos(sudut) - bb * np.sin(sudut), 0.5 + a * np.sin(sudut) + bb * np.cos(sudut)])
rata = X_pca.mean(axis=0)
kov = np.cov((X_pca - rata).T, bias=True)
nilai_eigen, vektor = np.linalg.eigh(kov)
arah_terbaik = vektor[:, 1]
total_var = np.trace(kov)
isi = []
for j, (arah, nama) in enumerate([(arah_terbaik, "arah PCA (komponen 1)"), (np.array([-arah_terbaik[1], arah_terbaik[0]]), "arah buruk (tegak lurus)")]):
    proyeksi = (X_pca - rata) @ arah
    simpan_persen = (proyeksi ** 2).mean() / total_var * 100
    out = [f"\\begin{{scope}}\\clip (0,0) rectangle ({AR},1);",
           f"\\draw[biru1,thick] ({num(rata[0]-arah[0]*2)},{num(rata[1]-arah[1]*2)})--({num(rata[0]+arah[0]*2)},{num(rata[1]+arah[1]*2)});",
           "\\end{scope}"]
    for i in range(150):
        kaki = rata + proyeksi[i] * arah
        out.append(f"\\draw[black!30,line width=0.2pt] ({num(X_pca[i,0])},{num(X_pca[i,1])})--({num(kaki[0])},{num(kaki[1])});")
    for i in range(150):
        out.append(f"\\fill[{C0},draw=white,line width=0.2pt] ({num(X_pca[i,0])},{num(X_pca[i,1])}) circle (1.5pt);")
    out.append(f"\\draw[black!35] (0.05,-0.14)--({AR-0.05},-0.14);")
    for i in range(150):
        kaki = rata + proyeksi[i] * arah
        out.append(f"\\fill[{C1}] ({num(kaki[0])},{num(kaki[1])}) circle (0.6pt);")
        out.append(f"\\fill[{C1}!80] ({num(AR/2 + proyeksi[i]*1.2)},-0.14) circle (0.8pt);")
    out.append(f"\\node[font=\\tiny,black!60,anchor=north] at ({AR/2},-0.17) {{hasil: 1 kolom}};")
    nama_persen = f"{simpan_persen:.0f}"
    out.append(bingkai(f"{nama}: menyimpan {nama_persen}\\,\\% sebaran"))
    isi.append(panel("\n".join(out), j * 1.75))
    angka("pcasatu" if j == 0 else "pcadua", simpan_persen, 0)
simpan("pca", gambar2d("\n".join(isi)))

# ---------------------------------------------------------------- Isolation Forest
rng = np.random.default_rng(31)
X_if = np.column_stack([0.78 + rng.normal(0, 0.14, 90), 0.48 + rng.normal(0, 0.11, 90)])
X_if = np.vstack([X_if, [[0.18, 0.86], [1.42, 0.16], [1.36, 0.88]]])
tengah = int(np.argmin(((X_if - [0.78, 0.48]) ** 2).sum(axis=1)))


def isolasi(idx, rng_cut, catat):
    sisa = np.arange(len(X_if))
    kotak = [0.0, AR, 0.0, 1.0]
    potong = []
    while len(sisa) > 1 and len(potong) < 60:
        f = int(rng_cut.integers(0, 2))
        lo = X_if[sisa, f].min()
        hi = X_if[sisa, f].max()
        if hi - lo < 1e-9:
            continue
        t = rng_cut.uniform(lo, hi)
        sisi = X_if[idx, f] < t
        if catat:
            if f == 0:
                potong.append((t, kotak[2], t, kotak[3]))
            else:
                potong.append((kotak[0], t, kotak[1], t))
        else:
            potong.append(None)
        if f == 0:
            if sisi:
                kotak[1] = t
            else:
                kotak[0] = t
        else:
            if sisi:
                kotak[3] = t
            else:
                kotak[2] = t
        sisa = sisa[(X_if[sisa, f] < t) == sisi]
    return potong, kotak


isi = []
for j, (idx, seed) in enumerate([(90, 4), (tengah, 4)]):
    potong, kotak = isolasi(idx, np.random.default_rng(seed), True)
    out = [f"\\fill[biru3!12] ({num(kotak[0])},{num(kotak[2])}) rectangle ({num(kotak[1])},{num(kotak[3])});"]
    for (u0, v0, u1, v1) in potong:
        out.append(f"\\draw[biru3!80!black,line width=0.45pt] ({num(u0)},{num(v0)})--({num(u1)},{num(v1)});")
    for i in range(len(X_if)):
        out.append(f"\\fill[{C0},draw=white,line width=0.2pt] ({num(X_if[i,0])},{num(X_if[i,1])}) circle (1.6pt);")
    out.append(f"\\draw[{C1},line width=1pt] ({num(X_if[idx,0])},{num(X_if[idx,1])}) circle (3pt);")
    nama = "titik terpencil" if j == 0 else "titik di tengah kerumunan"
    out.append(bingkai(f"{nama}: terisolasi dalam {len(potong)} potongan"))
    isi.append(panel("\n".join(out), j * 1.75))
simpan("iforest", gambar2d("\n".join(isi)))
rng_sim = np.random.default_rng(5)
total_luar = 0
total_tengah = 0
for ulang in range(300):
    total_luar += len(isolasi(90, rng_sim, False)[0])
    total_tengah += len(isolasi(tengah, rng_sim, False)[0])
angka("ifluar", total_luar / 300, 1)
angka("iftengah", total_tengah / 300, 1)

# ---------------------------------------------------------------- TF-IDF
dokumen = ["pengiriman cepat dan barang bagus",
           "barang rusak dan pengiriman lambat sekali",
           "pelayanan ramah dan barang bagus sekali",
           "barang bagus harga murah murah"]
kata_dok = []
for d in dokumen:
    kata_dok.append(d.split())
N = len(dokumen)
df = {}
for kata_list in kata_dok:
    for kata in set(kata_list):
        df[kata] = df.get(kata, 0) + 1
baris_tabel = []
dok4 = kata_dok[3]
hitung = {}
for kata in dok4:
    hitung[kata] = hitung.get(kata, 0) + 1
hasil = []
for kata in hitung:
    tf = hitung[kata] / len(dok4)
    idf = np.log((1 + N) / (1 + df[kata])) + 1
    hasil.append((tf * idf, kata, tf, idf))
hasil.sort(reverse=True)
for skor_kata, kata, tf, idf in hasil:
    lebar = skor_kata / hasil[0][0] * 2.6
    baris_tabel.append(f"\\textbf{{{kata}}} & {hitung[kata]}/{len(dok4)} = {koma(tf, 2)} & {df[kata]} dari {N} & "
                       f"{koma(idf, 2)} & {koma(skor_kata, 3)} & \\tikz\\fill[biru1] (0,0) rectangle ({lebar:.2f}cm,0.18cm);\\\\")
with open(f"{FIG_FOLDER}/tfidf_tabel.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(baris_tabel) + "\n\\bottomrule\n")

# ---------------------------------------------------------------- exponential smoothing
rng = np.random.default_rng(13)
t_es = np.arange(40)
y_es = 100 + 1.1 * t_es + 9 * np.sin(t_es / 3.2) + rng.normal(0, 6, 40)
y_es[22] += 25
garis_es = []
for alpha, warna in [(0.1, "biru1"), (0.8, "merah3")]:
    level = [y_es[0]]
    mae = 0
    for i in range(1, 40):
        mae += abs(y_es[i] - level[-1])
        level.append(alpha * y_es[i] + (1 - alpha) * level[-1])
    mae /= 39
    nama = "esmaekecil" if alpha == 0.1 else "esmaebesar"
    angka(nama, mae, 1)
    garis_es.append(f"\\addplot[{warna},very thick] coordinates {{{koordinat(t_es, level)}}};")
    garis_es.append(f"\\addlegendentry{{$\\alpha={koma(alpha, 1)}$, MAE {koma(mae, 1)}}}")
    garis_es.append(f"\\addplot[{warna},thick,densely dashed,forget plot] coordinates {{(39,{level[-1]:.2f}) (47,{level[-1]:.2f})}};")
simpan("es", f"""\\begin{{tikzpicture}}
\\begin{{axis}}[width=12cm,height=4.8cm,xmin=0,xmax=48,ymin=85,ymax=175,xlabel={{bulan}},ylabel={{penjualan}},
  grid=both,grid style={{black!8}},tick label style={{font=\\scriptsize}},label style={{font=\\scriptsize}},
  legend style={{font=\\tiny,at={{(0.02,0.98)}},anchor=north west,draw=black!30}}]
\\addplot[black!45,mark=*,mark size=0.9pt] coordinates {{{koordinat(t_es, y_es)}}};
\\addlegendentry{{data asli}}
{chr(10).join(garis_es)}
\\draw[black!50,densely dotted] (axis cs:39.5,85)--(axis cs:39.5,175);
\\node[font=\\tiny,black!60,anchor=north west] at (axis cs:40,172) {{ramalan}};
\\node[font=\\tiny,merah1,anchor=south] at (axis cs:22,{y_es[22]+1:.1f}) {{lonjakan}};
\\end{{axis}}
\\end{{tikzpicture}}
""")

# ---------------------------------------------------------------- matrix factorization
users = ["Ani", "Budi", "Citra", "Dodi", "Eka"]
films = ["Horor A", "Horor B", "Komedi A", "Komedi B", "Drama A", "Drama B"]
R = [[5, 4, 1, None, 2, None], [4, None, 2, 1, None, 2], [1, 2, 5, 4, None, 4],
     [None, 1, 4, 5, 4, None], [2, None, None, 4, 5, 5]]
rng = np.random.default_rng(9)
diketahui = []
for u in range(5):
    for i in range(6):
        if R[u][i] is not None:
            diketahui.append((u, i, R[u][i]))
mu = np.mean([r for _, _, r in diketahui])
bu = np.zeros(5)
bi = np.zeros(6)
P = rng.normal(0, 0.1, (5, 2))
Q = rng.normal(0, 0.1, (6, 2))
for epoch in range(3000):
    for u, i, r in diketahui:
        e = r - (mu + bu[u] + bi[i] + P[u] @ Q[i])
        bu[u] += 0.01 * (e - 0.05 * bu[u])
        bi[i] += 0.01 * (e - 0.05 * bi[i])
        pu = P[u].copy()
        P[u] += 0.01 * (e * Q[i] - 0.05 * P[u])
        Q[i] += 0.01 * (e * pu - 0.05 * Q[i])
baris_mf = []
for u in range(5):
    sel = [f"\\textbf{{{users[u]}}}"]
    for i in range(6):
        if R[u][i] is None:
            tebak = min(5, max(1, mu + bu[u] + bi[i] + P[u] @ Q[i]))
            sel.append(f"\\textcolor{{biru1}}{{\\textbf{{\\emph{{{tebak:.1f}}}}}}}".replace(".", "{,}"))
        else:
            sel.append(str(R[u][i]))
    baris_mf.append(" & ".join(sel) + "\\\\")
with open(f"{FIG_FOLDER}/mf_tabel.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(baris_mf) + "\n\\bottomrule\n")
angka("mfmu", mu)
angka("mfbu", bu[0])
angka("mfbi", bi[3])
angka("mfpq", P[0] @ Q[3])
angka("mfhasil", mu + bu[0] + bi[3] + P[0] @ Q[3])

# ---------------------------------------------------------------- graf + modularitas
rng = np.random.default_rng(6)
pusat_grup = [(0.55, 0.62), (1.45, 0.66), (1.0, 0.2)]
simpul = []
for g in range(3):
    for i in range(6):
        sudut_n = i / 6 * 2 * np.pi + g
        simpul.append((pusat_grup[g][0] + np.cos(sudut_n) * 0.2, pusat_grup[g][1] + np.sin(sudut_n) * 0.15, g))
sisi = []
for g in range(3):
    for i in range(6):
        for j in range(i + 1, 6):
            if rng.uniform() < 0.7:
                sisi.append((g * 6 + i, g * 6 + j))
sisi += [(0, 7), (3, 13), (9, 15), (5, 16)]


def modularitas(bagian):
    m = len(sisi)
    derajat = np.zeros(len(simpul))
    for a_, b_ in sisi:
        derajat[a_] += 1
        derajat[b_] += 1
    q = 0
    for c in set(bagian):
        dalam = 0
        for a_, b_ in sisi:
            if bagian[a_] == c and bagian[b_] == c:
                dalam += 1
        total_derajat = 0
        for i in range(len(simpul)):
            if bagian[i] == c:
                total_derajat += derajat[i]
        q += dalam / m - (total_derajat / (2 * m)) ** 2
    return q


bagian_grup = []
for s in simpul:
    bagian_grup.append(s[2])
angka("qgrup", modularitas(bagian_grup), 3)
angka("qsatu", modularitas([0] * len(simpul)), 3)
angka("jumlahsisi", len(sisi), 0)
WARNA_G = [C0, C1, "hijau2"]
out_g = []
for a_, b_ in sisi:
    out_g.append(f"\\draw[black!30,line width=0.6pt] ({num(simpul[a_][0])},{num(simpul[a_][1])})--({num(simpul[b_][0])},{num(simpul[b_][1])});")
for x_, y_, g in simpul:
    out_g.append(f"\\fill[{WARNA_G[g]},draw=white,line width=0.6pt] ({num(x_)},{num(y_)}) circle (3.2pt);")
simpan("graf", gambar2d("\n".join(out_g), "3.2cm"))

# ---------------------------------------------------------------- threshold
rng = np.random.default_rng(19)
kelas_th = (np.arange(70) < 25).astype(int)
p_th = 1 / (1 + np.exp(-(rng.normal(0, 1.2, 70) + np.where(kelas_th == 1, 1.1, -1.3))))
jitter = rng.uniform(-0.12, 0.12, 70)
titik_th = []
for i in range(70):
    warna = C1 if kelas_th[i] == 1 else C0
    y_pos = (0.75 if kelas_th[i] == 1 else 0.27) + jitter[i]
    benar = (p_th[i] >= 0.5) == (kelas_th[i] == 1)
    if benar:
        titik_th.append(f"\\fill[{warna},draw=white,line width=0.2pt] (axis cs:{p_th[i]:.3f},{y_pos:.3f}) circle (1.6pt);")
    else:
        titik_th.append(f"\\draw[{warna},fill=white,line width=0.6pt] (axis cs:{p_th[i]:.3f},{y_pos:.3f}) circle (1.6pt);")
simpan("threshold", f"""\\begin{{tikzpicture}}
\\begin{{axis}}[width=12cm,height=4.2cm,xmin=0,xmax=1,ymin=0,ymax=1,xlabel={{peluang kelas 1 dari model}},
  ytick={{0.27,0.75}},yticklabels={{asli 0,asli 1}},xtick={{0,0.25,0.5,0.75,1}},xticklabels={{0,0{{,}}25,0{{,}}5,0{{,}}75,1}},
  grid=major,grid style={{black!8}},tick label style={{font=\\scriptsize}},label style={{font=\\scriptsize}}]
\\fill[biru3!10] (axis cs:0.5,0) rectangle (axis cs:1,1);
\\draw[biru3!80!black,thick] (axis cs:0.5,0)--(axis cs:0.5,1);
\\node[font=\\tiny,biru3!80!black,anchor=north east] at (axis cs:0.99,0.99) {{ditebak kelas 1 $\\rightarrow$}};
\\node[font=\\tiny,biru3!80!black,anchor=north west] at (axis cs:0.01,0.99) {{$\\leftarrow$ ditebak kelas 0}};
{chr(10).join(titik_th)}
\\end{{axis}}
\\end{{tikzpicture}}
""")
baris_th = []
for ambang in [0.3, 0.5, 0.7]:
    tebak = (p_th >= ambang).astype(int)
    tp = int(((tebak == 1) & (kelas_th == 1)).sum())
    fp = int(((tebak == 1) & (kelas_th == 0)).sum())
    fn = int(((tebak == 0) & (kelas_th == 1)).sum())
    tn = int(((tebak == 0) & (kelas_th == 0)).sum())
    presisi = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * presisi * recall / (presisi + recall)
    baris_th.append(f"{ambang:.1f} & {tp} & {fp} & {fn} & {tn} & {presisi:.2f} & {recall:.2f} & \\textbf{{{f1:.2f}}}\\\\".replace(".", "{,}"))
with open(f"{FIG_FOLDER}/threshold_tabel.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(baris_th) + "\n\\bottomrule\n")

with open("angka.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(macros) + "\n")
print("selesai:", len(macros), "angka")
