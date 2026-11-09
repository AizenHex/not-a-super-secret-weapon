"""Tampilan profil kolom, temuan, dan hasil model: HTML untuk QTextBrowser + teks ringkas untuk tabel profil.
Semua warna diambil dari `W()` (ikut tema terang/gelap Windows), tidak ada warna mati di HTML."""
from html import escape

from mesin import langkah
from mesin.temuan import persen, rb

NAMA_TINGKAT = {2: "Penting", 1: "Perhatian", 0: "Info"}


# ---------- warna ----------
def gelap():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:  # uji.py di python tanpa Qt: pakai palet terang
        return False
    app = QApplication.instance()
    return bool(app) and app.palette().window().color().lightness() < 128


def W():
    """Palet HTML. Teks redup >= 4,5:1 di latar panel pada kedua tema."""
    if gelap():
        return {"redup": "#a3a9b2", "kode": "#16191d", "kode_teks": "#e3e6ea", "garis": "#40464f",
                "penting": "#ff7b72", "perhatian": "#e8a94f", "ok": "#6fcf86", "info": "#a3a9b2",
                "batang": "#6ea8e6", "panas": (110, 168, 230), "dasar": (37, 39, 43), "terang_di_panas": 0.45}
    return {"redup": "#5b6068", "kode": "#f1f3f5", "kode_teks": "#1f2328", "garis": "#d6dae0",
            "penting": "#b3261e", "perhatian": "#94570a", "ok": "#1e7a34", "info": "#5b6068",
            "batang": "#3d6fa8", "panas": (52, 101, 164), "dasar": (255, 255, 255), "terang_di_panas": 0.55}


def tingkat(t):
    """(nama, warna) tingkat temuan untuk pohon temuan dan rincian."""
    w = W()
    return NAMA_TINGKAT[t], {2: w["penting"], 1: w["perhatian"], 0: w["info"]}[t]


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(x)) for x in rgb)


def _campur(f):
    """Warna sel heatmap: dari latar (f=0) ke warna panas (f=1)."""
    w = W()
    return _hex([a + (b - a) * f for a, b in zip(w["dasar"], w["panas"])])


def redup(teks):
    return f"<span style='color:{W()['redup']}'>{teks}</span>"


def kode(teks):
    """Blok kode yang terbaca di kedua tema: latar dan teksnya sendiri, baris panjang dibungkus."""
    w = W()
    return (f"<table width='100%' cellspacing='0' cellpadding='8' style='margin-top:4px'><tr><td bgcolor='{w['kode']}'>"
            f"<p style='white-space:pre-wrap; font-family:Consolas,monospace; color:{w['kode_teks']}; margin:0'>"
            f"{escape(teks)}</p></td></tr></table>")


def batang_warna():
    return W()["batang"]


# ---------- profil ----------
def angka(v):
    if isinstance(v, float):
        return rb(round(v)) if (v.is_integer() or abs(v) >= 100) and abs(v) < 1e15 else f"{v:.4g}".replace(".", ",")
    return rb(v) if isinstance(v, int) else str(v)


def kosong(d):
    """Jumlah sel tanpa isi: null + '' + spasi saja."""
    return d["null"] + d.get("kosong", 0) + d.get("spasi", 0)


def ringkas(d):
    """Satu kalimat untuk kolom Ringkasan di tabel profil."""
    n = d["baris"] or 1
    if d["teratas"] and d["unik"] is not None and d["unik"] <= 20:
        return " · ".join(f"{str(v)[:25]} {persen(c, n)}" for v, c in d["teratas"])
    if d.get("panjang"):
        p, k = d["panjang"], d["kata"]
        return f"panjang median {angka(p['median'])} huruf (p95 {angka(p['p95'])}, maks {angka(p['maks'])}) · " \
               f"median {angka(k['median'])} kata"
    if d.get("angka"):
        a = d["angka"]
        o = next((x[4] for x in d.get("outlier", []) if x[0] == "IQR"), 0)
        return f"min {angka(a['min'])} · median {angka(a['median'])} · maks {angka(a['maks'])} · " \
               f"rata-rata {angka(d['rata'])}" + (f" · outlier IQR {rb(o)}" if o else "")
    if d.get("rentang"):
        return f"{d['rentang'][0]} s.d. {d['rentang'][1]}"
    return " · ".join(f"{str(v)[:25]} ({rb(c)})" for v, c in d["teratas"][:3])


def sebaran(d):
    """Angka untuk grafik mini: jumlah tiap kategori (≤ 20 nilai) atau isi tiap rentang histogram."""
    if d["teratas"] and d["unik"] is not None and d["unik"] <= 20:
        return [c for _, c in d["teratas"]]
    return [c for _, _, c in d.get("hist") or []]


def _batang(c, maks, lebar=220):
    w = max(1, round(lebar * c / maks)) if c and maks else 0
    isi = f"<td bgcolor='{W()['batang']}' width='{w}' height='12'></td>" if w else ""
    return f"<table cellspacing='0' cellpadding='0'><tr>{isi}<td width='{lebar - w}'></td></tr></table>"


def _baris(*sel, kanan=()):
    return "<tr>" + "".join(f"<td style='white-space:nowrap; padding:3px 12px 3px 0' {'align=right' if i in kanan else ''}>{x}</td>"
                            for i, x in enumerate(sel)) + "</tr>"


def _tabel(kepala, isi, kanan=(), bungkus=()):
    """Tabel rapat; kolom di `bungkus` boleh pindah baris (teks panjang), sisanya nowrap."""
    w = W()
    h = "<table cellspacing='0'><tr>" + "".join(
        f"<td style='white-space:nowrap; padding:2px 12px 4px 0; color:{w['redup']}' {'align=right' if i in kanan else ''}>{k}</td>"
        for i, k in enumerate(kepala)) + "</tr>"
    for r in isi:
        h += "<tr>" + "".join(
            f"<td style='{'' if i in bungkus else 'white-space:nowrap; '}padding:3px 12px 3px 0' "
            f"{'align=right' if i in kanan else ''}>{x}</td>" for i, x in enumerate(r)) + "</tr>"
    return h + "</table>"


def html_kolom(d):
    n = d["baris"] or 1
    h = [f"<h2>{escape(d['kolom'])}</h2><p>{redup(escape(d['tipe']) + ' · ' + rb(d['baris']) + ' baris')}</p>",
         "<table cellspacing='0'>"]
    h.append(_baris("Null", f"{rb(d['null'])}", persen(d["null"], n), kanan=(1, 2)))
    if "kosong" in d:
        h.append(_baris("Teks kosong ''", rb(d["kosong"]), persen(d["kosong"], n), kanan=(1, 2)))
        h.append(_baris("Spasi saja", rb(d["spasi"]), persen(d["spasi"], n), kanan=(1, 2)))
    h.append(_baris("Nilai unik", rb(d["unik"]) if d["unik"] is not None else "-",
                    persen(d["unik"], n) if d["unik"] is not None else "", kanan=(1, 2)))
    h.append("</table>")
    for k, judul in (("panjang", "Panjang (huruf)"), ("kata", "Jumlah kata"), ("angka", "Nilai")):
        if d.get(k):
            h.append(f"<h3>{judul}</h3>" + _tabel(list(d[k]), [[f"<b>{angka(v)}</b>" for v in d[k].values()]]))
    if d.get("rata") is not None:
        h.append(f"<p>Rata-rata <b>{angka(d['rata'])}</b></p>")
    if d.get("outlier"):
        h.append("<h3>Outlier per cara</h3>")
        h.append(_tabel(["Cara", "Batas bawah", "Batas atas", "Di luar", ""],
                        [(f"<b>{c}</b> {redup(str(f).replace('.', ','))}", angka(lo), angka(hi), rb(k), persen(k, n))
                         for c, f, lo, hi, k in d["outlier"]], kanan=(1, 2, 3, 4)))
        h.append(f"<p>{redup('IQR dan z peka pada ekor panjang (uang, luas); MAD dan persentil lebih tahan. Tangani '
                             'lewat Resep → + Langkah → Tangani outlier: batas dihitung dari data latih.')}</p>")
    if d.get("rentang"):
        h.append(f"<h3>Rentang</h3><p><b>{d['rentang'][0]}</b> s.d. <b>{d['rentang'][1]}</b></p>")
    hist = d.get("hist") or []
    if hist:
        maks = max(c for _, _, c in hist) or 1
        h.append(f"<h3>Sebaran {'panjang' if d.get('panjang') else 'nilai'}</h3><table cellspacing='0'>")
        for lo, hi, c in hist:
            h.append(_baris(f"{angka(lo)} – {angka(hi)}", _batang(c, maks, 100), rb(c), kanan=(2,)))
        h.append("</table>")
    if d["teratas"]:
        maks = d["teratas"][0][1] or 1
        h.append("<h3>Nilai terbanyak</h3><table cellspacing='0'>")
        for v, c in d["teratas"]:
            h.append(_baris(escape(str(v)[:40]), _batang(c, maks, 100), rb(c), persen(c, n), kanan=(2, 3)))
        h.append("</table>")
    return "".join(h)


def html_temuan(t):
    nama, warna = tingkat(t["tingkat"])
    h = [f"<p><span style='color:{warna}'><b>{nama}</b></span> "
         + redup("· " + escape(t["grup"]) + (f" · tabel {escape(t['tabel'])}" if t.get("tabel") else "")) + "</p>",
         f"<h3>{escape(t['judul'])}</h3>"]
    if t["rincian"]:
        h.append("".join(f"<p>{escape(p)}</p>" for p in t["rincian"].split("\n") if p.strip()))
    if t["langkah"]:
        h.append("<h4>Langkah usulan</h4>"
                 + redup("<p>Tekan <b>Jadikan langkah…</b> untuk membuka form yang sudah terisi; kode ini yang akan "
                         "dijalankan dan diekspor.</p>")
                 + kode(langkah.buat(*t["langkah"])["kode"]))
    return "".join(h)


# ---------- hasil tugas model (M4) ----------
JUDUL = {"tabular": "Baseline tabular (LightGBM)", "baseline": "Baseline teks + label mencurigakan",
         "kata": "Kata khas per label", "acak": "Uji label acak", "adversarial": "Adversarial train vs test",
         "submisi": "Validasi submission", "fold": "Ekspor kolom fold", "bocor": "Cek bocor per kolom",
         "asosiasi": "Peta asosiasi + pola kosong"}


def d3(v):
    return f"{v:.3f}".replace(".", ",")


def _teks(r):
    return redup(escape(r[:160]))


def ringkas_hasil(r):
    """Satu frasa untuk daftar riwayat."""
    h = r.get("hasil") or {}
    if r["jenis"] in ("baseline", "tabular") and "macro_f1" in h:
        return f"macro-F1 {d3(h['macro_f1'])}"
    if r["jenis"] == "tabular":
        return f"RMSE {d3(h['rmse'])}"
    if r["jenis"] == "acak":
        return f"macro-F1 acak {d3(h['macro_f1'])}"
    if r["jenis"] == "adversarial":
        return f"AUC {d3(h['auc'])}"
    if r["jenis"] == "kata":
        return f"{len(h['kelas'])} label"
    if r["jenis"] == "fold":
        return f"{len(h['per_fold'])} fold"
    if r["jenis"] == "bocor":
        n = sum(_curiga(h, x) for x in h["kolom"])
        return f"{n} kolom curiga" if n else "tidak ada yang curiga"
    if r["jenis"] == "asosiasi":
        return f"{len(h['kolom'])} kolom"
    n = sum(t == "galat" for t, _ in h["cek"])
    return f"{n} galat" if n else "lulus"


def html_hasil(r):
    h = r.get("hasil")
    detik = f" · {h['detik']} s" if h and "detik" in h else ""
    out = [f"<h2>{JUDUL[r['jenis']]}</h2><p>{redup(escape(r['ket']) + detik)}</p>"]
    if r["status"] != "ok":
        pesan = {"antre": "Menunggu giliran.", "jalan": "Sedang berjalan; kemajuan di LOG kiri bawah.",
                 "batal": "Dibatalkan.", "gagal": "Gagal. Ekor log:"}[r["status"]]
        out.append(f"<p>{pesan}</p>")
        if r["status"] == "gagal":
            out.append(kode("\n".join(r["log"][-25:])))
        return "".join(out)
    out += {"tabular": _html_tabular, "baseline": _html_baseline, "adversarial": _html_adv, "acak": _html_acak,
            "kata": _html_kata, "submisi": _html_submisi, "fold": _html_fold, "bocor": _html_bocor,
            "asosiasi": _html_asosiasi}[r["jenis"]](h)
    return "".join(out)


def html_banding(r, sama):
    """Skor run ini vs run lain (jenis + tabel sama) beserta beda resepnya: langkah mana yang menaikkan skor."""
    if r["status"] != "ok" or r["jenis"] not in ("tabular", "baseline") or len(sama) < 2:
        return ""
    ini = r.get("resep", [])
    isi = []
    for x in sama:
        tambah = [s for s in x["resep"] if s not in ini]
        kurang = [s for s in ini if s not in x["resep"]]
        beda = "; ".join(["+ " + escape(s[:50]) for s in tambah] + ["− " + escape(s[:50]) for s in kurang])
        isi.append([("<b>" + x["jam"] + "</b>") if x is r else x["jam"], ringkas_hasil(x), rb(len(x["resep"])),
                    "run ini" if x is r else beda or redup("resep sama")])
    return ("<h3>Banding versi resep</h3><p>" + redup("run lain dengan tugas dan tabel sama; + = langkah yang ada di "
                                                    "run itu tapi tidak di run ini, − = sebaliknya") + "</p>"
            + _tabel(["Jam", "Skor", "Langkah", "Beda resep dengan run ini"], isi, kanan=(2,), bungkus=(3,)))


def _matriks(h):
    """Matriks kebingungan sebagai heatmap: warna = porsi baris (recall), angka = jumlah."""
    w = W()
    kelas, m = h["kelas"], h["matriks"]
    out = ["<h3>Matriks kebingungan</h3><p>" + redup("baris = label asli, kolom = tebakan; warna = porsi tiap baris, "
                                                       "diagonal = benar") + "</p>",
           "<table cellspacing='1' cellpadding='5'><tr><td></td>"
           + "".join(f"<td align='center' style='white-space:nowrap; color:{w['redup']}'>{escape(k[:10])}</td>" for k in kelas)
           + f"<td align='right' style='white-space:nowrap; color:{w['redup']}'>recall</td></tr>"]
    for i, (k, baris) in enumerate(zip(kelas, m)):
        tot = sum(baris) or 1
        sel = []
        for j, v in enumerate(baris):
            f = v / tot
            warna = _campur(f)
            teks = "#ffffff" if f > w["terang_di_panas"] else "inherit"
            isi = f"<b>{rb(v)}</b>" if i == j else rb(v)
            sel.append(f"<td align='right' bgcolor='{warna}' style='white-space:nowrap; color:{teks}'>{isi}</td>")
        out.append(f"<tr><td style='white-space:nowrap; padding-right:8px'><b>{escape(k[:14])}</b></td>" + "".join(sel)
                   + f"<td align='right' style='white-space:nowrap'>{d3(baris[i] / tot)}</td></tr>")
    out.append("</table>")
    return out


def _per_label(h):
    lap = h["laporan"]
    return ["<h3>Per label</h3>", _tabel(["Label", "Presisi", "Recall", "F1", "", "Baris"],
                                         [(f"<b>{escape(k)}</b>", d3(lap[k]["precision"]), d3(lap[k]["recall"]),
                                           d3(lap[k]["f1-score"]), _batang(lap[k]["f1-score"], 1, 120),
                                           rb(int(lap[k]["support"]))) for k in h["kelas"]], kanan=(1, 2, 3, 5))]


def _skor_utama(h, metode):
    return (f"<p style='font-size:large'><b>macro-F1 {d3(h['macro_f1'])}</b> &nbsp; akurasi {d3(h['akurasi'])}</p>"
            f"<p>{redup('acuan: tebak kelas terbanyak = akurasi ' + d3(h['kelas_terbanyak']) + ' · validasi ' + escape(h.get('validasi', '')) + ' · per fold ' + ', '.join(d3(x) for x in h['per_fold']) + ' · ' + metode)}</p>")


def _html_tabular(h):
    w = W()
    if h["klasifikasi"]:
        out = [_skor_utama(h, f"LightGBM di {h['n_fitur']} kolom")]
    else:
        out = [f"<p style='font-size:large'><b>RMSE {d3(h['rmse'])}</b> &nbsp; MAE {d3(h['mae'])} &nbsp; "
               f"R² {d3(h['r2'])}</p><p>" + redup(f"acuan: tebak rata-rata = RMSE {d3(h['rmse_rata'])} · validasi "
                                                   f"{escape(h['validasi'])} · per fold "
                                                   + ", ".join(d3(x) for x in h["per_fold"])) + "</p>"]
    if h.get("kembar_lintas"):
        out.append(f"<p style='color:{w['perhatian']}'><b>{rb(h['kembar_lintas'])} baris kembar (fitur sama persis) "
                   "jatuh di fold berbeda</b>: skor CV bisa terlalu optimis. Buang duplikat atau pakai validasi grup.</p>")
    if h.get("dilewati"):
        out.append(f"<p>{redup('Kolom dilewati: ' + escape(', '.join(h['dilewati'])))}</p>")
    maks = h["penting"][0][1] if h["penting"] else 1
    out.append("<h3>Kolom paling berpengaruh</h3><p>" + redup("gain LightGBM dijumlah semua fold; kolom tanpa "
                                                              "pengaruh kandidat dibuang") + "</p>")
    out.append(_tabel(["Kolom", "", "Porsi"], [(escape(k[:50]), _batang(v, maks, 160), f"{100 * v:.1f}%".replace(".", ","))
                                               for k, v in h["penting"]], kanan=(2,)))
    if h["klasifikasi"]:
        out += _per_label(h)
        if len(h["kelas"]) <= 15:
            out += _matriks(h)
    return out


def _pct(v):
    return f"{100 * v:.1f}%".replace(".", ",")


def _html_fold(h):
    w = W()
    out = [f"<p style='font-size:large'><b>{len(h['per_fold'])} fold</b> &nbsp; {escape(h['validasi'])}</p>",
           f"<p>fold.csv: kolom <b>{escape(h['kunci'])}</b> + <b>fold</b> (0 s.d. {len(h['per_fold']) - 1}"
           + (f"; {rb(h['tanpa_fold'])} baris -1 = hanya latih" if h["tanpa_fold"] else "") + ")</p>",
           kode(h["path"])]
    kepala = ["Fold", "Baris"]
    isi = []
    if h["klasifikasi"] and len(h["kelas"]) <= 10:
        kepala += [escape(k[:8]) for k in h["kelas"]]
        isi.append(["semua", rb(sum(f["n"] for f in h["per_fold"]))] + [_pct(v) for v in h["sebaran_semua"]])
        isi += [[str(f["fold"]), rb(f["n"])] + [_pct(v) for v in f["sebaran"]] for f in h["per_fold"]]
    elif h["klasifikasi"]:
        kepala.append("Selisih porsi kelas terbesar")
        isi = [[str(f["fold"]), rb(f["n"]), _pct(max(abs(a - b) for a, b in zip(f["sebaran"], h["sebaran_semua"])))]
               for f in h["per_fold"]]
    else:
        kepala.append(f"Rata-rata {escape(h['label'])}")
        isi = [["semua", "", d3(h["sebaran_semua"])]] + [[str(f["fold"]), rb(f["n"]), d3(f["sebaran"])]
                                                        for f in h["per_fold"]]
    out += ["<h3>Isi tiap fold</h3><p>" + redup("porsi label tiap fold harus mirip baris 'semua'") + "</p>",
            _tabel(kepala, isi, kanan=tuple(range(1, len(kepala))))]
    k = h["kunci"]
    sambung = (f'df = df.join(fold, on="{k}", how="left")' if k != "baris"
               else 'df = df.with_columns(fold["fold"])  # urutan baris harus sama dengan tabel di Laya')
    out += ["<h3>Pakai di notebook</h3><p>" + redup("salin fold.csv ke folder kerja; semua model memakai split yang "
                                                    "sama, jadi OOF bisa di-ensemble") + "</p>",
            kode(f'import polars as pl\nfold = pl.read_csv(r"{h["path"]}")\n{sambung}\n'
                 f'for k in range({len(h["per_fold"])}):\n'
                 f'    latih, val = df.filter(pl.col("fold") != k), df.filter(pl.col("fold") == k)\n'
                 f'# scikit-learn: PredefinedSplit(df["fold"].to_numpy())'),
            f"<p style='color:{w['redup']}'>Fold dihitung dari tabel di tahap resep yang terbuka saat tugas jalan.</p>"]
    return out


def _curiga(h, x):
    return (x["skor"] >= 0.9 or x.get("f1_kelas_maks", 0) >= 0.95) if h["klasifikasi"] else x["skor"] >= 0.9


def _html_bocor(h):
    w = W()
    curiga = [x for x in h["kolom"] if _curiga(h, x)]
    out = [f"<p style='font-size:large'><b>{len(curiga)} kolom curiga</b> dari {len(h['kolom'])}</p>" if curiga else
           "<p style='font-size:large'><b>Tidak ada kolom yang sendirian menebak label</b></p>",
           "<p>" + redup(f"Satu model kecil per kolom ({escape(h['validasi'])}, 3 fold pertama). Acuan tebak kelas "
                         f"terbanyak: {h['metrik']} {d3(h['acuan'])}. Curiga = {h['metrik']} ≥ 0,9"
                         + (" atau satu kelas F1 ≥ 0,95" if h["klasifikasi"] else "") + ".") + "</p>"]
    if curiga:
        out.append(f"<p style='color:{w['perhatian']}'>Cek tiap kolom curiga: apakah nilainya sudah diketahui sebelum "
                   "label ditentukan? Kalau dibuat dari label (atau sesudahnya), buang. Kalau sah (aturan bisnis, "
                   "mis. penerima beasiswa = golongan 0), pertahankan dan sebut di presentasi.</p>")
    kepala = ["", "Kolom", h["metrik"], ""] + (["Kelas paling tertebak", "F1 kelas", ""] if h["klasifikasi"] else [])
    maks = max([x["skor"] for x in h["kolom"]] + [1e-9])
    isi = []
    for x in sorted(h["kolom"], key=lambda x: (not _curiga(h, x), -x["skor"])):
        tanda = f"<b style='color:{w['penting']}'>curiga</b>" if _curiga(h, x) else ""
        isi.append([tanda, escape(x["kolom"][:36]), d3(x["skor"]), _batang(max(x["skor"], 0), maks, 90)]
                   + ([escape(x["kelas_maks"]), d3(x["f1_kelas_maks"]), _batang(x["f1_kelas_maks"], 1, 90)]
                      if h["klasifikasi"] else []))
    out.append(_tabel(kepala, isi, kanan=(2, 5) if h["klasifikasi"] else (2,)))
    if h.get("dilewati"):
        out.append(f"<p>{redup('Kolom dilewati: ' + escape(', '.join(h['dilewati'])))}</p>")
    return out


def _heat(kol, m, judul, ket):
    """Heatmap persegi 0..1; kolom diberi nomor supaya tabel tidak melebar."""
    w = W()
    out = [f"<h3>{judul}</h3><p>{redup(ket)}</p><table cellspacing='1' cellpadding='3'><tr><td></td>"
           + "".join(f"<td align='center' style='font-size:8pt; color:{w['redup']}'>{i}</td>"
                     for i in range(1, len(kol) + 1)) + "</tr>"]
    for i, (c, baris) in enumerate(zip(kol, m), 1):
        sel = []
        for j, v in enumerate(baris, 1):
            f = min(abs(v), 1.0)
            teks = "#ffffff" if f > w["terang_di_panas"] else "inherit"
            angka = round(100 * f) if i != j and f >= 0.3 else ""  # angka kecil hanya mengotori; warna sudah cukup
            sel.append(f"<td align='center' width='22' bgcolor='{_campur(f)}' "
                       f"style='white-space:nowrap; font-size:8pt; color:{teks}'>{angka}</td>")
        out.append(f"<tr><td style='white-space:nowrap; padding-right:8px'>{i} {escape(c[:24])}</td>" + "".join(sel) + "</tr>")
    out.append("</table>")
    return out


def _html_asosiasi(h):
    out = [f"<p style='font-size:large'><b>{len(h['kolom'])} kolom</b></p><p>"
           + redup(f"{rb(h['baris'])} baris; angka-angka: |Spearman|, selain itu: Cramér's V (angka dibagi 10 ember). "
                   "0 = tidak terkait, 1 = sama persis") + "</p>", "<h3>Pasangan paling terkait</h3>",
           _tabel(["Kolom", "Kolom", "", "Nilai", "Cara"],
                  [(escape(a[:30]), escape(b[:30]), _batang(v, 1, 80), d3(v), cara.split("'")[0])
                   for v, a, b, cara in h["pasangan"]], kanan=(3,)),
           "<p>" + redup("≥ 0,95: hampir duplikat, satu bisa dibuang (mis. kota dan lokasi). Kolom yang sangat terkait "
                         "dengan label: cek kebocoran (tugas Cek bocor per kolom).") + "</p>"]
    out += _heat(h["kolom"], h["matriks"], "Peta asosiasi", "angka sel = nilai x 100; nomor kolom = nomor baris")
    k = h["kosong"]
    if k["matriks"]:
        out += _heat(k["kolom"], k["matriks"], "Pola kosong bersama",
                     "korelasi penanda kosong: tinggi = kolom-kolom ini kosong di baris yang sama (satu sumber "
                     "data hilang; satu kolom penanda kosong cukup)")
    elif k["kolom"]:
        out.append(f"<p>{redup('Hanya satu kolom yang punya nilai kosong.')}</p>")
    return out


def _html_baseline(h):
    w = W()
    out = [_skor_utama(h, "TF-IDF kata 1–2 + regresi logistik")]
    if h.get("pendek"):
        out.append(f"<p style='color:{w['perhatian']}'><b>Kolom teks ini isinya pendek/berulang (kategori atau "
                   "angka, bukan kalimat).</b> Model teks tidak cocok; pakai <b>Baseline tabular</b>.</p>")
    out += _per_label(h)
    if len(h["kelas"]) <= 15:
        out += _matriks(h)
    out.append("<h3>Kata paling berbobot per label</h3>")
    out.append(_tabel(["Label", "Kata"], [(f"<b>{escape(k)}</b>", escape(", ".join(v[:12]) or "-"))
                                          for k, v in h["kata"].items()], bungkus=(1,)))
    if "curiga_galat" in h:
        out.append(f"<p style='color:{w['perhatian']}'>Label mencurigakan dilewati: {escape(h['curiga_galat'])}.</p>")
        return out
    out.append(f"<h3>Label mencurigakan ({rb(h['curiga_total'])} baris)</h3>")
    if h["macro_f1"] < 0.8:
        out.append(f"<p style='color:{w['perhatian']}'>Model masih lemah, jadi banyak baris di sini hanya salah tebak "
                   "model, bukan salah label. Periksa dari yang paling atas; jangan buang massal.</p>")
    out.append(_tabel(["Baris/id", "Label → tebakan", "Yakin", "Teks"],
                      [(escape(c["id"] if c["id"] is not None else str(c["baris"])),
                        f"{escape(c['label'])} → <b>{escape(c['tebak'])}</b>", d3(c["yakin"]), _teks(c["teks"]))
                       for c in h["curiga"][:30]], kanan=(2,), bungkus=(3,)))
    return out


def _html_adv(h):
    a = h["auc"]
    arti = ("sebaran train dan test mirip; CV bisa dipercaya" if a < 0.55 else
            "train dan test agak berbeda; lihat kata pembeda" if a < 0.7 else
            "train dan test jelas berbeda: skor CV bisa terlalu optimis. Lihat kata pembeda dan baris test yang paling "
            "beda, pertimbangkan validasi pada baris train yang mirip test")
    return [f"<p style='font-size:large'><b>AUC {d3(a)}</b></p><p>{arti}.</p><p>"
            + redup("Model ditanya: baris ini dari train atau test? AUC 0,5 = tidak bisa membedakan.") + "</p>",
            _tabel(["Lebih sering di test", "Lebih sering di train"],
                   [(escape(u), escape(l)) for u, l in zip(h["kata_uji"][:15], h["kata_latih"][:15])]),
            "<h3>Baris test yang paling beda dari train</h3>",
            _tabel(["Baris test", "p(test)", "Teks"], [(rb(x["baris"]), d3(x["p"]), _teks(x["teks"]))
                                                       for x in h["paling_beda"][:15]], kanan=(0, 1), bungkus=(2,))]


def _html_acak(h):
    w = W()
    acuan = max(h["tebakan_seragam"], h["tebakan_terbanyak"])
    bocor = h["macro_f1"] > acuan + 0.05
    vonis = (f"<span style='color:{w['penting']}'><b>Jauh di atas tebakan: ada yang bocor</b></span> (teks memuat "
             "label, duplikat lintas fold, urutan baris, dsb.). Cek temuan duplikat dan kolom yang terlalu cocok dengan "
             "label." if bocor else f"<span style='color:{w['ok']}'><b>Wajar</b></span>: dengan label acak model tidak "
             "bisa lebih baik dari menebak, jadi skor baseline bukan hasil kebocoran pipeline.")
    return [f"<p style='font-size:large'><b>macro-F1 dengan label diacak {d3(h['macro_f1'])}</b></p><p>"
            + redup(f"acuan tebakan: seragam {d3(h['tebakan_seragam'])}, kelas terbanyak {d3(h['tebakan_terbanyak'])}"
                    " · per fold " + ", ".join(d3(x) for x in h["per_fold"])) + f"</p><p>{vonis}</p>"]


def _html_kata(h):
    out = ["<p>" + redup("Log-odds dengan prior Dirichlet (skor z): kata yang khas untuk satu label dibanding label "
                         "lain. n = kemunculan di label itu.") + "</p>"]
    for k in h["kelas"]:
        ks = [x for x in h["kata"][k] if x["z"] > float("-inf")][:15]
        maks = max([x["z"] for x in ks] + [1e-9])
        out.append(f"<h3>{escape(k)}</h3>")
        out.append(_tabel(["Kata", "z", "", "n"], [(escape(x["kata"]), f"{x['z']:.1f}".replace(".", ","),
                                                    _batang(max(x["z"], 0), maks, 120), rb(x["n"])) for x in ks],
                          kanan=(1, 3)))
    return out


def _html_submisi(h):
    w = W()
    warna = {"galat": w["penting"], "peringatan": w["perhatian"], "ok": w["ok"]}
    return ["<table cellspacing='0'>" + "".join(f"<tr><td style='white-space:nowrap; padding:3px 10px 3px 0; color:{warna[t]}'><b>{t}</b></td>"
                                                f"<td style='padding:3px 0'>{escape(p)}</td></tr>" for t, p in h["cek"])
            + "</table>"]
