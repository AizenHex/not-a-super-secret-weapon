"""Pahami data (M3): profil kolom (C1), temuan integritas (C2), train vs test (C3), teks (C4), relasi antartabel (C5),
cari dan sampel (B2, B3). Tanpa Qt.

Tiap temuan = dict: tingkat (2 penting, 1 perhatian, 0 info), grup, judul, rincian, `saring` (kode ekspresi polars
untuk melihat barisnya) dan `langkah` ((jenis, param) usulan langkah resep) bila ada. Satu cek yang gagal menjadi
temuan "cek gagal", tidak menjatuhkan yang lain."""
import re
import traceback

import polars as pl

from mesin import bantu
from mesin.langkah import PERLU_PY, PRESET_REGEX, lit, lits

NIL = ["", "-", "--", "na", "n/a", "nan", "null", "none", "?", "[]", "{}", "tidak ada"]
POLA_TEKS = {  # nama: (regex polars, preset langkah regex_ganti)
    "URL": (r"https?://\S+|www\.\S+", "URL"),
    "tag HTML": (r"<[a-zA-Z/][^<>]{0,200}>", "tag HTML"),
    "mention @": (r"(^|\s)@\w+", "mention @"),
    "hashtag #": (r"(^|\s)#\w+", "hashtag #"),
    "emoji": (r"[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]", "emoji / non-ASCII"),
    "huruf berulang 3+": ("(?i)(" + "|".join(f"{c}{{3,}}" for c in "abcdefghijklmnopqrstuvwxyz") + ")",
                          "huruf berulang (kerennnn → keren)"),
}
MOJIBAKE = r"‚Ä.|¬\S|Ã[\x{80}-\x{BF}]|Â[\x{80}-\x{BF}]|â€|\x{FFFD}"
SPESIAL = r"[\x00-\x08\x0B\x0C\x0E-\x1F\x{200B}-\x{200F}\x{2028}\x{2029}\x{FEFF}\x{A0}]"
DAFTAR = r"(?s)^\s*\[.*\]\s*$"
ANGKA = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
ANGKA_ID = r"^[+-]?\d{1,3}(\.\d{3})+(,\d+)?$|^[+-]?\d+,\d+$"


def T(tingkat, grup, judul, rincian="", saring=None, langkah=None):
    return {"tingkat": tingkat, "grup": grup, "judul": judul, "rincian": rincian, "saring": saring,
            "langkah": langkah}


def rb(n):
    return f"{n:,}".replace(",", ".")


def persen(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


# ---------- jenis kolom ----------
def bersarang(dt):
    return dt.is_nested() or dt == pl.Object


def kolom_teks(df):
    """Kolom teks bebas: banyak nilai unik, tidak pendek, bukan id/list."""
    out, buang = [], set(kolom_id(df)) | set(kolom_daftar(df))
    for c in df.columns:
        if df[c].dtype != pl.String or c in buang:
            continue
        s = df[c].drop_nulls()
        if s.len() and s.str.len_chars().median() >= 10 and s.n_unique() > max(50, 0.05 * s.len()):
            out.append(c)
    return out


def kolom_kategori(df, maks=2000):
    return [c for c in df.columns if df[c].dtype in (pl.String, pl.Categorical, pl.Boolean)
            and 1 < df[c].n_unique() <= max(50, min(maks, 0.05 * df.height))]


def kolom_id(df):
    return [c for c in df.columns if df.height > 20 and (df[c].dtype == pl.String or df[c].dtype.is_integer())
            and df[c].null_count() == 0 and df[c].n_unique() == df.height
            and (df[c].dtype != pl.String or df[c].str.len_chars().median() <= 40)]


def tebak_label(df, pasangan=None, contoh=None):
    """Kolom di train yang tidak ada di test; lalu kolom sample_submission (`contoh` = daftar kolomnya) selain id;
    terakhir nama khas label dengan sedikit nilai."""
    if pasangan is not None:
        beda = [c for c in df.columns if c not in pasangan.columns]
        if len(beda) == 1:
            return beda[0]
    if contoh:
        sama = [c for c in contoh if c in df.columns and c not in kolom_id(df)]
        if len(sama) == 1:
            return sama[0]
    for c in reversed(df.columns):
        if re.search(r"label|target|kelas|class|sentimen|kategori_target|^y$", c, re.I) and 1 < df[c].n_unique() <= 50:
            return c
    return None


# ---------- C1 profil ----------
def kuantil(s):
    s = s.drop_nulls()
    if not s.len():
        return {}
    return {k: s.quantile(q, "nearest") for k, q in
            [("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95), ("maks", 1)]}


def histogram(s, n=10):
    s = s.drop_nulls().cast(pl.Float64)
    if not s.len():
        return []
    lo, maks = s.min(), s.max()
    if lo == maks:
        return [(lo, maks, s.len())]
    hi = s.quantile(0.99)
    hi = maks if hi <= lo else hi  # ekor panjang (1% teratas) masuk batang terakhir supaya batang lain terbaca
    lebar = (hi - lo) / n
    b = ((s - lo) / lebar).floor().clip(0, n - 1).cast(pl.Int32).value_counts()
    hit = dict(zip(b[:, 0].to_list(), b["count"].to_list()))
    return [(lo + i * lebar, maks if i == n - 1 else lo + (i + 1) * lebar, hit.get(i, 0)) for i in range(n)]


def teratas(s, n=5):
    try:
        v = s.drop_nulls().value_counts(sort=True).head(n)
        return list(zip(v[v.columns[0]].to_list(), v["count"].to_list()))
    except Exception:
        return []


def profil(df):
    """Satu dict per kolom: tipe, null, kosong, spasi saja, unik, nilai teratas, kuantil + histogram."""
    out = []
    for c in df.columns:
        s = df[c]
        d = {"kolom": c, "tipe": str(s.dtype), "null": s.null_count(), "baris": df.height, "teratas": teratas(s)}
        try:
            d["unik"] = s.n_unique()
        except Exception:
            d["unik"] = None
        if s.dtype == pl.String:
            d["kosong"] = int((s == "").sum())
            d["spasi"] = int(((s.str.strip_chars() == "") & (s != "")).sum())
            p = s.str.len_chars()
            d["panjang"], d["hist"] = kuantil(p), histogram(p)
            d["kata"] = kuantil(s.str.count_matches(r"\S+"))
        elif s.dtype.is_numeric():
            d["angka"], d["hist"], d["rata"] = kuantil(s), histogram(s), s.mean()
            if s.dtype != pl.Boolean and (d["unik"] or 0) > 10:
                d["outlier"] = outlier_semua(s)
        elif s.dtype.is_temporal():
            d["rentang"] = (s.min(), s.max())
        out.append(d)
    return out


def outlier_semua(s):
    """Tiap cara di bantu.OUTLIER: (cara, faktor, bawah, atas, jumlah di luar batas). Gagal (mis. sd 0) = dilewati."""
    out = []
    for cara, f in bantu.OUTLIER.items():
        try:
            lo, hi = bantu.batas_outlier(s, cara)
        except (TypeError, ValueError):
            continue
        out.append((cara, f, lo, hi, int((~s.drop_nulls().cast(pl.Float64).is_between(lo, hi)).sum())))
    return out


def kolom_angka(df, label=None):
    """Kolom angka yang bermakna untuk outlier: bukan id, bukan label, bukan boolean, > 10 nilai unik."""
    idk = set(kolom_id(df))
    return [c for c, t in df.schema.items() if t.is_numeric() and c not in idk and c != label
            and df[c].n_unique() > 10]


PENANDA = {2147483647, -2147483648, 4294967295, 9223372036854775807, 999, 9999, 99999, 999999, 9999999, 99999999,
           -1, -9, -99, -999, -9999}


def cek_penanda(df, label=None):
    """Angka penanda kosong (2147483647 = batas int32, 9999, -1, ...): nilai ujung yang berulang dan bukan data."""
    per_kol = {}
    for c in kolom_angka(df, label):
        s = df[c].drop_nulls()
        p99 = abs(s.quantile(0.99) or 0)
        for v in {s.max(), s.min()}:
            n = int((s == v).sum())
            if n >= 5 and (v in PENANDA or abs(v) > 50 * max(p99, 1)):
                per_kol.setdefault(c, []).append((v, n))
    if not per_kol:
        return []
    rinci = "\n".join(f"{c}: " + ", ".join(f"{angka_pendek(v)} × {rb(n)}" for v, n in vs) for c, vs in per_kol.items())
    kode = "\n".join(f"df = df.with_columns(pl.when(pl.col({lit(c)}).is_in({[v for v, _ in vs]!r}))"
                     f".then(None).otherwise(pl.col({lit(c)})).alias({lit(c)}))" for c, vs in per_kol.items())
    saring = " | ".join(f"pl.col({lit(c)}).is_in({[v for v, _ in vs]!r})" for c, vs in per_kol.items())
    return [T(2, "outlier", f"Angka penanda tersamar di {len(per_kol)} kolom (bukan data asli)",
              rinci + "\nNilai ujung yang sama persis dan berulang biasanya kode 'tidak diisi' (2.147.483.647 = batas "
              "int32). Jadikan kosong dulu, baru tangani outlier; kalau tidak, rata-rata dan batas IQR ikut rusak.",
              saring, ("kode", {"isi": kode}))]


def cek_outlier(df, label=None):
    """Outlier IQR 1,5 per kolom angka, digabung jadi satu temuan supaya daftar tidak banjir."""
    baris, kol, saring = [], [], []
    for c in kolom_angka(df, label):
        lo, hi = bantu.batas_outlier(df[c], "IQR")
        n = int((~df[c].drop_nulls().cast(pl.Float64).is_between(lo, hi)).sum())
        if not n:
            continue
        kol.append(c)
        saring.append(f"~pl.col({lit(c)}).is_between({lo!r}, {hi!r})")
        baris.append((n, f"{c}: {rb(n)} baris ({persen(n, df.height)}) di luar {angka_pendek(lo)} s.d. "
                         f"{angka_pendek(hi)} · maks {angka_pendek(df[c].max())}"))
    if not kol:
        return []
    baris.sort(reverse=True)
    besar = baris[0][0] > 0.01 * df.height
    return [T(1 if besar else 0, "outlier", f"Outlier (IQR 1,5) di {len(kol)} kolom angka",
              "\n".join(b for _, b in baris) + "\nIQR menandai ekor panjang (uang, luas, harga) sebagai outlier walau "
              "nilainya sah; bandingkan dengan MAD dan persentil di tab Profil. Model pohon (LightGBM) tahan outlier, "
              "model linear dan KNN tidak. Batas langkah dihitung dari data latih lalu dipakai juga untuk test.",
              "pl.any_horizontal([" + ", ".join(saring) + "])",
              ("outlier", {"kolom": kol, "cara": "IQR", "faktor": "", "aksi": "potong ke batas (winsorize)"}))]


def angka_pendek(v):
    if v is None:
        return "-"
    v = float(v)
    return rb(round(v)) if abs(v) >= 100 or v.is_integer() else f"{v:.3g}".replace(".", ",")


# ---------- C2 integritas ----------
def _norm(e):
    return e.cast(pl.String).str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()


def cek_duplikat(df, label):
    out = []
    idk = kolom_id(df)
    kol = [c for c in df.columns if c not in idk and not bersarang(df[c].dtype)]
    if not kol or not df.height:
        return out
    dup = df.select(pl.struct(kol).is_duplicated()).to_series()
    if dup.sum():
        grup = df.filter(dup).select(kol).n_unique()
        out.append(T(2, "integritas", f"{rb(dup.sum() - grup)} salinan berlebih: {rb(dup.sum())} baris kembar persis dalam {rb(grup)} grup"
                     + (f" di luar kolom id ({', '.join(idk)})" if idk else ""),
                     "Baris sama di semua kolom. Dalam CV, salinan bisa jatuh di lipatan latih dan validasi sekaligus.",
                     f"pl.struct({lits(kol)}).is_duplicated()",
                     ("buang_duplikat", {"kolom": kol, "simpan": "pertama"})))
    teks = sorted(kolom_teks(df), key=lambda c: -(df[c].str.len_chars().mean() or 0))[:2]
    for t in teks:
        g = df.filter(pl.col(t).is_duplicated() & pl.col(t).is_not_null())
        n_norm = df.select(_norm(pl.col(t)).is_duplicated() & pl.col(t).is_not_null()).to_series().sum()
        if not g.height and not n_norm:
            continue
        lain = [c for c in df.columns if c != t and c not in idk and not bersarang(df[c].dtype)]
        beda = {}
        if g.height and lain:
            a = g.group_by(t).agg([pl.col(c).n_unique().alias(c) for c in lain])
            beda = {c: int((a[c] > 1).sum()) for c in lain}
        ngrup = g[t].n_unique() if g.height else 0
        rinci = []
        if beda:
            bedanya = sorted(((n, c) for c, n in beda.items() if n), reverse=True)
            samanya = [c for c, n in beda.items() if not n]
            if bedanya:
                rinci.append("Dalam grup, berbeda di: " + ", ".join(f"{c} ({rb(n)} grup)" for n, c in bedanya))
            if samanya:
                rinci.append("Selalu sama: " + ", ".join(samanya))
            kecuali = [c for c in df.columns if c in idk]
            if kecuali:
                n_kec = df.select(pl.struct(kol).is_duplicated()).to_series().sum()
                rinci.append(f"{rb(n_kec)} baris kembar persis kecuali {', '.join(kecuali)}.")
        if label and label in beda:
            rinci.append(f"Label bertentangan dalam {rb(beda[label])} grup.")
        if n_norm > g.height:
            rinci.append(f"Setelah huruf kecil + rapikan spasi: {rb(n_norm)} baris kembar (+{rb(n_norm - g.height)}).")
        rinci.append(f"CV harus dikelompokkan per {t} (GroupKFold) agar skor tidak bocor; atau buang duplikat.")
        tingkat = 2 if (label and beda.get(label)) or g.height > 0.01 * df.height else 1
        out.append(T(tingkat, "integritas", f"{t}: {rb(g.height - ngrup)} salinan berlebih ({rb(g.height)} baris berisi sama, {rb(ngrup)} grup)",
                     "\n".join(rinci), f"pl.col({lit(t)}).is_duplicated() & pl.col({lit(t)}).is_not_null()",
                     ("buang_duplikat", {"kolom": [t], "simpan": "pertama"})))
    return out


def cek_kolom(df):
    out = []
    daftar = set(kolom_daftar(df))
    nil_kolom, nil_nilai = {}, set()
    for c in df.columns:
        s = df[c]
        if s.n_unique() <= 1 and df.height > 1:
            out.append(T(1, "integritas", f"{c}: hanya satu nilai ({teratas(s, 1)[0][0] if teratas(s, 1) else 'null'})",
                         "Tidak membawa informasi untuk model.", None, ("buang_kolom", {"kolom": [c]})))
            continue
        if s.dtype != pl.String:
            continue
        st = s.str.strip_chars()
        nil = [x for x in NIL if not (c in daftar and x in ("[]", "{}"))]
        cocok = s.filter(st.str.to_lowercase().is_in(nil))
        if cocok.len():
            nil_kolom[c] = cocok.str.strip_chars().value_counts(sort=True)
            nil_nilai |= set(nil_kolom[c][c].to_list())
        isi = st.filter(st.is_not_null() & (st != ""))
        if not isi.len() or c in daftar:
            continue
        f_angka = isi.str.contains(ANGKA).mean()
        f_id = isi.str.contains(ANGKA_ID).mean()
        if 0.8 <= f_angka + f_id < 1:
            bukan = isi.filter(~isi.str.contains(ANGKA) & ~isi.str.contains(ANGKA_ID))
            contoh = ", ".join(repr(v) for v, _ in teratas(bukan, 5))
            out.append(T(1, "integritas", f"{c}: tipe campur ({persen(f_angka + f_id, 1)} angka, {rb(bukan.len())} bukan)",
                         f"Nilai bukan angka terbanyak: {contoh}. Kolom tetap teks agar tidak ada yang hilang diam-diam.",
                         f"~pl.col({lit(c)}).str.strip_chars().str.contains({lit(ANGKA)}) & pl.col({lit(c)}).is_not_null()",
                         ("ubah_tipe", {"kolom": [c], "tipe": "angka Indonesia (1.234,5)" if f_id > f_angka else "angka"})))
        elif f_angka + f_id == 0 or not isi.str.contains(r"^\d+$").all():
            contoh = isi.head(2000)
            f_tgl = pl.select(bantu.tanggal_campur(pl.lit(contoh))).to_series().is_not_null().mean()
            if f_tgl >= 0.8:
                tgl = df.select(bantu.tanggal_campur(pl.col(c))).to_series()
                out.append(T(0, "integritas", f"{c}: tanggal tersimpan sebagai teks ({tgl.min()} s.d. {tgl.max()})",
                             f"{persen(tgl.is_not_null().sum(), isi.len())} nilai terbaca sebagai tanggal.", None,
                             ("ubah_tipe", {"kolom": [c], "tipe": "tanggal"})))
    if nil_kolom:
        rinci = "\n".join(f"{c}: " + ", ".join(f"{v!r} {rb(n)}" for v, n in vc.iter_rows()) for c, vc in nil_kolom.items())
        out.append(T(1, "integritas", f"Penanda kosong tersamar di {len(nil_kolom)} kolom (bukan null)",
                     rinci + "\nPolars menganggapnya nilai biasa; model melihatnya sebagai kategori.",
                     f"pl.any_horizontal([pl.col(c).str.strip_chars().str.to_lowercase().is_in({lits(NIL)}) "
                     f"for c in {lits(list(nil_kolom))}])",
                     ("jadi_kosong", {"kolom": list(nil_kolom), "nilai": "\n".join(sorted(v for v in nil_nilai if v))})))
    return out


def cek_ejaan(df):
    """Nilai kategori yang sama setelah huruf kecil/spasi/tanda baca dibuang: Jakarta / JAKARTA / jakarta."""
    out = []
    for c in kolom_kategori(df):
        if df[c].dtype != pl.String:
            continue
        v = df[c].drop_nulls().value_counts(sort=True)
        v = v.with_columns(k=pl.col(c).str.to_lowercase().str.replace_all(r"[^\w]+", "").str.strip_chars())
        g = v.group_by("k", maintain_order=True).agg(pl.col(c), pl.col("count")).filter(pl.col(c).list.len() > 1)
        if not g.height:
            continue
        peta, contoh = [], []
        for vals, cnt in zip(g[c].to_list(), g["count"].to_list()):
            contoh.append(" / ".join(f"{x!r} ({rb(n)})" for x, n in zip(vals, cnt)))
            peta += [f"{x} => {vals[0]}" for x in vals[1:]]
        out.append(T(1, "integritas", f"{c}: {g.height} nilai ditulis beda-beda", "\n".join(contoh[:8]),
                     f"pl.col({lit(c)}).is_in({lits([p.split(' => ')[0] for p in peta])})",
                     ("ganti_nilai", {"kolom": [c], "peta": "\n".join(peta)})))
    return out


def kolom_daftar(df):
    out = []
    for c in df.columns:
        if df[c].dtype == pl.String:
            s = df[c].drop_nulls().head(5000)
            if s.len() and s.str.contains(DAFTAR).mean() >= 0.5:
                out.append(c)
    return out


# ---------- C4 teks ----------
def _ujung(e, awal, n):
    """Daftar frasa 2-4 kata di 4 posisi pertama/terakhir teks (unik per baris)."""
    potong = e.str.slice(0, 100) if awal else e.str.slice(-160)
    kata = potong.str.extract_all(r"\S+")
    if not awal:
        kata = kata.list.reverse()
    frasa = []
    for off in range(4):
        for k in range(2, 5):
            ambil = kata.list.slice(off, k)
            if not awal:
                ambil = ambil.list.reverse()
            frasa.append(pl.when(kata.list.len() >= off + k + (0 if awal else 1)).then(
                pl.lit(f"{k}|") + ambil.list.join(" ")))
    return pl.concat_list(frasa).list.drop_nulls().list.unique()


def boilerplate(df, c, awal):
    n = df.height
    f = df.select(_ujung(pl.col(c).fill_null(""), awal, n).alias("f")).explode("f").drop_nulls()
    vc = f.group_by("f").len().filter(pl.col("len") >= max(20, 0.02 * n)).sort("len", descending=True)
    hasil = []
    for fr, jml in vc.iter_rows():
        teks = fr.split("|", 1)[1]
        # jendela yang tumpang tindih dari frasa yang sama: jumlah hampir sama + berbagi pasangan kata → simpan terpanjang
        for i, (t2, j2) in enumerate(hasil):
            if abs(j2 - jml) <= 0.1 * j2 and _bigram(teks) & _bigram(t2):
                if len(teks) > len(t2):
                    hasil[i] = (teks, j2)
                break
        else:
            hasil.append((teks, jml))
    # buang potongan yang terkandung di frasa lain yang muncul hampir sebanyak itu
    hasil = [(t, j) for t, j in hasil if not any(t != t2 and t in t2 and abs(j - j2) <= 0.1 * j for t2, j2 in hasil)]
    return hasil[:4]


def _bigram(teks):
    w = teks.split(" ")
    return set(zip(w, w[1:]))


def _pola_frasa(teks):
    return r"\s+".join(re.escape(w) for w in teks.split(" "))


def cek_teks(df):
    out = []
    n = df.height
    daftar = kolom_daftar(df)
    if daftar:  # satu temuan untuk semua kolom sejenis, bukan satu per kolom
        out.append(T(1, "teks", f"List Python tersimpan sebagai teks di {len(daftar)} kolom: {', '.join(daftar)}",
                     "\n".join(f"{c}: {df[c].drop_nulls()[0]!r:.100}" for c in daftar)
                     + "\n'[]' berarti kosong. Urai jadi list agar bisa dihitung/di-explode.", None,
                     ("list_teks", {"kolom": daftar})))
    moji = {}
    for c in [c for c in df.columns if df[c].dtype == pl.String]:
        k = int(df[c].str.contains(MOJIBAKE).sum())
        if k:
            moji[c] = k
    if moji:
        kol = sorted(moji, key=lambda c: -moji[c])
        seq = pl.concat([df[c].drop_nulls().str.extract_all(MOJIBAKE).explode() for c in kol]).drop_nulls()
        seq = seq.value_counts(sort=True).head(8)
        out.append(T(2 if max(moji.values()) > 0.01 * n else 1, "teks",
                     f"Mojibake di {len(kol)} kolom (terbanyak {kol[0]}: {rb(moji[kol[0]])} baris)",
                     "Teks UTF-8 yang pernah terbaca sebagai MacRoman/cp1252.\nPer kolom: "
                     + ", ".join(f"{c} {rb(moji[c])}" for c in kol) + "\nUrutan rusak terbanyak: "
                     + ", ".join(f"{v!r} {rb(k)}" for v, k in seq.iter_rows()),
                     f"pl.any_horizontal([pl.col(c).str.contains({lit(MOJIBAKE)}) for c in {lits(kol)}])",
                     ("kode", {"isi": f"df = df.with_columns([perbaiki_mojibake(pl.col(c)) for c in {lits(kol)}])"})))
    for c in kolom_teks(df):
        s = df[c].drop_nulls()
        for awal in (True, False):
            for teks, jml in boilerplate(df, c, awal):
                pola = (r"^(\S+\s+){0,3}?" + _pola_frasa(teks) + r"\s*") if awal else (r"\s*" + _pola_frasa(teks) + r"(\s+\S+){0,3}\s*$")
                out.append(T(1, "teks", f"{c}: {'awalan' if awal else 'akhiran'} berulang “{teks}” di {rb(jml)} baris "
                             f"({persen(jml, n)})", "Boilerplate sumber berita; bisa menjadi jalan pintas model "
                             "(sumber → label) atau sekadar derau.", f"pl.col({lit(c)}).str.contains({lit(pola)})",
                             ("regex_ganti", {"kolom": [c], "pola": pola, "ganti": ""})))
        for nama, (pola, preset) in POLA_TEKS.items():
            k = s.str.contains(pola).sum()
            if k >= max(5, 0.005 * n):
                pg, gg = PRESET_REGEX[preset]
                out.append(T(0, "teks", f"{c}: {nama} di {rb(k)} baris ({persen(k, n)})", "",
                             f"pl.col({lit(c)}).str.contains({lit(pola)})",
                             ("regex_ganti", {"kolom": [c], "preset": preset, "pola": pg, "ganti": gg})))
        huruf = s.str.replace_all(r"[^A-Za-z]", "")
        besar = ((huruf.str.len_chars() >= 10) & (huruf == huruf.str.to_uppercase())).sum()
        if besar >= max(5, 0.005 * n):
            out.append(T(0, "teks", f"{c}: {rb(besar)} baris HURUF BESAR semua", "",
                         f"pl.col({lit(c)}).str.replace_all('[^A-Za-z]', '').str.len_chars().ge(10) & "
                         f"(pl.col({lit(c)}).str.replace_all('[^A-Za-z]', '') == pl.col({lit(c)}).str.replace_all('[^A-Za-z]', '').str.to_uppercase())",
                         ("rapikan_teks", {"kolom": [c], "nfkc": True, "kecil": True, "spasi": True})))
        sp = s.str.contains(SPESIAL).sum()
        if sp:
            out.append(T(1, "teks", f"{c}: karakter tak terlihat/kontrol di {rb(sp)} baris",
                         "Zero-width, BOM, NBSP, atau karakter kontrol: dua teks yang tampak sama jadi berbeda.",
                         f"pl.col({lit(c)}).str.contains({lit(SPESIAL)})",
                         ("regex_ganti", {"kolom": [c], "pola": SPESIAL, "ganti": " "})))
        p = s.str.len_chars()
        pendek = (s.str.strip_chars().str.len_chars() < 3).sum()
        out.append(T(1 if pendek else 0, "teks", f"{c}: panjang median {rb(int(p.median()))} huruf, p95 "
                     f"{rb(int(p.quantile(.95)))}, maks {rb(p.max())}" + (f"; {rb(pendek)} hampir kosong (< 3)" if pendek else ""),
                     "Panjang token kira-kira 1,3–1,6 × jumlah kata untuk IndoBERT; teks di atas MAX_LENGTH terpotong.",
                     f"pl.col({lit(c)}).str.strip_chars().str.len_chars() < 3" if pendek else None))
    return out


# ---------- label ----------
def cek_label(df, label):
    if not label or label not in df.columns:
        return []
    out = []
    vc = df[label].value_counts(sort=True)
    n = df.height
    sebaran = ", ".join(f"{v} {persen(k, n)}" for v, k in vc.iter_rows())
    minor = vc["count"].min() / n if vc.height else 0
    out.append(T(1 if minor < 0.2 else 0, "label", f"Label {label}: {sebaran}",
                 ("Timpang: pakai stratified split, metrik makro (F1-macro), atau bobot kelas." if minor < 0.2 else "")
                 + (f"\n{rb(df[label].null_count())} label kosong." if df[label].null_count() else ""),
                 f"pl.col({lit(label)}).is_null()" if df[label].null_count() else None))
    mayor = vc["count"].max() / n if vc.height else 0
    nilai = set(vc[label].drop_nulls().to_list())
    for c in df.columns:
        if c == label or df[c].dtype != df[label].dtype:
            continue
        lain = set(df[c].drop_nulls().unique().head(100).to_list())
        if not lain or len(lain & nilai) < max(1, len(nilai) // 2):
            continue
        sama = (df[c] == df[label]).sum()
        if sama / n > mayor:
            out.append(T(1, "label", f"{c} sama dengan label di {persen(sama, n)} baris (tebak kelas terbanyak "
                         f"{persen(mayor, 1)})", "Fitur kuat, bukan kunci jawaban (tidak 100%). Cek apakah tersedia "
                         "juga di test.", f"pl.col({lit(c)}) != pl.col({lit(label)})"))
    return out


def temuan(df, label=None):
    """C2 + C4 + label untuk satu tabel."""
    out = []
    for f in (lambda: cek_duplikat(df, label), lambda: cek_kolom(df), lambda: cek_ejaan(df), lambda: cek_teks(df),
              lambda: cek_penanda(df, label), lambda: cek_outlier(df, label), lambda: cek_label(df, label)):
        try:
            out += f()
        except Exception as e:
            out.append(T(1, "cek gagal", f"Satu cek gagal: {type(e).__name__}: {e}", traceback.format_exc()))
    return urut(out)


def urut(out):
    return sorted(out, key=lambda t: -t["tingkat"])


# ---------- C3 train vs test ----------
def banding(latih, uji, label=None, nama_uji="test"):
    out = []
    try:
        for c in kolom_teks(latih):
            if c not in uji.columns or uji[c].dtype != pl.String:
                continue
            ada = uji[c].is_in(latih[c].drop_nulls().implode()) & uji[c].is_not_null()
            n = int(ada.sum())
            n_norm = int(uji.select(_norm(pl.col(c)).is_in(latih.select(_norm(pl.col(c))).to_series().drop_nulls().implode())
                                    & pl.col(c).is_not_null()).to_series().sum())
            if not n and not n_norm:
                continue
            rinci = []
            if label and label in latih.columns:
                kon = latih.filter(pl.col(c).is_in(uji[c].filter(ada).implode())).group_by(c).agg(
                    pl.col(label).n_unique().alias("u"))
                konflik = int((kon["u"] > 1).sum())
                rinci.append(f"Label train untuk teks itu: {rb(konflik)} konflik dari {rb(kon.height)} teks"
                             + (" → label train bisa dipakai langsung (tanya panitia apakah boleh)." if not konflik else "."))
            if n_norm > n:
                rinci.append(f"Setelah huruf kecil + rapikan spasi: {rb(n_norm)} baris.")
            rinci.append("Skor validasi acak terlalu optimistis bila duplikat tidak dikelompokkan.")
            out.append(T(2, "train-test", f"{c}: {rb(n)} dari {rb(uji.height)} baris {nama_uji} ({persen(n, uji.height)}) "
                         "persis ada di train", "\n".join(rinci)))
        for c in kolom_kategori(latih):
            if c not in uji.columns or c == label or bersarang(uji[c].dtype):
                continue
            u = uji[c].drop_nulls().cast(pl.String)  # test bisa terbaca dengan tipe lain (mis. Int64 vs String)
            baru = u.filter(~u.is_in(latih[c].drop_nulls().cast(pl.String).implode()))
            if baru.len():
                out.append(T(1, "train-test", f"{c}: {baru.n_unique()} nilai di {nama_uji} tidak ada di train "
                             f"({rb(baru.len())} baris)", "Contoh: " + ", ".join(repr(v) for v, _ in teratas(baru, 5))))
        for c in kolom_teks(latih):
            if c in uji.columns and uji[c].dtype == pl.String and uji[c].drop_nulls().len():
                a, b = latih[c].str.len_chars().median(), uji[c].str.len_chars().median()
                if a and b and abs(a - b) / max(a, b) > 0.2:
                    out.append(T(1, "train-test", f"{c}: panjang median train {rb(int(a))} vs {nama_uji} {rb(int(b))}"))
        for c in latih.columns:
            if c not in uji.columns:
                continue
            ra, rb_ = _rentang_tanggal(latih[c]), _rentang_tanggal(uji[c])
            if ra and rb_:
                acak = ra == rb_ or (rb_[0] <= ra[1] and ra[0] <= rb_[1])
                out.append(T(0, "train-test", f"{c}: train {ra[0]} s.d. {ra[1]}, {nama_uji} {rb_[0]} s.d. {rb_[1]}",
                             "Rentang tumpang tindih: split acak, bukan per waktu; validasi acak wajar." if acak else
                             "Test sesudah train: validasi harus per waktu (train awal → validasi akhir)."))
    except Exception as e:
        out.append(T(1, "cek gagal", f"Banding train-test gagal: {type(e).__name__}: {e}", traceback.format_exc()))
    return urut(out)


def _rentang_tanggal(s):
    if s.dtype.is_temporal():
        return (s.min(), s.max()) if s.drop_nulls().len() else None
    if s.dtype != pl.String or not s.drop_nulls().len() or s.drop_nulls().str.contains(r"^\d+$").all():
        return None
    t = pl.select(bantu.tanggal_campur(pl.lit(s))).to_series()
    return (t.min(), t.max()) if t.is_not_null().mean() >= 0.8 else None


# ---------- C5 relasi ----------
def relasi(tabel, maks_kolom=40):
    """Kolom bernama sama di dua tabel: unik di mana, % nilai cocok, jenis hubungan."""
    out, nama = [], list(tabel)
    for i, a in enumerate(nama):
        for b in nama[i + 1:]:
            A, B = tabel[a], tabel[b]
            for c in [c for c in A.columns if c in B.columns][:maks_kolom]:
                try:
                    x, y = A[c], B[c]
                    if bersarang(x.dtype) or bersarang(y.dtype):
                        continue
                    if x.dtype != y.dtype:
                        x, y = x.cast(pl.String), y.cast(pl.String)
                    ux, uy = x.drop_nulls().is_unique().all(), y.drop_nulls().is_unique().all()
                    if not (ux or uy):  # dua-duanya berulang: kolom kategori biasa, bukan kunci
                        continue
                    cx = x.drop_nulls().is_in(y.drop_nulls().implode()).mean() or 0
                    cy = y.drop_nulls().is_in(x.drop_nulls().implode()).mean() or 0
                    if cx == 0 and cy == 0:
                        continue
                    # langkah usulan dipasang di sisi "banyak": tabel fakta mengambil kolom tabel dimensi (m:1)
                    if ux and uy:
                        jenis, milik, ambil, cakup, val = ("1:1", a, b, cx, "1:1") if cx >= cy else ("1:1", b, a, cy, "1:1")
                    elif uy:
                        jenis, milik, ambil, cakup, val = f"{a} → {b} banyak:1", a, b, cx, "m:1"
                    else:
                        jenis, milik, ambil, cakup, val = f"{b} → {a} banyak:1", b, a, cy, "m:1"
                    tipe = "" if A[c].dtype == B[c].dtype else f" Tipe beda ({A[c].dtype} vs {B[c].dtype}): samakan sebelum join."
                    yatim = f" {persen(1 - cakup, 1)} nilai {milik}.{c} tidak punya pasangan di {ambil}." if cakup < 0.99 else ""
                    t = T(1 if tipe or yatim else 0, "relasi", f"{a} ↔ {b} lewat {c}: {jenis}",
                          f"{persen(cx, 1)} nilai {a}.{c} ada di {b}; {persen(cy, 1)} nilai {b}.{c} ada di {a}.{tipe}{yatim}",
                          None, ("join", {"tabel": ambil, "kiri": [c], "cara": "left", "validasi": val}))
                    t["tabel"], t["antara"] = milik, (a, b)
                    out.append(t)
                except Exception as e:
                    out.append(T(1, "cek gagal", f"Relasi {a}/{b}.{c}: {type(e).__name__}: {e}"))
    return out


# ---------- B2 cari, B3 sampel ----------
SEMUA = "(semua kolom teks)"


def ekspresi_cari(mode, kolom, isi, df):
    """Kode ekspresi polars (string) untuk baris yang cocok; dipakai untuk saring tampilan dan langkah resep."""
    if mode == "SQL":
        return f"pl.sql_expr({lit(isi)}).fill_null(False)"
    if mode == "kode polars":
        return isi
    kol = [c for c in df.columns if df[c].dtype == pl.String] if kolom == SEMUA else [kolom]

    def satu(k):
        if mode == "teks":
            return f"pl.col({lit(k)}).cast(pl.String).str.to_lowercase().str.contains({lit(isi.lower())}, literal=True)"
        if PERLU_PY.search(isi):
            return f"cocok_regex_py(pl.col({lit(k)}), {lit(isi)})"
        return f"pl.col({lit(k)}).cast(pl.String).str.contains({lit(isi)})"

    if len(kol) == 1:
        return satu(kol[0]) + ".fill_null(False)"
    return "pl.any_horizontal([" + ", ".join(satu(k) for k in kol) + "]).fill_null(False)"


def evaluasi(kode):
    return eval(kode, {"pl": pl, **{k: v for k, v in vars(bantu).items() if not k.startswith("__")}})


def cari(df, kode, label=None):
    """(baris cocok, jumlah per label) untuk ekspresi dari ekspresi_cari / saring temuan."""
    hasil = df.filter(evaluasi(kode))
    per = hasil[label].value_counts(sort=True).rows() if label and label in hasil.columns else []
    return hasil, per


def sampel(df, n=1000, seed=42):
    return df.sample(min(n, df.height), seed=seed) if df.height > n else df


def per_label(df, label, n=20, seed=42):
    return df.sample(fraction=1.0, shuffle=True, seed=seed).group_by(label, maintain_order=True).head(n).sort(
        label, maintain_order=True)
