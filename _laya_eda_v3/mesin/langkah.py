"""Jenis langkah resep (D1). Tiap jenis = form isian + pembuat kode polars. Kode yang dibuat bekerja pada `df`
(tabel saat ini) dan `LATIH` (tabel latih sebelum langkah ini; median/modus/peta dihitung dari sini), boleh
memakai `T["nama"]` (tabel lain) dan fungsi di bantu.py. Kode yang sama dijalankan app dan ditulis ke ekspor."""
import re

# tipe isian: kolom, kolom_banyak (kosong = semua kolom teks), tabel, tabel_banyak, teks, teks_panjang,
# pilihan (daftar), angka, cek, kolom_opsional ("-" = tidak dipakai)
PERLU_PY = re.compile(r"\(\?<?[=!]|\\[1-9]|\(\?P=|\\g<")  # lookaround/backreference: mesin regex polars tidak bisa

PRESET_REGEX = {
    "URL": (r"https?://\S+|www\.\S+", " "),
    "tag HTML": (r"<[^>]+>", " "),
    "mention @": (r"@\w+", " "),
    "hashtag #": (r"#\w+", " "),
    "angka": (r"\d+", " "),
    "tanda baca": (r"[^\w\s]", " "),
    "emoji / non-ASCII": (r"[^\x00-\x7F]+", " "),
    "huruf berulang (kerennnn → keren)": (r"(\w)\1{2,}", r"\1"),
    "spasi ganda": (r"\s+", " "),
    "baris baru": (r"[\r\n]+", " "),
}
TIPE = ["angka", "angka Indonesia (1.234,5)", "bulat", "tanggal", "tanggal-waktu", "kategori", "teks", "boolean"]
NIL_BAWAAN = "-\nNA\nN/A\nnull\nNone\n?\n[]\n{}"
AGG = ["count", "n_unique", "sum", "mean", "median", "min", "max", "std", "first", "last"]


def lit(s):
    """Literal Python yang terbaca: raw string kalau aman, kalau tidak repr."""
    s = str(s)
    if not any(c in s for c in "\"\r\n") and not s.endswith("\\"):
        return 'r"' + s + '"' if "\\" in s else '"' + s + '"'
    return repr(s)


def lits(xs):
    return "[" + ", ".join(lit(x) for x in xs) + "]"


def baris(teks):
    return [b for b in (x.strip() for x in str(teks or "").splitlines()) if b and not b.startswith("#")]


def peta(teks):
    """`lama => baru` per baris (komentar `# ...` di akhir baris dibuang)."""
    hasil = {}
    for b in baris(teks):
        b = re.sub(r"\s+#.*$", "", b)
        if "=>" not in b:
            raise ValueError(f"baris tanpa '=>': {b}")
        lama, baru = b.split("=>", 1)
        hasil[lama.strip()] = baru.strip()
    return hasil


def _wajib(p, *nama):
    for n in nama:
        if p.get(n) in (None, "", []):
            raise ValueError(f"isian '{n}' belum diisi")


def _kol(p, k="kolom"):
    """Daftar kolom terpilih sebagai kode; kosong = semua kolom teks saat langkah jalan."""
    return lits(p[k]) if p.get(k) else "df.select(pl.col(pl.String)).columns"


def _ganti_polars(g):
    """Template ganti gaya Python (\\1) -> gaya polars (${1}); `$` literal jadi `$$`."""
    return re.sub(r"\\(\d)", r"${\1}", g.replace("$", "$$"))


def gen_regex_ganti(p):
    _wajib(p, "pola")
    pola, ganti = p["pola"], p.get("ganti", "")
    if (pola, ganti) == PRESET_REGEX["huruf berulang (kerennnn → keren)"]:
        return f"df = df.with_columns([huruf_berulang(pl.col(c)).alias(c) for c in {_kol(p)}])"
    if PERLU_PY.search(pola) or "\\g<" in ganti:
        return (f"df = df.with_columns([ganti_regex_py(pl.col(c), {lit(pola)}, {lit(ganti)}).alias(c)\n"
                f"                     for c in {_kol(p)}])")
    return f"df = df.with_columns(pl.col({_kol(p)}).cast(pl.String).str.replace_all({lit(pola)}, {lit(_ganti_polars(ganti))}))"


def gen_regex_ambil(p):
    _wajib(p, "kolom", "pola", "baru")
    grup = int(p.get("grup") or 1)
    if PERLU_PY.search(p["pola"]):
        e = f"ambil_regex_py(pl.col({lit(p['kolom'])}), {lit(p['pola'])}, {grup})"
    else:
        e = f"pl.col({lit(p['kolom'])}).cast(pl.String).str.extract({lit(p['pola'])}, {grup})"
    return f"df = df.with_columns({e}.alias({lit(p['baru'])}))"


def gen_rapikan_teks(p):
    e = "pl.col(c).cast(pl.String)"
    if p.get("nfkc", True):
        e += '.str.normalize("NFKC")'
    if p.get("kecil", True):
        e += ".str.to_lowercase()"
    if p.get("spasi", True):
        e += r'.str.replace_all(r"\s+", " ").str.strip_chars()'
    return f"df = df.with_columns([{e}.alias(c) for c in {_kol(p)}])"


def gen_ubah_tipe(p):
    _wajib(p, "kolom", "tipe")
    e = {
        "angka": "pl.col(c).cast(pl.String).str.strip_chars().cast(pl.Float64, strict=False)",
        "angka Indonesia (1.234,5)": "angka_indonesia(pl.col(c))",
        "bulat": "pl.col(c).cast(pl.String).str.strip_chars().cast(pl.Int64, strict=False)",
        "tanggal": "tanggal_campur(pl.col(c))",
        "tanggal-waktu": "tanggal_campur(pl.col(c), waktu=True)",
        "kategori": "pl.col(c).cast(pl.String).cast(pl.Categorical)",
        "teks": "pl.col(c).cast(pl.String)",
        "boolean": "ke_boolean(pl.col(c))",
    }[p["tipe"]]
    return f"df = df.with_columns([{e}.alias(c) for c in {_kol(p)}])"


def gen_jadi_kosong(p):
    nil = baris(p.get("nilai", NIL_BAWAAN)) + [""]
    return (f"NIL = {lits(nil)}\n"
            f"df = df.with_columns([pl.when(pl.col(c).cast(pl.String).str.strip_chars().is_in(NIL)).then(None)\n"
            f"                     .otherwise(pl.col(c)).alias(c) for c in {_kol(p)}])")


def gen_isi_kosong(p):
    _wajib(p, "kolom", "cara")
    cara = p["cara"]
    if cara == "nilai tetap":
        _wajib(p, "nilai")
        e = f"pl.col(c).fill_null(lit_sebagai({lit(p['nilai'])}, df.schema[c]))"
    elif cara.endswith(" per grup"):
        if p.get("grup") in (None, "", "-"):
            raise ValueError("isian 'grup' belum diisi")
        return (f"for c in {_kol(p)}:\n"
                f"    df = isi_per_grup(df, LATIH, c, {lit(p['grup'])}, \"{cara.removesuffix(' per grup')}\")")
    elif cara in ("median", "rata-rata", "modus"):
        e = f'pl.col(c).fill_null(statistik(LATIH[c], "{cara}"))'
    else:
        e = f"pl.col(c).{'forward_fill' if cara == 'isi dari atas' else 'backward_fill'}()"
    return f"df = df.with_columns([{e}.alias(c) for c in {_kol(p)}])"


def gen_ganti_nilai(p):
    _wajib(p, "kolom", "peta")
    return f"df = df.with_columns([pl.col(c).replace({peta(p['peta'])!r}) for c in {_kol(p)}])"


def gen_ganti_nama(p):
    _wajib(p, "peta")
    return f"df = df.rename({peta(p['peta'])!r})"


def gen_buang_kolom(p):
    _wajib(p, "kolom")
    return f"df = df.drop({lits(p['kolom'])})"


def gen_saring_baris(p):
    _wajib(p, "cara")
    cara = p["cara"]
    if cara == "ekspresi SQL":
        _wajib(p, "isi")
        return f"df = df.filter(pl.sql_expr({lit(p['isi'])}))"
    _wajib(p, "kolom")
    k = lit(p["kolom"])
    if cara in ("cocok regex", "tidak cocok regex"):
        _wajib(p, "isi")
        e = (f"cocok_regex_py(pl.col({k}), {lit(p['isi'])})" if PERLU_PY.search(p["isi"])
             else f"pl.col({k}).cast(pl.String).str.contains({lit(p['isi'])})")
        e = e + ".fill_null(False)" if cara == "cocok regex" else e + ".fill_null(False).not_()"
    else:
        e = f"pl.col({k}).{'is_null' if cara == 'kosong' else 'is_not_null'}()"
    return f"df = df.filter({e})"


def gen_buang_duplikat(p):
    keep = {"pertama": "first", "terakhir": "last", "buang semua salinan": "none"}[p.get("simpan") or "pertama"]
    sub = lits(p["kolom"]) if p.get("kolom") else "None"
    return f'df = df.unique(subset={sub}, keep="{keep}", maintain_order=True)'


def gen_list_teks(p):
    _wajib(p, "kolom")
    return f"df = df.with_columns([urai_list(pl.col(c)).alias(c) for c in {lits(p['kolom'])}])"


def gen_pecah(p):
    _wajib(p, "kolom")
    return f"df = df.{'explode' if p.get('cara', 'list → baris') == 'list → baris' else 'unnest'}({lit(p['kolom'])})"


def gen_join(p):
    _wajib(p, "tabel", "kiri")
    kanan = [x.strip() for x in str(p.get("kanan") or "").split(",") if x.strip()] or p["kiri"]
    val = p.get("validasi") or "-"
    v = "" if val == "-" else f', validate="{val}"'
    return (f"df = df.join(T[{lit(p['tabel'])}], left_on={lits(p['kiri'])}, right_on={lits(kanan)},\n"
            f"             how=\"{p.get('cara') or 'left'}\", suffix=\"_kanan\", coalesce=True{v})")


def gen_tumpuk(p):
    _wajib(p, "tabel")
    if p.get("asal"):
        isi = ['df.with_columns(_asal=pl.lit("ini"))'] + [
            f"T[{lit(t)}].with_columns(_asal=pl.lit({lit(t)}))" for t in p["tabel"]]
    else:
        isi = ["df"] + [f"T[{lit(t)}]" for t in p["tabel"]]
    return f'df = pl.concat([{", ".join(isi)}], how="diagonal_relaxed")'


def gen_agregasi(p):
    _wajib(p, "kolom")
    aggs = []
    for b in baris(p.get("isi")):
        if ":" not in b:
            raise ValueError(f"baris tanpa ':': {b} (contoh: harga: mean, max)")
        k, fs = b.rsplit(":", 1)
        for f in (x.strip() for x in fs.split(",") if x.strip()):
            if f not in AGG:
                raise ValueError(f"fungsi '{f}' tidak dikenal; pilih dari {', '.join(AGG)}")
            aggs.append(f"pl.col({lit(k.strip())}).{f}().alias({lit(k.strip() + '_' + f)})")
    aggs.append('pl.len().alias("jumlah_baris")')
    return f"df = df.group_by({lits(p['kolom'])}, maintain_order=True).agg(\n    " + ",\n    ".join(aggs) + ")"


def gen_sql(p):
    _wajib(p, "isi")
    return f"df = pl.SQLContext(df=df).execute({lit(p['isi'])}, eager=True)"


def gen_outlier(p):
    """Batas dihitung dari data latih (LATIH), lalu dipakai sama untuk train dan test."""
    _wajib(p, "cara", "aksi")
    f = str(p.get("faktor") or "").replace(",", ".").strip()
    f = repr(float(f)) if f else "None"
    kol = lits(p["kolom"]) if p.get("kolom") else "[c for c, t in df.schema.items() if t.is_numeric()]"
    b = f'*batas_outlier(LATIH[c], {lit(p["cara"])}, {f})'
    if p["aksi"] == "potong ke batas (winsorize)":
        return f"df = df.with_columns([pl.col(c).clip({b}) for c in {kol}])"
    if p["aksi"] == "jadikan kosong":
        return f"df = df.with_columns([pl.when(pl.col(c).is_between({b})).then(pl.col(c)).alias(c) for c in {kol}])"
    return (f"df = df.filter(pl.all_horizontal([pl.col(c).is_between({b}) | pl.col(c).is_null() "
            f"for c in {kol}]))")


def _opsi(p, k):
    return None if p.get(k) in (None, "", "-") else p[k]


def gen_rentang(p):
    _wajib(p, "kolom")
    ambil = p.get("ambil") or "tengah"
    akhir = "c" if p.get("simpan") == "ganti kolom" else 'c + "_angka"'
    return f'df = df.with_columns([angka_rentang(pl.col(c), "{ambil}").alias({akhir}) for c in {lits(p["kolom"])}])'


ENCODE = {"one-hot (kolom 0/1 per nilai)": "one_hot(df, LATIH, c, {maks})",
          "ordinal (0, 1, 2 …)": "encode_ordinal(df, LATIH, c, {urutan})",
          "frekuensi (porsi di latih)": "encode_frekuensi(df, LATIH, c)",
          "target OOF (rata-rata label)": "encode_target(df, LATIH, c, {label})"}


def gen_encode(p):
    _wajib(p, "kolom", "cara")
    if p["cara"].startswith("target") and not _opsi(p, "label"):
        raise ValueError("target encoding butuh kolom label")
    f = ENCODE[p["cara"]].format(maks=int(p.get("maks") or 20), urutan=lits(baris(p.get("urutan"))) if baris(
        p.get("urutan")) else "None", label=lit(p.get("label") or ""))
    return f"for c in {lits(p['kolom'])}:\n    df = {f}"


def gen_langka(p):
    _wajib(p, "kolom")
    return (f"for c in {lits(p['kolom'])}:\n"
            f"    df = gabung_langka(df, LATIH, c, {int(p.get('minimal') or 10)}, {lit(p.get('nama') or 'lainnya')})")


def gen_penanda_kosong(p):
    kol = lits(p["kolom"]) if p.get("kolom") else "[c for c in LATIH.columns if LATIH[c].null_count()]"
    return f'df = df.with_columns([pl.col(c).is_null().cast(pl.Int8).alias(c + "_kosong") for c in {kol}])'


BAGIAN_TGL = ["tahun", "bulan", "hari", "hari_minggu", "akhir_pekan"]


def gen_tanggal(p):
    _wajib(p, "kolom")
    bag = [b for b in BAGIAN_TGL if p.get(b)]
    acuan = _opsi(p, "acuan")
    if not bag and not acuan:
        raise ValueError("pilih minimal satu bagian tanggal atau kolom acuan")
    kode = (f"for c in {lits(p['kolom'])}:\n"
            f"    df = fitur_tanggal(df, c, {lits(bag)}, {lit(acuan) if acuan else 'None'})")
    return kode + (f"\ndf = df.drop({lits(p['kolom'])})" if p.get("buang") else "")


def gen_skala(p):
    _wajib(p, "cara")
    kol = lits(p["kolom"]) if p.get("kolom") else "[c for c, t in df.schema.items() if t.is_numeric()]"
    return f"for c in {kol}:\n    df = skala(df, LATIH, c, {lit(p['cara'])})"


def gen_mirip(p):
    return f"for c in {_kol(p)}:\n    df = satukan_mirip(df, LATIH, c)"


def gen_kode(p):
    _wajib(p, "isi")
    return str(p["isi"])


JENIS = {
    "regex_ganti": {"judul": "Ganti pola teks (regex)", "gen": gen_regex_ganti, "isian": [
        ("kolom", "kolom_banyak", "Kolom (kosong = semua kolom teks)"), ("preset", ("pilihan", ["-"] + list(PRESET_REGEX)), "Preset"),
        ("pola", "teks", "Pola regex"), ("ganti", "teks", "Ganti dengan (\\1 = grup 1)")]},
    "regex_ambil": {"judul": "Ambil bagian teks ke kolom baru (regex)", "gen": gen_regex_ambil, "isian": [
        ("kolom", "kolom", "Kolom"), ("pola", "teks", "Pola regex dengan grup ( )"), ("grup", "angka", "Nomor grup"),
        ("baru", "teks", "Nama kolom baru")]},
    "rapikan_teks": {"judul": "Rapikan teks (NFKC, huruf kecil, spasi)", "gen": gen_rapikan_teks, "isian": [
        ("kolom", "kolom_banyak", "Kolom (kosong = semua kolom teks)"), ("nfkc", "cek", "Normalisasi Unicode NFKC"),
        ("kecil", "cek", "Huruf kecil"), ("spasi", "cek", "Rapikan spasi")]},
    "ubah_tipe": {"judul": "Ubah tipe kolom", "gen": gen_ubah_tipe, "isian": [
        ("kolom", "kolom_banyak", "Kolom"), ("tipe", ("pilihan", TIPE), "Jadi tipe")]},
    "jadi_kosong": {"judul": "Jadikan kosong (null): '-', 'NA', ''…", "gen": gen_jadi_kosong, "isian": [
        ("kolom", "kolom_banyak", "Kolom (kosong = semua kolom teks)"), ("nilai", "teks_panjang", "Nilai (satu per baris; teks kosong selalu ikut)")]},
    "isi_kosong": {"judul": "Isi nilai kosong", "gen": gen_isi_kosong, "isian": [
        ("kolom", "kolom_banyak", "Kolom"),
        ("cara", ("pilihan", ["median", "rata-rata", "modus", "median per grup", "rata-rata per grup",
                              "modus per grup", "nilai tetap", "isi dari atas", "isi dari bawah"]), "Cara"),
        ("grup", "kolom_opsional", "Kolom grup (cara per grup)"), ("nilai", "teks", "Nilai tetap")]},
    "penanda_kosong": {"judul": "Tambah kolom penanda kosong (c_kosong = 1/0)", "gen": gen_penanda_kosong, "isian": [
        ("kolom", "kolom_banyak", "Kolom (kosong = semua kolom yang punya null di latih)")]},
    "ganti_nilai": {"judul": "Ganti nilai (peta)", "gen": gen_ganti_nilai, "isian": [
        ("kolom", "kolom_banyak", "Kolom"), ("peta", "teks_panjang", "lama => baru (satu per baris)")]},
    "ganti_nama": {"judul": "Ganti nama kolom", "gen": gen_ganti_nama, "isian": [
        ("peta", "teks_panjang", "lama => baru (satu per baris)")]},
    "buang_kolom": {"judul": "Buang kolom", "gen": gen_buang_kolom, "isian": [("kolom", "kolom_banyak", "Kolom")]},
    "saring_baris": {"judul": "Saring baris (simpan yang …)", "gen": gen_saring_baris, "hanya_latih": True, "isian": [
        ("cara", ("pilihan", ["ekspresi SQL", "cocok regex", "tidak cocok regex", "kosong", "tidak kosong"]), "Simpan baris yang"),
        ("kolom", "kolom", "Kolom (selain SQL)"), ("isi", "teks", "Ekspresi SQL / pola regex")]},
    "buang_duplikat": {"judul": "Buang baris duplikat", "gen": gen_buang_duplikat, "hanya_latih": True, "isian": [
        ("kolom", "kolom_banyak", "Kolom penentu (kosong = semua kolom)"),
        ("simpan", ("pilihan", ["pertama", "terakhir", "buang semua salinan"]), "Simpan")]},
    "list_teks": {"judul": "Urai teks list \"['a','b']\" jadi list", "gen": gen_list_teks, "isian": [
        ("kolom", "kolom_banyak", "Kolom")]},
    "pecah": {"judul": "Pecah list/struct", "gen": gen_pecah, "isian": [
        ("kolom", "kolom", "Kolom"), ("cara", ("pilihan", ["list → baris", "struct → kolom"]), "Cara")]},
    "join": {"judul": "Gabung dengan tabel lain (join)", "gen": gen_join, "isian": [
        ("tabel", "tabel", "Tabel kanan"), ("kiri", "kolom_banyak", "Kunci di tabel ini"),
        ("kanan", "teks", "Kunci di tabel kanan (pisah koma; kosong = sama)"),
        ("cara", ("pilihan", ["left", "inner", "full", "semi", "anti"]), "Cara"),
        ("validasi", ("pilihan", ["-", "m:1", "1:1", "1:m"]), "Validasi kunci")]},
    "tumpuk": {"judul": "Tumpuk tabel lain di bawah (concat)", "gen": gen_tumpuk, "hanya_latih": True, "isian": [
        ("tabel", "tabel_banyak", "Tabel"), ("asal", "cek", "Tambah kolom _asal")]},
    "agregasi": {"judul": "Kelompokkan + ringkas (group by)", "gen": gen_agregasi, "isian": [
        ("kolom", "kolom_banyak", "Kunci kelompok"),
        ("isi", "teks_panjang", "kolom: fungsi, fungsi (per baris). Fungsi: " + ", ".join(AGG))]},
    "outlier": {"judul": "Tangani outlier (IQR / z / MAD / persentil)", "gen": gen_outlier, "isian": [
        ("kolom", "kolom_banyak", "Kolom (kosong = semua kolom angka)"),
        ("cara", ("pilihan", ["IQR", "z", "MAD", "persentil"]), "Cara hitung batas"),
        ("faktor", "teks", "Faktor (kosong = bawaan: IQR 1,5 · z 3 · MAD 3,5 · persentil 1)"),
        ("aksi", ("pilihan", ["potong ke batas (winsorize)", "jadikan kosong", "buang baris"]), "Nilai di luar batas")]},
    "rentang": {"judul": "Teks rentang → angka (Rp 1 jt s.d. 2 jt)", "gen": gen_rentang, "isian": [
        ("kolom", "kolom_banyak", "Kolom"), ("ambil", ("pilihan", ["tengah", "bawah", "atas"]), "Ambil nilai"),
        ("simpan", ("pilihan", ["kolom baru (c_angka)", "ganti kolom"]), "Simpan ke")]},
    "encode": {"judul": "Encoding kategori (one-hot / ordinal / frekuensi / target OOF)", "gen": gen_encode, "isian": [
        ("kolom", "kolom_banyak", "Kolom"), ("cara", ("pilihan", list(ENCODE)), "Cara"),
        ("label", "kolom_opsional", "Kolom label (target OOF)"), ("maks", "angka", "One-hot: maks nilai tersering"),
        ("urutan", "teks_panjang", "Ordinal: urutan nilai, satu per baris (kosong = abjad)")]},
    "langka": {"judul": "Gabung kategori langka jadi 'lainnya'", "gen": gen_langka, "isian": [
        ("kolom", "kolom_banyak", "Kolom"), ("minimal", "angka", "Minimal muncul di latih"),
        ("nama", "teks", "Ganti dengan")]},
    "mirip": {"judul": "Satukan nilai mirip (beda huruf besar, tanda baca, urutan kata)", "gen": gen_mirip, "isian": [
        ("kolom", "kolom_banyak", "Kolom (kosong = semua kolom teks)")]},
    "tanggal": {"judul": "Fitur tanggal (tahun, bulan, hari, selisih hari)", "gen": gen_tanggal, "isian": [
        ("kolom", "kolom_banyak", "Kolom tanggal"), ("tahun", "cek", "Tahun"), ("bulan", "cek", "Bulan"),
        ("hari", "cek", "Tanggal (1-31)"), ("hari_minggu", "cek", "Hari dalam minggu (1 = Senin)"),
        ("akhir_pekan", "cek", "Akhir pekan (1/0)"), ("acuan", "kolom_opsional", "Selisih hari dengan kolom"),
        ("buang", "cek", "Buang kolom tanggal asal")]},
    "skala": {"judul": "Skala angka (standar / robust / min-max / log1p)", "gen": gen_skala, "isian": [
        ("kolom", "kolom_banyak", "Kolom (kosong = semua kolom angka)"),
        ("cara", ("pilihan", ["standar", "robust", "min-max", "log1p"]), "Cara")]},
    "sql": {"judul": "Query SQL (tabel bernama df)", "gen": gen_sql, "isian": [("isi", "teks_panjang", "SQL")]},
    "kode": {"judul": "Kode polars bebas", "gen": gen_kode, "isian": [("isi", "teks_panjang", "Kode (ubah df)")]},
}
BAWAAN = {"rapikan_teks": {"nfkc": True, "kecil": True, "spasi": True}, "jadi_kosong": {"nilai": NIL_BAWAAN},
          "sql": {"isi": "SELECT * FROM df"}, "kode": {"isi": "df = df"}, "regex_ambil": {"grup": 1},
          "encode": {"maks": 20}, "langka": {"minimal": 10, "nama": "lainnya"},
          "tanggal": {"tahun": True, "bulan": True, "hari_minggu": True}}


def buat(jenis, param):
    """Langkah baru siap simpan di proyek."""
    return {"jenis": jenis, "param": param, "kode": JENIS[jenis]["gen"](param), "nyala": True,
            "hanya_latih": JENIS[jenis].get("hanya_latih", False) or param.get("aksi") == "buang baris",
            "manual": False}


def ringkas(l):
    """Satu baris untuk daftar resep."""
    j = JENIS.get(l["jenis"], {}).get("judul", l["jenis"])
    p = l.get("param", {})
    rinci = p.get("kolom") or p.get("tabel") or p.get("pola") or p.get("isi") or ""
    if isinstance(rinci, list):
        rinci = ", ".join(rinci)
    rinci = str(rinci).replace("\n", " ")
    teks = j + (f": {rinci[:60]}" if rinci else "")
    if l.get("manual"):
        teks += " ✎"
    if l.get("hanya_latih"):
        teks += " [latih saja]"
    return teks
