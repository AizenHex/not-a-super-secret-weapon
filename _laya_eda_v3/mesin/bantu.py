"""Fungsi bantu yang dipakai kode langkah Laya. File ini ikut tersalin apa adanya ke kode ekspor, jadi hasil
notebook = hasil Laya. Hanya boleh bergantung pada polars + pustaka standar."""
import ast as _ast
import json as _json
import re as _re

import polars as pl

FORMAT_TANGGAL = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y", "%m/%d/%Y", "%d %b %Y", "%d %B %Y",
                  "%Y%m%d"]
FORMAT_WAKTU = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"]
BENAR = ["1", "true", "t", "ya", "y", "yes", "benar"]
SALAH = ["0", "false", "f", "tidak", "n", "no", "salah"]


def _balik_mojibake(m):
    for kod in ("mac_roman", "cp1252"):
        try:
            return m.group().encode(kod).decode("utf-8")
        except UnicodeError:
            pass
    return m.group()


def perbaiki_mojibake(e):
    """UTF-8 yang terbaca sebagai MacRoman/cp1252 ('¬©' → '©', 'Ã©' → 'é'). Tiap rangkaian non-ASCII hanya diganti
    bila hasil baliknya UTF-8 sah, jadi teks yang sudah benar tidak berubah."""
    return e.cast(pl.String).map_elements(lambda s: _re.sub(r"[^\x00-\x7F]{2,}", _balik_mojibake, s),
                                          return_dtype=pl.String)


def ganti_regex_py(e, pola, ganti):
    """Regex ganti dengan mesin Python (backreference/lookaround yang tidak didukung polars)."""
    rx = _re.compile(pola)
    return e.cast(pl.String).map_elements(lambda s: rx.sub(ganti, s), return_dtype=pl.String)


def huruf_berulang(e):
    """`kerennnn` -> `keren`: 3+ huruf/angka sama berturut jadi satu. Sama dengan regex Python (\\w)\\1{2,} -> \\1
    di GDOC, ~20x lebih cepat. shortcut: hanya a-z A-Z 0-9 _; huruf non-ASCII berulang tidak diringkas, pakai
    langkah regex dengan pola sendiri bila perlu."""
    e = e.cast(pl.String)
    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_":
        e = e.str.replace_all(ch * 3 + "+", ch)
    return e


def ambil_regex_py(e, pola, grup=1):
    rx = _re.compile(pola)

    def f(s):
        m = rx.search(s)
        return m.group(grup) if m else None

    return e.cast(pl.String).map_elements(f, return_dtype=pl.String)


def cocok_regex_py(e, pola):
    rx = _re.compile(pola)
    return e.cast(pl.String).map_elements(lambda s: rx.search(s) is not None, return_dtype=pl.Boolean)


def tanggal_campur(e, waktu=False):
    """Format pertama yang cocok menang (urutan FORMAT_*: hari/bulan didahulukan dari bulan/hari). Gagal -> null."""
    s = e.cast(pl.String).str.strip_chars()
    if waktu:
        return pl.coalesce([s.str.to_datetime(f, strict=False) for f in FORMAT_WAKTU])
    return pl.coalesce([s.str.to_date(f, strict=False) for f in FORMAT_TANGGAL])


def ke_boolean(e):
    s = e.cast(pl.String).str.strip_chars().str.to_lowercase()
    return pl.when(s.is_in(BENAR)).then(True).when(s.is_in(SALAH)).then(False).otherwise(None)


def angka_indonesia(e):
    """`1.234,5` -> 1234.5 (titik = ribuan, koma = desimal). Gagal -> null."""
    s = e.cast(pl.String).str.strip_chars().str.replace_all(".", "", literal=True).str.replace(",", ".", literal=True)
    return s.cast(pl.Float64, strict=False)


def modus(s):
    """Nilai terbanyak; seri diputus dengan urutan nilai supaya hasil selalu sama."""
    v = s.drop_nulls()
    if v.len() == 0:
        return None
    return v.value_counts(sort=True).sort(["count", v.name], descending=[True, False])[v.name][0]


def statistik(s, cara):
    """Nilai pengisi null dari kolom latih: median / rata-rata / modus."""
    if cara != "modus" and not (s.dtype.is_numeric() or s.dtype.is_temporal()):
        raise TypeError(f"{cara} butuh kolom angka; '{s.name}' bertipe {s.dtype}. Ubah tipe dulu atau pakai modus")
    v = modus(s) if cara == "modus" else s.median() if cara == "median" else s.mean()
    if v is None:
        raise ValueError(f"'{s.name}' di data latih kosong semua; tidak ada {cara}")
    return v


OUTLIER = {"IQR": 1.5, "z": 3.0, "MAD": 3.5, "persentil": 1.0}  # cara: faktor bawaan


def batas_outlier(s, cara="IQR", faktor=None):
    """Batas (bawah, atas) dari kolom latih. IQR: Q1 - f·IQR s.d. Q3 + f·IQR; z: rata-rata ± f·sd;
    MAD: median ± f·1,4826·MAD (tahan ekor panjang); persentil: f% terbawah dan teratas (1 = p1 s.d. p99)."""
    if not (s.dtype.is_numeric() or s.dtype == pl.Null):
        raise TypeError(f"outlier butuh kolom angka; '{s.name}' bertipe {s.dtype}. Ubah tipe dulu")
    f = OUTLIER[cara] if faktor is None else float(faktor)
    s = s.drop_nulls().cast(pl.Float64)
    if not s.len():
        raise ValueError(f"'{s.name}' di data latih kosong semua; tidak ada batas outlier")
    if cara == "IQR":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return q1 - f * (q3 - q1), q3 + f * (q3 - q1)
    if cara == "z":
        m, sd = s.mean(), s.std() or 0.0
        return m - f * sd, m + f * sd
    if cara == "MAD":
        med = s.median()
        mad = (s - med).abs().median() * 1.4826
        return med - f * mad, med + f * mad
    if cara == "persentil":
        return s.quantile(f / 100), s.quantile(1 - f / 100)
    raise ValueError(f"cara outlier tidak dikenal: {cara}")


def lit_sebagai(v, dtype):
    """Teks isian form -> literal bertipe kolom (tanggal/boolean tidak bisa lewat cast biasa)."""
    e = pl.lit(str(v))
    if dtype == pl.Date:
        return e.str.to_date()
    if isinstance(dtype, pl.Datetime):
        return e.str.to_datetime().cast(dtype)
    if dtype == pl.Boolean:
        return ke_boolean(e)
    return e.cast(dtype)


def _urai(s):
    try:
        v = _ast.literal_eval(s)
    except Exception:
        try:
            v = _json.loads(s)
        except Exception:
            return None
    if isinstance(v, (set, frozenset)):
        v = sorted(v, key=str)
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else None


def urai_list(e):
    """Teks `"['a', 'b']"` atau `'["a"]'` -> list teks. Tidak bisa diurai -> null."""
    return e.cast(pl.String).map_elements(_urai, return_dtype=pl.List(pl.String))


def nama_aman(nama):
    return _re.sub(r"[^\w-]+", "_", _re.sub(r"\.\w+(?=:|$)", "", nama)).strip("_") or "tabel"


def jalankan_resep(T, latih, pasangan, langkah):
    """Jalankan resep tabel `latih`, lalu resep yang sama ke tabel `pasangan` (mis. test). Langkah menerima
    (df, LATIH): LATIH = data latih sebelum langkah itu, jadi median/modus/peta dihitung dari latih saja.
    Langkah `hanya_latih` (buang baris) dilewati untuk pasangan."""
    df = T[latih]
    ps = {p: T[p] for p in pasangan}
    for f, hanya_latih in langkah:
        sebelum = df
        df = f(df, sebelum)
        if not hanya_latih:
            ps = {p: f(v, sebelum) for p, v in ps.items()}
    T[latih] = df
    T.update(ps)
