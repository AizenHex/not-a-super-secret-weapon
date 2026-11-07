"""Mesin resep (D1-D3, A3). Resep = daftar langkah kode polars per tabel. Tabel `pasangan` (mis. test) ikut resep
tabel pemiliknya (train): langkah yang sama, statistik dari latih, langkah `hanya_latih` dilewati. Tabel turunan =
kode yang membuat `df` dari `T["..."]`. Tiap tahap di-cache dengan kunci hash (sumber + kode + tabel acuan), jadi
mengubah langkah ke-5 hanya menghitung ulang dari langkah ke-5."""
import ast
import hashlib
import math
import os
import re
import textwrap
import traceback
from collections import OrderedDict

import polars as pl

from mesin import bantu, muat

REF = re.compile(r"""T\[\s*[rR]?(["'])(.+?)\1\s*\]""")
MAKS_TAHAP = 80  # tahap tersimpan; kolom yang tidak berubah berbagi memori antar tahap
PETUNJUK = [  # pesan polars yang sering muncul -> apa yang harus dilakukan
    ("cannot cast List type", "kolom ini berisi list; pecah dulu (langkah Pecah) atau pilih kolom teks"),
    ("join keys don't match", "tipe kunci join berbeda; samakan dulu dengan langkah Ubah tipe"),
    ("`explode` operation not supported", "kolom ini bukan list"),
    ("expected 'Struct'", "kolom ini bukan struct"),
    ("conversion from `str`", "nilai teks tidak cocok dengan tipe kolom; ubah tipe kolom ke teks dulu"),
    ("casting from Utf8View to", "nilai teks tidak cocok dengan tipe kolom; ubah tipe kolom ke teks dulu"),
    ("ColumnNotFoundError", "nama kolom tidak ada di tahap ini (dibuang/diganti nama di langkah sebelumnya?)"),
]


def pesan_galat(e):
    m = f"{type(e).__name__}: {e}"
    saran = [s for k, s in PETUNJUK if k in m]
    return m + ("\n→ " + saran[0] if saran else "")


class TabelGagal(Exception):
    pass


class Hasil:
    def __init__(self, nama, pemilik, lap):
        self.nama, self.pemilik, self.lap = nama, pemilik, lap
        self.tahap, self.kunci = [], []
        self.galat = None  # (nomor langkah 1.., pesan, traceback)

    @property
    def akhir(self):
        return self.tahap[-1]


def _hash(*x):
    return hashlib.sha1(repr(x).encode()).hexdigest()


def acuan(kode):
    return sorted({m.group(2) for m in REF.finditer(kode or "")})


def pemilik(proyek, nama):
    """Tabel yang resepnya dipakai `nama`: dirinya sendiri, atau tabel latih kalau `nama` adalah pasangannya."""
    for n, r in proyek.get("resep", {}).items():
        if n != nama and nama in r.get("pasangan", []):
            return n
    return nama


def resep_dari(proyek, nama):
    """(pemilik, resep); resep kosong kalau belum ada."""
    pem = pemilik(proyek, nama)
    return pem, proyek.get("resep", {}).get(pem, {"langkah": [], "pasangan": []})


def kelompok(proyek, nama):
    pem, r = resep_dari(proyek, nama)
    return [pem] + [p for p in r.get("pasangan", []) if p != pem]


class _ProksiT:
    """`T["nama"]` di dalam kode langkah = tabel itu setelah resepnya (final)."""

    def __init__(self, mesin):
        self.m = mesin

    def __getitem__(self, nama):
        h = self.m.hasil(nama)
        if h.galat:
            raise TabelGagal(f"tabel {nama} gagal di langkah {h.galat[0]}: {h.galat[1]}")
        return h.akhir

    def __contains__(self, nama):
        return nama in self.m.nama_tabel()


class Mesin:
    def __init__(self, proyek, entri):
        self.proyek, self.entri = proyek, entri
        self._baca = {}
        self._tahap = OrderedDict()
        self._fungsi = {}
        self._sedang = []
        self.glob = {"pl": pl, "re": re, "ast": ast, "math": math, "T": _ProksiT(self),
                     **{k: v for k, v in vars(bantu).items() if not k.startswith("__") and k != "pl"}}

    # ---------- struktur proyek ----------
    def semua_resep(self):
        return self.proyek.setdefault("resep", {})

    def turunan(self):
        return self.proyek.setdefault("turunan", {})

    def nama_tabel(self):
        return list(self.entri) + [t for t in self.turunan() if t not in self.entri]

    def pemilik(self, nama):
        return pemilik(self.proyek, nama)

    def resep(self, nama):
        return resep_dari(self.proyek, nama)

    def kelompok(self, nama):
        return kelompok(self.proyek, nama)

    # ---------- hitung ----------
    def _fungsi_dari(self, kode, nama="_langkah"):
        f = self._fungsi.get(kode)
        if f is None:
            src = f"def {nama}(df=None, LATIH=None):\n" + textwrap.indent(kode, "    ") + "\n    return df\n"
            g = dict(self.glob)
            exec(compile(src, "<langkah>", "exec"), g)
            f = self._fungsi[kode] = g[nama]
        return f

    def _simpan_tahap(self, k, df):
        self._tahap[k] = df
        self._tahap.move_to_end(k)
        while len(self._tahap) > MAKS_TAHAP:
            self._tahap.popitem(last=False)

    def _kunci_acuan(self, kode):
        return tuple((t, self.hasil(t).kunci[-1]) for t in acuan(kode))

    def sumber(self, nama):
        """(df, kunci, lap) tabel sebelum resep: hasil baca file, atau hasil kode turunan."""
        if nama in self.turunan():
            kode = self.turunan()[nama]
            k = _hash("turunan", kode, self._kunci_acuan(kode))
            if k not in self._tahap:
                df = self._fungsi_dari(kode, "_turunan")()
                if not isinstance(df, pl.DataFrame):
                    raise TypeError("kode tabel turunan harus mengisi df dengan DataFrame polars")
                self._simpan_tahap(k, df)
            return self._tahap[k], k, {"format": "turunan", "catatan": [], "detik": 0}
        if nama not in self.entri:
            raise KeyError(f"tabel {nama} tidak ada di folder data")
        e = self.entri[nama]
        opsi = self.proyek.get("opsi_baca", {}).get(nama, {})
        st = os.stat(e["path"])
        k = _hash("baca", e["path"], e["sheet"], sorted(opsi.items()), st.st_mtime_ns, st.st_size)
        if self._baca.get(nama, (None,))[0] != k:
            df, lap = muat.baca(e["path"], e["sheet"], opsi)
            self._baca[nama] = (k, df, lap)
        return self._baca[nama][1], k, self._baca[nama][2]

    def hasil(self, nama):
        if nama in self._sedang:
            raise TabelGagal("acuan berputar: " + " → ".join(self._sedang + [nama]))
        self._sedang.append(nama)
        try:
            return self._hasil(nama)
        finally:
            self._sedang.pop()

    def _hasil(self, nama):
        pem, r = self.resep(nama)
        latih = self.hasil(pem) if pem != nama else None
        df, k, lap = self.sumber(nama)
        h = Hasil(nama, pem, lap)
        h.tahap.append(df)
        h.kunci.append(k)
        for i, l in enumerate(r["langkah"]):
            if not l.get("nyala", True) or (latih and l.get("hanya_latih")):
                h.tahap.append(h.tahap[-1])
                h.kunci.append(h.kunci[-1])
                continue
            try:
                if latih:
                    if i >= len(latih.tahap) - 1 and latih.galat:
                        raise TabelGagal(f"resep {pem} gagal di langkah {latih.galat[0]}")
                    data_latih, k_latih = latih.tahap[i], latih.kunci[i]
                else:
                    data_latih, k_latih = h.tahap[-1], None
                k = _hash(h.kunci[-1], l["kode"], self._kunci_acuan(l["kode"]), k_latih)
                if k not in self._tahap:
                    df = self._fungsi_dari(l["kode"])(h.tahap[-1], data_latih)
                    if not isinstance(df, pl.DataFrame):
                        raise TypeError("langkah harus menyisakan df berupa DataFrame polars")
                    self._simpan_tahap(k, df)
            except Exception as e:
                tb = traceback.format_exc()
                h.galat = (i + 1, pesan_galat(e), tb)
                break
            h.tahap.append(self._tahap[k])
            h.kunci.append(k)
        return h

    def pratinjau(self, nama, indeks, kode):
        """Jalankan `kode` sebagai langkah ke-`indeks` (0..) tanpa menyimpannya: (sebelum, sesudah)."""
        pem, r = self.resep(nama)
        h = self.hasil(nama)
        if indeks >= len(h.tahap):
            raise TabelGagal(f"langkah {h.galat[0]} sebelumnya gagal: {h.galat[1]}")
        sebelum = h.tahap[indeks]
        latih = self.hasil(pem).tahap[indeks] if pem != nama else sebelum
        return sebelum, self._fungsi_dari(kode)(sebelum, latih)


def ringkas(a, b):
    """Apa yang berubah dari tabel a ke b."""
    r = {"baris": (a.height, b.height), "tambah": [c for c in b.columns if c not in a.columns],
         "hapus": [c for c in a.columns if c not in b.columns], "ubah": {}, "null_baru": {}, "tipe": {}}
    for c in b.columns:
        if c not in a.columns:
            continue
        if a[c].dtype != b[c].dtype:
            r["tipe"][c] = (str(a[c].dtype), str(b[c].dtype))
        d = b[c].null_count() - a[c].null_count()
        if d:
            r["null_baru"][c] = d
        if a.height == b.height:
            n = _beda(a[c], b[c])
            if n is None or n:
                r["ubah"][c] = n
    return r


def _beda(x, y):
    try:
        if x.dtype != y.dtype:
            x, y = x.cast(pl.String), y.cast(pl.String)
        return int(x.ne_missing(y).sum())
    except Exception:
        return None


def contoh_ubah(a, b, n=5):
    """Sampai n sel yang berubah: (kolom, baris, sebelum, sesudah)."""
    out = []
    if a.height != b.height:
        return out
    for c in b.columns:
        if c not in a.columns or len(out) >= n:
            continue
        x, y = a[c], b[c]
        try:
            if x.dtype != y.dtype:
                x, y = x.cast(pl.String), y.cast(pl.String)
            idx = x.ne_missing(y).arg_true().head(n - len(out)).to_list()
        except Exception:
            continue
        out += [(c, i, a[c][i], b[c][i]) for i in idx]
    return out


def teks_ringkas(r):
    s = [f"baris {r['baris'][0]:,} → {r['baris'][1]:,}".replace(",", ".")]
    if r["tambah"]:
        s.append("kolom baru: " + ", ".join(r["tambah"]))
    if r["hapus"]:
        s.append("kolom dibuang: " + ", ".join(r["hapus"]))
    for c, (x, y) in r["tipe"].items():
        s.append(f"{c}: tipe {x} → {y}")
    for c, n in r["ubah"].items():
        s.append(f"{c}: {'?' if n is None else f'{n:,}'.replace(',', '.')} sel berubah")
    for c, n in r["null_baru"].items():
        s.append(f"{c}: null {'+' if n > 0 else ''}{n:,}".replace(",", "."))
    return "\n".join(s)
