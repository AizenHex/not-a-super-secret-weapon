"""Muat data: pindai folder dan baca file apa pun tanpa crash. Tanpa Qt (diuji lewat uji.py)."""
import csv
import io
import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path

import polars as pl

EKSTENSI = {".csv", ".tsv", ".tab", ".txt", ".dat", ".json", ".jsonl", ".ndjson", ".parquet", ".xlsx", ".xlsm", ".xls"}
PEMISAH = [",", ";", "\t", "|"]
CONTOH = 1 << 20  # byte awal untuk menebak encoding dan pemisah
ANGKA_BULAT = r"^[+-]?\d+$"
ANGKA_DESIMAL = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"

csv.field_size_limit(2**31 - 1)  # sel teks panjang (berita 30 rb huruf) jangan dianggap galat


class BacaGagal(Exception):
    """File tidak bisa dibaca. `mentah` = 100 baris awal apa adanya, untuk ditampilkan."""

    def __init__(self, pesan, mentah=""):
        super().__init__(pesan)
        self.mentah = mentah


def pindai(folder):
    """Semua file data di folder + subfolder (belum dibaca). xlsx dipecah per sheet: nama `file.xlsx:Sheet`."""
    folder = Path(folder)
    hasil = []
    for akar, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__")))
        for f in sorted(files):
            p = Path(akar) / f
            ext = p.suffix.lower()
            if ext not in EKSTENSI or f.startswith("~$"):
                continue
            rel = p.relative_to(folder).as_posix()
            ukuran = p.stat().st_size
            if ext in (".xlsx", ".xlsm"):
                try:
                    for s in _daftar_sheet(p):
                        hasil.append({"nama": f"{rel}:{s}", "path": str(p), "sheet": s, "ukuran": ukuran})
                    continue
                except Exception:
                    pass  # xlsx rusak tetap tampil; galatnya muncul saat dibuka
            hasil.append({"nama": rel, "path": str(p), "sheet": None, "ukuran": ukuran})
    return hasil


def baca(path, sheet=None, opsi=None):
    """-> (DataFrame, laporan). opsi: pemisah, encoding, header (bool), lewati (int), semua_teks (bool).
    Gagal -> BacaGagal berisi pesan + 100 baris mentah."""
    opsi = dict(opsi or {})
    p = Path(path)
    ext = p.suffix.lower()
    t0 = time.perf_counter()
    lap = {"file": p.name, "catatan": []}
    try:
        if ext == ".parquet":
            lap["format"] = "parquet"
            df = pl.read_parquet(p)
        elif ext == ".json":
            df = _json(p, lap)
        elif ext in (".jsonl", ".ndjson"):
            df = _jsonl(p, lap)
        elif ext in (".xlsx", ".xlsm"):
            df = _excel(p, sheet, opsi, lap)
        elif ext == ".xls":
            raise BacaGagal("Format .xls lama belum didukung: buka di Excel lalu simpan sebagai .xlsx.")
        else:
            df = _teks(p, ext, opsi, lap)
        df = _nama_unik(df, lap)
        if not opsi.get("semua_teks"):
            df = tebak_tipe(df, lap)
    except BacaGagal as e:
        e.mentah = e.mentah or mentah(p)
        raise
    except Exception as e:
        raise BacaGagal(f"{type(e).__name__}: {e}", mentah(p)) from e
    lap.update(baris=df.height, kolom=df.width, detik=round(time.perf_counter() - t0, 2))
    return df, lap


def tebak_encoding(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError as e:
        if e.start >= len(raw) - 4:  # contoh terpotong di tengah huruf
            return "utf-8"
    try:
        raw.decode("cp1252")
        return "cp1252"
    except UnicodeDecodeError:
        return "latin-1"


def tebak_pemisah(teks, lewati=0):
    """Pemisah yang memberi jumlah kolom paling konsisten di 50 baris awal (kutip dan baris baru dalam sel dihitung
    benar oleh modul csv). None = tidak ada pemisah (satu kolom)."""
    potong = teks.rfind("\n")
    teks = teks[: potong + 1] if potong > 0 else teks
    terbaik, skor = None, (0, 0)
    for sep in PEMISAH:
        try:
            rows = list(itertools.islice(csv.reader(io.StringIO(teks), delimiter=sep), lewati, lewati + 50))
        except csv.Error:
            continue
        n = [len(r) for r in rows if r]
        if not n:
            continue
        modus, kali = Counter(n).most_common(1)[0]
        if modus > 1 and (kali, modus) > skor:
            terbaik, skor = sep, (kali, modus)
    return terbaik


def tebak_tipe(df, lap=None):
    """Kolom teks yang seluruh isinya angka -> Int64/Float64. Angka ber-nol-depan (kode/ID) dan format Indonesia
    (`1.234,5`) tetap teks; nol depan dicatat karena mengubahnya merusak data."""
    ubah = []
    for c in df.columns:
        s = df[c]
        if s.dtype != pl.String:
            continue
        isi = s.drop_nulls()
        if isi.len() == 0 or not isi.head(1000).str.contains(ANGKA_DESIMAL).all():
            continue  # cek cepat dulu supaya kolom teks panjang tidak dipindai penuh
        if not isi.str.contains(ANGKA_DESIMAL).all():
            continue
        if isi.str.contains(r"^[+-]?0\d").any():
            if lap is not None:
                lap["catatan"].append(f"kolom '{c}': angka ber-nol-depan, dibiarkan teks (kode/ID)")
            continue
        bulat = isi.str.contains(ANGKA_BULAT).all()
        v = s.cast(pl.Int64 if bulat else pl.Float64, strict=False)
        if v.null_count() == s.null_count():  # bulat > Int64 gagal di sini dan tetap teks
            ubah.append(v)
    return df.with_columns(ubah) if ubah else df


def mentah(p, n=100):
    try:
        with open(p, "rb") as f:
            raw = f.read(256 * 1024)
        return "\n".join(raw.decode(tebak_encoding(raw), errors="replace").splitlines()[:n])
    except OSError as e:
        return f"(tidak bisa dibuka: {e})"


def _teks(p, ext, opsi, lap):
    with open(p, "rb") as f:
        raw = f.read(CONTOH)
    if not raw.strip():
        lap.update(format="teks")
        lap["catatan"].append("file kosong")
        return pl.DataFrame()
    enc = opsi.get("encoding") or tebak_encoding(raw)
    lewati = int(opsi.get("lewati") or 0)
    sep = opsi.get("pemisah") or ("\t" if ext in (".tsv", ".tab") else tebak_pemisah(raw.decode(enc, errors="replace"), lewati))
    header = opsi.get("header", True)
    lap.update(format="teks", encoding=enc, pemisah=sep, header=header, lewati=lewati)
    if sep is None and ext == ".txt":  # dokumen biasa: satu baris = satu sel
        baris = p.read_bytes().decode(enc, errors="replace").splitlines()[lewati:]
        lap["catatan"].append("tidak ada pemisah kolom: tiap baris jadi satu sel di kolom 'teks'")
        return pl.DataFrame({"teks": baris}, schema={"teks": pl.String})
    lap["pemisah"] = sep = sep or ","
    utf8 = enc in ("utf-8", "utf-8-sig")
    # polars hanya membaca utf-8: encoding lain diubah dulu di memori
    src = str(p) if utf8 else io.BytesIO(p.read_bytes().decode(enc, errors="replace").encode("utf-8"))
    # Semua kolom dibaca sebagai teks dulu; angka ditentukan tebak_tipe agar nol depan tidak hilang diam-diam.
    df = pl.read_csv(src, separator=sep, has_header=header, skip_rows=lewati, infer_schema=False,
                     truncate_ragged_lines=True, encoding="utf8-lossy", quote_char='"')
    if utf8 and len(raw) == CONTOH:  # hanya 1 MB awal yang dicek; sisanya dibaca lossy
        rusak = sum(df[c].str.contains("�", literal=True).sum() for c in df.columns)
        if rusak:
            lap["catatan"].append(f"{rusak} sel memuat byte bukan utf-8 (jadi '�'); coba atur encoding")
    return df


def _json(p, lap):
    lap["format"] = "json"
    with open(p, "rb") as f:
        awal = f.read(4096).lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
    if awal == b"[":  # daftar record: jalur cepat polars
        try:
            return pl.read_json(p)
        except Exception:
            pass
    data = json.loads(p.read_text(encoding="utf-8-sig", errors="replace"))
    if isinstance(data, dict):
        kunci = next((k for k, v in data.items() if isinstance(v, list)), None)
        if kunci is None:
            return _dari_records([data], lap)
        lap["catatan"].append(f"objek JSON: yang dipakai daftar di kunci '{kunci}'")
        data = data[kunci]
    return _dari_records(data, lap)


def _jsonl(p, lap):
    lap["format"] = "jsonl"
    try:
        return pl.read_ndjson(p, infer_schema_length=None)
    except Exception:
        pass
    rows, rusak = [], 0
    for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rusak += 1
    if rusak:
        lap["catatan"].append(f"{rusak} baris bukan JSON, dilewati")
    return _dari_records(rows, lap)


def _dari_records(data, lap):
    if not all(isinstance(r, dict) for r in data):
        return pl.DataFrame({"nilai": [_sebagai_teks(r) for r in data]}, schema={"nilai": pl.String})
    try:
        return pl.from_dicts(data, infer_schema_length=None)
    except Exception:
        lap["catatan"].append("tipe campur antarbaris: nilai bersarang disimpan sebagai teks JSON")
        kolom = list(dict.fromkeys(k for r in data for k in r))
        return pl.DataFrame({k: [_sebagai_teks(r.get(k)) for r in data] for k in kolom},
                            schema={k: pl.String for k in kolom})


def _sebagai_teks(v):
    return v if v is None or isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _daftar_sheet(p):
    from openpyxl import load_workbook

    wb = load_workbook(p, read_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def _excel(p, sheet, opsi, lap):
    from openpyxl import load_workbook

    wb = load_workbook(p, read_only=True, data_only=True)
    try:
        ws = wb[sheet] if sheet else wb.worksheets[0]
        rows = [r for r in ws.iter_rows(values_only=True)]
        judul = ws.title
    finally:
        wb.close()
    lewati = int(opsi.get("lewati") or 0)
    header = opsi.get("header", True)
    lap.update(format="xlsx", sheet=judul, header=header, lewati=lewati)
    rows = rows[lewati:]
    while rows and all(v is None for v in rows[-1]):
        rows.pop()
    if not rows:
        return pl.DataFrame()
    lebar = max(len(r) for r in rows)
    nama = [str(v) if v not in (None, "") else "" for v in rows[0]] if header else [""] * lebar
    nama += [""] * (lebar - len(nama))
    isi = rows[1:] if header else rows
    nama = _unikkan(nama, lap)
    return pl.DataFrame([_kolom_excel(n, [r[j] if j < len(r) else None for r in isi]) for j, n in enumerate(nama)])


def _kolom_excel(nama, vals):
    tipe = {type(v) for v in vals if v is not None}
    if tipe and tipe <= {int, float}:
        return pl.Series(nama, vals, dtype=pl.Float64 if float in tipe else pl.Int64)
    if len(tipe) == 1 and tipe != {str}:
        try:
            return pl.Series(nama, vals)
        except Exception:
            pass
    return pl.Series(nama, [None if v is None else str(v) for v in vals], dtype=pl.String)


def _nama_unik(df, lap):
    baru = _unikkan(df.columns, lap)
    return df.rename(dict(zip(df.columns, baru))) if baru != df.columns else df


def _unikkan(nama, lap):
    """Nama kolom kosong -> kolom_i; nama ganda -> nama_2, nama_3 (dicatat di laporan)."""
    baru, dipakai = [], set()
    for i, n in enumerate(nama):
        dasar = n.strip() or f"kolom_{i + 1}"
        cand, k = dasar, 2
        while cand in dipakai:
            cand, k = f"{dasar}_{k}", k + 1
        dipakai.add(cand)
        baru.append(cand)
    if baru != list(nama):
        lap["catatan"].append("nama kolom kosong/ganda diganti: " + ", ".join(
            f"{a!r}→{b!r}" for a, b in zip(nama, baru) if a != b))
    return baru
