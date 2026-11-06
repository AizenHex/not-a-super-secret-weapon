"""Baca CSV umum (VISI.md G5): BOM, pemisah otomatis, file tanpa baris judul, 'NA' yang berarti 'tidak punya'.

Opsi baca = {"encoding", "sep", "judul": bool, "na_teks": [kolom yang 'NA'-nya teks, bukan kosong]}.
`read_code` menulis blok baca yang sama dengan yang dijalankan aplikasi (exec), jadi skrip ekspor membaca file
persis seperti aplikasi. `sniff` menebak opsi dari isi file + mengembalikan temuan baca (tanpa judul, 'NA').
"""
from __future__ import annotations

import codecs
import csv
import io

import pandas as pd

# token yang dibaca pandas sebagai kosong secara bawaan (selain sel kosong)
NA_TOKENS = ["#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A",
             "NA", "NULL", "NaN", "None", "n/a", "nan", "null"]
DEFAULT = {"encoding": "utf-8-sig", "sep": ",", "desimal": ".", "judul": True, "na_teks": []}


def _head(source, limit: int = 65536) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source[:limit])
    if hasattr(source, "getvalue"):
        return source.getvalue()[:limit]
    with open(source, "rb") as handle:
        return handle.read(limit)


def _stream(source):
    """Sumber → yang bisa dibaca pandas: path apa adanya, bytes/unggahan → BytesIO."""
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    if hasattr(source, "getvalue"):
        return io.BytesIO(source.getvalue())
    return source


def _is_number(text: str) -> bool:
    try:
        float(str(text).strip())
    except ValueError:
        return False
    return True


def sniff(source) -> tuple[dict, list[dict]]:
    """(opsi baca, temuan baca). Temuan memakai format core/findings.py."""
    from core.findings import finding, option

    raw = _head(source)
    encoding = "utf-8-sig"
    try:
        text = codecs.getincrementaldecoder("utf-8-sig")().decode(raw)   # huruf terpotong di ujung 64 KB bukan error
    except UnicodeDecodeError:
        encoding, text = "cp1252", raw.decode("cp1252", errors="replace")
    lines = "\n".join(text.splitlines()[:-1] or text.splitlines())        # baris terakhir bisa terpotong
    try:
        sep = csv.Sniffer().sniff(lines[:20000], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    # dari file utuh, bukan 64 KB pertama: potongan itu bisa berhenti di tengah teks berkutip (EOF inside string)
    rows = pd.read_csv(_stream(source), sep=sep, header=None, dtype=str, keep_default_na=False, nrows=200,
                       encoding=encoding, encoding_errors="replace")
    numeric = [c for c in rows.columns if rows[c].iloc[1:].map(_is_number).mean() >= 0.9]
    header = not numeric or sum(_is_number(rows.loc[0, c]) for c in numeric) < 0.8 * len(numeric)
    comma = sep != "," and any(rows[c].iloc[1:].str.fullmatch(r"-?\d+,\d+").mean() >= 0.5 for c in rows.columns)
    options = {**DEFAULT, "encoding": encoding, "sep": sep, "desimal": "," if comma else ".", "judul": bool(header)}
    notes = []
    if not header:
        notes.append(finding("tanpa_judul", [], "File tampaknya TIDAK punya baris judul",
                             f"Baris pertama berisi data (mis. {', '.join(rows.iloc[0, :4])}), bukan nama kolom. Kolom "
                             "diberi nama kolom_1, kolom_2, … Ubah di sidebar Data bila tebakan ini salah.",
                             [option("Sudah dipakai: baca tanpa judul", None)], None, 1))
    body = rows.iloc[1:] if header else rows
    names = list(rows.iloc[0]) if header else [f"kolom_{i + 1}" for i in range(rows.shape[1])]
    tokens = []
    for position, name in enumerate(names):
        cells = body.iloc[:, position]
        share = cells.isin(NA_TOKENS).mean()
        others = cells[~cells.isin(NA_TOKENS + [""])]
        if share >= 0.01 and len(others) and others.map(_is_number).mean() < 0.5:
            tokens.append({"kolom": name, "'NA' di 200 baris pertama": int(cells.isin(NA_TOKENS).sum())})
    if tokens:
        columns = [row["kolom"] for row in tokens]
        notes.append(finding("kosong_bermakna", columns, f"'NA' tertulis di {len(columns)} kolom teks",
                             "Di banyak data, 'NA' berarti 'tidak punya' (tidak ada gang, tidak ada kolam), bukan "
                             "data hilang. Pandas membacanya sebagai kosong. Pilih kolom yang 'NA'-nya teks di sidebar "
                             "Data (opsi baca), lalu 'NA' jadi kategori sendiri.",
                             [option("Atur di sidebar Data → 'NA' tetap teks", None)], pd.DataFrame(tokens), 2))
    return options, notes


def read_code(options: dict, source: str = "TRAIN_CSV", target: str = "train") -> str:
    """Blok baca lurus (tanpa def). `source` = nama variabel path/berkas, `target` = nama DataFrame."""
    options = {**DEFAULT, **(options or {})}
    args = f"encoding={options['encoding']!r}, sep={options['sep']!r}"
    if options.get("desimal", ".") != ".":
        args += f", decimal={options['desimal']!r}"
    lines = []
    if options["na_teks"]:
        lines += [f"NA_TEKS = {options['na_teks']!r}   # di kolom ini 'NA' = teks ('tidak punya'), bukan kosong",
                  f"TOKEN_KOSONG = {[''] + NA_TOKENS!r}   # sel kosong + token kosong bawaan pandas"]
    if not options["judul"]:
        lines.append(f"{target} = pd.read_csv({source}, {args}, header=None)   # file tanpa baris judul")
        lines.append(f'{target}.columns = ["kolom_" + str(nomor + 1) for nomor in range({target}.shape[1])]')
        if options["na_teks"]:
            lines.append(f"{target}[NA_TEKS] = {target}[NA_TEKS].fillna(\"NA\")   # kosong di kolom ini = 'NA'")
        return "\n".join(lines)
    if options["na_teks"]:
        lines.append(f"nama = pd.read_csv({source}, {args}, nrows=0).columns")
        if source not in ("TRAIN_CSV", "TEST_CSV"):         # berkas unggahan (BytesIO): kembali ke awal
            lines.append(f"if hasattr({source}, 'seek'):\n    {source}.seek(0)")
        lines.append("na_per_kolom = {kolom: TOKEN_KOSONG for kolom in nama if kolom not in NA_TEKS}")
        lines.append(f"{target} = pd.read_csv({source}, {args}, keep_default_na=False, na_values=na_per_kolom)")
        lines.append(f"for kolom in NA_TEKS:\n    {target}[kolom] = {target}[kolom].replace(\"\", float(\"nan\"))")
        return "\n".join(lines)
    return f"{target} = pd.read_csv({source}, {args})"


def read(source, options: dict | None = None) -> pd.DataFrame:
    """Jalankan read_code apa adanya (satu sumber dengan skrip)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    namespace = {"pd": pd, "SUMBER": source}
    exec(read_code(options or DEFAULT, "SUMBER", "data"), namespace)
    return namespace["data"]


__all__ = ["NA_TOKENS", "DEFAULT", "sniff", "read_code", "read"]
