"""Kumpulkan bahan presentasi yang BOLEH keluar dari PC lomba ke ../outputs/bawa_pulang/.

Aturan panitia: hanya luaran notebook (PNG, tabel ringkas, catatan .md). Apa pun yang bisa dipakai
merekonstruksi data (tabel per baris, teks asli, OOF, data bersih) TIDAK disalin.

Pakai (dari root repo):
  python _alat/ekspor_hasil.py                          semua subfolder ../outputs/text
  python _alat/ekspor_hasil.py ../outputs/text/eda_lengkap ../outputs/text/topik_lengkap
  python _alat/ekspor_hasil.py --notebook hasil1.ipynb hasil2.ipynb --model ../outputs/text/x/model
Baca bawa_pulang/MANIFEST.md sesudahnya: daftar yang disalin, yang ditolak + alasan, peringatan.
"""
import argparse
import csv
import glob
import os
import re
import shutil
import time

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.abspath(os.path.join(HERE, "..", "..", "outputs"))
TARGET = os.path.join(OUTPUTS, "bawa_pulang")

NAMA_TERLARANG = ["topik_dokumen", "oof", "pseudo", "train_bersih", "test_bersih", "contoh",
                  "duplikat_contoh", "cocok_acuan"]   # cocok_acuan (fuzzy_kategori) = nilai asli kolom
KOLOM_TERLARANG = {"id", "baris", "text", "teks"}
MAKS_BARIS_CSV = 200
MAKS_KARAKTER_OUTPUT = 3000
MAKS_BARIS_TABEL = 30


def alasan_tolak_csv(path):
    """Alasan CSV ditolak, atau None bila tabel ringkas."""
    with open(path, encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    kolom = sorted({name.strip().lower() for name in header} & KOLOM_TERLARANG)
    if kolom:
        return f"ada kolom per baris/teks: {kolom}"
    if rows > MAKS_BARIS_CSV:
        return f"{rows} baris (> {MAKS_BARIS_CSV}): bukan tabel ringkas"
    return None


def alasan_tolak(path):
    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1]
    if ext not in (".png", ".md", ".csv"):
        return f"jenis file {ext or '(tanpa ekstensi)'} tidak dibawa"
    for kata in NAMA_TERLARANG:
        if kata in name:
            return f"nama mengandung '{kata}'"
    if ext == ".csv":
        return alasan_tolak_csv(path)
    return None


def peringatan_notebook(path):
    """Sel yang output-nya berpotensi memuat data mentah (teks panjang / tabel banyak baris)."""
    notebook = nbformat.read(path, as_version=4)
    hasil = []
    for number, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            text = output.get("text", "") or data.get("text/plain", "")
            html = data.get("text/html", "")
            if isinstance(text, list):
                text = "".join(text)
            if isinstance(html, list):
                html = "".join(html)
            baris_tabel = len(re.findall(r"<tr", html))
            if baris_tabel > MAKS_BARIS_TABEL:
                hasil.append(f"sel {number}: tabel html {baris_tabel} baris")
            elif len(text) > MAKS_KARAKTER_OUTPUT or len(html) > 10 * MAKS_KARAKTER_OUTPUT:
                hasil.append(f"sel {number}: output teks {max(len(text), len(html))} karakter")
    return hasil


def salin(path, subfolder, disalin):
    tujuan = os.path.join(TARGET, subfolder, os.path.basename(path))
    os.makedirs(os.path.dirname(tujuan), exist_ok=True)
    shutil.copy2(path, tujuan)
    disalin.append((os.path.relpath(tujuan, TARGET), os.path.getsize(tujuan)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="*", help="folder hasil (default: semua ../outputs/text/*)")
    parser.add_argument("--notebook", nargs="*", default=[], help="notebook ber-output yang dibawa")
    parser.add_argument("--model", default=None, help="folder model yang dibawa apa adanya")
    args = parser.parse_args()
    folders = args.folder or sorted(glob.glob(os.path.join(OUTPUTS, "text", "*")))
    folders = [os.path.abspath(folder) for folder in folders if os.path.isdir(folder)]

    if os.path.basename(TARGET) != "bawa_pulang":      # jangan pernah menghapus folder lain
        raise SystemExit(f"TARGET bukan bawa_pulang: {TARGET}")
    shutil.rmtree(TARGET, ignore_errors=True)
    os.makedirs(TARGET)

    disalin, ditolak, peringatan = [], [], []
    for folder in folders:
        nama = os.path.basename(folder)
        for root, _, files in os.walk(folder):
            for file in sorted(files):
                path = os.path.join(root, file)
                rel = os.path.join(nama, os.path.relpath(path, folder))
                if file.lower().startswith("submission") and file.lower().endswith(".csv"):
                    salin(path, os.path.join("submission", nama), disalin)
                    continue
                alasan = alasan_tolak(path)
                if alasan:
                    ditolak.append((rel, alasan))
                else:
                    salin(path, os.path.dirname(rel), disalin)
    for path in args.notebook:
        for item in peringatan_notebook(path):
            peringatan.append(f"{os.path.basename(path)}: {item}")
        salin(path, "notebook", disalin)
    if args.model:
        tujuan = os.path.join(TARGET, "model", os.path.basename(os.path.normpath(args.model)))
        shutil.copytree(args.model, tujuan)
        disalin.append((os.path.relpath(tujuan, TARGET) + os.sep, 0))

    lines = ["# MANIFEST bawa_pulang", "", f"Dibuat {time.strftime('%Y-%m-%d %H:%M')}.",
             "Hanya luaran ringkas (PNG, .md, tabel <= 200 baris tanpa kolom id/teks), file kumpul,",
             "notebook, dan model yang diminta. Periksa bagian PERINGATAN sebelum menyalin keluar.",
             "", f"## PERINGATAN ({len(peringatan)})"]
    lines += [f"- {item}: kosongkan output sel itu bila berisi data mentah" for item in peringatan]
    lines += ["", f"## Disalin ({len(disalin)})"]
    lines += [f"- `{rel}` ({size / 1024:.0f} KB)" for rel, size in disalin]
    lines += ["", f"## DITOLAK ({len(ditolak)})"]
    lines += [f"- `{rel}`: {alasan}" for rel, alasan in ditolak]
    with open(os.path.join(TARGET, "MANIFEST.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print(f"disalin {len(disalin)}, ditolak {len(ditolak)}, peringatan {len(peringatan)}")
    for item in peringatan:
        print("PERINGATAN:", item)
    print("lihat", os.path.join(TARGET, "MANIFEST.md"))


main()
