"""Ekspor resep ke skrip .py / notebook .ipynb mandiri (E1). Skrip memuat salinan muat.py + bantu.py, jadi hasilnya
sama persis dengan yang terlihat di Laya, dan jalan tanpa Laya (cukup polars; openpyxl untuk xlsx)."""
import json
import os
import sys
import textwrap
from pathlib import Path

from mesin import bantu
from mesin.resep import acuan


def _sumber(nama):
    if getattr(sys, "frozen", False):
        return (Path(sys._MEIPASS) / "sumber" / nama).read_text(encoding="utf-8")
    return (Path(__file__).parent / nama).read_text(encoding="utf-8")


def _urutan(mesin, target):
    """Kelompok (pemilik + pasangan) yang perlu dihitung, acuan lebih dulu."""
    urut, sudah = [], set()

    def kunjungi(nama):
        pem, r = mesin.resep(nama)
        if pem in sudah:
            return
        sudah.add(pem)
        grup = mesin.kelompok(pem)
        perlu = set()
        for t in grup:
            perlu |= set(acuan(mesin.turunan().get(t)))
        for l in r["langkah"]:
            if l.get("nyala", True):
                perlu |= set(acuan(l["kode"]))
        for t in sorted(perlu - set(grup)):
            kunjungi(t)
        urut.append(pem)

    kunjungi(target)
    return urut


def skrip(mesin, target, keluar):
    folder = mesin.proyek["folder"]
    sel = [f"# %% [markdown]\n# Pembersihan data dari Laya: {target}\n# Ubah FOLDER/KELUAR bila folder data pindah.",
           "# %%\nimport ast\nimport math\nimport os\nimport re\n\nimport polars as pl\n\n"
           f"FOLDER = {folder!r}  # UBAH bila perlu\nKELUAR = {str(keluar)!r}  # UBAH bila perlu",
           "# %%\n# ---- salinan mesin/muat.py (cara baca file Laya) ----\n" + _sumber("muat.py"),
           "# %%\n# ---- salinan mesin/bantu.py (fungsi bantu langkah) ----\n" + _sumber("bantu.py"),
           "# %%\nT = {}"]
    for pem in _urutan(mesin, target):
        _, r = mesin.resep(pem)
        grup = mesin.kelompok(pem)
        bagian = [f"# %%\n# ---- {pem}" + (f" (+ {', '.join(grup[1:])})" if grup[1:] else "") + " ----"]
        for t in grup:
            if t in mesin.turunan():
                f = "turunan_" + bantu.nama_aman(t)
                bagian.append(f"def {f}():\n{textwrap.indent(mesin.turunan()[t], '    ')}\n    return df\n\n\n"
                              f"T[{t!r}] = {f}()")
            else:
                e = mesin.entri[t]
                rel = Path(os.path.relpath(e["path"], folder)).as_posix()
                opsi = mesin.proyek.get("opsi_baca", {}).get(t, {})
                bagian.append(f"T[{t!r}] = baca(os.path.join(FOLDER, {rel!r}), {e['sheet']!r}, {opsi!r})[0]")
        daftar = []
        for i, l in enumerate(r["langkah"], 1):
            if not l.get("nyala", True):
                continue
            f = f"langkah_{bantu.nama_aman(pem)}_{i}"
            bagian.append(f"def {f}(df, LATIH):\n{textwrap.indent(l['kode'], '    ')}\n    return df")
            daftar.append(f"({f}, {bool(l.get('hanya_latih'))})")
        if daftar:
            bagian.append(f"jalankan_resep(T, {pem!r}, {grup[1:]!r}, [\n    " + ",\n    ".join(daftar) + "])")
        sel.append("\n\n\n".join(bagian))
    tulis = "\n".join(f"T[{t!r}].write_parquet(os.path.join(KELUAR, {bantu.nama_aman(t) + '.parquet'!r}))"
                      for t in mesin.kelompok(target))
    sel.append(f"# %%\nos.makedirs(KELUAR, exist_ok=True)\n{tulis}\n"
               f"print({{k: T[k].shape for k in {mesin.kelompok(target)!r}}})\n"
               f"# df = T[{target!r}].to_pandas()  # kalau model butuh pandas")
    return "\n\n\n".join(sel) + "\n"


def ekspor_py(mesin, target, path, keluar):
    Path(path).write_text(skrip(mesin, target, keluar), encoding="utf-8")


def ekspor_ipynb(mesin, target, path, keluar):
    sel = []
    for blok in skrip(mesin, target, keluar).split("# %%")[1:]:
        md = blok.startswith(" [markdown]")
        isi = blok.split("\n", 1)[1].strip("\n") if "\n" in blok else ""
        if md:
            isi = "\n".join(b.removeprefix("# ").removeprefix("#") for b in isi.splitlines())
        sel.append({"cell_type": "markdown" if md else "code", "metadata": {}, "source": isi.splitlines(True),
                    **({} if md else {"execution_count": None, "outputs": []})})
    nb = {"cells": sel, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                     "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def simpan_tabel(df, folder, nama):
    """E2: parquet selalu; csv juga kalau semua kolom datar."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    keluar = [folder / (bantu.nama_aman(nama) + ".parquet")]
    df.write_parquet(keluar[0])
    if all(not (t.is_nested()) for t in df.dtypes):
        keluar.append(folder / (bantu.nama_aman(nama) + ".csv"))
        df.write_csv(keluar[1])
    return keluar
