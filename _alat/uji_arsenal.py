"""Uji potongan arsenal: gabungkan sel kode terpilih (bisa dari beberapa file), ganti baris
`NAMA = ...`, lalu jalankan di folder kerja di luar repo.

Pakai:
  python _alat/uji_arsenal.py arsenal/vision/deteksi/anotasi_baca.ipynb:1,5 \
      arsenal/vision/deteksi/anotasi_tulis.ipynb \
      --set COCO_JSON='"Z:/data/train/_annotations.coco.json"' --set IMAGE_FOLDER='"Z:/data/train"'

file[:sel] : sel = nomor sel KODE (mulai 1, markdown tidak dihitung). Tanpa :sel = semua sel.
--set      : NAMA=ekspresi Python, mengganti ruas kanan SEMUA baris `NAMA = ...` di sel terpilih.
--name     : nama folder kerja (default: nama file terakhir).
Folder kerja: ../outputs/uji_arsenal/<nama>/ (path relatif di potongan jatuh di sini).
"""
import argparse
import os
import re
import subprocess
import sys

from sel import baca_teks_sel

parser = argparse.ArgumentParser()
parser.add_argument("specs", nargs="+")
parser.add_argument("--set", action="append", default=[])
parser.add_argument("--name", default="")
parser.add_argument("--define", action="append", default=[])   # kode Python, ditaruh di awal
parser.add_argument("--tail", action="append", default=[])     # kode Python, ditaruh di akhir
args = parser.parse_args()

replacements = {}
for item in args.set:
    name, value = item.split("=", 1)
    replacements[name.strip()] = value


def code_cells_of(file_name):
    text = baca_teks_sel(file_name)
    parts = re.split(r"^(# %%.*)$", text, flags=re.M)
    cells = []
    for index in range(1, len(parts), 2):
        if "[markdown]" not in parts[index]:
            cells.append(parts[index + 1])
    return cells


# plt.show() → simpan fig_1.png, fig_2.png, ... di folder kerja (untuk dilihat)
script = ["import matplotlib", "matplotlib.use('Agg')", "import matplotlib.pyplot as _plt",
          "_figs = [0]",
          "def _show():\n    _figs[0] += 1\n    _plt.savefig(f'fig_{_figs[0]}.png', dpi=80)",
          "    _plt.close('all')",
          "_plt.show = _show"]
script = script + args.define
used = set()
last_file = ""
for spec in args.specs:
    file_name, _, selection = spec.rpartition(":") if re.search(r":[\d,]+$", spec) else (spec, "", "")
    cells = code_cells_of(file_name)
    chosen = list(range(1, len(cells) + 1))
    if selection:
        chosen = [int(number) for number in selection.split(",")]
    last_file = file_name
    for number in chosen:
        script.append(f"print('=== {os.path.basename(file_name)} sel {number}', flush=True)")
        for line in cells[number - 1].splitlines():
            match = re.match(r"^([A-Z_][A-Z0-9_]*) = ", line)
            if match and match.group(1) in replacements:
                used.add(match.group(1))
                line = f"{match.group(1)} = {replacements[match.group(1)]}"
            script.append(line)
missing = set(replacements) - used
if missing:
    sys.exit(f"--set tidak ditemukan di sel terpilih: {sorted(missing)}")

script = script + args.tail
name = args.name or os.path.splitext(os.path.basename(last_file))[0]
here = os.path.dirname(os.path.abspath(__file__))
work = os.path.normpath(os.path.join(here, "..", "..", "outputs", "uji_arsenal", name))
os.makedirs(work, exist_ok=True)
with open(os.path.join(work, "uji.py"), "w", encoding="utf-8") as file:
    file.write("\n".join(script))
env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
result = subprocess.run([sys.executable, "uji.py"], cwd=work, env=env)
print("LOLOS" if result.returncode == 0 else "GAGAL", " ".join(args.specs), "| kerja:", work)
sys.exit(result.returncode)
