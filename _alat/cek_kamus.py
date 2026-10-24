"""Jalankan semua blok ```python di kamus/*.md (tiap blok terpisah) di folder sementara.

Pakai: python _alat/cek_kamus.py   → cetak keluaran tiap blok, akhirnya jumlah blok yang error.
"""
import contextlib
import glob
import io
import os
import re
import tempfile
import traceback

KAMUS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kamus")

failed = 0
total = 0
work_folder = tempfile.mkdtemp(prefix="cek_kamus_")
os.chdir(work_folder)
for path in sorted(glob.glob(f"{KAMUS_FOLDER}/*.md")):
    text = open(path, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    for number, block in enumerate(blocks, 1):
        total += 1
        output = io.StringIO()
        print(f"===== {os.path.basename(path)} blok {number}")
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(block, f"{os.path.basename(path)}#{number}", "exec"), {})
            print(output.getvalue(), end="")
        except Exception:
            failed += 1
            print(output.getvalue(), end="")
            print("ERROR:\n" + traceback.format_exc(limit=1))
print(f"\n{total} blok, {failed} error (folder kerja: {work_folder})")
