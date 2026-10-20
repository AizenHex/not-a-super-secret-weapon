"""Uji pembuat kode EDA: buat kode untuk 2 skenario lalu JALANKAN kodenya (data kecil).
Kode hasil wajib tanpa Laya: baris 'import laya' / 'import torch' = gagal.
Jalankan: python _uji/test_codegen.py (dari folder _laya_eda atau mana saja)."""
import os
import shutil
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
APP_FOLDER = os.path.dirname(HERE)
sys.path.insert(0, APP_FOLDER)
import codegen

# folder kerja 2 tingkat di bawah GEMASTIK-DATABASE, seperti folder notebook toolkit
RUN_FOLDER = os.path.join(os.path.dirname(APP_FOLDER), "_uji_codegen_tmp")
shutil.rmtree(RUN_FOLDER, ignore_errors=True)
os.makedirs(RUN_FOLDER)

properti = pd.read_csv(os.path.join(APP_FOLDER, "contoh", "properti.csv")).head(40)
properti.to_csv(os.path.join(RUN_FOLDER, "properti_kecil.csv"), index=False)
emot = pd.read_csv(os.path.join(APP_FOLDER, "..", "..", "data_latihan", "text",
                                "classification", "emot_train.csv")).head(30)
emot.to_csv(os.path.join(RUN_FOLDER, "emot_kecil.csv"), index=False)
emot_test = pd.read_csv(os.path.join(APP_FOLDER, "..", "..", "data_latihan", "text",
                                     "classification", "emot_test.csv")).head(20)
emot_test.to_csv(os.path.join(RUN_FOLDER, "emot_test_kecil.csv"), index=False)

configs = {
    "properti": {
        "csv_path": "properti_kecil.csv", "test_csv": None,
        "columns": ["judul", "deskripsi", "harga", "alamat", "luas_m2"],
        "label_column": "jenis_asli", "number_columns": ["harga", "luas_m2"],
        "indonesian_format": True, "use_log": True, "ratio": ["harga", "luas_m2"],
        "manual_drop": {3: "harga 'hubungi wa'", 7: ""}, "new_labels": {5: "kost"},
        "drop_rules": ["duplikat", "teks_kosong", "tidak_terbaca", "negatif", "outlier"],
        "outlier_columns": ["harga", "harga_per_luas_m2"], "outlier_category": "jenis_asli",
        "features": ["gaya", "negasi", "panjang_kolom", "kosong", "log_angka", "rasio_median",
                     "frekuensi"],
        "category_columns": ["alamat"], "date_columns": [],
    },
    "emot": {
        "csv_path": "emot_kecil.csv", "test_csv": "emot_test_kecil.csv",
        "columns": ["text"], "label_column": "label", "number_columns": [],
        "indonesian_format": True, "use_log": True, "ratio": None,
        "manual_drop": {}, "new_labels": {}, "drop_rules": ["duplikat", "teks_kosong"],
        "outlier_columns": [], "outlier_category": None,
        "features": ["gaya", "negasi", "panjang_kolom", "rasio_median"],
        "category_columns": [], "date_columns": [],
    },
}

environment = dict(os.environ)
failed = 0
long_lines = 0   # semua baris kode hasil harus <= 100 karakter
for name in configs:
    cells = codegen.build_cells(configs[name])
    script_path = os.path.join(RUN_FOLDER, f"eda_{name}.py")
    with open(script_path, "w", encoding="utf-8") as file:
        file.write(codegen.to_script(cells))
    with open(os.path.join(RUN_FOLDER, f"eda_{name}.ipynb"), "w", encoding="utf-8") as file:
        file.write(codegen.to_notebook(cells))
    for line in codegen.to_script(cells).splitlines():
        if len(line) > 100:
            long_lines = long_lines + 1
        if "import laya" in line or "import torch" in line:
            print("  kode memanggil Laya/torch:", line)
            failed = failed + 1
    done = subprocess.run([sys.executable, script_path], cwd=RUN_FOLDER, env=environment,
                          capture_output=True, text=True)
    print(f"== {name}: kode {len(cells)} sel, baris >100: {long_lines}, exit {done.returncode}")
    if done.returncode != 0:
        failed = failed + 1
        print(done.stderr[-2500:])
    else:
        print(done.stdout[-800:])

notebook = os.path.join(RUN_FOLDER, "eda_emot.ipynb")
done = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                       "--execute", "--inplace", notebook], cwd=RUN_FOLDER, env=environment,
                      capture_output=True, text=True)
print("== ipynb emot dijalankan nbconvert: exit", done.returncode)
if done.returncode != 0:
    failed = failed + 1
    print(done.stderr[-2500:])
if long_lines > 0:
    failed = failed + 1
result = "LOLOS"
if failed > 0:
    result = f"GAGAL {failed}"
print("HASIL:", result)
shutil.rmtree(RUN_FOLDER)
