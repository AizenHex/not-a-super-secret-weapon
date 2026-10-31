"""Uji generator kode di data riil: resep penuh, test tanpa label, kode bisa dijalankan mandiri.

Jalankan dari folder _laya_eda_v2:  python -m tests.test_codegen
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core import recipe as rcp
from core.codegen import build_script, to_notebook
from core.runner import load_script

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "_laya_eda" / "data" / "train.csv"
LABEL = "Golongan UKT"
ROOF = "Bahan Atap Rumah"
SENTINEL_COLUMNS = ["Pengeluaran Bulanan Keluarga", "Jumlah (Perkiraan Harga Pasar) Kendaraan yang Dimiliki",
                    "Luas Tanah Rumah (m2)", "NJOP per Meter", "Total Hutang", "Cicilan Hutang per Bulan"]


def full_recipe() -> dict:
    r = rcp.new_recipe(LABEL)
    r["drop_columns"] = ["id", "Lokasi Tempat Tinggal"]
    r["text_columns"] = ["Pendidikan Orang Tua/Wali", "Kota", ROOF, "Riwayat Beasiswa atau Bantuan Pendidikan",
                         "Status Kepesertaan Program Bantuan Pemerintah", "Program Studi", "Jalur Masuk"]
    r["sentinels"] = {c: rcp.SENTINEL_FLOOR for c in SENTINEL_COLUMNS}
    r["blank_as"] = {"Status Kepesertaan Program Bantuan Pemerintah": "Tidak"}
    r["category_maps"] = {"Pendidikan Orang Tua/Wali": {"S- 3": "S-3", "S3": "S-3"},
                          "Kota": {"Kab. Semarang": "Kabupaten Semarang"}}
    r["free_text"] = {ROOF: {"gentng": "genting", "genteng": "genting", "asbas": "asbes", "bta": "bata"}}
    r["outliers"] = {"Luas Tanah Rumah (m2)": {"method": "iqr_log", "factor": 1.5, "action": "clip"},
                     "NJOP per Meter": {"method": "mad_log", "factor": 3.5, "action": "flag"},
                     "Total Hutang": {"method": "iqr", "factor": 3.0, "action": "nan"},
                     "Daya Listrik (Watt)": {"method": "mad", "factor": 3.5, "action": "drop"}}
    r["drop_duplicates"] = True
    r["max_missing_per_row"] = 10
    r["drop_queries"] = ["`Luas Bangunan Rumah (m2)` > 100000"]
    r["impute"] = {"Luas Tanah Rumah (m2)": {"strategy": "median"},
                   "Total Hutang": {"strategy": "median"},
                   "Jumlah Tanggungan dalam Keluarga": {"strategy": "most_frequent"},
                   "Pekerjaan Orang Tua/Wali": {"strategy": "constant", "value": "Tidak diketahui"},
                   "Kepemilikan Aset": {"strategy": "most_frequent"}}
    r["encoding"] = {"Pekerjaan Orang Tua/Wali": {"method": "onehot", "min_share": 0.01},
                     "Kepemilikan Aset": {"method": "ordinal"},
                     "Pendapatan Keluarga": {"method": "ordered", "order": None},
                     ROOF: {"method": "materials", "min_share": 0.01},
                     "Program Studi": {"method": "string", "n_components": 5}}
    r["split"] = {"method": "stratified", "valid_size": 0.2, "folds": 5, "seed": 42}
    return r


def main() -> None:
    train = pd.read_csv(DATA)
    test = train.drop(columns=[LABEL]).sample(300, random_state=1).reset_index(drop=True)
    # test berantakan: urutan kolom acak, satu kolom hilang, satu kolom tambahan, angka sebagai teks
    test = test.sample(frac=1, axis=1, random_state=2).drop(columns=["Kota"])
    test["kolom_aneh"] = 1
    test["Total Hutang"] = test["Total Hutang"].map(lambda v: f"{v:,.0f}".replace(",", ".") if pd.notna(v) else v)

    r = full_recipe()
    r["number_columns"] = ["Total Hutang"]
    r["encoding"]["Pendapatan Keluarga"]["order"] = sorted(train["Pendapatan Keluarga"].dropna().unique())
    script = build_script(r, {"train": str(DATA), "test": None, "output": "hasil_uji"})
    module = load_script(script)
    result = module.run(train, test)
    out_train, out_test = result["train"], result["test"]

    print("train", train.shape, "->", out_train.shape, "| test", test.shape, "->", out_test.shape)
    print("dibuang per alasan:", result["drop_reasons"].sum().to_dict())
    print("peringatan:", result["warnings"])
    assert any("Kota" in w for w in result["warnings"]) and any("kolom_aneh" in w for w in result["warnings"])
    assert len(out_test) == len(test), "baris test tidak boleh dibuang"
    assert list(out_test.columns) == [c for c in out_train.columns if c != LABEL], "kolom test != train"
    for column in train.columns.drop(["id", "Lokasi Tempat Tinggal"]):
        encoded = column in r["encoding"]
        assert encoded or column in out_train.columns, f"kolom asli hilang: {column}"
    roof = [c for c in out_train.columns if c.startswith(ROOF)]
    print("kolom atap:", roof)
    assert f"{ROOF}_genting" in roof and f"{ROOF}_lainnya" in roof
    assert not any("gentng" in c or "genteng" in c for c in roof)
    assert out_train[SENTINEL_COLUMNS[:2]].max().max() < rcp.SENTINEL_FLOOR, "sentinel masih ada"
    low, high = result["state"]["bounds"]["Luas Tanah Rumah (m2)"]
    land = out_train["Luas Tanah Rumah (m2)"]
    assert land[land > 0].between(low, high).all(), "clip gagal"      # mode log: nol tidak disentuh
    assert (land == 0).sum() == (train.loc[out_train.index, "Luas Tanah Rumah (m2)"] == 0).sum()
    assert out_test["Total Hutang"].notna().any(), "angka teks '1.500.000' gagal dibaca"
    assert out_train["Status Kepesertaan Program Bantuan Pemerintah"].isna().sum() == 0

    # joblib: simpan, muat, hasil identik
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "state.joblib"
        joblib.dump(result["state"], path)
        again = module.transform(test, joblib.load(path))
        pd.testing.assert_frame_equal(again, module.transform(test, result["state"]))

        # kode berdiri sendiri: python pipeline_eda.py
        file = Path(folder) / "pipeline_eda.py"
        file.write_text(build_script(r, {"train": str(DATA), "test": None, "output": str(Path(folder) / "hasil")}),
                        encoding="utf-8")
        run = subprocess.run([sys.executable, str(file)], capture_output=True, text=True, cwd=folder)
        print(run.stdout.strip() or run.stderr[-2000:])
        assert run.returncode == 0
        assert (Path(folder) / "hasil" / "valid_part.csv").exists()
        notebook = to_notebook(script)
        assert '"cell_type": "code"' in notebook

    # resep kosong = data tidak diubah selain buang kolom kosong
    empty = load_script(build_script(rcp.new_recipe(LABEL), {})).run(train)
    assert empty["train"].shape == train.shape
    pd.testing.assert_frame_equal(empty["train"][train.columns].astype(object),
                                  train.astype(object), check_dtype=False)
    print("SEMUA UJI LULUS")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
