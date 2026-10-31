"""Uji ketahanan pipeline di data buatan yang sengaja berantakan.

Jalankan dari folder _laya_eda_v2:  python -m tests.test_robust
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import recipe as rcp
from core.codegen import build_script
from core.runner import load_script


def messy(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "harga": rng.choice(["Rp 1.500.000", "2.000.000,5", "-", "?", "750.000", None], rows),
        "luas": np.where(rng.random(rows) < 0.05, 2_147_483_647, rng.lognormal(4, 1, rows).round()),
        "sama": np.zeros(rows),                                   # sebaran nol → outlier tanpa batas
        "kosong_semua": np.full(rows, np.nan),
        "kota": rng.choice(["Kab. Pati", "Kabupaten Pati", " kota  semarang", "-", None], rows),
        "bahan": rng.choice(["Genteng & asbas.", "gentng", "seng/asbes", "GENTENG dan Seng", "", None], rows),
        "punya": rng.choice(["Ya", None], rows),
        "label": rng.integers(0, 3, rows),
    })


def recipe() -> dict:
    r = rcp.new_recipe("label")
    r["number_columns"] = ["harga"]
    r["text_columns"] = ["kota", "bahan", "punya"]
    r["sentinels"] = {"luas": rcp.SENTINEL_FLOOR}
    r["blank_as"] = {"punya": "Tidak"}
    r["category_maps"] = {"kota": {"Kab. Pati": "Kabupaten Pati", "kota semarang": "Kota Semarang"}}
    r["free_text"] = {"bahan": {"gentng": "genteng", "asbas": "asbes"}}
    r["outliers"] = {"luas": {"method": "mad_log", "factor": 3.5, "action": "clip"},
                     "sama": {"method": "mad", "factor": 3.5, "action": "drop"}}
    r["impute"] = {"harga": {"strategy": "median"}, "kosong_semua": {"strategy": "median"},
                   "kota": {"strategy": "most_frequent"}}
    r["encoding"] = {"kota": {"method": "onehot", "min_share": 0.0},
                     "bahan": {"method": "materials", "min_share": 0.05},
                     "punya": {"method": "ordered", "order": ["Tidak", "Ya"]}}
    r["split"] = {"method": "kfold", "valid_size": 0.2, "folds": 3, "seed": 0}
    return r


def main() -> None:
    module = load_script(build_script(recipe(), {}))
    train = messy(400, 0)
    test = messy(50, 1).drop(columns=["label"])
    test.loc[0, "kota"] = "Kota Baru Sekali"                     # kategori yang tidak ada di train
    test.loc[1, "bahan"] = "jerami"                              # bahan baru → _lainnya
    result = module.run(train, test)
    out, out_test = result["train"], result["test"]
    print(out.head().to_string())

    assert len(out) == len(train), "'sama' sebaran nol tidak boleh membuang baris"
    assert result["state"]["bounds"]["sama"] == (-np.inf, np.inf)
    assert out["harga"].notna().all() and out["harga"].max() == 2_000_000.5
    assert out["luas"].max() < rcp.SENTINEL_FLOOR
    assert "kosong_semua" in out.columns, "kolom kosong semua tetap ada (keep_empty_features)"
    assert any("kosong_semua" in w for w in result["warnings"]), "harus ada peringatan kolom kosong semua"
    assert {"kota_Kabupaten Pati", "kota_Kota Semarang"} <= set(out.columns)
    assert not any("Kab. Pati" in c for c in out.columns)
    assert {"bahan_genteng", "bahan_asbes", "bahan_seng", "bahan_lainnya"} <= set(out.columns)
    assert set(out["punya"]) == {0.0, 1.0}
    assert out_test.loc[0, [c for c in out.columns if c.startswith("kota_")]].sum() == 0, "kota baru → semua 0"
    assert out_test.loc[1, "bahan_lainnya"] == 1
    assert len(out_test) == len(test)

    single = module.transform(test.iloc[[2]], result["state"])
    assert single.shape == (1, out_test.shape[1]), "satu baris harus bisa diproses"
    print("folds:", module.split(out).value_counts().to_dict())
    print("SEMUA UJI KETAHANAN LULUS")


if __name__ == "__main__":
    main()
