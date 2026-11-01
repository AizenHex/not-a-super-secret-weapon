"""Mesin langkah: semua operasi di data riil + test berantakan, skrip tanpa `def`, skrip mandiri = hasil aplikasi."""
import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import split, steps  # noqa: E402
from core.ops import OPS  # noqa: E402
from tests import bantu  # noqa: E402

TRAIN = bantu.train_csv()          # data riil bila ada, kalau tidak data sintetis (tests/sintetis.py)
LABEL = bantu.LABEL


def project() -> dict:
    data = pd.read_csv(TRAIN)
    texts = [c for c in data.columns if data[c].dtype == object and c != LABEL]
    p = steps.new_project(LABEL)
    add = lambda op, **params: steps.add_step(p, op, params)  # noqa: E731
    add("tidy_text", columns=texts)
    add("sentinel_to_nan", columns=list(data.select_dtypes("number").columns.drop(LABEL, errors="ignore")))
    add("free_text", column="Bahan Atap Rumah", words={"gentng": "genteng", "asbas": "asbes"})
    add("spelling", groups={"Pendidikan Orang Tua/Wali": {"sltasederajat": ["ma", "sma"]}, "Kota": {}},
        examples={"Pendidikan Orang Tua/Wali": {"sltasederajat": "SLTA Sederajat ← SMA, MA"}})
    add("map_values", column=texts[0], mapping={"zzz tidak ada": "y"})
    add("outlier", column="Luas Tanah Rumah (m2)", method="iqr_log", factor=3.0, action="clip")
    add("outlier", column="Luas Bangunan Rumah (m2)", method="mad", factor=3.5, action="drop")
    add("drop_duplicates")
    add("drop_missing_rows", max_missing=15)
    add("impute", columns=["Luas Tanah Rumah (m2)"], strategy="median")
    add("materials_onehot", column="Bahan Atap Rumah", min_share=0.01)
    add("ordinal", column=texts[1], order=[])
    add("onehot", column=texts[2], min_share=0.02)
    return p


def test_all_ops_have_text():
    for name, op in OPS.items():
        assert op["what"] and op["fits"] and op["danger"], name


def test_script_has_no_functions():
    script = steps.build_script(project(), str(TRAIN))
    tree = ast.parse(script)
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.Lambda, ast.ClassDef))]


def test_run_real_and_messy_test():
    train = pd.read_csv(TRAIN)
    test = train.drop(columns=[LABEL]).sample(500, random_state=1).reset_index(drop=True)
    test.iloc[0, 3] = "nilai baru sekali"
    test = test.drop(columns=[test.columns[-1]])                    # kolom hilang di test
    p = project()
    p["steps"] = [s for s in p["steps"]]
    result = steps.run(p["steps"], LABEL, train, test.reindex(columns=train.columns.drop(LABEL)))
    assert result["error"] is None, result["error"]
    assert len(result["test"]) == 500                               # test tidak pernah dibuang
    assert LABEL in result["train"] and len(result["train"]) < len(train)
    assert "Bahan Atap Rumah_lainnya" in result["test"]


def test_empty_project_changes_nothing():
    train = pd.read_csv(TRAIN)
    result = steps.run([], LABEL, train, None)
    pd.testing.assert_frame_equal(result["train"], train)


def test_error_is_reported_not_hidden():
    train = pd.read_csv(TRAIN, nrows=50)
    result = steps.run([{"op": "drop_columns", "params": {"columns": ["tidak ada"]}, "on": True}], LABEL, train, None)
    assert result["error"] and "KeyError" in result["error"]["galat"]


def test_standalone_script_matches_app(tmp_path):
    p = project()
    script = steps.build_script(p, str(TRAIN))
    (tmp_path / "pipa.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, "pipa.py"], cwd=tmp_path, check=True, capture_output=True)
    standalone = pd.read_csv(tmp_path / "train_bersih.csv")
    app = steps.run(p["steps"], LABEL, pd.read_csv(TRAIN), None)["train"].reset_index(drop=True)
    pd.testing.assert_frame_equal(standalone, app, check_dtype=False)


SPLIT = {"method": "acak", "share": 0.2, "seed": 42, "stratify": True}


def test_split_partitions_rows():
    """Setiap metode: train + valid = semua baris, tidak ada yang dobel; kelompok tidak terbelah."""
    train = pd.read_csv(TRAIN)
    text = [c for c in train.columns if train[c].dtype == object and c != LABEL][0]
    configs = [SPLIT, {**SPLIT, "stratify": False},
               {"method": "kelompok", "share": 0.2, "seed": 1, "column": text},
               {"method": "kelompok", "share": 0.2, "seed": 1, "column": split.ROW_GROUP}]
    for config in configs:
        part, valid = split.apply(config, LABEL, train)
        assert not part.index.intersection(valid.index).size, config
        assert sorted(part.index.append(valid.index)) == list(train.index), config
        assert 0.1 < len(valid) / len(train) < 0.3, (config, len(valid))
        if config["method"] == "kelompok" and config["column"] != split.ROW_GROUP:
            assert not set(part[text].astype(str)) & set(valid[text].astype(str))
    part, valid = split.apply(SPLIT, LABEL, train)            # stratified: proporsi kelas hampir sama
    gap = (part[LABEL].value_counts(normalize=True) - valid[LABEL].value_counts(normalize=True)).abs().max()
    assert gap < 0.01, gap
    print("baris valid kembar dengan train (acak):", split.twins_across(part, valid, LABEL))


def test_split_by_time():
    dates = pd.date_range("2020-01-01", periods=100, freq="D").astype(str).tolist()
    train = pd.DataFrame({"tanggal": dates[:95] + ["bukan tanggal"] * 5, "x": range(100), "y": [0, 1] * 50})
    part, valid = split.apply({"method": "waktu", "share": 0.2, "seed": 42, "column": "tanggal"}, "y", train)
    latest_train = pd.to_datetime(part["tanggal"], errors="coerce").max()
    assert len(valid) == 19 and pd.to_datetime(valid["tanggal"]).min() > latest_train, (len(valid), latest_train)
    assert (part["tanggal"] == "bukan tanggal").sum() == 5                # waktu kosong tetap di train
    part, valid = split.apply({"method": "waktu", "share": 0.2, "seed": 42, "column": "x", "numeric": True}, "y", train)
    assert valid["x"].min() > part["x"].max()


def test_statistics_from_train_part_only():
    """Median pengisi dari bagian train SAJA; baris valid tidak pernah dibuang."""
    train = pd.read_csv(TRAIN)
    column = "Luas Tanah Rumah (m2)"
    part, valid = split.apply(SPLIT, LABEL, train)
    p = steps.new_project(LABEL)
    steps.add_step(p, "impute", {"columns": [column], "strategy": "median"})
    steps.add_step(p, "drop_missing_rows", {"max_missing": 3})
    result = steps.run(p["steps"], LABEL, part, None, valid=valid)
    assert result["error"] is None, result["error"]
    holes = valid[column].isna()
    assert holes.any()
    assert (result["valid"].loc[holes, column] == part[column].median()).all()
    assert result["valid"].index.equals(valid.index)                   # valid utuh, walau train dibuang barisnya
    assert len(result["train"]) < len(part)


def test_standalone_script_with_split_matches_app(tmp_path):
    p = project()
    p["split"] = SPLIT
    script = steps.build_script(p, str(TRAIN))
    assert script.index("# %% split data") < script.index("# %% langkah 1")
    (tmp_path / "pipa.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, "pipa.py"], cwd=tmp_path, check=True, capture_output=True)
    part, valid = split.apply(SPLIT, LABEL, pd.read_csv(TRAIN))
    app = steps.run(p["steps"], LABEL, part, None, valid=valid)
    assert app["error"] is None, app["error"]
    assert len(app["valid"]) == len(valid)
    for name in ("train", "valid"):
        standalone = pd.read_csv(tmp_path / f"{name}_bersih.csv")
        pd.testing.assert_frame_equal(standalone, app[name].reset_index(drop=True), check_dtype=False)


def test_synthetic_messy():
    import numpy as np
    train = pd.DataFrame({
        "harga": ["Rp 1.500.000", "2.000,5", "-", None, "Rp 3.000.000", "abc"] * 5,
        "kota": [" Kab. Pati", "Kabupaten  Pati", "?", "Kota Semarang", None, "null"] * 5,
        "kosong": [np.nan] * 30,
        "luas": [0, 10, 12, 2_147_483_647, 11, 9_999] * 5,
        "y": ["a", "b"] * 15,
    })
    test = pd.DataFrame({"harga": ["5.000"], "kota": ["Kota Baru Sekali"], "luas": [13]})   # kolom 'kosong' hilang
    p = steps.new_project("y")
    add = lambda op, **params: steps.add_step(p, op, params)  # noqa: E731
    add("to_number", columns=["harga"], indonesian=True)
    add("tidy_text", columns=["kota"])
    add("sentinel_to_nan", columns=["luas"])
    add("outlier", column="luas", method="iqr_log", factor=1.5, action="nan")
    add("impute", columns=["harga", "luas"], strategy="median")
    add("onehot", column="kota", min_share=0.0)
    result = steps.run(p["steps"], "y", train, test.reindex(columns=train.columns.drop("y")))
    assert result["error"] is None, result["error"]
    assert result["train"]["harga"].iloc[0] == 1_500_000 and result["train"]["harga"].iloc[1] == 2000.5
    assert result["test"]["kota_lainnya"].iloc[0] == 1 and len(result["test"]) == 1
    assert result["train"]["luas"].max() < 2e9


if __name__ == "__main__":
    bantu.main(globals())
