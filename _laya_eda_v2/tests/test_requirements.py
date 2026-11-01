"""Uji requirement R1–R18 (VISI.md bagian 1, kolom "Cara memastikan") di data sintetis + test berantakan buatan.

Jalankan dari folder _laya_eda_v2:  python -m tests.test_requirements   (satu uji: ... test_r12_test_berantakan)
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tests import bantu, sintetis

from core import ops, split, steps  # noqa: E402
from core.ops import OPS  # noqa: E402

LABEL = bantu.LABEL
ROOT = bantu.ROOT
SPLIT = {"method": "acak", "share": 0.2, "seed": 42, "stratify": True}


def frame(n: int = 3000) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = sintetis.buat(n)
    test = sintetis.buat_test(train)
    for df in (train, test):                         # satu kolom angka-teks untuk op Teks → angka
        df["Biaya (teks)"] = (df["Pengeluaran Bulanan Keluarga"].fillna(0) // 1000 * 1000).map("{:,.0f}".format) \
            .str.replace(",", ".")
    return train, test


def all_ops() -> list[dict]:
    """Satu langkah untuk SETIAP op di katalog (kecuali AutoGluon generator: diuji terpisah)."""
    p = steps.new_project(LABEL)
    add = lambda op, **k: steps.add_step(p, op, k)  # noqa: E731 (uji)
    texts = ["Kota", "Pendidikan Orang Tua/Wali", "Riwayat Beasiswa atau Bantuan Pendidikan", "Bahan Atap Rumah"]
    add("tidy_text", columns=texts)
    add("change_case", columns=["Jenis Kelamin"], case="lower")
    add("to_number", columns=["Biaya (teks)"], indonesian=True)
    add("spelling", groups={"Pendidikan Orang Tua/Wali": {"sltasederajat": ["ma", "sma"]}, "Kota": {}})
    add("fuzzy_spelling", columns=["Kota"], min_count=5, cutoff=90)
    add("map_values", column="Pekerjaan Orang Tua/Wali", mapping={"Petani": "Tani"})
    add("free_text", column="Bahan Atap Rumah", words={"gentng": "genteng", "asbas": "asbes"})
    add("sentinel_to_nan", columns=sintetis.SENTINEL_COLUMNS)
    add("zero_to_nan", columns=["Daya Listrik"])
    add("fill_constant", columns=["Status Kepesertaan Program Bantuan Pemerintah"], value="Tidak")
    add("impute", columns=["Luas Tanah Rumah (m2)"], strategy="median")
    add("drop_duplicates", reason="uji: baris kembar")
    add("drop_missing_rows", max_missing=5)
    add("drop_rows_query", rule="`Jumlah Tanggungan dalam Keluarga` > 100")
    add("outlier", column="NJOP per Meter", method="iqr_log", factor=3.0, action="clip")
    add("outlier", column="Luas Bangunan Rumah (m2)", method="mad", factor=3.5, action="drop")
    add("outlier", column="Pengeluaran Bulanan Keluarga", method="iqr", factor=3.0, action="nan")
    add("log_transform", columns=["Total Hutang"])
    add("scale", columns=["Cicilan Hutang per Bulan"], method="standard")
    add("winsorize", columns=["Rerata Pengeluaran Listrik & Air per Bulan"], method="iqr", fold=1.5)
    add("power", columns=["Jumlah (Perkiraan Harga Pasar) Kendaraan yang Dimiliki"])
    add("rare_label", columns=["Pekerjaan Orang Tua/Wali"], tol=0.05)
    add("ordinal", column="Pendapatan Keluarga", order=sintetis.INCOME)
    add("onehot", column="Jalur Masuk", min_share=0.01)
    add("materials_onehot", column="Bahan Atap Rumah", min_share=0.01)
    add("target_encode", columns=["Kota"], kind="kelas")
    add("onehot_sklearn", columns=["Fakultas"], min_share=0.01)
    add("skrub_encoder", column="Program Studi", kind="string", n_components=4)
    add("drop_columns", columns=["Lokasi Tempat Tinggal"], reason="uji: kembar dengan Kota")
    used = {step["op"] for step in p["steps"]}
    assert used == set(OPS) - {"autogluon_features"}, set(OPS) - used
    return p["steps"]


def run_all(train, valid, test, chosen=None):
    return steps.run(chosen or all_ops(), LABEL, train, steps.align(train, test, LABEL), valid=valid)


def no_functions(code: str) -> bool:
    return not [n for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.FunctionDef, ast.Lambda, ast.ClassDef))]


# ---------------------------------------------------------------- R1–R5

def test_r1_proyek_kosong_tidak_mengubah_apa_pun():
    train, test = frame()
    before = pd.util.hash_pandas_object(train).sum()
    part, valid = split.apply(SPLIT, LABEL, train)
    result = steps.run([], LABEL, part, test, valid=valid)
    pd.testing.assert_frame_equal(result["train"], part)
    pd.testing.assert_frame_equal(result["valid"], valid)
    pd.testing.assert_frame_equal(result["test"], test)
    assert pd.util.hash_pandas_object(train).sum() == before


def test_r2_statistik_hanya_dari_train():
    train, test = frame()
    part, valid = split.apply(SPLIT, LABEL, train)
    assert len(valid) and len(test)
    first = run_all(part, valid, test)
    assert first["error"] is None, first["error"]
    other_valid, other_test = valid.copy(), test.copy()
    for df in (other_valid, other_test):             # isi valid/test diubah total
        for column in df.columns.drop([LABEL, "id"], errors="ignore"):
            df[column] = df[column].sample(frac=1, random_state=5).to_numpy()
        df["Kota"] = "Kota Baru Sekali"
        df["NJOP per Meter"] = df["NJOP per Meter"] * 1000
    second = run_all(part, other_valid, other_test)
    assert second["error"] is None, second["error"]
    pd.testing.assert_frame_equal(first["train"], second["train"])
    assert [e["keluaran"] for e in first["log"]] == [e["keluaran"] for e in second["log"]], "parameter langkah berubah"


def test_r3_baris_valid_dan_test_tidak_dibuang():
    train, test = frame()
    part, valid = split.apply(SPLIT, LABEL, train)
    result = run_all(part, valid, test)
    assert result["error"] is None, result["error"]
    assert len(valid) > 0 and len(test) > 0
    assert len(result["valid"]) == len(valid) and len(result["test"]) == len(test)
    assert result["valid"].index.equals(valid.index) and len(result["train"]) < len(part)


def test_r3_autogluon_generator():
    bantu.perlu_modul("autogluon")
    train, test = frame(1500)
    part, valid = split.apply(SPLIT, LABEL, train)
    chosen = all_ops()[:11] + [{"op": "autogluon_features", "params": {}, "on": True}]
    result = run_all(part, valid, test, chosen)
    assert result["error"] is None, result["error"]
    assert len(result["valid"]) == len(valid) and len(result["test"]) == len(test)


def test_r4_skrip_tanpa_def_lambda_class():
    project = {"label": LABEL, "steps": all_ops(), "split": SPLIT}
    assert no_functions(steps.build_script(project, "train.csv", "test.csv"))


def test_r5_skrip_mandiri_sama_dengan_aplikasi(tmp_path):
    train, test = frame()
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    project = {"label": LABEL, "steps": all_ops(), "split": SPLIT}
    (tmp_path / "pipa.py").write_text(steps.build_script(project, "train.csv", "test.csv"), encoding="utf-8")
    run = subprocess.run([sys.executable, "pipa.py"], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    raw_train, raw_test = pd.read_csv(tmp_path / "train.csv"), pd.read_csv(tmp_path / "test.csv")
    part, valid = split.apply(SPLIT, LABEL, raw_train)
    app = run_all(part, valid, raw_test)
    for name in ("train", "valid", "test"):
        standalone = pd.read_csv(tmp_path / f"{name}_bersih.csv")
        pd.testing.assert_frame_equal(standalone, app[name].reset_index(drop=True), check_dtype=False)


# ---------------------------------------------------------------- R6–R10

def test_r6_entri_peta_dihitung_dan_dibatasi():
    spelling = {"op": "spelling", "params": {"groups": {"Pendidikan": {"slta": ["ma", "sma"]}, "Kota": {}}}}
    words = {"op": "free_text", "params": {"column": "Bahan", "words": {"gentng": "genteng", "geting": "genteng",
                                                                        "asbas": "asbes"},
                                           "phrases": {"bata merah": "batu bata"}}}
    mapping = {"op": "map_values", "params": {"column": "Kota", "mapping": {"a": "b", "c": "d"}}}
    assert ops.map_entries(steps.step_code(spelling)) == 3          # 1 kunci + 2 isi list
    assert ops.map_entries(steps.step_code(words)) == 4             # (?:geting|gentng) = 2, (?:asbas) = 1, frasa 1
    assert ops.map_entries(steps.step_code(mapping)) == 2
    assert ops.map_entries(steps.step_code({"op": "tidy_text", "params": {"columns": ["Kota"]}})) == 0
    current = [{**spelling, "on": True}, {**words, "on": True}]
    big = {"op": "map_values", "params": {"column": "Kota", "mapping": {f"v{i}": "x" for i in range(294)}}}
    assert steps.map_budget(current, big)["total"] == 301 and not steps.map_budget(current, big)["ok"]
    small = {"op": "map_values", "params": {"column": "Kota", "mapping": {f"v{i}": "x" for i in range(293)}}}
    assert steps.map_budget(current, small)["ok"]
    project = {"label": LABEL, "steps": all_ops(), "split": SPLIT}
    script = steps.build_script(project, "train.csv")
    assert ops.map_entries(script) <= ops.MAP_LIMIT and len(script.splitlines()) < 400 + 200


def test_r7_satu_library_per_tugas():
    assert set(ops.LIBRARY) == set(OPS), set(OPS) ^ set(ops.LIBRARY)
    sample = {step["op"]: step for step in all_ops()}
    for name, library in ops.LIBRARY.items():
        if library not in ("pandas",) and name in sample:
            module = {"numpy": "np."}.get(library, library)
            assert module in steps.step_code(sample[name]), (name, library)


def test_r8_nama_kolom_asli_tidak_diganti():
    train, test = frame()
    part, valid = split.apply(SPLIT, LABEL, train)
    current = {"train": part, "valid": valid, "test": steps.align(part, test, LABEL)}
    for step in all_ops():
        before = list(current["train"].columns)
        result = steps.run([step], LABEL, current["train"], current["test"], valid=current["valid"])
        assert result["error"] is None, result["error"]
        after = list(result["train"].columns)
        removed = ops.removed_columns(step)
        lost = [c for c in before if c not in after]
        added = [c for c in after if c not in before]
        assert sorted(lost) == sorted(removed), (step["op"], lost, removed)
        assert all(any(str(c).startswith(f"{source}_") for source in removed) for c in added), (step["op"], added)
        assert [c for c in before if c in after] == [c for c in after if c in before], "urutan kolom asli tetap"
        current = result


def test_r9_semua_pilihan_punya_apa_kapan_bahaya():
    from core import training
    for catalog in (OPS, training.MODELS, training.IMBALANCE):
        for name, spec in catalog.items():
            assert spec["what"] and spec["fits"] and spec["danger"], name


def test_r10_semua_tab_punya_flowchart():
    from ui.flows import FLOWS, TAB_FLOWS
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    tabs = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TABS" for t in node.targets))
    names = [key.value for key in tabs.keys]
    assert len(names) == 14 and set(names) == set(TAB_FLOWS), set(names) ^ set(TAB_FLOWS)
    assert all(TAB_FLOWS[name] in FLOWS for name in names)
    assert "Latih" in names and "Submission" in names


# ---------------------------------------------------------------- R11–R15

def test_r11_galat_tampil_dan_berhenti():
    train, _ = frame(200)
    chosen = all_ops()[:3] + [{"op": "drop_columns", "params": {"columns": ["tidak ada"]}, "on": True}] + all_ops()[3:6]
    result = steps.run(chosen, LABEL, train, None)
    assert result["error"]["langkah"] == 4 and "KeyError" in result["error"]["galat"]
    assert len(result["log"]) == 4, "langkah sesudah galat tidak dijalankan"


def messy_variants(test: pd.DataFrame) -> dict:
    base = test.copy()
    spelled = base.copy()
    spelled.loc[spelled.index[:20], "Kota"] = ["Kab.  Patii ", "KOTA SEMARANG", "Kota Bandung Baru", "Ꮶota Ǥ"] * 5
    spelled.loc[spelled.index[:20], "Pendidikan Orang Tua/Wali"] = "S - 3"
    spelled.loc[spelled.index[:20], "Bahan Atap Rumah"] = "genteng & ijuk"
    spelled.loc[spelled.index[:20], "Pekerjaan Orang Tua/Wali"] = "Astronot"
    typed = base.copy()
    for column in ("Luas Tanah Rumah (m2)", "NJOP per Meter", "Total Hutang"):
        typed[column] = typed[column].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}".replace(",", "."))
    empty = base.copy()
    empty["Kota"], empty["Luas Tanah Rumah (m2)"], empty["Bahan Atap Rumah"] = np.nan, np.nan, np.nan
    return {
        "kategori/ejaan baru": spelled,
        "kolom hilang": base.drop(columns=["Fakultas", "Kota", "NJOP per Meter"]),
        "kolom tambahan": base.assign(kolom_asing=1, **{"kolom asing 2": "x"}),
        "urutan kolom beda": base[list(base.columns[::-1])],
        "tipe beda ('1.500' teks)": typed,
        "kolom kosong semua": empty,
        "test 1 baris": base.iloc[:1],
        "test 0 baris": base.iloc[:0],
    }


def test_r12_test_berantakan():
    train, test = frame()
    part, valid = split.apply(SPLIT, LABEL, train)
    for name, variant in messy_variants(test).items():
        result = run_all(part, valid, variant)
        assert result["error"] is None, (name, result["error"])
        assert len(result["test"]) == len(variant), name
        print(f"{name}: test {variant.shape} -> {result['test'].shape}")
    typed = steps.align(part, messy_variants(test)["tipe beda ('1.500' teks)"], LABEL)
    got, want = typed["Luas Tanah Rumah (m2)"], test["Luas Tanah Rumah (m2)"].round()
    assert got.dtype == float and ((got == want) | (got.isna() & want.isna())).all(), "'1.500' = 1500"


def test_r12_skrip_mandiri_test_berantakan(tmp_path):
    train, test = frame()
    train.to_csv(tmp_path / "train.csv", index=False)
    project = {"label": LABEL, "steps": all_ops(), "split": SPLIT}
    (tmp_path / "pipa.py").write_text(steps.build_script(project, "train.csv", "test.csv"), encoding="utf-8")
    variants = messy_variants(test)
    for name in ("kolom hilang", "tipe beda ('1.500' teks)", "test 0 baris", "kategori/ejaan baru"):
        variants[name].to_csv(tmp_path / "test.csv", index=False)
        run = subprocess.run([sys.executable, "pipa.py"], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, (name, run.stderr[-2000:])
        assert len(pd.read_csv(tmp_path / "test_bersih.csv")) == len(variants[name]), name


def test_r13_cp1252_dan_karakter_asing(tmp_path):
    column = "Kota ✓ Ł"
    train = pd.DataFrame({column: ["Kota Łódź", "kota łódź ", "Ǥenteng", "-", None] * 20, "y": [0, 1] * 50})
    train.to_csv(tmp_path / "train.csv", index=False)
    project = {"label": "y", "steps": [
        {"op": "tidy_text", "params": {"columns": [column]}, "on": True},
        {"op": "spelling", "params": {"groups": {column: {}}}, "on": True},       # mencetak nama kolom
        {"op": "fuzzy_spelling", "params": {"columns": [column]}, "on": True}], "split": None}
    script = steps.build_script(project, "train.csv")
    assert 'sys.stdout.reconfigure(errors="replace")' in script.split("\n# %%")[0], "blok pertama"
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    (tmp_path / "pipa.py").write_text(script, encoding="utf-8")
    run = subprocess.run([sys.executable, "pipa.py"], cwd=tmp_path, capture_output=True, env=env)
    assert run.returncode == 0, run.stderr.decode("utf-8", "replace")[-1500:]
    (tmp_path / "tanpa.py").write_text(script.replace('sys.stdout.reconfigure(errors="replace")', "pass"),
                                       encoding="utf-8")
    broken = subprocess.run([sys.executable, "tanpa.py"], cwd=tmp_path, capture_output=True, env=env)
    assert broken.returncode != 0, "tanpa reconfigure skrip seharusnya gagal (uji ini bermakna)"


def test_r14_reproducible_dan_versi(tmp_path):
    train, test = frame(1500)
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    project = {"label": LABEL, "steps": all_ops(), "split": SPLIT}
    script = steps.build_script(project, "train.csv", "test.csv")
    assert "versi: pandas" in script.split("\n# %%")[0] and "sklearn.__version__" in script
    (tmp_path / "pipa.py").write_text(script, encoding="utf-8")
    outputs = []
    for _ in range(2):
        subprocess.run([sys.executable, "pipa.py"], cwd=tmp_path, check=True, capture_output=True)
        outputs.append([(tmp_path / f"{n}_bersih.csv").read_bytes() for n in ("train", "valid", "test")])
    assert outputs[0] == outputs[1]
    for step in all_ops():
        code = steps.step_code(step)
        if any(word in code for word in ("GapEncoder", "TargetEncoder", "KFold", "SMOTE")):
            assert "random_state" in code, step["op"]
    assert "random_state=42" in split.split_code(SPLIT)


def test_r15_sebelum_sesudah_dan_alasan():
    train, test = frame()
    part, valid = split.apply(SPLIT, LABEL, train)
    result = run_all(part, valid, test)
    for entry in result["log"]:
        for key in ("baris train", "kolom baru", "kolom hilang", "sel berubah", "kosong baru", "baris valid", "baris test"):
            assert key in entry, (entry["langkah"], key)
    assert any(entry["sel berubah"] for entry in result["log"])
    drops = [s for s in all_ops() if s["op"].startswith("drop_") or s["params"].get("action") == "drop"]
    assert len(drops) == 5
    for step in drops:
        code = steps.step_code(step)
        assert code.startswith("# alasan: ") or "\n# alasan: " in code, step["op"]
        if step["params"].get("reason"):
            assert step["params"]["reason"] in code


# ---------------------------------------------------------------- R16–R18

def test_r16_tanpa_laya_dan_library_berat_hanya_saat_dipakai():
    child = ("import sys\nsys.path.insert(0, '.')\n"
             "import core.ops, core.steps, core.audit, core.text_tools, core.split, core.ag, core.pipeline\n"
             "import core.training, core.schema, core.ensemble, core.diagnose, core.submission\n"
             "import ui.common, ui.tab_clean, ui.tab_code, ui.tab_explore, ui.tab_laya, ui.tab_model, ui.tab_split\n"
             "import ui.tab_train, ui.tab_submit\n"
             "heavy = ['autogluon', 'skrub', 'lightgbm', 'xgboost', 'catboost', 'torch', 'sentence_transformers',\n"
             "         'laya', 'pandera', 'feature_engine', 'imblearn', 'shap', 'matplotlib']\n"
             "print(sorted(name for name in heavy if name in sys.modules))\n")
    run = subprocess.run([sys.executable, "-c", child], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "[]", run.stdout
    if importlib.util.find_spec("laya") is not None:
        raise bantu.Lewati("Laya terpasang di venv ini: 'jalan tanpa Laya' dibuktikan oleh test_app di venv tanpa Laya")


def test_r17_rerun_kurang_dari_2_detik():
    """AppTest dengan train sintetis 30.380 baris (ukuran data riil): ganti widget → rerun < 2 s sesudah cache hangat."""
    import tempfile

    from streamlit.testing.v1 import AppTest

    paths = bantu.folder_sintetis(30380)
    os.environ["LAYA_TRAIN"], os.environ["LAYA_TEST"] = str(paths["train"]), str(paths["test"])
    os.environ["LAYA_SESI"] = str(Path(tempfile.mkdtemp()) / "sesi_r17.json")
    try:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
        app.run()
        chosen = [step for step in all_ops()[:13] if step["op"] != "to_number"]    # 'Biaya (teks)' tidak ada di file
        app.session_state["project"] = {"label": LABEL, "steps": chosen, "split": SPLIT}
        timings = {}
        for page, change in (("Data", lambda a: a.selectbox(key="insight_column").select("Kota")),
                             ("Outlier", lambda a: a.radio(key="out_mode").set_value("satu kolom (rinci)")),
                             ("Kode & ekspor", lambda a: a.text_input(key="train_path_input").set_value("x.csv"))):
            app.radio(key="page").set_value(page).run()          # hitung + cache (pertama kali boleh lama)
            app.run()
            start = time.time()
            change(app).run()
            timings[page] = time.time() - start
            assert not app.exception, app.exception[0].value
            assert not app.error, [e.value for e in app.error]
        print("rerun (s):", {k: round(v, 2) for k, v in timings.items()})
        slow = {k: v for k, v in timings.items() if v >= 2}
        assert not slow, slow
    finally:
        for key in ("LAYA_TRAIN", "LAYA_TEST"):
            os.environ.pop(key, None)


def test_r18_semua_tabel_lewat_show_table():
    offenders = []
    for path in (ROOT / "ui").glob("*.py"):
        if path.name == "kamus.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    node.func.attr in ("dataframe", "data_editor", "table"):
                offenders.append(f"{path.name}:{node.lineno} .{node.func.attr}")
    assert not offenders, offenders
    from ui import kamus
    try:
        kamus.show_table(pd.DataFrame({"kolom tanpa arti xyz": [1]}))
        raise AssertionError("kolom tanpa arti harus ditolak")
    except ValueError as error:
        assert "R18" in str(error)


if __name__ == "__main__":
    bantu.main(globals())
