"""Uji 'umum' G1–G7 (VISI.md bagian 0b): Laya di 10 dataset tabular selain data UKT.

Data di `_laya_eda/data/umum/` (tidak di-commit; diunduh dari cermin GitHub bila belum ada, lihat tests/umum.py).
Tanpa jaringan dan tanpa file = uji dataset itu dilewati.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

from tests import bantu, umum

from core import findings, reader, steps, vocab  # noqa: E402
from core.ops import OPS  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = bantu.ROOT


def load(name: str) -> tuple[pd.DataFrame, str, dict, list]:
    path = umum.path(name)
    if path is None:
        raise bantu.Lewati(f"dataset {name} tidak tersedia (tanpa jaringan / file unggahan tidak ada)")
    options, notes = reader.sniff(path)
    data = reader.read(path, options)
    label = umum.DATASETS[name][1] or data.columns[-1]
    return data, label, options, notes


def test_g5_baca_umum():
    for name, expect_header in (("adult", False), ("heart", True), ("titanic", True)):
        data, label, options, notes = load(name)
        assert options["judul"] is expect_header, (name, options)
        assert not any(str(c).startswith("\ufeff") for c in data.columns), (name, "BOM di nama kolom")
    data, _, options, notes = load("adult")
    assert list(data.columns[:2]) == ["kolom_1", "kolom_2"] and len(data) == 48842, "baris pertama = data, bukan judul"
    assert "tanpa_judul" in {n["jenis"] for n in notes}
    data, _, options, notes = load("ames")
    meaning = next(n for n in notes if n["jenis"] == "kosong_bermakna")
    kept = reader.read(umum.path("ames"), {**options, "na_teks": meaning["kolom"]})
    assert kept[meaning["kolom"][0]].isin(["None", "NA"]).any(), "'None'/'NA' tetap teks bila dipilih"
    folder = Path(tempfile.mkdtemp())
    (folder / "a.csv").write_text("a;b\n1,5;x\n2,5;y\n", encoding="utf-8")
    options, _ = reader.sniff(folder / "a.csv")
    assert options["sep"] == ";" and reader.read(folder / "a.csv", options)["a"].tolist() == [1.5, 2.5]


def test_g5_teks_panjang_berkutip():
    """Teks berkutip + baris baru melewati 64 KB pertama, 'é' terbelah di batas 64 KB: dulu EOF inside string."""
    folder = Path(tempfile.mkdtemp())
    rows = ["id,teks,label"] + [f'{i},"pendek {i}",a' for i in range(12)]
    rows.append('12,"awal\n' + "x" * 65530 + 'é\nbaris kedua, ada koma",b')   # baris baru SEBELUM batas 64 KB
    rows += [f'{i},"ulasan {i}\nlanjut",a' for i in range(13, 30)]
    for source in (folder / "t.csv", "\n".join(rows).encode("utf-8")):
        if isinstance(source, Path):
            source.write_text("\n".join(rows), encoding="utf-8")
        options, _ = reader.sniff(source)
        assert options["encoding"] == "utf-8-sig" and options["sep"] == "," and options["judul"], options
        data = reader.read(source, options)
        assert data.shape == (30, 3) and data["teks"][12].endswith("ada koma"), data.shape


def test_g1_temuan_umum_di_10_dataset():
    from ui.kamus import meaning_of

    missing_meaning, results = [], {}
    for name, (_, _, expected) in umum.DATASETS.items():
        try:
            data, label, options, notes = load(name)
        except bantu.Lewati as reason:
            results[name] = f"dilewati ({reason})"
            continue
        items = findings.scan(data, label) + notes
        kinds = {item["jenis"] for item in items}
        assert expected <= kinds, (name, expected - kinds)
        for item in items:
            assert item["pilihan"] and item["judul"] and item["alasan"], (name, item["jenis"])
            for choice in item["pilihan"]:
                assert choice["op"] is None or choice["op"] in OPS, (name, choice)
            detail = item.get("detail")
            if detail is not None:
                missing_meaning += [f"{name}:{c}" for c in detail.columns if meaning_of(c) is None]
        results[name] = sorted(kinds)
    print(json.dumps(results, indent=1, ensure_ascii=False))
    assert not missing_meaning, missing_meaning


def test_g1_g6_semua_usulan_bisa_dijalankan():
    """Setiap pilihan aksi setiap temuan di setiap dataset dijalankan sebagai langkah: tanpa galat, baris valid/test
    tidak berkurang (R3), dan skrip mandiri = aplikasi (R5) untuk gabungan usulan pertama."""
    from core import split

    config = {"method": "acak", "share": 0.2, "seed": 0, "stratify": False}
    for name in umum.DATASETS:
        try:
            data, label, options, notes = load(name)
        except bantu.Lewati:
            continue
        part, valid = split.apply(config, label, data)
        test = part.drop(columns=[label]).sample(min(200, len(part)), random_state=0)
        for item in findings.scan(part, label):            # setiap pilihan setiap temuan, satu per satu
            for choice in item["pilihan"]:
                if choice["op"] is None:
                    continue
                step = {"op": choice["op"], "params": choice["params"], "on": True}
                result = steps.run([step], label, part, test, valid=valid)
                assert result["error"] is None, (name, item["judul"], choice["judul"], result["error"])
                assert len(result["valid"]) == len(valid) and len(result["test"]) == len(test), (name, choice)
        chosen, done, current = [], set(), {"train": part, "valid": valid, "test": test}
        for _ in range(30):                                  # seperti aplikasi: tambah usulan pertama, pindai ulang
            pending = [item for item in findings.scan(current["train"], label)
                       if item["pilihan"][0]["op"] and (item["jenis"], tuple(item["kolom"])) not in done]
            if not pending:
                break
            item = pending[0]
            done.add((item["jenis"], tuple(item["kolom"])))
            step = {"op": item["pilihan"][0]["op"], "params": item["pilihan"][0]["params"], "on": True}
            chosen.append(step)
            current = steps.run([step], label, current["train"], current["test"], valid=current["valid"])
            assert current["error"] is None, (name, item["judul"], current["error"])
        result = steps.run(chosen, label, part, test, valid=valid)
        assert result["error"] is None, (name, result["error"])
        print(f"{name}: {len(chosen)} usulan pertama berurutan OK → train {result['train'].shape}")


def test_g4_kamus_proyek():
    vocab.use(vocab.UMUM)
    project = {"label": "y", "steps": [], "split": None}
    steps.add_step(project, "spelling", {"groups": {"kota": {}}})
    steps.add_step(project, "free_text", {"column": "bahan", "words": {}})
    script = steps.build_script(project, "train.csv")
    for word in ("kabupaten", "\\bdan\\b", "genteng"):
        assert word not in script, f"kata UKT '{word}' tidak boleh ada di proyek umum"
    vocab.use(vocab.PRESET_UKT)
    project = {"label": "y", "steps": [], "split": None}
    steps.add_step(project, "spelling", {"groups": {"kota": {}}})
    assert "kabupaten" in steps.build_script(project, "train.csv")
    assert vocab.separator_regex() == vocab.LEGACY_SEPARATORS, "preset UKT = pemisah lama"
    vocab.use(vocab.UMUM)


def test_g3_kode_bebas():
    data = pd.DataFrame({"a": [1.0, 2.0, None], "y": [0, 1, 0]})
    good = {"op": "custom_code", "params": {"title": "x2", "code": "for df in (train, valid, test):\n    df['a_x2'] = df['a'] * 2"},
            "on": True}
    result = steps.run([good], "y", data, None)
    assert result["error"] is None and result["train"]["a_x2"].tolist()[:2] == [2.0, 4.0]
    for bad in ("f = lambda x: x", "def f():\n    pass", "class A:\n    pass"):
        result = steps.run([{"op": "custom_code", "params": {"code": bad}, "on": True}], "y", data, None)
        assert result["error"] and "R4" in result["error"]["galat"], bad
    impute = {"op": "impute", "params": {"columns": ["a"], "strategy": "median"}, "on": True}
    converted = {"op": "custom_code", "params": {"title": "dari impute", "code": steps.step_code(impute)}, "on": True}
    pd.testing.assert_frame_equal(steps.run([impute], "y", data, None)["train"], steps.run([converted], "y", data, None)["train"])


def test_g2_g3_g7_aplikasi_di_dataset_lain():
    """AppTest per dataset (proses terpisah): semua halaman tanpa galat, usulan Temuan dengan parameter DIUBAH
    tersimpan, langkah dibuka lewat ✎ lalu diubah jadi kode bebas, Latih jalan, skrip mandiri jalan."""
    chosen = [name for name in ("titanic", "taxis", "ames", "adult", "fitness") if umum.path(name)]
    if not chosen:
        raise bantu.Lewati("dataset umum tidak tersedia")
    failed = {}
    for name in chosen:
        run = subprocess.run([sys.executable, "-m", "tests.app_umum", name], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, "LAYA_KAMUS_KETAT": "1"}, timeout=1500)
        lines = [line for line in run.stdout.splitlines() if line.startswith(("OK", "GAGAL", "info"))]
        print(name, "|", " ; ".join(lines)[-600:])
        if run.returncode != 0:
            failed[name] = (run.stdout + run.stderr)[-1500:]
    assert not failed, failed


if __name__ == "__main__":
    bantu.main(globals())
