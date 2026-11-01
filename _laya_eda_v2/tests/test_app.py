"""Uji aplikasi tanpa browser (streamlit AppTest): semua tab tampil tanpa galat, tanpa langkah dan dengan langkah.

Jalankan dari folder _laya_eda_v2:  python -m tests.test_app
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

# file simpan otomatis milik uji, BUKAN hasil/sesi_terakhir.json milik pengguna
SESSION = Path(tempfile.mkdtemp()) / "sesi_uji.json"
os.environ["LAYA_SESI"] = str(SESSION)

from tests import bantu
from tests.test_steps import LABEL, TRAIN, project

os.environ.setdefault("LAYA_TRAIN", str(TRAIN))
ROWS = sum(1 for _ in open(TRAIN, encoding="utf-8")) - 1


def check(app: AppTest, title: str) -> None:
    errors = [e.value for e in app.exception]
    shown = [e.value for e in app.error]
    print(f"{title}: {len(app.tabs)} tab, exception {len(errors)}, st.error {len(shown)}")
    assert not errors, errors[0]
    assert not shown, shown


def button(app: AppTest, start: str):
    return next(b for b in app.button if b.label.startswith(start))


def wait_for(app: AppTest, title: str, done, seconds: int = 600) -> None:
    """Tunggu tugas latar selesai (rerun berkala, seperti fragment di browser)."""
    start = time.time()
    while not done():
        assert time.time() - start < seconds, f"{title} tidak selesai dalam {seconds} dtk"
        time.sleep(2)
        app.run()
    check(app, f"{title} selesai ({time.time() - start:.0f} dtk menunggu)")


def go(app: AppTest, page: str) -> None:
    app.radio(key="page").set_value(page).run()
    check(app, f"halaman {page}")


def test_app() -> None:
    app = AppTest.from_file("../app.py", default_timeout=300)
    app.run()
    check(app, "tanpa langkah")
    pages = app.radio(key="page").options
    assert len(pages) == 12 and pages[1] == "Split", pages
    for page in pages:                                 # setiap halaman tampil tanpa galat, tanpa langkah
        go(app, page)

    app.session_state["project"] = copy.deepcopy(project())
    for page in pages:                                 # dan dengan langkah
        go(app, page)
    go(app, "Kode & ekspor")
    metrics = {m.label: m.value for m in app.metric}
    print("metrik:", metrics)
    assert int(metrics["baris train"].replace(",", "")) < ROWS
    scripts = [c.value for c in app.code if "LABEL =" in c.value]
    assert scripts and LABEL in scripts[0] and "def " not in scripts[0]

    # split (urutan buku teks: baca → split → langkah → ekspor); sisa uji di bawah berjalan DENGAN split
    go(app, "Split")
    app.radio(key="split_method").set_value("acak").run()
    check(app, "split acak")
    assert app.checkbox(key="split_stratify").value, "stratified menyala bila label berupa kelas"
    button(app, "Pakai split ini").click().run()
    check(app, "pakai split")
    assert app.session_state["project"]["split"]["method"] == "acak"
    assert any("Split:" in c.value and "acak" in c.value for c in app.sidebar.caption)
    for page in pages:                                 # semua halaman tetap jalan dengan split + langkah
        go(app, page)
    go(app, "Kode & ekspor")
    metrics = {m.label: m.value for m in app.metric}
    print("metrik dengan split:", metrics)
    assert 0.19 * ROWS < int(metrics["baris valid"].replace(",", "")) < 0.21 * ROWS, metrics
    script = next(c.value for c in app.code if "LABEL =" in c.value)
    assert script.index("# %% split data") < script.index("# %% langkah 1")
    assert "train_test_split" in script and "valid_bersih.csv" in script

    # temuan: pratinjau usulan pertama lalu tambah langkah
    go(app, "Temuan")
    before = len(app.session_state["project"]["steps"])
    next(t for t in app.toggle if t.key.startswith("pv_")).set_value(True).run()
    check(app, "pratinjau temuan")
    button(app, "Tambah langkah").click().run()
    check(app, "tambah langkah")
    assert len(app.session_state["project"]["steps"]) == before + 1
    button(app, "Cari kolom kembar").click().run()
    check(app, "kolom kembar")

    # sidebar: matikan, pindah, hapus langkah terakhir
    app.button(key="on_0").click().run()
    assert app.session_state["project"]["steps"][0]["on"] is False
    app.button(key="on_0").click().run()
    app.button(key="down_0").click().run()
    app.button(key=f"del_{before}").click().run()
    check(app, "atur langkah")
    assert len(app.session_state["project"]["steps"]) == before
    assert app.session_state["project"]["steps"][1]["op"] == "tidy_text"

    go(app, "Outlier")
    check(app, "outlier tabel")
    assert not [b for b in app.button if b.key == "out_table_add"], "tabel: semua 'biarkan' → tanpa tombol tambah"
    app.radio(key="out_mode").set_value("satu kolom (rinci)").run()
    app.selectbox(key="outlier_column").select("NJOP per Meter").run()
    check(app, "outlier NJOP")
    assert not [b for b in app.button if str(b.key).startswith("add_out_")], "biarkan → tidak ada tombol tambah"
    app.radio(key="out_choice_NJOP per Meter").set_value("perbaiki dengan langkah").run()
    app.selectbox(key="out_m").select("manual").run()
    app.number_input(key="out_high").set_value(1e9).run()
    check(app, "outlier manual")
    assert [b for b in app.button if b.key == "add_out_NJOP per Meter"], "perbaiki → tombol tambah muncul"
    go(app, "Data")
    go(app, "Outlier")
    assert app.selectbox(key="outlier_column").value == "NJOP per Meter", "pilihan harus diingat antar halaman"
    go(app, "Ejaan")
    check(app, "ejaan semua kolom")
    count = len(app.session_state["project"]["steps"])
    add = [b for b in app.button if str(b.key).startswith("add_spell_all")]
    if add:                                            # aturan kunci + kamus → satu langkah untuk semua kolom
        add[0].click().run()
        check(app, "ejaan tambah sekaligus")
        assert len(app.session_state["project"]["steps"]) > count
        app.button(key=f"del_{len(app.session_state['project']['steps']) - 1}").click().run()
    app.radio(key="spell_all_method").set_value("+ salah ketik rapidfuzz (periksa)").run()
    check(app, "ejaan semua kolom salah ketik")
    app.radio(key="spell_mode").set_value("satu kolom (rinci)").run()
    app.radio(key="spell_method").set_value("+ salah ketik rapidfuzz (periksa)").run()
    check(app, "ejaan satu kolom salah ketik")
    go(app, "Operasi")
    app.radio(key="op_group").set_value("Encoding").run()
    app.selectbox(key="op_name").select("onehot").run()
    check(app, "panel operasi one-hot")
    app.selectbox(key="edit_index").select(3).run()
    check(app, "ubah langkah")
    # proses lama TIDAK boleh mati saat pindah halaman (dulu: pindah halaman = rerun = proses terputus)
    go(app, "Label")
    button(app, "Jalankan cleanlab").click().run()
    check(app, "cleanlab dimulai")
    assert any("berjalan di latar belakang" in i.value for i in app.info), "cleanlab harus jalan di latar"
    go(app, "Data")
    go(app, "Outlier")
    assert any("cleanlab" in c.value and "⏳" in c.value for c in app.sidebar.caption), "masih jalan sesudah pindah"
    go(app, "Label")
    wait_for(app, "cleanlab", lambda: any("Akurasi model" in m.value for m in app.markdown))

    go(app, "Outlier")
    button(app, "Jalankan IsolationForest").click().run()
    go(app, "Data")
    go(app, "Outlier")
    wait_for(app, "IsolationForest", lambda: any("baris ditandai" in m.value for m in app.markdown))

    # simpan otomatis: sesi baru (= server dinyalakan ulang) membaca langkah yang sama
    saved = json.loads(SESSION.read_text(encoding="utf-8"))
    assert saved["steps"] == app.session_state["project"]["steps"]
    assert saved["split"] == app.session_state["project"]["split"]
    fresh = AppTest.from_file("../app.py", default_timeout=300)
    fresh.run()
    check(fresh, "sesi baru")
    assert fresh.session_state["project"]["steps"] == saved["steps"], "langkah harus kembali sesudah restart"
    assert fresh.session_state["project"]["split"] == saved["split"], "split harus kembali sesudah restart"
    print("SEMUA UJI APLIKASI LULUS")



if __name__ == "__main__":
    bantu.main(globals())
