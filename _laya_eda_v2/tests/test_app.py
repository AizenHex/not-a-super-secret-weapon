"""Uji aplikasi tanpa browser (streamlit AppTest): semua tab tampil tanpa galat, tanpa langkah dan dengan langkah.

Jalankan dari folder _laya_eda_v2:  python -m tests.test_app
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

# file simpan otomatis milik uji, BUKAN hasil/sesi_terakhir.json milik pengguna
SESSION = Path(tempfile.mkdtemp()) / "sesi_uji.json"
os.environ["LAYA_SESI"] = str(SESSION)

from tests.test_steps import LABEL, project


def check(app: AppTest, title: str) -> None:
    errors = [e.value for e in app.exception]
    shown = [e.value for e in app.error]
    print(f"{title}: {len(app.tabs)} tab, exception {len(errors)}, st.error {len(shown)}")
    assert not errors, errors[0]
    assert not shown, shown


def button(app: AppTest, start: str):
    return next(b for b in app.button if b.label.startswith(start))


def go(app: AppTest, page: str) -> None:
    app.radio(key="page").set_value(page).run()
    check(app, f"halaman {page}")


def main() -> None:
    app = AppTest.from_file("../app.py", default_timeout=300)
    app.run()
    check(app, "tanpa langkah")
    pages = app.radio(key="page").options
    assert len(pages) == 11, pages
    for page in pages:                                 # setiap halaman tampil tanpa galat, tanpa langkah
        go(app, page)

    app.session_state["project"] = copy.deepcopy(project())
    for page in pages:                                 # dan dengan langkah
        go(app, page)
    go(app, "Kode & ekspor")
    metrics = {m.label: m.value for m in app.metric}
    print("metrik:", metrics)
    assert int(metrics["baris train"].replace(",", "")) < 30380
    scripts = [c.value for c in app.code if "LABEL =" in c.value]
    assert scripts and LABEL in scripts[0] and "def " not in scripts[0]

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
    add = [b for b in app.button if str(b.key).startswith("add_many_spell_all")]
    if add:                                            # fingerprint dicentang otomatis → tambah sekaligus
        add[0].click().run()
        check(app, "ejaan tambah sekaligus")
        assert len(app.session_state["project"]["steps"]) > count
        app.button(key=f"del_{len(app.session_state['project']['steps']) - 1}").click().run()
    app.radio(key="spell_all_method").set_value("nearest neighbor: rapidfuzz (periksa)").run()
    check(app, "ejaan semua kolom nearest neighbor")
    app.radio(key="spell_mode").set_value("satu kolom (rinci)").run()
    app.radio(key="spell_method").set_value("nearest neighbor: rapidfuzz (periksa)").run()
    check(app, "nearest neighbor")
    go(app, "Operasi")
    app.radio(key="op_group").set_value("Encoding").run()
    app.selectbox(key="op_name").select("onehot").run()
    check(app, "panel operasi one-hot")
    app.selectbox(key="edit_index").select(3).run()
    check(app, "ubah langkah")
    go(app, "Label")
    button(app, "Jalankan cleanlab").click().run()
    check(app, "cleanlab")
    assert any("Akurasi model" in m.value for m in app.markdown)

    # simpan otomatis: sesi baru (= server dinyalakan ulang) membaca langkah yang sama
    saved = json.loads(SESSION.read_text(encoding="utf-8"))
    assert saved["steps"] == app.session_state["project"]["steps"]
    fresh = AppTest.from_file("../app.py", default_timeout=300)
    fresh.run()
    check(fresh, "sesi baru")
    assert fresh.session_state["project"]["steps"] == saved["steps"], "langkah harus kembali sesudah restart"
    print("SEMUA UJI APLIKASI LULUS")



if __name__ == "__main__":
    main()
