"""Uji aplikasi tanpa browser (streamlit AppTest): semua tab tampil tanpa galat, tanpa langkah dan dengan langkah.

Jalankan dari folder _laya_eda_v2:  python -m tests.test_app
"""
from __future__ import annotations

import copy

from streamlit.testing.v1 import AppTest

from tests.test_steps import LABEL, project


def check(app: AppTest, title: str) -> None:
    errors = [e.value for e in app.exception]
    shown = [e.value for e in app.error]
    print(f"{title}: {len(app.tabs)} tab, exception {len(errors)}, st.error {len(shown)}")
    assert not errors, errors[0]
    assert not shown, shown


def button(app: AppTest, start: str):
    return next(b for b in app.button if b.label.startswith(start))


def main() -> None:
    app = AppTest.from_file("../app.py", default_timeout=300)
    app.run()
    check(app, "tanpa langkah")
    assert len(app.tabs) == 11

    app.session_state["project"] = copy.deepcopy(project())
    app.run()
    check(app, "dengan langkah")
    metrics = {m.label: m.value for m in app.metric}
    print("metrik:", metrics)
    assert int(metrics["baris train"].replace(",", "")) < 30380

    # temuan: pratinjau usulan pertama lalu tambah langkah
    before = len(app.session_state["project"]["steps"])
    next(t for t in app.toggle if t.key.startswith("pv_")).set_value(True).run()
    check(app, "pratinjau temuan")
    button(app, "Tambah langkah").click().run()
    check(app, "tambah langkah")
    assert len(app.session_state["project"]["steps"]) == before + 1

    # sidebar: matikan, pindah, hapus langkah terakhir
    app.button(key="on_0").click().run()
    assert app.session_state["project"]["steps"][0]["on"] is False
    app.button(key="on_0").click().run()
    app.button(key="down_0").click().run()
    app.button(key=f"del_{before}").click().run()
    check(app, "atur langkah")
    assert len(app.session_state["project"]["steps"]) == before
    assert app.session_state["project"]["steps"][1]["op"] == "tidy_text"

    button(app, "Cari kolom kembar").click().run()
    check(app, "kolom kembar")
    app.selectbox(key="outlier_column").select("NJOP per Meter").run()
    check(app, "outlier NJOP")
    app.radio(key="spell_method").set_value("nearest neighbor: rapidfuzz (periksa)").run()
    check(app, "nearest neighbor")
    app.radio(key="op_group").set_value("Encoding").run()
    app.selectbox(key="op_name").select("onehot").run()
    check(app, "panel operasi one-hot")
    app.selectbox(key="edit_index").select(3).run()
    check(app, "ubah langkah")
    button(app, "Jalankan cleanlab").click().run()
    check(app, "cleanlab")
    assert any("Akurasi model" in m.value for m in app.markdown)
    scripts = [c.value for c in app.code if "LABEL =" in c.value]
    assert scripts and LABEL in scripts[0] and "def " not in scripts[0]
    print("SEMUA UJI APLIKASI LULUS")


if __name__ == "__main__":
    main()
