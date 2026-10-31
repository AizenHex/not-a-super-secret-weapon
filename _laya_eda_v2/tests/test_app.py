"""Uji aplikasi tanpa browser (streamlit AppTest): semua tab tampil tanpa galat, resep kosong dan penuh.

Jalankan dari folder _laya_eda_v2:  python -m tests.test_app
"""
from __future__ import annotations

import copy

import pandas as pd
from streamlit.testing.v1 import AppTest

from tests.test_codegen import DATA, full_recipe


def check(app: AppTest, title: str) -> None:
    errors = [e.value for e in app.exception]
    shown = [e.value for e in app.error]
    print(f"{title}: {len(app.tabs)} tab, exception {len(errors)}, st.error {len(shown)}")
    assert not errors, errors[0]
    assert not shown, shown


def main() -> None:
    app = AppTest.from_file("../app.py", default_timeout=300)
    app.run()
    check(app, "resep kosong")
    assert len(app.tabs) == 15

    recipe = full_recipe()
    recipe["number_columns"] = ["Total Hutang"]
    recipe["encoding"]["Pendapatan Keluarga"]["order"] = sorted(
        pd.read_csv(DATA)["Pendapatan Keluarga"].dropna().unique())
    app.session_state["recipe"] = copy.deepcopy(recipe)
    app.run()
    check(app, "resep penuh")
    metrics = {m.label: m.value for m in app.metric}
    print("metrik:", metrics)
    assert metrics["baris train"] == "29,948"

    # klik semua tombol 'Cari kolom kembar' + pilih isian bebas lain
    next(b for b in app.button if b.label == "Cari kolom kembar").click().run()
    check(app, "kolom kembar")
    app.selectbox(key="outlier_column").select("NJOP per Meter").run()
    check(app, "outlier NJOP")
    app.selectbox(key="encoding_column").select("Jalur Masuk").run()
    next(s for s in app.selectbox if s.label == "Encoder").select("onehot").run()
    next(b for b in app.button if b.label == "Pratinjau kolom baru").click().run()
    check(app, "pratinjau encoding")
    print("pratinjau:", [c.value for c in app.code][:1])
    print("SEMUA UJI APLIKASI LULUS")


if __name__ == "__main__":
    main()
