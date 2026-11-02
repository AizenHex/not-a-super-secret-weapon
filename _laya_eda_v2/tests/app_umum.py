"""AppTest satu dataset umum (dipanggil tests/test_umum.py dalam proses terpisah agar cache/sesi tidak terbawa).

python -m tests.app_umum titanic
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tests import umum

NAME = sys.argv[1]
PATH = umum.path(NAME)
ROOT = Path(__file__).resolve().parents[1]
os.environ["LAYA_TRAIN"] = str(PATH)
os.environ["LAYA_SESI"] = str(Path(tempfile.mkdtemp()) / "sesi.json")
os.environ["LAYA_OUTPUT"] = tempfile.mkdtemp()

from streamlit.testing.v1 import AppTest  # noqa: E402


def check(app: AppTest, where: str) -> None:
    problems = [e.value for e in app.exception] + [str(e.value) for e in app.error]
    if problems:
        print(f"GAGAL {where}: {problems[0][:500]}")
        sys.exit(1)


def main() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=900)
    app.run()
    label = umum.DATASETS[NAME][1] or app.session_state["train"].columns[-1]
    app.session_state["project"]["label"] = label
    app.run()
    check(app, "awal")
    pages = app.radio(key="page").options
    for page in pages:
        app.radio(key="page").set_value(page).run()
        check(app, page)
    print(f"OK {len(pages)} halaman tanpa galat")

    # G2: ubah parameter usulan sebelum ditambah
    app.radio(key="page").set_value("Temuan").run()
    edited = None
    for toggle in [t for t in app.toggle if t.key.startswith("pv_")]:
        toggle.set_value(True).run()
        check(app, f"pratinjau {toggle.key}")
        own = toggle.key.removeprefix("pv_") + "_"          # widget milik usulan yang sedang dibuka saja
        radios = [r for r in app.radio if str(r.key).startswith(own) and str(r.key).endswith("_form_st")]
        texts = [t for t in app.text_input if str(t.key).startswith(own) and str(t.key).endswith("_form_reason")]
        if radios:
            radios[0].set_value("mean").run()
            edited = ("strategy", "mean")
        elif texts:
            texts[0].set_value("alasan diubah uji").run()
            edited = ("reason", "alasan diubah uji")
        else:
            app.toggle(key=toggle.key).set_value(False).run()
            continue
        before = len(app.session_state["project"]["steps"])
        mine = [b for b in app.button if str(b.key).startswith("add_" + own)]
        mine[0].click().run()
        check(app, "tambah usulan yang diubah")
        steps = app.session_state["project"]["steps"]
        assert len(steps) == before + 1 and steps[-1]["params"].get(edited[0]) == edited[1], steps[-1]
        print(f"OK usulan diubah ({edited[0]} = {edited[1]!r}) lalu ditambah: {steps[-1]['op']}")
        break
    for _ in range(30):                                    # tambah usulan lain (pilihan pertama) satu per satu
        toggles = [t for t in app.toggle if t.key.startswith("pv_") and not t.value]
        if not toggles:
            break
        toggles[0].set_value(True).run()
        check(app, f"pratinjau {toggles[0].key}")
        buttons = [b for b in app.button if b.label.startswith("Tambah langkah") and not b.disabled]
        if buttons:
            buttons[0].click().run()
            check(app, "tambah usulan")
    steps = app.session_state["project"]["steps"]
    print(f"info {len(steps)} langkah: {[s['op'] for s in steps]}")
    for page in pages:
        app.radio(key="page").set_value(page).run()
        check(app, page + " (dengan langkah)")
    print("OK semua halaman dengan langkah")

    # G3: ✎ di sidebar → halaman Operasi pada langkah itu → ubah jadi kode bebas
    if steps:
        app.button(key="edit_0").click().run()
        check(app, "buka langkah ✎")
        assert app.radio(key="page").value == "Operasi" and app.session_state["edit_index"] == 0
        old = app.session_state["project"]["steps"][0]["op"]
        if old != "custom_code":
            app.button(key="to_code_0").click().run()
            check(app, "ubah jadi kode bebas")
            assert app.session_state["project"]["steps"][0]["op"] == "custom_code"
        print(f"OK langkah 1 ({old}) dibuka lewat ✎ dan diubah jadi kode bebas")

    app.radio(key="page").set_value("Latih").run()
    app.multiselect(key="train_models").set_value(["histgb"]).run()
    app.radio(key="train_mode").set_value("cepat").run()
    next(b for b in app.button if b.label.startswith("Latih + papan")).click().run()
    start = time.time()
    while time.time() - start < 900:
        time.sleep(2)
        app.run()
        boards = [f.value for f in app.dataframe if "status" in f.value]
        if boards:
            break
    check(app, "Latih")
    status = boards[0]["status"].tolist() if boards else ["tidak selesai"]
    assert status == ["selesai"], status
    score = boards[0]["skor CV"].iloc[0]
    print(f"OK Latih: tugas {app.radio(key='train_task').value}, metrik {app.selectbox(key='train_metric').value}, "
          f"skor CV {score:.4f}")

    app.radio(key="page").set_value("Kode & ekspor").run()
    script = next(c.value for c in app.code if "LABEL =" in c.value)
    folder = Path(tempfile.mkdtemp())
    (folder / "pipa.py").write_text(script.replace("TRAIN_CSV = 'train.csv'", f"TRAIN_CSV = {str(PATH)!r}"),
                                    encoding="utf-8")
    run = subprocess.run([sys.executable, "pipa.py"], cwd=folder, capture_output=True, text=True,
                         env={**os.environ, "PYTHONIOENCODING": "cp1252"})
    if run.returncode:
        print(f"GAGAL skrip mandiri: {run.stderr.strip().splitlines()[-1][:400]}")
        sys.exit(1)
    print("OK skrip mandiri jalan")


if __name__ == "__main__":
    main()
