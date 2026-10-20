"""Uji otomatis tab_fitur.py dan tab_test.py lewat harness_b.py (Streamlit AppTest).
Jalankan dari mana saja:  python _uji/test_b.py
Hasil perantara (vektor, jawaban Laya CPU) disimpan di _uji/hasil_b/ dan dipakai lagi."""
import os
import sys
import time

import pandas as pd
from streamlit.testing.v1 import AppTest

HERE = os.path.dirname(os.path.abspath(__file__))
APP_FOLDER = os.path.dirname(HERE)
os.chdir(APP_FOLDER)
sys.path.insert(0, APP_FOLDER)
at = AppTest.from_file(f"{HERE}/harness_b.py", default_timeout=3000)
started = time.time()


def check(step):
    begin = time.time()
    at.run()
    if at.exception:
        print("ERROR at", step)
        for error in at.exception:
            print(error.message)
            print("\n".join(error.stack_trace[-8:]))
        raise SystemExit(1)
    print(f"ok ({time.time() - begin:5.1f} dtk): {step}")


def by_prefix(widgets, prefix):
    for widget in widgets:
        if widget.key is not None and widget.key.startswith(prefix):
            return widget
    raise KeyError(prefix)


def click(key):
    at.button(key=key).click()
    check("klik " + key)


def has_button(key):
    for button in at.button:
        if button.key == key:
            return True
    return False


def show_results(words):
    # cetak kalimat hasil (kotak hijau / peringatan) yang memuat kata tertentu
    for group in [at.success, at.error]:
        for item in group:
            for word in words:
                if word in item.value:
                    print("   >", item.value[:400].replace("\n", " "))
                    break


def run_fitur_tab():
    by_prefix(at.slider, "fitur_maks_baris").set_value(3500)
    check("slider baris")
    click("fitur_tombol_lab")
    show_results(["Kami menguji"])
    click("fitur_tombol_ablation")
    show_results(["Laya"])
    click("fitur_tombol_kata")
    click("fitur_tombol_padat")
    show_results(["permutation", "mutual"])
    if has_button("fitur_tombol_issues"):
        click("fitur_tombol_issues")
        show_results(["ditandai"])
        click("fitur_tombol_noise")
        show_results(["sengaja diacak"])
    if has_button("fitur_tombol_imbalance"):
        click("fitur_tombol_imbalance")
        show_results(["F1 macro"])
    click("fitur_tombol_bocor")
    show_results(["fitur", "Tidak ada"])


def run_test_tab():
    click("test_tombol_adversarial")
    show_results(["AUC"])
    by_prefix(at.number_input, "test_laya_baris").set_value(10)
    click("test_tombol_laya")
    show_results(["Laya"])
    print("   temuan terkumpul di harness:", len(at.dataframe[-1].value))


def make_property_test():
    properties = pd.read_csv("contoh/properti.csv")
    sample = properties.sample(n=600, random_state=7).reset_index(drop=True)
    sample["luas_m2"] = (sample["luas_m2"] * 1.5).astype(int)
    sample = sample.drop(columns=["jenis_asli", "masalah"])
    os.makedirs("_uji/hasil_b", exist_ok=True)
    sample.to_csv("_uji/hasil_b/properti_test.csv", index=False)


# ---------- skenario 1: emot (teks + label) ----------
make_property_test()
check("awal: emot (Laya CPU dan MiniLM dibuat bila belum ada)")
print("   subheader pertama:", at.subheader[0].value)
run_fitur_tab()
run_test_tab()

# ---------- skenario 2: properti (mode baris) ----------
at.sidebar.radio(key="harness_skenario").set_value("properti (baris + label)")
check("ganti skenario: properti")
by_prefix(at.selectbox, "fitur_dimensi").set_value("jenis_properti")
check("dimensi = jenis_properti")
by_prefix(at.text_input, "fitur_kata_kunci").set_value("/bulan,shm")
check("kata kunci")
run_fitur_tab()
at.text_input(key="test_path").set_value("_uji/hasil_b/properti_test.csv")
check("test = properti_test.csv")
run_test_tab()

# ---------- skenario 3: label angka (regresi) ----------
at.sidebar.radio(key="harness_skenario").set_value("properti regresi (label luas_m2)")
check("ganti skenario: regresi")
run_fitur_tab()
at.text_input(key="test_path").set_value("_uji/hasil_b/properti_test.csv")
check("test untuk regresi")
run_test_tab()
print(f"SEMUA LOLOS dalam {time.time() - started:.0f} detik")
