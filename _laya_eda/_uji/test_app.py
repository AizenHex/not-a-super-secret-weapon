"""Uji otomatis semua tab (Streamlit AppTest). Jalankan dari folder _laya_eda:
    python _uji/test_app.py
Memakai GPU sebentar (Laya 300 + 100 baris, MiniLM 1500 + 3521 teks).
Hanya folder hasil/_uji_* dan file data/_uji_*.csv yang dihapus di awal dan akhir, jadi
unggahan pengguna di data/ aman. Hasil Laya uji diambil dari _uji/hasil_b; hasil/emot_train
(milik pengguna) hanya dibaca, tidak diubah."""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

HERE = os.path.dirname(os.path.abspath(__file__))
APP_FOLDER = os.path.dirname(HERE)
APP = os.path.join(APP_FOLDER, "app.py")
# hanya file dan folder milik uji (awalan _uji_) yang dihapus; unggahan pengguna di data/ aman
TEMP_FOLDERS = ["hasil/_uji_properti", "hasil/_uji_emot"]
TEMP_FILES = ["data/_uji_properti.csv", "data/_uji_emot.csv", "data/_uji_emot_test.csv"]
os.chdir(APP_FOLDER)
sys.path.insert(0, APP_FOLDER)
for folder in TEMP_FOLDERS:
    shutil.rmtree(folder, ignore_errors=True)
for path in TEMP_FILES:
    if os.path.exists(path):
        os.remove(path)
os.makedirs("data", exist_ok=True)
shutil.copy("contoh/properti.csv", "data/_uji_properti.csv")
at = AppTest.from_file(APP, default_timeout=900)


def check(step):
    at.run()
    if at.exception:
        print("ERROR at", step)
        for error in at.exception:
            print(error.message)
            print("\n".join(error.stack_trace[-8:]))
        raise SystemExit(1)
    print("ok:", step)


def expect(condition, message):
    if not condition:
        print("GAGAL:", message)
        raise SystemExit(1)
    print("   benar:", message)


def by_label(widgets, text):
    for widget in widgets:
        if widget.label.startswith(text):
            return widget
    raise KeyError(text)


def by_key(kind, key):
    # kunci yang bergantung pada CSV ditulis "<kunci>__<folder>"
    for widget in at.get(kind):
        if widget.key == key or str(widget.key).startswith(key + "__"):
            return widget
    raise KeyError(key)


def show_markdown(word):
    for item in at.markdown:
        if word in item.value:
            print("   ", item.value[:160].encode("ascii", "replace").decode())


def has_text(word):
    for item in at.markdown:
        if word in item.value:
            return True
    return False


# ---------- skenario 0: data pengguna (emot_train), file Laya LAMA tanpa pilihan kedua ----------
check("awal (emot_train, file Laya lama)")
expect(has_text("Cara kerja") or len(at.expander) > 10, "kotak 'Cara kerja & kenapa' tampil")
by_label(at.sidebar.selectbox, "Kolom label").set_value("label")
check("emot_train + label")
by_key("selectbox", "relation_col").set_value("teks_panjang")
check("hubungan teks_panjang")

# ---------- skenario A: properti, mode baris ----------
by_label(at.sidebar.text_input, "File CSV").set_value("data/_uji_properti.csv")
check("ganti csv")
by_label(at.sidebar.radio, "Mode").set_value("baris")
check("mode baris")
by_label(at.sidebar.multiselect, "Kolom yang dibaca").set_value(
    ["judul", "deskripsi", "harga", "alamat", "luas_m2"])
check("kolom")
by_label(at.sidebar.selectbox, "Kolom label").set_value("jenis_asli")
check("label")
by_label(at.multiselect, "Kolom angka").set_value(["harga", "luas_m2"])
check("cek angka")
by_label(at.checkbox, "Buat kolom rasio").set_value(True)
check("kolom rasio harga per luas_m2")
by_key("selectbox", "profile_column").set_value("harga")
check("profil kolom harga")
by_key("selectbox", "relation_col").set_value("harga")
check("hubungan harga vs label")
by_key("selectbox", "consistency_column").set_value("jenis_asli")
check("konsistensi jenis_asli")
by_key("selectbox", "ngram_class").set_value("kost")
check("ngram kelas kost")
check("tab Outlier tanpa vektor & tanpa lensa (hanya label)")
for preset in ["jenis_properti", "kelengkapan_alamat", "spam"]:
    by_label(at.selectbox, "Lensa siap pakai").set_value(preset)
    by_label(at.button, "Tambah lensa ini").click()
    check("tambah " + preset)
by_label(at.number_input, "Jumlah baris").set_value(300)
check("n baris")
by_label(at.button, "Jalankan Laya").click()
check("jalankan laya")
for name in ["jenis_properti", "kelengkapan_alamat", "spam"]:
    result = pd.read_csv(f"hasil/_uji_properti/laya_{name}.csv")
    expect({"second", "second_value", "first_value"} <= set(result.columns),
           f"laya_{name}.csv punya kolom pilihan kedua ({len(result)} baris)")
    expect(result["second"].notna().all(), f"pilihan kedua terisi untuk lensa {name}")
by_label(at.number_input, "Maksimal baris").set_value(1500)
by_label(at.slider, "Jumlah kelompok").set_value(8)
by_label(at.button, "Kelompokkan").click()
check("kelompokkan")
by_label(at.button, "Jadikan lensa").click()
check("jadikan lensa")
print("   lensa:", list(pd.read_json("hasil/_uji_properti/lenses.json", typ="series").index))
by_key("selectbox", "summary_dim").set_value("jenis_properti")
check("ringkasan per kategori")
by_label(at.radio, "Isi sel").set_value("% per baris")
check("silang % per baris")
by_key("selectbox", "reason_dim").set_value("jenis_asli")
check("pembanding alasan = jenis_asli")
by_label(at.selectbox, "Kolom angka").set_value("harga_per_luas_m2")
check("outlier angka")
by_label(at.selectbox, "Histogram").set_value("apartemen")
check("histogram")
by_label(at.radio, "Tampilkan").set_value("nyasar")
check("teks nyasar")
show_markdown("pasangan")
by_label(at.slider, "Batas kemiripan").set_value(0.85)
check("kemiripan 0.85")
show_markdown("pasangan")
by_label(at.number_input, "Nomor baris (kolom").set_value(2144)
check("kartu baris 2144")
show_markdown("Alasan baris")
by_label(at.checkbox, "Hanya baris yang membuat Laya ragu").set_value(True)
check("telusuri ragu")
by_label(at.checkbox, "Hanya baris yang punya alasan").set_value(True)
check("telusuri beralasan")
by_label(at.button, "Simpan catatan").click()
check("simpan catatan")
by_key("slider", "tsne_points").set_value(500)
by_key("button", "tsne_button").click()
check("peta 2D (t-SNE 500 titik)")
by_key("selectbox", "tsne_color").set_value("jenis_properti")
check("peta 2D warna lensa")
expect(has_text("Silhouette rata-rata"), "angka silhouette tampil")
by_label(at.button, "Tulis file").click()
check("ekspor")
export = pd.read_csv("hasil/_uji_properti/data_berlensa.csv")
export["alasan"] = export["alasan"].fillna("")
print("   ekspor:", export.shape)
expect("alasan" in export.columns, "kolom alasan ada di file ekspor")
mustahil = export[export["masalah"] == "harga_mustahil"]
caught = mustahil["alasan"].str.contains("kali median")
expect(caught.mean() > 0.9,
       f"alasan angka untuk {caught.sum()} dari {len(mustahil)} harga mustahil")
normal = export[export["masalah"].isna()]
print("   contoh alasan harga mustahil:")
print("    ", mustahil["alasan"].iloc[0][:400].encode("ascii", "replace").decode())
unreadable = export[export["masalah"] == "harga_teks"]
expect(unreadable["alasan"].str.contains("tidak terbaca").mean() > 0.9, "alasan harga teks")
negative = export[export["masalah"] == "luas_negatif"]
expect(negative["alasan"].str.contains("negatif").mean() > 0.9, "alasan luas negatif")
doubt = export[export["alasan"].str.contains("Laya")]
expect(len(doubt) > 0, f"alasan Laya ragu ada ({len(doubt)} baris)")
by_key("button", "temuan_tulis").click()
check("tulis laporan temuan")
with open("hasil/_uji_properti/laporan_eda.md", encoding="utf-8") as file:
    report = file.read()
expect("## Temuan utama" in report and "**Alasan:**" in report, "laporan berisi temuan + alasan")
# catatan uji: buang baris 5 dan 6 (ditulis langsung, seperti hasil tab Telusuri)
notes = pd.DataFrame({"keputusan": ["buang", "buang"], "label_baru": ["", ""],
                      "catatan": ["uji buang", "uji buang"]}, index=[5, 6])
notes.to_csv("hasil/_uji_properti/catatan.csv", index_label="row")
by_key("multiselect", "ekspor_fitur").set_value(["gaya", "log_angka"])
check("ekspor: pilih fitur gaya + log angka")
by_key("button", "ekspor_train_tulis").click()
check("ekspor: tulis train_siap")
ready = pd.read_csv("hasil/_uji_properti/train_siap.csv")
expect(len(ready) == 2998 and 5 not in ready["row"].tolist(), "train_siap: 2 baris dibuang")
expect("panjang" in ready.columns and "log_harga" in ready.columns, "train_siap: kolom fitur ada")
by_key("multiselect", "buang_kolom").set_value(["deskripsi"])
by_key("text_area", "buang_aturan").set_value('jenis_asli == "kost"')
by_key("button", "ekspor_train_tulis").click()
check("ekspor: buang kustom")
ready = pd.read_csv("hasil/_uji_properti/train_siap.csv")
expect("deskripsi" not in ready.columns and (ready["jenis_asli"] != "kost").all(),
       f"buang kustom: kolom deskripsi + baris kost dibuang ({len(ready)} baris)")
by_key("selectbox", "kode_category").set_value("jenis_asli")
by_key("multiselect", "kode_features").set_value(["gaya", "log_angka", "rasio_median"])
by_key("button", "kode_write").click()
check("kode: tulis eda_otomatis")
with open("hasil/_uji_properti/eda_otomatis.py", encoding="utf-8") as file:
    script = file.read()
expect("MANUAL_DROP" in script and "for column in OUTLIER_COLUMNS:" in script,
       "kode: buang manual + loop outlier")
expect('CATEGORY = "jenis_asli"' in script and "DROP_ROWS" not in script,
       "kode: kategori jenis_asli, tanpa daftar DROP_ROWS")
expect("import laya" not in script and "torch" not in script, "kode: tanpa Laya, hanya pandas")
expect(os.path.exists("hasil/_uji_properti/eda_otomatis.ipynb"), "notebook .ipynb tertulis")
print("   contoh alasan Laya:", doubt["alasan"].iloc[0][:300].encode("ascii", "replace").decode())

# ---------- skenario B: teks emosi, label diacak sebagian, kolom tambahan ----------
emot = pd.read_csv("../../data_latihan/text/classification/emot_train.csv")
emot["label_asli"] = emot["label"]
flip = emot.sample(frac=0.1, random_state=3).index
labels = sorted(emot["label"].unique())
for row in flip:
    others = []
    for label in labels:
        if label != emot.at[row, "label"]:
            others.append(label)
    emot.at[row, "label"] = others[row % len(others)]
label_codes = {}
for number, name in enumerate(labels):
    label_codes[name] = number
codes = emot["label_asli"].map(label_codes)
months = (codes * 2 + emot.index % 3) % 12 + 1
emot["tanggal"] = "2024-" + months.astype(str).str.zfill(2) + "-15"
variants = ["Twitter", "twitter ", "TWITTER", "Twiter", "Instagram", "instagram", "Tiktok"]
sources = []
for number in emot.index:
    sources.append(variants[number % len(variants)])
emot["sumber"] = sources
emot["umur"] = 20 + emot.index % 40
emot.loc[(emot["label_asli"] == "anger") & (emot.index % 3 == 0), "umur"] = np.nan
os.makedirs("data", exist_ok=True)
emot.to_csv("data/_uji_emot.csv", index=False)
os.makedirs("hasil/_uji_emot", exist_ok=True)
# hasil Laya milik uji sendiri (_uji/hasil_b, 300 baris), bukan folder hasil pengguna.
# 'emosi_lama' = 100 baris format lama (tanpa pilihan kedua) untuk uji "lanjutkan file lama".
FIXTURE = "_uji/hasil_b/emot_train"
for name in ["emosi", "kejelasan", "label_cocok"]:
    shutil.copy(f"{FIXTURE}/laya_{name}.csv", f"hasil/_uji_emot/laya_{name}.csv")
old_file = pd.read_csv(f"{FIXTURE}/laya_emosi.csv").head(100)
old_file = old_file[["row", "answer", "value", "confidence"]]
old_file.to_csv("hasil/_uji_emot/laya_emosi_lama.csv", index=False)
with open(f"{FIXTURE}/lenses.json", encoding="utf-8") as file:
    test_lenses = json.load(file)
test_lenses["emosi_lama"] = test_lenses["emosi"]
with open("hasil/_uji_emot/lenses.json", "w", encoding="utf-8") as file:
    json.dump(test_lenses, file, ensure_ascii=False, indent=2)
by_label(at.sidebar.text_input, "File CSV").set_value("data/_uji_emot.csv")
check("csv emot")
by_label(at.sidebar.radio, "Mode").set_value("teks")
check("mode teks")
by_label(at.sidebar.selectbox, "Kolom label").set_value("label")
check("emot teks + label")
by_label(at.sidebar.selectbox, "Kolom tanggal").set_value("tanggal")
check("kolom tanggal")
expect(has_text("Separuh awal vs akhir"), "uji pergeseran label menurut waktu tampil")
show_markdown("Separuh awal")
by_key("selectbox", "consistency_column").set_value("sumber")
check("konsistensi sumber")
expect(has_text("Nilai yang bisa dirapikan"), "ringkasan nilai tidak konsisten tampil")
show_markdown("Nilai yang bisa dirapikan")
by_label(at.sidebar.selectbox, "Kolom teks").set_value("text")
check("kolom teks")
by_key("selectbox", "relation_col").set_value("umur")
check("hubungan umur vs label")
by_label(at.number_input, "Jumlah baris").set_value(200)
by_label(at.button, "Jalankan Laya").click()
check("lanjutkan file Laya lama (emosi_lama 100 -> 200)")
new_spam = pd.read_csv("hasil/_uji_emot/laya_emosi_lama.csv")
expect("second" in new_spam.columns and len(new_spam) >= 190, "file lama diperbarui "
       f"({len(new_spam)} baris, kolom second ada)")
expect(new_spam["second"].isna().sum() == 100,
       "100 baris lama tanpa pilihan kedua, baris baru terisi")
by_label(at.number_input, "Maksimal baris").set_value(len(emot))
by_label(at.button, "Kelompokkan").click()
check("kelompokkan emot")
by_key("selectbox", "reason_dim").set_value("label")
check("pembanding alasan = label")
by_label(at.radio, "Tampilkan").set_value("nyasar")
check("outlier teks per label")
groups = pd.read_csv("hasil/_uji_emot/kelompok.csv")
vectors = np.load("hasil/_uji_emot/vektor.npy")
import outliers
distances = outliers.text_outliers(vectors, groups["row"].to_numpy(), emot["label"])
is_flipped = distances.index.isin(flip)
nyasar_flipped = int((distances["nyasar"] & is_flipped).sum())
print("   nyasar:", int(distances["nyasar"].sum()), "| di antaranya label diacak:", nyasar_flipped,
      "dari", len(flip), "label diacak")
print("   persen label diacak di antara nyasar:",
      round(100 * nyasar_flipped / distances["nyasar"].sum(), 1), "% (dasar 10%)")
first_stray = int(distances.index[distances["nyasar"]][0])
by_label(at.number_input, "Nomor baris (kolom").set_value(first_stray)
check("kartu baris nyasar")
show_markdown("Alasan baris")
by_key("slider", "tsne_points").set_value(1000)
by_key("button", "tsne_button").click()
check("peta 2D emot (1000 titik)")
by_key("selectbox", "tsne_color").set_value("label")
check("peta 2D warna label")
by_key("button", "salah_tombol_train").click()
check("salah tebak train")
expect(has_text("Pasangan label"), "tabel pasangan label yang tertukar tampil")
# test berlabel: 300 baris emot dengan label asli, supaya baris salah bisa dihitung
emot.head(300).to_csv("data/_uji_emot_test.csv", index=False)
by_key("text_input", "salah_test_path").set_value("data/_uji_emot_test.csv")
by_key("button", "salah_tombol_test").click()
check("salah tebak test berlabel")
found = False
for item in at.success:
    if "Di test, model baseline salah menebak" in item.value:
        print("   ", item.value[:200].encode("ascii", "replace").decode())
        found = True
expect(found, "ringkasan salah tebak test tampil")
by_label(at.button, "Tulis file").click()
check("ekspor emot")
export = pd.read_csv("hasil/_uji_emot/data_berlensa.csv")
export["alasan"] = export["alasan"].fillna("")
expect((export["alasan"] != "").sum() > 100, f"{(export['alasan'] != '').sum()} baris beralasan")
print("   contoh alasan (nyasar):", export["alasan"][export["alasan"].str.contains("lebih dekat")]
      .iloc[0][:300].encode("ascii", "replace").decode())
conflict = export["alasan"].str.contains("KONFLIK")
expect(conflict.sum() >= 2, f"alasan konflik label ({conflict.sum()} baris)")
by_key("radio", "split_method").set_value("kelompok")
by_key("button", "split_tulis").click()
check("split kelompok teks sama")
train_split = pd.read_csv("hasil/_uji_emot/train_split.csv")
valid_split = pd.read_csv("hasil/_uji_emot/valid_split.csv")
same = set(train_split["text"].str.lower()) & set(valid_split["text"].str.lower())
expect(len(same) == 0, "split kelompok: tidak ada teks sama di train dan validasi")
by_key("radio", "split_method").set_value("waktu")
by_key("button", "split_tulis").click()
check("split waktu")
train_split = pd.read_csv("hasil/_uji_emot/train_split.csv")
valid_split = pd.read_csv("hasil/_uji_emot/valid_split.csv")
expect(train_split["tanggal"].max() <= valid_split["tanggal"].min(), "split waktu: validasi terbaru")
for folder in TEMP_FOLDERS:
    shutil.rmtree(folder, ignore_errors=True)
for path in TEMP_FILES:
    os.remove(path)
print("SEMUA LOLOS")
