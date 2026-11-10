# =====================================================================================
#  konfig.py: SATU-SATUNYA file yang diubah saat lomba. Semua notebook 1_..7_ membacanya
#  di akhir sel Konfigurasi. Sesudah mengubah file ini: Restart Kernel + Run All.
#  Nilai awal = data demo SmSA (sentimen, 3 kelas) supaya semua notebook bisa dicoba dulu.
# =====================================================================================

# ---- 1. File soal (garis miring /, bukan \) ----
TRAIN_MENTAH = "demo/smsa_train.csv"          # GANTI: mis. "C:/GEMASTIK/soal/train.csv"
TEST_MENTAH = "demo/smsa_test.csv"            # GANTI: mis. "C:/GEMASTIK/soal/test.csv"
SAMPLE_PANITIA = None                         # GANTI: "C:/GEMASTIK/soal/sample_submission.csv"
#                                               None = sample demo dibuat sendiri (format BELUM dicek)

# ---- 2. Kolom ----
ID_COL = "id"                                 # GANTI: kolom id
TEXT_COLS = ["text"]                          # GANTI: kolom teks. 4_finetune hanya memakai yang pertama
TARGET = "label"                              # GANTI: kolom kelas
SUBMIT_COL = TARGET                           # GANTI bila kolom prediksi di sample bernama lain
LABEL_COLS = None                             # MULTILABEL: daftar kolom label 0/1, mis. ["a", "b"].
#                                               None = single label. Bila diisi: pakai 3m, 4m, 5m
GROUP_COL = None                              # kolom penulis/sumber yang berulang (GroupKFold), None = tidak
DATE_COL = None                               # kolom tanggal (tren di EDA dan topik), None = tidak
LANGUAGE = "id"                               # "id" = stopword Indonesia, "en" = Inggris

# ---- 3. Hasil ----
HASIL = "../../outputs/final"                 # GANTI NAMA bila ganti dataset (mis. "../../outputs/lomba"),
#                                               supaya OOF demo tidak tercampur dengan OOF soal
TFIDF_BERAT = False                           # True = 3_baseline memakai kolom <teks>_tfidf (bersih berat).
#                                               Nyalakan hanya bila tabel "Ukur" di 2_praproses naik
SCORING = "f1_macro"                          # metrik sel ukur 2_praproses

# Coba jalur multilabel di data demo CASA (ulasan mobil, 6 label): pakai empat baris ini
# TRAIN_MENTAH = "demo/casa_train.csv"
# TEST_MENTAH = "demo/casa_test.csv"
# LABEL_COLS = ["fuel", "machine", "others", "part", "price", "service"]
# HASIL = "../../outputs/final_multilabel"

# Soal multilabel: 1_eda, 2_praproses, dan "topik x kelas" butuh SATU kolom kelas. Bila TARGET bukan
# salah satu LABEL_COLS, dipakai label pertama (ganti TARGET ke label yang paling penting).
if LABEL_COLS is not None and TARGET not in LABEL_COLS:
    TARGET = LABEL_COLS[0]
    SUBMIT_COL = TARGET

# Metrik lomba BUKAN F1 makro? Ganti juga baris `return` di sel "Metrik" pada
# 3_baseline, 4_finetune, 5_ensemble (multilabel: 3m, 4m, 5m); pilihannya sudah tertulis ber-#.
