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
GROUP_COL = None                              # kolom penulis/sumber yang berulang (GroupKFold), None = tidak
DATE_COL = None                               # kolom tanggal (tren di EDA dan topik), None = tidak
LANGUAGE = "id"                               # "id" = stopword Indonesia, "en" = Inggris

# ---- 3. Hasil ----
HASIL = "../../outputs/final"                 # GANTI NAMA bila ganti dataset (mis. "../../outputs/lomba"),
#                                               supaya OOF demo tidak tercampur dengan OOF soal
TFIDF_BERAT = False                           # True = 3_baseline memakai kolom <teks>_tfidf (bersih berat).
#                                               Nyalakan hanya bila tabel "Ukur" di 2_praproses naik
SCORING = "f1_macro"                          # metrik sel ukur 2_praproses

# Metrik lomba BUKAN F1 makro? Ganti juga baris `return` di sel "Metrik" pada
# 3_baseline, 4_finetune, 5_ensemble (pilihannya sudah tertulis ber-#).
