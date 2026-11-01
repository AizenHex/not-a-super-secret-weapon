# Keputusan

Terbaru di atas. Format: `- YYYY-MM-DD: <keputusan>. Alasan: <...>.`

- 2026-11-02: uji Laya v2 tanpa pytest: runner sendiri `tests/bantu.py` (lulus/gagal/dilewati, `Lewati` bila data
  riil/library tidak ada) + `tests/semua.py`. Data uji = `tests/sintetis.py` (sintetis, seed tetap) bila data riil
  tidak ada. Alasan: pytest tidak ada di requirements.txt dan prompt melarang menambah library.
- 2026-11-02: R15 alasan buang = parameter `reason` opsional; bila kosong kode menulis alasan bawaan dari
  parameter langkah (mis. "baris train dengan lebih dari 5 kolom kosong"). Alasan: langkah lama/usulan Temuan tetap
  sah tanpa memaksa pengguna mengetik, tetapi setiap langkah buang tetap punya komentar `# alasan:`.
- 2026-11-02: R6 dilampaui → aplikasi menolak menambah langkah dan menawarkan op baru `fuzzy_spelling` (kosakata
  dari train + rapidfuzz `token_sort_ratio` + `default_process`, nilai train tidak diubah), sesuai VISI bagian 3/4.
  Alasan: lapisan 3 R6 butuh op nyata, bukan hanya pesan.
- 2026-11-02: R12 tahan test beda bentuk lewat blok "selaraskan test" (kolom hilang → kosong, urutan & tipe =
  train, '1.500' → 1500) yang sama di skrip dan aplikasi (`steps.align`). Kolom tambahan tetap ada di belakang.
  Alasan: langkah lain tidak perlu cabang per kolom; satu sumber kode (R5).

- 2026-11-02: E1 Pipeline = satu `ColumnTransformer` per langkah (remainder passthrough) + tahap `teks_sisa`
  (OrdinalEncoder, nilai baru/kosong = -1) + model; yang masuk pipeline hanya AKHIRAN langkah aktif yang semuanya
  punya transformer (`ops.transformer`), sisanya blok EDA pandas sebelum pipeline. Op pandas lama yang dipetakan
  (impute, fill_constant, scale, log_transform, ordinal, outlier manual clip) dibuktikan sama angkanya di uji E1;
  `scale` diubah ke std ddof=0 agar = StandardScaler. Pipeline dilatih di baris berlabel saja (statistik pipeline
  dari baris berlabel; blok EDA dari seluruh train). Alasan: urutan langkah tidak pernah diubah, kode tetap lurus,
  dan kesetaraan bisa diuji persis (prediksi identik 100% di data sintetis).
- 2026-11-02: op `rename_column` dihapus dari katalog. Alasan: R8 (nama kolom asli tidak pernah diganti) adalah
  aturan yang tidak boleh dilanggar menurut prompt pengguna; langkah lama di sesi tersimpan tampil sebagai
  "operasi tidak dikenal" dan berhenti dengan galat (R11), tidak diam-diam diabaikan.
- 2026-11-02: venv uji cloud memakai `torch==2.11.0` dari PyPI karena index PyTorch (cu128/cpu) diblokir proxy
  (403). Versi sama dengan requirements.txt, varian build berbeda; hanya dipakai AutoGluon di uji E5.

- 2026-11-02: VISI.md R6 dihitung per entri peta (kunci dict + isi list + alternatif regex), bukan per baris; contoh rapidfuzz memakai `token_sort_ratio` + `utils.default_process` dan tidak mengubah nilai yang sudah ada di train. Alasan: temuan review 7 dan 4 (baris bisa dipadatkan, WRatio menyamakan potongan `Bandung` ~ `Kota Bandung`).

- 2026-11-02: fitur extra Laya v2 (VISI.md bagian 0): langkah pandas (ejaan, isian bebas, penanda kosong)
  tetap blok EDA lurus sebelum Pipeline; hanya langkah yang punya transformer sklearn/feature_engine masuk
  `Pipeline`. `FunctionTransformer` + `def` hanya opsi. pandera default mode lapor, mode ketat opsi.
  Alasan: R4 tetap terjaga tanpa membatasi joblib; `def` di file joblib wajib didefinisikan ulang
  sebelum `load`; mode lapor tidak mematikan prediksi saat lomba.

- 2026-11-02: rencana Laya v2 lama di `RENCANA.md` (P1–P7, flowchart per tab, checklist, keputusan terbuka)
  dinilai lalu dilebur ke `VISI.md`. Masuk: R8 diperketat (nama kolom asli tidak pernah diganti, tidak
  hilang diam-diam), R12 + daftar uji bentuk test, R15 sebelum→sesudah + alasan, R16 jalan tanpa
  Laya/MiniLM (offline), R17 rerun < 2 s, R18 arti kolom tabel, baris penanda kosong per kolom dan logika
  antar kolom, pemisah hanya di isian bebas, one-hot per bahan, ambang peta kata default, ekspor
  `laporan_temuan.md`, uji terima data riil. Tidak masuk: (sklearn `Pipeline` + joblib, pandera, split k-fold:
  usang 2026-11-02, pengguna menjadikannya fitur extra E1–E5 di atas R1–R18),
  persetujuan flowchart sebelum kode (cara kerja, bukan fitur), usulan Laya ubin~tegel sebagai syarat
  selesai (Laya opsional). Alasan: pengguna minta VISI.md jadi satu-satunya sumber.

- 2026-11-02: `evidently` dan `ydata-profiling` dihapus dari rencana Laya v2 (VISI.md) dan tidak
  masuk `requirements.txt`. Alasan: dry-run menunjukkan evidently 0.7.23 menurunkan plotly 7.1.0 ->
  5.24.1 dan ydata-profiling 4.18.4 menurunkan numpy 2.5.3 -> 2.3.5, scipy 1.18.1 -> 1.16.3,
  matplotlib 3.11.2 -> 3.10.0; fungsinya sudah tertutup `skrub.TableReport` (profil) dan
  `ks_2samp` + adversarial validation (drift). Enam library EDA lain dikunci ke versi terbaru PyPI.

- 2026-10-31: encoding kedekatan makna = vektor MiniLM nilai kategori → PCA (numpy SVD) dari nilai
  train, `MEANING_SIZE` = 8 angka per kolom. Alasan: isian bebas yang berantakan ("genting",
  "Genteng") dapat angka mirip dan nilai baru di test tetap terbaca; model yang sama sudah dipakai
  aplikasi (`../../pretrained/minilm`).
- 2026-10-31: Laya EDA membuat train_siap dengan menjalankan (exec) sel kode dari `codegen.py`, bukan
  fungsi terpisah. Alasan: file di aplikasi dan notebook yang ditempel dijamin sama. Akibatnya jawaban
  lensa Laya dan fitur non-pandas (MiniLM, alay, regex, bin) tidak masuk train_siap karena test tidak
  bisa dibuat ulang tanpa model; jawaban Laya tetap di data_berlensa.csv.

- 2026-10-30: repo dibuat compact: output 199 notebook `contoh/` dikosongkan, `_alat/sumber_arsenal/`
  dan `_alat/sumber_contoh/` dihapus setelah dicek sel per sel identik dengan notebook. Alasan: pengguna
  minta "compact"; output bisa dibuat ulang dengan Run All, skor tiap alternatif sudah ada di
  `contoh/KATALOG.md`. `_laya_eda/_uji/hasil_b/` tetap di git karena dipakai sebagai fixture
  `test_app.py` dan membuatnya ulang butuh model Laya.
- 2026-10-30: `ocr_end_to_end_v2` ditulis ulang ke gaya repo (menggantikan keputusan "apa adanya" di
  bawah). Perilaku dipertahankan; yang berubah: semua path ke `WORK_FOLDER`, `try` di `paddle_field`
  diganti cek `isinstance(dict)`, `assert` → `raise`, GUI dipecah jadi fungsi top-level + flag
  `RUN_LABEL_TOOL`, `DRAFT_N = 0` melewati draf, val YOLO ditulis ke `RUNS_FOLDER`.
- 2026-10-30: salinan repo lama di `Z:/GEMASTIK` diarsipkan, bukan dihapus. Alasan: semua commit-nya
  sudah ada di repo utama, sisa perubahan hanya path lokal; menghapus ±35 GB menunggu izin pengguna.

- 2026-10-30: angka deteksi figur PPT (664/674 citra uji berkotak entete, 10 tanpa) diambil dari
  prediksi ulang CPU `Downloads/yolo_2cls_v2.pt` (imgsz 416, conf 0,1) ke
  `Vignate-2/outputs/figures/_cache/det_v2_test.csv`, bukan dari output notebook (640/674). Alasan:
  pengguna memilih "pakai v2 aja"; `debug_v2.csv` dan `yolo_2cls.pt` lama tidak ada di laptop ini.
  Checkpoint di Downloads berupa folder (torch zip yang sudah diekstrak); disalin + di-zip ulang
  (ZIP_STORED) di scratchpad, aslinya tidak diubah. Model ini berbeda dari run notebook (conf 005e28
  0,823 vs 0,867). `f08_11_v2.py` (tabel02, funnel) masih memakai angka lama 639/35.
- 2026-10-30: PaperBanana tidak dipasang; yang diambil hanya daftar cek gaya dan kritiknya (lihat
  `aturan.md` bagian Figur). Alasan: butuh kunci API Gemini/OpenRouter dan internet (lomba luring),
  hasilnya gambar raster dari model pembuat gambar yang bisa mengarang teks/angka, kode plot statistiknya
  belum dirilis, dan isi figur terkirim ke layanan luar.
- 2026-10-30: pengingat memori dipasang sebagai hook Claude Code (SessionStart + UserPromptSubmit +
  Stop) yang hanya memakai bash + git. Alasan: aturan di CLAUDE.md saja bisa terlewat; jq/python belum
  tentu ada di PATH hook. Stop memblokir hanya sekali per giliran (`stop_hook_active`) supaya tidak
  berputar, dan hanya menilai perubahan selama giliran itu (potret saat prompt masuk), jadi perubahan
  lama milik pengguna tidak memicu.

- 2026-10-30: memori agen disimpan di dalam repo (`_memori/`) dan ikut di-commit, bukan hanya di
  memori lokal Claude. Alasan: sesi baru di komputer lain (atau agen lain) harus langsung tahu keadaan.
- 2026-10-30: notebook vignette patokan (`ocr_end_to_end_v2`) masuk `arsenal/pipeline/` apa adanya,
  output dikosongkan, isi kode tidak diubah. Alasan: pilihan pengguna; penyesuaian lint ditunda.
- 2026-10-24: arsenal ditulis langsung sebagai `.ipynb`, bukan `.py` bersel `# %%`. Alasan: pengguna
  menyalin sel dari notebook; `.py` sering lupa dikonversi.
- 2026-10-24: output semua notebook ke `../../outputs/`. Alasan: repo di-push, folder notebook harus bersih.
