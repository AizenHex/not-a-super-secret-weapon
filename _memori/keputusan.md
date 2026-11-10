# Keputusan

Terbaru di atas. Format: `- YYYY-MM-DD: <keputusan>. Alasan: <...>.`

- 2026-11-10: PaddleOCR dipasang lewat `requirements_paddle.txt` dengan `--no-deps` (paddlepaddle 3.3.1 CPU,
  paddlex 3.7.2, paddleocr 3.7.0, opencv-contrib-python 5.0.0.93 = versi opencv-python), dependensi lain di
  requirements.txt; model PP-OCRv6_medium det/rec + PP-LCNet doc_ori dari HF ke `pretrained/`. Alasan: paddlex
  mematok numpy<2.4 dan PyYAML==6.0.2, pemasangan biasa menurunkan numpy 2.5.3 seluruh venv dan membuat
  `pip install -r` gagal resolve; diuji jalan dengan numpy 2.5.3. opencv-contrib 4.10 bawaan paddlex
  ditolak karena menimpa cv2 milik opencv-python 5.0. paddlex hanya mengecek NAMA paket opencv-contrib.
  Versi CPU (bukan paddlepaddle-gpu) supaya tidak bentrok CUDA torch; Windows wajib `enable_mkldnn=False`
  (oneDNN error ConvertPirAttribute2RuntimeAttribute).

- 2026-11-10: metode niche dipilih yang belum ada di repo dan jalan di CPU: metaheuristik bobot (hill negatif,
  annealing, evolusi diferensial, genetik, Optuna), FWLS, gzip/BM25-kNN, null importance/Boruta/SA/GA, co/tri-
  training, kNN OOF + mini OpenFE, RuleFit. AWP/R-Drop/distilasi tidak dibuat. Alasan: hill Caruana, Nelder-Mead,
  rank, NB-SVM, FGM/SWA/LLRD, EM Saerens, cleanlab sudah ada; AWP/R-Drop butuh GPU untuk diuji. Semua tanpa paket
  baru (Boruta, BM25, OpenFE ditulis manual; `rank_bm25` lambat O(korpus) per kueri). gzip-kNN dibatasi
  `NCD_POOL` 500 karena ±20 µs per pasangan.

- 2026-11-10: fuzzy matching dibuat sebagai `fuzzy_kategori` (rapikan kategori, cocok ke acuan, hampir kembar),
  bukan record linkage dua tabel. Alasan: `contoh/multitabel/entity_resolution` sudah mencakup linkage.
  `interval_prediksi` dan `uji_beda_model` bekerja di atas file OOF, tanpa melatih ulang. Alasan: berlaku untuk
  model apa pun yang sudah menulis OOF, cepat dipakai menjelang presentasi.

- 2026-11-09: `klasifikasi_hierarkis` memakai SGDClassifier log-loss (α 1e-5) sebagai model dasar, LogReg jadi alt.
  Alasan: banyak model (per tingkat + per induk × 5 lipatan); LogReg C4 = 14 menit di baju, SGD 1 menit dengan F1
  satu lipatan lebih tinggi (0,465 vs 0,457). Simpul = jalur penuh supaya nama anak yang sama di dua induk aman.

- 2026-11-09: teks + kolom tabel digabung lewat stacking (OOF model teks + fitur tetangga → LightGBM) dan alt
  `VERBAL_COLS` di `finetune_oof`, bukan jaringan fusi (BERT + MLP tabel). Alasan: CPU-only cukup, memakai lipatan
  dan format OOF yang sama, terbukti di baju (0,468 vs SVD 0,399); fusi butuh kode training baru + GPU.
- 2026-11-09: `pasca_prediksi` menyimpan `oof_pasca/test_pasca.npy` di OUTPUT_FOLDER, bukan OOF_FOLDER. Alasan:
  isinya turunan blend; kalau masuk OOF_FOLDER, `ensemble_oof` akan menggabungkannya lagi dengan sumbernya.
- 2026-11-09: regresi teks+tabel memakai fitur kelas-bin (LogReg pada bin kuantil target) di samping Ridge.
  Alasan: menangkap bentuk sebaran target (rating 1 vs 5); di baju nilai harapan bin 0,684 < Ridge 0,714.

- 2026-11-08: Laya v3 target encoding OOF di langkah resep: fungsi tahu sedang di train lewat `df is LATIH` (train
  dijalankan `f(df, df)`), maka nilai train dihitung out-of-fold (k=5, smoothing m=10) dan test memakai rata-rata seluruh
  train. Alasan: tidak perlu jalur khusus di mesin resep; ekspor .py berperilaku sama.
- 2026-11-08: drift (PSI pada desil train + KS) dan skema test dihitung dengan polars + stdlib di `temuan.py`, bukan
  numpy/scipy. Alasan: Temuan berjalan di exe (tanpa venv); hanya tugas berat yang lewat python venv.
- 2026-11-08: cek bocor per kolom = LightGBM 60 pohon per satu kolom, 3 fold pertama dari skema validasi; curiga bila
  macro-F1 >= 0,9 atau F1 satu kelas >= 0,95. Alasan: menangkap kolom yang menebak satu kelas saja (Riwayat Beasiswa
  UKT: kelas 0 F1 0,998 walau macro-F1 0,18), yang tidak terlihat dari skor rata-rata.
- 2026-11-08: undo/redo resep menyimpan salinan seluruh resep (semua tabel, maks 50) sebelum tiap perubahan, bukan
  operasi per langkah. Alasan: satu titik (`_ubah_resep`) menangkap semua jenis ubahan; resep kecil.
- 2026-11-07: Laya v3 baseline tabular memakai LightGBM (200 pohon, lr 0,1, n_jobs <= 8: 4 s/fold di UKT 30 rb baris);
  teks bebas (unik > 50% baris) dilewati, teks pendek jadi kategori. Alasan: model teks salah untuk data tabel
  (screenshot pengguna), LightGBM sudah di venv dan tahan outlier/kosong. Hasil model tampil di tab Model sendiri
  (panel kanan disembunyikan), bukan panel sempit. Validasi disimpan per proyek dan dipakai semua tugas.
- 2026-11-07: `ensemble_oof` memilih cara gabung dari SKOR JUJUR (bobot dicari di 4/5 baris OOF, dinilai di 1/5) dan memilih yang paling
  sederhana dalam `MIN_GAIN` dari yang terbaik; OOF gabungan ditulis ke `OUTPUT_FOLDER`, bukan `OOF_FOLDER`; OOF dataset lain disaring
  dengan bentuk + skor di atas OOF yang barisnya diacak. Alasan: hill climbing/stacking di OOF yang sama dengan penilaiannya
  optimistis (smsa: hill dalam sampel 0,8658, jujur 0,8589); tebakan konstan terlalu rendah di F1 makro sehingga OOF dataset
  lain sebentuk lolos; menyimpan gabungan di folder bersama akan ikut tergabung di run berikutnya.

- 2026-11-07: Tugas model Laya (M4) memakai HashingVectorizer kata 1–2 paralel + SGDClassifier log-loss, tanpa
  n-gram huruf, dijalankan python venv lewat QProcess; validator submission (H1) jalan di dalam Laya. Alasan:
  gerbang VISI F1 ≤ 60 s di GDOC; TfidfVectorizer kata 51 s dan + n-gram huruf 143 s, lbfgs menit-an, sedangkan
  hashing paralel ± 8 s; n-gram huruf tidak menaikkan macro-F1 GDOC (0,571 vs 0,586). Proses terpisah supaya exe
  tetap kecil (tanpa sklearn) dan crash/Batal tidak menjatuhkan Laya. H1 cukup polars, jadi tidak perlu venv.

- 2026-11-06: Grafik Laya v3 digambar sendiri dengan QPainter (grafik mini per baris profil) dan batang HTML di
  rincian, bukan QtCharts/matplotlib; histogram dipotong di p99. Alasan: pengguna minta grafik; QtCharts ada di
  PySide6-Addons (tidak terpasang, exe membesar), matplotlib berat; grafik mini sebaris membuat semua kolom bisa
  dibandingkan sekilas. Skala p99 karena panjang teks berekor panjang (satu teks 18 ribu huruf membuat 96% baris
  masuk satu batang).

- 2026-11-06: Rombak UI Laya v3 memakai pola "pilih di tengah → rincian di kanan" dengan widget Qt bawaan
  (QTableWidget untuk profil, QTreeWidget untuk temuan, QTextBrowser + HTML untuk rincian, batang histogram dari
  tabel HTML), bukan grafik. Temuan sejenis lintas kolom (list-sebagai-teks, mojibake) digabung jadi satu butir.
  Alasan: pengguna menilai teks polos "chaos" dan susah dinavigasi; pola tiga panel konsisten di semua tab; tanpa
  pustaka tambahan (matplotlib/QtCharts) dan exe tidak membesar. Ditolak: dasbor kartu/grafik (lebih banyak kode,
  lebih lambat di 50 ribu baris), tab terpisah per tingkat temuan (navigasi bertambah).

- 2026-11-06: Laya v3 M3: temuan dihitung pada **tahap resep yang sedang dilihat** (bukan selalu data mentah), hanya
  saat tab Profil/Temuan dibuka, satu kali per tahap. Tiap temuan membawa ekspresi polars untuk "Lihat baris" dan
  langkah usulan yang lewat form biasa (pengguna tetap melihat dan mengubah kodenya). Label ditebak sebagai kolom
  train yang tidak ada di test, lalu disimpan dan bisa diganti. Pencarian yang dijadikan langkah = langkah kode
  `df.filter(...)` bertanda latih saja. Profil berupa teks + histogram huruf, bukan grafik. Mojibake diperbaiki per
  rangkaian non-ASCII yang hasil baliknya UTF-8 sah. Alasan: setelah langkah, temuan menunjukkan apakah langkah itu
  berhasil; menghitung semua tab terus-menerus memakan ~11 s waktu mesin di GDOC; saring baris test merusak
  submission; grafik butuh pustaka tambahan (matplotlib/QtCharts) untuk nilai yang kecil. Ditolak: cek temuan
  per sel (lambat), perbaiki mojibake seluruh teks sekaligus (gagal bila teks campur benar dan rusak).

- 2026-11-06: Laya v3 M2: langkah resep disimpan sebagai **kode polars** (plus isian form untuk diubah lagi), bukan
  sebagai operasi abstrak; app dan ekspor menjalankan string kode yang sama, dan `bantu.py`/`muat.py` disalin apa
  adanya ke skrip ekspor. Tabel test = "pasangan" yang ikut resep train: tiap langkah menerima `LATIH` (train sebelum
  langkah itu) untuk median/modus; saring baris, buang duplikat, tumpuk default "latih saja". Cache tahap = hash
  (kunci tahap sebelumnya, kode, kunci tabel acuan, kunci tahap latih). Preset "huruf berulang" memakai 63 kali
  `replace_all` polars, bukan regex backreference Python. Alasan: satu sumber kode = ekspor pasti sama dengan app
  (gerbang 6.5, diuji di subprocess); LATIH mencegah bocor statistik test; regex Python 8,8 s vs polars 0,4 s di
  konten GDOC dengan hasil identik (batas: huruf non-ASCII berulang tidak diringkas). Ditolak: kode ekspor dibuat
  ulang dari operasi abstrak (dua jalur yang bisa beda), pipeline fit/transform ala sklearn (berat, pandas).

- 2026-11-06: Laya v3 membaca CSV dengan semua kolom sebagai teks, lalu `tebak_tipe` mengubah ke angka hanya bila
  seluruh isi kolom angka biasa; nol depan (`007`), format Indonesia (`1.234,5`), dan bulat > Int64 tetap teks.
  Alasan: tebakan tipe polars membuang nol depan kode/ID tanpa jejak (melanggar "tidak menebak diam-diam") dan bisa
  gagal di baris > 10 rb. JSON berbentuk objek: dipakai daftar record di kunci pertama berisi list (dicatat di
  laporan), bukan satu baris struct. Proyek + log disimpan di `proyek/` di samping exe (cadangan LOCALAPPDATA/Laya),
  bukan di folder data (N4). Exe mengecualikan torch/pandas/numpy/geo/cloud (513 → 303 MB); latih berat tetap di venv.

- 2026-11-06: Laya v3 = kokpit lomba. Exe (PySide6 + polars) memuat, memahami, merapikan, dan mengekspor; latih berat
  (AutoGluon Tabular, fine-tune transformer, ensemble) jalan sebagai proses terpisah memakai venv, lewat notebook
  arsenal (`# UBAH` jadi formulir, nbclient) dan AutoGluon. Fitur yang belum lulus gerbangnya disembunyikan di build
  lomba. Menggantikan "latih model tidak masuk Laya" (entri di bawah). Alasan: pengguna minta fitur sebanyak mungkin
  termasuk AutoGluon; proses terpisah menjaga exe kecil dan Laya tidak jatuh saat GPU kehabisan memori; notebook
  arsenal sudah teruji, jadi tidak ada kode latih kedua. Ditolak: AutoGluon preset extreme (VRAM 20–32 GB + unduh
  bobot saat fit; PC lomba 8 GB, offline), pasang `autogluon.multimodal` sebelum lomba (versi torch/transformers bisa
  berubah), asisten LLM di Laya (aturan tanpa AI).
- 2026-11-06: Laya v3 (usulan, menunggu persetujuan pengguna di `_laya_eda_v3/VISI.md` bagian 8) = PySide6 + polars +
  PyInstaller; langkah = kode polars yang di-`exec` apa adanya dan diekspor sama persis; latih model tidak masuk Laya
  (bagian latih diganti entri di atas).
  Alasan: pengguna minta `.exe` bukan web; tabel Qt virtual; polars membaca GDOC 229 MB 0,5 s (v2 buka app 20 s);
  satu sumber kode mencegah beda hasil app vs ekspor; fitur latih v2 (E1–E12) melebarkan lingkup dan sudah ada di
  notebook arsenal. Ditolak: Tkinter (tabel lemah), Electron/Tauri (web), C# (ekspor bukan Python), DuckDB (belum perlu).

- 2026-11-06: Rombak `tabular/multitabel` tetap memakai Berka sebagai data demo (bukan data triase GDOC), tetapi semua
  langkah jadi fungsi berparameter + tabel "jenis tabel → fungsi → sel". Alasan: Berka sudah ada di `data_latihan`
  (punya kunci jawaban test); data GDOC di luar repo. Keumuman diuji dengan menjalankan sel 1–2 di data GDOC.

- 2026-11-06: T11 menambah sel di AKHIR notebook arsenal (bagian baru 8/9/5/10/9), bukan menyisip di tengah; satu-satunya
  penyisipan adalah sel `score` F1 makro (bagian 1b, ber-`#`) di ensembling. Alasan: nomor bagian di `arsenal/PETA.md` dan
  rujukan lain tidak bergeser. Rank average dan kalibrasi tetap ditulis walau rank average buruk untuk multikelas, karena
  berguna untuk AUC biner; peringatan ditaruh di markdown. Kalibrasi memakai CV 5 fold untuk membandingkan temperature vs
  isotonic (isotonic mudah overfit), bukan menilai di OOF yang sama dengan fitnya.

- 2026-11-06: T8 memakai `baju_wanita` (Women's E-Commerce Clothing Reviews, CC0, target `rating` 5 kelas) sebagai data demo,
  `recommended_ind` dibuang. Alasan: teks panjang + angka + kategori, lisensi terbuka, multikelas; `recommended_ind` turunan rating.
  Model: GBDT di atas SVD (0,399) sedikit di bawah TF-IDF+LogReg (0,434); tetap dipakai karena inti T8 menggabungkan kolom tabel.
- 2026-11-06: `run_cv` memakai `copy.deepcopy(model)`, bukan `sklearn.clone`. Alasan: `clone` menolak CatBoost dengan `cat_features`.

- 2026-11-06: `baseline_oof` memakai SVM terkalibrasi (bukan LogReg) sebagai model aktif; LogReg jadi alt. Alasan: OOF
  smsa LogReg 0,8528 < target 0,86, SVC ≈0,860. Pengali kelas dipakai hanya bila naik ≥ `MIN_GAIN` 0,002 (dihitung di
  OOF yang sama, optimistis). Sel sample demo hanya menulis bila `SAMPLE_CSV` di `OUTPUT_FOLDER`, agar sample panitia tak tertimpa.

- 2026-11-06: validator submission (T3) dibuat dua sel berurutan (#15 struktur+id, #16 isi+label) dan tipe kolom
  dibandingkan lewat `dtype` pandas, bukan teks mentah. Alasan: satu sel tidak muat 30 baris; dtype memberi
  pembeda "3" vs "3.0" tanpa fungsi tambahan (harga: sample palsu semua 0 vs probabilitas memicu pesan tipe).
- 2026-11-06: tebakan encoding `cp1250`/`iso8859_2` dari charset_normalizer ditimpa `cp1252`. Alasan: teks soal
  Indonesia/Inggris; pada file latin-1 kecil `ï` terbaca `ď` (hanya cp1250 yang jadi kandidat) → data rusak diam-diam.
- 2026-11-06: `unduh_model.py` melewati bobot ganda hanya per folder (`SKIP_EXTRA` untuk xlmr_base/e5_base) dan
  berkas latih (`optimizer.pt` dkk) global. Alasan: indobert_large/indobertweet hanya punya `.bin`, skip global
  `pytorch_model.bin` akan merusaknya; hemat ± 3,9 GB unduhan hari setup.

- 2026-11-06: `baseline_oof`, `finetune_oof`, `ensemble_oof` jadi notebook contoh baru; `klasifikasi_tfidf` dan
  `klasifikasi_finetune` tetap sebagai jalur cepat. Alasan: k-fold + simpan OOF mengubah alur latih (aturan
  pemisah pengguna), dan ensemble lintas notebook butuh OOF dari semua model.
- 2026-11-06: format OOF bersama (`oof_<MODEL_NAME>.npy`/`test_<MODEL_NAME>.npy` di `outputs/text/oof`, urutan
  kelas `np.unique`, StratifiedKFold(5, shuffle, seed 42) pada urutan train asli, `percobaan.csv`). Alasan: tanpa
  urutan baris/kelas/fold yang sama, hill climbing di OOF tidak sah.
- 2026-11-06: prediksi test = rata-rata model per fold, tidak latih ulang di semua data. Alasan: 5 jam lomba;
  latih ulang menambah satu putaran penuh tanpa bisa divalidasi.
- 2026-11-06: file bersama punya pemilik tunggal: `requirements.txt` = T1, `_alat/unduh_model.py` = T4,
  KATALOG/PANDUAN/PETA/SOURCES_text = T13; butir lain menulis blok usulan di STATUS-nya. Alasan: butir G1 jalan
  paralel; dua sesi menulis file yang sama = konflik.
- 2026-11-06: model baru = indobert_large, indobertweet, xlmr_base, e5_base (± 4 GB, dugaan). Ditolak: bge-m3
  (± 2,2 GB, e5 cukup), mdeberta-v3-base (nli_mdeberta sudah ada sebagai backbone), mT5/IndoT5 (peluang soal
  generasi kecil, B9). Alasan: menutup Indonesia formal besar, medsos, campur bahasa, embedding lebih kuat.
- 2026-11-06: garis potong sesudah T13; TAPT/MLM (B1) dan NER-BERT end-to-end (B2) di bawah garis. Alasan:
  jalur kritis GPU ± 17 jam sudah memakan sisa waktu sampai 2026-11-08; klasifikasi lebih mungkin muncul (dugaan).
  Disetujui pengguna ("Ya, semua").
- 2026-11-06: uji notebook teks memakai `.venv` baru dari `setup.bat`, bukan Python sistem. Alasan: pilihan
  pengguna (rekomendasi); sama dengan cara pasang di PC panitia, jadi sekaligus gladi setup.

- 2026-11-06: Prompt A memakai subagent tipe general-purpose (bukan Explore) untuk membaca, perencana membaca
  sendiri file inti (memori, README, RENCANA_LOMBA, PETA, PANDUAN, KATALOG teks, unduh_model.py) dan mengecek
  klaim penentu. Alasan: bahan ±1 MB tidak muat satu konteks; Explore hanya membaca potongan, padahal laporan
  per notebook butuh isi penuh.
- 2026-11-06: perencana tidak mengukur VRAM/waktu; angkanya "dugaan" dan pengukuran jadi uji terima butir model.
  Alasan: Prompt A melarang menulis kode, sedangkan "ukur, jangan tebak" butuh menjalankan model.
- 2026-11-06: cara menangani file bersama (KATALOG, PANDUAN, PETA, unduh_model.py, requirements.txt,
  SOURCES_text.md) diserahkan ke perencana dengan dua opsi (butir integrasi per grup, atau blok milik sendiri +
  baca ulang sebelum edit). Alasan: butir contoh/model paralel pasti menyentuh file yang sama; pilihan terbaik
  bergantung pada jumlah butir yang belum diketahui.

- 2026-11-02: overhaul umum = "inti umum dulu" (pilihan pengguna dari 3 opsi): mesin Temuan baru
  `core/findings.py` (detektor profil kolom, bukan kamus), `core/reader.py` (BOM/pemisah/tanpa judul/'NA' teks,
  blok baca yang sama di skrip), kamus proyek `core/vocab.py` (UMUM default, PRESET_UKT opsional; nilai kamus
  disalin ke parameter langkah saat dibuat), op umum baru (fitur tanggal, statistik teks, jadikan kategori, nilai →
  kosong, penanda kosong, kode bebas), ✎ di sidebar + "ubah jadi kode bebas". Arsitektur langkah → kode, Latih,
  Submission dipertahankan. Alasan: lomba 2026-11-11; tulis ulang total berisiko belum stabil.
- 2026-11-06: persiapan final teks dibagi dua tahap: perencana (Opus xhigh/max) menulis `RENCANA_TEKS.md`
  tanpa kode, lalu pekerja (Opus medium/high) per butir dengan uji terima tertulis. Alasan: pilihan
  pengguna; peta + uji terima membuat pekerja tidak perlu konteks percakapan, dan grup paralel mencegah
  dua sesi menyentuh file yang sama. Prompt di `PROMPT_FINAL_TEKS.md`.
- 2026-11-02: sesi tersimpan tanpa `kosakata` yang sudah punya langkah dianggap proyek UKT (preset UKT); proyek
  baru = UMUM. Langkah lama tanpa parameter `singkatan`/`pemisah` memakai nilai lama. Alasan: sesi UKT pengguna
  tidak berubah perilakunya sesudah overhaul.
- 2026-11-02: dataset uji umum diambil dari cermin GitHub (Hugging Face, OpenML, UCI, Kaggle diblokir proxy
  sesi cloud, 403) dan disimpan di `_laya_eda/data/umum/` (diabaikan git). Alasan: tidak boleh commit data.

- 2026-11-02: R18 di aplikasi = peringatan (tabel tetap tampil), galat keras hanya di uji (`LAYA_KAMUS_KETAT`).
  Alasan: satu tooltip yang kurang tidak boleh mematikan seluruh halaman Temuan (laporan pengguna); kekurangan tetap
  terlihat dan uji tetap gagal, ditambah cek statis kunci `pd.DataFrame({...})`.

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
