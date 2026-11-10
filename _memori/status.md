# Status repo

Terakhir diperbarui: 2026-11-10.

## Tujuan

Arsenal offline untuk final GEMASTIK XIX Penambangan Data: setup 2026-11-10, pengerjaan
2026-11-11 13.00–18.00 (5 jam, luring, tanpa internet; AI boleh hanya untuk bantu kode hal tak terduga, koreksi pengguna 2026-11-07), presentasi 2026-11-12.
Pengguna berangkat 2026-11-09. Prioritas: [`../RENCANA_TEKS.md`](../RENCANA_TEKS.md) (teks, rinci) dan [`../RENCANA_LOMBA.md`](../RENCANA_LOMBA.md) (bobot nilai, pohon lama).

**Konfirmasi panitia (pengguna, 2026-11-10): topik final = KLASIFIKASI TEKS + TOPIC MODELING.** Zero-shot dan
few-shot DILARANG kecuali modelnya di-fine-tune sendiri (jangan pakai `zero_shot_nli`, `llm_prompt`, `rag_llm`,
`arsenal/teks/zero_shot` sebagai prediksi; backbone yang sama boleh bila di-fine-tune). Submission ke Kaggle hanya
**3 kali, di 1 jam terakhir** (internet baru ada saat itu): tidak ada umpan balik leaderboard selama 4 jam pertama,
jadi keputusan model sepenuhnya dari CV lokal + validator. Boleh menyalin model dan screenshot ke laptop untuk
menyiapkan presentasi. Belum diketahui: bagaimana topic modeling dinilai (leaderboard atau hanya presentasi).
Celah repo untuk topic modeling (dicek 2026-11-10): `topik`, `topik_bertopic`, `topik_dinamis` hanya menilai dengan
NMI/ARI terhadap label; belum ada koherensi (NPMI/c_v), keragaman topik, pemilihan K, topik x kelas. `gensim` tidak
terpasang.

**Info panitia (pengguna, 2026-11-06): data final 2026 (tuan rumah BINUS) dipastikan TEKS.** Bentuk
spesifik dan jebakannya belum diketahui. Pengguna minta repo dikembangkan agar siap untuk soal ML teks apa
pun; prioritas di bawah (deteksi, RLE, dll.) perlu disusun ulang ke teks, menunggu persetujuan rencana.
**Email panitia (2026-11-06):** tanpa internet; tiap tim dapat 1 PC panitia (i7-12700, RAM 32 GB, SSD 1 TB,
**RTX 3070 LHR 8 GB VRAM**). Akses dataset, training, dan submission **wajib di PC panitia**; laptop pribadi
hanya pendukung (tidak boleh menyentuh data). Akibatnya: arsenal + venv/wheelhouse + model pretrained harus
bisa dipasang di PC asing. **Hari setup (1 hari sebelum lomba) ada internet** (pengguna, 2026-11-06): pasang
via `setup.bat` + `download_model.bat` online di PC panitia (keduanya sudah ada dan sudah sering dipakai
pengguna; tidak perlu skrip baru); wheelhouse/flashdisk hanya cadangan. Model > 0,5 GB boleh dipakai. Belum
diketahui: OS, Python/driver NVIDIA terpasang, apakah PC tidak di-reset antara setup dan hari lomba. Torch di requirements = cu128 (butuh driver NVIDIA ≥ 570).
Persiapan teks berjalan lewat `PROMPT_FINAL_TEKS.md` (2026-11-06): Prompt A membuat `RENCANA_TEKS.md`
(belum ada), Prompt B mengerjakan butirnya satu per satu. `RENCANA_TEKS.md` jadi sumber daftar kerja teks.
Revisi prompt 2026-11-06: dua pilar diperbarui, `arsenal/` dan `contoh/text/` (soal teks baru, metode baru,
model kecil baru; notebook sendiri hanya bila loader/alur latih berubah, ganti satu fungsi = alternatif
berkomentar); peta memuat metode ML standar/niche/SOTA; subagent boleh untuk membaca.
**Prompt A sudah dijalankan 2026-11-06**: [`../RENCANA_TEKS.md`](../RENCANA_TEKS.md) = daftar kerja teks (butir
T1–T13 di atas garis potong, B1–B12 di bawah; STATUS tiap butir masih kosong). Keempat pertanyaan di atas file itu
sudah dijawab "Ya, semua" (venv lewat `setup.bat`, pip `emoji`/`ipywidgets`, unduh 4 model ± 4 GB, garis potong
sesudah T13). Berikutnya Prompt B: G0 = T1 + T4 bersamaan. T1 selesai 2026-11-06 (32/32 contoh/text + 13/13 arsenal teks LOLOS; STATUS di RENCANA_TEKS.md). Temuan terbuka: kernelspec `python3` pengguna menunjuk venv yang tidak ada, `bangun_contoh.bat` perlu `JUPYTER_PATH`/`ipykernel install --user` (menunggu keputusan pengguna).
**T4 selesai 2026-11-06**: 4 model ada di `pretrained/` (indobert_large, indobertweet, xlmr_base, e5_base; ± 2,9 GB),
dimuat offline; alt e5 di `klasifikasi_embedding` F1 0.8476. Blok KATALOG menunggu T13 (di STATUS T4).
**Venv `Z:/GEMASTIK/Vignate-GEMASTIK/.venv` dibuat T1 (2026-11-06)**, Python 3.13.15, requirements + emoji/ipywidgets terpasang; hasil uji Run All di STATUS T1 `RENCANA_TEKS.md`.
GPU laptop: RTX 5060 Laptop 8 GB.
Latihan terakhir pengguna: soal 2024 Golongan UKT di `Z:/GEMASTIK/GEMASTIK-2024` (`train.ipynb`, XGBoost,
F1 makro val 0,427; kelas ke-2 F1 0).

## Keadaan sekarang

- 2026-11-10: `contoh/text/katalog_algoritma.ipynb` = contoh kode tiap algoritma dari daftar pengguna (ML klasik,
  RNN/LSTM/GRU/Bi-/Attention/Text-CNN/Transformer dari nol, fine-tune HF satu fungsi, Word2Vec/GloVe/FastText
  dari nol di torch), skor di split sama. Lint 0, Run All lulus di SmSA 3000 baris (GPU). ALBERT/XLNet/ERNIE/T5/
  BART (T5 = IndoNanoT5) ber-`#`, sudah di `unduh_model.py` dan terunduh; semua 12 model C diuji jalan.
  Versi awal di-commit (2ffd26d); perubahan unduh_model + komentar MODEL_PATHS belum di-commit.
- 2026-11-10: `contoh/text/eda_lengkap.ipynb` = EDA teks + angka/kategori (hanya membaca, tulis `temuan.md`),
  dijalankan paling awal. Lint 0; Run All lulus di demo baju, SmSA regresi (target buatan) dan SmSA klasifikasi,
  keduanya dengan `DATE_COL`, `LANGUAGE="id"`, tanpa kolom tabel. Belum diuji: target kosong (qcut gagal), data
  tanpa kolom id. Sudah di-commit.
- 2026-11-10: `contoh/text/praproses_lengkap.ipynb` = langkah sesudah EDA di jalur teks (periksa → bersih ringan/berat →
  ukur per langkah → `train_bersih.csv`/`test_bersih.csv`); notebook model membacanya lewat `TRAIN_CSV`/`TEST_CSV`,
  `TEXT_COLS` asli untuk BERT, `<kolom>_tfidf` untuk TF-IDF. Lint 0, Run All demo lulus, sudah di-commit.

- Branch kerja: `rapikan-struktur` (remote `origin` = github.com/AizenHex/not-a-super-secret-weapon).
  Sesi cloud 2026-11-02 (Laya v2) bekerja di `new-gen` atas perintah pengguna.
  Default remote: `new-gen`, disamakan dengan `rapikan-struktur` 2026-10-31 (pengguna mulai
  memakai repo dengan serius). Branch `origin/kode-run` ("Add files via upload") bukan dari agen.
- `.claude/` (2026-10-30): hook memori (`settings.json`, `hooks/memori.sh`). Aktif kalau Claude Code
  dibuka dari root repo ini; tinjau/matikan lewat `/hooks`.
- `arsenal/`: potongan siap tempel per modalitas (vision, teks, tabular, jembatan, umum, pipeline).
  Peta: [`../arsenal/PETA.md`](../arsenal/PETA.md).
- `contoh/`: 199 notebook utuh, 19 modalitas, disimpan **tanpa output** sejak 2026-10-30
  (repo 86 MB → 6,8 MB). `_alat/sumber_arsenal/` dan `_alat/sumber_contoh/` dihapus (isinya identik
  dengan notebook); `.ipynb` satu-satunya sumber. `_alat/bangun_contoh.bat` menjalankan notebook
  langsung, salinan berisi output ke `Z:/GEMASTIK/Vignate-GEMASTIK/outputs/uji_contoh/`.
- `arsenal/pipeline/ocr_end_to_end_v2.ipynb` (2026-10-30): notebook vignette patokan (YOLOv8n 2 kelas +
  orientasi Paddle + PaddleOCR → submission), ditulis ulang ke gaya repo: lint 0 masalah, semua hasil
  ke `WORK_FOLDER` (default `../../../outputs/pipeline/ocr_end_to_end_v2`). Bagian 0–2 (termasuk GUI
  label) diuji pada `Vignate-GEMASTIK/data`; bagian 3–6 **belum diuji** (`yolo_2cls.pt` tidak ada di laptop). PaddleOCR terpasang 2026-11-10,
  model dari `pretrained/paddleocr_det|_rec|_ori`; `get_paddle`/`get_ori` diuji di luar notebook.
- Di luar repo (2026-10-30): salinan repo lama (`PythonProject`, `Sen-3`..`Sen-6`, `Sentiment-2`, ±35 GB)
  dan file lepas di root `Vignate-GEMASTIK` dipindah ke `Z:/GEMASTIK/_arsip/`, tidak dihapus.
- `_laya_eda/` (Laya v1) dihapus pengguna (commit 3a375ec "laya v1 removed", 2026-11-06); yang dipakai
  `_laya_eda_v2/`. Data riil Laya sekarang dirujuk sebagai `../../HAND-WRITE/train.csv`.
- Laya v2 perbaikan 2026-11-06 (belum di-commit): batas unggah 2048 MB + peringatan KEEP dimatikan
  (`.streamlit/config.toml`), `reader.sniff` aman untuk teks berkutip panjang, langkah `regex_replace`/`regex_extract`
  (+ pola siap pakai pembersih teks), kolom teks panjang dikenali (`text_tools.is_long_text`) sehingga halaman tidak
  lagi jeda 15–150 s di CSV 229 MB (sisa: Temuan kunjungan pertama 23 s). Pengguna menjalankan Laya dari salinan
  `Z:/GEMASTIK/GDOC/not-a-super-secret-weapon`, yang belum punya perbaikan ini sampai ditarik.
- **Laya DIBEKUKAN (2026-11-06)**: pengguna tidak memakainya untuk final. Perubahan Laya yang belum di-commit
  dibiarkan apa adanya.
- **Laya v3 (`_laya_eda_v3/`, 2026-11-06)**: pengguna memilih mulai sekarang, sebagai `.exe` desktop, dan minta fitur
  "tank" sebanyak mungkin (termasuk AutoGluon/latih). Baru ada `VISI.md`: kokpit lomba, 79 fitur (17 P0, 44 P1,
  17 P2) dengan gerbang, latih lewat proses venv + komandan notebook arsenal, jadwal M1–M5 s.d. 8 Nov malam, garis
  potong Sabtu malam. 5 keputusan di bagian 8 menunggu pengguna (termasuk pasang ftfy/datasketch/grex dan 3 pertanyaan
  untuk panitia). Katalog disetujui pengguna; **M1 selesai 2026-11-06** (muat folder, baca tahan banting, grid
  virtual, panel isi, opsi baca, proyek otomatis, `Laya.exe` 303 MB via `buat_exe.bat`, cadangan `jalankan.bat`,
  `python uji.py` lulus). Kecepatan N1 belum diukur pengguna. **M2 selesai 2026-11-06 malam**: resep 17 jenis langkah
  (`mesin/langkah.py`), mesin tahap ber-cache + pasangan train/test + tabel turunan `T[...]` (`mesin/resep.py`),
  fungsi bantu yang ikut tersalin ke ekspor (`mesin/bantu.py`), ekspor .py/.ipynb + simpan parquet/csv
  (`mesin/ekspor.py`), panel resep + form langkah dengan pratinjau (`tampilan/form.py`). Gerbang 6.3/6.5 lulus di
  `uji.py`; belum dicoba pengguna dengan klik nyata. **M3 selesai 2026-11-06 malam**: `mesin/temuan.py` (profil
  kolom, temuan integritas/teks/label, banding train-test, relasi antartabel, cari teks/regex/SQL/kode, sampel acak
  dan per label) + tab Data/Profil/Temuan di jendela ("Lihat baris", "Jadikan langkah"). Gerbang 6.4 lulus otomatis
  (`uji.py` → `uji_emas`, GDOC + Berka). **UI dirombak 2026-11-06 malam** setelah kritik pengguna (susah navigasi,
  profil kacau): kiri tabel/resep, tengah tab Data/Profil/Temuan, kanan rincian dari yang dipilih
  (`tampilan/pahami.py`), plus grafik mini sebaran per kolom di tabel Profil; pengguna oke dengan UI ini (2026-11-07).
  **M4 selesai kode 2026-11-07**: tab Model (Ctrl+4, `tampilan/tab_model.py`) menjalankan `mesin/tugas.py` di python
  venv lewat QProcess (antrean, log langsung, Batal membunuh proses): baseline + label mencurigakan (cleanlab),
  kata khas per label, uji label acak, adversarial train vs test; hasil di panel kanan HASIL MODEL; "Lihat baris" /
  "Jadikan langkah" untuk baris curiga. Menu Model → Validasi file submission (Ctrl+M, `mesin/submisi.py`). Python
  venv dicari: `proyek/python.txt` (tombol Python…), VIRTUAL_ENV, lalu `.venv`/`venv` di folder Laya/data dan
  induknya (`mesin/venv.py`). GDOC: baseline 40,8 s (macro-F1 0,586), adversarial 37 s (AUC 0,499), kata 11 s di
  laptop 32 thread; PC panitia 20 thread belum diukur. Exe uji di scratchpad lulus (`--uji` menjalankan tugas kata
  sungguhan lewat python venv); `dist\Laya` di-build ulang 2026-11-07 00.44: `--uji` GDOC 0 gagal, python ditemukan otomatis dari `.venv` repo, tugas kata exit 0. Belum dicoba pengguna dengan klik nyata. **2026-11-07 siang: perbaikan UI/UX setelah keluhan pengguna**: bilah alur
  5 langkah, warna ikut tema, hasil model lebar, langkah outlier + temuan penanda/outlier, kontrol validasi (6 cara),
  Baseline tabular (LightGBM); exe di-build ulang 11.10 (`--uji` 0 gagal). Celah fitur berprioritas di VISI 8b. **2026-11-08: semua butir VISI 8b selesai** (rentang, encode, langka, penanda kosong, tanggal, skala, nilai mirip, isi per grup, fold.csv, cek bocor, asosiasi, drift/skema, undo/redo, banding versi resep); `uji.py` lulus; exe di-build ulang 2026-11-08.
  Jenis langkah encoding (D11 di VISI, dulu ditunda) sudah dibuat 2026-11-08. PySide6 6.12.0 +
  pyinstaller 6.22.3 terpasang di `../.venv`; `requirements.txt` + polars/PySide6_Essentials/pyinstaller. Semua
  belum di-commit.
- Temuan data GDOC (berguna untuk latihan, diukur 2026-11-06): 32% baris test punya `konten_berita` persis sama
  dengan train (label kembar di train 0 konflik); 11.806 baris train berkonten kembar = artikel sama yang terambil
  lewat beberapa kategori/kata kunci (CV harus per konten); `sentimen_kutipan` = label 58%; footer
  `COPYRIGHT ¬© ANTARA`; kolom list-sebagai-teks. Rincian di `_laya_eda_v3/VISI.md` 6.4.
- **Rencana pengguna (2026-11-06): UI Laya diganti dari Streamlit ke Flask + web, "nanti"** (belum dimulai, waktu
  belum ditentukan). Ukuran: `ui/` + `app.py` 2.867 baris (379 baris memanggil `st.*`), `core/` 3.546 baris tanpa
  Streamlit (dipakai ulang). Ukur per halaman (CSV 229 MB): jeda berasal dari hitungan, bukan Streamlit (Temuan 94 s,
  Ejaan 75 s, Isian bebas 155 s/47 s, Latih 14 s dan Laya 16 s tiap kunjungan; halaman ringan 0,1 s). Rincian di log.
- `_laya_eda_v2/` (2026-11-02, sesi cloud, branch `new-gen`): `VISI.md` = satu-satunya sumber target (extra
  E1–E12 di atas R1–R18; 8 temuan review sudah masuk). E1–E12 diimplementasikan (tab **Latih** dan **Submission**,
  14 halaman) dan R1–R18 punya uji. `python -m tests.semua` di venv cloud (versi persis requirements.txt, torch
  2.11.0 dari PyPI): lulus 47, gagal 0, dilewati 12. Semua uji memakai data **sintetis** (`tests/sintetis.py`);
  dengan data riil unggahan pengguna (`../_laya_eda/data/`, diabaikan git): lulus 57, gagal 0, dilewati 2.
  **Belum**: uji di venv Windows `Z:/GEMASTIK/Vignate-GEMASTIK/.venv` (paket
  pandera/feature-engine/skrub/autogluon belum dipasang di sana per 2026-11-02); R17 belum diukur di browser;
  penanda satuan campur + nol mencurigakan; tab Kosong/Encoding. Rincian di STATUS `RENCANA.md`.
- Laya v2 overhaul UMUM (2026-11-02): Temuan/baca/kamus umum, semua bisa diedit; uji 10 dataset di
  `_laya_eda/data/umum/` (diabaikan git). Semua uji: lulus 63, gagal 0, dilewati 2. Rincian STATUS E di `_laya_eda_v2/RENCANA.md`.
- T3 + T7 selesai (2026-11-06, belum di-commit): `arsenal/umum/baca_folder` bagian 12–16 (#14–#20) dan
  `arsenal/umum/simpan_muat` #15–#16 (validator submission vs sample), #17 (zip cadangan, bergeser dari #16),
  #18 (paket kumpul). Bukti dan blok PETA di STATUS T3/T7 `RENCANA_TEKS.md`; T13 yang menyalin ke `arsenal/PETA.md`.
- T11 a, b, d, e selesai (2026-11-06, belum di-commit): `teks/klasifikasi/tfidf` bagian 8-9 (NB-SVM), `teks/dasar/praproses`
  bagian 5 (train↔test), `tabular/ensembling` bagian 1b + 10 (F1 makro, rank average), `tabular/metrik` bagian 9 (temperature
  scaling + isotonic). **T11c (submission NER-BERT, GPU) belum.** Bukti dan blok PETA di STATUS T11 `RENCANA_TEKS.md`.
- `viginte-train.ipynb` dipindah dari root ke `_.Testing/` (diabaikan git).
- PPT vignette (penyisihan) selesai 2026-10-30. Figur + skripnya di `Z:/GEMASTIK/Vignate-2/outputs/figures/`
  (di luar repo, bukan git); catatan lengkap `Z:/GEMASTIK/Vignate-2/HANDOFF.md`.

## Pekerjaan terbuka

- **Jalur lomba siap pakai (2026-11-10 malam): `../final-pipeline/`** (di luar repo, bukan git, harus disalin
  sendiri ke PC panitia sejajar repo). Notebook `1_eda` .. `7_laporan_model` + `6b_topik_bertopic` = salinan
  `contoh/text` yang membaca `konfig.py`; `8_ekspor.bat`; `PETUNJUK_AKHIR.pdf`. Hasil ke `../outputs/final`.
  Run All berurutan lulus di SmSA (finetune hanya 1 epoch). Bangun ulang dari repo: `python _alat/buat_ulang.py`
  di folder itu (menimpa notebook). Belum diuji: metrik lain, banyak kolom teks, sample panitia nyata.

- Dokumentasi presentasi aman NDA (2026-11-10, di-push): `topik_lengkap`, `laporan_model`, saklar `AMAN_NDA` di
  `eda_lengkap`/`eda_presentasi`, `ekspor_hasil.bat` → `../outputs/bawa_pulang/` + MANIFEST. Semua lint 0 dan lulus
  Run All demo. Belum: gladi menyusun slide hanya dari `bawa_pulang/` (R5), `ekspor_hasil.bat` lewat cmd.
  Aturan baru: selalu commit + push; agen paling tinggi Opus (bawaan Sonnet high). Catatan serah terima lokal:
  `HANDOFF.md` (diabaikan git).

- Cek terakhir 2026-11-10 (rinci di log): model 48/48 terunduh (21 GB), venv = requirements (+ jiwer/kmodes/ruptures),
  PANDUAN/KATALOG sudah memuat semua notebook teks, semua di-push. Sisa: angka "9 GB / 18 model" di
  `download_model.bat`/`unduh_model.py`/README usang; `data_latihan` (junction ke `Z:/GEMASTIK-DATABASE`) harus disalin
  sendiri; wheelhouse tidak ada; tautan mati PANDUAN (`visualisasi/PILIH_GRAFIK.md`, `eda_otomatis.ipynb`); model
  `bge_m3`/`e5_large_instruct`/`qwen3_06b` tidak diunduh (hanya alternatif ber-`#`); T12 gladi belum.

- Perbaikan review jalur runbook (2026-11-10, di-push bersama `run_baseline.bat` untuk cek PC lomba): 12 notebook `contoh/text/` (penjaga sample demo,
  NUM_COLS ⊂ kolom test, opsi grup teks kembar, peringatan validator, path pseudo-label). Hari lomba:
  kosongkan `outputs/text/oof` di PC panitia sebelum mulai.
- NusaBERT-base (2026-11-10, belum di-commit): `pretrained/nusabert` + alternatif di finetune_oof; belum diuji skor.
- Qwen3-Embedding-0.6B (2026-11-09, belum di-commit): `_alat/unduh_model.py` + folder `pretrained/qwen3_emb`,
  alternatif ber-`#` di `contoh/text/klasifikasi_embedding.ipynb` (sentence-transformers, tanpa Ollama). SmSA
  (skrip uji, split sama): + instruksi `TEXT_PREFIX` + SVC C3 **0.8935** (tanpa instruksi 0.8656; MiniLM SVC
  0.8609); scaler+LogReg merusak qwen (0.80). Notebook belum Run All ulang; KATALOG belum diisi.
  `embedding_ollama.ipynb` dihapus (penghapusan ter-stage, file itu ada di commit 7533908).
  Asisten kode laptop (OpenCode Desktop + Ollama, terpasang 2026-11-09): model `qwen3.6-32k` (Modelfile dari
  `qwen3.6:35b-a3b-q4_K_M` + num_ctx 32768, karena app Ollama memaksa 16384), 35 tok/s, 77% CPU/23% GPU, tool call
  lewat /v1 lulus. Config `~/.config/opencode/opencode.json`. Ollama harus dibuka saat GPU aktif (bukan mode hemat).
- `contoh/text/eda_presentasi.ipynb` (2026-11-09, belum di-commit, lint 0): 10 grafik EDA → PNG+CSV +
  `fakta.md` di OUTPUT_FOLDER untuk bikin PPT tanpa data. Lolos SmSA/HS/teks bertanggal multilabel. Sisa: legenda
  `08_waktu` (12 kelas) menutupi garis. Sisa rencana (dijeda, tunggu pengguna): `laporan_model.ipynb`, hook
  autosave di `arsenal/umum/plot`, `_alat/ekspor_hasil.bat`, alt EM Saerens/lookup (ensemble_oof), R-Drop/AWP
  (finetune_oof), baris PANDUAN.
- Notebook teks+tabel baru (2026-11-09, belum di-commit, lint 0, sudah di PANDUAN): `stacking_teks_tabular`,
  `stacking_teks_tabular_regresi`, `pasca_prediksi`, `penjelasan_teks`, alt `VERBAL_COLS` di `finetune_oof`
  (belum diuji GPU). KATALOG belum diisi. Kandidat berikut: klasifikasi hierarkis (baju division > department >
  class), fuzzy matching/record linkage (rapidfuzz), multi-target. `klasifikasi_hierarkis` selesai 2026-11-09
  (lint 0, baju 1 menit, gabung datar+rantai 0,437 vs datar 0,426; di PANDUAN). 2026-11-10: + 9 notebook
  (`fuzzy_kategori`, `multi_target`, `ordinal_teks`, `regresi_hitungan`, `interval_prediksi`, `data_besar_teks`,
  `uji_beda_model`, `graf_label_teks`, `geser_distribusi`), lint 0, sekali jalan, di PANDUAN; belum di KATALOG,
  belum di-commit. 2026-11-10 (metode niche): + 7 notebook `bobot_ensemble_metaheuristik`,
  `pengklasifikasi_alternatif`, `seleksi_fitur_metaheuristik`, `co_training_teks`, `fitur_otomatis_tetangga`,
  `stacking_fwls`, `rulefit_aturan`; lint 0, sekali jalan, di PANDUAN + RENCANA_TEKS; belum di KATALOG, belum
  di-commit. Belum ada: AWP/R-Drop/distilasi (butuh GPU untuk uji). Panduan pakai ketujuhnya (kapan,
  data apa, hasil demo, urutan saat waktu sempit): `contoh/PANDUAN_METODE.md`, ditautkan dari PANDUAN. Versi cetak `PANDUAN_METODE.pdf` (+ `.tex`, isi
  harus disamakan manual kalau md berubah).
- Qwen3.6 lewat OpenCode dipakai pengguna sepanjang 5 jam lomba (perbaiki kode, dll.; pengguna 2026-11-09).
  Masalah: OpenCode memuat `AGENTS.md` repo otomatis (docs opencode, dicek), dan isinya menyuruh baca status +
  aturan (±28 KB) lalu menulis memori tiap tugas → buang konteks 32k dan waktu. Usulan (belum dikerjakan):
  file aturan lomba khusus Qwen lewat `instructions` opencode.json. Belum jelas Qwen jalan di laptop (tidak boleh
  diberi data) atau PC panitia (rebutan VRAM dengan training).

Teks (utama sejak 2026-11-06): butir T1–T13 di `RENCANA_TEKS.md` bagian 4, mulai T1 (lingkungan + uji Run All).
T8 selesai 2026-11-06 (`contoh/text/teks_tabular.ipynb`; sisa: `e5_base` belum diukur; uji muat di `ensemble_oof` sudah lulus 2026-11-07;
blok KATALOG/PANDUAN/SOURCES ada di STATUS T8 untuk T13).
T6 `contoh/text/ensemble_oof.ipynb` selesai sebagian 2026-11-07 (lint 0, LOLOS di smsa/emot/regresi/biner/OOF T8; sisa: uji dengan OOF
`indobert` dari T5 dan `finetune_oof` memuat `pseudo_label.csv`; blok KATALOG/PANDUAN/PETA di STATUS T6 untuk T13).
Daftar lama di bawah (vision-sentris) turun prioritas; validator (2), zip (3), log percobaan (8) masuk T3/T7/T6.

1. Deteksi objek sampai `submission.csv` + mAP lokal (pembaca YOLO generik, split val, format
   per-kotak/PredictionString, gambar tanpa deteksi tetap punya baris).
2. ~~Validator submission universal~~ selesai (T3, `simpan_muat#15–#16`).
3. Paket kumpul: zip selesai (T7, `simpan_muat#18`); potongan grafik → pptx belum.
4. Metrik mAP / mIoU / Dice / CER.
5. RLE segmentasi multi-kelas + resize balik.
6. Pembaca dataset Hugging Face parquet.
7. Deret waktu ke arsenal.
8. Log percobaan + latihan 5 jam pakai stopwatch.
- Tambahan: uji bagian 3–6 `pipeline/ocr_end_to_end_v2` (PaddleOCR sudah terpasang, butuh `yolo_2cls.pt`). Kalau pengguna setuju, hapus `pretrained/`/data dobel di `Z:/GEMASTIK/_arsip/`.
