# PANDUAN — buka notebook mana, kapan

Baca ini di menit pertama final. Detail riset: [RISET.md](RISET.md).

## 0. Sebelum mulai (sekali saja)

1. Salin folder proyek ke komputer lomba. Pakai path pendek (misalnya `C:/GEMASTIK/`), karena
   path panjang bisa membuat pemasangan library gagal ("Long Path").
   - Dobel klik `_.NEW-GEN/setup.bat` (sekali saja). File ini membuat `.venv` lalu memasang semua
     library dari `requirements.txt`. Kalau `env/wheelhouse` ikut disalin, pemasangan tanpa internet.
   - `jupyter.bat`: buka JupyterLab. `aktifkan.bat`: buka terminal dengan `.venv` aktif.
   - Belum punya folder `pretrained`? Dobel klik `download_model.bat` (sekali, sekitar 9 GB,
     butuh internet). Model masuk ke folder `pretrained` di sebelah folder `_.NEW-GEN`.
2. Path di notebook **relatif** terhadap folder notebook, jadi langsung jalan di laptop mana pun
   asalkan susunan foldernya begini:
   ```
   <folder apa saja>/_.NEW-GEN/      ← repo ini
   <folder apa saja>/pretrained/     ← hasil download_model.bat
   <folder apa saja>/data_latihan/   ← data demo (opsional)
   ```
   Saat lomba, ganti baris pertama sel Konfigurasi ke folder data soal:
   ```python
   DATA_FOLDER = "C:/GEMASTIK/soal1"   # GANTI saat lomba (garis miring /, bukan \)
   ```
3. Ganti nama file dan nama kolom (huruf besar semua) di sel Konfigurasi.
4. Hapus sel terakhir `## Cek dengan kunci jawaban (HAPUS saat lomba)`. Kunci jawaban hanya ada
   di data latihan.
5. Hasil selalu ditulis ke `submission.csv` di folder yang sama dengan notebook.

Semua notebook urutannya sama: Import → Konfigurasi → Baca → EDA → Split → Model → Evaluasi →
Latih ulang + submission. Ganti model = pindahkan `#` di sel Model.
Semua alternatif yang di-comment sudah diuji jalan satu per satu (324 varian). Daftar lengkap
metode + skor demo per notebook: [KATALOG.md](KATALOG.md). Tanda ⭐ = lebih baik dari model aktif.

## 1. Peta cepat: tanda di soal → notebook

| Tanda di soal / data | Buka (urut) | Validasi |
|---|---|---|
| CSV kuesioner/survei, kategori kotor, kolom teks bebas, multikelas, F1 macro | **`tabular/survei_kotor`** (jalankan 2–3 model) → **`tabular/blending`** | StratifiedKFold 5 (OOF) |
| CSV, target 2 kelas | `tabular/klasifikasi` → `tuning_klasifikasi` → `ensembling_klasifikasi` | Stratified |
| CSV, target > 2 kelas, fitur bersih | `tabular/multikelas`, atau `survei_kotor` (sudah ada class weight + OOF) | Stratified |
| CSV, kelas positif sangat langka | `tabular/imbalanced` (class weight + threshold) | Stratified |
| CSV, target angka | `tabular/regresi` → `regresi_lanjutan` (log target, interval) → `tuning_regresi` | KFold |
| Tiap baris punya banyak label | `tabular/multilabel` / `text/multilabel` | KFold |
| Durasi sampai kejadian + sensor | `tabular/survival` | — |
| Ada entitas berulang (user, pasien, sekolah, speaker) | notebook tabular + `GroupKFold` (lihat `tabular/validasi`) | GroupKFold |
| Train dan test terasa beda | `tabular/validasi` (adversarial validation) | — |
| Perlu menjelaskan model ke juri | `tabular/explain_klasifikasi` / `explain_regresi` (SHAP, PDP) | — |
| Teks + label | `text/klasifikasi_tfidf` → `klasifikasi_embedding` → `klasifikasi_finetune` | Stratified |
| Teks: entitas | `text/ner` | — |
| Teks tanpa label | `text/clustering`, `text/topik`, `text/kata_kunci_ringkasan` | Baca manual |
| Cari dokumen mirip | `text/retrieval` | — |
| Satu deret waktu, ramal ke depan | `timeseries/peramalan_statistik` → `peramalan_ml` | Split waktu |
| Deret waktu + variabel penjelas | `timeseries/peramalan_exog` (pastikan eksogen masa depan ada!) | Split waktu |
| Deret waktu: klasifikasi / anomali / cek pola | `timeseries/klasifikasi` / `anomali` / `diagnostik` | — |
| Gambar per kelas | `vision/klasifikasi_embedding` → `klasifikasi_clip` → `klasifikasi_cnn` | Stratified |
| Gambar: kotak objek / mask / teks | `vision/deteksi` / `segmentasi` / `ocr` | — |
| Audio | `audio/klasifikasi_embedding` → `klasifikasi_mfcc` / `klasifikasi_cnn`; ucapan → `audio/asr` | Group per speaker |
| Graf / jaringan | `graph/klasifikasi_node`, `prediksi_link`, `komunitas`, `analisis` | — |
| Rating / rekomendasi | `recsys/prediksi_rating`, `recsys/top_n` | — |
| Tanpa label: kelompokkan / anomali / pola belanja | `unsupervised/clustering` / `anomali` / `aturan_asosiasi` | Silhouette |
| Tabel + kolom teks | `multimodal/klasifikasi` / `regresi`, atau `survei_kotor` (n-gram teks bebas) | — |
| Beberapa tabel (induk + transaksi/riwayat), kunci id | `multitabel/agregasi_klasifikasi`; ada waktu → `multitabel/agregasi_waktu` | Stratified / split waktu |
| Dua tabel, cari baris yang sama (nama/alamat beda tulis) | `multitabel/entity_resolution` | — |
| Banyak deret sekaligus (harga x provinsi) | `timeseries/peramalan_global`, bandingkan `timeseries/peramalan_chronos` (zero-shot) | Split waktu |
| Video: hitung/lacak objek, garis, zona | `video/tracking`; kejadian aneh → `video/anomali` | — |
| GPS / lintasan | `trajektori/fitur_gps` (moda), `trajektori/clustering` (tempat sering dikunjungi) | GroupKFold per pengguna |
| Sensor/akselerometer per jendela waktu | `sinyal/klasifikasi_jendela` | GroupKFold per subjek |
| Citra medis (rontgen, MNIST medis) | `medis/klasifikasi_citra` (pilih ambang sensitivitas) | Stratified |
| Klip video + suara | `multimodal/audio_video` | Group per aktor |
| File PDF (laporan, faktur, formulir), PDF scan | `vision/dokumen_pdf` (teks, key-value, tabel, QR, deskew + OCR) | — |
| Objek mudah dipisah warna/terang, tanpa label | `vision/citra_klasik` (HSV, morfologi, connected components) | — |
| Keputusan terbaik di bawah batasan (produksi, lokasi, anggaran) | `tabular/optimasi` (LP, MIP, Monte Carlo) | — |
| Teks tanpa label, topik bermakna | `text/topik_bertopic` (lebih baik dari LDA di demo) | Baca manual |
| Citra satelit/udara: kotak objek, nilai angka per petak, perubahan dua waktu | `satelit/deteksi_objek`, `satelit/regresi`, `satelit/deteksi_perubahan` | Split per wilayah |
| Data per wilayah (peta, koordinat, poligon) | `geo/klasifikasi_wilayah` (fitur tetangga); titik panas → `geo/hotspot` (Moran, LISA) | Split per wilayah |
| Log aplikasi/server per sesi | `log/klasifikasi_sesi`; tanpa label → `log/anomali_log` | Stratified per sesi |
| Graf: simpul aneh / target angka per simpul | `graph/anomali`, `graph/regresi_node` | — |
| Suara mesin/alat tidak normal | `audio/anomali` (latih dari suara normal saja) | — |
| Gambar, kelas berurutan (tingkat, grade, kerusakan) | `vision/klasifikasi_ordinal` (regresi + ambang, QWK) | Stratified |
| Dua gambar: sama/mirip atau tidak | `vision/kemiripan_pasangan`; wajah → `vision/wajah` (YuNet + SFace) | — |
| Gambar tanpa label, kelompokkan | `vision/clustering` (CLIP → UMAP → KMeans/HDBSCAN) | Baca manual |
| Pertanyaan tentang isi gambar | `vision/vqa` (ViLT, pertanyaan bahasa Inggris) | — |
| Kenali dataset gambar, gambar kembar train↔test | `vision/statistik_dataset` (imagehash) | — |
| Plat nomor kendaraan | `vision/plat_nomor` (YOLO plat + EasyOCR + koreksi format) | — |
| Produk cacat (inspeksi), sedikit/tanpa contoh cacat | `vision/defect` (PatchCore dari gambar normal, peta panas) | — |
| Teks aneh/spam/di luar topik tanpa label | `text/anomali_teks` | — |
| Ringkas / parafrase teks, nilai ROUGE/BLEU | `text/generasi` (Qwen 0.5B lokal; bandingkan Lead-3) | — |
| Ujaran kebencian / kata kasar | `text/ujaran_kebencian` (multilabel, kamus alay, ambang per label) | Holdout stratified (label HS) |
| Transaksi kartu/e-wallet, deteksi penipuan | `tabular/fraud_transaksi` (fitur kecepatan dari masa lalu) | Split waktu |

### Label tidak ideal

| Situasi | Buka |
|---|---|
| Belum ada label, harus melabeli sendiri | `vision/anotasi_klasifikasi` / `anotasi_deteksi`, `text/anotasi_teks`; pilih data yang dilabeli → `tabular/active_learning` |
| Label dari beberapa anotator | `tabular/kesepakatan_anotator` (kappa, voting) |
| Label dari aturan / kata kunci | `text/weak_supervision` |
| Curiga banyak label salah | `text/label_berisik`, `vision/label_berisik`, `tabular/label_noise` (cleanlab) |
| Hanya sebagian kecil berlabel | `tabular/semi_supervised`, `text/semi_supervised_teks`, `vision/semi_supervised`, `audio/semi_supervised` |
| Hanya contoh positif yang diketahui | `tabular/pu_learning` (Elkan-Noto) |
| Test punya kelas yang tidak ada di train ("lainnya") | `vision/kelas_baru` (CLIP + ambang) |

### Setelah model jadi (naikkan skor, bahan presentasi)

| Tujuan | Buka |
|---|---|
| Tuning | `tabular/tuning_*`, `text/tuning` (TF-IDF + C) |
| Gabung model | `tabular/ensembling_*`, `tabular/blending`, `text/ensemble`, `vision/tta_ensemble` |
| Di mana model salah? | `tabular/analisis_kesalahan`, `text/analisis_kesalahan`, `vision/analisis_kesalahan`, `timeseries/analisis_kesalahan` |
| Grafik untuk slide | `visualisasi/` (16 notebook); pilih jenis grafik lewat [visualisasi/PILIH_GRAFIK.md](visualisasi/PILIH_GRAFIK.md) |

### Data kotor → `preprocessing/` (sebelum notebook model)

| Isi kolom | Buka | Inti |
|---|---|---|
| Kode/nomor/email/alamat, perlu ambil sebagian | `preprocessing/regex_ekstraksi` | grup bernama `(?P<nama>...)` + `str.extract` + laporan baris tidak cocok |
| Uang `1,5 jt`, `3.700rb`, `2-3 juta`; berat `70.500 gr`, `0,7 kwintal`; tanggal `12 Agu 2024`, `45516`; lama `2 thn 3 bln` | `preprocessing/angka_satuan_tanggal` | satu pola angka+satuan, tabel `SKALA_*`, format tanggal dicoba berurutan |
| Kategori typo/singkatan (`Tehnik Informatika`, `T. Informatika`, `TI`) | `preprocessing/kategori_fuzzy` | nilai unik → normalisasi → `difflib` → draf CSV → `KOREKSI` |
| Beda kata sama makna (`ojol`, `sopir`, `pns gol iii`) | `preprocessing/peta_semantik` | Laya + MiniLM dua juri; setuju diterima, sisanya manual |
| Tweet/ulasan Indonesia | `preprocessing/teks_indonesia` | regex, kamus alay 4.330 kata, stopword tanpa membuang `tidak`, stemming; skor per tahap |

Urutan hemat waktu: regex/angka dulu, lalu fuzzy, lalu semantik (hanya untuk sisa yang maknanya
beda). Pembersihan per baris tidak memakai target, jadi aman dilakukan sebelum split. Terapkan
fungsi dan peta yang **sama** ke train dan test.

Log jaringan/IDS → `tabular/log_serangan`.

## 2. Metrik → yang dilakukan

| Metrik | Lakukan |
|---|---|
| F1 macro (multikelas) | Class weight `balanced` + penskalaan probabilitas per kelas di OOF (`survei_kotor`) |
| F1 kelas positif (biner) | Class weight + cari threshold (`tabular/imbalanced`) |
| ROC-AUC / PR-AUC | Submit **probabilitas** (`predict_proba`), bukan label |
| Log loss | Jangan SMOTE / class weight ekstrem; submit probabilitas |
| Akurasi | Argmax biasa, tanpa class weight |
| RMSE / RMSLE | Target miring → latih di `log1p(y)`, balik dengan `expm1` (`regresi_lanjutan`) |
| MAE / MAPE | Objective MAE; untuk MAPE hati-hati nilai ≈ 0 |
| mAP / mIoU | `vision/deteksi` / `vision/segmentasi` |
| WER / CER | `audio/asr` (fungsi jarak edit sudah ada) |

## 3. Validasi — pilih satu

1. Ada kolom waktu dan test lebih baru → split waktu (`TimeSeriesSplit`).
2. Ada entitas berulang → `GroupKFold` / `StratifiedGroupKFold`.
3. Selain itu → `StratifiedKFold(5)` untuk klasifikasi, `KFold(5)` untuk regresi.
4. Pilih submission final berdasarkan **skor CV/OOF**, bukan public leaderboard.
   Public LB kecil bisa meleset ±0.01, sementara juara 1–2 tahun 2024 hanya beda 0.003.

## 4. Checklist 30 menit pertama

1. [ ] Baca soal: metrik **persis**, format submission (nama kolom, label atau probabilitas), batas submit.
2. [ ] `shape`, tipe kolom, jumlah nilai unik, % missing (sel profil di `survei_kotor`).
3. [ ] Target: distribusi, kelas langka, ordinal atau tidak.
4. [ ] Kolom ID / waktu / entitas → pilih validasi (§3).
5. [ ] Kebocoran: kolom yang sama dengan target, kolom "masa depan", duplikat train↔test (RISET §4).
6. [ ] Kategori kotor / teks bebas → `value_counts().head(30)` per kolom, isi `PETA`
   (lihat tabel `preprocessing/` di §1).
7. [ ] Jalankan baseline → **submit pertama sebelum menit 60**.
8. [ ] Cek `outputs/new-gen/eksperimen.csv` (skor OOF tiap jalan `survei_kotor` tercatat otomatis).

Alur 5 jam lengkap: [RISET.md §5](RISET.md).

## 5. Kesalahan fatal

- Preprocessing di-fit pada train+test (selalu taruh di dalam `Pipeline`).
- SMOTE di luar pipeline sebelum CV.
- Memilih threshold/bobot di data test.
- Lupa mengganti `DATA_FOLDER` / `PRETRAINED` → membaca data latihan, bukan data soal.
- Submit label padahal metriknya AUC/log loss.
