# Riset: tipe data, tipe tugas, dan preprocessing untuk GEMASTIK Penambangan Data

Disusun 2026-10-10. Sumber utama: slide sosialisasi resmi GEMASTIK XVIII 2025, arsip soal/pemenang
tahun lalu, lomba sejenis di Indonesia, dan paper benchmark (tautan di tiap bagian dan di akhir).

---

## 1. Format lomba (yang menentukan cara kita bersiap)

Dari slide sosialisasi resmi 2024 dan 2025:

| Tahap | Isi | Implikasi |
|---|---|---|
| Penyisihan | Laporan teknis 4–5 halaman format IEEE, dataset **pilihan sendiri** dari internet sesuai tema. 2024: *"Penambangan Data untuk Peningkatan Pelayanan Publik menuju Masyarakat Indonesia yang Sejahtera"*. 2025: *"Penambangan Data untuk Peningkatan TIK menuju Kemandirian Bangsa"*. Isi wajib: judul, abstrak, pendahuluan, kajian terkait, solusi usulan (dataset, metode, beda dengan solusi sebelumnya, metrik), hasil eksperimen, analisis, kesimpulan. Dinilai: keaslian (cek plagiarisme), kebaruan dataset/metode, manfaat, kejelasan, kelengkapan. | Butuh kemampuan menulis + validasi yang jujur (lihat §4 kebocoran data). |
| Final (luring, 20 tim) | Set-up workstation di perangkat panitia → **studi kasus 5 jam (13.00–18.00)** membangun model dan evaluasi; boleh pakai tools/library/framework → presentasi hari berikutnya. | Notebook siap-pakai + kebiasaan kerja cepat lebih penting daripada model eksotis. Paket offline (pretrained) harus dibawa. |
| Bobot final | **Leaderboard 30%**, ketepatan metode 25%, presentasi 25%, nilai penyisihan 20%. | 50% nilai final dari *alasan* metode + presentasi → setiap langkah preprocessing harus bisa dijelaskan. |

Slide 2024 menyebut data bisa **terstruktur atau tidak terstruktur (teks, gambar, video, suara/speech)**.

## 2. Tipe data & tugas yang pernah / mungkin muncul

### 2.1 Riwayat soal GEMASTIK yang bisa dilacak

| Tahun | Soal final / kasus | Tipe |
|---|---|---|
| 2011 | Prediksi pelunasan kredit | Tabular, klasifikasi biner |
| GEMASTIK 5 (2012) | Klasifikasi teks paper ilmiah berbahasa Inggris (judul + abstrak), hampir tanpa label | Teks, klasifikasi / semi-unsupervised |
| 2019, 2022 | Final 5 jam, data latih hanya ~50% | Format leaderboard |
| XIV (2021) | Pemenang: deteksi depresi dari tweet | Teks Indonesia (sosmed) |
| 2023 | Log jaringan → prediksi **tipe serangan**, ada kelas langka | Tabular/log, multikelas imbalanced. Pemenang menekankan feature engineering + domain knowledge |
| **2024 (XVII)** | **Penggolongan UKT**: prediksi golongan UKT (8 kelas ordinal) dari data survei ekonomi mahasiswa yang berantakan. Detail di §2.1a | Tabular kotor + kolom teks bebas, multikelas imbalanced, F1 makro |
| 2024, 2025 | Juara: "Three Outliers" (UI), "Three Vectors" (UI); 2025 diikuti 258 tim | — |

### 2.1a Studi kasus final 2024 — presentasi juara 1 "Three Outliers" (UI)

Sumber: `ukt_classification_final_presentation.pptx` (slide presentasi tim juara).

- **Data**:
  - Train 30.380 baris, test 7.595 baris. Target **Golongan UKT** 0–7; golongan 1 sangat sedikit.
  - Fitur dari formulir survei:
    - Ekonomi: pendapatan, pengeluaran, hutang, cicilan, daya listrik.
    - Aset dan rumah: luas tanah/bangunan, NJOP, kendaraan, bahan atap, status rumah.
    - Keluarga dan akademik: pekerjaan/pendidikan orang tua, ayah/ibu hidup, tanggungan, riwayat beasiswa, prodi, kota, jalur masuk.
- **Format**: kompetisi gaya Kaggle. Leaderboard private ≈69% test. Skor 3 besar 0,48391 / 0,48062 / 0,47951,
  jadi selisihnya hanya ~0,003. Juara hanya submit 2 kali.
- **Langkah juara** (skor CV F1 makro):
  1. Bersihkan string tidak konsisten: `S- 3`/`S3`/`S-3` → `S3`. Lalu `-`/`Tidak`/`Tidak Ada`/`Tidak Dapat` → `Tidak`.
  2. Missing value: isi heuristik dari logika domain (NaN tanggungan anak → 0). Sisanya diisi non-heuristik / sentinel -1.
  3. Buang outlier di atas batas atas IQR.
  4. Kolom teks bebas dengan **3.266 nilai unik** diubah jadi fitur: n-gram kata (1–3) → ambil n-gram terbanyak → one-hot.
     Fitur "NLP" ini **paling penting** di semua model (importance ≈0,7 di XGBoost).
  5. Bandingkan baseline: CatBoost 0,467 > LGBM 0,463 > XGB 0,457 ≫ RF 0,416 > DT 0,381.
  6. Cost-sensitive learning, bobot kelas `n / (k · n_c)`: 0,467 → 0,472.
  7. Hill-climbing ensemble CatBoost+LGBM+XGB: 0,472 → 0,475.
  8. Rencana lanjutan: autocorrect + text embedding untuk kategori kotor.
- **Pelajaran**:
  - Yang menang adalah **pembersihan data + feature engineering**, bukan model eksotis.
  - Notebook harus siap untuk: normalisasi string kategori, teks bebas di dalam tabel, F1 makro multikelas, class weight, hill climbing.

### 2.2 Lomba data mining Indonesia sejenis (indikator gaya soal)

| Lomba | Kasus |
|---|---|
| Datavidia 2024 (Kaggle) | Peramalan harga pangan harian, 34 provinsi × 13 komoditas, horizon 92 hari, metrik MAPE |
| Datavidia 2020 | Cross-selling pesawat + hotel (tabular, imbalanced) |
| Datavidia (lain) | Regresi curah hujan |
| Satria Data 2021 | Gender/usia dari foto (vision) |
| Satria Data 2022 final | Sentimen BPJS + *dynamic topic modeling* (teks) |
| LOGIKA UI 2025 | Klasifikasi rumah adat Nusantara (vision) |

### 2.3 Benchmark NLP Indonesia (tugas teks yang paling mungkin)

IndoNLU, IndoNLG, dan NusaX memetakan tugas teks Indonesia standar:
- Klasifikasi: sentimen (SmSA), emosi (EmoT), aspek (CASA, HoASA, multilabel), entailment (WReTE).
- Sequence labeling: NER (NERGrit, NERP), POS, ekstraksi kata kunci.
- QA, ringkasan, terjemahan, sentimen bahasa daerah (NusaX).

### 2.4 Peta kemungkinan → notebook yang sudah ada

| Kemungkinan soal | Peluang | Notebook di `_.NEW-GEN` | Celah |
|---|---|---|---|
| Tabular klasifikasi/regresi (layanan publik, kredit, kesehatan) | Tinggi | `tabular/*` (19) | — |
| Multikelas imbalanced dari log/jaringan (pola 2023) | Tinggi | `tabular/imbalanced`, `tabular/multikelas` | Belum ada notebook khusus log/IDS (§3.6) |
| Teks Indonesia (sentimen, emosi, hoaks, aspek) | Tinggi | `text/*` (9) | Normalisasi slang belum ada (§3.3) |
| Peramalan deret waktu panel (harga, cuaca) | Sedang–tinggi | `timeseries/peramalan_*` | Belum ada LightGBM global multi-seri (§3.5) |
| Gambar (klasifikasi; deteksi lebih jarang) | Sedang | `vision/*` (10) | — |
| Audio / speech | Rendah–sedang | `audio/*` (5) | — |
| **Video** (disebut eksplisit di slide 2024) | Rendah–sedang | **tidak ada** | Perlu `video/klasifikasi` (§3.8) |
| Clustering/segmentasi tanpa label, topik | Sedang | `unsupervised/*`, `text/topik`, `text/clustering` | Dynamic topic modeling belum ada |
| Graf, recsys, multimodal | Rendah | `graph/*`, `recsys/*`, `multimodal/*` | — |

---

## 3. Preprocessing yang lebih baik (bukti dari paper)

Prinsip umum: **semua langkah yang "belajar" dari data (imputer, scaler, encoder, seleksi fitur,
oversampling) harus di dalam `Pipeline` dan di-fit hanya di fold latih.** Ini sumber kebocoran
nomor satu (Kapoor & Narayanan 2023).

### 3.1 Tabular — pilih preprocessing sesuai model

| Langkah | Model pohon (LightGBM/XGB/CatBoost/RF) | Model linear / SVM / KNN / MLP | Bukti |
|---|---|---|---|
| Scaling | **Tidak perlu** (invarian skala) | Wajib. MLP: `QuantileTransformer(output_distribution="normal")` atau robust scaling + clipping | Scaling study 2025; RealMLP 2024. Terbukti di `tabular/deep_regresi`: R² −322 → 0.91 setelah ganti ke Quantile |
| Missing value | Biarkan NaN (LightGBM/XGB menanganinya), opsional tambah kolom indikator | Imputasi. Default terbaik: **random forest / KNN imputer**; mean/mode cukup untuk kategori dengan missing MNAR berat | Jäger et al. 2021 |
| Kategori kardinalitas rendah | Ordinal/native categorical | One-hot | — |
| Kategori kardinalitas tinggi (ID, kota, kode) | **Target encoding teregularisasi, dihitung out-of-fold** (`sklearn.preprocessing.TargetEncoder` sudah cross-fit otomatis) atau CatBoost | Sama | Pargent et al. 2022 |
| Fitur tidak informatif | Pohon relatif tahan | MLP sangat terganggu → seleksi fitur | Grinsztajn et al. 2022 |
| Outlier target regresi | Latih di `log1p(y)` bila skewed | Sama | Praktik umum |

Pilihan model awal (TabArena 2025, TabPFN 2025):
- **CatBoost default** adalah model terkuat tanpa tuning → baseline pertama yang cepat.
- Data kecil (≤10K baris, ≤500 fitur): **TabPFN** sering menang tanpa tuning (TabPFN-2.5: ≤50K baris). Belum terpasang di `.venv`; perlu diunduh beserta bobot sebelum final.
- Dengan waktu tuning: LightGBM/RealMLP/TabM + ensembling. Untuk 5 jam, cukup GBDT + OOF stacking/rata-rata.

### 3.2 Imbalanced

- **SMOTE hanya membantu model lemah.** Untuk model kuat (GBDT), balancing tidak menaikkan skor;
  random oversampling ≈ SMOTE (Elor & Averbuch-Elor 2022). SMOTE juga merusak kalibrasi probabilitas.
- Yang efektif: **atur threshold** di data validasi (sudah ada di `tabular/imbalanced`: F1 0.62 → 0.66
  di threshold 0.4), `class_weight="balanced"` / `scale_pos_weight`, dan metrik yang benar
  (F1 macro, PR-AUC).
- Multikelas dengan kelas sangat langka: gabungkan stratified CV + class weight; cek confusion matrix per kelas.

### 3.3 Teks Indonesia

- **Normalisasi slang/kata tidak baku adalah langkah paling berpengaruh** untuk IndoBERT
  (+0.061 macro F1 menurut Budianoor et al., JCTA). Pakai kamus seperti *colloquial-indonesian-lexicon*
  (Salsabila et al. 2018) → `map` kata per kata. Belum ada di notebook teks.
- **Stemming (Sastrawi) dan hapus stopword tidak terbukti membantu model subword/transformer**, dan bisa
  merugikan (membuang negasi "tidak", "bukan"). Untuk TF-IDF masih boleh dicoba sebagai variasi.
- Bersihkan: lowercase, URL→token, mention→token, emoji→hapus atau ubah ke kata, huruf berulang
  ("bangetttt"→"banget").
- Urutan model: TF-IDF (char + word n-gram) + LogReg → embedding (sentence-transformers) + LogReg →
  fine-tune IndoBERT bila waktu cukup.

### 3.4 Label berisik (semua modalitas)

- Test set benchmark populer pun punya ≥3.3% label salah (Northcutt et al. 2021).
- **Confident learning (cleanlab)**: dari probabilitas OOF, tandai sampel yang labelnya kemungkinan
  salah → periksa / buang → latih ulang. Sangat relevan untuk data sosmed hasil anotasi.
  Paket `cleanlab` belum terpasang.

### 3.5 Deret waktu

Pelajaran M5 (Makridakis et al. 2022), mirip soal Datavidia harga pangan:
- **Satu LightGBM global untuk semua seri** (cross-learning) mengalahkan model statistik per seri.
- Fitur: lag, rolling mean/std (digeser agar tidak bocor), kalender (hari, minggu, libur nasional),
  harga, ID seri sebagai kategori.
- Gabungkan strategi recursive dan direct; validasi dengan **split berdasarkan waktu** (time-based
  holdout / `TimeSeriesSplit`), tidak pernah acak.
- Target skewed → log; metrik MAPE → hati-hati nilai mendekati nol.

### 3.6 Log jaringan / intrusion detection (pola soal 2023)

Engelen et al. 2021 (analisis CICIDS2017): lebih dari 25% flow adalah artefak dan ada label salah.
- **Buang fitur jalan pintas**: IP, MAC, port sumber, timestamp, flow ID. Fitur ini membuat skor
  validasi sangat tinggi tapi tidak menggeneralisasi.
- Cek duplikat antar train/test; split berdasarkan waktu bila ada timestamp.
- Feature engineering domain: rasio byte/paket, durasi, flag TCP, agregasi per host per jendela waktu.

### 3.7 Gambar

- Data kecil: **embedding pretrained beku + LogReg** dulu (`vision/klasifikasi_embedding` 0.887 >
  CNN fine-tune 0.674 di data demo).
- Augmentasi saat fine-tune: **TrivialAugment** (`torchvision.transforms.TrivialAugmentWide`), tanpa
  tuning dan setara/lebih baik dari RandAugment. Untuk few-shot di fitur beku, satu augmentasi yang baik
  cukup (FroFA 2024).
- Normalisasi pakai mean/std ImageNet bawaan bobot pretrained; resize sesuai ukuran input model.

### 3.8 Video

Baseline kuat untuk data kecil (CLIP4Clip, CLIP-Hitchhiker, EVL): **ambil N frame seragam (8–16) →
CLIP beku → rata-rata embedding → LogReg**. Tidak perlu model video khusus. Belum ada notebooknya.

### 3.9 Audio

- Resample ke laju yang diharapkan model (16 kHz untuk wav2vec2/Whisper/BEATs), mono, potong/pad ke durasi tetap.
- Model dari nol: log-mel + **SpecAugment** (masking waktu & frekuensi).
- Embedding: **BEATs** > PANNs (AudioSet mAP 50.6 vs 43.9; ESC-50 98.1%); wav2vec2 sudah dipakai
  (`audio/klasifikasi_embedding` 0.875). BEATs memakai normalisasi fbank sendiri.

---

## 4. Checklist kebocoran data (wajib sebelum submit dan saat menulis laporan)

Delapan tipe kebocoran menurut Kapoor & Narayanan (2023):

1. [ ] Preprocessing (scaler/imputer/encoder) di-fit di train+test → pindahkan ke `Pipeline`.
2. [ ] Seleksi fitur dilakukan sebelum split → lakukan di dalam CV.
3. [ ] Oversampling sebelum split → `imblearn.pipeline` atau jangan pakai.
4. [ ] Duplikat baris muncul di train dan validasi/test.
5. [ ] Fitur yang tidak akan tersedia saat prediksi (fitur "masa depan", ID, kolom turunan target).
6. [ ] Kebocoran temporal: data masa depan dipakai melatih untuk memprediksi masa lalu.
7. [ ] Non-independensi: baris dari entitas yang sama (pasien, user, host) ada di train dan validasi → `GroupKFold`.
8. [ ] Bias sampling: distribusi validasi tidak mirip test → cek **adversarial validation**
       (sudah ada di `tabular/validasi`; AUC ≈ 0.5 berarti aman).

---

## 5. Alur kerja 5 jam final (usulan)

| Waktu | Kegiatan |
|---|---|
| 0:00–0:30 | Baca soal, metrik, format submission. EDA: ukuran, target, missing, duplikat, kolom ID/waktu. Tentukan skema validasi (stratified / group / time). |
| 0:30–1:00 | Buka notebook `_.NEW-GEN` yang sesuai, ganti CONFIG, jalankan baseline → **submit pertama** (jaga-jaga). |
| 1:00–3:00 | Preprocessing + feature engineering (bagian 3), uji satu perubahan per eksperimen, catat skor CV di tabel. |
| 3:00–4:00 | Model kedua yang berbeda jenis + rata-rata/stacking OOF; threshold tuning bila klasifikasi. |
| 4:00–4:30 | Latih ulang di semua data, submission final, cek format. |
| 4:30–5:00 | Simpan plot/tabel untuk presentasi (importance/SHAP, confusion matrix, tabel eksperimen). |

Kebiasaan dari *Kaggle Grandmasters Playbook* (NVIDIA): eksperimen cepat (GPU), validasi yang
dipercaya, OOF stacking, *hill climbing* ensemble, pseudo-label bila test besar.

---

## 6. Sumber

**Resmi / lomba**
- Slide sosialisasi GEMASTIK XVIII 2025 (Penambangan Data); teks hasil ekstraksi ada di `_alat/sosialisasi2025.txt`.

**Tabular**
- Grinsztajn, Oyallon, Varoquaux (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS. https://arxiv.org/abs/2207.08815
- Erickson et al. (2025). *TabArena: A Living Benchmark for Machine Learning on Tabular Data.* NeurIPS. https://arxiv.org/abs/2506.16791
- Holzmüller, Grinsztajn, Steinwart (2024). *Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data* (RealMLP). https://arxiv.org/abs/2407.04491
- Hollmann et al. (2025). *Accurate predictions on small data with a tabular foundation model* (TabPFN). Nature. https://www.nature.com/articles/s41586-024-08328-6
- *Feature scaling study* (2025). https://arxiv.org/abs/2506.08274
- Pargent et al. (2022). *Regularized target encoding outperforms traditional methods…* https://arxiv.org/abs/2104.00629
- Elor & Averbuch-Elor (2022). *To SMOTE, or not to SMOTE?* https://arxiv.org/abs/2201.08528
- Jäger, Allhorn, Bießmann (2021). *A Benchmark for Data Imputation Methods.* Frontiers in Big Data. https://pmc.ncbi.nlm.nih.gov/articles/PMC8297389/
- NVIDIA. *The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data.* https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/

**Validasi & kualitas data**
- Kapoor & Narayanan (2023). *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns. https://arxiv.org/abs/2207.07048
- Northcutt, Athalye, Mueller (2021). *Pervasive Label Errors in Test Sets…* https://arxiv.org/abs/2103.14749
- Northcutt, Jiang, Chuang (2021). *Confident Learning.* https://arxiv.org/abs/1911.00068

**Deret waktu & log**
- Makridakis, Spiliotis, Assimakopoulos (2022). *M5 accuracy competition: Results, findings, and conclusions.* IJF. https://doi.org/10.1016/j.ijforecast.2021.11.013
- Engelen, Rimmer, Joosen (2021). *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study.* IEEE S&P Workshops (WTMC).

**Teks Indonesia**
- Wilie et al. (2020). *IndoNLU.* https://arxiv.org/abs/2009.05387
- Cahyawijaya et al. (2021). *IndoNLG.* https://arxiv.org/abs/2104.08200
- Winata et al. (2023). *NusaX.* https://arxiv.org/abs/2205.15960
- Budianoor et al. *Studi preprocessing teks untuk IndoBERT* (JCTA) — normalisasi slang paling berpengaruh.
- Salsabila et al. (2018). *Colloquial Indonesian Lexicon* (kamus slang).

**Gambar, video, audio**
- Müller & Hutter (2021). *TrivialAugment.* ICCV. https://arxiv.org/abs/2103.10158
- Bär et al. (2024). *FroFA: Frozen Feature Augmentation for Few-Shot Image Classification.* CVPR. https://arxiv.org/abs/2403.10519
- Luo et al. (2021). *CLIP4Clip.* https://arxiv.org/abs/2104.08860
- Bain et al. (2022). *A CLIP-Hitchhiker's Guide to Long Video Retrieval.* https://arxiv.org/abs/2205.08508
- Lin et al. (2022). *Frozen CLIP Models are Efficient Video Learners* (EVL). https://arxiv.org/abs/2208.03550
- Park et al. (2019). *SpecAugment.* https://arxiv.org/abs/1904.08779
- Chen et al. (2022). *BEATs: Audio Pre-Training with Acoustic Tokenizers.* https://arxiv.org/abs/2212.09058
