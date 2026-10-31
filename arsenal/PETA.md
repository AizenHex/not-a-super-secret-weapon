# PETA ARSENAL

Cara pakai: cari soalnya di tabel, buka file potongannya, salin sel yang perlu ke notebook kosong,
lalu ubah baris yang bertanda `# UBAH`. Tiap sel punya import sendiri.

Butuh baseline utuh sekali jalan (Run All)? Lihat notebook kasus di
[`../contoh/PANDUAN.md`](../contoh/PANDUAN.md). Sintaks Python yang lupa: [`../kamus/`](../kamus/).

## Aturan menempel (berlaku di semua file)

- Tiap file diawali **Butuh / Menghasilkan** dan tabel **Cara menempel**: urutan sel, mana yang
  wajib, mana yang opsional. Kalau muncul `NameError`, sel yang membuat variabel itu belum
  dijalankan: lihat tabel itu.
- **Bagian 0 = data contoh** (`DEMO_FOLDER = "data_latihan"`). Saat lomba lewati, ganti dengan
  baca data soal. Sel terakhir "Cek ke kunci jawaban (hanya demo)" juga dilewati.
- Sel "Fungsi ..." hanya **membuat** fungsi dan mencobanya pada satu contoh. Pekerjaan untuk
  semua data ada di sel "... SEMUA + simpan". Jangan lupa sel itu.
- `PRETRAINED_FOLDER = "pretrained"`: ubah ke folder hasil `download_model.bat`.
- Hasil tiap tahap disimpan ke `outputs/`. Kernel mati? Jangan ulang dari awal, baca file-nya:

| tahap | file | baca ulang |
|---|---|---|
| vektor gambar/teks | `outputs/train_vectors.npy`, `test_vectors.npy` | `np.load(...)` |
| latih YOLO | `outputs/yolo_runs/latih/weights/best.pt` | `BEST_WEIGHTS` di sel Konfigurasi `deteksi_yolo` |
| kotak terbaik | `outputs/best_boxes.csv` | `best_boxes = pd.read_csv("outputs/best_boxes.csv")` |
| potong + perbaiki + OCR | `outputs/crops.csv` | `crops = pd.read_csv("outputs/crops.csv")` |
| proba OOF model tabular | `outputs/oof_*.npy` | untuk `tabular/ensembling` |

## Umum (semua modalitas): `umum/`

Bagian bernomor; tiap file diawali tabel "ingin → bagian". Bagian 0 = data contoh, lewati.

| ingin | file | bagian |
|---|---|---|
| gambar per folder kelas → tabel path, label | `umum/baca_folder.ipynb` | 1 |
| folder gambar + CSV label, cek file hilang | `umum/baca_folder.ipynb` | 2 |
| banyak CSV / txt / JSON / JSONL / Excel → satu tabel | `umum/baca_folder.ipynb` | 3–6 |
| ambil info dari nama file (regex) | `umum/baca_folder.ipynb` | 7 |
| gambar → array `X` (n, SIZE, SIZE, 3) + `y` | `umum/baca_folder.ipynb` | 8 |
| CSV sangat besar, zip, PDF | `umum/baca_folder.ipynb` | 9–11 |
| laporan cepat: tipe, kosong, unik | `umum/tabel_bersih.ipynb` | 1 |
| nama kolom, `-`/`?` → NaN, `Rp 1.250.000` → angka | `umum/tabel_bersih.ipynb` | 2–4 |
| tanggal format campur, kategori tidak konsisten, ya/tidak | `umum/tabel_bersih.ipynb` | 5–7 |
| baris dobel, isi kosong, outlier | `umum/tabel_bersih.ipynb` | 8–10 |
| filter, urut, N teratas per kelompok | `umum/tabel_olah.ipynb` | 1–2 |
| `groupby` ringkasan / `transform` / pivot / melt | `umum/tabel_olah.ipynb` | 3–6 |
| gabung tabel, fitur dari tabel riwayat | `umum/tabel_olah.ipynb` | 7–8 |
| fitur waktu: lag, rolling, selisih hari | `umum/tabel_olah.ipynb` | 9 |
| angka → rentang (`cut` / `qcut`) | `umum/tabel_olah.ipynb` | 10 |
| contekan regex, bersihkan teks, kata gaul | `umum/teks_regex.ipynb` | 1–2 |
| ambil field dari teks (kode, tanggal, email, HP) | `umum/teks_regex.ipynb` | 3 |
| `1,5 juta` / `3 Okt 2026` → angka / tanggal | `umum/teks_regex.ipynb` | 6 |
| cocokkan nama mirip ke daftar baku (fuzzy) | `umum/teks_regex.ipynb` | 7 |
| histogram, hitungan kelas, scatter, waktu, boxplot | `umum/plot.ipynb` | 1–5 |
| heatmap korelasi, confusion matrix, grid gambar, kurva loss | `umum/plot.ipynb` | 6–9 |
| simpan/muat tabel, numpy, sklearn, PyTorch, JSON, pickle | `umum/simpan_muat.ipynb` | 1–6 |
| **file jawaban** + cek dengan `sample_submission.csv` | `umum/simpan_muat.ipynb` | 7 |
| cadangkan `outputs/` jadi zip | `umum/simpan_muat.ipynb` | 8 |

## Vision: `vision/<tugas>/`

| soal / kendala | file | sel |
|---|---|---|
| buka gambar aman (GIF, PNG transparan, foto HP miring EXIF) | `vision/dasar/gambar_dasar` | 1 |
| cek ukuran/mode semua gambar, padding persegi | `vision/dasar/gambar_dasar` | 2–3 |
| augmentasi, `Dataset` + `DataLoader` PyTorch | `vision/dasar/gambar_dasar` | 4–5 |
| **gambar per kelas, baseline tercepat** (vektor + LogReg) | `vision/klasifikasi/embedding` | 1–5 |
| akurasi lebih tinggi, ada GPU (fine-tune CNN) | `vision/klasifikasi/finetune` | 1–7 |
| tidak ada data latih / kelas baru (CLIP zero-shot) | `vision/klasifikasi/zero_shot_clip` | 1–5 |
| satu gambar banyak label, ambang per label | `vision/multilabel/multilabel` | 1–6 |
| gambar tanpa label → kelompok + label awal | `vision/pengelompokan/pengelompokan` | 1–5 |
| cari gambar termirip, pasangan sama/tidak, gambar kembar | `vision/kemiripan/kemiripan` | 1–4 |
| cacat produk, latih dari gambar normal saja (PatchCore) | `vision/anomali/anomali` | 1–7 |
| tebak kelas tiap piksel (mask), IoU, RLE | `vision/segmentasi/semantik` | 1–7 |
| hitung objek + luas tanpa training (Otsu), poligon ↔ mask, YOLO-seg | `vision/segmentasi/mask_kontur` | 1–6 |
| anotasi COCO json / YOLO txt / VOC xml / CVAT xml → tabel | `vision/deteksi/anotasi_baca` | sesuai format |
| tabel → dataset YOLO + `data.yaml`, putar 90°, COCO json | `vision/deteksi/anotasi_tulis` | 1–4 |
| latih detektor kotak, mAP, prediksi (4 rotasi), kotak terbaik | `vision/deteksi/deteksi_yolo` | 1–6 |
| potong kotak, mendatarkan, mempertajam, deskew, 4 titik | `vision/ocr/potong_perbaiki` | 1–5 |
| baca teks (EasyOCR offline), pilih rotasi, regex field, CER | `vision/ocr/ocr` | 1–6 |
| gambar terbalik/menyamping → CNN tebak orientasi | `vision/ocr/orientasi_cnn` | semua |
| latih OCR sendiri dari teks sintetis (CNN + Transformer + CTC) | `vision/ocr/ocr_cnn_transformer` | semua |
| aksara non-Latin (Jawa) dengan label buatan sendiri (CVAT) | `vision/ocr/karakter_label_sendiri` | semua |
| kotak → potong → OCR → submission, sekali jalan | `pipeline/deteksi_potong_ocr` | Run All |
| vignette: YOLO 2 kelas → orientasi Paddle → PaddleOCR → submission (patokan v2) | `pipeline/ocr_end_to_end_v2` | 0–6 |

Format tabel standar vision:

| tabel | kolom |
|---|---|
| `images` | path, width, height |
| `boxes` / `pred_boxes` / `best_boxes` | path, label, x1, y1, x2, y2 (piksel), + score untuk prediksi |
| `crops` | crop_path, path, label, width, height, + text, conf, rotation setelah OCR |

Alur soal "baca teks di bagian gambar" (contoh: vignette):

1. `deteksi/anotasi_baca` (COCO) → `deteksi/anotasi_tulis` (YOLO + putar 90°) → `deteksi/deteksi_yolo`
2. `deteksi_yolo` (4 rotasi, kotak terbaik) → `ocr/potong_perbaiki` (potong, `fix`, SEMUA)
3. `ocr/ocr` (EasyOCR + pilih rotasi) → opsional `ocr/orientasi_cnn` → `ocr/ocr` (rapikan)
4. submission: sel terakhir `pipeline/deteksi_potong_ocr`

Train tanpa teks = OCR tidak bisa dilatih dari soal. Pakai pretrained, atau latih dari teks
sintetis (`ocr_cnn_transformer`), lalu fine-tune bila sempat mengetik label beberapa potongan.

## Teks: `teks/<tugas>/`

| soal / kendala | file | sel |
|---|---|---|
| kenali data teks: panjang, kembar, kembar beda label | `teks/dasar/praproses` | 1 |
| bersihkan (URL, username, gaul), stemming Sastrawi, stopword | `teks/dasar/praproses` | 2–3 |
| kata khas per kelas (chi²) untuk EDA/slide | `teks/dasar/praproses` | 4 |
| **teks + label, baseline tercepat** (TF-IDF kata+huruf) | `teks/klasifikasi/tfidf` | 1–5 |
| target angka dari teks (Ridge) | `teks/klasifikasi/tfidf` | 7 |
| vektor kalimat MiniLM + gabung proba dengan TF-IDF | `teks/klasifikasi/embedding` | 1–5 |
| akurasi tertinggi, ada GPU (fine-tune IndoBERT) | `teks/klasifikasi/finetune_bert` | 1–7 |
| tanpa data berlabel (mDeBERTa NLI) + self-training | `teks/zero_shot/zero_shot_nli` | 1–6 |
| satu teks banyak label, ambang per label | `teks/multilabel/multilabel` | 1–5 |
| sentimen per aspek (ABSA) | `teks/aspek/aspek_sentimen` | 1–4 |
| entitas per kata (NER): CRF / IndoBERT | `teks/ner/ner` | 1–5 / 6–9 |
| pasangan kalimat: sama makna, entailment, duplikat | `teks/pasangan/pasangan` | 1–6 |
| cari dokumen (BM25 + embedding + RRF), jawab pertanyaan (QA) | `teks/kemiripan/retrieval` | 1–5 |
| teks kembar / hampir kembar | `teks/kemiripan/retrieval` | 6 |
| topik tanpa label (NMF, UMAP+HDBSCAN, c-TF-IDF), tren per bulan | `teks/pengelompokan/topik` | 1–7 |
| ringkasan ekstraktif (LEAD, TextRank, embedding) + ROUGE | `teks/ringkasan/ringkasan` | 1–6 |

## Tabular: `tabular/<tugas>/`

Urutan kerja yang disarankan (playbook Kaggle Grandmaster): validasi dulu → baseline GBDT →
fitur → tuning seperlunya → ensembling. Semua skor dipilih dari **OOF/CV**, bukan test.

| soal / kendala | file | sel |
|---|---|---|
| **pilih fold**: stratified / group / kfold → kolom `fold` | `tabular/dasar/validasi` | 1–2 |
| kolom bocor (satu kolom saja sudah menebak target) | `tabular/dasar/validasi` | 3 |
| train vs test berbeda? (adversarial validation) | `tabular/dasar/validasi` | 4 |
| split waktu vs acak, TimeSeriesSplit, stabilitas seed | `tabular/dasar/validasi` | 5–6 |
| fitur kategori: code / count / one-hot / target encoding OOF | `tabular/fitur/fitur` | 2–4 |
| fitur tanggal (sin/cos), agregasi per entitas, rolling, jarak GPS, log/bin | `tabular/fitur/fitur` | 5–8 |
| cek kelompok fitur mana yang membantu | `tabular/fitur/fitur` | 9 |
| **klasifikasi**: LogReg, LightGBM, XGBoost, CatBoost, rata-rata | `tabular/klasifikasi/gbdt` | semua |
| curiga label salah (OOF yakin tapi beda label) | `tabular/klasifikasi/gbdt` | 10 |
| **regresi**: target log, Ridge vs GBDT, plot sisaan, clip | `tabular/regresi/gbdt` | semua |
| kelas positif langka: bobot kelas, ambang, SMOTE yang BENAR | `tabular/imbalanced/imbalanced` | semua |
| tuning Optuna (8 parameter LightGBM), ruang XGB/Cat | `tabular/tuning/optuna_tuning` | semua |
| gabung model: rata-rata, hill climbing, stacking, seed, pseudo-label | `tabular/ensembling/ensembling` | semua |
| jelaskan model ke juri: gain, permutation, SHAP, PDP/ICE | `tabular/penjelasan/penjelasan` | semua |
| metrik ditulis sendiri (F1, AUC, logloss, RMSE, QWK, MAP@k, NDCG) | `tabular/metrik/metrik` | 1–5 |
| optimasi ambang QWK / pengali kelas F1 macro | `tabular/metrik/metrik` | 6–8 |
| urutkan item per query (LGBMRanker, NDCG@k) | `tabular/ranking/ranking` | semua |
| banyak tabel (induk + transaksi), agregasi sebelum tanggal kejadian | `tabular/multitabel/multitabel` | semua |
| segmentasi pelanggan (RFM + KMeans/GMM), profil segmen | `tabular/pengelompokan/segmentasi` | semua |
| anomali tanpa label (robust z, Mahalanobis, IsolationForest, LOF) | `tabular/anomali/anomali` | semua |
| "yang beli A juga beli B" (apriori, lift, rekomendasi) | `tabular/asosiasi/asosiasi` | semua |
| durasi sampai kejadian + sensor (KM, Cox, C-index) | `tabular/survival/survival` | semua |
| A/B test: siapa yang terpengaruh promo (uplift, Qini) | `tabular/uplift/uplift` | semua |

## Jembatan antar modalitas: `jembatan/`

| ingin | file |
|---|---|
| gambar → vektor `.npy` (ResNet/ConvNeXt/CLIP) | `jembatan/gambar_ke_vektor` |
| teks → vektor (TF-IDF, MiniLM, IndoBERT) | `jembatan/teks_ke_vektor` |
| vektor + kolom tabel → satu model (gabung awal/akhir, stacking) | `jembatan/vektor_ke_model` |

Contoh skor di data demo (gambar CIFAR + kolom tabel, akurasi test): tabel saja 0,743, vektor
saja 0,845, gabung akhir 0,903, gabung awal 0,934, stacking 0,936.

## Mengubah atau menambah notebook

`.ipynb` di `arsenal/` dan `contoh/` adalah satu-satunya sumber (tidak ada salinan `.py` lagi).
Edit notebook-nya langsung, simpan **tanpa output** (Kernel → Restart & Clear Output), lalu cek:

```bash
python _alat/lint.py arsenal/tabular/survival/survival.ipynb
python _alat/uji_arsenal.py arsenal/tabular/survival/survival.ipynb:1,2 --set NAMA=nilai
_alat\bangun_contoh.bat tabular        # Run All contoh/, hasil ke ../outputs/uji_contoh/
```

Draf baru boleh ditulis sebagai `.py` bersel `# %%` lalu diubah dengan
`python _alat/py2nb.py draf.py arsenal/<kategori>/<nama>.ipynb`; drafnya jangan disimpan di repo.
