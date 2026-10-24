# PETA ARSENAL

Cara pakai: cari soalnya di tabel, buka file potongannya, salin sel yang perlu ke notebook kosong,
lalu ubah baris yang bertanda `# UBAH`. Tiap sel punya import sendiri.

## Aturan menempel (berlaku di semua file)

- Tiap file diawali bagian **Cara menempel**: tabel urutan sel, mana yang wajib, mana yang opsional.
- Tiap sel diawali **Butuh:** (variabel/file dari sel sebelumnya) dan **Menghasilkan:**. Kalau
  muncul `NameError`, lihat baris **Butuh:** sel itu: sel yang membuatnya belum dijalankan.
- Sel "Fungsi ..." hanya **membuat** fungsi dan mencobanya pada satu contoh. Pekerjaan untuk semua
  data ada di sel "... SEMUA + simpan". Jangan lupa sel itu.
- Hasil tiap tahap disimpan ke disk. Kernel mati? Jangan ulang dari awal, cukup baca CSV-nya:

| tahap | file | baca ulang |
|---|---|---|
| latih YOLO | `outputs/yolo_runs/latih/weights/best.pt` | `BEST_WEIGHTS` di sel Konfigurasi `deteksi_yolo` |
| kotak terbaik | `outputs/best_boxes.csv` | `best_boxes = pd.read_csv("outputs/best_boxes.csv")` |
| potong + perbaiki + OCR | `outputs/crops.csv` | `crops = pd.read_csv("outputs/crops.csv")` |

## Format tabel standar (semua potongan memakai ini)

| tabel | kolom |
|---|---|
| `images` | path, width, height |
| `boxes` / `pred_boxes` / `best_boxes` | path, label, x1, y1, x2, y2 (piksel), + score untuk prediksi |
| `crops` | crop_path, path, label, width, height, + text, conf, rotation setelah OCR |

## Soal → potongan

| soal / kendala | file | sel |
|---|---|---|
| anotasi COCO json (Roboflow, CVAT COCO, Label Studio) | `vision/anotasi_baca.ipynb` | COCO json |
| anotasi YOLO txt / Pascal VOC xml / CVAT xml | `vision/anotasi_baca.ipynb` | sesuai format |
| cek jumlah kotak, kelas induk kosong, ukuran kotak | `vision/anotasi_baca.ipynb` | Cek isi |
| buang/gabung/ganti nama kelas | `vision/anotasi_tulis.ipynb` | Pilih kelas |
| tabel → dataset YOLO + data.yaml | `vision/anotasi_tulis.ipynb` | Dataset YOLO |
| foto test terputar/terbalik, train selalu tegak | `vision/anotasi_tulis.ipynb` | Augmentasi putar 90° |
| tabel → COCO json | `vision/anotasi_tulis.ipynb` | COCO json |
| latih detektor kotak, mAP val | `vision/deteksi_yolo.ipynb` | Latih, Evaluasi |
| prediksi folder → tabel kotak | `vision/deteksi_yolo.ipynb` | Prediksi |
| prediksi foto terputar (4 rotasi) | `vision/deteksi_yolo.ipynb` | 4 rotasi |
| satu kotak per gambar, ada gambar tanpa deteksi | `vision/deteksi_yolo.ipynb` | Kotak terbaik |
| potong kotak dari gambar | `vision/potong_perbaiki.ipynb` | Potong |
| teks berdiri, buram, kontras rendah | `vision/potong_perbaiki.ipynb` | Mendatarkan + mempertajam |
| teks miring sedikit | `vision/potong_perbaiki.ipynb` | Deskew |
| stiker/dokumen difoto miring (perspektif) | `vision/potong_perbaiki.ipynb` | 4 titik |
| baca teks (pretrained, offline) | `vision/ocr.ipynb` | EasyOCR |
| teks terbalik → pilih rotasi dengan skor OCR | `vision/ocr.ipynb` | Pilih rotasi |
| gambar terbalik/menyamping → CNN tebak orientasi | `vision/orientasi_cnn.ipynb` | semua |
| rapikan teks, ambil field (LOT/EXP) dengan regex | `vision/ocr.ipynb` | Rapikan |
| CER / exact match | `vision/ocr.ipynb` | Metrik |
| latih OCR sendiri (tanpa label: teks sintetis) | `vision/ocr_cnn_transformer.ipynb` | semua |
| potongan banyak baris → per baris | `vision/ocr_cnn_transformer.ipynb` | split_lines |
| kotak → potong → OCR → submission, sekali jalan | `pipeline/deteksi_potong_ocr.ipynb` | Run All |

## Alur soal "baca teks di bagian gambar" (contoh: vignette)

1. `anotasi_baca` (COCO) → `anotasi_tulis` (YOLO + putar 90°) → `deteksi_yolo` (latih)
2. `deteksi_yolo` (4 rotasi, kotak terbaik) → `potong_perbaiki` (potong, `fix`, terapkan ke SEMUA)
3. `ocr` (EasyOCR + pilih rotasi) → opsional `orientasi_cnn` (pseudo-label dari OCR) → `ocr` (rapikan)
4. submission: lihat sel terakhir `pipeline/deteksi_potong_ocr.ipynb`

Train tanpa teks = OCR tidak bisa dilatih dari soal. Pakai pretrained, atau latih dari teks
sintetis (`ocr_cnn_transformer`), lalu fine-tune bila sempat mengetik label beberapa potongan.
