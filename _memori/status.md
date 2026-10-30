# Status repo

Terakhir diperbarui: 2026-10-30.

## Tujuan

Arsenal offline untuk final GEMASTIK XIX Penambangan Data: setup 2026-11-10, pengerjaan
2026-11-11 13.00–18.00 (5 jam, luring, tanpa internet dan tanpa AI), presentasi 2026-11-12.
Pengguna berangkat 2026-11-09. Prioritas dan pohon pekerjaan: [`../RENCANA_LOMBA.md`](../RENCANA_LOMBA.md).

## Keadaan sekarang

- Branch kerja: `rapikan-struktur` (remote `origin` = github.com/AizenHex/not-a-super-secret-weapon).
  Default remote: `new-gen`.
- `.claude/` (2026-10-30): hook memori (`settings.json`, `hooks/memori.sh`). Aktif kalau Claude Code
  dibuka dari root repo ini; tinjau/matikan lewat `/hooks`.
- `arsenal/`: potongan siap tempel per modalitas (vision, teks, tabular, jembatan, umum, pipeline).
  Peta: [`../arsenal/PETA.md`](../arsenal/PETA.md).
- `contoh/`: 199 notebook utuh, 19 modalitas.
- `arsenal/pipeline/ocr_end_to_end_v2.ipynb` (2026-10-30): salinan notebook vignette patokan
  (YOLOv8n 2 kelas + orientasi Paddle + PaddleOCR → submission). **Belum sesuai aturan repo**:
  204 masalah `_alat/lint.py`, path relatif `data/`, `work/`, `models/` (menulis ke dalam folder kerja,
  bukan `../../outputs/`).
- `viginte-train.ipynb` dipindah dari root ke `_.Testing/` (diabaikan git).

## Pekerjaan terbuka (urut prioritas, dari RENCANA_LOMBA.md)

1. Deteksi objek sampai `submission.csv` + mAP lokal (pembaca YOLO generik, split val, format
   per-kotak/PredictionString, gambar tanpa deteksi tetap punya baris).
2. Validator submission universal (id, urutan, nilai label, tipe, NaN).
3. Paket kumpul: zip dokumentasi + kode + PPT; potongan grafik → pptx.
4. Metrik mAP / mIoU / Dice / CER.
5. RLE segmentasi multi-kelas + resize balik.
6. Pembaca dataset Hugging Face parquet.
7. Deret waktu ke arsenal.
8. Log percobaan + latihan 5 jam pakai stopwatch.
- Tambahan: rapikan `pipeline/ocr_end_to_end_v2` agar lolos lint dan output ke luar repo (kalau
  pengguna minta).
