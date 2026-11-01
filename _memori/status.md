# Status repo

Terakhir diperbarui: 2026-11-02.

## Tujuan

Arsenal offline untuk final GEMASTIK XIX Penambangan Data: setup 2026-11-10, pengerjaan
2026-11-11 13.00–18.00 (5 jam, luring, tanpa internet dan tanpa AI), presentasi 2026-11-12.
Pengguna berangkat 2026-11-09. Prioritas dan pohon pekerjaan: [`../RENCANA_LOMBA.md`](../RENCANA_LOMBA.md).

## Keadaan sekarang

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
  label) diuji pada `Vignate-GEMASTIK/data`; bagian 3–6 **belum diuji** (PaddleOCR belum dipasang,
  `yolo_2cls.pt` tidak ada di laptop).
- Di luar repo (2026-10-30): salinan repo lama (`PythonProject`, `Sen-3`..`Sen-6`, `Sentiment-2`, ±35 GB)
  dan file lepas di root `Vignate-GEMASTIK` dipindah ke `Z:/GEMASTIK/_arsip/`, tidak dihapus.
- `_laya_eda/` (2026-10-31): tiga tab terakhir Split data → Kode EDA → Ekspor. Kode EDA memilih
  langkah (buang baris, isi angka kosong, fitur, buang kolom) dan menulis kode train + test yang juga
  dijalankan aplikasi untuk train_siap; nama kolom asli tidak diubah. Rerun mode baris ±4 s di
  `data/train.csv`. Kamus kolom (tooltip judul kolom + sidebar) dan encoding kategori (one-hot,
  kode, target, kedekatan makna MiniLM) di Kode EDA. `_uji/test_app.py` dan
  `_uji/test_codegen.py` belum dijalankan ulang penuh.
- `_laya_eda_v2/` (2026-11-02, sesi cloud, branch `new-gen`): `VISI.md` = satu-satunya sumber target (extra
  E1–E12 di atas R1–R18; 8 temuan review sudah masuk). E1–E12 diimplementasikan (tab **Latih** dan **Submission**,
  14 halaman) dan R1–R18 punya uji. `python -m tests.semua` di venv cloud (versi persis requirements.txt, torch
  2.11.0 dari PyPI): lulus 47, gagal 0, dilewati 12. Semua uji memakai data **sintetis** (`tests/sintetis.py`);
  yang dilewati = uji yang butuh data riil `../_laya_eda/data/train.csv` (tidak ada di clone) + sample_submission
  lomba 2024. **Belum**: uji di data riil dan di venv Windows `Z:/GEMASTIK/Vignate-GEMASTIK/.venv` (paket
  pandera/feature-engine/skrub/autogluon belum dipasang di sana per 2026-11-02); R17 belum diukur di browser;
  penanda satuan campur + nol mencurigakan; tab Kosong/Encoding. Rincian di STATUS `RENCANA.md`.
- `viginte-train.ipynb` dipindah dari root ke `_.Testing/` (diabaikan git).
- PPT vignette (penyisihan) selesai 2026-10-30. Figur + skripnya di `Z:/GEMASTIK/Vignate-2/outputs/figures/`
  (di luar repo, bukan git); catatan lengkap `Z:/GEMASTIK/Vignate-2/HANDOFF.md`.

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
- Tambahan: uji bagian 3–6 `pipeline/ocr_end_to_end_v2` setelah PaddleOCR dipasang (pengguna yang
  memutuskan kapan). Kalau pengguna setuju, hapus `pretrained/`/data dobel di `Z:/GEMASTIK/_arsip/`.
