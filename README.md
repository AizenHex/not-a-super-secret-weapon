# Arsenal GEMASTIK (offline)

Bahan lomba data mining yang harus jalan **tanpa internet dan tanpa AI**. Semua kode Python
memakai gaya sederhana (pandas/numpy, tanpa comprehension/lambda/class rumit) supaya bisa dibaca
dan diubah oleh orang yang terbiasa C++.

## Isi repo

| folder | isi | kapan dibuka |
|---|---|---|
| [`arsenal/`](arsenal/PETA.md) | potongan **siap tempel** per modalitas → per tugas (`vision/`, `teks/`, `tabular/`, `jembatan/`, `umum/`, `pipeline/`); tiap file punya tabel "Cara menempel" | membangun notebook sendiri, sel demi sel |
| [`contoh/`](contoh/PANDUAN.md) | notebook **utuh** per kasus (199 notebook, 19 modalitas), Run All → `submission.csv`; skor tiap alternatif di `KATALOG.md` | baseline cepat di jam pertama |
| [`kamus/`](kamus/README.md) | padanan C++ → Python, jebakan, numpy/pandas, arti pesan error | lupa sintaks |
| `_laya_eda/` | web lokal untuk melihat data (`jalankan.bat`) + ekspor kode EDA | menit awal, kenali data |
| `_ringkasan_metode/` | ringkasan metode (PDF) | bahan presentasi |
| `_alat/` | `lint.py`, `uji_arsenal.py`, `bangun_contoh.bat`, `py2nb.py`, skrip unduh data/model | mengubah/menambah notebook |

Mulai dari [`arsenal/PETA.md`](arsenal/PETA.md) (soal → potongan) atau
[`contoh/PANDUAN.md`](contoh/PANDUAN.md) (tanda di soal → notebook utuh).

## Susunan folder yang diharapkan

```
<folder apa saja>/
├── <repo ini>/        setup.bat, arsenal/, contoh/, kamus/
├── pretrained/        hasil download_model.bat (±12 GB)
├── data_latihan/      data demo (opsional, untuk mencoba notebook)
└── env/wheelhouse/    file .whl untuk pasang library tanpa internet (opsional)
```

- `contoh/<modalitas>/x.ipynb` membaca data demo dari `../../../data_latihan` dan model dari
  `../../../pretrained`.
- Potongan `arsenal/` memakai `DEMO_FOLDER = "data_latihan"` dan `PRETRAINED_FOLDER =
  "pretrained"`: ubah ke path folder tadi (garis miring `/`) di notebook tempat menempel.

## Persiapan saat jendela setup (masih ada internet, ±5 jam)

Urut dari yang paling penting. Centang satu per satu.

1. [ ] **Salin repo + `pretrained/`** ke path pendek (`C:/GEMASTIK/`). Path panjang membuat
   pemasangan library gagal ("Long Path").
2. [ ] **`setup.bat`** → harus berakhir `OK. GPU: True` lalu `SUKSES`. Python 3.13 64-bit.
3. [ ] **Model**: `pretrained/` belum ikut? `download_model.bat` (bisa dijalankan ulang, yang
   sudah ada dilewati). Isi: 18 model Hugging Face (MiniLM, IndoBERT, mDeBERTa NLI/QA, CLIP,
   TrOCR, Qwen 0.5B, Whisper, Chronos, ...), 9 backbone torchvision, YOLOv8n (deteksi/seg/pose),
   wajah OpenCV, EasyOCR (`en`, `id`), Laya.
4. [ ] **Wheelhouse cadangan** (kalau komputer lomba berbeda dari laptop sendiri):
   ```bash
   python -m pip download -r requirements.txt -d ../env/wheelhouse
   ```
   Harus dijalankan di Windows + Python 3.13 yang sama dengan komputer lomba. `setup.bat`
   otomatis memakai `../env/wheelhouse` kalau folder itu ada.
5. [ ] **Jalankan sekali setiap notebook yang mungkin dipakai** selagi online. Library tertentu
   diam-diam mengunduh file saat pertama jalan:
   - ultralytics (YOLO) mengecek AMP dengan mengunduh `yolo26n.pt`. Salinannya ada di
     `pretrained/ultralytics/yolo26n.pt` → taruh di folder kerja notebook, atau latih dengan
     `amp=False`. Font `Arial.ttf` diambil dari font Windows (aman).
   - Hugging Face: setelah online selesai, set `HF_HUB_OFFLINE=1` supaya tidak mencoba internet
     dan memberi error lambat; semua notebook memuat model dari folder lokal.
   - EasyOCR: wajib `model_storage_directory=pretrained/easyocr` + `download_enabled=False`.
6. [ ] **Font aksara** (soal OCR aksara Jawa / non-Latin): `vision/ocr/karakter_label_sendiri`
   memakai font Windows `C:/Windows/Fonts/javatext.ttf` (Javanese Text). Unduh cadangan
   **Noto Sans Javanese** (Google Fonts, `.ttf`) ke flashdisk dan isi `FONT_PATH` kalau font
   Windows tidak ada. Untuk aksara lain, unduh Noto Sans <aksara> yang sesuai.
7. [ ] **Alat label** (kalau soal butuh melabeli sendiri): CVAT self-host butuh Docker; jalankan
   `docker compose pull` saat online supaya image tersimpan, lalu coba buka `localhost:8080`.
   Cadangan tanpa Docker: Label Studio (`pip install label-studio`, ikut wheelhouse).
8. [ ] **Dokumentasi offline**: `kamus/`, `arsenal/PETA.md`, `contoh/PANDUAN.md`,
   `contoh/RISET.md`. Opsional: simpan halaman dok scikit-learn / pandas / LightGBM sebagai PDF.
9. [ ] Uji akhir: matikan Wi-Fi, buka `jupyter.bat`, jalankan satu notebook vision, teks, dan
   tabular dari awal sampai `submission.csv`.

## Aturan kode

Notebook di repo ini dicek oleh `_alat/lint.py`: tanpa comprehension, lambda, ternary, try,
assert, rantai method ≥ 3; sel ≤ 30 baris; baris ≤ 100 karakter; fungsi ≤ 20 baris dan langsung
dicoba di sel yang sama. Notebook disimpan tanpa output; cara mengubah/menambah: bagian
terakhir [`arsenal/PETA.md`](arsenal/PETA.md).
