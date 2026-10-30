# Rencana lomba: apa yang harus selesai sebelum final

Dibuat 24 Okt 2026. Final GEMASTIK XIX Data Mining: setup 10 Nov, **pengerjaan 11 Nov 13.00–18.00
(5 jam)**, presentasi luring 12 Nov. Sisa ±17 hari.

Ukuran keberhasilan arsenal ini **bukan jumlah notebook**, tapi:
1. tiap alur berakhir di file yang bisa diunggah + skor validasi lokal,
2. tahan terhadap bentuk data soal yang belum pernah dilihat,
3. baseline pertama terunggah secepat mungkin.

## Bobot nilai final (Panduan GEMASTIK XIX 2026)

| komponen | bobot | catatan |
|---|---|---|
| penyisihan | 20% | sudah terkunci |
| leaderboard | 30% | file jawaban di platform (2021: eval.ai, ±10 kirim di fase uji) |
| ketepatan metode | 25% | dinilai juri dari kode + dokumentasi + presentasi |
| kejelasan & kelengkapan presentasi | 25% | PPT, temuan dari data |
| syarat kumpul | gugur | zip: dokumentasi pengembangan model + source code + PPT; video profil tim |

Data bisa terstruktur, gambar, video, atau teks. Riwayat: 2021 klasifikasi (Macro-F1),
2023 jenis serangan dari log jaringan (kelas langka), 2024 tabular, 2025 OCR aksara Jawa
(data tidak lengkap, harus melabeli sendiri).

## Pohon pekerjaan

Tanda: ✅ ada dan teruji · ⚠️ ada tapi bolong · ❌ belum ada

```
Menang final 11 Nov
├── Leaderboard 30%  (file valid + skor)
│   ├── Alur per tipe data sampai submission.csv
│   │   ├── ✅ Tabular klasifikasi/regresi        arsenal/tabular, umum/simpan_muat §7
│   │   ├── ✅ Tabel kotor                         umum/tabel_bersih
│   │   ├── ✅ Klasifikasi gambar (folder/kelas)   vision/klasifikasi
│   │   ├── ✅ OCR aksara + label sendiri          vision/ocr/karakter_label_sendiri
│   │   ├── ✅ Teks klasifikasi / teks→angka       teks/klasifikasi, jembatan/
│   │   ├── ❌ Deteksi objek → submission          berhenti di best_boxes.csv
│   │   │   ├── ⚠️ baca YOLO susunan folder apa pun   (draf, belum masuk)
│   │   │   ├── ❌ split val dari data / acak bila tak ada
│   │   │   ├── ❌ sel submission: per-kotak ATAU PredictionString (1 variabel)
│   │   │   └── ❌ gambar tanpa deteksi tetap ada baris; conf rendah untuk mAP
│   │   ├── ⚠️ Segmentasi RLE                      vision/segmentasi/semantik: 1 kelas saja
│   │   │   └── ❌ multi-kelas `<img>_<kelas>`, urutan baris/kolom, resize balik
│   │   ├── ⚠️ Deret waktu                         hanya di contoh/timeseries
│   │   └── ❌ Dataset Hugging Face (parquet: gambar bytes + ClassLabel)
│   ├── Format jawaban bisa diubah di satu tempat
│   │   ├── ✅ label teks ("Ya"/"Tidak") lewat map
│   │   └── ⚠️ cek vs sample_submission: kolom/jumlah/NaN ada;
│   │         ❌ id sama & urutan, nilai label ⊂ contoh, tipe angka
│   └── Strategi kuota kirim
│       ├── ⚠️ baseline ≤ 45 menit                 RISET.md §5, belum diuji pakai stopwatch
│       └── ❌ log percobaan (skor lokal vs publik → pilih 2 final)
├── Ketepatan metode 25%
│   ├── ✅ CV + metrik dasar                       tabular/metrik (F1, AUC, AP, QWK, RMSE, ...)
│   ├── ✅ ambang/cut optimal                      optimize_cuts, optimize_class_weights
│   ├── ❌ metrik vision/teks: mAP@50, mAP@.5:.95, mIoU, Dice, CER/WER
│   └── ⚠️ alasan pemilihan metode tertulis         _ringkasan_metode PDF, tidak nyambung ke notebook
├── Presentasi 25%
│   ├── ✅ EDA cepat                               _laya_eda
│   ├── ⚠️ grafik → PNG → .pptx                    hanya contoh/visualisasi/presentasi (Titanic)
│   └── ❌ kerangka slide: masalah, data, temuan, metode, validasi, hasil, batasan
└── Syarat kumpul (gugur kalau tidak ada)
    ├── ❌ zip: dokumentasi model + source code + PPT   (yang ada hanya zip outputs/)
    ├── ✅ checklist setup offline                  README.md
    ├── ⚠️ uji offline penuh vision/teks/tabular    belum dijalankan dengan Wi-Fi mati
    └── ⬜ video profil tim                          di luar repo, urusan tim
```

## Urutan kerja (nilai × peluang muncul ÷ usaha)

| # | pekerjaan | alasan |
|---|---|---|
| 1 | Deteksi: pembaca YOLO generik → split val → latih → **submission + mAP lokal** | 2025 soal vision + label sendiri; celah terbesar |
| 2 | Validator submission universal (id, urutan, nilai label, tipe, NaN) | satu salah format = skor 0, berlaku untuk semua tipe |
| 3 | Paket kumpul: zip dokumentasi + kode + PPT, dan potongan grafik → pptx di arsenal | syarat gugur + 25% presentasi |
| 4 | Metrik mAP / mIoU / Dice / CER di tabular/metrik | validasi lokal harus = metrik soal |
| 5 | RLE multi-kelas + resize balik | format segmentasi Kaggle yang umum |
| 6 | Pembaca dataset HF parquet → folder/csv | bentuk data yang sering diberikan sekarang |
| 7 | Deret waktu ke arsenal | sudah ada di contoh/, tinggal dipotong |
| 8 | Log percobaan + latihan 5 jam pakai stopwatch | strategi kuota, ukur menit sampai baseline |

## Sumber

- Panduan GEMASTIK XIX 2026, bagian Data Mining (jadwal, bobot, isi zip, jenis data).
- Sosialisasi GEMASTIK 2025 (`_alat/sosialisasi2025.txt`): bobot sama, final 27–29 Okt 2025.
- eval.ai challenge 1274 (GEMASTIK 2021): Macro-F1, `.csv`, fase uji ±5 jam, maks 10 kirim.
- Kaggle: 272 kompetisi berbahasa Indonesia dari ±1.500 yang dipanen (format jawaban, metrik,
  ±5 kirim/hari, pilih 2 final, papan publik/privat).
- Hugging Face: `setilanaji/aksara-jawa-ocr` (imagefolder), `RayFox52/Aksara-jawa-carakan-murda`
  (parquet).
