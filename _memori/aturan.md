# Aturan kerja

Koreksi terbaru pengguna di chat menang atas isi file ini; setelah itu perbarui file ini.

## Komunikasi

- Bahasa Indonesia. Jawab inti dulu (1–3 kalimat), detail hanya kalau diminta. (2026-10-24)
- Pengguna paham konsep ML (latar OSN Informatika, C++) tapi tidak hafal sintaks Python. Notebook harus
  lengkap dan bisa Run All; yang diubah cukup sel Konfigurasi / baris `# UBAH`. (2026-10-23)
- "Mode latihan" hanya kalau pengguna menyalakannya: saat itu jangan menulis kode, hanya arahan dan
  review. Default: mode biasa. Ragu → tanya satu baris. (2026-10-23)
- Jangan menulis tebakan sebagai fakta (pola dari nama file, isi data yang tidak dibaca pasti). (2026-10-30)

## Kode

- Potongan `arsenal/` disimpan sebagai `.ipynb`, itu satu-satunya sumber. Lint/uji membaca `.ipynb`
  langsung. (2026-10-24)
- Gaya dicek `_alat/lint.py`: tanpa try/except, assert, comprehension, lambda bertumpuk, ternary
  (kecuali baris `DEVICE`); fungsi ≤ 20 baris, sel ≤ 30 baris, baris ≤ 100 karakter; `class` hanya
  untuk `nn.Module`/`Dataset`; submission = `pd.DataFrame()` baru, kolom ditambah satu per satu.
- Arsenal tanpa path dataset tertentu; baris yang perlu diubah diberi `# UBAH`.
- Semua output notebook ke luar repo: `OUTPUT_FOLDER = "../../outputs/<kategori>/<nama>"`. Jangan
  membuat file di folder notebook. (2026-10-24)
- Uji potongan: `_alat/uji_arsenal.py file.ipynb:1,2 --set NAMA=nilai`.
- Notebook `arsenal/` dan `contoh/` disimpan tanpa output (pengguna minta repo "compact"). Jangan
  membuat lagi salinan sumber `.py` di repo. (2026-10-30)

## Laya EDA (`_laya_eda/`)

- Jangan pernah mengganti nama kolom asli di file hasil (tidak ada `_angka` / `__angka`): kolom angka
  diisi angka hasil baca di kolom yang sama. Kolom baru (fitur) boleh, kolom asli tetap. Pengguna:
  "niatnya untuk mempermudah EDA, ini malah dipersulit". (2026-10-31)
- Tiga tab terakhir selalu Split data → Kode EDA → Ekspor (urutan yang pengguna kenal). (2026-10-31)
- Kode EDA = semua yang dilakukan ke data dan harus diulang saat inferensi, siap tempel: langkah isi
  data (angka, isi kosong, fitur, buang kolom) untuk train + test dengan nilai dari train; buang baris
  dan ganti label hanya train. train_siap di aplikasi harus = hasil kode itu. (2026-10-31)
- Wajib ada pilihan isi angka kosong (median/mean/modus/0, per kolom). (2026-10-31)
- Kecepatan penting: klik widget tidak boleh menghitung ulang hal berat (pakai `st.cache_data` dengan
  kunci `ctx["data_key"]`, unduhan CSV lewat callable). Ukur rerun mode baris pada `data/train.csv`
  (30k baris) sesudah mengubah tab. (2026-10-31)
- Setiap kolom yang tampil di tabel harus punya arti di `kamus.py` (nilai apa, dari apa); tabel
  baru pakai `kamus.show_table`, bukan `st.dataframe`. Pengguna bingung arti "nilai". (2026-10-31)

## Lingkungan

- Venv: `Z:\GEMASTIK\Vignate-GEMASTIK\.venv` (Python 3.13). Pretrained offline:
  `Z:\GEMASTIK\Vignate-GEMASTIK\pretrained`.
- Laptop pernah mati mendadak saat beban GPU berat: jalankan build GPU satu per satu di foreground,
  batch kecil (mis. 8), jangan menaikkan batch yang komentarnya menyebut panas. (2026-10-24)
- Tanya dulu sebelum memasang paket pip.

## Figur (PPT/laporan)

- Satu pesan per figur, 16:9, huruf besar; tidak ada label/klaim tebakan di figur. (2026-10-30)
- Daftar cek sebelum figur dianggap selesai (diadaptasi dari kritik PaperBanana, 2026-10-30):
  isi setia ke data/metode (tidak ada yang dikarang, tidak ada komponen penting yang hilang); cek
  ejaan dan label; contoh di figur benar; judul/caption tidak ditulis di dalam gambar; buang legenda
  yang berulang dan beri label langsung di garis/batang.
- Plot: grid tipis putus-putus di belakang data, garis diberi marker, nilai ditulis di sel heatmap,
  jangan colormap jet/rainbow, warna dibantu bentuk/pola. Diagram: kotak sudut membulat untuk proses,
  garis putus-putus untuk alur tambahan/tahap logis, warna jenuh hanya untuk keluaran akhir.

- Figur kartu proses untuk PPT (2026-10-30): kanvas 13×5,4 in (≈1:1 di slide 13,33 in), font
  Be Vietnam Pro, judul kartu 22, angka besar 23, keterangan 15,5, legenda ≥ 14. Desain kartu polos:
  pita judul + pita bawah pastel, chevron antar kartu. **Jangan** tambah pita "kenapa", bayangan,
  atau kurung fase; pengguna menilai versi polos lebih rapi. Satu angka = satu sumber (semua figur
  deteksi pakai prediksi `yolo_2cls_v2.pt` yang sama).

## Git

- Commit/push hanya kalau pengguna memintanya di sesi itu. Kalau diizinkan ("kalau ada perubahan kode,
  silahkan di push"), commit perubahan kode + memori bersama lalu push ke branch kerja. (2026-10-30)
- Jangan commit `outputs/`, data, atau hasil Run All (sudah diatur `.gitignore`).
