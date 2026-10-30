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

## Git

- Commit/push hanya kalau pengguna memintanya di sesi itu. Kalau diizinkan ("kalau ada perubahan kode,
  silahkan di push"), commit perubahan kode + memori bersama lalu push ke branch kerja. (2026-10-30)
- Jangan commit `outputs/`, data, atau hasil Run All (sudah diatur `.gitignore`).
