# Keputusan

Terbaru di atas. Format: `- YYYY-MM-DD: <keputusan>. Alasan: <...>.`

- 2026-10-30: angka deteksi figur PPT (664/674 citra uji berkotak entete, 10 tanpa) diambil dari
  prediksi ulang CPU `Downloads/yolo_2cls_v2.pt` (imgsz 416, conf 0,1) ke
  `Vignate-2/outputs/figures/_cache/det_v2_test.csv`, bukan dari output notebook (640/674). Alasan:
  pengguna memilih "pakai v2 aja"; `debug_v2.csv` dan `yolo_2cls.pt` lama tidak ada di laptop ini.
  Checkpoint di Downloads berupa folder (torch zip yang sudah diekstrak); disalin + di-zip ulang
  (ZIP_STORED) di scratchpad, aslinya tidak diubah. Model ini berbeda dari run notebook (conf 005e28
  0,823 vs 0,867). `f08_11_v2.py` (tabel02, funnel) masih memakai angka lama 639/35.
- 2026-10-30: PaperBanana tidak dipasang; yang diambil hanya daftar cek gaya dan kritiknya (lihat
  `aturan.md` bagian Figur). Alasan: butuh kunci API Gemini/OpenRouter dan internet (lomba luring),
  hasilnya gambar raster dari model pembuat gambar yang bisa mengarang teks/angka, kode plot statistiknya
  belum dirilis, dan isi figur terkirim ke layanan luar.
- 2026-10-30: pengingat memori dipasang sebagai hook Claude Code (SessionStart + UserPromptSubmit +
  Stop) yang hanya memakai bash + git. Alasan: aturan di CLAUDE.md saja bisa terlewat; jq/python belum
  tentu ada di PATH hook. Stop memblokir hanya sekali per giliran (`stop_hook_active`) supaya tidak
  berputar, dan hanya menilai perubahan selama giliran itu (potret saat prompt masuk), jadi perubahan
  lama milik pengguna tidak memicu.

- 2026-10-30: memori agen disimpan di dalam repo (`_memori/`) dan ikut di-commit, bukan hanya di
  memori lokal Claude. Alasan: sesi baru di komputer lain (atau agen lain) harus langsung tahu keadaan.
- 2026-10-30: notebook vignette patokan (`ocr_end_to_end_v2`) masuk `arsenal/pipeline/` apa adanya,
  output dikosongkan, isi kode tidak diubah. Alasan: pilihan pengguna; penyesuaian lint ditunda.
- 2026-10-24: arsenal ditulis langsung sebagai `.ipynb`, bukan `.py` bersel `# %%`. Alasan: pengguna
  menyalin sel dari notebook; `.py` sering lupa dikonversi.
- 2026-10-24: output semua notebook ke `../../outputs/`. Alasan: repo di-push, folder notebook harus bersih.
