# Keputusan

Terbaru di atas. Format: `- YYYY-MM-DD: <keputusan>. Alasan: <...>.`

- 2026-10-30: memori agen disimpan di dalam repo (`_memori/`) dan ikut di-commit, bukan hanya di
  memori lokal Claude. Alasan: sesi baru di komputer lain (atau agen lain) harus langsung tahu keadaan.
- 2026-10-30: notebook vignette patokan (`ocr_end_to_end_v2`) masuk `arsenal/pipeline/` apa adanya,
  output dikosongkan, isi kode tidak diubah. Alasan: pilihan pengguna; penyesuaian lint ditunda.
- 2026-10-24: arsenal ditulis langsung sebagai `.ipynb`, bukan `.py` bersel `# %%`. Alasan: pengguna
  menyalin sel dari notebook; `.py` sering lupa dikonversi.
- 2026-10-24: output semua notebook ke `../../outputs/`. Alasan: repo di-push, folder notebook harus bersih.
