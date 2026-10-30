# CLAUDE.md

Repo arsenal offline untuk final GEMASTIK XIX Penambangan Data. Memori kerja agen ada di
[`_memori/`](_memori/README.md) dan ikut di-commit.

## Wajib di awal sesi

1. Baca `_memori/status.md` dan `_memori/aturan.md`. Baca `_memori/log.md` bagian teratas kalau
   butuh konteks sesi terakhir.
2. Sesuai tugas: `README.md`, `RENCANA_LOMBA.md` (prioritas), `arsenal/PETA.md` (soal → notebook → sel).
3. Instruksi pengguna di chat menang atas memori. Kalau bertentangan, ikuti chat lalu perbarui memori.

## Wajib selalu memperbarui memori

Sebelum memberi jawaban akhir untuk tugas yang mengubah sesuatu (file, keputusan, status, atau
pengguna mengoreksi cara kerja):

1. `_memori/log.md`: tambah entri di atas (format di `_memori/README.md`).
2. `_memori/status.md`: perbarui kalau keadaan repo, pekerjaan terbuka, atau langkah berikut berubah.
3. `_memori/aturan.md`: tambah/ubah kalau pengguna mengoreksi perilaku atau menetapkan cara kerja.
   Koreksi pengguna tidak cukup ditulis di log saja.
4. `_memori/keputusan.md`: kalau memilih satu cara dari beberapa, tulis alasannya.
5. Kalau perubahan kode di-commit, commit perubahan memori di commit yang sama.

Hook di `.claude/settings.json` menempel `status.md` + `aturan.md` saat sesi mulai dan menahan
jawaban akhir sekali kalau ada file berubah tanpa `_memori/` ikut berubah. Hook itu pengingat,
bukan pengganti langkah di atas.

Pertanyaan biasa yang tidak mengubah apa pun tidak perlu entri. Jangan menulis rahasia atau tebakan
sebagai fakta.

## Aturan kode (ringkas, detail di `_memori/aturan.md`)

- Arsenal = `.ipynb`, dicek `_alat/lint.py`; output notebook ke `../../outputs/<kategori>/<nama>`.
- Bahasa Indonesia, jawab inti dulu.
- Commit/push hanya kalau pengguna memintanya.
