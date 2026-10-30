# Memori agen

Memori kerja bersama untuk agen AI (Claude Code, Codex, dll.) yang menyentuh repo ini. Ikut di-commit,
jadi sesi baru di komputer mana pun langsung tahu keadaan terakhir.

| file | isi | kapan diubah |
|---|---|---|
| [`status.md`](status.md) | keadaan repo sekarang, pekerjaan terbuka, langkah berikut | setiap ada yang selesai/berubah |
| [`aturan.md`](aturan.md) | aturan kerja + koreksi pengguna yang berlaku seterusnya | saat pengguna mengoreksi atau menetapkan cara kerja |
| [`keputusan.md`](keputusan.md) | keputusan desain beserta alasannya | saat memilih satu cara dari beberapa |
| [`log.md`](log.md) | kronologi sesi, terbaru di atas | akhir setiap sesi/tugas |

## Format entri `log.md`

```
## YYYY-MM-DD <agen/model> - <judul singkat>
- Dikerjakan: <apa yang berubah, path, commit>
- Dikoreksi pengguna: <kutipan singkat> (atau "-")
- Berikutnya: <langkah lanjut>
```

## Aturan menulis

- Bahasa Indonesia, singkat, fakta saja. Tulis tanggal absolut (2026-10-30), bukan "kemarin".
- Satu fakta di satu file. Status di `status.md`, aturan di `aturan.md`; `log.md` hanya merujuk.
- Fakta yang sudah tidak benar: perbarui atau tandai `(usang YYYY-MM-DD: ...)`, jangan biarkan dua versi.
- Jangan menulis rahasia (token, password) atau data pribadi.
- Tebakan ditandai "dugaan", jangan ditulis seolah fakta.
