# Log sesi

Terbaru di atas. Format di [`README.md`](README.md).

## 2026-10-31 claude-opus-5-5 - Laya EDA: tab Split data + buang kustom
- Dikerjakan: `_laya_eda/tab_split.py` (tab baru "Split data" sesudah Ekspor: acak + stratify,
  waktu, kelompok / teks sama; cek teks bocor; tulis `train_split.csv` + `valid_split.csv`; kode
  notebook setara) dan `_laya_eda/custom_drop.py` (buang kolom + aturan baris `train.query`,
  disimpan di `hasil/<csv>/buang_kustom.json`, diterapkan sesudah catatan Telusuri). Sambungan di
  `tab_ekspor.py` (`ctx["ready_train"]`), `app.py`, `context.py`, `penjelasan.py`. Commit 21ab935
  di `new-gen` (di-rebase ke atas `e0b2bfa`, lalu di-push). Uji cepat AppTest (di luar repo) lolos
  sebelum dan sesudah rebase; langkah uji ditambah ke `_uji/test_app.py` tetapi belum dijalankan.
- Dikoreksi pengguna: -
- Berikutnya: jalankan `python _uji/test_app.py` dari `_laya_eda` (butuh GPU); aturan buang kustom
  belum ikut ke kode tab Kode EDA (kode itu membaca CSV mentah tanpa kolom fitur).

## 2026-10-31 claude-opus-5-5 - commit + push, new-gen = rapikan-struktur
- Dikerjakan: semua perubahan repo compact (sesi 2026-10-30) di-commit dan di-push ke
  `rapikan-struktur`; `new-gen` (default) dimajukan ke commit yang sama atas permintaan pengguna.
- Dikoreksi pengguna: -
- Berikutnya: pekerjaan terbuka di `status.md`; uji bagian 3–6 OCR v2 setelah PaddleOCR dipasang.

## 2026-10-30 claude-opus-5-5 - repo compact + rapikan Z:/GEMASTIK
- Dikerjakan: output `contoh/` dikosongkan (contoh/ 81 MB → 2,6 MB); `_alat/sumber_arsenal/` +
  `_alat/sumber_contoh/` dihapus; `bangun_contoh.bat` menjalankan `.ipynb` langsung ke
  `../outputs/uji_contoh/` (diuji: `sinyal` LOLOS); docstring `lint.py`/`uji_arsenal.py`; PETA.md bagian
  akhir + README `_alat/`; `pipeline/ocr_end_to_end_v2` ditulis ulang (lint 204 → 0, bagian 0–2 diuji).
  Di luar repo: 6 salinan repo lama + file lepas root `Vignate-GEMASTIK` → `Z:/GEMASTIK/_arsip/`;
  `README.dataset.txt`/`README.roboflow.txt` → `Z:/GEMASTIK/Indonesian-plat/`. Di-commit 2026-10-31.
- Dikoreksi pengguna: "aku mau itu jadi ace compact" → masuk `aturan.md`.
- Berikutnya: pengguna meninjau diff lalu commit; uji bagian 3–6 OCR v2 setelah PaddleOCR dipasang.

## 2026-10-30 claude-opus-5-5 - figur PPT vignette (di Vignate-2, bukan repo ini)
- Dikerjakan: figur kartu `gambar18` (proses YOLO), `gambar20` (inferensi), `gambar21` (orientasi →
  enhance → PaddleOCR → teks), `gambar22` (10 citra uji tanpa entete), `gambar23` (4 contoh ujung ke
  ujung, pengganti figur lama Times). Skrip di `Z:/GEMASTIK/Vignate-2/outputs/figures/_src/v2/`, rincian
  di `Z:/GEMASTIK/Vignate-2/HANDOFF.md` bagian A4. Angka deteksi dari prediksi ulang `yolo_2cls_v2.pt`
  (lihat `keputusan.md`). PPT selesai dibuat pengguna hari ini.
- Dikoreksi pengguna: tolak pita "kenapa" + bayangan + label fase di kartu ("design sebelumnya malah
  lebih rapi"); teks kecil di PPT → kanvas 13 in. Masuk `aturan.md` bagian Figur.
- Berikutnya: tidak ada untuk PPT. PaddleOCR belum dipasang (pengguna: "jangan dulu"), jadi orientasi +
  teks gambar21/23 masih dari run notebook lama.

## 2026-10-30 claude-opus-5-5 - hook agar memori selalu diperbarui
- Dikerjakan: `.claude/settings.json` + `.claude/hooks/memori.sh`. SessionStart menempel
  `status.md` + `aturan.md` ke konteks; UserPromptSubmit memotret keadaan file; Stop memblokir sekali
  kalau giliran itu mengubah file di luar `_memori/` tanpa menyentuh `_memori/`. Diuji di repo
  sementara (perubahan kotor, commit tanpa memori, commit dengan memori, nama file aneh).
- Dinilai: PaperBanana (github.com/dwzhu-pku/PaperBanana). Tidak dipasang, lihat `keputusan.md`.
- Dikoreksi pengguna: -
- Berikutnya: hook baru aktif di sesi Claude Code berikutnya yang dibuka dari root repo ini.

## 2026-10-30 claude-opus-5-5 - memori agen + CLAUDE.md
- Dikerjakan: `_memori/` (README, status, aturan, keputusan, log), `CLAUDE.md` (wajib baca dan
  perbarui memori), `AGENTS.md` (rujukan untuk Codex). Isi awal diambil dari README,
  RENCANA_LOMBA.md, `../KONTEKS_CLAUDE.md`, dan memori lokal Claude proyek ini.
- Dikoreksi pengguna: -
- Berikutnya: pekerjaan terbuka di `status.md`.

## 2026-10-30 claude-opus-5-5 - notebook vignette v2 masuk arsenal
- Dikerjakan: `2d58f25` (RENCANA_LOMBA.md, `.gitignore` + `_.Testing`, `viginte-train.ipynb` dipindah),
  `4c654db` (`arsenal/pipeline/ocr_end_to_end_v2.ipynb`), `d59ab4b` (entri PETA.md). Di-push ke
  `rapikan-struktur`.
- Dikoreksi pengguna: -
- Berikutnya: notebook belum lolos lint dan masih menulis output di folder kerja.
