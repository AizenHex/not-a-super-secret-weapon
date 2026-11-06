# Status repo

Terakhir diperbarui: 2026-11-06.

## Tujuan

Arsenal offline untuk final GEMASTIK XIX Penambangan Data: setup 2026-11-10, pengerjaan
2026-11-11 13.00–18.00 (5 jam, luring, tanpa internet dan tanpa AI), presentasi 2026-11-12.
Pengguna berangkat 2026-11-09. Prioritas: [`../RENCANA_TEKS.md`](../RENCANA_TEKS.md) (teks, rinci) dan [`../RENCANA_LOMBA.md`](../RENCANA_LOMBA.md) (bobot nilai, pohon lama).

**Info panitia (pengguna, 2026-11-06): data final 2026 (tuan rumah BINUS) dipastikan TEKS.** Bentuk
spesifik dan jebakannya belum diketahui. Pengguna minta repo dikembangkan agar siap untuk soal ML teks apa
pun; prioritas di bawah (deteksi, RLE, dll.) perlu disusun ulang ke teks, menunggu persetujuan rencana.
**Email panitia (2026-11-06):** tanpa internet; tiap tim dapat 1 PC panitia (i7-12700, RAM 32 GB, SSD 1 TB,
**RTX 3070 LHR 8 GB VRAM**). Akses dataset, training, dan submission **wajib di PC panitia**; laptop pribadi
hanya pendukung (tidak boleh menyentuh data). Akibatnya: arsenal + venv/wheelhouse + model pretrained harus
bisa dipasang di PC asing. **Hari setup (1 hari sebelum lomba) ada internet** (pengguna, 2026-11-06): pasang
via `setup.bat` + `download_model.bat` online di PC panitia (keduanya sudah ada dan sudah sering dipakai
pengguna; tidak perlu skrip baru); wheelhouse/flashdisk hanya cadangan. Model > 0,5 GB boleh dipakai. Belum
diketahui: OS, Python/driver NVIDIA terpasang, apakah PC tidak di-reset antara setup dan hari lomba. Torch di requirements = cu128 (butuh driver NVIDIA ≥ 570).
Persiapan teks berjalan lewat `PROMPT_FINAL_TEKS.md` (2026-11-06): Prompt A membuat `RENCANA_TEKS.md`
(belum ada), Prompt B mengerjakan butirnya satu per satu. `RENCANA_TEKS.md` jadi sumber daftar kerja teks.
Revisi prompt 2026-11-06: dua pilar diperbarui, `arsenal/` dan `contoh/text/` (soal teks baru, metode baru,
model kecil baru; notebook sendiri hanya bila loader/alur latih berubah, ganti satu fungsi = alternatif
berkomentar); peta memuat metode ML standar/niche/SOTA; subagent boleh untuk membaca.
**Prompt A sudah dijalankan 2026-11-06**: [`../RENCANA_TEKS.md`](../RENCANA_TEKS.md) = daftar kerja teks (butir
T1–T13 di atas garis potong, B1–B12 di bawah; STATUS tiap butir masih kosong). Keempat pertanyaan di atas file itu
sudah dijawab "Ya, semua" (venv lewat `setup.bat`, pip `emoji`/`ipywidgets`, unduh 4 model ± 4 GB, garis potong
sesudah T13). Berikutnya Prompt B: G0 = T1 + T4 bersamaan. T1 selesai 2026-11-06 (32/32 contoh/text + 13/13 arsenal teks LOLOS; STATUS di RENCANA_TEKS.md). Temuan terbuka: kernelspec `python3` pengguna menunjuk venv yang tidak ada, `bangun_contoh.bat` perlu `JUPYTER_PATH`/`ipykernel install --user` (menunggu keputusan pengguna).
**T4 selesai 2026-11-06**: 4 model ada di `pretrained/` (indobert_large, indobertweet, xlmr_base, e5_base; ± 2,9 GB),
dimuat offline; alt e5 di `klasifikasi_embedding` F1 0.8476. Blok KATALOG menunggu T13 (di STATUS T4).
**Venv `Z:/GEMASTIK/Vignate-GEMASTIK/.venv` dibuat T1 (2026-11-06)**, Python 3.13.15, requirements + emoji/ipywidgets terpasang; hasil uji Run All di STATUS T1 `RENCANA_TEKS.md`.
GPU laptop: RTX 5060 Laptop 8 GB.
Latihan terakhir pengguna: soal 2024 Golongan UKT di `Z:/GEMASTIK/GEMASTIK-2024` (`train.ipynb`, XGBoost,
F1 makro val 0,427; kelas ke-2 F1 0).

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
- `_laya_eda/` (Laya v1) dihapus pengguna (commit 3a375ec "laya v1 removed", 2026-11-06); yang dipakai
  `_laya_eda_v2/`. Data riil Laya sekarang dirujuk sebagai `../../HAND-WRITE/train.csv`.
- `_laya_eda_v2/` (2026-11-02, sesi cloud, branch `new-gen`): `VISI.md` = satu-satunya sumber target (extra
  E1–E12 di atas R1–R18; 8 temuan review sudah masuk). E1–E12 diimplementasikan (tab **Latih** dan **Submission**,
  14 halaman) dan R1–R18 punya uji. `python -m tests.semua` di venv cloud (versi persis requirements.txt, torch
  2.11.0 dari PyPI): lulus 47, gagal 0, dilewati 12. Semua uji memakai data **sintetis** (`tests/sintetis.py`);
  dengan data riil unggahan pengguna (`../_laya_eda/data/`, diabaikan git): lulus 57, gagal 0, dilewati 2.
  **Belum**: uji di venv Windows `Z:/GEMASTIK/Vignate-GEMASTIK/.venv` (paket
  pandera/feature-engine/skrub/autogluon belum dipasang di sana per 2026-11-02); R17 belum diukur di browser;
  penanda satuan campur + nol mencurigakan; tab Kosong/Encoding. Rincian di STATUS `RENCANA.md`.
- Laya v2 overhaul UMUM (2026-11-02): Temuan/baca/kamus umum, semua bisa diedit; uji 10 dataset di
  `_laya_eda/data/umum/` (diabaikan git). Semua uji: lulus 63, gagal 0, dilewati 2. Rincian STATUS E di `_laya_eda_v2/RENCANA.md`.
- T3 + T7 selesai (2026-11-06, belum di-commit): `arsenal/umum/baca_folder` bagian 12–16 (#14–#20) dan
  `arsenal/umum/simpan_muat` #15–#16 (validator submission vs sample), #17 (zip cadangan, bergeser dari #16),
  #18 (paket kumpul). Bukti dan blok PETA di STATUS T3/T7 `RENCANA_TEKS.md`; T13 yang menyalin ke `arsenal/PETA.md`.
- T11 a, b, d, e selesai (2026-11-06, belum di-commit): `teks/klasifikasi/tfidf` bagian 8-9 (NB-SVM), `teks/dasar/praproses`
  bagian 5 (train↔test), `tabular/ensembling` bagian 1b + 10 (F1 makro, rank average), `tabular/metrik` bagian 9 (temperature
  scaling + isotonic). **T11c (submission NER-BERT, GPU) belum.** Bukti dan blok PETA di STATUS T11 `RENCANA_TEKS.md`.
- `viginte-train.ipynb` dipindah dari root ke `_.Testing/` (diabaikan git).
- PPT vignette (penyisihan) selesai 2026-10-30. Figur + skripnya di `Z:/GEMASTIK/Vignate-2/outputs/figures/`
  (di luar repo, bukan git); catatan lengkap `Z:/GEMASTIK/Vignate-2/HANDOFF.md`.

## Pekerjaan terbuka

Teks (utama sejak 2026-11-06): butir T1–T13 di `RENCANA_TEKS.md` bagian 4, mulai T1 (lingkungan + uji Run All).
T8 selesai 2026-11-06 (`contoh/text/teks_tabular.ipynb`; sisa: uji muat di `ensemble_oof` setelah T6, `e5_base` belum diukur;
blok KATALOG/PANDUAN/SOURCES ada di STATUS T8 untuk T13).
Daftar lama di bawah (vision-sentris) turun prioritas; validator (2), zip (3), log percobaan (8) masuk T3/T7/T6.

1. Deteksi objek sampai `submission.csv` + mAP lokal (pembaca YOLO generik, split val, format
   per-kotak/PredictionString, gambar tanpa deteksi tetap punya baris).
2. ~~Validator submission universal~~ selesai (T3, `simpan_muat#15–#16`).
3. Paket kumpul: zip selesai (T7, `simpan_muat#18`); potongan grafik → pptx belum.
4. Metrik mAP / mIoU / Dice / CER.
5. RLE segmentasi multi-kelas + resize balik.
6. Pembaca dataset Hugging Face parquet.
7. Deret waktu ke arsenal.
8. Log percobaan + latihan 5 jam pakai stopwatch.
- Tambahan: uji bagian 3–6 `pipeline/ocr_end_to_end_v2` setelah PaddleOCR dipasang (pengguna yang
  memutuskan kapan). Kalau pengguna setuju, hapus `pretrained/`/data dobel di `Z:/GEMASTIK/_arsip/`.
