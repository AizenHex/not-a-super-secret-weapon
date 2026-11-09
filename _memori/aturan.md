# Aturan kerja

Koreksi terbaru pengguna di chat menang atas isi file ini; setelah itu perbarui file ini.

## Komunikasi

- Bahasa Indonesia. Jawab inti dulu (1–3 kalimat), detail hanya kalau diminta. (2026-10-24)
- Pengguna paham konsep ML (latar OSN Informatika, C++) tapi tidak hafal sintaks Python. Notebook harus
  lengkap dan bisa Run All; yang diubah cukup sel Konfigurasi / baris `# UBAH`. (2026-10-23)
- "Mode latihan" hanya kalau pengguna menyalakannya: saat itu jangan menulis kode, hanya arahan dan
  review. Default: mode biasa. Ragu → tanya satu baris. (2026-10-23)
- Jangan menulis tebakan sebagai fakta (pola dari nama file, isi data yang tidak dibaca pasti). (2026-10-30)
- Sebelum mengusulkan alat/skrip baru, baca dulu yang sudah ada (`setup.bat`, `download_model.bat`,
  `_alat/`, arsenal) dan usulkan hanya celahnya. Pengguna kesal: "repo sekarang kan udh berkali-kali aku
  pakai. kamu gak bahkan ngecek setup.bat ama download model.bat kah?" (2026-11-06)
- Saat soal belum diketahui, rencana/prompt dimulai dari peta bidang yang lengkap, lalu dicocokkan ke repo,
  baru diprioritaskan. Jangan menyempitkan ke tebakan soal di awal. Pengguna: "kenapa gak suruh dia bikin
  map ML dalam bidang data text secara keseluruhan? daripada mengarahkannya kepada satu mosi" (2026-11-06)
- Subagent (alat Agent) boleh dipakai untuk membaca dan memadatkan bahan besar; klaim yang menentukan
  keputusan tetap dicek sendiri. Pengguna: "boleh pakai agent untuk memadatkan kan?" (2026-11-06)
- Persiapan teks memperbarui DUA pilar: `arsenal/` (potongan) dan `contoh/text/` (notebook end-to-end).
  Model/metode baru yang cukup ganti satu fungsi atau beberapa baris (loader dan alur latih tetap) →
  alternatif ber-`#` dengan komentar di notebook yang sama; yang loader/cara muat/alur latihnya beda →
  notebook sendiri. Pengguna: "model baru yang cara loadnya berbeda (butuh notebook terpisah ...)" lalu
  "kalo misalnya cuma perlu ganti 1 fungsi, gak perlu ganti loader, dll. itu boleh 1 note, cuma nanti
  ber komentar" (2026-11-06)
- Sesi pekerja (Prompt B, satu butir `RENCANA_TEKS.md` per sesi) dan subagent baca memakai Sonnet effort high,
  bukan Opus; Opus hanya untuk perencana. Saat membuat sesi, isi `model` secara eksplisit (kalau kosong, sesi
  mewarisi Opus). Pengguna: "pakai sonnet high bisa itu, tapi kenapa malah pake opus high?" (2026-11-06)
- Peta metode wajib memuat metode ML standar, niche, dan SOTA (mis. hill climbing) serta model kecil;
  LLM one/zero-shot bukan fokus. Pengguna: "metode ML dan model kecil, bukan model one shot kayak LLM"
  (2026-11-06)
- Aturan AI saat lomba dikoreksi pengguna (2026-11-07): AI **boleh** dipakai untuk membantu menulis kode untuk
  hal tak terduga, bukan "tanpa AI" mutlak. Arsenal tetap utama; asisten kode lokal hanya penambal. Dasarnya
  ucapan pengguna di chat, belum ada kutipan guidebook.

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

- **Laya dibekukan (pengguna, 2026-11-06: "capek sama laya", tidak percaya hasil maupun prosesnya).** Jangan
  memperbaiki, menyarankan, atau memakai Laya v2 (Streamlit) untuk final; EDA/pembersihan final lewat notebook
  arsenal. Pengguna sendiri memulai Laya v3 (2026-11-06, lihat di bawah); v3 tidak boleh jadi jalur wajib final.
- **Visi Laya baru (pengguna, 2026-11-06)**: bukan EDA biasa ("kalau gitu langsung pandas"), tapi workbench
  persiapan data: loader banyak file sekaligus (multi-tabel/relasional, mis. GDOC `train/multitabel.ipynb` +
  folder `riwayat-medis/`), data jenis apa pun (teks, tabular, multi-level), bisa diatur sesuai keinginan pengguna.
  Bukan alat yang dioptimasi untuk satu dataset (pelajaran UKT).
- Jangan menyajikan ukuran uji (AppTest, tanpa browser) sebagai waktu yang dirasakan pengguna; yang dirasakan
  pengguna di browser yang benar (dikoreksi 2026-11-06: "0,1–0,3 s ... bullshit").
- **Laya v3 = aplikasi desktop `.exe`, bukan web, bukan Streamlit** (pengguna, 2026-11-06: "buatin jadi .exe aku
  gak mau jadi web"; menggantikan aturan lama "Flask + web"). `_laya_eda_v3/VISI.md` = satu-satunya sumber tujuan,
  fitur (P0/P1/P2), requirement, dan uji terima v3. Pengguna ingin `.exe` tinggal ditempel di repo lalu dipakai.
- Pengguna tidak tahu fitur apa yang terbaik untuknya dan meminta agen (dipanggil "max") menyusunnya: beri
  rekomendasi tegas dengan alasan dan bukti data, bukan daftar pilihan tanpa saran. (2026-11-06)
- **Laya v3 = "tank"** (pengguna, 2026-11-06: "aku harus menyiapkan tank sekalipun meski lawan-lawanku hanya membawa
  kapak batu"): fitur sebanyak mungkin dalam arti luas, bukan hanya EDA: latih model (mis. AutoGluon), submission,
  presentasi, kesiapan PC lomba, dokumentasi offline. Riset alat lain (bukan hanya ide sendiri) sebelum menyusun fitur.
- **UI Laya v3 harus jelas dan visual** (pengguna, 2026-11-07, marah: "asli yang kamu buat itu membingungkan"):
  pakai skill `impeccable` untuk kerja UI; tiap perubahan UI diverifikasi dengan screenshot tema gelap DAN terang
  (Windows dark mode); warna HTML lewat `pahami.W()`, bukan hex mati; alur langkah selalu terlihat (bilah alur).
  Fitur yang ada di v2 (outlier, split/validasi, dsb.) harus ikut ke v3 sebelum fitur baru. Saat menambah fitur
  preprocessing, pikirkan juga fitur yang belum diminta (lihat VISI 8b).
- **UI Laya v3 harus mudah dinavigasi dan tiap layar jelas menampilkan apa** (pengguna, 2026-11-06: "ui/ux ini jelek
  ... susah di navigasi ... profiling nya gak jelas nampilin apa, chaos"). Pola tetap: kiri = tabel + resep, tengah =
  daftar yang bisa dipilih (Data/Profil/Temuan), kanan = rincian dari yang dipilih. Jangan menaruh dinding teks
  polos; pakai tabel ringkas + rincian per butir. Sebelum menyebut UI selesai, tangkap layar offscreen
  (`QT_QPA_FONTDIR=C:/Windows/Fonts`) dengan data riil dan periksa sendiri.

- Laya v2 (`_laya_eda_v2/`): `VISI.md` = satu-satunya sumber tujuan, requirement (R1–R18), prinsip, library,
  dan uji terima. `RENCANA.md` hanya status, arsitektur yang jalan, dan bukti uji; jangan menulis prinsip
  atau pilihan library di sana. (2026-11-02)
- Laya v2 punya fitur **extra** di luar EDA: sklearn Pipeline + joblib, split/k-fold, pandera, latih cepat,
  tab Submission, metrik, log, ensemble, mode aman, figur, kelas timpang (VISI.md bagian 0, E1–E12). Kata pengguna: kedudukannya **lebih tinggi dari R1–R18**;
  bila bertentangan, extra menang. (2026-11-02)

- Jangan pernah mengganti nama kolom asli di file hasil (tidak ada `_angka` / `__angka`): kolom angka
  diisi angka hasil baca di kolom yang sama. Kolom baru (fitur) boleh, kolom asli tetap. Pengguna:
  "niatnya untuk mempermudah EDA, ini malah dipersulit". (2026-10-31)
- Tiga tab terakhir selalu Split data → Kode EDA → Ekspor (urutan yang pengguna kenal). (2026-10-31)
- Kode EDA = semua yang dilakukan ke data dan harus diulang saat inferensi, siap tempel: langkah isi
  data (angka, isi kosong, fitur, buang kolom) untuk train + test dengan nilai dari train; buang baris
  dan ganti label hanya train. train_siap di aplikasi harus = hasil kode itu. (2026-10-31)
- Wajib ada pilihan isi angka kosong (median/mean/modus/0, per kolom). (2026-10-31)
- Kecepatan penting: klik widget tidak boleh menghitung ulang hal berat (pakai `st.cache_data` dengan
  kunci `ctx["data_key"]`, unduhan CSV lewat callable). Ukur rerun mode baris pada `../../HAND-WRITE/train.csv`
  (30k baris) sesudah mengubah tab. (2026-10-31)
- Laya v2 harus **umum** untuk data tabular apa pun, bukan alat `Golongan UKT`; semua yang dibuatnya bisa diedit
  (usulan Temuan, langkah, kode langkah, kamus kata domain). Pengguna: "terlalu fokus ke data train golongan UKT ...
  bukan software yang general ... aku bahkan gak bisa mengeditnya ... aku butuh overhaul". Kata domain (Indonesia/UKT)
  hanya sebagai preset kamus proyek, bukan tertanam di kode. Uji di banyak dataset (`tests/test_umum.py`). (2026-11-02)
- Notebook arsenal/contoh harus bisa dipakai untuk domain apa pun: nama variabel dan komentar menjelaskan pola
  (tabel utama, 1-ke-1, long per sampel, riwayat per entitas + filter waktu), bukan domain demo. Pengguna soal
  `arsenal/tabular/multitabel` (data bank): "terlalu menjerumus ke perhitungan uang dan gak umum, jadinya aku bingung". (2026-11-06)
- Laya v2: uji yang lulus di data sintetis belum bukti; jalankan juga di data riil (`_laya_eda/data/train.csv`, tidak di-commit) sebelum menyebut selesai. (2026-11-02)
- Setiap kolom yang tampil di tabel harus punya arti di `kamus.py` (nilai apa, dari apa); tabel
  baru pakai `kamus.show_table`, bukan `st.dataframe`. Pengguna bingung arti "nilai". (2026-10-31)

## Lingkungan

- Venv yang dicari alat: `Z:\GEMASTIK\Vignate-GEMASTIK\.venv` (Python 3.13), dibuat `setup.bat`
  (T1, 2026-11-06). Python sistem (cadangan):
  `C:\Users\EYZARIM\AppData\Local\Programs\Python\Python313\python.exe`. Pretrained offline:
  `Z:\GEMASTIK\Vignate-GEMASTIK\pretrained`. GPU laptop RTX 5060 Laptop 8 GB; PC lomba RTX 3070 8 GB.
  `.venv` dibuat 2026-11-06 lewat `setup.bat`; kernelspec Jupyter `python3` pengguna menunjuk venv itu (tanpa
  `JUPYTER_PATH`).
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

- Koreksi pengguna 2026-11-07 (figur pipeline GDOC):
  - **Baca bagian Figur ini sebelum membuat figur apa pun.** Kanvas selalu ~13×5,4 in (atau 16:9); **jangan
    pita pipih/menyamping** (mis. 13×3,2): di PPT tidak bisa di-scale dengan benar. Huruf besar (judul ≥ 18).
  - Istilah teknis ditulis seperti di kode, jangan diterjemahkan ("patience 2", bukan "sabar 2"; max_length, epoch).
  - Keterangan sedikit dan kecil; satu baris catatan per kotak paling banyak. Figur pipeline hanya pipeline:
    jangan tambah jalur contoh/ilustrasi (mis. "contoh bersihkan()"); rincian data masuk figur "what we see".
  - Pipeline wajib menyebut input yang benar-benar dipakai: kolom mana yang dimuat (mis. hanya konten_berita →
    sentimen_berita) dan bahwa kolom lain diabaikan; keluaran submission sebut kolomnya (id, label).
  - Figur hasil model: pakai output notebook/file yang sudah ada; jangan menjalankan ulang model/inferensi tanpa
    diminta (limit pengguna, 2026-11-07).
  - Pengecualian kanvas: gambar pengganti foto/kolase di slide (mis. kolase kartu berita) boleh persegi, ikuti
    ukuran tempatnya di slide.

## Git

- Commit/push hanya kalau pengguna memintanya di sesi itu. Kalau diizinkan ("kalau ada perubahan kode,
  silahkan di push"), commit perubahan kode + memori bersama lalu push ke branch kerja. (2026-10-30)
- Jangan commit `outputs/`, data, atau hasil Run All (sudah diatur `.gitignore`).
