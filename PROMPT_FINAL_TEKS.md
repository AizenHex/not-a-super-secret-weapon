# Prompt persiapan final teks

Dua prompt yang ditempel ke Claude Code dari root repo ini. Jalankan A sekali (Opus effort xhigh/max).
Setelah `RENCANA_TEKS.md` jadi dan kamu setujui, jalankan B per butir (Sonnet effort high, satu sesi
per butir; butir di grup paralel yang sama boleh jalan bersamaan).

Revisi 2026-11-06 (pengguna): cakupan = dua pilar, `arsenal/` dan `contoh/text/`; keduanya ditambah soal
teks baru, metode baru, dan model baru yang cara muatnya beda (notebook sendiri). Kedua prompt boleh
memakai subagent (alat Agent) untuk membaca dan memadatkan bahan besar.
Revisi kedua (pengguna, 2026-11-06): notebook terpisah hanya bila loader/alur latih berubah; ganti satu
fungsi cukup jadi alternatif berkomentar di notebook yang sama. Peta wajib memuat metode ML standar, niche,
dan SOTA (mis. hill climbing) serta model kecil; LLM one/zero-shot bukan fokus.

---

## Prompt A: perencana (Opus xhigh/max)

```
Kamu perencana. Tugasmu membuat PETA KERJA, bukan menulis kode. Hasil akhirmu adalah file
RENCANA_TEKS.md di root repo yang akan dikerjakan agen lain (Sonnet effort high) butir demi butir,
tanpa melihat percakapan ini.

## Fakta lomba (dari pengguna, 2026-11-06)
- Final GEMASTIK XIX Divisi Data Mining, tuan rumah BINUS. Setup 2026-11-10 (ADA internet, 1 hari),
  pengerjaan 2026-11-11 13.00–18.00 (5 jam, TANPA internet, tanpa AI), presentasi 2026-11-12.
  Pengguna berangkat 2026-11-09, jadi semua persiapan selesai 2026-11-08 malam.
- Data final DIPASTIKAN TEKS. Bentuk tugas, bahasa, metrik, dan jebakannya belum diketahui.
- Tiap tim 1 PC panitia: i7-12700, RAM 32 GB, SSD 1 TB, RTX 3070 LHR 8 GB VRAM. Akses dataset, training,
  dan submission WAJIB di PC itu; laptop pribadi hanya pendukung (tidak boleh menyentuh data).
  Model pretrained > 0,5 GB boleh dipakai.
- Bobot nilai: penyisihan 20% (terkunci), leaderboard 30%, ketepatan metode 25% (kode + dokumentasi +
  presentasi), presentasi 25%; syarat kumpul (zip dokumentasi + kode + PPT) = gugur bila tidak ada.
- Riwayat soal: 2021 klasifikasi (Macro-F1), 2023 log jaringan kelas langka, 2024 tabular Golongan UKT
  berisi isian teks bebas yang harus diurai (latihan pengguna: Z:/GEMASTIK/GEMASTIK-2024/train.ipynb,
  F1 makro val 0,427, satu kelas F1 0), 2025 OCR aksara Jawa dengan data tidak lengkap. Panitia suka
  data kotor dan kejutan.
- Pengguna paham konsep ML tapi tidak hafal sintaks Python: yang dia ubah saat lomba hanya sel
  Konfigurasi / baris `# UBAH` / `# GANTI saat lomba`, lalu Run All.

## Sasaran (dari pengguna, 2026-11-06)
Repo punya dua pilar yang dipakai saat lomba, dan KEDUANYA diperbarui:
- arsenal/ = potongan siap tempel per tugas (sel pendek, lolos _alat/lint.py, peta di arsenal/PETA.md).
- contoh/text/ = notebook utuh end-to-end per kasus (Import → Konfigurasi → Baca → EDA → Split → Model →
  Evaluasi → Latih ulang + submission.csv; Run All langsung jadi). Alternatif model ditulis ber-`#` dan
  dicatat dengan skor demo di contoh/KATALOG.md; notebook dipilih lewat tabel di contoh/PANDUAN.md.
Yang ditambahkan ke keduanya:
1. Lebih banyak soal teks: kasus, bentuk data, dan tugas yang belum punya notebook end-to-end.
2. Metode baru, termasuk yang niche dan SOTA (praproses, fitur, validasi, pelatihan, loss, pencarian
   hiperparameter, ensemble seperti hill climbing, kalibrasi/ambang, dll.).
3. Model baru. Fokus model kecil yang bisa dilatih/fine-tune di 3070 8 GB. LLM yang dipakai one/zero-shot
   lewat prompt BUKAN fokus: cukup tercatat satu baris di peta.
Aturan pemisah untuk model DAN metode baru:
- Satu notebook, alternatif berkomentar: kalau yang berubah cukup path checkpoint, satu fungsi, atau
  beberapa baris, sementara loader data, cara muat model, dan alur latih tetap (mis. encoder BERT lain,
  loss lain, cara ensemble lain). Alternatif ditulis ber-`#` dengan komentar singkat kapan dipakai,
  + baris KATALOG (+ entri _alat/unduh_model.py bila model). BUKAN notebook baru.
- Notebook contoh SENDIRI: kalau loader, cara muat, alur latih, atau inferensinya harus diganti (kelas
  model lain, library lain, kuantisasi, adapter LoRA/PEFT, format file lain, loop latih lain, dll.),
  + potongan arsenal bila bagiannya berguna ditempel ke notebook lain, entri unduh_model.py, dan
  requirements.txt bila butuh library baru.
Pilih model dan metode dari peta langkah 0 (sumbu "keluarga model" dan "metode ML"), bukan dari daftar
yang sudah ada di repo.
Jumlah soal/metode/model baru ditentukan dari celah peta × peluang muncul ÷ usaha, bukan target angka.

## Cara membaca (subagent boleh dan disarankan)
Bahan wajib ±1 MB (md ±190 KB, notebook ±780 KB); tidak muat dibaca utuh dalam satu konteks.
- Baca SENDIRI: semua file di butir 1 daftar bacaan, contoh/PANDUAN.md, bagian teks contoh/KATALOG.md,
  _alat/unduh_model.py.
- Sisanya bagi ke subagent (alat Agent, tipe general-purpose; jangan Explore karena ia hanya membaca
  potongan), beberapa berjalan paralel, misalnya: contoh/text (dibagi dua), arsenal/teks, arsenal
  jembatan+umum+tabular, data latihan + SOURCES, _laya_eda_v2 + kamus + RISET. Subagent HANYA membaca:
  tidak menulis file dan tidak menjalankan notebook. Isi tiap notebook dibaca penuh (sel kode + markdown;
  output boleh dilewati).
- Laporan subagent per notebook: path; tugas + bentuk data demo (set, kolom, jumlah baris); model/metode
  aktif dan alternatif ber-`#`; cara muat model (kelas/library/fungsi); baris yang diubah saat lomba;
  metrik + skor demo (dari KATALOG bila ada); status uji (LOLOS/GAGAL/belum + sumbernya); bolong yang
  terlihat (bagian hilang, nama kolom tertanam, tidak sampai submission, dll.).
  Per set data: nama file, kolom, jumlah baris, format/encoding, bahasa, 2 baris contoh, ada kunci
  jawaban atau tidak.
- Tulis ringkasan ke RENCANA_TEKS.md bagian "Bahan yang sudah dibaca" segera setelah tiap laporan datang;
  file itu jadi memori kerja kalau konteks dipadatkan.
- Klaim subagent yang menentukan butir (mis. "notebook X tidak punya Y") dicek sendiri dengan membuka sel
  itu sebelum ditulis sebagai fakta.

## Daftar bacaan (isi, bukan hanya grep atau 3 baris pertama; boleh lewat subagent seperti di atas)
1. CLAUDE.md, _memori/status.md, _memori/aturan.md (aturan kode + koreksi pengguna), _memori/keputusan.md,
   _memori/log.md bagian atas, README.md, RENCANA_LOMBA.md, arsenal/PETA.md.
2. Semua yang sudah ada untuk teks: arsenal/teks/**, arsenal/jembatan/teks_ke_vektor.ipynb +
   vektor_ke_model.ipynb, arsenal/umum/teks_regex.ipynb + tabel_bersih.ipynb + simpan_muat.ipynb,
   arsenal/tabular/metrik + imbalanced + ensembling (yang relevan untuk teks), contoh/text/ (32 notebook),
   contoh/KATALOG.md + PANDUAN.md + RISET.md, _laya_eda_v2/VISI.md + RENCANA.md (bagian Latih/Submission/
   metrik/kelas timpang), kamus/03_jebakan.md.
3. Alat yang sudah ada dan TIDAK boleh dibuat ulang: setup.bat (venv + pip online/wheelhouse + cek import
   & GPU), download_model.bat + _alat/unduh_model.py (daftar model HF), _alat/lint.py,
   _alat/uji_arsenal.py, _alat/bangun_contoh.bat (Run All contoh; LOLOS/GAGAL di
   _alat/log/hasil_build.txt), _alat/unduh_data_tambahan.py (pola unduh + rapikan data demo),
   requirements.txt. Catatan: generator contoh/KATALOG.md tidak ada di _alat/ (dicek 2026-11-06).
   Gaya contoh berbeda dari arsenal (path string relatif `../../../`, `# GANTI saat lomba`); cek sendiri
   apakah contoh/text lolos lint.py, lalu tetapkan gaya untuk notebook contoh baru. Jangan menebak.
4. Data latihan teks: ../data_latihan/text/ (16 set: absa, arxiv, classification, classification_en,
   english, generasi, multilabel, ner, nusax, pair, pos, qa, regression, sentiment, topik_waktu,
   ujaran_kebencian) dan ../data_latihan/SOURCES_*.md. Model: ../pretrained/.
Tulis di RENCANA_TEKS.md bagian "Bahan yang sudah dibaca" (file + ringkasan 1 baris + status teruji/belum).
Kalau sesuatu tidak sempat dibaca, tulis "BELUM DIBACA", jangan menebak isinya.

## Yang harus kamu hasilkan di RENCANA_TEKS.md
0. PETA ML TEKS SECARA KESELURUHAN, disusun dari pengetahuanmu tentang bidangnya, BUKAN dari isi repo dan
   bukan dari tebakan soal. Jangan menyempitkan ke satu jenis soal: soal final belum diketahui, jadi
   petanya harus lengkap dulu, baru dipangkas di langkah berikutnya. Minimal sumbu berikut, masing-masing
   dirinci sampai tingkat yang bisa dijadikan sel notebook:
   - Bentuk masukan/data: satu kolom, banyak kolom teks, teks + metadata tabular, pasangan/triplet, dokumen
     panjang, token per baris (CoNLL), percakapan/thread, teks berwaktu, teks semi-terstruktur (isian bebas
     yang harus diurai jadi angka/kategori), format file dan encoding.
   - Tugas: semua keluarga tugas supervised (klasifikasi biner/multikelas/multilabel/hierarkis/ordinal,
     regresi, pelabelan urutan/NER/span, pasangan kalimat, ranking/retrieval, QA ekstraktif), unsupervised
     (topik, clustering, anomali, deduplikasi), label sedikit (zero/few-shot, semi-supervised, weak
     supervision, active/self-labeling), generatif (ringkasan, parafrase, generasi), dan tugas lain yang
     menurutmu ada di bidang ini.
   - Representasi: aturan/regex/kamus, BoW/TF-IDF kata & karakter, hashing, embedding statis, embedding
     kalimat, transformer fine-tune, LLM kecil; fitur buatan (panjang, tanda baca, emoji, leksikon).
   - Keluarga model menurut cara muat/latih/inferensi (sumbu ini yang menentukan notebook terpisah):
     mis. encoder + head klasifikasi, token classification, seq2seq kecil, bi-encoder kalimat (pakai &
     fine-tune), cross-encoder/reranker, few-shot berbasis kontras, embedding statis, model klasik non-HF
     (CRF, dll.), head kustom (teks + tabular, ordinal), LLM kecil + LoRA, model terkuantisasi atau format
     lain; tambah yang lain menurutmu. LLM one/zero-shot lewat prompt cukup satu baris. Per keluarga:
     kandidat checkpoint (Indonesia, multibahasa, Inggris), ukuran unduhan, perkiraan VRAM latih/inferensi
     di 8 GB (tandai "dugaan" sampai diukur), library yang dibutuhkan.
   - Metode ML, dari yang standar sampai yang niche dan SOTA (metode + model kecil, bukan LLM one-shot).
     Rinci per tahap, contoh (bukan batas):
     model klasik (linear, NB, NB-SVM, SVM, boosting di atas fitur/embedding);
     fine-tune (truncation kepala+ekor, fp16, grad accumulation, layer-wise LR decay, re-init lapisan atas,
     pooling lain, multi-sample dropout, R-Drop, FGM/AWP, EMA/SWA, lanjut pretraining MLM di teks soal,
     k-fold); loss (focal, class-balanced, label smoothing, loss ordinal);
     data (pseudo-label, self-training, distilasi, augmentasi);
     pencarian hiperparameter (Optuna/TPE, random search, successive halving);
     ensemble (rata-rata, rank average, bobot lewat hill climbing pada OOF, ensemble selection Caruana,
     stacking/blending);
     pasca-proses (kalibrasi, ambang per kelas, optimasi ambang langsung ke metrik, aturan).
     Tambahkan yang kamu tahu dari paper dan solusi kompetisi teks. Per metode tulis: tingkat
     (standar/niche/SOTA), kapan membantu, biaya waktu di 3070 (dugaan), dan masuk mana menurut aturan
     pemisah (alternatif berkomentar di notebook yang ada, atau notebook sendiri).
   - Masalah data & jebakan: kelas timpang, duplikat & duplikat beda label, bocor train↔test, label
     berisik, pergeseran distribusi, bahasa campur/slang/daerah/typo/emoji, teks sangat panjang/kosong,
     PII/noise, format label/submission aneh, dan yang lain menurutmu.
   - Evaluasi: semua metrik yang mungkin dipakai panitia (akurasi, F1 mikro/makro/berbobot, QWK, RMSE/MAE,
     F1 entitas/span, MAP@k/NDCG/MRR, ROUGE/BLEU, log loss, AUC) + skema validasi (stratified, group,
     waktu, adversarial validation) + analisis kesalahan.
   - Penjelasan & presentasi: kata khas per kelas, contoh salah, atribusi, figur yang meyakinkan juri.
   - Kendala eksekusi: 5 jam, offline, RTX 3070 8 GB, RAM 32 GB (ukuran model, batch, waktu per epoch).
   Setiap sel peta diberi kode (mis. P.tugas.3) supaya bisa dirujuk butir kerja.
1. Peta cakupan repo: peta langkah 0 dicocokkan ke repo, dengan DUA kolom per sel: arsenal (file:sel) dan
   contoh/text (notebook:sel + skor demo dari KATALOG). Isi tiap kolom: sudah ada / ada tapi bolong /
   belum ada, dan teruji di data apa. Ini menunjukkan celah secara utuh, bukan hanya celah yang terpikir
   lebih dulu.
2. Prioritas: baru di sini peta diurutkan menurut peluang muncul di final (dengan alasan dari riwayat
   soal dan fakta lomba) × dampak ke nilai. Sel berpeluang kecil tetap tercatat di peta, hanya turun
   ke bawah garis potong.
3. Jalur 5 jam (runbook): menit ke berapa melakukan apa di PC panitia, dari salin data sampai submission
   pertama (target baseline terunggah ≤ 45 menit), upgrade (embedding, fine-tune IndoBERT/model besar
   di 3070 8 GB, ensemble), pemilihan submission final, dan bahan slide. Sebut notebook contoh dan
   potongan arsenal mana yang dibuka di tiap langkah.
4. Daftar butir kerja T1, T2, ... diturunkan dari celah di peta cakupan. Setiap butir WAJIB punya:
   - Tujuan (1–2 kalimat), kode sel peta yang ditutupnya, pilar (arsenal / contoh / keduanya), dan alasan
     nilainya (leaderboard / metode / presentasi / syarat gugur).
   - File yang dibuat/diubah (path persis) dan file yang dipakai ulang (path:sel).
   - Butir contoh baru: set data demo yang dipakai. Kalau tidak ada set yang cocok di ../data_latihan/text/,
     butir itu (atau butir data terpisah sebelumnya) mengunduh dan merapikan set baru dengan pola
     _alat/unduh_data_tambahan.py dan mencatat sumbernya di ../data_latihan/SOURCES_text.md. Internet
     hanya pasti ada sampai 2026-11-08.
   - Butir model baru: id checkpoint HF, ukuran unduhan, entri _alat/unduh_model.py, library tambahan, dan
     pengukuran VRAM puncak + waktu per epoch di GPU laptop sebagai bagian uji terima, dengan batas yang
     harus dipenuhi agar muat di RTX 3070 8 GB.
   - Uji terima: perintah persis pada set di ../data_latihan/text/ yang disebut namanya, dan hasil yang
     harus terlihat. Arsenal: `python _alat/lint.py <file>` + `_alat/uji_arsenal.py ... --set ...`.
     Contoh: `_alat\bangun_contoh.bat text\<nama>` → LOLOS di _alat/log/hasil_build.txt, skor tercetak,
     submission.csv lolos validator, waktu < N menit. Minimal satu uji pada data yang BENTUKNYA BEDA dari
     data demo notebook itu (kolom lain nama, format lain), karena soal final belum diketahui.
   - Ketergantungan (butir mana harus selesai dulu) dan grup paralel (butir dalam satu grup TIDAK boleh
     menyentuh file yang sama). File bersama (contoh/KATALOG.md, contoh/PANDUAN.md, arsenal/PETA.md,
     _alat/unduh_model.py, requirements.txt, SOURCES_text.md): tetapkan satu cara dan tulis di atas
     daftar butir, mis. butir integrasi per grup, atau tiap butir hanya menambah blok miliknya dan membaca
     ulang file tepat sebelum mengedit.
   - Ukuran (S ≤ 1 jam, M ≤ 3 jam, L > 3 jam) dan effort agen yang cocok (medium untuk potong/rapikan/uji,
     high untuk desain alur atau logika baru).
   - Kolom STATUS kosong (diisi pekerja: belum / dikerjakan / selesai + bukti).
   Urutkan menurut (nilai × peluang muncul) ÷ usaha. Tandai garis potong: butir di bawah garis hanya
   dikerjakan kalau waktu sisa. Total butir di atas garis harus realistis selesai 2026-11-08, termasuk
   waktu unduh model/data dan Run All contoh.
5. Cek silang (bukan batas, hanya agar tidak terlewat; putuskan sendiri, tulis alasannya bila ditolak):
   pembaca data teks format apa pun; pemindai jebakan teks yang hasilnya jadi bahan slide; satu alur
   end-to-end sampai submission.csv yang divalidasi terhadap sample_submission; metrik teks + ambang per
   kelas; model pretrained tambahan di _alat/unduh_model.py yang muat di 8 GB; total ukuran unduhan
   ../pretrained sesudah penambahan; pengukuran waktu fine-tune di GPU laptop sebagai patokan; latihan
   5 jam pakai stopwatch pada set yang belum pernah dilihat; paket kumpul (zip dokumentasi + kode + PPT);
   setiap contoh baru masuk tabel tanda-soal di contoh/PANDUAN.md.
6. Perbarui RENCANA_LOMBA.md (pohon pekerjaan dan urutan kerja diarahkan ke teks; butir vision/deteksi/RLE
   diturunkan, jangan dihapus diam-diam) dan _memori/ (log.md, status.md, keputusan.md untuk setiap
   pilihan desain beserta alasannya).

## Batasan
- JANGAN menulis kode atau mengubah notebook. Hanya RENCANA_TEKS.md, RENCANA_LOMBA.md, _memori/.
- Kamu tidak menjalankan pengukuran. Angka VRAM/waktu/ukuran ditulis "dugaan" (sumber: kartu model atau
  hitungan), lalu pengukurannya dijadikan uji terima butir model.
- Rancangan harus lolos aturan kode di _memori/aturan.md dan _alat/lint.py (tanpa try/except, assert,
  comprehension, ternary kecuali DEVICE, class selain nn.Module/Dataset, cabang `if METHOD`/`if MODEL`;
  fungsi ≤ 20 baris, sel ≤ 30 baris; output ke ../../outputs/<kategori>/<nama>; arsenal dan contoh =
  .ipynb tanpa output). Kalau sebuah butir butuh pengecualian aturan, tulis jelas dan tandai
  "PERLU PERSETUJUAN".
- Library baru boleh (hari setup ada internet) asal masuk requirements.txt dan alasan ditulis; utamakan
  yang sudah terpasang. Pemasangan pip saat bekerja tetap minta izin pengguna (aturan.md).
- Pakai ulang yang sudah ada; butir baru hanya untuk celah nyata. Tidak ada aplikasi/GUI baru kecuali
  kamu bisa menunjukkan alasannya lebih kuat dari notebook.
- Bahasa Indonesia, singkat, fakta. Tebakan ditandai "dugaan".
- Kalau ada keputusan yang benar-benar milik pengguna (mis. memakai Laya v2 di PC lomba atau tidak),
  kumpulkan di bagian "Pertanyaan untuk pengguna" di atas RENCANA_TEKS.md, lalu tanyakan di akhir.
- Jangan commit/push.

Di akhir: ringkas di chat (5–10 baris) jumlah sel peta (metode per tingkat standar/niche/SOTA), berapa
yang sudah tertutup repo (arsenal /
contoh), jumlah butir (berapa contoh baru, potongan arsenal baru, model baru), garis potong, perkiraan
total jam, tambahan ukuran unduhan, dan pertanyaan yang perlu dijawab pengguna.
```

---

## Prompt B: pekerja (Sonnet high), satu butir per sesi

Ganti `T?` dengan nomor butir.

```
Kerjakan butir T? di RENCANA_TEKS.md. Hanya butir itu.

1. Baca CLAUDE.md, _memori/status.md, _memori/aturan.md, lalu RENCANA_TEKS.md bagian fakta lomba, peta,
   cara menangani file bersama, dan butir T? lengkap. Baca SEMUA file yang disebut di butir itu (isi
   penuh, bukan grep) sebelum menulis apa pun. Subagent (alat Agent) boleh dipakai untuk membaca atau
   mencari; beban GPU tidak pernah dijalankan paralel, termasuk oleh subagent.
2. Cek ketergantungan: kalau butir yang harus selesai dulu belum berstatus "selesai", berhenti dan
   laporkan.
3. Isi STATUS butir T? = "dikerjakan (<tanggal>)".
4. Kerjakan sesuai rancangan. Pakai ulang kode yang disebut (path:sel). Jangan menyentuh file di luar
   daftar butir T?; kalau ternyata perlu, berhenti dan tulis alasannya di "Catatan pekerja" butir itu.
   Kalau rancangan salah atau tidak mungkin, jangan improvisasi besar: tulis di "Catatan pekerja" dan
   tanyakan ke pengguna.
5. Aturan kode di _memori/aturan.md berlaku penuh. Jalankan `python _alat/lint.py <file>` sampai 0 masalah
   (untuk contoh: sesuai gaya contoh yang ditetapkan RENCANA_TEKS.md).
   Notebook contoh baru: urutan Import → Konfigurasi → Baca → EDA → Split → Model → Evaluasi → Latih ulang
   + submission; semua yang diganti saat lomba di sel Konfigurasi (`# GANTI saat lomba`); sel terakhir
   `## Cek dengan kunci jawaban (HAPUS saat lomba)`; alternatif ber-`#` di sel Model, masing-masing
   dengan komentar singkat kapan dipakai. Notebook baru hanya bila RENCANA_TEKS.md memintanya (loader/alur
   latih berubah); ganti satu fungsi = alternatif berkomentar di notebook yang ada.
   Library baru: minta izin pengguna sebelum `pip install`, lalu tambahkan ke requirements.txt.
   Model baru: tambahkan ke _alat/unduh_model.py dan unduh lewat alat itu (cek cara pakainya di file).
6. Jalankan uji terima PERSIS seperti tertulis di butir (venv: Z:/GEMASTIK/Vignate-GEMASTIK/.venv). Beban
   GPU berat dijalankan satu per satu di foreground (laptop pernah mati mendadak). Butir model: catat tipe
   GPU (`nvidia-smi`), VRAM puncak (`torch.cuda.max_memory_allocated`), dan waktu per epoch. Simpan
   notebook arsenal dan contoh tanpa output.
7. Isi STATUS butir T? = "selesai (<tanggal>)" + bukti: perintah yang dijalankan, angka/hasil yang keluar,
   waktu jalan. Kalau ada uji yang gagal atau dilewati, tulis apa adanya, jangan ditandai selesai.
8. Perbarui file bersama sesuai cara di RENCANA_TEKS.md (baca ulang tepat sebelum mengedit; hanya blok
   milik butir ini): arsenal/PETA.md bila ada potongan baru; contoh/KATALOG.md (metrik, skor asli,
   alternatif + skor) dan contoh/PANDUAN.md (tanda soal → notebook) bila ada contoh baru. Lalu _memori/
   (log.md; status.md; keputusan.md bila memilih satu cara dari beberapa).
9. Jangan commit/push kecuali pengguna memintanya.

Di akhir: ringkas di chat apa yang berubah (file), hasil uji, dan apa yang belum.
```
