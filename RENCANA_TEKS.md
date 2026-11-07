# RENCANA_TEKS: peta kerja persiapan final teks

Dibuat 2026-11-06 oleh perencana (Prompt A di `PROMPT_FINAL_TEKS.md`). Pekerja mengerjakan butir T di
bagian 4 satu per satu dengan Prompt B. Fakta tanpa tanda = dibaca dari repo/data; tanda "dugaan" =
pengetahuan perencana yang belum diukur di repo ini.

## Pertanyaan untuk pengguna

**Dijawab pengguna 2026-11-06: "Ya, semua" (keempat rekomendasi disetujui).** Akibatnya: (1) T1 menjalankan
`setup.bat` untuk membuat `.venv`, uji memakai venv itu; (2) T1 boleh `pip install emoji ipywidgets` dan
menambahkannya ke `requirements.txt`; (3) T4 boleh mengunduh keempat model; (4) garis potong tetap sesudah T13.

1. **Python untuk uji.** `.venv` tidak ada di repo maupun di `Z:/GEMASTIK/Vignate-GEMASTIK/`, jadi
   `cari_python.bat` kosong dan `_alat/bangun_contoh.bat` akan GAGAL semua. Pilih: (a, disarankan) jalankan
   `setup.bat` sekarang untuk membuat `.venv` (pip online, unduhan besar karena torch cu128; sekaligus gladi
   setup PC panitia), atau (b) pakai Python sistem 3.13 (torch 2.11 cu128 sudah ada) dengan perintah
   `nbconvert` langsung. Jawaban: (a).
2. **Izin pip install + masuk requirements:** `emoji` (dipakai `contoh/text/inggris`) dan `ipywidgets`
   (`contoh/text/anotasi_teks`). Keduanya tidak terpasang dan tidak ada di `requirements.txt`, jadi dua notebook
   itu gagal di PC panitia.
3. **Izin unduh 4 model teks baru** (± 4 GB, dugaan): `indobert_large`, `indobertweet`, `xlmr_base`, `e5_base`
   (butir T4). Akibatnya unduhan `download_model.bat` hari setup naik dari ± 12 GB ke ± 16 GB.
4. **Garis potong:** setuju lanjut-pretraining MLM (TAPT) dan notebook NER-BERT ada di bawah garis (dikerjakan
   hanya bila waktu sisa)?

## Fakta lomba (dari pengguna, 2026-11-06)

- Final GEMASTIK XIX Divisi Data Mining, tuan rumah BINUS. Setup 2026-11-10 (ada internet), pengerjaan
  2026-11-11 13.00–18.00 (5 jam, tanpa internet, tanpa AI), presentasi 2026-11-12. Pengguna berangkat
  2026-11-09: semua persiapan selesai 2026-11-08 malam.
- Data final dipastikan TEKS. Bentuk tugas, bahasa, metrik, jebakan belum diketahui.
- 1 PC panitia per tim: i7-12700, RAM 32 GB, SSD 1 TB, RTX 3070 LHR 8 GB. Dataset, training, submission
  wajib di PC itu; laptop hanya pendukung. Model pretrained > 0,5 GB boleh.
- Bobot: penyisihan 20% (terkunci), leaderboard 30%, ketepatan metode 25% (kode + dokumentasi +
  presentasi), presentasi 25%; zip dokumentasi + kode + PPT wajib (gugur bila tidak ada).
- Riwayat: 2021 klasifikasi (Macro-F1), 2023 log jaringan kelas langka, 2024 tabular Golongan UKT dengan
  isian teks bebas (latihan pengguna F1 makro val 0,427, satu kelas F1 0), 2025 OCR aksara Jawa dengan data
  tidak lengkap. Panitia suka data kotor dan kejutan.
- Pengguna mengubah hanya sel Konfigurasi / `# UBAH` / `# GANTI saat lomba`, lalu Run All.

## Bahan yang sudah dibaca

Dibaca sendiri: `CLAUDE.md`, `_memori/` (status, aturan, keputusan, log teratas), `README.md`,
`RENCANA_LOMBA.md`, `arsenal/PETA.md`, `contoh/PANDUAN.md`, bagian teks `contoh/KATALOG.md`,
`_alat/unduh_model.py`, `_alat/bangun_contoh.bat`, `cari_python.bat`, `setup.bat` (bagian venv/pip).
Dibaca 6 subagent (sonnet, hanya membaca; laporan penuh tidak disimpan, ringkasannya di bawah):
32 notebook `contoh/text/`, 12 notebook `arsenal/teks/`, `arsenal/jembatan|umum|tabular`, alat uji,
16 set `data_latihan/text/`, `SOURCES_*.md`, `pretrained/`, paket terpasang, GPU, `_laya_eda_v2/`
(VISI, RENCANA), `kamus/`, `contoh/RISET.md`. Tidak dibaca: `arsenal/vision/*`, `contoh/` selain `text/`.

### Fakta yang dicek ulang perencana (2026-11-06)

- **Venv tidak ada.** `Z:/GEMASTIK/Vignate-GEMASTIK/.venv`, `.venv`/`venv` di repo: tidak ada.
  `cari_python.bat` hanya mencari VIRTUAL_ENV atau `.venv`/`venv` di repo/satu tingkat di atas, lalu
  `bangun_contoh.bat` memanggil `"%PY%" -m jupyter ...` → dengan PY kosong semua GAGAL. `python` di PATH =
  `C:\Users\EYZARIM\AppData\Local\Programs\Python\Python313\python.exe` (3.13), nbconvert 7.17.1, torch
  2.11.0+cu128, transformers 5.18.0, sentence-transformers 6.1.0 (versi dari subagent).
- **`_alat/log/hasil_build.txt` ditimpa tiap run** (`echo build ... >`). Isinya sekarang hanya
  `LOLOS sinyal\klasifikasi_jendela`, jadi bukan bukti contoh/text tidak pernah jalan; KATALOG memuat skor
  8 notebook teks (clustering, klasifikasi_embedding/finetune/tfidf, multilabel, ner, retrieval, topik).
- **Paket hilang** (import di notebook teks vs terpasang vs `requirements.txt`): `emoji` tidak terpasang,
  tidak di requirements (`contoh/text/inggris`); `ipywidgets` tidak terpasang, tidak di requirements
  (`anotasi_teks`); `nltk` terpasang tapi tidak tertulis (dugaan: terbawa `rouge_score`). Tidak terpasang:
  setfit, peft, seqeval, gliner, iterstrat, gensim, datasets, bitsandbytes, faiss.
- **Data:** `Z:/GEMASTIK/Vignate-GEMASTIK/data_latihan/text/` ada (16 set), jadi path
  `../../../data_latihan/text/...` dari `contoh/text/` benar (satu subagent salah menyebut tidak ada).
  Tidak ada `sample_submission` di set teks mana pun. Di luar `text/` tidak ada set teks panjang + kolom
  angka (hanya `multimodal/classification`: judul film + tahun, biner; `multitabel/dblp_acm`).
- `contoh/text/` = 32 notebook. Lint 0 masalah di 32 contoh + 12 arsenal/teks (dijalankan subagent).
- GPU laptop: RTX 5060 Laptop 8 GB, driver 617.42 (subagent, nvidia-smi). PC lomba: RTX 3070 8 GB.

### Ringkasan contoh/text (32 notebook)

Pola: Konfigurasi → baca CSV (`id`, teks, target) → EDA → holdout 80/20 → model (alternatif ber-#) →
evaluasi → latih ulang → `submission.csv` di folder kerja → sel kunci "HAPUS saat lomba".
Kelemahan bersama: tidak ada yang membaca/mengecek `sample_submission`; nama kolom `id`/target tertulis di sel
submission; output ke folder kerja (bukan `../../../outputs/...`); hampir tanpa CV/OOF (hanya
analisis_kesalahan, label_berisik), tanpa fp16, ensemble lintas model, ambang per kelas, atau log percobaan.
Beberapa praproses khusus Inggris (`stop_words="english"`, regex `a-z`).

| notebook | tugas, model aktif | skor KATALOG / catatan |
|---|---|---|
| absa | ABSA 10 aspek, TF-IDF union + LogReg per aspek | tanpa skor; split tidak stratified |
| analisis_kesalahan | OOF 5-fold TF-IDF+LogReg, salah per panjang/negasi/Laya | butuh `pretrained/laya` |
| anomali_teks | MiniLM + jarak kNN (alt IsolationForest, LOF) | sel data demo harus diganti |
| anotasi_teks | aturan + tombol ipywidgets + margin sampling | `ipywidgets` hilang |
| augmentasi | hapus/tukar/slang + TF-IDF LogReg | `train.sample(400)` tertanam |
| bahasa_daerah | NusaX, TF-IDF char+kata, GroupShuffleSplit `id % 600`, deteksi bahasa | grup khusus NusaX |
| clustering | TF-IDF + KMeans | ARI 0.2691; SVD+KMeans 0.4161 |
| ensemble | 3 TF-IDF soft voting (alt stacking) | md: ensemble 0.862 < anggota terbaik 0.867 |
| generasi | Qwen2.5-0.5B few-shot vs Lead-3, ROUGE/chrF/BLEU | md: model kecil sering kalah Lead-3 |
| inggris | demojize + stem Porter + TF-IDF LogReg | `emoji` hilang |
| kata_kunci_ringkasan | TF-IDF kata kunci + kalimat | tanpa metrik |
| klasifikasi_embedding | MiniLM + LogReg (alt 7 model, indobert) | F1 0.8461; indobert 0.8660 |
| klasifikasi_finetune | IndoBERT, loop manual, holdout, MAX_LENGTH 64 | F1 0.9242; distilbert_id 0.9033; tanpa fp16/latih ulang |
| klasifikasi_tfidf | TF-IDF + LinearSVC (alt 10) | F1 0.8636; LogReg 0.8637 |
| label_berisik | OOF + cleanlab, buang/perbaiki | sel "rusak label" demo |
| llm_prompt | Qwen 0.5B few-shot | bukan fokus |
| multilabel | TF-IDF + OvR LogReg | F1 0.7264; char_wb 0.7911; tanpa ambang |
| ner | CRF lbfgs | F1w 0.6299 (per token, bukan per entitas) |
| pasangan_teks | MiniLM [|a-b|, a*b] + leksikal → LogReg | tanpa skor |
| pos_tagging | CRF | butuh file valid |
| rag_llm | TF-IDF retrieval + Qwen | bukan fokus |
| regresi_teks | TF-IDF union + Ridge, clip, QWK | tanpa skor; clean_text khusus Inggris |
| retrieval | BM25 (alt TF-IDF, MiniLM) | MRR 0.8673; MiniLM 0.9138; sel Baca membaca kunci |
| semi_supervised_teks | pseudo-label N teratas per kelas | tanpa skor; pool bisa terduplikasi |
| tanya_jawab | mDeBERTa SQuAD2 zero-shot, max 384 | tanpa sliding window |
| topik | LDA (alt NMF) | ARI 0.3943; NMF 0.4384 |
| topik_bertopic | MiniLM + UMAP + HDBSCAN | stopword Indonesia belum ada |
| topik_dinamis | NMF + tren bulanan | clean_text khusus Inggris |
| tuning | Optuna TF-IDF + LogReg, CV f1_macro | ruang cari tertanam |
| ujaran_kebencian | kamus alay + TF-IDF OvR LogReg balanced + ambang per label | ambang dari model validasi |
| weak_supervision | aturan regex + vote → LogReg | kelas tanpa aturan tak terprediksi |
| zero_shot_nli | mDeBERTa XNLI pipeline | tanpa self-training (PETA menyebut ada) |

### Ringkasan arsenal (nomor `#k` = sel kode ke-k, nomor yang dipakai `uji_arsenal.py`)

- `teks/dasar/praproses` #2 kembar + kembar beda label, #3 clean+slang, #4 Sastrawi (cache), #5 chi².
  Hanya train; test tidak ikut dibersihkan.
- `teks/klasifikasi/tfidf` #2 FeatureUnion kata+char + LogReg balanced, #4 CV 3 kandidat, #5 koefisien,
  #8 proba test, #9 Ridge regresi. `embedding` #2–#6 MiniLM + blend bobot dengan TF-IDF (bobot dicari di val).
  `finetune_bert` #2 persentil token, #5–#7 bf16 + clip + warmup + early stopping (patience 2); markdown
  menjanjikan kepala+ekor dan multi-seed, kode tidak; test dari model 80%.
- `teks/zero_shot` #2–#6 NLI + pilih template + self-training 1 putaran. `teks/multilabel` #4 OOF KFold,
  #5 ambang per label. `teks/aspek` OOF per aspek + gating kata kunci. `teks/ner` CRF #2–#9, BERT #10–#15
  (tanpa sel submission BERT). `teks/pasangan` leksikal + MiniLM + NLI dua arah → LogReg.
  `teks/kemiripan/retrieval` BM25, MiniLM, RRF, QA ekstraktif, kembar/hampir kembar.
  `teks/pengelompokan/topik` NMF, UMAP+HDBSCAN, c-TF-IDF, tren. `teks/ringkasan` LEAD, TextRank, ROUGE sendiri.
- `jembatan/teks_ke_vektor` TF-IDF, MiniLM, IndoBERT mean-pool → npy (tanpa versi test di #8).
  `jembatan/vektor_ke_model` OOF helper, blend 2 model, stacking (demo gambar, metrik akurasi).
- `umum/teks_regex` regex, slang, ekstrak field, uang/tanggal Indonesia, rapidfuzz. `umum/baca_folder` csv,
  txt per label, json/jsonl, xlsx, zip, pdf (tanpa tsv, parquet, deteksi encoding). `umum/simpan_muat` #15
  validator minimal (kolom, jumlah baris, NaN; hanya print; tanpa cek himpunan/urutan id, label, tipe).
- `tabular/metrik` metrik numpy (#3 F1, #4 AUC, #7 QWK, #8 MAP@k, #9 NDCG), #11 ambang QWK, #14 pengali kelas
  F1 makro. `tabular/ensembling` #9 hill climbing Caruana (logloss saja), #11 stacking, #12 seed, #13
  pseudo-label. `tabular/dasar/validasi` #2 fold stratified/group, #4 leak_scan biner, #5 adversarial
  (numerik). `tabular/tuning` Optuna LGBM. `tabular/imbalanced` #5 ambang biner. `tabular/ranking` LambdaMART.
- Tidak ada di mana pun: kalibrasi (kode), rank average proba, NB-SVM, seqeval, metrik ensemble selain
  logloss, adversarial validation teks, overlap train↔test teks, pembaca tsv/parquet/encoding.

### Data, model, paket, alat

- 16 set teks: absa (hoasa 2568/286), arxiv (2000, ±10 berlabel per kelas), classification (emot 3521/440,
  5 kelas), classification_en (news4 2140/1429), english (tweet 8000/2000), generasi (3 contoh + 50 test),
  multilabel (casa 810/90, 6 label), ner (nerp 2500/600 kalimat, test berlabel), nusax (7200/4800, 12 bahasa),
  pair (wrete 350/100), pos (posp), qa (facqa 2495/311/311), regression (yelp 6000/2000, Inggris),
  sentiment (smsa 5148/1260, 3 kelas), topik_waktu (7768 tweet, tanggal), ujaran_kebencian (10418/2605,
  12 label). Semua CSV koma UTF-8; kunci di `*_test_labels.csv`. Sumber 9 set tidak tercatat di
  `SOURCES_text.md` (ada di kode `_alat/unduh_data_tambahan.py`).
- Pola set baru: fungsi di `_alat/unduh_data_tambahan.py` + `simpan_split(...)` → `{nama}_train/test/test_labels.csv`.
- `pretrained/` 12 GB. Teks: minilm (926 MB, multibahasa), minilm_en, distilbert_id, distilbert_en, indobert
  (base-p1), nli_mdeberta (mDeBERTa-v3 XNLI), qa_mdeberta, qwen_05b, laya. `unduh_model.py` mengunduh semua
  dan melewati folder yang sudah ada (tidak bisa pilih satu model).
- Laya v2 (`_laya_eda_v2`): untuk tabular; teks hanya `text_stats`, skrub encoder, isian bebas, ejaan; tidak
  ada TF-IDF/embedding/fine-tune. `kamus/03_jebakan.md`: jebakan Python umum, tidak ada yang khusus NLP.
- `contoh/RISET.md` teks: normalisasi slang paling berpengaruh untuk IndoBERT (+0,061 F1, sumber di RISET);
  stemming/stopword bisa merugikan transformer (negasi hilang); urutan TF-IDF → embedding → fine-tune.

---

## 0. Peta ML teks secara keseluruhan

Disusun dari pengetahuan bidang, bukan dari isi repo dan bukan dari tebakan soal. Kode sel dipakai di
bagian 1–4. "alt" = cukup alternatif berkomentar di notebook yang ada (ganti satu fungsi/beberapa
baris); "nb" = butuh notebook sendiri (loader/cara muat/alur latih beda).

### 0.1 Bentuk masukan/data (P.data)

| kode | sel |
|---|---|
| P.data.1 | satu kolom teks + satu label |
| P.data.2 | banyak kolom teks (judul + isi, pertanyaan + konteks); sebagian kosong |
| P.data.3 | teks + metadata tabular (angka, kategori, tanggal) |
| P.data.4 | pasangan teks (premis–hipotesis, dua pertanyaan, query–dokumen) |
| P.data.5 | query + banyak kandidat (ranking per query, grup) |
| P.data.6 | dokumen panjang > 512 token (chunk, kepala+ekor, agregasi per dokumen) |
| P.data.7 | token per baris CoNLL/BIO, kalimat dipisah baris kosong |
| P.data.8 | span dengan offset karakter (JSON start/end) |
| P.data.9 | percakapan/thread (giliran bicara, balasan, konteks sebelumnya) |
| P.data.10 | teks berwaktu (tanggal posting; urutan; drift) |
| P.data.11 | teks semi-terstruktur / isian bebas → angka, satuan, kategori (pola soal 2024) |
| P.data.12 | entitas berulang (penulis, pengguna, produk) di train dan test |
| P.data.13 | format file: csv, tsv, xlsx, json, jsonl, parquet, txt per baris, folder txt per kelas, zip |
| P.data.14 | encoding dan pemisah aneh: BOM, latin-1/cp1252, `;`, kutip rusak, baris multi-line |
| P.data.15 | label berformat aneh: string list `"['a','b']"`, one-hot banyak kolom, angka vs nama |
| P.data.16 | bahasa campur / daerah / transliterasi |
| P.data.17 | teks hasil OCR/ASR (noise karakter) |
| P.data.18 | HTML, markdown, emoji, hashtag, mention, URL |

### 0.2 Tugas (P.tugas)

| kode | sel | kelompok |
|---|---|---|
| P.tugas.1 | klasifikasi biner | supervised |
| P.tugas.2 | klasifikasi multikelas | supervised |
| P.tugas.3 | multilabel | supervised |
| P.tugas.4 | hierarkis (kategori bertingkat) | supervised |
| P.tugas.5 | ordinal (rating 1–5, golongan, tingkat) | supervised |
| P.tugas.6 | regresi (skor, harga, umur) | supervised |
| P.tugas.7 | penilaian esai / skor teks panjang (QWK) | supervised |
| P.tugas.8 | NER / pelabelan token BIO | supervised |
| P.tugas.9 | ekstraksi span/slot (field dari teks bebas) | supervised |
| P.tugas.10 | POS tagging / pelabelan urutan lain | supervised |
| P.tugas.11 | ABSA (aspek + sentimen per aspek) | supervised |
| P.tugas.12 | pasangan: NLI, parafrase, duplikat | supervised |
| P.tugas.13 | kemiripan semantik bernilai (STS, skor 0–5) | supervised |
| P.tugas.14 | ranking/retrieval (query → dokumen) | supervised |
| P.tugas.15 | QA ekstraktif | supervised |
| P.tugas.16 | intent + slot pada percakapan | supervised |
| P.tugas.17 | emosi / stance / sarkasme / ujaran kebencian (klasifikasi khusus) | supervised |
| P.tugas.18 | atribusi penulis / gaya (stylometri) | supervised |
| P.tugas.19 | deteksi bahasa | supervised |
| P.tugas.20 | topik | unsupervised |
| P.tugas.21 | clustering dokumen | unsupervised |
| P.tugas.22 | anomali / spam / di luar topik | unsupervised |
| P.tugas.23 | deduplikasi / near-duplicate | unsupervised |
| P.tugas.24 | kata kunci / keyphrase | unsupervised |
| P.tugas.25 | entity resolution / cocokkan nama | unsupervised |
| P.tugas.26 | zero-shot (NLI, kemiripan ke deskripsi label) | label sedikit |
| P.tugas.27 | few-shot (SetFit, prototipe embedding) | label sedikit |
| P.tugas.28 | semi-supervised (self-training, pseudo-label) | label sedikit |
| P.tugas.29 | weak supervision (labeling function, kamus) | label sedikit |
| P.tugas.30 | active learning / melabeli sendiri | label sedikit |
| P.tugas.31 | PU learning / kelas "lainnya" di test | label sedikit |
| P.tugas.32 | ringkasan ekstraktif | generatif |
| P.tugas.33 | ringkasan abstraktif / parafrase (seq2seq kecil) | generatif |
| P.tugas.34 | generasi / terjemahan / normalisasi ejaan (seq2seq) | generatif |
| P.tugas.35 | tren topik/volume berdasarkan waktu | lain |

### 0.3 Representasi (P.rep)

| kode | sel |
|---|---|
| P.rep.1 | aturan / regex / kamus kata kunci |
| P.rep.2 | leksikon sentimen/emosi (InSet, kamus kasar) |
| P.rep.3 | normalisasi: lowercase, alay → baku, stemming Sastrawi, stopword tanpa membuang negasi |
| P.rep.4 | BoW / Count biner |
| P.rep.5 | TF-IDF kata n-gram |
| P.rep.6 | TF-IDF karakter (`char_wb` 2–5), tahan typo |
| P.rep.7 | hashing vectorizer (data besar) |
| P.rep.8 | BM25 |
| P.rep.9 | rasio log-count NB (fitur NB-SVM) |
| P.rep.10 | embedding statis (word2vec/fastText dilatih di teks soal) + rata-rata/SIF |
| P.rep.11 | embedding kalimat bi-encoder (MiniLM, e5, LaBSE, bge-m3) |
| P.rep.12 | embedding dari encoder (CLS/mean pooling, tanpa fine-tune) |
| P.rep.13 | representasi transformer yang di-fine-tune |
| P.rep.14 | fitur buatan: panjang, jumlah kata, huruf besar, tanda baca, emoji, angka, URL, mention, rasio alay, OOV |
| P.rep.15 | fitur metadata + interaksi dengan teks |
| P.rep.16 | reduksi dimensi: SVD/LSA, PCA, UMAP |
| P.rep.17 | fitur stylometri (panjang kata, fungsi kata, char n-gram) |
| P.rep.18 | LLM kecil sebagai pembuat fitur (bukan fokus) |

### 0.4 Keluarga model menurut cara muat/latih/inferensi (P.kel)

Angka ukuran dan VRAM = dugaan sampai diukur (lihat butir model).

| kode | keluarga (cara muat) | kandidat checkpoint (dugaan) | unduhan / VRAM latih 8 GB (dugaan) | library | tempat |
|---|---|---|---|---|---|
| P.kel.1 | sklearn linear/NB/SVM di atas sparse | – | – / CPU | sklearn | alt |
| P.kel.2 | GBDT di atas fitur/SVD/embedding; CatBoost `text_features` | – | – / CPU (GPU opsional) | lightgbm, xgboost, catboost | alt |
| P.kel.3 | encoder + head klasifikasi (`AutoModelForSequenceClassification`) | indobert-base-p1 (ada), indobert-large-p1, IndoBERTweet, NusaBERT, XLM-R base/large, mDeBERTa-v3-base | base 0,5–1,1 GB / 3–5 GB; large 1,3–2,2 GB / 6–8 GB dengan fp16 + grad checkpointing + batch 4–8 | transformers | alt (ganti checkpoint) |
| P.kel.4 | encoder + head kustom `nn.Module` (regresi, ordinal CORAL, teks + tabular, multi-sample dropout, pooling lain) | encoder di P.kel.3 | sama P.kel.3 | torch | nb |
| P.kel.5 | token classification + perataan label subword | indobert, XLM-R | sama P.kel.3 | transformers | nb |
| P.kel.6 | QA ekstraktif + offset mapping | mdeberta-v3-base-squad2 (ada) | 1,1 GB / 4–6 GB | transformers | nb |
| P.kel.7 | cross-encoder / reranker (pasangan masuk bersama) | mmarco-mMiniLMv2, bge-reranker-v2-m3 | 0,5–2,2 GB / 3–7 GB | sentence-transformers | nb |
| P.kel.8 | bi-encoder: encode saja | MiniLM (ada), multilingual-e5-base/large, LaBSE, bge-m3 | 0,5–2,2 GB / inferensi 1–3 GB | sentence-transformers | alt (ganti path + prefix) |
| P.kel.9 | bi-encoder fine-tune (MultipleNegativesRankingLoss, CoSENT) | sama P.kel.8 | / 3–6 GB | sentence-transformers | nb |
| P.kel.10 | SetFit (fine-tune kontras + head) | paraphrase-multilingual-mpnet-base-v2, e5 | 1,1 GB / 3–5 GB | setfit | nb |
| P.kel.11 | seq2seq kecil (`AutoModelForSeq2SeqLM` + generate) | mT5-small/base, IndoT5-base, flan-t5-base | 0,9–2,3 GB / 4–7 GB | transformers | nb |
| P.kel.12 | lanjut pretraining MLM (`AutoModelForMaskedLM`) di teks soal | encoder P.kel.3 | sama P.kel.3 | transformers | nb |
| P.kel.13 | sekuens klasik: CRF, BiLSTM-CRF | – | – / CPU atau < 1 GB | sklearn-crfsuite, torch | nb |
| P.kel.14 | dilatih dari nol: fastText supervised, TextCNN/BiLSTM | – | – / < 1 GB | fasttext/gensim, torch | nb |
| P.kel.15 | embedding statis dilatih di teks soal (word2vec/fastText) | – | – / CPU | gensim | nb |
| P.kel.16 | zero-shot NLI pipeline | mDeBERTa NLI (ada) | 1,1 GB / inferensi 1–2 GB | transformers | alt |
| P.kel.17 | zero-shot NER (GLiNER) | gliner_multi | 1,2 GB / inferensi 2 GB | gliner | nb |
| P.kel.18 | topik berbasis embedding (BERTopic) | MiniLM | – | bertopic, umap, hdbscan | nb |
| P.kel.19 | model tugas-spesifik Indonesia sebagai fitur (sentimen/emosi/HS siap pakai) | dugaan: banyak di HF | 0,5 GB / inferensi | transformers | alt |
| P.kel.20 | LLM kecil + LoRA untuk klasifikasi | Qwen2.5-0.5B (ada) / 1.5B | 1–3 GB / 6–8 GB (QLoRA) | peft, bitsandbytes | nb, bukan fokus |
| P.kel.21 | LLM prompt one/zero-shot | Qwen2.5-0.5B (ada) | – | transformers | bukan fokus (satu baris) |
| P.kel.22 | kuantisasi/ONNX untuk inferensi cepat | – | – | optimum, onnxruntime | nb, bukan fokus |

### 0.5 Metode ML: standar, niche, SOTA (P.met)

Tingkat: S = standar, N = niche, T = SOTA/kompetisi teratas. Biaya = waktu tambahan di 3070 (dugaan).

| kode | metode | tingkat | kapan membantu | biaya | tempat |
|---|---|---|---|---|---|
| P.met.k.1 | LogReg / LinearSVC / Ridge di TF-IDF | S | baseline semua klasifikasi | detik | alt |
| P.met.k.2 | MultinomialNB / ComplementNB | S | teks pendek, data kecil | detik | alt |
| P.met.k.3 | NB-SVM (fitur rasio log NB + LogReg) | N | sentimen, data kecil–sedang; sering kalahkan SVM biasa | detik | alt |
| P.met.k.4 | SGD + early stopping, hashing | S | data sangat besar | menit | alt |
| P.met.k.5 | kalibrasi LinearSVC (`CalibratedClassifierCV`) | S | perlu probabilitas untuk ensemble/AUC | detik | alt |
| P.met.k.6 | GBDT di atas SVD TF-IDF / embedding + fitur buatan | S | teks + metadata, fitur non-linear | menit | alt |
| P.met.k.7 | kNN / prototipe di embedding | S | few-shot, label rapi | detik | alt |
| P.met.k.8 | gabung TF-IDF + embedding (hstack atau rata proba) | S | hampir selalu naik sedikit | menit | alt |
| P.met.f.1 | max_len dari distribusi panjang; truncation kepala+ekor | N | teks panjang, info di akhir | 0 | alt |
| P.met.f.2 | dynamic padding + urut panjang (batch per panjang) | N | mempercepat 1,5–3× | 0 | alt |
| P.met.f.3 | fp16/bf16 AMP | S | wajib di 8 GB | menghemat | alt |
| P.met.f.4 | gradient accumulation | S | batch efektif besar di 8 GB | 0 | alt |
| P.met.f.5 | gradient checkpointing | S | model large di 8 GB | +30% waktu | alt |
| P.met.f.6 | layer-wise LR decay (LLRD) | N | fine-tune stabil, +0,5–1 poin | 0 | alt |
| P.met.f.7 | re-init lapisan atas | N | data kecil | 0 | alt |
| P.met.f.8 | pooling lain (mean, attention, concat 4 lapisan terakhir) | N | sering > CLS | 0 | alt (head kustom) |
| P.met.f.9 | multi-sample dropout | N | regresi/klasifikasi stabil | kecil | alt (head kustom) |
| P.met.f.10 | R-Drop (dua forward + KL) | T | regularisasi, data kecil | 2× waktu | alt |
| P.met.f.11 | FGM (adversarial di embedding) | T | +0,5–1 poin di kompetisi teks | +50% waktu | alt |
| P.met.f.12 | AWP (adversarial weight perturbation) | T | solusi juara Kaggle teks | +50–100% waktu | alt |
| P.met.f.13 | EMA bobot | N | hasil akhir stabil | kecil | alt |
| P.met.f.14 | SWA | N | rata-rata bobot akhir | kecil | alt |
| P.met.f.15 | lanjut pretraining MLM di teks soal (TAPT) | T | domain beda dari pretraining (slang, istilah) | 10–30 menit | nb |
| P.met.f.16 | k-fold fine-tune + OOF + rata-rata model fold | S | skor stabil, bahan stacking | k× waktu | alt |
| P.met.f.17 | warmup + jadwal linear/cosine | S | stabil | 0 | alt |
| P.met.f.18 | early stopping pada metrik lomba per setengah epoch | S | hemat waktu | 0 | alt |
| P.met.f.19 | sliding window/chunk + agregasi untuk dokumen panjang | N | > 512 token | 2–4× | alt |
| P.met.f.20 | 8-bit Adam / freeze lapisan bawah | N | VRAM kurang | 0 | alt |
| P.met.l.1 | cross-entropy + class weight | S | kelas timpang | 0 | alt |
| P.met.l.2 | focal loss | N | kelas langka, contoh sulit | 0 | alt |
| P.met.l.3 | class-balanced loss (effective number) | N | timpang ekstrem | 0 | alt |
| P.met.l.4 | label smoothing | S | label berisik | 0 | alt |
| P.met.l.5 | BCE + pos_weight (multilabel) | S | multilabel timpang | 0 | alt |
| P.met.l.6 | asymmetric loss (ASL) | N | multilabel dengan banyak negatif | 0 | alt |
| P.met.l.7 | loss ordinal (CORAL/CORN) atau regresi + ambang | N | target berurutan, QWK | 0 | alt (head kustom) |
| P.met.l.8 | loss kontras (MNRL, SupCon) | N | embedding/few-shot | – | nb (P.kel.9/10) |
| P.met.l.9 | MSE / Huber / L1 untuk regresi | S | regresi | 0 | alt |
| P.met.l.10 | soft-F1 loss | N | metrik F1 langsung | 0 | alt |
| P.met.d.1 | pseudo-label test yang yakin | T | test besar, model sudah bagus | 1 putaran latih | alt |
| P.met.d.2 | self-training iteratif | N | label sedikit | beberapa putaran | alt |
| P.met.d.3 | distilasi (teacher besar → student / soft label) | T | inferensi cepat, ensemble jadi satu | 1 putaran latih | nb |
| P.met.d.4 | augmentasi EDA (tukar, hapus, sinonim) | S | data kecil | menit | alt |
| P.met.d.5 | augmentasi noise typo/alay | N | test lebih kotor dari train | menit | alt |
| P.met.d.6 | back-translation | N | data kecil; butuh model terjemah offline | lama | nb, bukan fokus |
| P.met.d.7 | mixup di embedding | N | regularisasi | kecil | alt |
| P.met.d.8 | over/undersampling teks | S | timpang | 0 | alt |
| P.met.d.9 | buang duplikat; selesaikan duplikat beda label | S | data kotor | detik | alt |
| P.met.d.10 | deteksi label berisik (cleanlab / OOF yakin tapi beda) | N | label manusia kotor | menit | alt |
| P.met.h.1 | grid kecil | S | 1–2 parameter | menit | alt |
| P.met.h.2 | random search | S | ruang besar | menit | alt |
| P.met.h.3 | Optuna TPE + pruning | N | GBDT / TF-IDF + C | 10–30 menit | alt |
| P.met.h.4 | successive halving (`HalvingRandomSearchCV`) | N | anggaran waktu ketat | menit | alt |
| P.met.e.1 | rata-rata probabilitas | S | 2+ model beda jenis | detik | alt |
| P.met.e.2 | rank average | S | skala proba beda (AUC) | detik | alt |
| P.met.e.3 | bobot lewat hill climbing pada OOF | T | 3+ model; tiap model sedikit beda | detik | alt |
| P.met.e.4 | ensemble selection Caruana (dengan pengembalian) | T | banyak kandidat model | detik | alt |
| P.met.e.5 | optimasi bobot `scipy.optimize` (Nelder-Mead) | N | sedikit model, metrik halus | detik | alt |
| P.met.e.6 | stacking (LogReg/LGBM di OOF) | N | model beragam, OOF tersedia | menit | alt |
| P.met.e.7 | blending holdout | S | tidak ada OOF | detik | alt |
| P.met.e.8 | seed averaging / bagging | S | varians tinggi (fine-tune) | k× waktu | alt |
| P.met.e.9 | voting mayoritas | S | hanya label | detik | alt |
| P.met.e.10 | rata-rata model antar-fold untuk test | S | k-fold | 0 | alt |
| P.met.p.1 | kalibrasi Platt / isotonic | S | log loss, ensemble | detik | alt |
| P.met.p.2 | temperature scaling | N | fine-tune terlalu yakin | detik | alt |
| P.met.p.3 | ambang biner optimal di OOF | S | F1 kelas positif | detik | alt |
| P.met.p.4 | pengali/ambang per kelas untuk F1 makro (coordinate ascent di OOF) | N | multikelas timpang | detik | alt |
| P.met.p.5 | ambang per label (multilabel) | S | multilabel | detik | alt |
| P.met.p.6 | ambang ordinal (OptimizedRounder) untuk QWK | N | ordinal/regresi dinilai QWK | detik | alt |
| P.met.p.7 | aturan/kamus menimpa prediksi (kasus pasti) | N | pola jelas di data | detik | alt |
| P.met.p.8 | penyesuaian prior kelas test (EM Saerens) | N | proporsi kelas train ≠ test | detik | alt |
| P.met.p.9 | perbaikan BIO + gabung span (NER) | S | NER | detik | alt |
| P.met.p.10 | clip / round regresi ke rentang valid | S | regresi | 0 | alt |

### 0.6 Masalah data dan jebakan (P.jeb)

| kode | jebakan |
|---|---|
| P.jeb.1 | kelas timpang ekstrem; kelas dengan < 10 contoh |
| P.jeb.2 | duplikat di train |
| P.jeb.3 | teks sama, label beda (konflik) |
| P.jeb.4 | bocor train↔test (teks sama/hampir sama) |
| P.jeb.5 | bocor lewat kolom lain (id berurutan, metadata menyimpan label) |
| P.jeb.6 | label berisik |
| P.jeb.7 | pergeseran distribusi train vs test |
| P.jeb.8 | pergeseran waktu |
| P.jeb.9 | slang/alay/singkatan |
| P.jeb.10 | bahasa daerah / campur |
| P.jeb.11 | typo / ejaan bebas |
| P.jeb.12 | emoji/emotikon bermakna (dibuang = info hilang) |
| P.jeb.13 | teks kosong/NaN/spasi/hanya simbol |
| P.jeb.14 | teks sangat panjang atau sangat pendek |
| P.jeb.15 | PII / token tersamar (`[USERNAME]`, nomor HP, URL) |
| P.jeb.16 | mojibake / HTML entity (`Ã©`, `&amp;`) |
| P.jeb.17 | label berformat aneh; spasi/huruf besar beda; label test tidak ada di train |
| P.jeb.18 | format submission aneh (urutan id, label string vs angka, proba per kelas, multilabel dipisah spasi) |
| P.jeb.19 | kelas baru / "lainnya" di test |
| P.jeb.20 | id duplikat / baris test hilang di submission |
| P.jeb.21 | negasi dan sarkasme ("tidak bagus", "mantap banget lemotnya") |
| P.jeb.22 | penulis/entitas sama di train dan test (bocor via group) |
| P.jeb.23 | data sangat kecil (< 500) atau sangat besar (> 1 juta baris) |
| P.jeb.24 | banyak kolom teks, sebagian kosong |
| P.jeb.25 | isian bebas berisi angka + satuan (pola UKT) |
| P.jeb.26 | metrik khusus/tidak biasa dari panitia |
| P.jeb.27 | stopword/stemming membuang kata penting (negasi, intensitas) |

### 0.7 Evaluasi (P.eval)

| kode | sel |
|---|---|
| P.eval.1 | akurasi |
| P.eval.2 | F1 makro |
| P.eval.3 | F1 mikro |
| P.eval.4 | F1 berbobot |
| P.eval.5 | precision/recall/F1 per kelas (classification_report) |
| P.eval.6 | QWK |
| P.eval.7 | RMSE / MAE / RMSLE |
| P.eval.8 | Pearson / Spearman (STS) |
| P.eval.9 | F1 entitas seqeval (strict) |
| P.eval.10 | exact match + F1 token (QA) |
| P.eval.11 | MAP@k |
| P.eval.12 | NDCG@k |
| P.eval.13 | MRR / Recall@k |
| P.eval.14 | ROUGE-1/2/L |
| P.eval.15 | BLEU / chrF |
| P.eval.16 | log loss |
| P.eval.17 | ROC-AUC / PR-AUC |
| P.eval.18 | metrik multilabel (hamming, F1 sampel/mikro/makro) |
| P.eval.19 | ARI / NMI / silhouette / koherensi topik |
| P.eval.20 | holdout stratified |
| P.eval.21 | StratifiedKFold + OOF |
| P.eval.22 | stratifikasi multilabel (iterative stratification) |
| P.eval.23 | GroupKFold (penulis, klaster duplikat) |
| P.eval.24 | split waktu |
| P.eval.25 | adversarial validation |
| P.eval.26 | log percobaan: skor CV vs LB, pilih final |
| P.eval.27 | analisis kesalahan: confusion matrix, contoh salah, salah per panjang/kelas |
| P.eval.28 | bootstrap CI / beda skor bermakna |
| P.eval.29 | validator submission terhadap sample_submission (id, urutan, kolom, nilai, tipe, NaN) |

### 0.8 Penjelasan dan presentasi (P.pres)

| kode | sel |
|---|---|
| P.pres.1 | kata khas per kelas (chi², log-odds) |
| P.pres.2 | koefisien model linear: kata paling berpengaruh |
| P.pres.3 | contoh salah + dugaan penyebab |
| P.pres.4 | confusion matrix ternormalisasi |
| P.pres.5 | atribusi token (Integrated Gradients / LIME / SHAP teks) |
| P.pres.6 | peta embedding UMAP diwarnai kelas |
| P.pres.7 | tabel ablation (tiap teknik → +x skor CV) |
| P.pres.8 | EDA: distribusi panjang, kelas, bahasa |
| P.pres.9 | temuan jebakan (duplikat, konflik label, bocor) sebagai slide |
| P.pres.10 | diagram alur pipeline |
| P.pres.11 | tabel model: skor CV, waktu, ukuran |
| P.pres.12 | dokumentasi + paket zip (syarat gugur) |

### 0.9 Kendala eksekusi (P.eks)

| kode | sel |
|---|---|
| P.eks.1 | offline: model + library terpasang, `HF_HUB_OFFLINE=1` |
| P.eks.2 | VRAM 8 GB: base aman, large perlu fp16 + grad checkpointing + batch kecil |
| P.eks.3 | waktu per epoch (dari jumlah token × throughput) |
| P.eks.4 | RAM 32 GB: TF-IDF besar tetap sparse |
| P.eks.5 | simpan OOF + prediksi test per model (kernel mati, ensemble nanti) |
| P.eks.6 | anggaran waktu per tahap (stopwatch) |
| P.eks.7 | seed / reproducibility |
| P.eks.8 | CPU paralel (`n_jobs`) saat GPU dipakai fine-tune |
| P.eks.9 | kuota submit (±10 kali, pilih final) |
| P.eks.10 | GPU tidak terdeteksi / driver lama → jalur CPU tetap jalan |

---

## 1. Peta cakupan repo

Status: ✅ ada dan bisa dipakai, ⚠️ sebagian / ada tapi bolong, ❌ tidak ada. Kolom arsenal memakai nomor sel
kode `#k`; kolom contoh = notebook `contoh/text/` + skor KATALOG bila ada. "Butir" = butir di bagian 4 yang
menutup celahnya; "diterima" = celah sengaja dibiarkan (alasan singkat).

Jumlah: 243 sel peta. ✅ 105, ⚠️ 62, ❌ 76. Arsenal menyentuh 141 sel, contoh/text 119 sel (dihitung dari
tabel di bawah). Metode ML (P.met, 72 sel: 33 standar, 31 niche, 8 SOTA): ✅ 22, ⚠️ 16, ❌ 34.

### 1.1 Bentuk data

| kode | arsenal | contoh | status | butir |
|---|---|---|---|---|
| P.data.1 | teks/klasifikasi/tfidf, embedding, finetune_bert | klasifikasi_tfidf 0.8636, _embedding 0.8461, _finetune 0.9242 | ✅ | T2, T5 |
| P.data.2 | – | pasangan_teks, tanya_jawab (khusus) | ⚠️ | T2 (gabung TEXT_COLS) |
| P.data.3 | jembatan/teks_ke_vektor + vektor_ke_model; tabular/klasifikasi/gbdt | – | ⚠️ | T8 |
| P.data.4 | teks/pasangan | pasangan_teks | ✅ | T5 (alt pasangan) |
| P.data.5 | teks/kemiripan/retrieval; tabular/ranking | retrieval MRR 0.8673 | ✅ | B7 |
| P.data.6 | – (truncation kepala saja) | – (MAX_LENGTH 64) | ❌ | T5 (kepala+ekor), B4 |
| P.data.7 | teks/ner (CSV panjang) | ner, pos_tagging (CSV panjang) | ⚠️ | T3 (pembaca CoNLL) |
| P.data.8 | – | – | ❌ | diterima (peluang kecil, dugaan) |
| P.data.9 | – | – | ❌ | diterima (gabung jadi satu teks, T2) |
| P.data.10 | teks/pengelompokan/topik#7; tabular/dasar/validasi#7–#8 | topik_dinamis | ⚠️ | T2 (alt split waktu) |
| P.data.11 | umum/teks_regex#4, #7, #8 | – (Laya v2 tab Isian bebas) | ✅ | – |
| P.data.12 | tabular/dasar/validasi#2 | bahasa_daerah (GroupShuffleSplit) | ⚠️ | T2 (alt GroupKFold) |
| P.data.13 | umum/baca_folder#5–#8, #12, #13 | – (CSV saja) | ⚠️ | T3 |
| P.data.14 | umum/baca_folder#6 (ENCODING manual) | – | ❌ | T3 |
| P.data.15 | teks/multilabel#3 (get_dummies) | – | ⚠️ | T3 |
| P.data.16 | – | bahasa_daerah | ✅ | T5 (xlmr_base) |
| P.data.17 | teks/klasifikasi/tfidf (char_wb) | char_wb di banyak contoh | ⚠️ | diterima |
| P.data.18 | umum/teks_regex#3; teks/dasar/praproses#3 | inggris (demojize), ujaran_kebencian | ⚠️ | T1 (emoji), T2 (alt clean) |

### 1.2 Tugas

| kode | arsenal | contoh | status | butir |
|---|---|---|---|---|
| P.tugas.1 | tabular/imbalanced#5 (ambang biner) | klasifikasi_tfidf (komentar pos_label) | ✅ | T2 (alt ambang) |
| P.tugas.2 | teks/klasifikasi/* | klasifikasi_* | ✅ | T2, T5 |
| P.tugas.3 | teks/multilabel#4–#5 | multilabel 0.7264 (char_wb 0.7911), ujaran_kebencian | ✅ | T9 (BERT) |
| P.tugas.4 | – | – | ❌ | diterima (ratakan jadi kelas gabungan) |
| P.tugas.5 | tabular/metrik#10–#12 | regresi_teks (QWK pembulatan) | ⚠️ | T10 |
| P.tugas.6 | teks/klasifikasi/tfidf#9 | regresi_teks | ✅ | T10 (BERT) |
| P.tugas.7 | tabular/metrik#7 | regresi_teks | ⚠️ | T10 |
| P.tugas.8 | teks/ner (CRF; BERT tanpa submission) | ner (CRF) | ⚠️ | T11c, B2 |
| P.tugas.9 | umum/teks_regex#4 | – | ✅ | – |
| P.tugas.10 | – | pos_tagging | ✅ | – |
| P.tugas.11 | teks/aspek | absa | ✅ | – |
| P.tugas.12 | teks/pasangan | pasangan_teks | ✅ | T5 (cross-encoder) |
| P.tugas.13 | – | – | ❌ | T10 (alt pasangan + regresi) |
| P.tugas.14 | teks/kemiripan/retrieval; tabular/ranking | retrieval | ✅ | B7 |
| P.tugas.15 | teks/kemiripan/retrieval#7–#9 | tanya_jawab, rag_llm | ⚠️ | B8 |
| P.tugas.16 | – | – | ❌ | diterima (= klasifikasi + NER) |
| P.tugas.17 | demo emot di teks/klasifikasi/* | ujaran_kebencian, analisis_kesalahan | ✅ | – |
| P.tugas.18 | – | – | ❌ | diterima (char n-gram di T2) |
| P.tugas.19 | – | bahasa_daerah | ✅ | – |
| P.tugas.20 | teks/pengelompokan/topik | topik 0.3943 (NMF 0.4384), topik_bertopic, topik_dinamis | ✅ | B3 |
| P.tugas.21 | – | clustering 0.2691 (SVD+KMeans 0.4161) | ✅ | B3 |
| P.tugas.22 | tabular/anomali | anomali_teks | ✅ | – |
| P.tugas.23 | teks/dasar/praproses#2; teks/kemiripan/retrieval#10–#11 | – | ✅ | T2, T11b (train↔test) |
| P.tugas.24 | – | kata_kunci_ringkasan | ✅ | – |
| P.tugas.25 | umum/teks_regex#8 | – | ✅ | – |
| P.tugas.26 | teks/zero_shot | zero_shot_nli, llm_prompt | ✅ | – |
| P.tugas.27 | – | llm_prompt; klasifikasi_embedding alt kNN | ⚠️ | B6 |
| P.tugas.28 | teks/zero_shot#6; tabular/ensembling#13 | semi_supervised_teks | ✅ | – |
| P.tugas.29 | – | weak_supervision | ✅ | – |
| P.tugas.30 | – | anotasi_teks | ⚠️ | T1 (ipywidgets) |
| P.tugas.31 | – | – | ❌ | diterima |
| P.tugas.32 | teks/ringkasan | kata_kunci_ringkasan | ✅ | – |
| P.tugas.33 | – | generasi (Qwen 0.5B) | ⚠️ | B9 |
| P.tugas.34 | teks/dasar/praproses#3 (slang) | generasi; ujaran_kebencian (kamus alay) | ⚠️ | B9 |
| P.tugas.35 | teks/pengelompokan/topik#7 | topik_dinamis | ✅ | – |

### 1.3 Representasi

| kode | arsenal | contoh | status | butir |
|---|---|---|---|---|
| P.rep.1 | umum/teks_regex | weak_supervision, anotasi_teks | ✅ | – |
| P.rep.2 | – | – | ❌ | B11 |
| P.rep.3 | teks/dasar/praproses#3–#4; umum/teks_regex#3 | ujaran_kebencian, augmentasi (kamus alay) | ✅ | T2 (alt) |
| P.rep.4 | – | klasifikasi_tfidf alt Count 0.8335 | ✅ | – |
| P.rep.5 | teks/klasifikasi/tfidf#2 | banyak | ✅ | – |
| P.rep.6 | teks/klasifikasi/tfidf#2 | multilabel alt, bahasa_daerah | ✅ | – |
| P.rep.7 | – | – | ❌ | diterima (RAM 32 GB) |
| P.rep.8 | teks/kemiripan/retrieval#3 | retrieval | ✅ | – |
| P.rep.9 | – | – | ❌ | T2, T11a |
| P.rep.10 | – | – | ❌ | diterima (gensim tidak terpasang) |
| P.rep.11 | teks/klasifikasi/embedding#2–#3; jembatan/teks_ke_vektor#4 | klasifikasi_embedding | ✅ | T4 (e5_base) |
| P.rep.12 | jembatan/teks_ke_vektor#5–#6 | klasifikasi_embedding alt indobert 0.8660 | ✅ | – |
| P.rep.13 | teks/klasifikasi/finetune_bert | klasifikasi_finetune | ⚠️ | T5 |
| P.rep.14 | – | analisis_kesalahan (jumlah kata) | ⚠️ | T8 |
| P.rep.15 | jembatan/vektor_ke_model | – | ⚠️ | T8 |
| P.rep.16 | – | clustering alt SVD, topik_bertopic | ✅ | T8 (SVD) |
| P.rep.17 | – | – | ❌ | diterima |
| P.rep.18 | – | – | ❌ | bukan fokus |

### 1.4 Keluarga model

| kode | arsenal | contoh | status | butir |
|---|---|---|---|---|
| P.kel.1 | teks/klasifikasi/tfidf, teks/multilabel, teks/aspek | banyak | ✅ | – |
| P.kel.2 | tabular/klasifikasi/gbdt; jembatan/vektor_ke_model#6 | alt LGBM di klasifikasi_tfidf 0.8044 / _embedding 0.8400 | ⚠️ | T8 (+CatBoost text_features) |
| P.kel.3 | teks/klasifikasi/finetune_bert | klasifikasi_finetune (indobert, distilbert_id, minilm) | ⚠️ | T4, T5 |
| P.kel.4 | – | – | ❌ | T10 |
| P.kel.5 | teks/ner#10–#15 | – | ⚠️ | T11c, B2 |
| P.kel.6 | teks/kemiripan/retrieval#7–#9 | tanya_jawab | ✅ | B8 |
| P.kel.7 | – | – | ❌ | T5 (pasangan), B7 (reranker) |
| P.kel.8 | teks/klasifikasi/embedding, jembatan, retrieval | klasifikasi_embedding, retrieval, anomali_teks | ✅ | T4 |
| P.kel.9 | – | – | ❌ | B6 |
| P.kel.10 | – | – | ❌ | B6 |
| P.kel.11 | – | – | ❌ | B9 |
| P.kel.12 | – | – | ❌ | B1 |
| P.kel.13 | teks/ner#6 | ner, pos_tagging | ✅ | – |
| P.kel.14 | – | – | ❌ | diterima (TF-IDF menutup) |
| P.kel.15 | – | – | ❌ | diterima |
| P.kel.16 | teks/zero_shot; teks/pasangan#5 | zero_shot_nli | ✅ | – |
| P.kel.17 | – | – | ❌ | B12 |
| P.kel.18 | teks/pengelompokan/topik#4 | topik_bertopic | ✅ | – |
| P.kel.19 | – | – | ❌ | B11 |
| P.kel.20 | – | – | ❌ | B12 (bukan fokus) |
| P.kel.21 | – | llm_prompt, generasi, rag_llm | ✅ | – |
| P.kel.22 | – | – | ❌ | bukan fokus |

### 1.5 Metode ML

| kode | arsenal | contoh | status | butir |
|---|---|---|---|---|
| P.met.k.1 | teks/klasifikasi/tfidf#2–#4 | klasifikasi_tfidf | ✅ | T2 |
| P.met.k.2 | teks/klasifikasi/tfidf#4 | klasifikasi_tfidf alt ComplementNB 0.8271 | ✅ | T2 |
| P.met.k.3 | – | – | ❌ | T2, T11a |
| P.met.k.4 | – | klasifikasi_tfidf alt SGD 0.8531 | ✅ | – |
| P.met.k.5 | – (markdown tfidf) | klasifikasi_tfidf alt Calibrated 0.8633 | ✅ | T2 |
| P.met.k.6 | tabular/klasifikasi/gbdt | alt LGBM 0.8044 / 0.8400 | ⚠️ | T8 |
| P.met.k.7 | teks/klasifikasi/embedding#4 (ber-#) | klasifikasi_embedding alt kNN 0.8040 | ✅ | – |
| P.met.k.8 | teks/klasifikasi/embedding#5–#6 | – | ✅ | T6 |
| P.met.f.1 | teks/klasifikasi/finetune_bert#2 (persentil) | – | ⚠️ | T5 |
| P.met.f.2 | finetune_bert#4 (padding dinamis) | klasifikasi_finetune | ⚠️ | T5 (urut panjang) |
| P.met.f.3 | finetune_bert#6 (bf16) | – | ⚠️ | T5 |
| P.met.f.4 | – | – | ❌ | T5 |
| P.met.f.5 | – | – | ❌ | T5 |
| P.met.f.6 | – | – | ❌ | T5 |
| P.met.f.7 | – | – | ❌ | T5 |
| P.met.f.8 | – | – | ❌ | T10 |
| P.met.f.9 | – | – | ❌ | T10 |
| P.met.f.10 | – | – | ❌ | B5 |
| P.met.f.11 | – | – | ❌ | T5 |
| P.met.f.12 | – | – | ❌ | B5 |
| P.met.f.13 | – | – | ❌ | T5 |
| P.met.f.14 | – | – | ❌ | diterima (EMA cukup) |
| P.met.f.15 | – | – | ❌ | B1 |
| P.met.f.16 | – | – | ❌ | T5 |
| P.met.f.17 | finetune_bert#5 | klasifikasi_finetune | ✅ | – |
| P.met.f.18 | finetune_bert#7 (patience) | – | ⚠️ | T5 |
| P.met.f.19 | – | – | ❌ | B4 |
| P.met.f.20 | – | – | ❌ | diterima (bitsandbytes tidak ada; VRAM ditutup grad checkpointing T5) |
| P.met.l.1 | teks/klasifikasi/tfidf#2; finetune_bert#5 (ber-#) | absa, inggris, ujaran_kebencian, weak_supervision | ✅ | T2, T5 |
| P.met.l.2 | – | – | ❌ | T5 |
| P.met.l.3 | – | – | ❌ | diterima (bobot kelas + focal) |
| P.met.l.4 | – | – | ❌ | T5 |
| P.met.l.5 | – | – | ❌ | T9 |
| P.met.l.6 | – | – | ❌ | diterima |
| P.met.l.7 | tabular/metrik#11 (ambang) | – | ⚠️ | T10 |
| P.met.l.8 | – | – | ❌ | B6 |
| P.met.l.9 | teks/klasifikasi/tfidf#9 | regresi_teks | ✅ | T10 |
| P.met.l.10 | – | – | ❌ | diterima |
| P.met.d.1 | teks/zero_shot#6; tabular/ensembling#13 | semi_supervised_teks | ✅ | T6 (tulis), T5 (alt) |
| P.met.d.2 | – | semi_supervised_teks | ✅ | – |
| P.met.d.3 | – | – | ❌ | diterima |
| P.met.d.4 | – | augmentasi | ✅ | – |
| P.met.d.5 | – | augmentasi (slang) | ✅ | – |
| P.met.d.6 | – | – | ❌ | bukan fokus |
| P.met.d.7 | – | – | ❌ | diterima |
| P.met.d.8 | tabular/imbalanced#6 | – | ⚠️ | diterima (bobot kelas) |
| P.met.d.9 | teks/dasar/praproses#2 (deteksi) | – | ⚠️ | T2 |
| P.met.d.10 | tabular/klasifikasi/gbdt#12–#13 | label_berisik | ✅ | – |
| P.met.h.1 | teks/klasifikasi/embedding#5 (grid bobot) | – | ⚠️ | – |
| P.met.h.2 | – | – | ❌ | diterima (Optuna) |
| P.met.h.3 | tabular/tuning/optuna_tuning | tuning | ✅ | – |
| P.met.h.4 | – | – | ❌ | diterima |
| P.met.e.1 | tabular/ensembling#8 | ensemble | ✅ | T6 |
| P.met.e.2 | – (RRF hanya retrieval) | – | ❌ | T6, T11d |
| P.met.e.3 | tabular/ensembling#9 (logloss saja) | – | ⚠️ | T6 |
| P.met.e.4 | tabular/ensembling#9 | – | ⚠️ | T6 |
| P.met.e.5 | – | – | ❌ | T6 (alt) |
| P.met.e.6 | tabular/ensembling#11; jembatan/vektor_ke_model#9 | ensemble alt StackingClassifier | ✅ | T6 |
| P.met.e.7 | teks/klasifikasi/embedding#5 | – | ✅ | – |
| P.met.e.8 | tabular/ensembling#12 (LGBM) | – | ⚠️ | T5 (SEEDS) |
| P.met.e.9 | – | – | ❌ | diterima |
| P.met.e.10 | tabular/ensembling#3 | – | ⚠️ | T2, T5 |
| P.met.p.1 | – (markdown) | klasifikasi_tfidf alt Calibrated | ⚠️ | T11e |
| P.met.p.2 | – | – | ❌ | T6, T11e |
| P.met.p.3 | tabular/imbalanced#5 | – | ✅ | T2 (alt) |
| P.met.p.4 | tabular/metrik#14 | – | ⚠️ | T2, T6 |
| P.met.p.5 | teks/multilabel#5 | ujaran_kebencian | ✅ | T9 |
| P.met.p.6 | tabular/metrik#11 | – | ✅ | T10 |
| P.met.p.7 | teks/aspek#6 | – | ⚠️ | diterima |
| P.met.p.8 | – | – | ❌ | diterima |
| P.met.p.9 | – | – | ❌ | diterima |
| P.met.p.10 | teks/klasifikasi/tfidf#9 | regresi_teks | ✅ | – |

### 1.6 Jebakan

| kode | arsenal | contoh | status | butir |
|---|---|---|---|---|
| P.jeb.1 | tfidf balanced; tabular/metrik#14; tabular/imbalanced | ujaran_kebencian, absa | ✅ | T2, T5 |
| P.jeb.2 | teks/dasar/praproses#2; umum/tabel_bersih#9 | – | ✅ | T2 |
| P.jeb.3 | teks/dasar/praproses#2 | – | ✅ | T2 |
| P.jeb.4 | – | – | ❌ | T2, T11b |
| P.jeb.5 | tabular/dasar/validasi#4 (biner) | – | ⚠️ | T8 |
| P.jeb.6 | tabular/klasifikasi/gbdt#12 | label_berisik | ✅ | – |
| P.jeb.7 | tabular/dasar/validasi#5 (numerik) | – | ⚠️ | T2, T11b |
| P.jeb.8 | tabular/dasar/validasi#7–#8 | topik_dinamis | ⚠️ | T2 (alt) |
| P.jeb.9 | teks/dasar/praproses#3; umum/teks_regex#3 | ujaran_kebencian, augmentasi | ✅ | T2 (alt) |
| P.jeb.10 | – | bahasa_daerah | ✅ | T5 (xlmr_base) |
| P.jeb.11 | teks/klasifikasi/tfidf (char) | alt char_wb | ✅ | – |
| P.jeb.12 | – (dibuang regex) | inggris (demojize) | ⚠️ | T1, T2 |
| P.jeb.13 | jembatan/teks_ke_vektor#2 (fillna) | sebagian fillna | ⚠️ | T2, T3 |
| P.jeb.14 | teks/klasifikasi/finetune_bert#2 | – | ⚠️ | T5 |
| P.jeb.15 | umum/teks_regex#3 | ujaran_kebencian, augmentasi | ✅ | – |
| P.jeb.16 | – | – | ❌ | T3 |
| P.jeb.17 | teks/multilabel#3 | klasifikasi_finetune (LABELS manual) | ⚠️ | T3, T5 |
| P.jeb.18 | umum/simpan_muat#14–#15 | – | ⚠️ | T3, semua butir contoh |
| P.jeb.19 | – | – | ❌ | diterima (tak bisa ditebak) |
| P.jeb.20 | – | – | ❌ | T3 |
| P.jeb.21 | – | analisis_kesalahan (negasi), inggris | ⚠️ | – |
| P.jeb.22 | tabular/dasar/validasi#2 | bahasa_daerah | ⚠️ | T2 (alt) |
| P.jeb.23 | – | zero_shot_nli, semi_supervised_teks | ⚠️ | T5 (catatan data besar) |
| P.jeb.24 | – | – | ❌ | T2 |
| P.jeb.25 | umum/teks_regex#7 | – (Laya v2) | ✅ | – |
| P.jeb.26 | tabular/metrik | – | ⚠️ | – |
| P.jeb.27 | teks/dasar/praproses#3 (kata dipertahankan) | inggris | ✅ | – |

### 1.7 Evaluasi

| kode | arsenal | contoh | status | butir |
|---|---|---|---|---|
| P.eval.1 | tabular/metrik#3 | banyak | ✅ | – |
| P.eval.2 | tabular/metrik#3 | banyak | ✅ | – |
| P.eval.3 | tabular/metrik#3 | ujaran_kebencian | ✅ | – |
| P.eval.4 | tabular/metrik#3 | ner (F1w) | ✅ | – |
| P.eval.5 | teks/klasifikasi/tfidf#3 | banyak | ✅ | – |
| P.eval.6 | tabular/metrik#7 | regresi_teks | ✅ | T10 |
| P.eval.7 | tabular/metrik#6 | regresi_teks | ✅ | – |
| P.eval.8 | – | – | ❌ | T10 (alt STS) |
| P.eval.9 | teks/ner#4–#5 | – (ner memakai F1 per token) | ✅ | B10 |
| P.eval.10 | teks/kemiripan/retrieval#9 | tanya_jawab, rag_llm | ✅ | – |
| P.eval.11 | tabular/metrik#8 | – | ✅ | – |
| P.eval.12 | tabular/metrik#9; tabular/ranking#2 | retrieval | ✅ | – |
| P.eval.13 | teks/kemiripan/retrieval#2 | retrieval | ✅ | – |
| P.eval.14 | teks/ringkasan#3–#5 | generasi | ✅ | – |
| P.eval.15 | – | generasi | ✅ | – |
| P.eval.16 | tabular/metrik#5; tabular/ensembling#2 | – | ✅ | T6 |
| P.eval.17 | tabular/metrik#4 | anomali_teks | ✅ | – |
| P.eval.18 | teks/multilabel#5 | multilabel (hamming, subset) | ✅ | – |
| P.eval.19 | teks/pengelompokan/topik#8 | clustering, topik | ✅ | – |
| P.eval.20 | teks/klasifikasi/tfidf#3 | banyak | ✅ | – |
| P.eval.21 | teks/multilabel#4; teks/aspek#3 | analisis_kesalahan, label_berisik | ⚠️ | T2 (format OOF) |
| P.eval.22 | – | – | ❌ | T9 (catatan: KFold biasa) |
| P.eval.23 | tabular/dasar/validasi#2 | bahasa_daerah | ✅ | T2 (alt) |
| P.eval.24 | tabular/dasar/validasi#7–#8 | – | ✅ | T2 (alt) |
| P.eval.25 | tabular/dasar/validasi#5 (numerik) | – | ⚠️ | T2, T11b |
| P.eval.26 | – | – | ❌ | T2, T5, T6 |
| P.eval.27 | teks/klasifikasi/tfidf#3 | analisis_kesalahan | ✅ | – |
| P.eval.28 | – | – | ❌ | diterima |
| P.eval.29 | umum/simpan_muat#15 (minimal) | – | ⚠️ | T3 |

### 1.8 Penjelasan, presentasi, eksekusi

| kode | arsenal | contoh | status | butir |
|---|---|---|---|---|
| P.pres.1 | teks/dasar/praproses#5 | – | ✅ | – |
| P.pres.2 | teks/klasifikasi/tfidf#5 | – | ✅ | – |
| P.pres.3 | teks/klasifikasi/tfidf#3; finetune_bert#8 | analisis_kesalahan | ✅ | – |
| P.pres.4 | umum/plot | klasifikasi_* | ✅ | – |
| P.pres.5 | – | – | ❌ | diterima (captum tidak terpasang) |
| P.pres.6 | teks/pengelompokan/topik#4 | topik_bertopic | ⚠️ | diterima |
| P.pres.7 | – | – | ❌ | T6 (tabel dari percobaan.csv) |
| P.pres.8 | teks/dasar/praproses#2 | banyak | ✅ | – |
| P.pres.9 | teks/dasar/praproses#2 | – | ⚠️ | T2 (sel Temuan) |
| P.pres.10 | – | – | ❌ | diterima (PPT pengguna) |
| P.pres.11 | – | – (skor hanya di KATALOG) | ⚠️ | T6 |
| P.pres.12 | umum/simpan_muat#16 (zip outputs saja) | – | ⚠️ | T7 |
| P.eks.1 | path lokal | path lokal; finetune `local_files_only` | ⚠️ | T12 |
| P.eks.2 | teks/klasifikasi/finetune_bert (bf16) | – | ⚠️ | T5 (ukur) |
| P.eks.3 | – | – | ❌ | T5 |
| P.eks.4 | sparse | sparse | ✅ | – |
| P.eks.5 | tabular/ensembling#6; finetune_bert#9 | – | ⚠️ | T2, T5 |
| P.eks.6 | – | – | ❌ | bagian 3, T12 |
| P.eks.7 | SEED | SEED | ✅ | – |
| P.eks.8 | – (tidak dicek) | – (tidak dicek) | ⚠️ | diterima |
| P.eks.9 | – | – | ❌ | bagian 3 |
| P.eks.10 | baris DEVICE | baris DEVICE | ✅ | – |

## 2. Prioritas

Urutan = peluang dipakai saat final × naiknya skor/nilai metode ÷ usaha. Peluang = dugaan: final GEMASTIK
dan soal teks Indonesia paling sering klasifikasi (biner/multikelas, F1 makro, timpang, teks medsos kotor);
berikutnya multilabel, teks + kolom tabel (pola 2024), regresi/ordinal; NER/QA/generasi lebih jarang.

| urutan | butir | kenapa |
|---|---|---|
| 1 | T1 lingkungan + uji dasar | tanpa Python yang jalan semua uji GAGAL; 24 dari 32 contoh/text belum punya bukti jalan |
| 2 | T2 `baseline_oof` | submission pertama ≤ 45 menit untuk soal klasifikasi apa pun; fondasi OOF untuk ensemble; menutup 20+ sel ⚠️/❌ |
| 3 | T3 baca + validator | format file/submission aneh = gagal total (panitia suka kejutan); murah |
| 4 | T4 model baru | prasyarat ukur VRAM dan pilihan checkpoint di T5; harus selagi ada internet |
| 5 | T5 `finetune_oof` | sumber skor terbesar (IndoBERT 0.9242 vs TF-IDF 0.8636 di smsa); banyak metode SOTA cukup jadi alternatif ber-# |
| 6 | T6 `ensemble_oof` | hill climbing/Caruana sudah ada tapi terkunci logloss; F1 makro + pengali kelas = naik murah di menit akhir |
| 7 | T7 paket kumpul | zip wajib (gugur); 1 jam |
| 8 | T8 `teks_tabular` | pola final 2024 (fitur teks paling penting); sekarang tidak ada contoh end-to-end |
| 9 | T9 `finetune_multilabel` | multilabel umum di data Indonesia (HS, aspek); BERT belum ada |
| 10 | T10 `finetune_regresi` | regresi/ordinal + QWK; contoh head kustom (pooling, multi-sample dropout) |
| 11 | T11 tambahan arsenal | NB-SVM, train↔test, submission NER-BERT, rank average, kalibrasi; kecil-kecil |
| 12 | T12 gladi | satu-satunya bukti runbook muat 5 jam dan jalan offline |
| 13 | T13 gabung file bersama | KATALOG/PANDUAN/PETA/SOURCES jadi peta pengguna saat lomba |

## 3. Jalur 5 jam (runbook)

Belum diukur; angka menit = target. T12 mengukurnya dan memperbarui tabel ini.

| waktu | langkah | alat |
|---|---|---|
| 0:00–0:10 | baca soal, metrik, format `sample_submission`, kuota submit; `nvidia-smi`; cek GPU di Python | – |
| 0:10–0:20 | buka data: ukuran, kolom, encoding/pemisah; tentukan bentuk soal (tabel di bawah) | `arsenal/umum/baca_folder` (T3) |
| 0:20–0:45 | `baseline_oof`: isi Konfigurasi → Run All → validator → **submit #1** | T2 |
| 0:45–1:00 | baca sel Temuan (duplikat, konflik label, overlap train↔test, adversarial AUC, panjang token, kelas timpang); putuskan split (stratified/group/waktu) dan pembersihan | T2 |
| 1:00–2:30 | GPU: `finetune_oof` IndoBERT base, `FOLDS_TO_RUN=[0]` dulu untuk waktu/epoch, lalu semua fold. Sambil menunggu (CPU): varian baseline (char, NB-SVM, LinearSVC), `teks_tabular` bila ada kolom tabel | T5, T2, T8 |
| 2:30–3:30 | GPU: checkpoint kedua (indobertweet untuk medsos, xlmr_base untuk campur bahasa, indobert_large bila waktu/epoch cukup) atau seed kedua; **submit** yang CV-nya naik | T5 |
| 3:30–4:00 | `ensemble_oof`: hill climbing di metrik lomba + pengali kelas → **submit** | T6 |
| 4:00–4:30 | pilih final dari CV (bukan LB saja); validator; paket zip; tabel ablation dari `percobaan.csv` | T6, T7 |
| 4:30–5:00 | figur presentasi (confusion matrix, kata khas, contoh salah, ablation); cek isi zip | arsenal teks/praproses#5, tfidf#5 |

Bentuk soal → notebook (✅ ada, T = dibuat butir):

| bentuk | baseline (≤ 45 menit) | lanjutan GPU | gabung |
|---|---|---|---|
| klasifikasi satu/banyak kolom teks | `baseline_oof` (T2) | `finetune_oof` (T5) | `ensemble_oof` (T6) |
| teks + kolom tabel | `teks_tabular` (T8) | `finetune_oof` pada kolom teks | `ensemble_oof` |
| multilabel | `multilabel` / `ujaran_kebencian` ✅ | `finetune_multilabel` (T9) | rata-rata proba + ambang per label (T9) |
| regresi / ordinal / esai | `regresi_teks` ✅ | `finetune_regresi` (T10) | `ensemble_oof` metrik RMSE/QWK (T6 alt) |
| pasangan (NLI, parafrase) | `pasangan_teks` ✅ | `finetune_oof` alt TEXT_B (T5) | `ensemble_oof` |
| NER / POS | `ner`, `pos_tagging` (CRF) ✅ | arsenal `teks/ner` #10–#16 BERT (T11c) | – |
| ABSA | `absa` ✅ | – | – |
| QA / ranking | `tanya_jawab`, `retrieval` ✅ | – | – |
| topik / klaster / anomali | `topik`, `topik_bertopic`, `clustering`, `anomali_teks` ✅ | – | – |
| label sedikit / tanpa label | `zero_shot_nli`, `semi_supervised_teks`, `weak_supervision`, `anotasi_teks` ✅ | – | – |
| bahasa daerah / Inggris | `bahasa_daerah`, `inggris` ✅ | `finetune_oof` xlmr_base / distilbert_en | `ensemble_oof` |
| ringkasan / generasi | `kata_kunci_ringkasan`, arsenal `teks/ringkasan` (LEAD) ✅ | `generasi` (Qwen) | – |

## 4. Butir kerja

### Kesepakatan bersama (wajib untuk semua butir contoh baru)

- Gaya `contoh/`: urutan Import → Konfigurasi → Baca → EDA → Split → Model → Evaluasi → Latih/prediksi test →
  Submission → Cek submission → `## Cek dengan kunci jawaban (HAPUS saat lomba)`. Baris yang diubah saat lomba
  diberi `# GANTI saat lomba`; alternatif ber-`#` masing-masing diberi komentar kapan dipakai. Lint
  (`_alat/lint.py`) 0 masalah. Disimpan tanpa output.
- Folder: `OUTPUT_FOLDER = "../../../outputs/text/<nama_notebook>"`, `OOF_FOLDER = "../../../outputs/text/oof"`
  (di luar repo, `Z:/GEMASTIK/Vignate-GEMASTIK/outputs/`).
- **Format OOF** (agar `ensemble_oof` bisa memakai keluaran notebook mana pun): `oof_<MODEL_NAME>.npy`
  (float32, n_train × n_kelas; regresi n_train; multilabel n_train × n_label) dan `test_<MODEL_NAME>.npy`
  (n_test × sama). Urutan kelas = `np.unique(train[TARGET])`. Fold = `StratifiedKFold(N_FOLDS=5, shuffle=True,
  random_state=SEED=42)` pada urutan baris train apa adanya (regresi/multilabel: `KFold` seed sama). Notebook
  yang membuang baris train sebelum fold tidak menulis ke `OOF_FOLDER`. Test = rata-rata model per fold (tidak
  latih ulang di semua data; alasan di `_memori/keputusan.md`).
- `percobaan.csv` di `OOF_FOLDER`: satu baris per run (waktu, notebook, model, metrik, skor_cv, detik, catatan).
- Submission: baca `SAMPLE_CSV`; `submission = pd.DataFrame()`; isi kolom id dari sample, prediksi dipetakan
  lewat id; lalu sel validator (spesifikasi di T3). Data latihan tidak punya sample: sel
  "Buat sample_submission demo (HAPUS saat lomba)" menulisnya ke `OUTPUT_FOLDER`.
- Model baru hanya dari `pretrained/` lewat path lokal; tidak ada unduhan saat notebook jalan.

### Uji

- Arsenal: `python _alat/lint.py <file>` lalu `python _alat/uji_arsenal.py <file>:<sel> [--set NAMA=nilai]`
  (nomor sel = sel kode; data demo dan pretrained di `Z:/GEMASTIK/Vignate-GEMASTIK/`, pakai `--set` untuk path,
  lihat docstring alat).
- Contoh: `cmd /c "_alat\bangun_contoh.bat text\<nama> < nul"` setelah T1 menyediakan venv. Kalau pakai
  Python sistem: dari root repo `python -m jupyter nbconvert --to notebook --execute --output-dir
  ../outputs/uji_contoh/text --ExecutePreprocessor.timeout=1800 contoh/text/<nama>.ipynb` → keluar 0 = LOLOS.
  `hasil_build.txt` ditimpa tiap run: salin hasilnya ke STATUS butir.
- Tiap butir: sekali di data demo + sekali di data **berbentuk beda** (disebut per butir).
- Butir GPU: catat GPU (`nvidia-smi`), VRAM maks (`torch.cuda.max_memory_allocated`), detik per epoch. Satu
  butir GPU berjalan pada satu waktu (laptop pernah mati saat beban berat; batch 8–16).

### Gelombang dan file bersama

- G0 (mulai sekarang): T1 (GPU), T4 (jaringan; boleh bersamaan dengan T1).
- G1 (CPU, boleh paralel, mulai sekarang): T2, T3, T7, T8, T11 (a, b, d, e). T3 dan T7 sama-sama mengedit
  `arsenal/umum/simpan_muat.ipynb`: kerjakan berurutan di satu sesi (T3 lalu T7), jangan paralel.
- G2 (GPU, satu per satu, setelah T1 dan T4): T5 → T9 → T10 → T11c.
- G1b (CPU): T6 setelah T2 punya OOF; uji penuh setelah T5.
- G3 (berurutan, terakhir): T12 → T13.
- Pemilik file bersama: `requirements.txt` hanya T1; `_alat/unduh_model.py` hanya T4;
  `contoh/KATALOG.md`, `contoh/PANDUAN.md`, `arsenal/PETA.md`, `data_latihan/SOURCES_text.md` hanya T13.
  Butir lain menulis blok usulan (baris KATALOG/PANDUAN/PETA/SOURCES, paket yang dibutuhkan) di STATUS-nya
  sendiri; T13 menyalinnya. Di `RENCANA_TEKS.md` tiap pekerja hanya mengubah STATUS butirnya (baca ulang tepat
  sebelum mengedit). `_memori/log.md`: tambah entri di atas, baca ulang tepat sebelum mengedit.

### T1. Lingkungan + uji dasar semua notebook teks

- Tujuan: bukti LOLOS/GAGAL + waktu untuk 32 `contoh/text` dan 13 notebook `arsenal/teks` + `jembatan/teks_ke_vektor`
  di Python yang dipakai; paket hilang masuk requirements.
- Peta: P.eks.1, P.eks.10, P.tugas.30, P.jeb.12. Pilar: keduanya (uji). Nilai: tinggi.
- File: `requirements.txt` (tambah `emoji`, `ipywidgets`; `pyarrow` bila `pip show pyarrow` tidak menunjukkan
  paket lain yang membutuhkannya). Notebook hanya diperbaiki bila ≤ 5 baris; yang lebih besar jadi baris
  baru di STATUS (perencana/pengguna memutuskan).
- Langkah: jawaban pertanyaan 1–2 → pasang → jalankan tiap notebook satu per satu (GPU berurutan) → tabel
  notebook | LOLOS/GAGAL | menit | penyebab.
- Uji terima: tabel lengkap di STATUS; `inggris` dan `anotasi_teks` LOLOS (Run All tanpa klik).
  Bentuk beda: tidak berlaku (uji dasar).
- Dependensi: pertanyaan 1, 2. Gelombang G0. Ukuran M, ± 3 jam (sebagian besar menunggu Run All).
- STATUS: selesai (2026-11-06). Run All 32/32 contoh/text LOLOS (0 GAGAL), uji arsenal 13/13 LOLOS (12 arsenal/teks + jembatan/teks_ke_vektor); `inggris` dan `anotasi_teks` LOLOS.
  - Lingkungan: `setup.bat` membuat `Z:/GEMASTIK/Vignate-GEMASTIK/.venv` (Python 3.13.15), seluruh requirements terpasang, `pip check` bersih, GPU True. `emoji==2.16.0`, `ipywidgets==8.1.9` ditambah ke requirements.txt; pyarrow tidak (terbawa streamlit/openfe/autogluon.common).
  - TEMUAN: kernelspec `python3` pengguna (`%APPDATA%\jupyter\kernels\python3`) menunjuk `Z:\GEMASTIK-DATABASE\.venv` yang tidak ada, jadi `bangun_contoh.bat` apa adanya GAGAL semua (FileNotFoundError). Uji ini memakai `JUPYTER_PATH` ke kernelspec venv + venv di depan PATH. Usulan (pengguna/perencana): `python -m ipykernel install --user --name python3` dari venv, atau `bangun_contoh.bat` mengisi JUPYTER_PATH. Tidak diubah oleh T1.
  - Catatan: menit memuat kontensi dengan unduhan T4 dan uji arsenal CPU yang berjalan bersamaan; baseline_oof (T2) tidak diuji; klasifikasi_embedding = versi kerja T4 (belum commit).

  | notebook (contoh/text) | hasil | menit |
  |---|---|---|
  | inggris | LOLOS | 0.32 |
  | anotasi_teks | LOLOS | 0.32 |
  | absa | LOLOS | 1.19 |
  | analisis_kesalahan | LOLOS | 1.54 |
  | anomali_teks | LOLOS | 0.5 |
  | augmentasi | LOLOS | 0.2 |
  | bahasa_daerah | LOLOS | 0.43 |
  | clustering | LOLOS | 0.16 |
  | ensemble | LOLOS | 0.31 |
  | kata_kunci_ringkasan | LOLOS | 0.16 |
  | klasifikasi_embedding | LOLOS | 0.91 |
  | klasifikasi_tfidf | LOLOS | 0.31 |
  | label_berisik | LOLOS | 0.51 |
  | multilabel | LOLOS | 0.32 |
  | ner | LOLOS | 0.61 |
  | pasangan_teks | LOLOS | 0.68 |
  | pos_tagging | LOLOS | 1.97 |
  | regresi_teks | LOLOS | 0.4 |
  | retrieval | LOLOS | 1.26 |
  | semi_supervised_teks | LOLOS | 0.71 |
  | topik | LOLOS | 0.43 |
  | topik_bertopic | LOLOS | 1.34 |
  | topik_dinamis | LOLOS | 0.26 |
  | tuning | LOLOS | 5.84 |
  | ujaran_kebencian | LOLOS | 0.39 |
  | weak_supervision | LOLOS | 0.25 |
  | generasi | LOLOS | 2 |
  | klasifikasi_finetune | LOLOS | 2.13 |
  | llm_prompt | LOLOS | 2.57 |
  | rag_llm | LOLOS | 2.95 |
  | tanya_jawab | LOLOS | 1.43 |
  | zero_shot_nli | LOLOS | 0.73 |

  | arsenal (data demo, semua sel aktif, `uji_arsenal.py`) | hasil | menit |
  |---|---|---|
  | praproses | LOLOS | 0.22 |
  | tfidf | LOLOS | 2.00 |
  | multilabel | LOLOS | 0.08 |
  | aspek_sentimen | LOLOS | 1.30 |
  | ringkasan | LOLOS | 0.40 |
  | topik | LOLOS | 2.73 |
  | ner | LOLOS | 2.45 |
  | retrieval | LOLOS | 0.78 |
  | embedding | LOLOS | 0.30 |
  | pasangan | LOLOS | 0.28 |
  | zero_shot_nli | LOLOS | 0.45 |
  | jembatan/teks_ke_vektor | LOLOS | 0.47 |
  | finetune_bert (`--set EPOCHS=1 BATCH_SIZE=8`, F1 makro test 0.7173) | LOLOS | 0.68 |

  - Perintah: `cmd /c "_alatangun_contoh.bat text\<nama> < nul"` per notebook (driver serial di scratchpad; `JUPYTER_PATH` -> kernelspec venv, venv di depan PATH); arsenal: `python _alat/uji_arsenal.py <file> --set DEMO_FOLDER=... --set PRETRAINED_FOLDER=...` (venv). GPU: RTX 5060 Laptop 8 GB, driver 617.42, suhu puncak 58 C; tidak diukur: VRAM puncak/detik per epoch (T1 bukan butir model; finetune_bert hanya 1 epoch).
  - Sumber tidak diubah T1: tidak ada notebook yang diedit, `git status` hanya menunjukkan file milik butir lain (T2/T3/T4/T8). Batas uji: arsenal hanya data demo, `# UBAH` tidak diubah selain path dan EPOCHS/BATCH_SIZE finetune; hasil LOLOS = keluar 0, nilai skor tidak dibandingkan dengan KATALOG.
  - Usulan untuk pengguna/perencana (di luar file T1): kernelspec `python3` pengguna menunjuk `Z:\GEMASTIK-DATABASE\.venv` (tidak ada) sehingga `bangun_contoh.bat` apa adanya GAGAL semua; perbaiki dengan `python -m ipykernel install --user --name python3` dari venv, atau `bangun_contoh.bat` mengisi `JUPYTER_PATH`. Catatan menit: berbagi CPU/jaringan dengan T4 dan uji arsenal CPU.

### T2. `contoh/text/baseline_oof.ipynb` (baru)

- Tujuan: jalur submission pertama ≤ 45 menit untuk klasifikasi teks apa pun, sekaligus OOF bersama.
- Peta: P.data.1, .2, .10, .12, P.tugas.1–2, P.met.k.1–3, k.5, l.1, d.9, e.10, p.3, p.4, P.jeb.1–4, .7–9, .13,
  .22, .24, P.eval.21, .23–26, P.pres.9, P.eks.5. Pilar: contoh. Nilai: tertinggi.
- Notebook baru karena alur latih beda (k-fold + OOF + sample_submission), bukan ganti satu fungsi.
- Isi:
  - Konfigurasi: `DATA_FOLDER`, `TRAIN_CSV`, `TEST_CSV`, `SAMPLE_CSV`, `ID_COL`, `TEXT_COLS` (list; lebih dari
    satu = digabung dengan spasi), `TARGET`, `N_FOLDS`, `SEED`, `MODEL_NAME`, folder.
  - Sel Temuan (dicetak sebagai daftar): sebaran kelas, panjang kata, teks kosong, kembar, kembar beda label,
    teks test yang persis ada di train, hampir kembar train↔test (TF-IDF char + NearestNeighbors cosine, pola
    `arsenal/teks/kemiripan/retrieval#11`), adversarial AUC train vs test (TF-IDF + LogReg, OOF).
  - Bersihkan: aktif = lowercase + URL/mention jadi token + `html.unescape`; alt ber-#: kamus alay
    (`data_latihan/preprocessing/kamus_alay_salsabila.csv`), huruf berulang, emoji dipertahankan.
  - Model aktif: FeatureUnion TF-IDF kata (1–2) + char_wb (2–5) + LogReg balanced. Alt: LinearSVC +
    `CalibratedClassifierCV`, NB-SVM (fitur rasio log-count), ComplementNB, SGD modified_huber, Ridge.
  - CV: `StratifiedKFold`; alt `GroupKFold` (kolom grup), alt split waktu (urut tanggal, 20% terakhir).
  - Pasca: pengali kelas F1 makro di OOF (pola `tabular/metrik#14`); alt ambang biner.
  - Simpan OOF/test + baris `percobaan.csv`; submission + validator (T3).
- Data demo: `sentiment` (smsa). Target: F1 makro OOF ≥ 0,86 (KATALOG klasifikasi_tfidf holdout 0.8636).
  Bentuk beda: (1) `classification` (emot, 5 kelas); (2) `ujaran_kebencian` dengan `TARGET="HS"` (biner, 0/1);
  (3) `pair` wrete dengan `TEXT_COLS=["sent_A","sent_B"]`.
- Uji terima: lint 0; LOLOS di demo; tiga bentuk beda jalan dengan hanya mengubah Konfigurasi; catat detik
  per fold. Blok KATALOG/PANDUAN di STATUS.
- Dependensi: spesifikasi validator T3 (salin bila T3 belum selesai). Gelombang G1. Ukuran L, ± 3,5 jam.
- STATUS: selesai (2026-11-06). `contoh/text/baseline_oof.ipynb` (42 sel, 333 baris kode, tanpa output), `python _alat/lint.py` → 0 masalah.
  - Lingkungan uji: `.venv` (sklearn 1.9.1) + `JUPYTER_PATH` ke kernelspec venv di folder sementara (kernelspec `python3` pengguna menunjuk `Z:\GEMASTIK-DATABASE\.venv` yang tidak ada). Perintah: `jupyter nbconvert --to notebook --execute --output-dir ../outputs/uji_contoh/text contoh/text/baseline_oof.ipynb`.
  - **Demo smsa (Run All LOLOS, 15–22 dtk, CPU, 2,0–2,5 dtk/fold)**: F1 makro OOF **0,8602** (target ≥ 0,86 tercapai, margin tipis: SVC dipilih dari ±20 konfigurasi di fold yang sama), F1 test (kunci) 0,8343 (≈ 0,026 di bawah OOF; patokan "pilih final dari CV, bukan LB"). `MASALAH: 0`. Menulis `oof_tfidf_svc.npy`, `test_tfidf_svc.npy` (+ `oof_tfidf_logreg.npy` dari run awal) dan baris `percobaan.csv` ke `outputs/text/oof`. Temuan smsa: kembar 44 baris, 11 test persis ada di train, 10 hampir kembar (sim ≥ 0,9; contoh "melelahkan" vs "tidak melelahkan" beda label), adversarial AUC 0,562.
  - **Penyimpangan dari rancangan**: model aktif = SVM terkalibrasi (`CalibratedClassifierCV(LinearSVC(C=0.5, balanced), cv=3)`), bukan LogReg (0,8528 < 0,86); LogReg jadi alt. Pengali kelas dipakai hanya bila naik ≥ `MIN_GAIN` 0,002. Sel sample demo tidak menimpa `SAMPLE_CSV` di luar `OUTPUT_FOLDER` (diuji dengan file sentinel: md5 tetap). Validator = salinan T3 `simpan_muat` #15/#16. `ANSWER_CSV` dipindah ke Konfigurasi. Nama splitter dicatat di `percobaan.csv` (PERHATIAN dicetak bila bukan `StratifiedKFold`: OOF tidak sefold dengan model lain, tetap out-of-fold sah).
  - **Alternatif ber-# (semua dijalankan, F1 makro OOF smsa)**: aktif SVC 0,8602 | Ridge 0,8603 | LogReg union 0,8528 | NB-SVM 0,8562 (juga jalan di biner `hs`: 0,8443) | kata saja 0,8564 | char saja 0,8520 | SGD 0,8448 | ComplementNB 0,8209 | bersih alay+huruf berulang 0,8611 | DROP_DUPLICATES 0,8601 | metrik akurasi jalan | ambang biner (hs) jalan | split waktu (`DATE_COL`) jalan, OOF tidak lengkap sehingga tak disimpan | GroupKFold & StratifiedGroupKFold (wrete `category`, grup kosong di-`fillna`) jalan | submission proba per kelas (sample berkolom kelas, `LABEL_COLUMNS=[]`) jalan, `MASALAH: 0`.
  - **Bentuk beda, hanya ubah Konfigurasi, LOLOS (`MASALAH: 0`)**: emot (5 kelas) OOF 0,6926, test 0,6714, 1,2 dtk/fold; ujaran_kebencian `TARGET="HS"` OOF 0,8654, test 0,8652, 2,0 dtk/fold; wrete `TEXT_COLS=["sent_A","sent_B"]` OOF 0,4779 (pengali → 0,5844), test 0,6011, 0,1 dtk/fold (hanya bukti jalan: struktur pasangan hilang saat digabung, pakai `pasangan_teks`/T5 `TEXT_B`).
  - **Validator diuji dengan 5 kesalahan sengaja** (submission smsa): id hilang → "jumlah baris" + "id hilang"; urutan acak → "urutan id beda"; NaN → "nilai kosong"; label asing → "di luar ALLOWED"; id float → "tipe float64, sample int64". Kelimanya terdeteksi, submission bersih `MASALAH: 0`.
  - Batas: uji CPU saja, sesi lain memakai GPU/CPU; belum diuji di data > 100 ribu baris (adversarial/near-duplicate memakai `head(5000)` test dan NearestNeighbors brute).
  - **Blok KATALOG (untuk T13)**: `baseline_oof` | k-fold klasifikasi teks + OOF + sample_submission + validator | F1 makro OOF 0,8602 (smsa; test 0,8343) | alt: Ridge 0,8603, LogReg 0,8528, NB-SVM 0,8562, kata saja 0,8564, char saja 0,8520, SGD 0,8448, ComplementNB 0,8209, bersih alay 0,8611 | emot 0,6926, HS 0,8654, wrete 0,4779 (hanya jalan).
  - **Blok PANDUAN (untuk T13)**: "klasifikasi teks apa pun, submission pertama ≤ 45 menit, satu atau beberapa kolom teks, sumber OOF untuk ensemble" → `baseline_oof` (ubah Konfigurasi: `DATA_FOLDER/TRAIN_CSV/TEST_CSV/SAMPLE_CSV/ID_COL/TEXT_COLS/TARGET/SUBMIT_COL/MODEL_NAME`, lalu Run All, baca sel Temuan, hapus sel `HAPUS saat lomba` dan `ANSWER_CSV`).
  - **Catatan untuk T5/T6**: OOF/test pakai urutan kelas `np.unique(train[TARGET])`, test = rata-rata model fold; `MODEL_NAME` harus unik tiap variasi; file OOF di `outputs/text/oof` berisi run demo smsa (`tfidf_svc`, `tfidf_logreg`), jangan dicampur dengan data lain.

### T3. Baca data apa saja + validator submission (arsenal/umum)

- Tujuan: format file/submission aneh tidak menghabiskan waktu lomba.
- Peta: P.data.7, .13–15, P.jeb.13, .16–18, .20, P.eval.29. Pilar: arsenal. Nilai: tinggi.
- File: `arsenal/umum/baca_folder.ipynb` (sel baru), `arsenal/umum/simpan_muat.ipynb` (#15 diganti).
- Isi baca: encoding (`charset_normalizer.from_path(...).best().encoding`) + pemisah (`csv.Sniffer`) →
  `read_csv(sep=, encoding=, on_bad_lines="warn")`; tsv; parquet; jsonl; txt satu teks per baris; CoNLL
  (token TAB tag, baris kosong = kalimat baru) → tabel `sentence_id, token, tag` seperti `contoh/text/ner`;
  label string list `"['a','b']"` → kolom 0/1 (`ast.literal_eval`); `html.unescape`.
- Isi validator (dipakai ulang semua butir contoh): kolom dan urutannya = sample; jumlah baris sama; himpunan id
  sama; urutan id sama dengan sample; id kembar; NaN; tipe tiap kolom sama dengan sample ("3" vs "3.0");
  label ⊆ `ALLOWED` (label train). Cetak tiap masalah dan `MASALAH: n` (tanpa assert/try).
- Data demo: berkas dibuat sel demo dari smsa/nerp di folder output (tsv, `;` + latin-1, jsonl, parquet, CoNLL)
  lalu dibaca balik sama persis. Bentuk beda: validator diberi submission smsa dengan 5 kesalahan sengaja
  (id hilang, urutan acak, NaN, label asing, float) → menemukan kelimanya.
- Uji terima: lint 0; `uji_arsenal.py` LOLOS untuk sel baru. Blok PETA di STATUS.
- Gelombang G1. Ukuran S, ± 1,5 jam.
- STATUS: selesai (2026-11-06)
  - File: `arsenal/umum/baca_folder.ipynb` (sel kode baru #14–#20 = bagian 12–16, tabel awal ikut diperbarui) dan
    `arsenal/umum/simpan_muat.ipynb` (#15 lama diganti DUA sel berurutan #15 struktur + #16 isi; zip cadangan
    outputs yang dulu #16 sekarang #17). Python sistem 3.13 (pandas 2.3.3, charset_normalizer 3.5.2, pyarrow
    24.0.0); venv `.venv` belum berisi paket saat diuji.
  - Lint: `python _alat/lint.py arsenal/umum/baca_folder.ipynb arsenal/umum/simpan_muat.ipynb` → TOTAL MASALAH: 0.
  - uji_arsenal: `baca_folder.ipynb:14,15,16` / `14,17` / `14,18` / `14,19` / `14,20` dan seluruh notebook, lalu
    `simpan_muat.ipynb:1,14,15,16` (default, hasil `MASALAH: 0`) dan seluruh notebook → semua LOLOS.
  - Baca balik sama persis (`DataFrame.equals`), lewat sel yang sama dengan `--set TABLE_PATH=...`, data
    `data_latihan/text`: smsa_train (5.148 baris) → tsv, jsonl, parquet; 2.002 baris smsa + 2 baris beraksen
    → `;`+latin-1, tsv, CSV ber-BOM utf-8, pemisah `|` dalam .txt; smsa tiap baris → .txt (`ENCODING` default);
    nerp_train (57.630 token, 2.500 kalimat) → CoNLL TAB dan CoNLL spasi + `-DOCSTART-`; casa_train → kolom
    `"['fuel', 'price']"` → 0/1 sama dengan enam kolom asli; smsa lewat `html.escape` → `html.unescape` sama persis.
    12 uji, 0 gagal (skrip: scratchpad `uji_t3.py`, berkas uji di `../outputs/uji_t3/`).
  - Validator (`simpan_muat#15–#16`) pada smsa_test (1.260 baris, sample label tunggal): file bersih → `MASALAH: 0`;
    file dengan lima kesalahan sekaligus (2 id hilang, urutan acak, 1 NaN, label "sedih", id float) → 6 masalah,
    kelimanya terlaporan (jumlah baris, id hilang, urutan id, tipe float64 vs int64, nilai kosong, label di luar
    ALLOWED). Tiap kesalahan sendiri + id kembar + nama kolom beda + urutan kolom terbalik juga terdeteksi
    (masing-masing 1–2 pesan). Skrip: `uji_validator.py`.
  - Temuan saat uji: (1) `charset_normalizer` menebak `cp1250` untuk file latin-1 berisi `ï` (dan hanya cp1250
    di `could_be_from_charset`) → huruf rusak; sel menimpa `cp1250`/`iso8859_2` jadi `cp1252` (asumsi teks Indonesia/
    Inggris). (2) Lint melarang `;` di baris kode (regex "dua statement") termasuk di string/komentar → pemisah titik
    koma ditulis `chr(59)` dan entitas HTML demo ditulis dengan `!` lalu diganti. (3) `csv.Sniffer` hanya dipakai bila
    header memuat ≥ 2 kandidat pemisah; satu kandidat dipakai langsung, nol berarti satu kolom (`,`); Sniffer bisa
    melempar `csv.Error` (tanpa try/except dilarang) → komentar di sel menyuruh isi `separator` manual.
  - Penyimpangan dari rancangan: validator dua sel (struktur + isi, sel 30 baris tidak muat satu sel); tipe kolom
    dibandingkan lewat `dtype` pandas (bukan teks mentah): sample palsu semua 0 vs jawaban probabilitas menghasilkan
    pesan tipe (sudah dicatat di komentar sel). Sel demo 12a memakai data sintetis kecil (arsenal tanpa path dataset);
    uji smsa/nerp di atas memakai sel pembaca yang sama dengan `--set`.
  - Blok PETA (usulan untuk T13, tambahkan ke bagian `umum`):
    | `umum/baca_folder#14` (12a) | data contoh format (tsv, `;`+latin-1, parquet, jsonl, txt, CoNLL, list, HTML), lewati saat lomba |
    | `umum/baca_folder#15–#16` (12b–12c) | CSV/TSV/TXT bertabel: tebak encoding (charset_normalizer) + pemisah (Sniffer), `on_bad_lines="warn"`; parquet; jsonl → `table` |
    | `umum/baca_folder#17` (13) | .txt satu teks per baris → `table.text` |
    | `umum/baca_folder#18` (14) | CoNLL (TAB/spasi, `-DOCSTART-`) → `sentence_id, token, tag` |
    | `umum/baca_folder#19` (15) | label teks `"['a', 'b']"` → kolom 0/1 per label (`ast.literal_eval`) |
    | `umum/baca_folder#20` (16) | `html.unescape` (dua putaran) kolom teks |
    | `umum/simpan_muat#15–#16` | validator submission vs sample: kolom+urutan, jumlah baris, id hilang/asing/kembar/urutan, NaN, tipe (dtype), label ⊆ ALLOWED; cetak `MASALAH: n` (ganti #15 lama; zip cadangan jadi #17) |
    Peta yang ditutup: P.data.7, .13–15, P.jeb.13?, .16–18, .20, P.eval.29 (selain P.jeb.13 yang bagian fillna-nya ada di T2).
  - Belum: tidak ada. Catatan untuk butir lain: `from simpan_muat#15–#16` cukup di-copy ke sel "Cek submission" contoh
    (butuh `ID_COLUMN`, `LABEL_COLUMNS`, `ALLOWED` = `np.unique(train[TARGET])`); data latihan tanpa sample →
    sel "Buat sample_submission demo" dulu.

### T4. Model teks baru (unduh + embedding alt)

- Tujuan: checkpoint untuk Indonesia formal (besar), medsos, campur bahasa, embedding lebih kuat.
- Peta: P.kel.3, P.kel.8, P.rep.11, P.data.16. Pilar: contoh (alt) + alat. Nilai: tinggi untuk T5.
- Model (ukuran dugaan; pekerja catat ukuran nyata):

  | folder | HF id | ukuran | dipakai di |
  |---|---|---|---|
  | `indobert_large` | `indobenchmark/indobert-large-p1` | ± 1,3 GB | T5 alt (fp16 + grad checkpointing, batch 8) |
  | `indobertweet` | `indolem/indobertweet-base-uncased` | ± 0,5 GB | T5 alt (teks medsos; cek praproses di model card) |
  | `xlmr_base` | `FacebookAI/xlm-roberta-base` | ± 1,1 GB | T5 alt (bahasa campur/daerah/Inggris) |
  | `e5_base` | `intfloat/multilingual-e5-base` | ± 1,1 GB | `klasifikasi_embedding` alt (awalan `"query: "`) |

- File: `_alat/unduh_model.py` (tambah 4 baris di dict `HF`, pemilik tunggal); `contoh/text/klasifikasi_embedding`
  (alt `EMBEDDING_MODEL` e5 + baris awalan, ber-#).
- Library: transformers, sentence-transformers (sudah ada).
- Uji terima: `python _alat/unduh_model.py` selesai; tiap folder dimuat `from_pretrained(..., local_files_only=True)`;
  alt e5 di klasifikasi_embedding LOLOS + F1 dicatat (pembanding 0.8461). VRAM/waktu fine-tune diukur di T5.
- Dependensi: pertanyaan 3. Gelombang G0 (jaringan). Ukuran S, ± 1,5 jam (tergantung internet).
- STATUS: selesai (2026-11-06)
  - **Dijalankan:** `python _alat/unduh_model.py` (Python sistem 3.13; `.venv` belum ada) → "SELESAI", exit 0;
    model lain dilewati. Folder tujuan `Z:/GEMASTIK/Vignate-GEMASTIK/pretrained/`.
  - **Ukuran nyata** (sesudah membuang berkas tak perlu, lihat bawah): `indobert_large` 1279 MB (335 M param,
    `pytorch_model.bin`), `indobertweet` 424 MB (110 M), `xlmr_base` 1077 MB (278 M, `model.safetensors`),
    `e5_base` 1082 MB (278 M, `model.safetensors`). Total ± 2,9 GB (dugaan rencana ± 4 GB). Hasil unduhan mentah
    lebih besar: `xlmr_base`/`e5_base` 2,1 GB (bobot ganda `.safetensors` + `.bin`), `indobertweet` 1,27 GB (ikut
    `optimizer.pt`, `scheduler.pt`, `training_args.bin`). Berkas itu dihapus dan `unduh_model.py` diberi
    `SKIP` untuk `optimizer.pt|scheduler.pt|training_args.bin` serta `SKIP_EXTRA` (`pytorch_model.bin` untuk
    `xlmr_base` dan `e5_base` saja; `indobert*` hanya punya `.bin`, jadi tidak boleh diskip global). Unduhan
    hari setup jadi ± 2,9 GB, bukan ± 6,8 GB (diverifikasi `snapshot_download(..., dry_run=True)`:
    hanya satu berkas bobot per folder).
  - **Muat offline** (CPU, `HF_HUB_OFFLINE=1`): `AutoTokenizer` + `AutoModel.from_pretrained(..., local_files_only=True)`
    OK untuk keempatnya (2,1 / 0,7 / 4,4 / 4,3 detik); `SentenceTransformer(e5_base, local_files_only=True)`
    OK, encode → `(1, 768)`. Peringatan UNEXPECTED (`cls.predictions.*`, `lm_head.*`) = kepala MLM, normal.
  - **Model card indobertweet** (huggingface.co/indolem/indobertweet-base-uncased, dibaca 2026-11-06, bukan dari
    berkas lokal): praproses wajib sebelum tokenisasi = (1) huruf kecil, (2) mention → `@USER`, URL → `HTTPURL`,
    (3) emotikon/emoji → teks lewat paket `emoji` (`emoji.demojize`). Tokenizer lokal `do_lower_case=True`,
    vocab 31923, `httpurl` satu token; `@USER` terpecah `@`,`user`. Kartu tidak menyebut panjang maks; data latih
    ± 409 juta kata tweet Indonesia (Des 2019 – Des 2020, ekonomi/kesehatan/pendidikan/pemerintah). **Untuk T5:**
    pakai `indobertweet` hanya dengan praproses itu (alt di `finetune_oof` + sel bersihkan sesuai kartu);
    `max_len` 128 aman untuk teks medsos (BERT-base 512).
  - **Alt e5** di `contoh/text/klasifikasi_embedding.ipynb`: sel Konfigurasi dapat baris ber-# `EMBEDDING_MODEL = .../e5_base`
    dan konstanta `TEXT_PREFIX = ""` (aktif) + baris ber-# `TEXT_PREFIX = "query: "` ("WAJIB bila e5_base"); sel embedding
    memakai `TEXT_PREFIX + train[TEXT_COL]`. Lint: 0 masalah. Aktif semula (minilm, awalan kosong), tanpa output.
  - **Uji alt e5** (salinan dengan dua baris itu diaktifkan di `outputs/uji_contoh/text/`, jalan di GPU laptop RTX 5060
    8 GB, `nbconvert --execute`, 84 detik termasuk muat model; salinan hasil dihapus): LOLOS, **F1 makro validasi
    0.8476** (pembanding minilm KATALOG 0.8461, selisih +0.0015, tidak signifikan di 1030 baris validasi; indobert
    di KATALOG 0.8660 tetap lebih baik di SmSA). Kelas: negative 0.84, neutral 0.81, positive 0.88.
  - **Catatan untuk T1/T13:** kernelspec `python3` di `C:\Users\EYZARIM\AppData\Roaming\jupyter\kernels\python3`
    menunjuk `Z:\GEMASTIK-DATABASE\.venv\Scripts\python.exe` (tidak ada) sehingga `nbconvert --execute` dengan Python
    sistem GAGAL `WinError 2`. Saya menyiasatinya dengan `JUPYTER_PATH` ke folder kernelspec sementara di scratchpad
    (tidak mengubah konfigurasi pengguna). Venv baru dari T1 perlu `ipykernel install --user` atau `JUPYTER_PATH`.
  - **Blok usulan KATALOG (untuk T13)**, tabel `text/klasifikasi_embedding`, tambah baris:
    `| EMBEDDING_MODEL = ".../pretrained/e5_base" + TEXT_PREFIX = "query: " (multibahasa kuat; awalan wajib) | 0.8476 ⭐ | LOLOS |`
    Catatan T13: tabel itu memakai `EMBED_DIR = ROOT / ...` (versi build lama); sesuaikan dengan nama variabel aktif.
  - **Usulan PANDUAN/PETA:** tidak ada baris baru (tidak ada potongan arsenal baru). Usulan satu kalimat di
    PANDUAN "Teks + label": embedding e5 butuh `TEXT_PREFIX = "query: "`; `indobertweet` butuh praproses `@USER`/`HTTPURL`/emoji.
  - **Belum:** VRAM/waktu fine-tune keempat model diukur di T5 (butir ini hanya memuat dan embedding).

### T5. `contoh/text/finetune_oof.ipynb` (baru)

- Tujuan: fine-tune transformer dengan k-fold, OOF bersama, dan metode SOTA sebagai alternatif ber-#.
- Peta: P.kel.3, P.kel.7 (pasangan), P.rep.13, P.met.f.1–7, f.11, f.13, f.16, f.18, l.1, l.2, l.4, d.1, e.8,
  e.10, P.jeb.10, .14, .17, .23, P.eks.2, .3, .5. Pilar: contoh. Nilai: tertinggi untuk leaderboard.
- Notebook baru (alur latih k-fold, simpan OOF); `klasifikasi_finetune` tetap sebagai jalur cepat.
- Isi:
  - Konfigurasi: `PRETRAINED` (alt ber-#: indobert, indobert_large, indobertweet, xlmr_base, nli_mdeberta sebagai
    backbone, distilbert_id, minilm), `MAX_LENGTH` (cetak persentil 50/95/99 token), `BATCH_SIZE`, `ACCUM_STEPS`,
    `EPOCHS`, `LEARNING_RATE`, `FOLDS_TO_RUN` (mis. `[0]` saat waktu mepet), `SEEDS`, `TEXT_A`, `TEXT_B`.
  - `LABELS = np.unique(...)` (bukan ditulis tangan).
  - Latih: AMP bf16/fp16, padding dinamis, urut panjang untuk inferensi, clip grad, warmup linear, evaluasi tiap
    epoch, simpan bobot terbaik per fold, cetak detik/epoch + perkiraan sisa + VRAM maks.
  - Alt ber-#: kepala+ekor truncation; grad checkpointing; class weight; focal loss; label smoothing; LLRD;
    re-init lapisan atas; FGM; EMA; input pasangan `tokenizer(a, b)`; pseudo-label (`pd.concat` dengan CSV dari T6).
  - Simpan OOF/test proba + `percobaan.csv`; submission + validator.
- Data demo: smsa, indobert. Target: F1 makro OOF ≥ 0,92 (KATALOG holdout 0.9242).
  Pengukuran (fold 0, 1 epoch): indobert, indobert_large, indobertweet, xlmr_base, nli_mdeberta → tabel
  VRAM maks / detik per epoch / F1 di STATUS (GPU laptop RTX 5060; PC lomba 3070 bisa beda).
  Bentuk beda: (1) `nusax` dengan xlmr_base `FOLDS_TO_RUN=[0]`; (2) `pair` wrete dengan alt pasangan.
- Uji terima: lint 0; LOLOS (Run All demo); dua bentuk beda jalan dengan Konfigurasi/alt saja.
- Dependensi: T1 (Python), T4 (checkpoint baru). Gelombang G2. Ukuran L, ± 5 jam.
- STATUS: dikerjakan (2026-11-07)

### T6. `contoh/text/ensemble_oof.ipynb` (baru)

- Tujuan: gabung OOF lintas notebook dengan metrik lomba, bukan logloss saja.
- Peta: P.met.e.1–5, e.6, p.2, p.4, d.1, P.eval.16, .26, P.pres.7, .11. Pilar: contoh. Nilai: tinggi.
- Notebook baru (pemuatnya file `.npy`, bukan model). Pola dari `arsenal/tabular/ensembling#8–#11`.
- Isi: muat semua `oof_*.npy`/`test_*.npy` di `OOF_FOLDER` + y; metrik di Konfigurasi (aktif F1 makro dari
  argmax; alt ber-#: logloss, akurasi, QWK, RMSE, AUC); korelasi antar model; rata-rata; hill climbing
  Caruana (dengan pengembalian) di metrik itu; alt rank average, `scipy.optimize` Nelder-Mead, stacking LogReg;
  pengali kelas F1 makro + alt temperature scaling sesudah gabung; tabel ablation dari `percobaan.csv`;
  alt tulis CSV pseudo-label (test yakin) untuk T5; submission + validator.
- Data demo: OOF smsa dari T2 (≥ 2 varian, mis. LogReg dan NB-SVM) + T5 (indobert). Target: hasil ≥ model
  tunggal terbaik di OOF (kalau tidak, notebook memilih model tunggal dan mengatakannya).
  Bentuk beda: OOF emot dari T2 saja (tanpa BERT, 5 kelas).
- Uji terima: lint 0; LOLOS. Dependensi: T2 (format), uji penuh setelah T5. Gelombang G1b. Ukuran M, ± 2,5 jam.
- STATUS: selesai sebagian (2026-11-07): semua uji yang bisa dijalankan sekarang LOLOS; **menunggu T5:** (a) demo
  "T2 + OOF indobert" (OOF `indobert` belum ada di folder bersama saat uji) dan (b) `finetune_oof` memuat `pseudo_label.csv`
  (T6 sudah menulisnya; kolom id, teks, target seperti train). Selebihnya selesai.
  - File: `contoh/text/ensemble_oof.ipynb` (baru, 52 sel, 30 sel kode, 409 baris kode, tanpa output). `python _alat/lint.py`
    → 0 masalah. Tidak ada file lain di repo yang diubah (skrip pembangun dan varian uji ada di scratchpad, bukan di repo).
  - Isi: Konfigurasi (`TASK` classification/regression, `MODEL_NAMES`, `MIN_GAIN`, `WRITE_PSEUDO`) → Metrik (`score(truth,
    pred)`, besar = baik; aktif F1 makro, alt ber-#: akurasi, logloss, QWK, AUC biner/multikelas, RMSE, QWK regresi) → baca →
    sample demo (HAPUS) → muat OOF → skor + korelasi → cara gabung → skor jujur → putusan → temperature (alt) → pengali
    kelas → simpan + `percobaan.csv` + ablation → CSV pseudo-label → submission → validator T3 → kunci (HAPUS).
  - **Penyaringan per dataset (catatan T8):** (1) pasangan `oof_`/`test_` lengkap; (2) bentuk = (n_train, n_kelas) dan
    (n_test, n_kelas); (3) angka terbatas, tanpa baris nol (OOF split waktu tak lengkap); baris dinormalkan ke jumlah 1
    (NB-SVM per kelas); (4) skor OOF harus melewati tebakan konstan DAN skor OOF yang barisnya diacak (+10% +0,005).
    Alasan (4): untuk F1 makro tebakan konstan terlalu rendah (0,19), jadi OOF dataset lain yang sebentuk (≈ acak 0,33)
    lolos; pembanding diacak menangkapnya. `MODEL_NAMES` diisi = saringan skor dimatikan.
  - **Skor jujur:** semua cara gabung berantarmuka `(fit_stack, truth, apply_stacks)`, jadi bobot dicari di 4/5 baris OOF dan
    dinilai di 1/5 (seed `SEED+1`, 5 putaran). Skor dalam sampel dicetak sebagai pembanding. Putusan = cara PALING SEDERHANA
    (urutan tunggal → rata → hill → alt) yang skor jujurnya ≥ terbaik − `MIN_GAIN`; bila model tunggal yang menang dicetak
    "TIDAK ADA gabungan yang mengalahkan model tunggal terbaik". Pengali kelas juga dinilai jujur.
  - Cara gabung: aktif tunggal, rata, hill climbing Caruana (dengan pengembalian, di metrik lomba); alt ber-#: rank average,
    Nelder-Mead (`scipy.optimize`, bobot softmax), stacking LogReg (regresi: Ridge). Pasca: pengali kelas (aktif, jujur) +
    temperature scaling (alt ber-#). Voting mayoritas tidak dibuat (diterima di peta).
  - Hasil gabungan disimpan di `OUTPUT_FOLDER` (`blend_oof_jujur.npy`, `blend_test.npy`, `ablation.csv`), BUKAN di `OOF_FOLDER`,
    supaya run berikut tidak menggabungkan hasil gabungan. Baris `percobaan.csv`: `ensemble_oof, ensemble_<cara>, <metrik>,
    <skor jujur>`. Sel ablation membandingkan skor OOF yang dihitung ulang dengan baris terakhir `percobaan.csv` per model
    (beda > 0,002 dicetak: tanda baris dari dataset/metrik lain).
  - **Uji terima 1, demo smsa, folder bersama `outputs/text/oof` apa adanya** (python `.venv`: `jupyter nbconvert --execute
    --output-dir ../outputs/uji_contoh/text contoh/text/ensemble_oof.ipynb`): LOLOS, 13 detik. Folder berisi `tfidf_svc`,
    `tfidf_logreg` (smsa) + `lgbm_teks_tabular` (baju, ditolak: bentuk 17614×5). F1 makro OOF: svc 0,8602, logreg 0,8528;
    skor jujur tunggal 0,8602, rata 0,8572, hill 0,8602 → dipilih tunggal (`tfidf_svc`), pengali −0,0015 tidak dipakai; test
    (kunci) 0,8343; `MASALAH: 0`. Dua model itu berkorelasi 0,97 sehingga tidak ada gabungan yang menang, dan notebook
    mengatakannya (sesuai target "hasil ≥ tunggal terbaik, bila tidak pilih tunggal"). Dua baris `ensemble_tunggal` masuk ke
    `percobaan.csv` bersama (dua run penerimaan).
  - **Uji 2, 6 varian smsa** (scratch: svc, logreg, nbsvm, ridge, kata, char dari salinan `baseline_oof`; ditambah OOF emot,
    baju, dan dua berkas palsu): saringan menolak `emot_*` dan `lgbm_teks_tabular` (bentuk), `tanpa_test` (tak ada test),
    `asing` (0,3298 ≤ acak 0,3303) dan `diacak` (baris tertukar, 0,3394 ≤ 0,3381 + margin); 6 model asli lolos. F1 makro:
    tunggal jujur 0,8576, hill 0,8589, **rata 0,8640** (dipilih, +0,0064); test (kunci) 0,8427 vs terbaik tunggal di test (ridge)
    0,8432, jadi setara. Akurasi: rata 0,8654 jujur vs tunggal 0,8563; test 0,8802 vs terbaik 0,8786. QWK: hill 0,7801 jujur,
    rata dipilih (selisih < MIN_GAIN); test 0,8117 vs 0,8109. AUC multikelas: stacking 0,9642 jujur, hill dipilih 0,9626
    (dalam MIN_GAIN); test 0,9667 vs terbaik 0,9638. **Logloss (alt dinyalakan): stacking −0,3539 jujur vs tunggal −0,3832; test
    −0,3175 vs terbaik −0,3375**; temperature scaling dinyalakan (T = 1,003: proba sudah terkalibrasi); rank average −0,65
    (jangan dipakai di logloss, sesuai peringatan di notebook).
  - **Bentuk beda emot** (5 varian `baseline_oof` di folder scratch, tanpa BERT; alt rank/Nelder-Mead/stacking dinyalakan):
    LOLOS, `MASALAH: 0`; stacking dipilih (jujur 0,6966 vs tunggal 0,6926), test 0,6776 vs terbaik tunggal 0,6714.
  - Bentuk lain: **regresi** (yelp, `TASK="regression"`, RMSE; 2 OOF Ridge dibuat di scratch): LOLOS, stacking Ridge jujur RMSE
    0,8744 vs tunggal 0,9003, test 0,8681 vs 0,8940; validator `MASALAH: 0` (sel sample demo diberi nilai float untuk regresi).
    **Biner** (ujaran_kebencian `HS`, 2 varian, AUC): LOLOS, tunggal dipilih (stacking 0,9405 vs 0,9392, dalam MIN_GAIN).
    **Satu model = OOF T8** (baju, `TARGET=rating`): dimuat dan dicatat; pengali kelas naik jujur +0,0407 (F1 0,399 → 0,440; test
    0,404 → 0,457). Jalur pengali dipaksa dipakai (smsa, `use_weights=True`): submission dan validator `MASALAH: 0`.
    `WRITE_PSEUDO=True`: 501 dari 1260 test yakin ≥ 0,9, `pseudo_label.csv` (kolom id, text, label).
  - **Menutup uji T8:** OOF `lgbm_teks_tabular` (T8) dimuat `ensemble_oof` tanpa masalah bentuk/urutan kelas.
  - Temuan di luar T6: `contoh/text/baseline_oof.ipynb` (milik T2) memuat `language_info.codemirror_mode: null` yang ditolak
    validasi `nbformat` Python sistem ini (`NotebookValidationError`); `ensemble_oof` ditulis tanpa field itu. Di uji varian saya
    mengganti metadata di salinan scratch. Usulan: T2/T13 hapus `codemirror_mode`/`version` null. (Run All sebelumnya LOLOS
    di T2, jadi mungkin versi nbformat berbeda.)
  - Batas: tidak diuji di data besar (> 100 ribu baris) dan paling banyak 6 model; Nelder-Mead di F1 (bertingkat) hampir tidak
    bergerak dari bobot sama (disebut di notebook); rank average buruk di multikelas (disebut di notebook).
  - Blok KATALOG (T13 salin, `contoh/KATALOG.md`): `## text/ensemble_oof` | Metrik: sesuai lomba (aktif F1 makro); asli: smsa 6 varian
    `baseline_oof`, rata 0.8640 jujur (tunggal terbaik 0.8576); logloss stacking −0.3539 vs −0.3832 | Aktif: tunggal, rata, hill
    climbing (Caruana), pengali kelas jujur | Alt: rank, Nelder-Mead, stacking LogReg/Ridge, temperature scaling, metrik
    akurasi/logloss/QWK/AUC/RMSE, regresi, CSV pseudo-label | LOLOS 13 dtk (CPU).
  - Blok PANDUAN (T13 salin, tabel tanda-soal): "Sudah ada ≥ 2 OOF dari notebook lain (baseline_oof / finetune_oof / teks_tabular),
    mau digabung dengan metrik lomba" → `text/ensemble_oof` (ganti Konfigurasi + sel Metrik; pilihan memakai skor JUJUR, bukan
    dalam sampel); lalu submit `submission.csv`, pilih final dari CV bukan LB saja.
  - Blok PETA (T13): peta yang ditutup: P.met.e.1–e.6, P.met.p.2 (alt), P.met.p.4, P.met.d.1 (tulis CSV; T5 memuat), P.eval.16,
    P.eval.26 (percobaan.csv dibandingkan), P.pres.7 (`ablation.csv`), P.pres.11 (tabel skor model). `tabular/ensembling` dan
    `tabular/metrik#14` diadaptasi ke `contoh/text/ensemble_oof`.

### T7. Paket kumpul (arsenal/umum)

- Tujuan: zip dokumentasi + kode + PPT benar isinya (syarat gugur).
- Peta: P.pres.12. Pilar: arsenal (`umum/simpan_muat`, sel baru sesudah #16).
- Isi: `FILES_TO_ZIP` (# UBAH: notebook, submission, dokumentasi, PPT), cek tiap file ada + ukuran, zip ke
  `OUTPUT_FOLDER`, cetak `namelist()` dan total ukuran, peringatan bila melewati `MAX_MB` (# UBAH).
- Uji terima: lint 0; uji_arsenal LOLOS. Bentuk beda: satu file sengaja hilang → dilaporkan.
- Gelombang G1. Ukuran S, ± 1 jam. Blok PETA di STATUS.
- STATUS: selesai (2026-11-06)
  - File: `arsenal/umum/simpan_muat.ipynb`: markdown "## 9. Paket kumpul" + sel kode baru **#18** (sesudah zip
    cadangan #17, yang dulu #16 sebelum T3 memecah validator jadi #15–#16); tabel awal notebook diberi baris 9.
  - Isi sel: `FILES_TO_ZIP` / `OUTPUT_FOLDER` / `ZIP_NAME` / `MAX_MB` (semua `# UBAH`); tiap berkas dicetak "ada" +
    ukuran MB atau `HILANG`; zip ke `OUTPUT_FOLDER` dengan `arcname=basename`; cetak `namelist()` dari arsip yang
    baru dibuka kembali, jumlah berkas dan ukuran zip; `PERINGATAN` bila ada berkas hilang atau zip > `MAX_MB`.
  - Lint: `python _alat/lint.py arsenal/umum/simpan_muat.ipynb arsenal/umum/baca_folder.ipynb` → TOTAL MASALAH: 0.
  - uji_arsenal (`simpan_muat.ipynb:18`, berkas 200 KB dibuat lewat `--define`): (1) empat berkas ada →
    `isi zip` memuat keempat nama, tanpa peringatan, LOLOS; (2) `dokumentasi.pdf` sengaja tidak ada →
    `HILANG  dokumentasi.pdf`, `3 dari 4 file`, `PERINGATAN: ada file HILANG`, LOLOS; (3) `MAX_MB=0.0001` →
    `PERINGATAN ... melewati batas`, LOLOS. Seluruh `simpan_muat.ipynb` tanpa pilihan sel juga LOLOS.
  - Catatan: ukuran zip dicetak 2 desimal (berkas uji yang mudah dikompres tampil 0.00 MB; berkas nyata normal).
    Sel tidak membuat dokumentasi/PPT; hanya memeriksa dan mengemas yang sudah ada.
  - Blok PETA (usulan untuk T13, bagian `umum`):
    | `umum/simpan_muat#18` (bagian 9) | paket kumpul: cek tiap berkas (ada/HILANG + ukuran), zip, cetak `namelist()` + ukuran, peringatan > `MAX_MB`; untuk syarat gugur (zip dokumentasi + kode + PPT) |
    Peta yang ditutup: P.pres.12.

### T8. `contoh/text/teks_tabular.ipynb` (baru)

- Tujuan: soal teks + kolom angka/kategori (pola final 2024) end-to-end.
- Peta: P.data.3, P.kel.2, P.rep.14–16, P.met.k.6, P.jeb.5. Pilar: contoh. Nilai: tinggi.
- Isi: TF-IDF → TruncatedSVD (alt: vektor MiniLM/e5) + statistik teks (panjang, kata, huruf besar, angka,
  tanda baca) + kolom angka/kategori → LightGBM OOF (alt XGBoost; alt CatBoost dengan `text_features`); cek
  bocor per kolom (pola `tabular/dasar/validasi#4`); simpan OOF format bersama; submission + validator.
- Data demo: tidak ada di repo (dicek: hanya `multimodal/classification` judul + tahun). Pekerja mencari set
  terbuka dengan ≥ 3.000 baris, satu kolom teks panjang, ≥ 3 kolom angka/kategori, target klasifikasi; unduh
  dengan pola `_alat/unduh_data_tambahan.py` (fungsi baru), catat sumber + lisensi di STATUS (T13 menyalin ke
  `SOURCES_text.md`). Internet hanya sampai 2026-11-08. Bila 30 menit tidak ketemu: pakai `nusax`
  (teks + `bahasa`) dan catat sebagai cadangan lemah.
  Bentuk beda: `multimodal/classification` (judul pendek + `release_year`, biner).
- Uji terima: lint 0; LOLOS di demo dan bentuk beda; OOF dimuat `ensemble_oof`.
- Gelombang G1. Ukuran L, ± 3,5 jam.
- STATUS: selesai (2026-11-06), kecuali dua hal yang tidak bisa diuji: OOF belum dimuat `ensemble_oof` (T6 belum
  ada; hanya diperiksa dengan skrip) dan alt `e5_base` belum diukur (CPU terlalu lama; alt `minilm_en` terukur).
  - File: `contoh/text/teks_tabular.ipynb` (baru, 39 sel, lint 0 masalah), `_alat/unduh_data_tambahan.py`
    (+fungsi `baju_wanita`, masuk daftar default). Notebook dibangun dari skrip di scratchpad (bukan di repo).
  - Data demo: Women's E-Commerce Clothing Reviews (Kaggle nicapotato, **CC0**), repack HF
    `chibifire/kaggle-womens-ecom-clothing-reviews`. Diunduh ke `data_latihan/text/teks_tabular/baju_wanita_{train,test,test_labels}.csv`
    (17614/5872 baris; teks `title`+`review_text`, angka `age`,`positive_feedback_count`, kategori `clothing_id`,
    divisi/departemen/kelas; target `rating` 1–5). `recommended_ind` dibuang (DROP_COLS; AUC sendiri 0,77, hampir turunan rating).
    Jebakan nyata: 631 teks kosong, 2850 judul kosong, 217 teks test persis ada di train (CV agak optimis; fold tidak bisa dikelompokkan).
  - Run All demo (venv `.venv`, CPU, kernelspec lewat `JUPYTER_PATH` sementara karena kernelspec pengguna
    menunjuk venv yang tidak ada): LOLOS, 126 detik. F1 makro OOF 0,399 (akurasi 0,649), test 0,404.
    Ablasi: tabel saja 0,181 → +statistik 0,244 → +SVD teks 0,396. Pembanding TF-IDF+LogReg (fold sama) F1 makro 0,434:
    GBDT di atas SVD sedikit di bawah linear murni (dicatat jujur; ide stacking OOF teks tidak dikerjakan).
  - OOF disimpan `outputs/text/oof/oof_lgbm_teks_tabular.npy` (17614×5 float32, baris jumlah 1,0) + `test_...npy` (5872×5)
    + baris `percobaan.csv`. Dimuat `ensemble_oof`: BELUM (notebook itu belum ada); diperiksa dengan skrip (bentuk, dtype, urutan kelas).
    Catatan T6: `oof_*.npy` set lain (mis. `tfidf_logreg` smsa dari T2) ada di folder yang sama → T6 harus menyaring per dataset.
  - Bentuk beda `multimodal/classification` (judul + `release_year`, biner, tanpa kategori): LOLOS 26 detik,
    OOF (1331×2), F1 makro OOF 0,515, test 0,539, validator MASALAH: 0 (OOF_FOLDER di scratch).
  - Cek bocor: kolom tanam `skor_internal` (salinan target) terdeteksi AUC 1,0 → "CURIGA BOCOR"; demo asli tanpa bocor.
  - Alt: XGBoost LOLOS (254 detik total, F1 OOF 0,383). CatBoost `text_features` LOLOS setelah `clone` diganti
    `copy.deepcopy` (628 detik total CPU, F1 OOF 0,335; lebih lambat dan lebih rendah di data ini). Alt embedding
    (`minilm_en` + `embed_svd`, ganti baris svd, CPU karena `embed_svd` memakai `device="cpu"`): LOLOS 545 detik total
    (encode ± 7 menit), F1 OOF 0,361, test 0,349 (di bawah TF-IDF→SVD 0,399 di data ini). `e5_base` (prefix "query: "): belum diukur.
  - Belum/tidak bisa: dimuat `ensemble_oof` (T6 belum ada). Tidak menyentuh `requirements.txt` (xgboost, catboost, sentence-transformers sudah ada).
  - Blok KATALOG (T13 salin, `contoh/KATALOG.md`): `## text/teks_tabular` | Metrik F1 makro; asli **0.399** (OOF 5 lipatan, test 0.404) |
    Aktif: LGBMClassifier(400 pohon) di atas statistik teks + TF-IDF→SVD64 + age/positive_feedback_count + 4 kategori |
    Alt: XGBClassifier enable_categorical 0.383 LOLOS; CatBoost text_features 0.335 LOLOS (lambat, 530 dtk CPU); embed_svd minilm_en 0.361 LOLOS;
    e5_base belum diukur; pembanding TF-IDF+LogReg (lipatan sama, bukan di notebook) 0.434.
  - Blok PANDUAN (T13 salin, `contoh/PANDUAN.md` tabel tanda-soal): "Tabel/CSV dengan kolom teks panjang + kolom angka/kategori
    (pola final 2024: isian teks bebas)" → `text/teks_tabular` (ganti Konfigurasi: TEXT_COLS, CAT_COLS, DROP_COLS) → `text/finetune_oof` untuk teksnya → `text/ensemble_oof`; validasi Stratified 5 (OOF).
  - Blok SOURCES: `text/teks_tabular (baju_wanita)` | Kaggle nicapotato/womens-ecommerce-clothing-reviews, repack HF chibifire | CC0-1.0 | teks+angka+kategori, target rating.

### T9. `contoh/text/finetune_multilabel.ipynb` (baru)

- Tujuan: multilabel dengan transformer.
- Peta: P.tugas.3, P.met.l.5, p.5, P.eval.22. Pilar: contoh.
- Notebook baru (target matriks, BCE, ambang per label mengubah latih dan pasca).
- Isi: `LABEL_COLS` di Konfigurasi; BCEWithLogits (alt `pos_weight`); sigmoid; ambang per label dari OOF;
  aturan minimal satu label (alt); `KFold` (iterstrat tidak terpasang); OOF format bersama; submission kolom 0/1.
  Alur latih/AMP disalin dari T5.
- Data demo: `ujaran_kebencian` (12 label). Pembanding: jalankan juga `contoh/text/ujaran_kebencian` (TF-IDF),
  catat F1 makro keduanya. Bentuk beda: `multilabel` casa (6 label, 810 baris).
- Uji terima: lint 0; LOLOS; VRAM/detik per epoch dicatat. Dependensi: T5 (alur). Gelombang G2. Ukuran M, ± 3 jam.
- STATUS:

### T10. `contoh/text/finetune_regresi.ipynb` (baru)

- Tujuan: regresi/ordinal/esai dengan transformer + contoh head kustom.
- Peta: P.tugas.5–7, .13, P.kel.4, P.met.f.8, f.9, l.7, l.9, p.6, P.eval.6, .8. Pilar: contoh.
- Notebook baru (model `nn.Module` sendiri: AutoModel + mean pooling + multi-sample dropout + Linear(1)).
- Isi: MSE (alt Huber); skala target; clip; ambang ordinal di OOF untuk QWK (pola `tabular/metrik#11`);
  alt pasangan untuk STS + Pearson/Spearman; OOF format bersama; submission + validator.
- Data demo: `regression` yelp (Inggris; xlmr_base atau distilbert_en), pembanding `contoh/text/regresi_teks`
  (jalankan, catat RMSE). Bentuk beda: smsa sebagai ordinal (negative=0, neutral=1, positive=2) dengan indobert,
  metrik QWK.
- Uji terima: lint 0; LOLOS; VRAM/waktu dicatat. Dependensi: T5. Gelombang G2. Ukuran M, ± 3 jam.
- STATUS:

### T11. Tambahan arsenal kecil

- Tujuan: menutup celah metode yang cukup satu sel.
- Pilar: arsenal. Isi (tiap sel `# UBAH` di tempat yang diubah):
  - a. `teks/klasifikasi/tfidf`: sel NB-SVM (rasio log-count per kelas + LogReg) + skor CV dibanding LogReg.
    Peta P.met.k.3, P.rep.9.
  - b. `teks/dasar/praproses`: sel train↔test (kembar persis, hampir kembar, adversarial AUC + kata paling
    membedakan) dan terapkan clean/stem ke test. Peta P.jeb.4, .7, P.eval.25.
  - c. `teks/ner`: sel submission dari BERT (format kode #7). Peta P.tugas.8, P.kel.5. Butuh GPU: jalankan
    saat tidak ada butir GPU lain.
  - d. `tabular/ensembling`: `score` alternatif F1 makro (ber-#) + sel rank average. Peta P.met.e.2–3.
  - e. `tabular/metrik`: sel temperature scaling (cari T di OOF, minimalkan logloss) + alt isotonic.
    Peta P.met.p.1–2.
- Data demo: data demo notebook masing-masing. Bentuk beda: a/b di emot; d/e di OOF smsa dari T2.
- Uji terima: lint 0; uji_arsenal LOLOS per sel. Blok PETA di STATUS. Gelombang G1 (c di G2).
  Ukuran M, ± 3 jam.
- STATUS: a, b, d, e selesai (2026-11-06); **c belum** (submission NER-BERT, butuh GPU, tidak dikerjakan sesi ini).
  - Nomor "sel" di bawah = nomor SEL KODE (untuk `uji_arsenal`), nomor "bagian" = judul `## n.` di notebook (dipakai PETA).
    Semua sel baru ditambahkan di akhir notebook (kecuali `1b` di ensembling), jadi sel/bagian lama tidak bergeser.
  - **a. `teks/klasifikasi/tfidf.ipynb`**: bagian 8 (sel #10 fungsi `log_count_ratio` + `fit_predict_nbsvm`, pola dari
    `contoh/text/baseline_oof` yang sudah ada; #11 CV 5 fold NB-SVM vs LogReg vs rata-rata, simpan `outputs/oof_nbsvm.npy`
    + `outputs/test_nbsvm.npy`) dan bagian 9 (sel #12 `submission_nbsvm.csv`); tabel awal diberi baris 8–9.
    Lint 0. uji_arsenal `tfidf.ipynb:1,2,10,11,12 --set DEMO_FOLDER=...` LOLOS (11,9 dtk), smsa F1 makro OOF: LogReg 0,8544,
    NB-SVM 0,8562, rata-rata 0,8603. Bentuk beda emot (5 kelas, `--define` train/test emot, sel 2,10,11,12) LOLOS:
    LogReg 0,6828, NB-SVM 0,6692, rata-rata 0,6834 (NB-SVM TIDAK selalu menang, makanya dibandingkan di CV).
  - **b. `teks/dasar/praproses.ipynb`**: bagian 0 sekarang juga memuat `test` (emot_test); bagian 5 (sel #6 `clean`/`stem` ke
    test + stem kata baru, #7 kembar persis + hampir kembar (kosinus TF-IDF huruf, `SIMILAR_LEVEL`), #8 adversarial AUC +
    kata paling khas test/train + panjang teks). Lint 0. uji_arsenal `praproses.ipynb:1,2,3,4,6,7,8` LOLOS (4,5 dtk) di emot:
    4 kembar persis, 7 hampir kembar, AUC 0,48. Bentuk beda (ujaran_kebencian, kolom diganti `Tweet`/`HS`, `--set TEXT/TARGET/TEST_TEXT`,
    sel 2,3,4,6,7,8) LOLOS: 23 kembar persis, 73 hampir kembar, AUC 0,501. Seluruh notebook tanpa pilihan sel LOLOS.
  - **d. `tabular/ensembling/ensembling.ipynb`**: sel #3 (bagian 1b, semua ber-`#`) = `score` alternatif F1 makro (nilai
    pertama = −F1 makro agar `hill_climb`/papan tetap mencari yang terkecil); bagian 10 = rank average (sel #17 `rank_average`
    + `rank_avg` di `oofs`/`test_probas`; #18 banding rata proba vs rank average per model: AUC, akurasi, F1 makro).
    Lint 0. Data: OOF smsa T2 (`outputs/text/oof/oof_tfidf_svc|logreg.npy`, via `--define`) sel 2,17,18 LOLOS: rata proba F1 0,8572,
    rank average 0,8269. Biner (ujaran_kebencian HS, OOF dibuat di `--define`) sel 2,3,17,18 LOLOS, alt F1 diaktifkan di salinan
    sementara (`score` mengembalikan (−0,8429, 0,848)). Run All utuh covertype LOLOS 82 dtk. **Temuan jujur**: rank average
    per kelas TIDAK membantu akurasi/F1 multikelas (covertype 7 kelas: akurasi 0,56 vs rata proba 0,79; smsa 0,830 vs 0,858)
    dan di smsa AUC OvR pun turun (0,953 vs 0,959); markdown sel menyebutnya. Berguna untuk AUC biner / model beda skala.
  - **e. `tabular/metrik/metrik.ipynb`**: bagian 9 (sel #15 muat OOF/test/label smsa (`# UBAH`), #16 temperature scaling
    (`best_temperature` minimalkan logloss, `minimize_scalar`), #17-#18 alt isotonic per kelas + banding temperature vs
    isotonic dengan CV 5 fold, #19 terapkan ke test + simpan `outputs/oof_dikalibrasi.npy`/`test_dikalibrasi.npy`); tabel awal
    diberi baris 9 dan baris logloss menyebut kalibrasi. Lint 0. uji_arsenal `metrik.ipynb:1,15-19 --set DEMO_FOLDER=...` LOLOS:
    smsa T=0,861, logloss 0,3859 → 0,3813 (CV: temperature 0,3814, isotonic 0,3818). Biner (ujaran_kebencian, SVM + sigmoid
    sendiri, T=1,298) LOLOS: 0,3805 → 0,3707 (CV isotonic 0,3743).
  - Seluruh isi `tfidf`, `praproses`, `metrik`, `ensembling` dijalankan tanpa pilihan sel dengan `--set DEMO_FOLDER=...`: semua LOLOS.
    Venv `Z:/GEMASTIK/Vignate-GEMASTIK/.venv`, CPU saja. Tidak di-commit. Catatan: file-file ini menyimpan output di `outputs/`
    relatif (mengikuti konvensi notebook masing-masing), bukan `../../outputs/<kategori>/`.
  - Blok PETA (usulan untuk T13; nomor = bagian):
    | NB-SVM (rasio log-count + LogReg per kelas), banding CV dengan LogReg, simpan OOF `.npy` | `teks/klasifikasi/tfidf` | 8 |
    | `submission_nbsvm.csv` dari NB-SVM | `teks/klasifikasi/tfidf` | 9 |
    | train ↔ test: `clean`/`stem` ke test, kembar persis, hampir kembar, adversarial AUC + kata khas test | `teks/dasar/praproses` | 5 |
    | metrik soal F1 makro di ensembling (`score` alt ber-`#`, sel kode #3) | `tabular/ensembling/ensembling` | 1b |
    | rank average (AUC / model beda skala; hati-hati multikelas) | `tabular/ensembling/ensembling` | 10 |
    | kalibrasi proba OOF: temperature scaling + isotonic (logloss turun, argmax tetap) | `tabular/metrik/metrik` | 9 |
    Peta yang ditutup: P.met.k.3, P.rep.9 (a); P.tugas.23, P.jeb.4, P.jeb.7, P.eval.25 (b); P.met.e.2 dan metrik F1 makro (d);
    P.met.p.1, P.met.p.2 (e, bagian kalibrasi saja). c (P.tugas.8, P.kel.5) tetap terbuka.

### T12. Gladi jalur 5 jam (pendek, offline)

- Tujuan: ukur runbook bagian 3 dan buktikan semuanya jalan tanpa internet.
- Peta: P.eks.1, .6, .9. Pilar: keduanya.
- Isi: set yang tidak dipakai demo baru (mis. `english` tweet atau `nusax`); `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1` (pengguna boleh mematikan wifi); jalankan T2 → T5 `FOLDS_TO_RUN=[0]` → T6 → validator
  → T7 dengan stopwatch; isi menit nyata di tabel bagian 3.
- Uji terima: baseline sampai submission tervalidasi ≤ 45 menit; tidak ada akses jaringan; tabel waktu terisi.
- Dependensi: T2, T5, T6, T7. Gelombang G3. Ukuran M, ± 2,5 jam.
- STATUS:

### T13. Gabung file bersama

- Tujuan: peta pengguna saat lomba memuat semua yang baru.
- File: `contoh/KATALOG.md` (bagian notebook baru + skor alt baru + skor yang didapat T1/T9/T10),
  `contoh/PANDUAN.md` (baris bentuk soal → notebook, rujukan ke bagian 3), `arsenal/PETA.md` (sel baru T3, T7,
  T11), `data_latihan/SOURCES_text.md` (set baru T8 + 9 set yang belum tercatat, sumber dari
  `_alat/unduh_data_tambahan.py`).
- Uji terima: tiap notebook/sel baru muncul di file yang sesuai; lint semua notebook baru sekali lagi 0.
- Dependensi: semua butir di atas garis. Gelombang G3. Ukuran S, ± 1,5 jam.
- STATUS:

### Jumlah di atas garis

13 butir: 6 notebook contoh baru (T2, T5, T6, T8, T9, T10); sel arsenal baru/diganti di 7 notebook arsenal
(T3: `umum/baca_folder`, `umum/simpan_muat`; T7: `umum/simpan_muat`; T11: `teks/klasifikasi/tfidf`,
`teks/dasar/praproses`, `teks/ner`, `tabular/ensembling`, `tabular/metrik`); 1 alt di `contoh/text/klasifikasi_embedding`;
4 model baru (T4). Usaha ± 35 jam kerja;
jalur kritis GPU (T1 → T5 → T9 → T10 → T11c → T12) ± 17 jam; dengan 2–3 sesi CPU paralel perkiraan selesai
± 20 jam kalender (dugaan), muat sebelum 2026-11-08 malam bila mulai 2026-11-06 malam. Unduhan tambahan:
± 4 GB model + satu set data (puluhan MB, dugaan).

---

**GARIS POTONG.** Butir di bawah hanya dikerjakan bila semua butir di atas LOLOS dan masih ada waktu
sebelum 2026-11-08 malam, urut dari atas.

| butir | isi | peta | ukuran |
|---|---|---|---|
| B1 | `lanjut_pretraining_mlm.ipynb` (TAPT: MLM di teks train+test, simpan checkpoint, dipakai `finetune_oof` lewat `PRETRAINED`) | P.kel.12, P.met.f.15 | M, 3 jam |
| B2 | `ner_bert.ipynb` contoh end-to-end (token classification, perataan subword, F1 entitas) | P.tugas.8, P.kel.5 | M, 3 jam |
| B3 | stopword Indonesia ke Konfigurasi di clustering, topik, topik_bertopic, topik_dinamis, regresi_teks, kata_kunci_ringkasan | P.tugas.20–21 | S, 1,5 jam |
| B4 | dokumen panjang: chunk + agregasi sebagai alt di `finetune_oof` | P.data.6, P.met.f.19 | M, 2 jam |
| B5 | AWP dan R-Drop sebagai alt di `finetune_oof` | P.met.f.10, f.12 | S, 1,5 jam |
| B6 | SetFit / fine-tune bi-encoder (pip `setfit`) | P.kel.9–10, P.tugas.27 | M, 3 jam |
| B7 | reranker cross-encoder untuk `retrieval` (unduh model) | P.kel.7, P.data.5 | M, 2,5 jam |
| B8 | QA fine-tune + sliding window | P.tugas.15 | L, 4 jam |
| B9 | seq2seq kecil (mT5/IndoT5) untuk ringkasan | P.kel.11, P.tugas.33 | L, 5 jam |
| B10 | bug kecil contoh lama: pool semi_supervised terduplikasi, ambang ujaran_kebencian dari model validasi, F1 entitas di `ner`, `retrieval` membaca kunci di sel Baca | – | S, 2 jam |
| B11 | leksikon sentimen (InSet) / model sentimen siap pakai sebagai fitur | P.rep.2, P.kel.19 | S, 2 jam |
| B12 | GLiNER, LLM kecil + LoRA | P.kel.17, P.kel.20 | bukan fokus |

## 5. Cek silang

- Tiap butir di atas garis punya: pilar, file, data demo, bentuk beda, perintah uji, gelombang, ukuran. ✅
- Model baru (T4) semua masuk `unduh_model.py`; paket baru di atas garis hanya `emoji`, `ipywidgets` (T1).
  Butir lain memakai paket terpasang (transformers, sklearn, lightgbm, catboost, charset_normalizer, pyarrow).
- Tiap langkah runbook menunjuk notebook yang ada atau butir T. ✅
- Sel ❌ yang tidak punya butir dan tidak di bawah garis = "diterima" di bagian 1, dengan alasan.
- Syarat gugur (zip) → T7 + `RENCANA_LOMBA.md`. Validator submission → T3, dipakai semua butir contoh.
- Risiko:
  - Venv: tanpa jawaban pertanyaan 1, uji memakai Python sistem yang tidak sama dengan PC panitia
    (`setup.bat` membuat `.venv` baru dari `requirements.txt`). Gladi `setup.bat` di PC lain belum ada buktinya.
  - Unduhan hari setup naik ± 4 GB; kecepatan internet BINUS belum diketahui. Cadangan: flashdisk.
  - Waktu/VRAM diukur di RTX 5060 Laptop; RTX 3070 desktop bisa lebih cepat atau lebih lambat (dugaan).
  - Semua angka waktu dan peluang soal di rencana ini dugaan sampai T5/T12 mengukurnya.
- Belum dicek perencana: isi `arsenal/vision/*` (tidak relevan teks), apakah `pyarrow` terbawa paket lain di
  `requirements.txt`, isi model card indobertweet.
