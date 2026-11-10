# RISET_TEKS_PENCARI: celah metode + visualisasi teks (2026-11-09)

Hasil agen pencari. Tujuannya dua: arsenal teks selengkap mungkin, dan **semua grafik serta tabel tersimpan
sebagai file** karena saat membuat PPT data lomba tidak bisa dibuka lagi. Status repo dicek dengan grep pada
`contoh/text/*.ipynb`, `arsenal/teks/**`, `contoh/visualisasi/*.ipynb`, dan paket di `../.venv`. Status
butir di `RENCANA_TEKS.md` bagian 1 (tanggal 06-11) sebagian sudah usang: T2–T10 sudah menutup banyak ❌.

Paket yang terpasang di venv: `shap 0.52`, `umap-learn`, `wordcloud`, `plotly`, `python-pptx`, `seaborn`,
`networkx`, `cleanlab`, `emoji`, `rapidfuzz`, `bertopic`, `hdbscan`, `optuna`, `Sastrawi`, `nltk`.
**Tidak terpasang:** `lime`, `scattertext`, `datasketch`, `captum`, `kaleido` (artinya plotly tidak bisa
diekspor ke PNG, hanya ke HTML), `setfit`. Usulan di bawah sengaja tidak menambah dependensi.

## Temuan paling penting (dicek)

1. **Tidak satu pun dari 46 notebook `contoh/text` dan 12 notebook `arsenal/teks` memanggil `savefig`.**
   25 notebook menampilkan grafik dengan `plt.show()` saja. Notebook disimpan tanpa output (aturan repo), jadi
   tanpa langkah tambahan semua grafik **hilang** setelah kernel ditutup.
2. `contoh/visualisasi/*.ipynb` memang menyimpan PNG, tetapi ke **folder kerja notebook**
   (`plt.savefig("teks_sebaran_kelas.png")`), bukan ke `OUTPUT_FOLDER`. Datanya juga data demo generik,
   bukan OOF model teks.
3. Beberapa notebook lama menulis CSV ke folder notebook (`topik_bertopic`, `analisis_kesalahan`,
   `inggris`, dll.); itu sebabnya ada `hasil.csv`, `topik_dokumen.csv`, dan sejenisnya di `contoh/text/`.
4. **Solusi tanpa mengubah notebook, sudah diuji di scratchpad:**
   `jupyter nbconvert --execute --to markdown --output-dir <OUT> nb.ipynb` menghasilkan `nb.md`, dan setiap
   grafik diekstrak ke `nb_files/*.png`. Perintah `--to html` menghasilkan satu HTML berisi semua grafik dan
   tabel. Kedua cara ini jalan di venv repo.

## (a) Celah metode, urut prioritas (dampak ke skor × peluang muncul)

| # | metode | status repo | catatan |
|---|---|---|---|
| 1 | Deteksi **near-duplicate** train↔test dan di dalam train (TF-IDF cosine NN / `rapidfuzz`) | **BELUM** (duplikat persis ada di `teks/dasar/praproses` #2) | Data medsos ID sering penuh retweet/copas. Selain itu bisa dipakai sebagai "tebak dari tetangga" dan jadi slide temuan |
| 2 | **Prior shift** test (EM Saerens) setelah kalibrasi | **BELUM** (`ensemble_oof` hanya punya prior baseline) | Naik kalau proporsi kelas test ≠ train. Murah, ±15 baris |
| 3 | **AWP** (adversarial weight perturbation) di fine-tune | **BELUM** (FGM sudah ada di `finetune_oof`) | Teknik juara seri Feedback Prize Kaggle. Waktu latih +50–100%. Jadikan alt ber-`#` di samping `FGM_EPS` |
| 4 | **R-Drop** / **multi-sample dropout** untuk klasifikasi | R-Drop BELUM; MSD hanya di `finetune_regresi` | Alt ber-`#` di `finetune_oof` |
| 5 | **Atribusi token** (kata mana yang mendorong prediksi) | **BELUM** (captum tidak ada) | Untuk linear: kontribusi = `coef × tfidf` (pasti, tanpa paket). Untuk transformer: gradient×input, ±15 baris torch |
| 6 | Kata pembeda **weighted log-odds** (Fightin' Words, Monroe 2008) | **BELUM** (chi² sudah ada di `praproses` #4) | Lebih jujur daripada frekuensi atau chi² untuk kata langka. Implementasi manual karena scattertext tidak terpasang |
| 7 | **OOV / tumpang tindih kosakata** train↔test | **BELUM** | Pemicu memilih char n-gram atau TAPT. Sekaligus jadi slide |
| 8 | Distilasi (soft label dari ensemble → 1 model) | **BELUM** (hasil grep "distil" hanya nama model distilbert) | Peluangnya kecil di 5 jam. Prioritas rendah |
| 9 | Retrieve + **rerank** cross-encoder | **BELUM** (tidak ada model reranker di `pretrained/`) | Pola juara Eedi/MAP (memakai LLM besar). Hanya relevan kalau soalnya ranking/retrieval. Perlu unduhan |
| 10 | Deteksi teks buatan AI | **BELUM** | Juara Kaggle "LLM Detect AI" memakai LLM fine-tune + data sintetis. Peluang muncul di GEMASTIK kecil (dugaan) |
| 11 | Klasifikasi hierarkis | **BELUM** (sengaja diterima di RENCANA) | Bisa didekati dengan meratakan label menjadi kelas gabungan |

Sudah **ADA** (tidak perlu dikerjakan): NB-SVM (`baseline_oof`, `teks/klasifikasi/tfidf`), adversarial
validation (`baseline_oof`), pseudo-label (`finetune_oof`, `semi_supervised_teks`), cleanlab
(`label_berisik`), hill climbing/Caruana (`ensemble_oof`, `embedding_llm_oof`), temperature scaling
(`ensemble_oof`), LLRD/FGM/EMA/focal/label smoothing/re-init/grad checkpoint/head+tail (`finetune_oof`),
TAPT (`mlm_tapt`), SetFit kontrastif, LoRA LLM kecil, ordinal/QWK (`finetune_regresi`), dokumen panjang
(chunk), multilabel + ambang per label, NER BERT, kamus alay (`ujaran_kebencian`), topik dinamis, BERTopic.

## (b) Daftar visualisasi untuk PPT

Status: ADA = ada dan memakai data/model teks · GENERIK = ada di `contoh/visualisasi` tetapi untuk data demo
(perlu disambungkan ke OOF teks) · BELUM. Semua cukup memakai matplotlib/seaborn kecuali disebut lain.

### EDA

| visual | status | cara buat (1 baris) |
|---|---|---|
| Sebaran kelas (bar + persen) | ADA (`visualisasi/teks` #1, banyak contoh) | `value_counts().plot.bar()` |
| Distribusi panjang (kata/karakter/token) per kelas | ADA (`visualisasi/teks` #5) | histogram/violin per kelas; untuk token pakai tokenizer model |
| Kata & n-gram teratas per kelas | ADA (`visualisasi/teks` #2–3, #6) | `CountVectorizer(ngram_range)` → barh per kelas |
| Wordcloud per kelas | ADA (`visualisasi/teks` #4) | `wordcloud.WordCloud().generate_from_frequencies` |
| Jaringan ko-okurensi kata | ADA (`visualisasi/teks` #7) | `networkx` |
| **Kata pembeda log-odds berbobot** (kata vs z-score, sumbu x frekuensi) | BELUM | rumus Monroe: δ = log-odds kelas A − B dengan prior Dirichlet dari korpus; z = δ/√var; scatter |
| **Kosakata train vs test**: % token test yang OOV, kata baru test teratas | BELUM | himpunan kata train/test → bar + tabel CSV |
| **Ciri medsos per kelas**: emoji, #hashtag, @mention, URL, angka, HURUF KAPITAL, tanda seru | SEBAGIAN (`inggris` hanya EDA emoji/hashtag) | regex `umum/teks_regex` → rata-rata per kelas → heatmap |
| Campur bahasa (rasio kata Inggris/daerah) | SEBAGIAN (`bahasa_daerah` per bahasa) | kamus kata + rasio per teks → box per kelas |
| **Near-duplicate** train↔test (jumlah, contoh, konflik label) | BELUM | `NearestNeighbors(metric="cosine")` di TF-IDF char → histogram jarak + tabel |
| Drift train↔test (adversarial AUC + token paling pembeda) | SEBAGIAN (`baseline_oof` hanya mencetak AUC) | koefisien LogReg adversarial → barh token |
| Konflik label (teks sama, label beda) | SEBAGIAN (`praproses` #2 deteksi) | groupby teks bersih → nunique label > 1 → tabel |
| Tren waktu / topik per waktu | ADA (`topik_dinamis`) | – |

### Model dan evaluasi (sumber: OOF yang sudah disimpan `*_oof`)

| visual | status | cara buat |
|---|---|---|
| Confusion matrix ternormalisasi | GENERIK (`visualisasi/evaluasi_klasifikasi` #1); `plt.show` di contoh | `ConfusionMatrixDisplay.from_predictions(normalize="true")` |
| ROC / PR per kelas | GENERIK (#2–3) | `RocCurveDisplay`/`PrecisionRecallDisplay` dari OOF |
| Kurva kalibrasi | GENERIK (#6) | `CalibrationDisplay.from_predictions` |
| F1 per kelas | GENERIK (#7) | `classification_report(output_dict=True)` → barh |
| Threshold / pengali kelas vs F1 | GENERIK (#5); pengali ada di `baseline_oof` tanpa plot | loop ambang → garis |
| Learning curve | GENERIK (#8) | `learning_curve` (hanya model cepat) |
| **Skor per fold (box) per model** | BELUM | baca `percobaan.csv` → boxplot |
| **Perbandingan model** (CV, waktu, ukuran) | BELUM (skor hanya di KATALOG/log) | `percobaan.csv` → barh terurut, sorot yang terbaik |
| Korelasi OOF antar model | ADA (`ensemble_oof` "Korelasi antar model") | heatmap |
| **Tabel/grafik ablation** (+x per teknik) | SEBAGIAN (`ensemble_oof` menulis `ablation.csv`) | csv → waterfall/barh |
| Bobot ensemble | BELUM (bobot hanya dicetak) | barh bobot hill climbing |
| Histogram keyakinan: benar vs salah | BELUM | `max(proba)` dipisah benar/salah → hist |

### Interpretasi & analisis kesalahan

| visual | status | cara buat |
|---|---|---|
| Kata berpengaruh model linear (koefisien top ±) | ADA (`teks/klasifikasi/tfidf` #5) tanpa simpan | barh koefisien per kelas |
| **Sorot kata per contoh** (atribusi token) | BELUM | linear: `coef[k] * x_tfidf` → teks berwarna HTML; transformer: gradient×input |
| SHAP | GENERIK tabular (`visualisasi/penjelasan_model`) | untuk teks: pakai fitur TF-IDF/SVD + `shap.LinearExplainer` (belum dicek di teks) |
| UMAP/t-SNE embedding diwarnai kelas (+ salah prediksi ditandai) | GENERIK (`visualisasi/embedding_klaster`), `topik_bertopic` | `umap.UMAP().fit_transform(emb)` → scatter warna kelas, marker x untuk salah |
| Peta topik / kata per topik | ADA (`topik`, `topik_bertopic`, `clustering`) | – |
| Kelas yang tertukar | ADA (`analisis_kesalahan` #1) | – |
| **Error per panjang teks / per kelompok** (bin panjang, ada emoji, ada negasi) | SEBAGIAN (`analisis_kesalahan` #2) | `pd.qcut(panjang)` → F1 per bin → bar |
| **Contoh salah paling yakin** (tabel CSV untuk slide) | SEBAGIAN (`analisis_kesalahan` #5, CSV ke folder notebook) | urut `proba` salah desc → `to_csv(OUTPUT_FOLDER)` |
| Label mencurigakan (cleanlab) | ADA (`label_berisik`) tanpa simpan grafik | – |

## (c) Notebook teks yang belum menyimpan figur ke file

Semuanya. `contoh/text`: 46/46 tanpa `savefig`. Yang punya `plt.show` (grafik akan hilang):
analisis_kesalahan, bahasa_daerah, baseline_oof (2), clustering, embedding_ollama, ensemble_oof,
finetune_oof, inggris (2), klasifikasi_embedding (2), klasifikasi_finetune (2), klasifikasi_tfidf (2),
llm_prompt, multilabel, regresi_teks (2), tanya_jawab, teks_tabular, topik, topik_dinamis (2),
zero_shot_nli. `arsenal/teks/*` 12/12 tanpa `savefig`; satu-satunya pola simpan di
arsenal adalah `arsenal/umum/plot.ipynb` (`PLOT_FOLDER`). `contoh/visualisasi/*` menyimpan ke folder
notebook, bukan `OUTPUT_FOLDER`.

## (d) Usulan pekerjaan berikutnya (urut)

| # | pekerjaan | bentuk | sumber |
|---|---|---|---|
| 1 | **Jalankan notebook lewat nbconvert** ke `../../outputs/<kategori>/<nama>/` (`--to markdown` untuk PNG terpisah + `.md`, `--to html` untuk satu file). Cukup satu baris di runbook atau satu `.bat` kecil. Semua grafik, tabel, dan cetakan tersimpan tanpa menyentuh 46 notebook. Sudah diuji di venv | perintah/`.bat` (bukan notebook) | https://nbconvert.readthedocs.io/en/latest/usage.html |
| 2 | Sel **simpan otomatis**: hook IPython `post_execute` yang menyimpan setiap figur terbuka ke `OUTPUT_FOLDER/fig_NN.png`. Alternatif kalau notebook dijalankan interaktif. **Belum diuji** | 1 sel di `arsenal/umum/plot` + baris tempel | https://ipython.readthedocs.io/en/stable/config/callbacks.html |
| 3 | `contoh/text/eda_presentasi.ipynb` (baru): semua EDA teks berbasis data soal, masing-masing **PNG + CSV** ke `OUTPUT_FOLDER`: kelas, panjang, n-gram, log-odds berbobot, OOV train↔test, ciri medsos, campur bahasa, near-dup, konflik label, token pembeda adversarial | notebook baru (loader sendiri) | https://convokit.cornell.edu/documentation/fightingwords.html · https://yorko.github.io/2023/practical-near-dup-detection |
| 4 | `contoh/text/laporan_model.ipynb` (baru): membaca `OOF_FOLDER` + `percobaan.csv` + `ablation.csv` dan mengeluarkan semua visual model (CM, PR, kalibrasi, F1 per kelas, ambang, box per fold, perbandingan model, bobot ensemble, keyakinan benar vs salah, error per bin panjang, contoh salah paling yakin, UMAP salah prediksi). Isinya bisa disalin dari `visualisasi/evaluasi_klasifikasi` | notebook baru | https://scikit-learn.org/stable/visualizations.html |
| 5 | Atribusi token: alt di `teks/klasifikasi/tfidf` (koefisien×tfidf → HTML teks berwarna, disimpan ke file) dan sel gradient×input di `finetune_oof` | alt ber-`#` / 1 sel | https://arxiv.org/abs/1703.01365 (IG; gradient×input adalah versi sederhananya) |
| 6 | `ensemble_oof`: alt **EM Saerens** setelah kalibrasi | alt ber-`#` | https://doi.org/10.1162/089976602753284446 · https://arxiv.org/abs/1901.06852 |
| 7 | `finetune_oof`: alt **AWP**, **R-Drop**, **multi-sample dropout** di samping `FGM_EPS` | alt ber-`#` (loader sama) | https://arxiv.org/abs/2004.05884 · https://arxiv.org/abs/2106.14448 · https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx |
| 8 | Rapikan notebook lama yang menulis CSV/PNG ke folder notebook (`topik_bertopic`, `analisis_kesalahan`, `inggris`, `visualisasi/*`) agar menulis ke `OUTPUT_FOLDER` | edit kecil | aturan repo (`_memori/aturan.md`) |
| 9 | (Opsional, hanya kalau soalnya ranking) reranker cross-encoder kecil di `download_model.bat` + alt di `retrieval` | unduh + alt | pola juara Eedi/MAP: https://eedi.com/blog/from-wrong-answers-to-real-insights-how-we-used-a-kaggle-challenge-to-map-student-misconceptions |

Belum dicek / catatan jujur:
- Write-up resmi juara Kaggle teks 2025 (Jigsaw Agile Community Rules, MAP) tidak berhasil diambil; yang
  ada hanya ringkasan pihak ketiga. Klaim "AWP dipakai juara Feedback Prize" berasal dari laporan W&B
  (sumber sekunder).
- API `shap.maskers.Text` untuk pipeline TF-IDF belum diverifikasi; usulan 5 sengaja memakai
  koefisien×tfidf (pasti benar untuk model linear).
- Hook `post_execute` (usulan 2) belum diuji. nbconvert (usulan 1) sudah diuji.
- Kamus alay Salsabila 2018 (±3.600 kata) sudah dipakai repo melalui `ujaran_kebencian`/`praproses`; isi
  kamusnya tidak dicek ulang. https://huggingface.co/datasets/SEACrowd/kamus_alay
