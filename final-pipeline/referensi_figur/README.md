# Referensi figur (training + EDA) dari paper

Contoh figur paper yang bagus untuk ditiru di `9_figur_ppt.py`.

| Apa | Isi | Di-commit? |
|---|---|---|
| `README.md` (ini) | Pilihan terkurasi + "versi kita" + data yang dibutuhkan | ya |
| `katalog.py` | Pengunduh + pembangun katalog | ya |
| `galeri.html` | **624 figur, 76 paper**, filter kelompok + kotak cari, klik gambar = perbesar | tidak (dibuat) |
| `katalog.md` | Daftar semua figur + caption asli; cari di sini dengan grep (`grep -i calibration katalog.md`) | tidak (dibuat) |
| `arxiv/<id>/` | Figur + `meta.json` (caption) dari 53 paper arXiv; paper tanpa figur HTML = halaman PDF `halNN.png` | tidak |
| `paperbench/` | 23 paper ICML 2024 dari PaperBench (`paper.md`, `paper.pdf`, `assets/`, rubrik) | tidak |
| `index.html` + `img/` | Galeri kurasi awal (24 figur di tabel bawah) | html ya, img tidak |

Repo **publik** dan gambar milik penulis paper, jadi semua gambar hanya ada di laptop pengguna. Bangun ulang:

```
hf download josancamon/paperbench --repo-type=dataset --local-dir paperbench   # 213 MB
python katalog.py unduh   # arxiv/, butuh internet, lewati yang sudah ada
python katalog.py         # katalog.md + galeri.html, offline
```

Pelajaran umum dari semua figur di bawah, dibanding `figur_ppt/` lama (dinilai pengguna "ampas", 2026-11-10):

1. Tampilkan **sebaran** (per contoh, per epoch, per seed, per kata), bukan satu angka ringkasan per model.
2. Satu figur = satu klaim; judul PPT berupa kesimpulan, bukan label ("Ensemble tidak menambah skor").
3. Angka penting/nama ditulis langsung di grafik (label langsung, bukan legenda jauh).
4. Sumbu jangan diperbesar sampai beda 0,001 terlihat besar; kalau beda < noise antar-seed, katakan itu.
5. Sertakan contoh teks asli (salah label, paling sulit, paling mudah). Juri ingat contoh, bukan angka.

Data yang dibutuhkan (kolom "Butuh") belum dicek ada di output pipeline; cek dulu sebelum membuat.

## Training

| File | Paper | Isi figur | Versi kita | Butuh |
|---|---|---|---|---|
| `carto_snli.png`, `carto_wino.png`, `carto_human.png` | Dataset Cartography, Swayamdipta dkk. 2020, [2009.10795](https://arxiv.org/abs/2009.10795) Gbr. 1 | Data map: titik per contoh, x = variabilitas, y = keyakinan ke label benar, warna = proporsi epoch benar; zona easy/ambiguous/hard | Data map fine-tune IndoBERT + 3 contoh teks per zona | prob per epoch per contoh latih |
| `forget_margin.png`, `forget_easy_hard.png`, `forget_stability.png` | Example Forgetting, Toneva dkk. 2019, [1812.05159](https://arxiv.org/abs/1812.05159) | Heatmap jumlah "lupa" vs margin; grid contoh tak-pernah-lupa vs sering-lupa; stabilitas peringkat antar seed | Tabel/grid teks "selalu benar" vs "bolak-balik" | prediksi per epoch |
| `ftbert_lr2e5.png`, `ftbert_lr4e4.png` | How to Fine-Tune BERT, Sun dkk. 2019, [1905.05583](https://arxiv.org/abs/1905.05583) | Kurva error train vs dev per iterasi; lr 4e-4 gagal total (error 0,5) | Kurva loss/F1 train vs val per epoch, dua lr berdampingan | log per epoch/step |
| `seeds_heatmap.png` | Revisiting Few-sample BERT, Zhang dkk. 2021, [2006.05987](https://arxiv.org/abs/2006.05987) | Heatmap skor seed-init x seed-data | Sebaran F1 antar fold/seed (strip/box) untuk tiap model | skor per fold |
| `scaling_trainvstest.png`, `scaling_curves.png` | Scaling Laws, Kaplan dkk. 2020, [2001.08361](https://arxiv.org/abs/2001.08361) | Kurva loss vs step, warna = ukuran data/model; train putus-putus vs test | Kurva belajar: F1 vs % data latih (25/50/75/100) untuk TF-IDF dan IndoBERT | latih ulang di subset |
| `dd_heatmap.png`, `dd_nlp.png` | Deep Double Descent, Nakkiran dkk. 2019, [1912.02292](https://arxiv.org/abs/1912.02292) | Heatmap error test (epoch x ukuran model); kurva NLP data kecil vs besar | Heatmap F1 (epoch x lr) atau (epoch x max_len) | grid kecil |
| `indonlu_f1.png` | IndoNLU, Wilie dkk. 2020, [2009.05387](https://arxiv.org/abs/2009.05387) | Bubble: F1 vs jumlah parameter (log), gelembung = biaya, label langsung | Pengganti `02_model`: F1 vs waktu latih/inferensi per model | skor + waktu per model |

## Evaluasi / keandalan

| File | Paper | Isi figur | Versi kita | Butuh |
|---|---|---|---|---|
| `calib_fig1.png` | On Calibration, Guo dkk. 2017, [1706.04599](https://arxiv.org/abs/1706.04599) Gbr. 1 | Atas: histogram keyakinan + garis rata-rata; bawah: reliability diagram dengan celah merah, teks "Error=" | Upgrade `05_keandalan`: TF-IDF vs IndoBERT berdampingan + ECE | prob OOF |
| `pik_calib.png`, `pik_token.png` | LMs (Mostly) Know What They Know, Kadavath dkk. 2022, [2207.05221](https://arxiv.org/abs/2207.05221) | Kurva kalibrasi diwarnai ukuran model; token diwarnai skor | Teks contoh dengan kata diwarnai bobot (koef TF-IDF/atribusi) | koef model linear |
| `labelerr_grid.png` | Pervasive Label Errors, Northcutt dkk. 2021, [2103.14749](https://arxiv.org/abs/2103.14749) | Grid contoh: label diberi vs label koreksi, per dataset | Grid teks: label asli vs prediksi yakin (calon salah label) | prob OOF |
| `shortcut_toy.png` | Shortcut Learning, Geirhos dkk. 2020, [2004.07780](https://arxiv.org/abs/2004.07780) | Diagram konsep rapi: train vs i.i.d. vs o.o.d. | Diagram kebocoran (mis. urutan baris memprediksi label) | - |

## EDA / data

| File | Paper | Isi figur | Versi kita | Butuh |
|---|---|---|---|---|
| `artifact_len.png` | Annotation Artifacts, Gururangan dkk. 2018, [1803.02324](https://arxiv.org/abs/1803.02324) | Kurva sebaran panjang teks per kelas | Ganti panel skor kolom di `01_data` dengan KDE panjang per kelas + kata pemicu per kelas | data latih |
| `nusax_transfer.png`, `nusax_vocab.png` | NusaX, Winata dkk. 2023, [2205.15960](https://arxiv.org/abs/2205.15960) | Heatmap train x test; heatmap irisan kosakata antar bahasa (angka di sel) | Heatmap irisan kosakata antar kelas/topik, atau train vs test | data latih/uji |

## Topik

| File | Paper | Isi figur | Versi kita | Butuh |
|---|---|---|---|---|
| `bertopic_comp.png` | BERTopic, Grootendorst 2022, [2203.05794](https://arxiv.org/abs/2203.05794) | Hanya waktu komputasi (paper-nya miskin figur) | Tiru visual dokumentasi BERTopic: peta topik, hirarki, dokumen 2D ([docs](https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html)) | embedding + topik |

## Pilihan tambahan dari katalog besar (dilihat satu per satu, 2026-11-10)

Path relatif ke folder ini. Kolom terakhir = untuk apa ditiru.

### EDA korpus teks (paling dekat dengan EDA kita)

| Path | Paper | Tiru untuk |
|---|---|---|
| `arxiv/2101.00027/00_pile_chart2.png` | The Pile | Treemap komposisi data per sumber/kategori → komposisi kelas x sumber |
| `arxiv/2101.00027/04_analysis_Pile_len_utf8bytes.png`, `05_analysis_overview_bytes_per_token.png` | The Pile | Histogram panjang; titik + galat per komponen → panjang/token per kelas |
| `arxiv/2101.00027/06_lda_perplexities.png`, `10_arXiv.png` | The Pile | Batang horizontal berurut + garis rata-rata; area bertumpuk topik per waktu |
| `arxiv/2104.08758/00_dates_freq.png`, `01_clean_country_freq.png`, `02_doc_embedding.png` | C4 docs | Sebaran waktu (log), batang negara berurut, peta embedding dokumen berwarna topik |
| `arxiv/2402.00159/00_log_counts.png`, `01_fertility.png` | Dolma | Frekuensi token (Zipf) per ID; batang dengan angka di atas |
| `arxiv/2406.17557/08_man_tfidf.png` | FineWeb | Tabel kata x kelompok berwarna (heatmap tabel) → kata khas per kelas |
| `arxiv/2310.20707/00_wimbd-fig1.png` | WIMBD | Diagram alur analisis data yang bersih |
| `arxiv/2309.10661/00_nulis-overlapping-lexicon.png`, `01_lm_quality_balanced.png`, `04_nulis-nlu-per-lang.png` | NusaWrites | Bubble irisan kosakata; violin per sumber; heatmap model x bahasa dengan angka |
| `arxiv/2212.09648/01_annotation_distribution.png`, `03_nusa_sankey_NLU.png` | NusaCrowd | Batang asal anotasi; **sankey** dataset → tugas → bahasa |

### Training / stabilitas

| Path | Paper | Tiru untuk |
|---|---|---|
| `arxiv/2002.06305/00_matrix.png`, `01_correlation_spearman.png` | Dodge dkk. 2020 | Heatmap skor seed-init x seed-data per tugas → variasi antar seed/fold |
| `arxiv/2006.04884/hal02.png`, `hal07.png`, `hal14.png` | Mosbach dkk. 2021 | Box plot run gagal vs berhasil; permukaan loss; box per konfigurasi (halaman PDF) |
| `arxiv/1803.09820/00_testLoss1.png`, `05_overfitting3.png`, `02_under-overfitting.png` | Smith 2018 | LR range test; kurva val loss per weight decay; skema under/overfitting |
| `arxiv/1712.09913/00_noshortLogRotatedLower.png` | Loss landscape | Gambar "wah" untuk slide motivasi (3D) |
| `paperbench/stay-on-topic-with-classifier-free-guidance/assets/fig3.jpg`, `fig14.jpg` | CFG | Batang menang/seri/kalah bertumpuk; scatter vs diagonal dengan teks "34 di atas, 23 di bawah" → model A vs B per contoh |
| `paperbench/ftrl/assets/asset_7.jpg` | FTRL | Kurva dengan pita (rata-rata ± std) dan legenda rapi → kurva belajar antar fold |
| `paperbench/what-will-my-model-forget/assets/fig_1.png`, `fig_2.png` | Forgetting | Kartu contoh teks + logit sebelum/sesudah → contoh salah prediksi |
| `paperbench/pinn/assets/asset_1.jpg`, `asset_8.jpg` | PINN | Kurva loss log per optimizer; batang + galat per lebar jaringan |
| `paperbench/test-time-model-adaptation/assets/asset_4.jpg` | FOA | Kurva akurasi vs jumlah sampel dengan baseline putus-putus |

### Evaluasi / interpretasi / topik

| Path | Paper | Tiru untuk |
|---|---|---|
| `arxiv/2003.07892/00_bert.png` | Desai & Durrett | Kalibrasi per dataset dalam satu panel (BERT/RoBERTa, in/out domain) |
| `arxiv/2211.09110/04_metrics_calibration.png` | HELM | Diagram penjelasan ECE dengan angka → slide "cara membaca kalibrasi" |
| `arxiv/1602.04938/00_mturk1_example.png` | LIME | Batang kata penting + dokumen dengan kata disorot → penjelasan prediksi teks |
| `arxiv/1703.01365/09_attributions.png`, `10_nmt.png` | Integrated Gradients | Kalimat dengan kata berwarna; heatmap token x token dengan angka |
| `arxiv/1906.05714/00_head_view_1_combined.png`, `04_bert_model_view.jpg` | BertViz | Garis atensi antar token; grid layer x head |
| `arxiv/2002.12327/00_wu_impact.png`, `02_bert-attn.png` | BERTology | Heatmap token x token; 5 pola atensi berdampingan |
| `arxiv/2008.09470/06_20news_semantic.png`, `07_20news_top2vec.png`, `09_topic_7_zoom.png` | Top2Vec | **Peta topik 2D berlabel kata** + zoom satu topik → ganti `10_topik_peta` |
| `arxiv/2008.09470/03_topic_vector.png`, `04_topic_words.png` | Top2Vec | Skema penjelasan cara kerja topik berbasis embedding (slide metode) |
| `arxiv/2107.02173/hal07.png` | Hoyle dkk. | Box koherensi otomatis vs penilaian manusia (halaman PDF) |
| `paperbench/lca-on-the-line/assets/asset_1.jpg`, `asset_3.jpg` | LCA | Scatter dengan elips + anotasi teks besar; pohon kelas → hirarki label/topik |
| `paperbench/mechanistic-understanding/assets/asset_1.jpg`, `asset_4.jpg` | DPO | Kurva dengan pita + garis abu penanda; dumbbell sebelum→sesudah |
| `paperbench/lbcs/assets/asset_3.jpg` | LBCS | Radar banyak metode x dataset (pakai hemat) |
| `paperbench/semantic-self-consistency/assets/asset_1.jpg` | SSC | Diagram pipeline berwarna → slide alur metode kita |

## Gaya presentasi

| File | Paper | Catatan |
|---|---|---|
| `pb_overview.png` | PaperBench, Starace dkk. 2025, [2504.01848](https://arxiv.org/abs/2504.01848) | Diagram alur bersih; grafik hasil ada di PDF |

Sumber gambar: `https://ar5iv.labs.arxiv.org/html/<id>` (bagian `assets/`), PaperBench dari `https://arxiv.org/html/2504.01848v3`,
Guo dkk. dipotong dari PDF halaman 1.
