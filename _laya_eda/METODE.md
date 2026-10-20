# Metode Laya EDA

Dokumen ini menjelaskan setiap metode yang dipakai aplikasi `_laya_eda`, alasan memilihnya,
batasannya, rujukannya, dan **angka hasil uji kita sendiri**. Bagian metode untuk fitur/pembuat
kode ada di bagian 9 dan 10.

Angka bertanda **[diukur 20 Okt 2026]** dihasilkan ulang oleh `_uji/ukur_metode.py` (CPU, kecuali
Laya 500 baris) atau `_uji/test_app.py`. Angka bertanda **[ukuran sebelumnya]** berasal dari
pengujian sebelumnya dan tidak diukur ulang di sesi ini.

## 1. Peta singkat: bagian aplikasi, metode, rujukan

| Bagian (tab) | Metode | Rujukan |
|---|---|---|
| Data | duplikat persis / mirip, panjang teks | - |
| Profil & label | skew, kategori langka, tipe campur; chi-square + Cramer V; Kruskal-Wallis + epsilon2; Spearman; chi-square untuk data kosong dan pergeseran waktu | Pearson 1900; Cramer 1946; Cohen 1988; Kruskal & Wallis 1952; Tomczak & Tomczak 2014; Spearman 1904; Little & Rubin 2019 |
| Cek angka | pembacaan angka format Indonesia; IQR Tukey, boleh skala log | Tukey 1977 |
| Teks & konsistensi | kunci sederhana + rapidfuzz ratio >= 90; konflik label; n-gram chi-square; kosakata; kamus alay; stopword | Levenshtein 1966; Salsabila dkk. 2018 |
| Temukan kategori | MiniLM (Sentence-BERT) + KMeans + kata kunci TF-IDF | Reimers & Gurevych 2019, 2020; Wang dkk. 2020; Arthur & Vassilvitskii 2007 |
| Lensa Laya | keputusan sekali jalan, probabilitas terkalibrasi, pilihan kedua | Guo dkk. 2017 (kalibrasi); dokumentasi paket `laya` 0.4.8 |
| Peta data | ringkasan per kategori, tabel silang, PCA + t-SNE, silhouette | van der Maaten & Hinton 2008; Rousseeuw 1987 |
| Outlier | IQR per kategori, jarak ke pusat makna, nyasar, kemiripan kosinus, alasan | Tukey 1977; Northcutt dkk. 2021 (konteks salah label) |
| Fitur → label | lab fitur CV 5 lipatan, ablation Laya, permutation importance, mutual information, confident learning, cek fitur bocor | Breiman 2001; Kraskov dkk. 2004; Northcutt dkk. 2021 |
| Train vs test | KS, chi-square + Cramer V, PSI, OOV, adversarial validation | Massey 1951; Siddiqi 2006; Zajac 2016 |
| Temuan utama, Kode EDA | temuan diurutkan nilai penting; codegen menulis notebook dari pilihan + catatan | - |

## 2. Alasan outlier (kolom `alasan`)

Setiap baris yang ditandai mendapat satu atau beberapa kalimat berbahasa Indonesia yang memuat
angka bukti. Jenis dan contohnya:

* **Angka per kategori** - "harga Rp 8,95 jt adalah 0,002 kali median jenis_asli 'komersial'
  (median Rp 4,23 M). Batas wajar jenis_asli 'komersial': Rp 684 jt sampai Rp 24,2 M. Persentil 0
  di jenis_asli 'komersial', 19 di seluruh data. Status: janggal di kategorinya dan di seluruh
  data." Batas dihitung pada log10 bila skala log, lalu dibalik ke satuan asli (10^batas).
* **Angka tak terbaca / negatif / nol** - isi aslinya disebut, mis. "harga berisi 'nego' yang
  tidak terbaca sebagai angka (isiannya tidak diawali angka)".
* **Angka di luar batas seluruh data** - dipakai untuk baris yang belum punya kategori.
* **Teks jauh / nyasar** - jarak makna ke pusat kategorinya, median dan batas jarak kategori itu,
  kategori lain terdekat beserta jaraknya.
* **Laya ragu** - "Laya (lensa 'kelengkapan_alamat') memilih '1 hanya kota' 53%, pilihan kedua
  '2 kecamatan dan kota' 40% - selisih 13 poin; keyakinan 0,528 di bawah batas 0,6." Selisih di
  bawah 10 poin disebut "hampir seri". Untuk ya/tidak yang disebut peluang ya dan tidak. File
  hasil lama (tanpa pilihan kedua) tetap terbaca; alasannya hanya menyebut pilihan dan keyakinan.
* **Duplikat, hampir duplikat, konflik label** - nomor baris lain dan labelnya, mis. "KONFLIK
  label: baris ini berlabel 'fear', tetapi baris 1268 berlabel 'sadness' padahal teksnya
  sama/hampir sama."

Aturan: status "janggal hanya di keseluruhan" (wajar untuk kategorinya) **tidak** dijadikan alasan
di kolom `alasan`; tabel di tab Outlier tetap menampilkannya. Alasan dihitung terhadap satu kategori
pembanding yang dipilih di panel samping. Kartu baris ("Periksa satu baris") menampilkan semua
alasan satu baris. Kolom `alasan` ikut di tab Telusuri, tab Outlier, dan file ekspor.

Format Laya di `laya_<lensa>.csv`: `row, answer, value, confidence, first_value, second,
second_value` (tiga kolom terakhir baru). `first_value` = peluang pilihan yang dipilih,
`second` = pilihan kedua, `second_value` = peluangnya. Untuk skala, pilihan = tingkat 0..k-1 yang
dibulatkan dari skor. Untuk ya/tidak, `second` = jawaban yang berlawanan.

## 2b. Temuan utama (`ctx["findings"]`)

Tiap bagian menambahkan temuan pentingnya berupa kamus `{bagian, temuan, alasan, saran, penting}`.
`penting` (0..1) adalah heuristik pengurutan: untuk masalah data = 10 x porsi baris yang terkena;
untuk hubungan dengan label = ukuran efek / 0,5; untuk Laya ragu dan teks nyasar dibatasi 0,7
karena pada kelas yang memang bertumpuk porsinya besar; temuan informatif (kelompok, kata khas)
0,3-0,4. Semuanya dipotong ke 0..1. Ini bukan peluang atau skor yang dikalibrasi.

## 3. Metode tabel

### 3.1 Profil kolom
Skew = momen ketiga terstandar (pandas). |skew| > 1 = miring; miring ke kanan dan semua nilai > 0 =
disarankan skala log. Kategori langka = di bawah 1% baris, hanya untuk kolom kategori (<= 50 nilai
unik, atau kolom angka <= 10 nilai unik). Tipe campur = kolom teks yang >= 50% terbaca angka tetapi
tidak 100%. Ambang adalah konvensi, bukan hukum.

### 3.2 Hubungan kolom dengan label
* Kolom kategori vs label kelas: uji chi-square independensi, ukuran efek **Cramer V** =
  sqrt(chi2 / (n (min(baris, kolom) - 1))).
* Kolom angka vs label kelas: **Kruskal-Wallis** H, ukuran efek **epsilon2** = (H - k + 1) / (n - k)
  (Tomczak & Tomczak 2014).
* Label angka (> 10 nilai unik): **Spearman** rho untuk kolom angka; Kruskal-Wallis untuk kolom
  kategori.
* Ambang efek (Cohen 1988): V dan rho 0,1 / 0,3 / 0,5 (kecil / sedang / besar); epsilon2 0,01 /
  0,06 / 0,14. Signifikan bila p < 0,05. Tafsir dibuat otomatis, mis. "signifikan (p < 0,001), efek
  kecil". Hubungan signifikan dengan efek diabaikan diberi catatan khusus.
* Kenapa ukuran efek: pada data besar p-value hampir selalu kecil. Kenapa berbasis peringkat:
  sebaran harga dan panjang teks miring.
* Batasan: satu kolom per uji (tanpa interaksi); V cenderung naik pada tabel kecil; hubungan terlalu
  kuat dapat berarti kolom itu bocoran label.

### 3.3 Korelasi, data kosong, waktu
* Korelasi Spearman antar kolom angka; pasangan redundan bila |rho| > 0,9.
* Data kosong: penanda kosong per kolom diuji chi-square terhadap label (+ Cramer V). Pasangan kolom
  dengan korelasi penanda kosong > 0,95 = kosong bersamaan. Tidak membedakan MCAR/MAR/MNAR secara
  formal (Little & Rubin 2019).
* Waktu: `to_datetime` (format campur), jumlah per bulan, sebaran label per bulan; baris diurutkan
  menurut waktu lalu dibagi dua sama banyak, sebaran label awal vs akhir diuji chi-square + V.

### 3.4 Cek angka: IQR Tukey, skala log
Q1 dan Q3 = persentil 25 dan 75, IQR = Q3 - Q1. Outlier bila nilai < Q1 - 1,5 IQR atau > Q3 + 1,5
IQR (Tukey 1977). Pada skala log semua dihitung pada log10 (hanya nilai > 0; nilai <= 0 dilaporkan
terpisah). Per kategori: Q1, Q3 dihitung di dalam tiap kategori; `kali_median` = nilai / median
kategori; status = janggal hanya di kategorinya / di keduanya / hanya di keseluruhan / normal.
Sekitar 0,7% data berdistribusi normal pun melewati batas 1,5 IQR; itu sebabnya hasilnya daftar
kandidat.

Pembacaan angka format Indonesia: "Rp" dan spasi dibuang, titik = ribuan, koma = desimal, lalu angka
di awal teks dibaca (`12abc` terbaca 12; `nego` tidak terbaca).

## 4. Metode teks dan makna

### 4.1 TF-IDF dan kata khas
`TfidfVectorizer(max_features=20000, min_df=2)`. Kata khas kelompok = rata-rata TF-IDF di kelompok
dikurangi rata-rata seluruh teks, 8 teratas (Sparck Jones 1972; Salton & Buckley 1988).

### 4.2 MiniLM / Sentence-BERT
Model `paraphrase-multilingual-MiniLM-L12-v2` (Reimers & Gurevych 2019, 2020; arsitektur MiniLM:
Wang dkk. 2020) mengubah teks menjadi vektor 384 dimensi, dinormalkan panjang 1 sehingga kemiripan
kosinus = perkalian titik. Dijalankan di GPU, batch 64.

### 4.3 KMeans
`KMeans(n_clusters=k, n_init=4, random_state=42)` pada vektor ternormalisasi (jarak Euklides pada
vektor panjang 1 setara kosinus). Inisialisasi k-means++ (Arthur & Vassilvitskii 2007) adalah
bawaan scikit-learn. Jumlah kelompok ditentukan pengguna; kelompok adalah usulan untuk diberi nama.

### 4.4 Jarak ke pusat, jauh, nyasar
Pusat kategori = rata-rata vektor anggotanya, dinormalkan. Jarak = 1 - kosinus ke pusat. **Jauh** =
jarak di atas Q3 + 1,5 IQR jarak kategori itu dan di atas median. **Nyasar** = pusat terdekat bukan
kategorinya sendiri. Alasan menyebut juga kategori lain terdekat.

### 4.5 Kemiripan kosinus (parafrase)
Pasangan dengan kosinus >= batas (bawaan 0,93), duplikat persis dibuang, maksimal 3000 pasangan,
dihitung per blok 1000 baris. "Suka" dan "tidak suka" bisa sangat mirip: ini penyaring, bukan hakim.

### 4.6 Nilai kategori tidak konsisten
1. Kunci sederhana = huruf kecil, hanya a-z dan 0-9. Nilai berkunci sama digabung ke yang paling
   sering muncul.
2. Antar kunci: `rapidfuzz.fuzz.ratio` >= 90 (kemiripan Indel ternormalisasi 0..100; Levenshtein 1966
   untuk gagasan jarak sunting). Digabung greedy ke kunci yang lebih sering. Pasangan yang hanya
   beda angka (kelompok_1 vs kelompok_2) dilewati. Maksimal 1500 nilai unik per kolom.
Keluaran: tabel saran + peta koreksi (`dict`) yang bisa ditempel ke notebook.

### 4.7 Konflik label
Teks dikelompokkan menurut kunci sederhana (sama persis atau hanya beda huruf besar/tanda baca).
Kelompok dengan >= 2 label berbeda = konflik. Konteks: salah label di dataset benchmark umum
(Northcutt dkk. 2021).

### 4.8 N-gram khas, kosakata, kata alay, campuran bahasa
* `CountVectorizer(ngram_range=(1,2), min_df=3, binary=True)`; per kelas, `chi2` satu-lawan-semua,
  hanya frasa yang lebih sering di kelas itu.
* Kosakata dan hapax (kata muncul sekali): kata = `\w+` huruf kecil.
* Kata alay: ada di `kamus_alay_salsabila.csv` (Salsabila dkk. 2018) + `kamus_slang.csv` di
  `data_latihan/preprocessing`, selain yang bentuk bakunya sama. Bila file tidak ada, bagian ini
  dilewati dengan pesan.
* Campuran bahasa: teks yang memuat >= 1 stopword Inggris dan >= 1 stopword Indonesia dari dua daftar
  kecil (sekitar 45 kata); deteksi sederhana dan tidak mengenali bahasa daerah.

### 4.9 Peta 2D makna dan silhouette
Sampel acak (maks 5000) vektor MiniLM -> PCA 50 dimensi -> t-SNE 2 dimensi (CPU, perplexity =
min(30, n/4), init PCA; van der Maaten & Hinton 2008). t-SNE hanya bermakna untuk kedekatan lokal:
jarak antar gugus dan ukuran gugus tidak bisa dibaca (Wattenberg dkk. 2016). **Silhouette**
(Rousseeuw 1987) dihitung pada vektor asli 384 dimensi dengan jarak kosinus: s = (b - a) / max(a, b).
Tafsir kasar (Kaufman & Rousseeuw 1990): > 0,5 terpisah jelas, 0,25-0,5 cukup, di bawahnya lemah.
Untuk teks, nilai di bawah 0,1 biasa, jadi bandingkan antar kelas.

## 5. Laya

Laya adalah model keputusan sekali jalan: untuk setiap teks dan pertanyaan ia mengeluarkan
peluang tiap pilihan dalam satu putaran. Tiga jenis: `choice` (pilih satu), `noul` (ya/tidak),
`score` (skala). "Yakin" = `answer_confidence`, keyakinan yang dikalibrasi (gagasan penskalaan suhu:
Guo dkk. 2017). Laya kuat soal makna dan lemah soal angka, jadi angka dicek dengan pandas.
Pilihan kedua disimpan agar alasan "ragu" bisa disebut dengan angka.

## 6. Angka hasil uji

### 6.1 Uji otomatis aplikasi
`python _uji/test_app.py` **[diukur 20 Okt 2026]** menjalankan seluruh tab pada tiga skenario:
data emot_train dengan file Laya lama, properti (mode baris, Laya 3 lensa x 300 baris, MiniLM 1500
teks, t-SNE 500 titik), dan emot dengan label diacak 10% serta kolom tambahan (MiniLM 3521 teks,
t-SNE 1000 titik). Hasil: lolos tanpa exception. Kasus khusus yang diperiksa: file Laya lama
(tanpa kolom `second`) tetap terbaca dan diperbarui otomatis saat ditambah (100 baris lama tanpa
pilihan kedua, baris baru terisi).

### 6.2 Outlier angka (properti demo, 3000 baris; 56 harga mustahil disisipkan) [diukur 20 Okt 2026]
| Cara | Harga mustahil tertangkap | Baris normal ikut |
|---|---|---|
| IQR skala log **per kategori** (`jenis_asli`) | 54 dari 56 | 0 dari 2710 |
| IQR skala log seluruh data (tanpa kategori) | 56 dari 56 | 529 dari 2710 |

Cakupan kolom `alasan` (angka + duplikat, tanpa Laya dan tanpa vektor teks), kategori pembanding
`jenis_asli`:

| Masalah yang disisipkan | Baris | Punya alasan |
|---|---|---|
| harga mustahil | 56 | 98,2% |
| harga berupa teks ("nego", "-") | 46 | 100% |
| luas negatif | 51 | 100% |
| salah jenis (label dibalik) | 50 | 52,0% |
| alamat kabur | 41 | 0% (butuh lensa Laya / teks) |
| deskripsi kosong | 46 | 0% (butuh lensa Laya / teks) |
| baris normal | 2710 | 0% |

Mengumpulkan semua alasan untuk 3000 baris memerlukan 0,07 detik. Pada 20 ribu baris (emot x 6)
satu putaran ulang seluruh aplikasi tanpa label memerlukan sekitar 2,5-3 detik dan dengan label
sekitar 7-8 detik pada putaran pertama (hasil uji ditahan cache sesudahnya).

### 6.3 Laya, pilihan kedua, dan "ragu" (lensa jenis_properti, 500 baris acak, mode baris) [diukur 20 Okt 2026]
* Akurasi pilihan pertama 94,1% (dari 492 baris yang labelnya benar). Pilihan pertama atau kedua
  benar: 100% (29 jawaban salah, seluruhnya benar di pilihan kedua).
* Laya ragu (yakin < 0,6) pada 33 baris: akurasi di baris ragu 60,6% vs 96,5% di baris yakin;
  13 dari 29 jawaban salah tertangkap oleh "ragu".
* Kecepatan hari ini 201 baris/detik (500 baris, 1 lensa, mode baris, termasuk pemanasan).
* **[ukuran sebelumnya]** Laya sekitar 317 teks/detik (1 lensa) dan 75 teks/detik (4 lensa), batch 32,
  GPU laptop; jenis_properti dari baris dict akurasi 92,6%.
* **[ukuran sebelumnya]** Lensa `label_cocok` (ya/tidak "apakah label ini benar") memberi AUC sekitar
  0,75 untuk label yang diacak; ambang 0,5 buruk, jadi urutkan kolom `label_cocok__nilai`.

### 6.4 Teks nyasar dan label diacak (emot_train, 10% label diacak = 352 baris) [diukur 20 Okt 2026]
"Nyasar" per label menandai 1717 teks; 304 dari 352 label salah ikut tertandai (recall 86,4%),
presisi 17,7% (dasar acak 10%). Ini penyaring kandidat, bukan hakim.

### 6.5 Konflik label [diukur 20 Okt 2026]
emot_train asli: 1 kelompok teks sama berlabel beda (2 baris: contoh "Terlalu naif jika berbicara
patah hati ..." berlabel fear dan sadness). Setelah menyisipkan 100 salinan berlabel beda: 101
kelompok ditemukan (100 + 1 bawaan). Deteksi bersifat pasti untuk teks yang sama persis atau sama
setelah huruf/tanda baca diabaikan, sehingga uji ini terutama memastikan tidak ada yang terlewat.

### 6.6 Nilai kategori tidak konsisten (kolom kota dari alamat, 10% baris diberi varian) [diukur 20 Okt 2026]
| Jenis varian | Disisipkan | Dikoreksi ke nilai benar | Nilai bersih salah digabung |
|---|---|---|---|
| huruf besar/spasi/tanda baca | 298 | 297 | 0 |
| salah ketik 1 huruf (hapus) | 298 | 219 (73,5%) | 0 |

Kelemahan terukur: salah ketik pada kata **5 huruf atau kurang** tidak tertangkap (Depok -> Deok,
Bogor -> Boor, Medan -> Mean tidak ditemukan), karena satu huruf hilang pada kata 5 huruf memberi
ratio 88,9 < 90. Kata 6 huruf ke atas (Malang, Bandung, Tangerang, Yogyakarta) tertangkap.

### 6.7 Uji hubungan dan pergeseran (kalibrasi pada data tanpa hubungan) [diukur 20 Okt 2026]
* Kolom kategori acak (4 nilai) vs label emot, 300 percobaan: p < 0,05 pada 6,7% (harapan ~5%).
* Kolom angka acak vs label, 300 percobaan: p < 0,05 pada 3,7%.
* Pergeseran label dengan tanggal diacak, 300 percobaan: p < 0,05 pada 4,7%.
* Kolom yang menyalin label pada separuh baris: Cramer V = 0,709, p mendekati 0.
* Hubungan nyata pada emot_train: `teks_jumlah_kata` vs label epsilon2 = 0,010 (p < 0,001) =
  efek kecil; `teks_panjang` epsilon2 = 0,002 (p = 0,013) = signifikan tetapi efek diabaikan;
  `id` epsilon2 = 0,000 (tidak signifikan).
* Data uji dengan kolom tanggal yang sengaja bergantung label: pergeseran awal vs akhir p < 0,001,
  Cramer V = 0,781, label paling berubah `anger` 47,7% menjadi 2,2%.

### 6.8 Peta makna dan pemisahan kelas (emot_train, 2000 titik) [diukur 20 Okt 2026]
Silhouette rata-rata 0,008 (label diacak penuh: -0,008): lima emosi pada dasarnya bertumpuk dalam
ruang makna. Per kelas: love 0,114 (paling terpisah), fear 0,013, anger 0,008, happy -0,017,
sadness -0,042; 88,5% teks sadness lebih mirip kelas lain. PCA 50 dimensi menyimpan 73% ragam;
t-SNE 2000 titik di CPU 4,3 detik.

### 6.9 Teks (emot_train) [diukur 20 Okt 2026]
Kosakata 16.926 kata dari 102.069 kata (29 kata per teks); 57,9% kata muncul hanya sekali. Rasio kata
alay rata-rata 12,9% (kamus 4.342 kata); 10,1% teks memuat stopword Inggris dan Indonesia
sekaligus. N-gram paling khas: happy "alhamdulillah" (7,6% vs 0,3%), fear "takut" (44,5% vs 1,1%),
love "sayang" (42,6% vs 3,0%), anger "username username" (14,8% vs 3,8%), sadness "sedih" (3,4% vs
0,6%). Waktu n-gram 3521 teks 0,45 detik.

## 7. Hal yang belum atau tidak diukur

* Kecepatan Laya 317/75 teks per detik, akurasi 92,6%, dan AUC `label_cocok` 0,75 tidak diukur ulang.
* Alasan untuk `alamat kabur` dan `deskripsi kosong` hanya muncul bila lensa Laya yang relevan sudah
  dijalankan; belum diukur berapa persen yang tertangkap lewat "Laya ragu".
* Peta 2D, n-gram, dan silhouette hanya diuji pada tweet emosi (emot_train) dan data properti; belum
  pada data lain atau 20 ribu baris dengan vektor.
* Pemilihan ambang (1%, 50%, 0,9, 90, 0,6) bersifat konvensi dan belum dioptimalkan.

## 8. Rujukan

* Arthur, D. & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. SODA.
* Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Erlbaum.
* Cramer, H. (1946). Mathematical Methods of Statistics. Princeton University Press.
* Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.
* Kaufman, L. & Rousseeuw, P. J. (1990). Finding Groups in Data. Wiley.
* Kruskal, W. H. & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. JASA 47.
* Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions and reversals.
  Soviet Physics Doklady 10.
* Little, R. J. A. & Rubin, D. B. (2019). Statistical Analysis with Missing Data (3rd ed.). Wiley.
* Northcutt, C. G., Athalye, A. & Mueller, J. (2021). Pervasive label errors in test sets
  destabilize machine learning benchmarks. NeurIPS Datasets and Benchmarks.
* Pearson, K. (1900). On the criterion that a given system of deviations ... (uji chi-square).
  Philosophical Magazine 50.
* Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks.
  EMNLP-IJCNLP. arXiv:1908.10084.
* Reimers, N. & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using
  knowledge distillation. EMNLP. arXiv:2004.09813.
* Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of
  cluster analysis. Journal of Computational and Applied Mathematics 20.
* Salsabila, N. A., Ardhito Winatmoko, Y., Septiandri, A. A. & Jamal, A. (2018). Colloquial
  Indonesian lexicon. IALP.
* Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
  Information Processing & Management 24.
* Spearman, C. (1904). The proof and measurement of association between two things. American
  Journal of Psychology 15.
* Sparck Jones, K. (1972). A statistical interpretation of term specificity. Journal of
  Documentation 28.
* Tomczak, M. & Tomczak, E. (2014). The need to report effect size estimates revisited. Trends in
  Sport Sciences 21(1).
* Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
* van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE. JMLR 9.
* Wang, W., Wei, F., Dong, L., Bao, H., Yang, N. & Zhou, M. (2020). MiniLM: Deep self-attention
  distillation for task-agnostic compression of pre-trained transformers. NeurIPS. arXiv:2002.10957.
* Wattenberg, M., Viegas, F. & Johnson, I. (2016). How to use t-SNE effectively. Distill.
* Perangkat lunak: scikit-learn (Pedregosa dkk. 2011), SciPy (Virtanen dkk. 2020), pandas,
  rapidfuzz, Streamlit, Altair.

## 9. Fitur → label dan Train vs test

File: `features.py` (hitungan, tanpa streamlit), `tab_fitur.py`, `tab_test.py` (tampilan,
masing-masing `show(ctx)`). Uji: `_uji/harness_b.py` + `_uji/test_b.py`.
Semua angka di bawah diukur di CPU, seed tetap (`SEED = 42`, `FLIP_SEED = 2024`).
Data: `emot_train.csv` (3521 baris, 5 emosi) dan `contoh/properti.csv` (3000 baris).

### 9.0. Aturan bersama

- Skor: label kategori → **F1 macro**, `StratifiedKFold(5)`; label angka → **R²** (dan MAE),
  `KFold(5)`. Dipilih otomatis (`task_of`): bukan angka, atau angka dengan ≤ 20 nilai = kategori.
- Baseline = **TF-IDF kata 1-2 gram + LogisticRegression** (saga, C = 3). Untuk label angka Ridge.
- Model: ada TF-IDF → model linear. Hanya kolom padat → **LightGBM** (200 pohon, 15 daun).
- Tidak ada kebocoran: TF-IDF, penskala, imputasi, dan target encoding berada **di dalam**
  `Pipeline` sklearn, jadi hanya di-fit pada lipatan latih.
- Nilai padat distandarkan lalu tiap kelompok dikali `1/√jumlah_kolom` agar normnya ≈ 1 seperti
  TF-IDF (supaya 384 kolom MiniLM tidak menenggelamkan kata).
- Keputusan kelompok: **bantu** bila selisih skor ≥ +0,01 **dan** lebih besar dari std antar
  lipatan; **merugikan** bila sebaliknya; selain itu **tidak beda**.
- Lensa Laya yang `with_label = True` (Laya melihat kolom label) **tidak** dipakai sebagai fitur
  (bocor). Mereka hanya muncul di cek fitur bocor, diberi catatan.
- Baris yang belum dibaca Laya / tidak punya vektor: tiap kelompok diuji pada barisnya sendiri dan
  baseline dihitung ulang **pada baris yang sama persis** (kolom `baris` di tabel).

### 9.1. Lab feature engineering (`score_group`, `assemble`)

Kelompok (17): TF-IDF kata (baseline), TF-IDF karakter 2-5 (`char_wb`), statistik gaya (panjang,
kata, rasio kapital, `!`, `?`, angka, URL, @mention, #hashtag, emoji, kata diulang, huruf diulang),
negasi (tidak, bukan, gak, nggak, ga, tak, jangan), rasio kata alay (kamus Salsabila,
`../../data_latihan/preprocessing/kamus_alay_salsabila.csv`), vektor MiniLM, lensa Laya (nilai +
one-hot), kelompok MiniLM (one-hot), log angka, binning kuantil 10 kotak, angka ÷ median
kategori, frequency encoding, target encoding (sklearn `TargetEncoder`, cross-fitting), penanda
kosong, regex (angka terbesar di teks + kata kunci seperti `/bulan`), panjang tiap kolom, bagian
tanggal.

Hasil pilihan disimpan ke `<folder>/fitur_terpilih.json` (nama kelompok, selisih, kolom, parameter).

Angka (emot_train, F1 macro, 5 lipatan, baseline 0,661 ± 0,013):

| kelompok | baris | selisih | keputusan |
|---|---|---|---|
| char TF-IDF 2-5 | 3521 | +0,021 | bantu (std 0,013) |
| gaya | 3521 | +0,002 | tidak beda |
| negasi | 3521 | -0,002 | tidak beda |
| alay | 3521 | -0,001 | tidak beda |
| regex | 3521 | +0,003 | tidak beda |
| panjang kolom | 3521 | +0,002 | tidak beda |
| MiniLM | 3000 | +0,005 | tidak beda |
| kelompok MiniLM (8 kluster) | 3000 | -0,032 | merugikan |
| lensa Laya | 300 | +0,035 (std 0,047) | tidak beda |
| gabungan gaya+negasi+alay+regex+panjang | 3521 | +0,000 | tidak beda |

Lab 9 kelompok pada emot ≈ 43 detik (CPU), 14 kelompok properti ≈ 50-100 detik.
Properti (jenis_asli, baseline 0,988): tidak ada kelompok yang membantu (judul sudah menentukan
jenis). Regresi luas_m2 (R² 0,562): tidak ada yang membantu.

### 9.2. Ablation Laya (`ablation_table`)

Pada baris yang dibaca Laya: tebakan kelas terbanyak, TF-IDF saja, TF-IDF + lensa, lensa saja.
emot_train, 300 baris (lensa emosi + kejelasan, Laya di CPU):

| model | F1 macro | std |
|---|---|---|
| tebakan kelas terbanyak | 0,088 | 0,000 |
| TF-IDF saja | 0,461 | 0,058 |
| TF-IDF + lensa Laya | 0,497 | 0,047 |
| lensa Laya saja | 0,392 | 0,095 |

Selisih +0,036 lebih kecil dari derau antar lipatan (0,047), jadi jawabannya "belum terbukti
menambah informasi". Lensa saja jauh di atas tebakan (0,392 vs 0,088): Laya memang mengenali
emosi, tetapi kata-kata sudah memuat informasi itu. Properti: +0,000 (0,987 → 0,987).
Kesimpulan ini bergantung pada 300 baris; tambah baris Laya lalu ulangi.

### 9.3. Importance

**3a Kata penentu** — koefisien LogisticRegression pada TF-IDF kata (15 teratas positif dan
negatif per kelas). Contoh emot: anger = username, kok, anda, marah, sebel; fear = takut, ngeri,
serem, trauma; happy = alhamdulillah, bahagia, wkwk; love = sayang, cinta, jatuh; sadness = sedih,
sakit, rindu. Batasan: korelasi antar kata, bukan sebab-akibat.

**3b Permutation importance** — Breiman (2001). LightGBM dilatih 75%, di 25% sisanya satu kolom
diacak (5 ulangan), turunnya F1 macro/R² = pentingnya kolom. emot (gaya+negasi+alay+regex+panjang
kolom, 3500 baris): `rasio_kapital` 0,054, `rasio_alay` 0,044, `panjang` 0,017; 7 dari 21 fitur
tidak berpengaruh (≈ 3 detik).

**3c Mutual information** — `mutual_info_classif/regression` (Kraskov dkk. 2004), menangkap
hubungan non-linear; emot: `rasio_kapital` 0,044, `jumlah_alay` 0,040.

### 9.4. Pencari label salah (`label_issues`)

Confident learning sederhana (Northcutt, Jiang, Chuang 2021):
1. Peluang tiap kelas out-of-fold (`cross_val_predict`, `predict_proba`, 5 lipatan).
2. Ambang kelas *k* = rata-rata peluang kelas *k* pada baris berlabel *k*.
3. Ditandai bila peluang label yang diberikan < ambangnya **dan** ada kelas lain dengan
   peluang ≥ ambang kelas itu. Saran label = kelas lain dengan peluang terbesar.
4. Urut menurut (peluang saran - peluang label). Kolom `alasan`, contoh:
   `model memberi label 'anger' 1%, 'fear' 98% (ambang 'fear' 38%)`.

**Diukur**: pada emot_train, 10% label (352 dari 3521) diacak ke kelas lain (`FLIP_SEED = 2024`),
lalu dihitung berapa yang ditandai:

| metode | ditandai | presisi | recall |
|---|---|---|---|
| tebakan acak (dasar) | sama | 10% | sebanding |
| **pencari ini, TF-IDF kata** (2 detik) | 955 | **26,8%** | **72,7%** |
| pencari ini, TF-IDF kata + karakter (16 detik) | 895 | 28,5% | 72,4% |
| pencari ini, TF-IDF kata + MiniLM (3000 baris) | 765 | 27,2% | 69,3% |
| "nyasar" MiniLM, dihitung ulang (3000 baris) | 1454 | 18,6% | 87,4% |
| "nyasar" MiniLM, uji sebelumnya | - | 17,7% | 86% |

Presisi pada K teratas (K = 352 = jumlah yang diacak): **47,2%** (acak: 10%). Jadi pencari ini
menandai lebih sedikit (presisi 1,5× lebih tinggi daripada "nyasar" dan 2,7× dasar) tetapi
kehilangan sekitar 14 poin recall dibanding "nyasar". Cocok untuk daftar periksa manual: urutan
teratas jauh lebih "bersih". Pada label asli emot (tanpa diacak) 17,3% baris ditandai; sebagian
adalah label yang memang ambigu, bukan kesalahan. Properti: 29 baris (1,0%) ditandai; bila 10%
diacak, recall 95% dan presisi 90% (jenis properti mudah dikenali dari judul).

### 9.5. Ketidakseimbangan kelas

Jumlah per kelas, rasio terbesar/terkecil, saran: rasio < 1,5 seimbang; 1,5-3 agak timpang (F1
macro, stratify, coba `class_weight='balanced'`); > 3 timpang (wajib). Efek diukur (CV sama):
emot rasio 1,73 (anger 881 : love 509), F1 macro 0,666 → 0,667 dengan `class_weight='balanced'`
(+0,001, praktis tidak ada efek pada rasio sekecil itu).

### 9.6. Cek fitur bocor (`single_feature_scores`)

Tiap kolom data asli (termasuk `id`), nomor baris, dan nilai lensa Laya dipakai **sendirian**
(angka: LightGBM, teks: TF-IDF + LogisticRegression, CV 5 lipatan, ≤ 3000 baris). Skor ≥ 0,95 =
"CURIGA BOCOR", ≥ 0,8 = "sangat kuat, periksa". emot: teks 0,658, `id` 0,189, nomor baris 0,184,
lensa Laya 0,22-0,27 (tidak ada yang bocor, 6-7 detik). Properti: `judul` dan `deskripsi` ≥ 0,95
(sah dalam data sintetis ini, judul memang menyebut jenis, tetapi harus dipastikan ada saat test).

### 9.7. Train vs test (`tab_test.py`)

- **Sebaran**: uji Kolmogorov-Smirnov dua sampel (D, p) untuk panjang teks, jumlah kata, dan
  kolom angka (Massey 1951). Kolom kategori: chi-square (Pearson), **Cramér's V** (ukuran efek,
  Cramér 1946), dan **PSI** (Σ (p_test - p_train)·ln(p_test/p_train); < 0,1 stabil, 0,1-0,25
  bergeser, > 0,25 beda besar; Siddiqi 2006). emot: panjang median 179 vs 172,5, D = 0,059,
  p = 0,126 → mirip.
- **Kosakata**: kata test yang tidak ada di train (OOV). emot: 10,6% kata test baru (1327 dari
  12506), 86,4% baris test punya minimal satu kata baru, 1220 jenis kata baru.
- **Kebocoran**: teks test sama persis / setelah huruf kecil + tanpa tanda baca dengan train,
  beserta label train. emot: 4 dari 440 teks test (0,9%) sudah ada di train.
- **Adversarial validation**: model membedakan train (0) dari test (1), AUC 5 lipatan.
  emot: **AUC 0,513 ± 0,029** (TF-IDF teks), 0,498 ± 0,027 (statistik gaya), kontrol train vs
  train 0,481 ± 0,026 → test mirip train (≈ 8 detik). Kata khas test/train ditampilkan dari
  koefisien. AUC < 0,5 berarti baris kembar antara train dan test (lihat Kebocoran). Ada AUC dari
  kolom angka saja (LightGBM): pada `properti_test.csv` (luas_m2 ×1,5) AUC 0,73 padahal AUC teks
  0,50, jadi angka perlu dicek terpisah.
- **Laya pada sampel test** (opsional, tombol, maks 300 baris, default CPU, hasil di
  `<folder>/test/`): sebaran jawaban lensa train vs test dibandingkan dengan PSI / chi-square /
  Cramér's V. Sampel kecil diberi peringatan.

### 9.8 Rujukan bagian 9

- Breiman, L. (2001). Random Forests. *Machine Learning* 45, 5-32 (permutation importance).
- Northcutt, C., Jiang, L., Chuang, I. (2021). Confident Learning: Estimating Uncertainty in
  Dataset Labels. *JAIR* 70, 1373-1411.
- Adversarial validation: praktik kompetisi data (Zając, 2016, "Adversarial validation", FastML);
  idenya: bila classifier bisa membedakan train dan test, distribusinya berbeda.
- Massey, F. (1951). The Kolmogorov-Smirnov test for goodness of fit. *JASA* 46, 68-78.
- Cramér, H. (1946). *Mathematical Methods of Statistics* (Cramér's V). Pearson (1900), chi-square.
- Kraskov, A., Stögbauer, H., Grassberger, P. (2004). Estimating mutual information.
  *Physical Review E* 69 (dipakai `mutual_info_*` sklearn).
- Cavnar, W., Trenkle, J. (1994). N-gram-based text categorization (TF-IDF karakter n-gram).
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes.
  *SIGKDD Explorations* 3(1) (target encoding; sklearn `TargetEncoder` memakai cross-fitting
  out-of-fold).
- Siddiqi, N. (2006). *Credit Risk Scorecards* (PSI).
- Ke, G. dkk. (2017). LightGBM. *NeurIPS*.
- Kohavi, R. (1995). A study of cross-validation and bootstrap (stratified k-fold).

## 10. Kode EDA dan laporan

- **Kode EDA** (`codegen.py`, tab *Kode EDA*): sel notebook disusun dari pilihan sidebar (kolom,
  format angka Indonesia, skala log, rasio), lensa Laya yang sudah dijalankan (lensa
  `with_label` dilewati), dan fitur pilihan (gaya, negasi, panjang, kosong, log, angka ÷ median
  kategori, frekuensi, tanggal; statistik diambil dari train saja). Baris dibuang lewat **aturan
  pandas** (duplikat, teks kosong, angka tidak terbaca, negatif/nol, outlier IQR per kategori):
  tiap aturan berupa loop per kolom yang di-`|` ke satu penanda `drop`, dicetak jumlahnya, lalu
  dibuang sekali. Hanya keputusan manual tab Telusuri yang ditulis per baris (`MANUAL_DROP`,
  `NEW_LABELS`, bersama catatannya). Hasilnya `.py` (sel `# %%`) dan `.ipynb`.
- **Train siap pakai** (tab *Ekspor*): data asli + catatan diterapkan + fitur pilihan dari
  `features.BUILDERS` → `train_siap.csv`.
  `_uji/test_codegen.py` membuat lalu **menjalankan** kode untuk dua skenario (properti; emot +
  test) dan notebooknya lewat nbconvert: lolos, 0 baris > 100 karakter [diukur 20 Okt 2026].
- Kelompok KMeans (tab *Temukan kategori*) tidak ditulis ulang karena bisa berubah tiap jalan.
- **Temuan utama** (tab *Temuan utama*): `ctx["findings"]` (lihat 2b) diurutkan menurut
  `penting`, temuan sama dibuang, lalu disusun jadi `laporan_eda.md`.
