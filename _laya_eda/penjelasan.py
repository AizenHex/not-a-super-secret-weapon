"""Penjelasan teknis untuk juri: tiap bagian app punya kotak "Cara kerja & kenapa".
Isi tiap bagian: apa, cara (rumus/langkah), kenapa dipilih, batasan, kalimat untuk juri.
Angka ukuran Laya/MiniLM berasal dari pengujian di METODE.md."""
import streamlit as st

TITLES = {"apa": "Apa", "cara": "Cara (rumus / langkah)", "kenapa": "Kenapa dipilih",
          "batasan": "Batasan", "juri": "Kalimat untuk juri"}

EXPLANATIONS = {
    "data_stats": {
        "apa": "Ringkasan bentuk data dan kebersihan teks: duplikat, teks terlalu pendek, panjang.",
        "cara": ("Duplikat persis = teks yang sama persis muncul lebih dari sekali. Duplikat mirip "
                 "= sama setelah huruf dibuat kecil dan semua tanda baca/spasi dibuang "
                 "(`[^a-z0-9]`). Teks pendek = 3 kata atau kurang. Panjang = jumlah huruf."),
        "kenapa": ("Duplikat yang jatuh di data latih dan data uji sekaligus membuat skor tampak "
                   "lebih bagus daripada kenyataan. Teks sangat pendek hampir tidak punya "
                   "informasi untuk ditebak labelnya."),
        "batasan": ("Duplikat mirip tidak menangkap parafrase (kata beda, makna sama): itu "
                    "dicari terpisah dengan MiniLM di tab Outlier. Di mode baris seluruh kolom "
                    "digabung menjadi satu teks."),
        "juri": ("Sebelum memodelkan, kami mengukur duplikat dan teks sangat pendek karena "
                 "keduanya menggelembungkan atau melemahkan skor."),
    },
    "profil_kolom": {
        "apa": ("Profil tiap kolom: tipe, persen kosong, nilai unik, nilai terbanyak, kemiringan "
                "sebaran (skew), kategori langka, dan tipe campur (angka bercampur teks)."),
        "cara": ("Skew = momen ketiga terstandar (pandas `.skew()`); |skew| > 1 dianggap miring, "
                 "dan kalau semua nilai > 0 serta miring ke kanan disarankan skala log. Kategori "
                 "langka = kategori di bawah 1% baris (hanya untuk kolom kategori: <= 50 nilai "
                 "unik, atau kolom angka <= 10 nilai unik). Tipe campur = kolom teks yang >= 50% "
                 "isinya terbaca sebagai angka tetapi tidak 100%."),
        "kenapa": ("Hampir semua kejutan di model berasal dari kolom yang 'kelihatannya angka "
                   "tapi bertipe teks', sebaran sangat miring, atau kategori yang hanya muncul "
                   "sekali. Profil singkat mencegahnya sebelum modelling."),
        "batasan": ("Ambang 1, 1%, dan 50% adalah konvensi, bukan hukum. Skew sensitif terhadap "
                    "nilai ekstrem. Kolom ID dan teks panjang tidak dihitung sebagai kategori."),
        "juri": ("Kami memprofilkan setiap kolom lebih dulu supaya keputusan seperti skala log "
                 "atau penggabungan kategori langka memakai angka, bukan perasaan."),
    },
    "hubungan_label": {
        "apa": ("Seberapa kuat tiap kolom berhubungan dengan label, dengan p-value DAN ukuran "
                "efek, plus kalimat tafsir."),
        "cara": ("Kolom kategori vs label kelas: uji chi-square independensi, ukuran efek "
                 "Cramer V = sqrt(chi2 / (n * (min(baris, kolom) - 1))). Kolom angka vs label "
                 "kelas: uji Kruskal-Wallis H, ukuran efek epsilon2 = (H - k + 1) / (n - k) "
                 "(Tomczak & Tomczak 2014). Label berupa angka (> 10 nilai unik): korelasi "
                 "Spearman rho untuk kolom angka, Kruskal-Wallis untuk kolom kategori. Ambang efek "
                 "(Cohen): V dan rho 0,1 kecil / 0,3 sedang / 0,5 besar; epsilon2 0,01 / 0,06 / "
                 "0,14. Signifikan bila p < 0,05."),
        "kenapa": ("Pada data besar p-value hampir selalu kecil, jadi p saja menyesatkan; ukuran "
                   "efek menjawab 'seberapa kuat'. Kruskal-Wallis dan Spearman berbasis peringkat "
                   "sehingga tahan sebaran miring dan outlier, yang umum pada harga dan panjang "
                   "teks."),
        "batasan": ("Satu kolom diuji sendirian: interaksi antar kolom tidak tertangkap. "
                    "Chi-square butuh frekuensi harapan yang cukup; V cenderung naik pada tabel "
                    "kecil. Hubungan kuat bukan bukti sebab-akibat, dan hubungan yang terlalu "
                    "kuat bisa berarti kolom itu bocoran dari label."),
        "juri": ("Kami menilai hubungan kolom dengan label memakai uji signifikansi dan ukuran "
                 "efek (Cramer V / epsilon2), sehingga kami bisa berkata 'signifikan tetapi "
                 "lemah' atau 'kuat', bukan hanya 'p kecil'."),
    },
    "korelasi": {
        "apa": "Korelasi antar kolom angka dan daftar pasangan yang hampir sama (redundan).",
        "cara": ("Korelasi Spearman = korelasi Pearson pada peringkat nilai. Pasangan dengan "
                 "|rho| > 0,9 dicatat sebagai redundan."),
        "kenapa": ("Spearman tahan outlier dan menangkap hubungan menaik/menurun yang tidak "
                   "lurus. Kolom redundan menambah beban tanpa menambah informasi dan membuat "
                   "koefisien model tidak stabil."),
        "batasan": ("Hanya mendeteksi hubungan monoton. Korelasi tinggi bisa muncul karena dua "
                    "kolom diturunkan dari kolom ketiga. Korelasi bukan sebab-akibat."),
        "juri": ("Kolom angka yang urutannya hampir identik (|rho| > 0,9) kami tandai supaya "
                 "salah satunya dibuang dan model tidak menghitung informasi yang sama dua kali."),
    },
    "kosong": {
        "apa": ("Kolom mana yang punya sel kosong, berapa persen, dan apakah kosongnya "
                "berkaitan dengan label."),
        "cara": ("Penanda kosong (benar/salah) dibuat per kolom, lalu diuji dengan chi-square "
                 "terhadap label dan diukur dengan Cramer V (ambang sama dengan hubungan label). "
                 "Dua kolom yang penanda kosongnya berkorelasi > 0,95 dianggap kosong bersamaan."),
        "kenapa": ("Kosong yang tidak acak adalah informasi: misalnya penulis kelas tertentu "
                   "lebih sering tidak mengisi kolom. Mengisinya mentah menghapus sinyal itu."),
        "batasan": ("Kami hanya menguji kaitan dengan label, bukan membedakan secara formal "
                    "jenis kosong (acak, bergantung kolom lain, bergantung nilainya sendiri). "
                    "Kolom yang hampir tidak pernah kosong membuat ujinya lemah."),
        "juri": ("Kami memeriksa apakah data kosong acak atau berkaitan dengan label, karena "
                 "kosong yang berkaitan dengan label perlu ditangani sebagai fitur, bukan "
                 "ditambal."),
    },
    "waktu": {
        "apa": ("Jumlah baris per bulan, sebaran label per bulan, dan tanda pergeseran label "
                "dari awal ke akhir."),
        "cara": ("Tanggal dibaca pandas (`to_datetime`, format campur boleh; yang gagal dihitung "
                 "terpisah). Baris diurutkan menurut waktu lalu dibagi dua sama banyak; sebaran "
                 "label separuh awal dan separuh akhir dibandingkan dengan chi-square + Cramer V."),
        "kenapa": ("Jika sebaran label bergeser menurut waktu, membagi data latih dan uji secara "
                   "acak memberi skor terlalu optimis; pembagian menurut waktu lebih jujur."),
        "batasan": ("Pembagian dua separuh itu kasar. Pola musiman bisa terbaca sebagai "
                    "pergeseran. Tanggal yang tidak terbaca dibuang dari analisis ini."),
        "juri": ("Kami menguji apakah label bergeser menurut waktu, sehingga strategi membagi "
                 "data latih dan uji dipilih dengan alasan terukur."),
    },
    "cek_angka": {
        "apa": ("Pandas mengecek kolom angka: yang tidak terbaca, kosong, negatif/nol, dan "
                "outlier."),
        "cara": ("Teks diubah jadi angka: 'Rp' dan spasi dibuang, titik = ribuan, koma = "
                 "desimal (bisa dimatikan), lalu angka di awal teks dibaca. Outlier IQR Tukey: "
                 "Q1 = persentil 25, Q3 = persentil 75, IQR = Q3 - Q1; outlier bila nilai < Q1 - "
                 "1,5 IQR atau > Q3 + 1,5 IQR. Pada skala log semua dihitung pada log10(nilai) "
                 "(hanya nilai > 0)."),
        "kenapa": ("Laya lemah soal angka, sedangkan pandas pasti. Skala log dipakai karena "
                   "harga dan luas berekor panjang: tanpa log, harga mahal yang wajar ikut "
                   "ditandai dan batas bawah bisa negatif."),
        "batasan": ("IQR seluruh data tidak tahu kategori: Rp 12 juta wajar untuk kost tetapi "
                    "mustahil untuk rumah. Nilai <= 0 tidak bisa di-log sehingga dilaporkan "
                    "terpisah. Sekitar 0,7% data berdistribusi normal pun akan terhitung "
                    "outlier."),
        "juri": ("Angka diperiksa dengan IQR Tukey pada skala log; kelemahannya tanpa kategori "
                 "adalah banyak baris normal ikut tertandai, makanya kami membandingkannya per "
                 "kategori di tab Outlier."),
    },
    "rasio": {
        "apa": "Kolom baru hasil pembagian dua kolom angka, misalnya harga per m2.",
        "cara": "rasio = pembilang / penyebut, hanya bila penyebut > 0; selain itu kosong.",
        "kenapa": ("Kejanggalan sering baru terlihat pada rasio: harga 12 juta wajar dan luas "
                   "500 m2 wajar, tetapi harga per m2 = Rp 24 ribu tidak masuk akal."),
        "batasan": ("Rasio menggandakan galat kedua kolom. Penyebut nol atau negatif membuat "
                    "baris itu kosong pada kolom rasio."),
        "juri": "Kami membuat kolom rasio karena sebagian kejanggalan hanya tampak pada rasionya.",
    },
    "kelompok": {
        "apa": ("Mengelompokkan teks yang maknanya mirip supaya kita tahu kategori apa yang "
                "sebenarnya ada, lalu kelompok bisa dijadikan lensa Laya."),
        "cara": ("1) MiniLM (Sentence-BERT, model paraphrase-multilingual-MiniLM-L12-v2; Reimers "
                 "& Gurevych 2019) mengubah tiap teks jadi vektor 384 dimensi, panjang 1. 2) "
                 "KMeans (n_init=4, seed 42) mengelompokkan vektor. 3) Kata kunci kelompok = kata "
                 "dengan selisih terbesar antara rata-rata TF-IDF di kelompok dan rata-rata "
                 "semua teks."),
        "kenapa": ("Laya hanya bisa memilih dari daftar kategori yang kita tulis; kelompok "
                   "memberi daftar yang berasal dari data. MiniLM multibahasa paham bahasa "
                   "Indonesia dan cepat di laptop."),
        "batasan": ("KMeans butuh jumlah kelompok yang kita tentukan dan cenderung membuat "
                    "kelompok bulat dengan ukuran sama. Kata kunci bisa generik. Kelompok adalah "
                    "usulan, bukan kebenaran: beri nama sendiri."),
        "juri": ("Kategori tidak kami tebak: MiniLM mengelompokkan teks berdasarkan makna, lalu "
                 "kami memberi nama dan memeriksanya dengan Laya."),
    },
    "lensa": {
        "apa": ("Satu lensa = satu pertanyaan dengan pilihan jawaban yang dijawab Laya untuk "
                "setiap baris."),
        "cara": ("Tiga jenis: pilih satu (choice), ya/tidak (noul), skala (score). Laya membaca "
                 "teks dan pertanyaan sekali jalan lalu mengeluarkan peluang tiap pilihan. "
                 "'Yakin' = answer_confidence terkalibrasi. Kami juga menyimpan pilihan KEDUA dan "
                 "peluangnya (untuk ya/tidak: peluang ya) sehingga alasan 'ragu' bisa disebutkan "
                 "dengan angka."),
        "kenapa": ("Laya cepat (sekitar 317 teks per detik untuk satu lensa, 75 teks per detik "
                   "untuk empat lensa, batch 32, GPU laptop) dan memberi ukuran ketidakpastian "
                   "per jawaban. Jenis properti dari baris data (mode baris) benar 92,6%."),
        "batasan": ("Laya lemah soal angka (pakai tab Cek angka). Jawaban hanya sebaik "
                    "deskripsi pilihan. Lensa label_cocok untuk mencari label salah kurang tepat "
                    "bila dibaca sebagai ya/tidak: dalam uji dengan label diacak AUC-nya sekitar "
                    "0,75 sedangkan ambang 0,5 buruk, jadi urutkan kolom nilainya."),
        "juri": ("Laya menjawab pertanyaan bermakna dengan keyakinan terukur: jenis properti "
                 "dari baris data benar 92,6% pada pengujian kami."),
    },
    "jalankan_laya": {
        "apa": "Menjalankan semua lensa pada sejumlah baris, hasil disimpan ke file CSV.",
        "cara": ("Urutan baris diacak dengan seed tetap 42 sehingga menambah jumlah baris hanya "
                 "menghitung baris baru. Teks dikirim per 32 (batch) ke GPU; hasil ditulis ke "
                 "file tiap 256 baris sehingga aman dihentikan. Perkiraan waktu = jawaban / 300 "
                 "per detik."),
        "kenapa": "Hasil tidak hilang saat proses dihentikan, dan laptop tidak dipakai berlebihan.",
        "batasan": ("Perkiraan 300 jawaban per detik berlaku untuk teks sekitar 180 huruf; teks "
                    "panjang atau banyak lensa lebih lambat (75 teks per detik untuk empat "
                    "lensa)."),
        "juri": ("Proses bisa dihentikan dan dilanjutkan tanpa menghitung ulang, sehingga "
                 "hasil Laya yang mahal selalu tersimpan."),
    },
    "ragu_laya": {
        "apa": "Baris yang membuat Laya ragu pada minimal satu lensa.",
        "cara": ("Ragu bila 'yakin' (answer_confidence) < batas di panel samping (bawaan 0,6). "
                 "Alasannya menyebut pilihan pertama, pilihan kedua, dan selisih peluangnya; "
                 "selisih di bawah 10 poin disebut 'hampir seri'."),
        "kenapa": ("Baris ambigu paling mungkin salah label, salah kategori, atau memang "
                   "membingungkan; memeriksanya dulu paling efisien."),
        "batasan": ("Batas 0,6 adalah pilihan kita, ubah dengan slider. Keyakinan Laya bukan "
                    "akurasi: Laya bisa yakin dan salah."),
        "juri": ("Kami memakai ketidakyakinan Laya sebagai penunjuk baris yang perlu dicek "
                 "manusia, lengkap dengan pilihan kedua dan peluangnya."),
    },
    "ringkasan_kategori": {
        "apa": ("Satu baris per kategori: jumlah, persen, panjang median, label terbanyak, dan "
                "kata khas."),
        "cara": ("Pengelompokan pandas (groupby). Kata khas = rata-rata TF-IDF kategori dikurangi "
                 "rata-rata semua teks, diambil 8 kata teratas."),
        "kenapa": "Memberi bahan insight untuk slide: seperti apa tiap kategori.",
        "batasan": "Kata khas bisa kata umum bila kategori sangat besar.",
        "juri": "Tiap kategori diringkas dengan angka dan kata khasnya, bukan hanya namanya.",
    },
    "tabel_silang": {
        "apa": "Tabel dua dimensi, misalnya label asli x lensa emosi.",
        "cara": "pandas crosstab; isi sel boleh jumlah, persen per baris, atau persen per kolom.",
        "kenapa": ("Sel yang tidak sejalan (label 'senang' tetapi Laya membaca 'marah') "
                   "adalah kandidat salah label."),
        "batasan": ("Ketidaksejalanan bisa berarti label salah atau Laya salah; periksa "
                    "contohnya dengan mata."),
        "juri": ("Kami menyilangkan label asli dengan pembacaan Laya untuk menemukan kandidat "
                 "salah label."),
    },
    "peta_2d": {
        "apa": ("Gambar 2 dimensi dari makna teks; titik berdekatan = maknanya mirip. Angka "
                "silhouette menunjukkan seberapa terpisah kelas-kelasnya."),
        "cara": ("Sampel acak (maks 5000) vektor MiniLM -> PCA ke 50 dimensi -> t-SNE ke 2 "
                 "dimensi (CPU, perplexity = min(30, n/4), init PCA). Silhouette dihitung pada "
                 "VEKTOR ASLI 384 dimensi dengan jarak kosinus: s = (b - a) / max(a, b), a = jarak "
                 "rata-rata ke kelas sendiri, b = jarak rata-rata ke kelas terdekat lain. Juga "
                 "dicari kelas terdekat menurut kemiripan pusat kelas."),
        "kenapa": ("t-SNE mempertahankan tetangga dekat sehingga gugus makna terlihat; "
                   "silhouette memberi angka untuk apa yang dilihat mata, supaya klaim 'kelas "
                   "bertumpuk' bukan sekadar kesan."),
        "batasan": ("Jarak antar gugus dan ukuran gugus di t-SNE tidak bermakna; hanya "
                    "kedekatan lokal. Gambar berubah bila seed berubah. Silhouette pada teks "
                    "hampir selalu rendah (di bawah 0,1 pun biasa), jadi bandingkan antar kelas, "
                    "bukan dengan angka mutlak."),
        "juri": ("Peta makna menunjukkan kelas mana yang saling bertumpuk, dan silhouette per "
                 "kelas menaruh angka pada kesan itu."),
    },
    "outlier_angka": {
        "apa": ("Angka yang janggal DI DALAM kategorinya, dengan alasan berupa kalimat dan "
                "angka bukti."),
        "cara": ("IQR Tukey dihitung per kategori (pada log10 bila skala log): batas bawah = "
                 "Q1 - 1,5 IQR, batas atas = Q3 + 1,5 IQR, lalu dibalik ke satuan asli (10^batas) "
                 "untuk ditampilkan. kali_median = nilai / median kategori. Status: janggal "
                 "hanya di kategorinya, di keduanya, hanya di keseluruhan, atau normal."),
        "kenapa": ("Rp 12 juta wajar untuk kost tetapi mustahil untuk rumah. Membandingkan "
                   "dengan baris sejenis menangkap yang tidak tertangkap pembanding global. "
                   "Pada data properti demo, cara ini menangkap 54 dari 56 harga mustahil "
                   "tanpa satu baris normal pun ikut, sedangkan tanpa kategori 529 baris "
                   "normal ikut tertandai."),
        "batasan": ("Kategori kecil (kurang dari sekitar 10 baris) membuat batas tidak stabil. "
                    "Bila kategorinya sendiri keliru (lensa salah), perbandingannya ikut "
                    "keliru. Pembanding kategori dipilih di panel samping. Baris yang janggal "
                    "hanya di seluruh data (wajar untuk kategorinya) tetap tampil di tabel, "
                    "tetapi tidak dijadikan alasan di kolom `alasan`."),
        "juri": ("Kami membandingkan tiap angka dengan baris sejenis, bukan seluruh data: "
                 "54 dari 56 harga mustahil tertangkap tanpa menyeret baris normal."),
    },
    "outlier_teks": {
        "apa": ("Teks yang maknanya jauh dari kategorinya, atau lebih dekat ke kategori lain "
                "('nyasar')."),
        "cara": ("Pusat kategori = rata-rata vektor MiniLM anggotanya, dinormalkan. Jarak = 1 - "
                 "kosinus ke pusat. 'Jauh' = jarak di atas Q3 + 1,5 IQR jarak kategori itu dan "
                 "di atas median. 'Nyasar' = pusat terdekat bukan pusat kategorinya sendiri."),
        "kenapa": ("Teks berlabel salah cenderung dekat ke kategori yang benar. Pada tweet "
                   "emosi dengan 10% label diacak, 'nyasar' per label menangkap 86% label salah "
                   "(recall), dengan presisi 17,7% (dasar acak 10%)."),
        "batasan": ("Presisi rendah: hasilnya daftar kandidat untuk dicek manusia, bukan vonis. "
                    "Pusat ikut tercemar label salah. Kategori yang memang tumpang tindih "
                    "menghasilkan banyak 'nyasar'."),
        "juri": ("Teks yang lebih dekat ke kelas lain menangkap 86% label yang sengaja diacak, "
                 "dengan presisi 17,7% dibanding dasar 10%, jadi ini penyaring, bukan hakim."),
    },
    "makna_sama": {
        "apa": "Pasangan teks yang kata-katanya beda tetapi maknanya hampir sama.",
        "cara": ("Kemiripan kosinus antar vektor (perkalian titik karena panjang 1) >= batas. "
                 "Duplikat persis dibuang lebih dulu; maksimal 3000 pasangan."),
        "kenapa": ("Parafrase lolos dari cek duplikat persis tetapi tetap menyebabkan "
                   "kebocoran antara data latih dan uji."),
        "batasan": ("Kemiripan tinggi tidak berarti kebenaran sama: 'suka' dan 'tidak suka' "
                    "bisa sangat mirip. Dihitung per blok 1000 baris."),
        "juri": "Kami mencari parafrase dengan kemiripan makna, bukan hanya teks identik.",
    },
    "kartu_baris": {
        "apa": "Semua alasan untuk satu baris dikumpulkan: angka, teks, Laya, dan duplikat.",
        "cara": ("Setiap pemeriksaan menambahkan kalimat berisi angka ke daftar baris itu: "
                 "(1) angka tak terbaca / negatif / nol / janggal di dalam kategorinya (di luar "
                 "batas wajar seluruh data bila baris belum punya kategori); (2) teks jauh atau "
                 "nyasar; (3) Laya ragu; (4) duplikat, hampir duplikat, dan konflik label. "
                 "Kolom `alasan` menggabungkan semuanya dengan tanda |."),
        "kenapa": ("Satu tempat untuk memutuskan apakah baris itu dibuang, diperbaiki, atau "
                   "dipertahankan."),
        "batasan": ("Alasan dihitung terhadap SATU kategori pembanding (panel samping). Ganti "
                    "pembanding bila ingin sudut pandang lain."),
        "juri": ("Setiap baris yang kami tandai punya alasan tertulis berisi angka, jadi "
                 "keputusan membuang atau memperbaiki data bisa dipertanggungjawabkan."),
    },
    "telusuri": {
        "apa": "Menyaring dan mencari baris, lalu menulis keputusan dan catatan manual.",
        "cara": ("Saring menurut kategori, cari kata, hanya yang Laya ragu atau punya alasan. "
                 "Keputusan (simpan / buang / ganti_label / periksa_lagi) disimpan di "
                 "catatan.csv."),
        "kenapa": "Keputusan manusia atas daftar kandidat dari mesin itulah hasil kerja EDA.",
        "batasan": "Hanya 500 baris pertama hasil saringan yang tampil; persempit dengan saringan.",
        "juri": "Mesin mengusulkan, manusia memutuskan, dan semua keputusan tercatat.",
    },
    "ekspor": {
        "apa": ("Satu file CSV berisi data asli, jawaban Laya, kelompok, cek teks, angka, "
                "alasan, dan catatan."),
        "cara": ("Semua tabel digabung menurut nomor baris (`row`). Notebook EDA program cukup "
                 "membaca file ini dan membuang baris berkeputusan 'buang'."),
        "kenapa": "Hasil EDA manual bisa dipakai langsung oleh notebook modelling.",
        "batasan": "Berisi hanya baris yang sudah dihitung; lensa yang belum jalan berisi kosong.",
        "juri": "Seluruh temuan dan alasannya diekspor ke satu file yang dipakai ulang oleh model.",
    },
    "konsistensi": {
        "apa": ("Nilai kolom kategori / teks pendek yang sebenarnya sama tetapi ditulis berbeda, "
                "dengan saran peta koreksi."),
        "cara": ("1) Kunci sederhana: huruf kecil dan hanya a-z, 0-9. Nilai berkunci sama = sama "
                 "kecuali huruf besar/spasi/tanda baca. 2) Antar kunci: kemiripan rapidfuzz "
                 "`fuzz.ratio` (0-100, berbasis jarak Indel) >= 90 digabung ke nilai yang lebih "
                 "sering muncul; pasangan yang hanya beda angka diabaikan (kelompok_1 vs "
                 "kelompok_2)."),
        "kenapa": ("'Rumah', 'rumah ', dan 'RUMAH' dianggap tiga kategori oleh model dan "
                   "mengacaukan hitungan, tabel silang, serta perbandingan antar kategori."),
        "batasan": ("Ambang 90 ketat tetapi tetap bisa menggabungkan dua nilai berbeda yang "
                    "kebetulan mirip; periksa tabel sebelum memakai peta koreksi. Hanya kolom "
                    "teks pendek dengan paling banyak 1500 nilai unik."),
        "juri": ("Kami mendeteksi kategori yang sama tetapi tertulis berbeda dengan aturan "
                 "sederhana dan kemiripan huruf, lalu memberi peta koreksi yang bisa diperiksa."),
    },
    "konflik_label": {
        "apa": "Teks yang sama (atau hampir sama) tetapi diberi label berbeda.",
        "cara": ("Teks dikelompokkan menurut kunci sederhana (huruf kecil, hanya a-z dan 0-9). "
                 "Kelompok yang punya dua label atau lebih adalah konflik. Dilaporkan apakah "
                 "teksnya sama persis atau hanya sama setelah penyederhanaan."),
        "kenapa": ("Dua label untuk teks yang sama berarti paling banyak satu yang benar; model "
                   "tidak mungkin belajar keduanya, dan skor evaluasi ikut terpotong."),
        "batasan": ("Hanya menangkap teks yang identik setelah penyederhanaan; parafrase "
                    "berlabel beda ditangkap tab Outlier. Dua label bisa sama-sama benar bila "
                    "tugasnya multi-label."),
        "juri": ("Kami menemukan teks identik yang berlabel berbeda; itu bukti salah label "
                 "tanpa perlu model."),
    },
    "ngram": {
        "apa": "Kata dan frasa (1-2 kata) yang paling khas untuk tiap kelas.",
        "cara": ("CountVectorizer biner (1-2 kata, muncul di minimal 3 teks). Untuk tiap kelas, "
                 "uji chi-square satu-lawan-semua (sklearn `chi2`); hanya frasa yang lebih sering "
                 "di kelas itu daripada di kelas lain yang dipertimbangkan."),
        "kenapa": ("Menunjukkan apa yang membedakan kelas dan memperlihatkan penanda yang "
                   "mencurigakan (mis. nama sumber data) yang bisa jadi bocoran."),
        "batasan": ("Frasa khas bisa berupa kata umum atau nama. Korelasi, bukan sebab. Kelas "
                    "kecil menghasilkan skor yang tidak stabil."),
        "juri": "Kami menunjukkan frasa yang secara statistik membedakan tiap kelas.",
    },
    "kosakata_alay": {
        "apa": ("Ukuran kosakata, kata yang hanya muncul sekali, rasio kata alay/slang, dan "
                "campuran bahasa Indonesia-Inggris."),
        "cara": ("Kata = deretan huruf/angka (huruf kecil). Kata muncul sekali (hapax) = "
                 "frekuensi 1. Kata alay = ada di kamus alay Salsabila + kamus slang proyek "
                 "(`data_latihan/preprocessing`), tidak termasuk kata yang bentuk bakunya sama. "
                 "Campuran bahasa = teks yang memuat sedikitnya satu stopword Inggris DAN satu "
                 "stopword Indonesia dari daftar kecil (sekitar 45 kata)."),
        "kenapa": ("Banyak kata yang muncul sekali berarti model berbasis kata sulit belajar; "
                   "kata alay perlu dinormalisasi lebih dulu; bahasa campuran menentukan pilihan "
                   "model (multibahasa)."),
        "batasan": ("Kamus alay tidak lengkap dan sebagian kata ambigu. Deteksi campuran bahasa "
                    "sangat sederhana dan tidak mengenali bahasa daerah."),
        "juri": ("Kami mengukur kosakata, kata alay, dan campuran bahasa untuk memutuskan "
                 "perlu tidaknya normalisasi slang dan model multibahasa."),
    },
    "kode": {
        "apa": ("Kode notebook EDA (.py dengan sel `# %%` atau .ipynb) yang mengulang semua "
                "langkah yang kamu pilih di web ini, tanpa membutuhkan `_laya_eda`."),
        "cara": ("codegen.py menyusun sel satu per satu dari pilihan sidebar (kolom, format "
                 "angka, skala log, rasio), lensa Laya yang sudah dijalankan, outlier IQR per "
                 "kategori, dan catatan buang / ganti label dari tab Telusuri. Nomor baris di "
                 "catatan = indeks baris CSV asli."),
        "kenapa": ("Keputusan manual harus bisa diulang. Kode yang bisa dijalankan dari awal "
                   "membuktikan bahwa data bersih untuk model berasal dari langkah yang jelas."),
        "batasan": ("Tab Temukan kategori (kelompok) tidak ikut dibuat ulang karena hasil "
                    "pengelompokan bisa berubah tiap jalan; lensa yang memakai label juga "
                    "dilewati. Path CSV dianggap dari folder notebook di _.NEW-GEN/<folder>/."),
        "juri": ("Semua yang kami temukan secara manual sudah diubah jadi kode yang bisa "
                 "dijalankan ulang (reproducible), bukan hanya catatan."),
    },
    "temuan": {
        "apa": "Ringkasan semua temuan dari tab lain, diurutkan dari yang paling penting.",
        "cara": ("Tiap tab mencatat temuan dengan alasan, saran, dan nilai penting 0–1 (misalnya "
                 "proporsi baris yang terkena). Temuan yang sama hanya ditampilkan sekali. "
                 "Laporan Markdown menyusun data, temuan, dan lensa yang dipakai."),
        "kenapa": ("Juri dan tim perlu melihat keputusan terpenting lebih dulu tanpa membuka "
                   "sepuluh tab."),
        "batasan": ("Nilai penting adalah aturan sederhana, bukan ukuran statistik; temuan "
                    "hanya muncul dari tab yang sudah dihitung (misalnya setelah Laya jalan)."),
        "juri": ("Setiap temuan punya alasan dan saran tindakan, dan laporan ini adalah "
                 "ringkasan EDA kami."),
    },
}


def show_explanation(key, evidence=""):
    # kotak lipat berisi lima bagian; evidence = angka dari data yang sedang dibuka (opsional)
    sections = EXPLANATIONS[key]
    with st.expander("ℹ️ Cara kerja & kenapa"):
        for part in ["apa", "cara", "kenapa", "batasan", "juri"]:
            st.markdown(f"**{TITLES[part]}**")
            st.markdown(sections[part])
        if evidence != "":
            st.markdown(f"**Bukti dari data ini:** {evidence}")
