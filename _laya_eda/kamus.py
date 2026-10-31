"""Kamus kolom: arti tiap nama kolom yang muncul di tabel aplikasi (nilai apa, dihitung dari
apa). Dipakai dua cara: tooltip di judul kolom (arahkan kursor ke judul kolom) lewat show_table,
dan daftar lengkap yang bisa dicari di sidebar (show_glossary)."""
import pandas as pd
import streamlit as st

# nama kolom persis -> arti
EXACT = {
    "row": "Nomor baris di CSV asli (mulai 0). Dipakai untuk mencari baris yang sama di tab lain.",
    "nilai": ("ISI sel aslinya di baris itu untuk kolom yang sedang dicek (angka atau teks), "
              "BUKAN jumlah. Jumlah baris ada di kolom `jumlah` / `count`."),
    "jumlah": "Banyak baris (hitungan), bukan isi sel.",
    "count": "Banyak baris (hitungan) yang tidak kosong.",
    "proportion": "Bagian dari seluruh baris (0–1).",
    "mean": "Rata-rata.",
    "std": "Simpangan baku: seberapa jauh angka menyebar dari rata-rata.",
    "min": "Angka terkecil.",
    "max": "Angka terbesar.",
    "25%": "Kuartil bawah: 25% baris punya angka di bawah ini.",
    "50%": "Median: setengah baris di bawah, setengah di atas.",
    "75%": "Kuartil atas: 75% baris punya angka di bawah ini.",
    "kolom": "Nama kolom CSV yang dibahas di baris tabel ini.",
    "kelompok": "Kategori pembanding (nilai kolom kategori pilihanmu) tempat baris ini dibanding.",
    "kategori": "Nilai kolom kategori untuk baris ini.",
    "label": "Label asli dari CSV.",
    "tebakan": "Tebakan model untuk label baris ini (model kecil dengan validasi silang).",
    "teks": "Teks yang dibaca (gabungan kolom yang dibaca Laya).",
    "alasan": "Semua alasan baris ini ditandai janggal (duplikat, outlier, Laya ragu, ...).",
    "alasan_kolom_ini": "Alasan baris ini janggal khusus untuk kolom angka yang sedang dicek.",
    "status": ("Hasil cek outlier: normal / janggal di keduanya / janggal hanya di kategorinya / "
               "janggal hanya di keseluruhan."),
    "outlier": "True = di luar batas wajar IQR di dalam kategorinya sendiri.",
    "outlier_keseluruhan": "True = di luar batas wajar IQR bila dibanding SEMUA baris.",
    "outlier_kategori": "True = di luar batas wajar IQR bila dibanding baris sekategori saja.",
    "outlier_di_kategori": "Banyak baris kategori ini yang janggal dibanding sesama kategorinya.",
    "outlier_di_keseluruhan": "Banyak baris kategori ini yang janggal dibanding seluruh data.",
    "median_kelompok": "Median angka kolom ini di antara baris sekategori.",
    "median_kategori": "Median angka kolom ini di antara baris sekategori.",
    "median_keseluruhan": "Median angka kolom ini di seluruh data.",
    "kali_median": ("nilai ÷ median kategorinya. 1 = biasa; 0.1 = sepersepuluh (terlalu "
                    "kecil/murah); 10 = sepuluh kali (terlalu besar/mahal)."),
    "batas_bawah": "Batas wajar bawah (IQR Tukey: Q1 − 1.5×IQR) di kategorinya, satuan asli.",
    "batas_atas": "Batas wajar atas (IQR Tukey: Q3 + 1.5×IQR) di kategorinya, satuan asli.",
    "persentil_keseluruhan": "Posisi angka ini di seluruh data, 0–100 (90 = di atas 90% baris).",
    "persentil_kategori": "Posisi angka ini di antara baris sekategori, 0–100.",
    "persentil": "Posisi angka ini, 0–100 (90 = lebih besar dari 90% baris).",
    "jarak_ke_kategorinya": ("Jarak makna (1 − kemiripan kosinus MiniLM) ke pusat kategorinya. "
                             "0 = sangat mirip, makin besar makin jauh."),
    "kategori_terdekat": "Kategori yang pusat maknanya paling dekat dengan teks ini.",
    "nyasar": "True = teks lebih dekat ke pusat kategori LAIN daripada kategorinya sendiri.",
    "jarak_lain": "Jarak makna ke pusat kategori terdekat (bisa kategorinya sendiri).",
    "jarak_median": "Median jarak makna di kategori ini.",
    "batas_jarak": "Batas wajar jarak makna di kategori ini; di atasnya = jauh.",
    "jauh": "True = jarak makna ke kategorinya di atas batas wajar.",
    "teks_a": "Teks pertama dari pasangan.",
    "teks_b": "Teks kedua dari pasangan.",
    "row_a": "Nomor baris teks pertama.",
    "row_b": "Nomor baris teks kedua.",
    "kemiripan": "Kemiripan makna dua teks, 0–1 (1 = sama persis maknanya).",
    "laya_ragu": "True = salah satu lensa Laya yakin di bawah batas ragu (sidebar).",
    "keputusan": "Keputusan manualmu di tab Telusuri: biarkan / buang / ganti_label.",
    "catatan": "Catatan manualmu untuk baris ini.",
    "label_baru": "Label pengganti bila keputusan = ganti_label.",
    "ganti_ke": "Bentuk label yang disarankan sebagai pengganti (penulisan yang paling sering).",
    "panjang": "Panjang teks dalam karakter.",
    "jumlah_kata": "Banyak kata dalam teks.",
    "rasio_bukan_huruf": "Bagian karakter yang bukan huruf (angka, simbol, emoji), 0–1.",
    "rasio_kapital": "Bagian huruf yang kapital, 0–1.",
    "duplikat": "True = teks persis sama muncul di baris lain.",
    "duplikat_mirip": "True = teks sama setelah huruf kecil dan tanda baca dibuang.",
    "kosong": "True = sel kosong.",
    "kosong_persen": "Persen baris yang kosong di kolom ini.",
    "persen_kosong": "Persen baris yang kosong di kolom ini.",
    "nilai_unik": "Banyak nilai berbeda di kolom ini.",
    "tipe": "Jenis kolom menurut pandas / aplikasi (angka, teks, kategori, tanggal).",
    "p_value": ("Peluang hubungan sebesar ini muncul karena kebetulan. < 0.05 = biasanya dianggap "
                "nyata; tapi lihat ukuran_efek untuk besarnya."),
    "ukuran_efek": "Besar hubungan (lihat nama_efek); 0 = tidak ada hubungan.",
    "nama_efek": "Ukuran efek yang dipakai: Cramér's V, eta², rho Spearman, dsb.",
    "efek": "Besar efek dalam kata: kecil / sedang / besar.",
    "tafsir": "Kesimpulan singkat dari p_value dan ukuran efek.",
    "rho": "Korelasi Spearman −1..1 (urutan naik bersama = positif).",
    "kolom_a": "Kolom pertama dari pasangan.",
    "kolom_b": "Kolom kedua dari pasangan.",
    "cramer_v": "Kekuatan hubungan dua kategori, 0–1.",
    "cramers_v": "Kekuatan hubungan dua kategori, 0–1.",
    "psi": ("Population Stability Index: seberapa beda sebaran train vs test. < 0.1 mirip, "
            "0.1–0.25 agak beda, > 0.25 beda jauh."),
    "grup": "Kode grup fitur.",
    "grup fitur": "Nama grup fitur yang diuji.",
    "baris": "Banyak baris yang dipakai dalam uji ini.",
    "skor_baseline": "Skor validasi silang model TANPA grup fitur ini.",
    "skor_dengan_grup": "Skor validasi silang model DENGAN grup fitur ini.",
    "selisih": ("Selisih dua angka di sebelahnya: skor_dengan_grup − skor_baseline (tab Fitur), "
                "tebakan − label (regresi), atau peluang_saran − peluang_label."),
    "selisih_mutlak": "Besar selisih tanpa tanda (tebakan meleset sejauh ini).",
    "std_fold": "Simpangan skor antar-lipatan validasi silang; selisih di bawah ini = belum pasti.",
    "detik": "Lama hitung dalam detik.",
    "fitur": "Nama kolom fitur.",
    "turun_skor": "Skor turun sebanyak ini bila fitur diacak (permutation importance).",
    "mutual_info": "Informasi bersama fitur dengan label; 0 = tidak memberi petunjuk.",
    "bobot": "Bobot kata di model linear: positif mendorong ke kelas ini, negatif menjauhkan.",
    "peluang_label": "Peluang (menurut model) bahwa label yang tertulis benar.",
    "label_saran": "Label yang menurut model paling cocok.",
    "peluang_saran": "Peluang label saran menurut model.",
    "ditandai": "Banyak baris yang ditandai metode ini.",
    "tepat": "Dari yang ditandai, berapa yang memang salah label (data uji label dibalik).",
    "presisi": "tepat ÷ ditandai.",
    "recall": "Bagian salah label yang berhasil ditemukan.",
    "dasar_acak": "Presisi kalau menandai acak (pembanding).",
    "presisi_top_k": "Presisi pada k baris teratas yang ditandai.",
    "kata_total": "Banyak kata seluruhnya.",
    "kosakata": "Banyak kata berbeda.",
    "muncul_1x": "Banyak kata yang hanya muncul sekali.",
    "kata_khas": "Kata yang paling khas untuk kategori ini dibanding kategori lain.",
    "kata_baru": "Kata di test yang tidak pernah ada di train.",
    "kata_test": "Banyak kata di test.",
    "persen_kata_baru": "Persen kata test yang tidak ada di train.",
    "silhouette_rata2": "Seberapa terpisah kelompok makna, −1..1 (besar = rapi terpisah).",
    "penting": "Bobot pentingnya temuan, 0–1.",
    "temuan": "Isi temuan.",
    "saran": "Saran tindakan.",
    "label_bentrok": "True = teks yang sama punya label berbeda.",
    "angka_terbesar_log": "log10 dari angka terbesar yang ada di teks.",
}

# awalan nama kolom -> arti (bagian sesudah awalan = nama kolom asal)
PREFIX = [
    ("teks__", "Statistik teks '{rest}' (panjang, jumlah kata, rasio bukan huruf, ...)."),
    ("kosong_", "1 = kolom '{rest}' kosong di CSV asli (sebelum diisi), 0 = ada isinya."),
    ("log_", "log(1 + angka '{rest}'): meredam angka yang sangat besar."),
    ("rasio_", "Angka '{rest}' dibagi median kategorinya (dihitung dari train). 1 = biasa."),
    ("frek_", "Berapa kali nilai '{rest}' muncul di train (nilai langka = kecil)."),
    ("oh_", "One-hot: 1 = kolom kategori bernilai '{rest}' (kolom_nilai), 0 = bukan."),
    ("kode_", "Kode angka kategori '{rest}' (0 = nilai tersering di train, -1 = nilai baru)."),
    ("target_", "Target encoding '{rest}' (kolom_kelas): porsi kelas itu / rata-rata "
                "label di train untuk nilai kategori baris ini, dihaluskan."),
    ("makna_", "Kedekatan makna '{rest}' (kolom_ke-n): posisi arti nilai kategori "
               "pada sumbu PCA ke-n dari vektor MiniLM; arti mirip = angka mirip."),
    ("laya_", "Jawaban lensa Laya '{rest}'."),
    ("persentil_di_", "Posisi angka ini 0–100 di antara baris dengan {rest}."),
]

# akhiran kolom lensa Laya -> arti
SUFFIX = [
    ("__yakin", "Keyakinan Laya (0–1) pada jawaban lensa '{rest}'. Di bawah batas ragu = ragu."),
    ("__pertama_nilai", "Peluang pilihan pertama Laya pada lensa '{rest}' (0–1)."),
    ("__kedua_nilai", "Peluang pilihan kedua Laya pada lensa '{rest}' (0–1)."),
    ("__kedua", "Jawaban pilihan kedua Laya pada lensa '{rest}'."),
    ("__nilai", ("Skor angka lensa '{rest}': peluang 'ya' untuk lensa ya/tidak, nilai pada skala "
                 "untuk lensa skala. Klik judul kolom untuk mengurutkan.")),
]


def explain(name, notes=None):
    # arti satu nama kolom; notes = arti khusus tabel ini (menimpa kamus umum)
    name = str(name)
    if notes is not None and name in notes:
        return notes[name]
    if name in EXACT:
        return EXACT[name]
    for suffix, text in SUFFIX:
        if name.endswith(suffix):
            return text.format(rest=name[:-len(suffix)])
    for prefix, text in PREFIX:
        if name.startswith(prefix) and len(name) > len(prefix):
            return text.format(rest=name[len(prefix):])
    return None


def lens_answer(name):
    # kolom bernama sama dengan lensa = jawaban Laya (daftar lensa diisi app.py)
    if name in st.session_state.get("kamus_lensa", []):
        return f"Jawaban Laya untuk lensa '{name}' (pilihan pertama)."
    return None


def frame_columns(frame):
    # nama kolom + nama index dari DataFrame, Series, atau Styler
    if hasattr(frame, "data") and isinstance(frame.data, pd.DataFrame):
        frame = frame.data
    if isinstance(frame, pd.Series):
        return [frame.name], frame.index.name
    return list(frame.columns), frame.index.name


def column_help(frame, notes=None):
    # column_config st.dataframe: tooltip di judul kolom yang punya arti
    columns, index_name = frame_columns(frame)
    config = {}
    for name in columns:
        text = explain(name, notes)
        if text is None:
            text = lens_answer(name)
        if text is not None:
            config[str(name)] = st.column_config.Column(help=text)
    if index_name is not None and explain(index_name, notes) is not None:
        config["_index"] = st.column_config.Column(help=explain(index_name, notes))
    return config


def show_table(frame, notes=None, **options):
    # pengganti st.dataframe: sama persis, ditambah tooltip arti kolom
    config = column_help(frame, notes)
    config.update(options.pop("column_config", {}))
    return st.dataframe(frame, column_config=config, **options)


def glossary_lines(query):
    lines = []
    for name in EXACT:
        if query in name or query in EXACT[name].lower():
            lines.append(f"- **`{name}`**: {EXACT[name]}")
    for prefix, text in PREFIX:
        sample = text.format(rest="X")
        if query in prefix or query in sample.lower():
            lines.append(f"- **`{prefix}X`**: {sample}")
    for suffix, text in SUFFIX:
        sample = text.format(rest="X")
        if query in suffix or query in sample.lower():
            lines.append(f"- **`X{suffix}`**: {sample}")
    return lines


def show_glossary():
    with st.sidebar.expander("Kamus kolom (arti nama kolom tabel)"):
        st.caption("Arahkan kursor ke judul kolom tabel mana pun untuk artinya. Daftar lengkap:")
        query = st.text_input("Cari nama kolom", "", key="kamus_cari").strip().lower()
        st.markdown("\n".join(glossary_lines(query)))
