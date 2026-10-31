"""Flowchart cara kerja tiap tab (DOT untuk st.graphviz_chart). Nama metode/library ditulis di node.

Bentuk: kotak = langkah otomatis, jajar genjang = keputusan pengguna, belah ketupat = cabang,
silinder = resep (yang masuk ke kode pipeline). Merah = bisa berbahaya, wajib diperiksa.
"""
from __future__ import annotations

import re

HEAD = ('digraph { rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.3;'
        'node [shape=box, style="rounded,filled", fillcolor="#eef2f7", fontname="Helvetica", fontsize=11];'
        'edge [fontname="Helvetica", fontsize=10, color="#8a94a6", fontcolor="#8a94a6"];')
SPECIAL = {"user": 'user [shape=parallelogram, fillcolor="#fff4cc"];',
           "recipe": 'recipe [shape=cylinder, fillcolor="#d9f2e3", label="resep → kode pipeline"];'}


def dot(body: str) -> str:
    """Node `user` (keputusan) dan `recipe` (resep) hanya dibuat bila dipakai di flowchart ini."""
    special = "".join(line for name, line in SPECIAL.items() if re.search(rf"\b{name}\b", body))
    return HEAD + special + body + "}"


FLOWS = {
    "data": dot('''
        a [label="baca CSV train (pd.read_csv)"]; b [label="baca CSV test (opsional)"];
        user [label="pilih kolom label"]; c [label="label tidak ada di test? → test tanpa label (normal)"];
        a -> user -> c; b -> c; c -> recipe [label="label"];'''),
    "ringkasan": dot('''
        a [label="profil per kolom: tipe tebakan, kosong %, unik, nilai tersering"];
        b [label="temuan baris: duplikat persis, duplikat beda label, kosong per baris"];
        c [label="skrub.TableReport (opsional, tombol)"];
        a -> b -> c; user [label="baca saja, tidak mengubah apa pun"]; c -> user;'''),
    "tipe": dot('''
        a [label="tebak tipe: angka / angka-teks / kategori / isian bebas / id"];
        b [label="kolom kembar: skrub.column_associations (Cramér\'s V ≥ 0,85)"];
        user [label="pilih: kolom dibuang, kolom angka-teks, kolom teks yang dirapikan"];
        c [label="angka-teks → to_number (format Indonesia 1.500.000,5)"];
        d [label="teks → tidy_text (strip, spasi ganda, penanda kosong → NaN)"];
        a -> b -> user; user -> c -> recipe; user -> d -> recipe;'''),
    "kosong": dot('''
        a [label="penanda kosong palsu di teks: \'-\', \'?\', \'null\', ..."];
        b [label="nilai penjaga ≥ 2.000.000.000 (2^31−1 dan varian)"];
        c [label="kolom angka dengan banyak nol"]; d [label="kolom 1 nilai + kosong (checkbox)"];
        user [label="setujui per kolom"];
        e [label="penanda → NaN, penjaga → NaN (mask)"]; f [label="nol → NaN hanya bila dipilih", fillcolor="#fde2e2"];
        g [label="kosong → nilai pilihan (fillna), mis. \'Tidak\'"];
        a -> user; b -> user; c -> user; d -> user; user -> e -> recipe; user -> f -> recipe; user -> g -> recipe;'''),
    "kategori": dot('''
        a [label="pilih kolom kategori"]; b [label="1. fingerprint (OpenRefine key collision)\\nhuruf kecil, tanda baca, urutan kata, singkatan Kab.→Kabupaten"];
        c [label="2. skrub.deduplicate (opsional)\\nklaster n-gram huruf", fillcolor="#fde2e2"];
        user [label="tabel usulan: centang yang setuju, ubah ejaan baku bila perlu"];
        d [label="peta {ejaan asli: ejaan baku}"]; e [label="sebelum → sesudah: value_counts"];
        a -> b -> user; a -> c -> user [style=dashed]; user -> d -> e -> recipe;'''),
    "isian": dot('''
        a [label="isian bebas, mis. \'Genteng & asbas.\'"];
        b [label="huruf kecil, pecah di , & / + ; dan atau ( )"];
        c [label="hitung kata (kolom ini + kolom terkait)"];
        d [label="kosakata baku = kata ≥ N kali (default 30)"];
        e [label="kata jarang → rapidfuzz.process.extractOne\\nfuzz.ratio ≥ 75 (≥ 80 untuk < 4 huruf)\\nSTOPWORDS (tanpa, tidak, seng, ...) dilewati"];
        f [label="kata baku mirip (genting/genteng)\\n= kandidat sinonim, tidak otomatis"];
        user [label="centang usulan yang benar"];
        g [label="canonical_materials: \'asbes, genting\'\\n(nilai diganti, kolom tetap satu)"];
        a -> b -> c -> d -> e -> user; d -> f -> user; user -> g -> recipe;'''),
    "outlier": dot('''
        a [label="pilih kolom angka (setelah penjaga → NaN)"];
        b [label="bandingkan: IQR, IQR-log, MAD, MAD-log\\n(batas dihitung dari train)"];
        c [shape=diamond, label="sebaran 0?", fillcolor="#ffffff"];
        d [label="tanpa batas + peringatan"];
        e [label="nilai di luar batas + contoh baris\\n(opsional: per kelompok label / jawaban Laya)"];
        user [label="pilih metode, k, aksi"];
        f [label="flag: tidak diubah"]; g [label="clip: potong ke batas (train+test)"];
        h [label="nan: jadi kosong → isi kosong"]; i [label="drop: buang baris (train saja)", fillcolor="#fde2e2"];
        a -> b -> c; c -> d [label="ya"]; c -> e [label="tidak"]; e -> user;
        user -> f; user -> g; user -> h; user -> i; f -> recipe; g -> recipe; h -> recipe; i -> recipe;'''),
    "baris": dot('''
        a [label="duplikat persis (df.duplicated)"]; b [label="baris dengan banyak kosong"];
        c [label="aturan logika (pandas eval), mis. luas bangunan > 100.000"];
        user [label="pilih aturan buang"]; d [label="hanya baris TRAIN yang dibuang\\ntest tidak pernah dibuang"];
        e [label="tabel alasan: satu kolom per aturan"];
        a -> user; b -> user; c -> user; user -> d -> e -> recipe;'''),
    "isi": dot('''
        a [label="kolom yang masih kosong setelah langkah sebelumnya"];
        user [label="pilih strategi per kolom"];
        b [label="sklearn SimpleImputer\\nmedian / mean / most_frequent / constant"];
        c [label="statistik dipelajari dari train (fit),\\ndipakai sama di test (transform)"];
        a -> user -> b -> c -> recipe;'''),
    "label": dot('''
        a [label="distribusi label"]; b [label="duplikat fitur dengan label berbeda"];
        c [label="cleanlab Datalab (opsional)\\nmodel cross-validation → label yang mungkin salah"];
        user [label="baca saja: hanya ditandai, label tidak diubah otomatis"];
        a -> b -> c -> user;'''),
    "test": dot('''
        a [label="kolom: hilang / lebih / urutan beda"]; b [label="kategori di test yang tidak ada di train"];
        c [label="pergeseran distribusi angka (scipy ks_2samp)"];
        d [label="di pipeline: reindex ke kolom train,\\nOneHotEncoder handle_unknown, pandera cek skema"];
        a -> b -> c -> d;'''),
    "encoding": dot('''
        a [label="pilih kolom"]; user [label="pilih encoder (lihat penjelasan tiap encoder)"];
        b [label="tidak di-encode (default)"]; c [label="OneHotEncoder: satu kolom 0/1 per nilai\\nnilai jarang → infrequent"];
        d [label="OrdinalEncoder: kode angka, nilai baru = −1"]; e [label="berurutan: urutan dari kamu"];
        f [label="one-hot per bahan: str.get_dummies + _lainnya"];
        g [label="skrub String/MinHash/Gap Encoder\\nkolom angka tanpa makna langsung", fillcolor="#fde2e2"];
        h [label="PRATINJAU nama kolom baru sebelum disimpan"];
        a -> user; user -> b; user -> c -> h; user -> d -> h; user -> e -> h; user -> f -> h; user -> g -> h; h -> recipe;'''),
    "split": dot('''
        user [label="pilih: tanpa split / acak / stratified / k-fold"];
        a [label="train_test_split (stratify=label bila stratified)"]; b [label="StratifiedKFold / KFold → folds.csv"];
        c [label="split dilakukan SETELAH pembersihan train"];
        user -> a -> c; user -> b -> c; c -> recipe;'''),
    "kode": dot('''
        recipe [label="resep (JSON)"]; a [label="codegen: resep → pipeline_eda.py"];
        b [label="prepare → batas outlier (fit) → buang baris train → clip/nan\\n→ one-hot bahan → ColumnTransformer (fit) → pandera skema"];
        c [label="transform(test, state): langkah dan statistik yang sama"];
        d [label="aplikasi MENJALANKAN kode yang sama (exec)"];
        e [label="unduh .py / .ipynb / resep / CSV bersih / joblib"];
        recipe -> a -> b -> c -> d -> e;'''),
    "laya": dot('''
        user [label="pilih kolom + pertanyaan + pilihan jawaban (lensa)"];
        a [label="Laya menjawab per baris (sampel)"];
        b [label="jawaban disimpan TERPISAH (hasil/laya_*.csv)\\ntidak pernah masuk data bersih"];
        c [label="dipakai sebagai konteks: kelompok pembanding outlier,\\ncek \'apakah dua ejaan bermakna sama?\'"];
        user -> a -> b -> c;'''),
}
