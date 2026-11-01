"""Flowchart cara kerja tiap tab (DOT untuk st.graphviz_chart). Nama metode/library ditulis di node.

Bentuk: kotak = langkah otomatis, jajar genjang = keputusan pengguna, belah ketupat = cabang,
silinder = daftar langkah (yang jadi kode). Merah = bisa berbahaya, wajib diperiksa.
"""
from __future__ import annotations

import re

HEAD = ('digraph { rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.3;'
        'node [shape=box, style="rounded,filled", fillcolor="#eef2f7", fontname="Helvetica", fontsize=11];'
        'edge [fontname="Helvetica", fontsize=10, color="#8a94a6", fontcolor="#8a94a6"];')
SPECIAL = {"user": 'user [shape=parallelogram, fillcolor="#fff4cc"];',
           "steps": 'steps [shape=cylinder, fillcolor="#d9f2e3", label="daftar langkah → kode lurus"];'}


def dot(body: str) -> str:
    """Node `user` (keputusan) dan `steps` (daftar langkah) hanya dibuat bila dipakai di flowchart ini."""
    special = "".join(line for name, line in SPECIAL.items() if re.search(rf"\b{name}\b", body))
    return HEAD + special + body + "}"


PREVIEW = 'p [label="pratinjau: sebelum → sesudah\\n(baris dibuang, kolom baru, sel berubah)"];'

FLOWS = {
    "langkah": dot('''
        a [label="data mentah → split (train / valid) + test"]; steps;
        b [label="langkah 1 … n dijalankan berurutan\\n(blok kode yang sama dengan file .py)"];
        c [shape=diamond, label="galat?", fillcolor="#ffffff"]; d [label="berhenti, tampilkan galat apa adanya", fillcolor="#fde2e2"];
        user [label="lihat data di langkah ke-k, matikan / pindah / hapus langkah"];
        a -> b; steps -> b; b -> c; c -> d [label="ya"]; c -> user [label="tidak"]; user -> steps;'''),
    "data": dot('''
        a [label="baca CSV train (pd.read_csv)"]; b [label="baca CSV test (opsional)"];
        user [label="pilih kolom label"]; c [label="grid data di langkah terpilih\\n+ wawasan kolom (tipe, kosong, unik, sebaran)"];
        a -> user; b -> user; user -> c;'''),
    "split": dot('''
        a [label="train mentah (sesudah pd.read_csv)"];
        user [label="pilih metode: acak (stratified) / kelompok / waktu\\n+ porsi valid + random_state"];
        b [label="kode split: sklearn train_test_split /\\nGroupShuffleSplit / tanggal batas"];
        c [label="cek: ukuran, proporsi label, baris kembar train↔valid,\\nkelompok di dua sisi, rentang waktu"];
        d [label="langkah EDA: statistik dari train SAJA\\n→ dipakai ke valid & test (tidak pernah dibuang)"];
        e [label="ekspor: .py (baca → split → langkah) +\\ntrain_bersih, valid_bersih, test_bersih"];
        a -> user -> b -> c; c -> user [label="belum pas", style=dashed]; c -> steps [label="pakai split"];
        steps -> d -> e;'''),
    "temuan": dot('''
        a [label="pindai data di langkah terakhir"];
        b [label="spasi berlebih, penanda kosong ('-' yang sering = jawaban, bukan kosong),\\nejaan berbeda, nilai penjaga 2^31−1, angka-teks, checkbox,\\nkolom id/kembar, duplikat, kolom berurutan"];
        c [label="setiap temuan → usulan langkah + penjelasan"];
        user [label="klik 'tambah langkah' (tidak ada yang otomatis)"];
        a -> b -> c -> user -> steps;'''),
    "operasi": dot(f'''
        a [label="pilih kelompok → operasi"]; b [label="apa / kapan cocok / kapan berbahaya"];
        user [label="isi parameter (kolom, nilai, metode)"]; {PREVIEW}
        k [label="kode pandas lurus langkah ini"];
        a -> b -> user -> p -> k -> steps [label="tambah / simpan ubahan"];'''),
    "ejaan": dot(f'''
        a [label="kolom kategori (isian bebas → tab Isian bebas)"];
        b [label="aturan kunci (key collision OpenRefine): huruf kecil,\\nKab.→Kabupaten, tanpa spasi/tanda baca\\n'S- 2' = 'S2' (tanpa daftar)"];
        e [label="ejaan baku tiap kunci = ejaan tersering di train"];
        g [label="usulan GABUNG + alasan: urutan kata (fingerprint),\\nkamus jenjang (SMA = MA = SLTA), kamus 'tidak'"];
        c [label="salah ketik: rapidfuzz (kunci → kunci jauh lebih sering)", fillcolor="#fde2e2"];
        d [label="skrub.deduplicate (opsional)", fillcolor="#fde2e2"];
        user [label="pilih kolom, centang gabungan yang benar"]; {PREVIEW}
        a -> b -> e -> user; a -> g -> user; a -> c -> user [style=dashed]; a -> d -> user [style=dashed];
        user -> p -> steps [label="Satukan ejaan\\n(±20 baris, berapa pun variasinya)"];'''),
    "isian": dot(f'''
        a [label="isian bebas, mis. \'Genteng & asbas.\'"];
        b [label="huruf kecil, pecah di , & / + ; dan atau ( )"];
        c [label="kosakata = kata ≥ N kali (kolom ini + kolom terkait)"];
        d [label="kata jarang → rapidfuzz extractOne, skor ≥ 75\\n(≥ 80 untuk < 4 huruf, STOPWORDS dilewati)"];
        e [label="kata baku mirip (genting/genteng) = kandidat sinonim"];
        f [label="kata pengisi (satu regex): kata nama kolom, 'dari', 'biasa'\\n'lantai keramik' = 'keramik'"];
        g [label="peta FRASA (bahan ≥ N baris): kamus KBBI (bata merah = batu bata),\\nurutan kata ('plester semen' = 'semen plester')"];
        h [label="'genteng beton' memuat 'genteng' (jauh lebih sering)\\n→ tidak dicentang: rincian bisa penting", fillcolor="#fde2e2"];
        user [label="centang usulan yang benar"]; {PREVIEW}
        a -> b -> c -> d -> user; c -> e -> user; d -> f -> g -> user; g -> h -> user [style=dashed];
        user -> p -> steps [label="bakukan isian bebas"];'''),
    "outlier": dot('''
        a [label="kolom angka di langkah terakhir"];
        b [label="bandingkan IQR, IQR-log, MAD, MAD-log (batas dari train)"];
        c [shape=diamond, label="sebaran 0?", fillcolor="#ffffff"]; d [label="tanpa batas + peringatan"];
        e [label="baris di luar batas (+ per kelompok label / Laya)"];
        user [label="pilih metode, k, aksi"];
        g [label="clip / nan: train + valid + test"]; i [label="drop: train saja", fillcolor="#fde2e2"];
        a -> b -> c; c -> d [label="ya"]; c -> e [label="tidak"]; e -> user; user -> g -> steps; user -> i -> steps;'''),
    "label": dot('''
        a [label="distribusi label, duplikat fitur beda label"];
        b [shape=diamond, label="model?", fillcolor="#ffffff"];
        c [label="cepat: HistGradientBoosting 5-fold\\n(kolom angka saja)"];
        d [label="kuat: AutoGluon bagging 5-fold\\npredict_proba_oof (semua kolom)"];
        e [label="cleanlab.find_label_issues + skor kualitas label"];
        user [label="baca saja: label tidak diubah otomatis"];
        a -> b; b -> c; b -> d; c -> e; d -> e; e -> user;'''),
    "test": dot('''
        a [label="train dan test setelah semua langkah"];
        b [label="kolom hilang / lebih"]; c [label="kategori baru di test"];
        d [label="pergeseran angka (scipy ks_2samp)"];
        a -> b -> c -> d;'''),
    "autogluon": dot('''
        a [label="split proyek (train / valid)\\natau 80/20 sementara bila belum di-split"];
        b [label="AutoGluon di data MENTAH"]; c [label="semua langkah: statistik dari train,\\nvalid ikut diubah (tidak dibuang)"];
        d [label="AutoGluon di data BERSIH"]; e [label="skor di baris uji yang SAMA\\n+ papan model"];
        f [label="permutation importance: kolom penting,\\nkolom id penting = curiga bocor", fillcolor="#fde2e2"];
        user [label="langkah yang menurunkan skor → periksa / matikan"];
        a -> b -> e; a -> c -> d -> e; e -> f -> user;'''),
    "kode": dot('''
        steps; a [label="satu blok # %% per langkah\\ntanpa def: pandas/numpy lurus"];
        s [label="blok split (sesudah baca data)"];
        b [label="statistik dari train → variabel biasa → dipakai ke train, valid & test"];
        c [label="aplikasi menjalankan blok yang sama (exec)"];
        d [label="unduh .py / .ipynb / langkah .json / CSV bersih (train, valid, test)"];
        s -> a; steps -> a -> b -> c -> d;'''),
    "laya": dot('''
        user [label="pilih kolom + pertanyaan + pilihan jawaban (lensa)"];
        a [label="Laya menjawab per baris (sampel)"];
        b [label="jawaban disimpan TERPISAH (hasil/laya_*.csv)\\ntidak pernah masuk data bersih"];
        c [label="konteks: kelompok pembanding outlier,\\n\'apakah dua ejaan bermakna sama?\'"];
        user -> a -> b -> c;'''),
}
