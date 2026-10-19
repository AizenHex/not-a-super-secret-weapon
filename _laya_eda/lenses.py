"""Lensa siap pakai untuk Laya. Satu lensa = satu pertanyaan + pilihan jawabannya.

type "choice" = pilih satu (criteria = {pilihan: penjelasan})
type "noul"   = ya / tidak (tanpa criteria)
type "score"  = skala (criteria = daftar tingkat, urut dari terendah)
with_label    = True bila Laya juga perlu melihat kolom label (mencari label yang salah)

Tambah lensa baru cukup dengan menyalin salah satu blok di bawah.
"""

PRESETS = {
    "emosi": {
        "type": "choice",
        "instructions": "Emosi apa yang paling terasa dari penulis teks ini?",
        "criteria": {
            "senang": "gembira, bangga, puas, bersyukur",
            "sedih": "kecewa, kehilangan, putus asa",
            "marah": "kesal, benci, jengkel",
            "takut": "cemas, khawatir, ngeri",
            "cinta": "sayang, rindu, kagum pada seseorang",
            "netral": "informasi biasa tanpa emosi",
        },
    },
    "sentimen": {
        "type": "choice",
        "instructions": "Bagaimana sikap penulis terhadap hal yang dibicarakan?",
        "criteria": {
            "positif": "suka, setuju, memuji",
            "negatif": "tidak suka, menolak, mengeluh",
            "netral": "tidak memihak, hanya informasi",
        },
    },
    "maksud": {
        "type": "choice",
        "instructions": "Apa maksud utama penulis teks ini?",
        "criteria": {
            "bertanya": "meminta informasi atau bantuan",
            "mengeluh": "menyampaikan masalah atau ketidakpuasan",
            "memuji": "menyampaikan kepuasan atau terima kasih",
            "berbagi_info": "memberi kabar, berita, atau cerita",
            "promosi": "menawarkan atau mengiklankan sesuatu",
            "lainnya": "tidak termasuk pilihan di atas",
        },
    },
    "nada": {
        "type": "choice",
        "instructions": "Bagaimana gaya bahasa teks ini?",
        "criteria": {
            "formal": "bahasa baku seperti berita atau surat resmi",
            "santai": "bahasa sehari-hari, gaul, singkatan",
            "kasar": "memaki atau menghina",
        },
    },
    "kejelasan": {
        "type": "score",
        "instructions": "Seberapa jelas dan bermakna teks ini?",
        "criteria": ["acak atau tidak bermakna", "sulit dipahami", "cukup jelas", "sangat jelas"],
    },
    "spam": {
        "type": "noul",
        "instructions": "Apakah teks ini spam, iklan, atau promosi?",
    },
    "sarkasme": {
        "type": "noul",
        "instructions": "Apakah teks ini sarkasme atau sindiran (kata-katanya berlawanan maksud)?",
    },
    "campur_bahasa": {
        "type": "noul",
        "instructions": "Apakah teks ini mencampur bahasa Indonesia dengan bahasa daerah atau "
                        "bahasa Inggris?",
    },
    "kata_kasar": {
        "type": "noul",
        "instructions": "Apakah teks ini memuat kata kasar, makian, atau hinaan?",
    },
    "data_pribadi": {
        "type": "noul",
        "instructions": "Apakah teks memuat data pribadi seperti nomor HP, email, "
                        "atau alamat rumah?",
    },
    "label_cocok": {
        "type": "noul",
        # jawaban ya/tidak kurang tepat; urutkan kolom label_cocok__nilai dari terkecil
        "instructions": "Apakah label ini benar menggambarkan isi teks?",
        "with_label": True,
    },
    # ---- lensa untuk mode baris (contoh data properti) ----
    "jenis_properti": {
        "type": "choice",
        "instructions": "Jenis properti apa yang diiklankan?",
        "criteria": {
            "rumah": "rumah tinggal",
            "apartemen": "unit apartemen",
            "kost": "kamar kost atau kontrakan bulanan",
            "tanah": "tanah kosong atau kavling",
            "komersial": "ruko, kantor, gudang",
        },
    },
    "kelengkapan_alamat": {
        "type": "score",
        "instructions": "Seberapa lengkap alamatnya?",
        "criteria": ["tidak ada lokasi jelas", "hanya kota", "kecamatan dan kota",
                     "jalan, nomor, dan kota lengkap"],
    },
    "deskripsi_jelas": {
        "type": "score",
        "instructions": "Seberapa jelas deskripsi iklan ini menggambarkan propertinya?",
        "criteria": ["kosong atau tidak nyambung", "sangat singkat", "cukup", "rinci dan jelas"],
    },
}
