"""Buat data latihan preprocessing SINTETIS (kolom kotor + kunci jawaban hasil bersih).

Pakai: python buat_data_kotor.py
Hasil: data_latihan/preprocessing/
- kotor_SINTETIS.csv        kolom mentah yang harus dibersihkan
- kotor_SINTETIS_kunci.csv  nilai bersih yang benar (untuk mengukur ketepatan pembersihan)
- kamus_slang.csv           kamus slang -> kata baku (boleh dipakai dan ditambah saat lomba)
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("Z:/GEMASTIK-DATABASE/data_latihan/preprocessing")
N = 3000
rng = np.random.default_rng(7)
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]
BULAN3 = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
PRODI = ["TI", "SI", "MN", "AK", "ST", "HK"]
KOTA = ["Bandung", "Surabaya", "Medan", "Makassar", "Bogor", "Malang", "Semarang", "Padang"]
JALAN = ["Mawar", "Melati", "Sudirman", "Diponegoro", "Merdeka", "Kenanga", "Ahmad Yani"]
DOMAIN = ["gmail.com", "yahoo.co.id", "ui.ac.id", "itb.ac.id", "outlook.com", "student.ugm.ac.id"]
JURUSAN = {
    "Teknik Informatika": ["Teknik Informatika", "teknik informatika ", "T. Informatika",
                           "Tek. Informatika", "TI", "Tehnik Informatika", "Teknik Imformatika",
                           "TEKNIK INFORMATIKA", "Informatika"],
    "Sistem Informasi": ["Sistem Informasi", "sistem informasi", "SI", "Sist. Informasi",
                         "Sistim Informasi", "Sistem Infromasi", "S. Informasi"],
    "Manajemen": ["Manajemen", "manajemen", "Managemen", "Menejemen", "MANAJEMEN", "Manajeman"],
    "Akuntansi": ["Akuntansi", "akuntansi", "Akutansi", "Akuntasi", "AKUNTANSI", "Akunting"],
    "Statistika": ["Statistika", "statistika", "Statistik", "Statiska", "Stastistika", "STAT"],
    "Ilmu Hukum": ["Ilmu Hukum", "ilmu hukum", "Hukum", "I. Hukum", "Ilmu Hukm", "FH - Hukum"],
}
# kata baku -> bentuk slang (isi kamus_slang.csv)
SLANG = {
    "tidak": ["gak", "ga", "nggak", "ngga", "gk", "tdk"], "banget": ["bgt", "bngt", "bangettt"],
    "tapi": ["tp", "tpi"], "yang": ["yg"], "dengan": ["dgn", "dg"], "sudah": ["udah", "udh", "sdh"],
    "belum": ["blm", "blum"], "karena": ["krn", "karna"], "juga": ["jg"], "saja": ["aja", "aj"],
    "sangat": ["sngt"], "bagus": ["bgs", "bagusss"], "mantap": ["mantul", "mantab", "mantappp"],
    "lama": ["lamaaa", "lma"], "kecewa": ["kcewa", "kecewaaa"], "pelayanan": ["pelayanannya"],
    "terima kasih": ["makasih", "thx", "mksh"], "kalau": ["kalo", "klo"], "lagi": ["lg"],
    "sama": ["sm"], "banyak": ["byk", "bnyk"], "buruk": ["jelek", "jlek"], "cepat": ["cpt", "cepet"],
    "murah": ["murmer"], "mahal": ["mhl"], "recommended": ["rekomen", "recomended", "rekom"],
}


def acak(pilihan):
    return pilihan[rng.integers(len(pilihan))]


def tulis_kode(tahun, prodi, nomor):
    bentuk = [f"{tahun}-{prodi}-{nomor:05d}", f"{tahun}{prodi}{nomor:05d}",
              f"{tahun % 100}/{prodi}/{nomor}", f"{prodi}-{tahun}-{nomor}",
              f"{tahun}.{prodi.lower()}.{nomor:05d}", f"No. {tahun}-{prodi}-{nomor:05d}"]
    return acak(bentuk)


def tulis_hp(digit):
    """digit = '812xxxxxxxx' (tanpa 0 di depan)."""
    a, b, c = digit[:3], digit[3:7], digit[7:]
    bentuk = [f"0{a}-{b}-{c}", f"+62 {a} {b} {c}", f"62{digit}", f"0{digit}",
              f"(+62) {a}-{b}-{c}", f"0{a} {b} {c}", f"+62{digit}", "-"]
    return acak(bentuk)


def tulis_email(nama, domain):
    bentuk = [f"{nama}@{domain}", f" {nama.title()}@{domain.upper()} ", f"{nama}@{domain}",
              f"{nama} at {domain.replace('.', ' dot ')}", f"{nama}@{domain}."]
    return acak(bentuk)


def tulis_alamat(jalan, nomor, rt, rw, kota, kodepos):
    bentuk = [f"Jl. {jalan} No. {nomor} RT {rt:02d}/RW {rw:02d}, Kota {kota} {kodepos}",
              f"jln {jalan.lower()} no{nomor} rt.{rt} rw.{rw} {kota.lower()}, {kodepos}",
              f"Jalan {jalan} {nomor}, RT{rt:03d} RW{rw:03d}, Kab. {kota} {kodepos}",
              f"JL {jalan.upper()} NO {nomor} RT {rt} RW {rw} {kota.upper()} - {kodepos}",
              f"{jalan} No {nomor} Rt {rt} Rw {rw} {kota}"]
    pilih = rng.integers(len(bentuk))
    return bentuk[pilih], (kodepos if pilih < 4 else np.nan)


def koma(angka):
    """70.5 -> '70,5' (desimal gaya Indonesia)."""
    return f"{angka:g}".replace(".", ",")


def titik_ribuan(angka):
    """1500000 -> '1.500.000'."""
    return f"{angka:,.0f}".replace(",", ".")


def tulis_uang(rupiah):
    """Kembalikan (teks, nilai_benar). rupiah kelipatan 100 ribu."""
    juta = rupiah / 1e6
    pilih = rng.integers(10)
    if pilih == 8:
        tengah = max(2, round(juta))
        return f"{tengah - 1}-{tengah + 1} juta", tengah * 1e6
    if pilih == 9:
        return acak(["-", "tidak tahu", "?"]), np.nan
    bentuk = [f"Rp {titik_ribuan(rupiah)}", f"{rupiah:.0f}", f"{koma(juta)} jt",
              f"{juta:g} juta", f"Rp{titik_ribuan(rupiah)},-", f"Rp. {titik_ribuan(rupiah)},00",
              f"{titik_ribuan(rupiah / 1000)}rb", f"± {koma(juta)} jt"]
    return bentuk[pilih], rupiah


def tulis_berat(kg):
    bentuk = [f"{koma(kg)} kg", f"{kg:g}kg", f"{koma(kg)} Kg", f"{titik_ribuan(kg * 1000)} gr",
              f"{kg * 1000:.0f} gram", f"{kg * 10:g} ons", f"{koma(kg / 100)} kwintal",
              f"{koma(kg)}"]
    return acak(bentuk)


def tulis_tanggal(tanggal):
    d, m, y = tanggal.day, tanggal.month, tanggal.year
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][tanggal.weekday()]
    serial = (tanggal - pd.Timestamp("1899-12-30")).days
    bentuk = [f"{d} {BULAN[m - 1]} {y}", f"{d} {BULAN3[m - 1]} {y}", f"{d:02d}/{m:02d}/{y}",
              tanggal.strftime("%Y-%m-%d"), f"{d:02d}-{m:02d}-{y % 100:02d}", str(serial),
              f"{hari}, {d} {BULAN[m - 1]} {y}", f"{d}-{BULAN3[m - 1]}-{y}"]
    return acak(bentuk)


def tulis_lama(bulan):
    """Kembalikan (teks, bulan_benar)."""
    if bulan < 12 and rng.random() < 0.3:
        return acak(["kurang dari 1 tahun", "< 1 thn", "blm setahun"]), 6
    tahun, sisa = divmod(bulan, 12)
    bentuk = [f"{tahun} tahun {sisa} bulan", f"{tahun} thn {sisa} bln", f"{bulan} bulan",
              f"{tahun}th{sisa}bl", f"{bulan} bln"]
    if sisa == 0:
        bentuk = [f"{tahun} tahun", f"{tahun} thn", f"{bulan} bulan"]
    if sisa == 6:
        bentuk.append(f"{tahun},5 tahun")
    return acak(bentuk), bulan


kotor = []
kunci = []
for i in range(1, N + 1):
    tahun, prodi, nomor = int(rng.integers(2018, 2025)), acak(PRODI), int(rng.integers(1, 99999))
    hp = "8" + "".join(rng.choice(list("0123456789"), 10))
    nama = acak(["budi", "siti.aminah", "andi_p", "dewi99", "rizky.f", "nur"])
    domain = acak(DOMAIN)
    rt, rw, kota = int(rng.integers(1, 20)), int(rng.integers(1, 15)), acak(KOTA)
    alamat, kodepos = tulis_alamat(acak(JALAN), int(rng.integers(1, 200)), rt, rw, kota,
                                   int(rng.integers(10000, 99999)))
    rupiah = float(np.clip(np.round(rng.lognormal(15, 0.8) / 1e5), 1, 500) * 1e5)
    harga, harga_benar = tulis_uang(rupiah)
    kg = round(float(rng.uniform(40, 110)), 1)
    tanggal = pd.Timestamp("2019-01-01") + pd.Timedelta(int(rng.integers(0, 2190)), "D")
    lama, lama_benar = tulis_lama(int(rng.integers(1, 121)))
    jurusan = acak(list(JURUSAN))
    teks_hp = tulis_hp(hp)
    kotor.append({"id": i, "kode_peserta": tulis_kode(tahun, prodi, nomor), "no_hp": teks_hp,
                  "email": tulis_email(nama, domain), "alamat": alamat, "harga": harga,
                  "berat": tulis_berat(kg), "tanggal_daftar": tulis_tanggal(tanggal),
                  "lama_kerja": lama, "jurusan": acak(JURUSAN[jurusan])})
    kunci.append({"id": i, "tahun_masuk": tahun, "kode_prodi": prodi, "nomor_urut": nomor,
                  "no_hp": np.nan if teks_hp == "-" else "0" + hp, "email": f"{nama}@{domain}",
                  "kodepos": kodepos, "rt": rt, "rw": rw, "kota": kota, "harga_rp": harga_benar,
                  "berat_kg": kg, "tanggal_daftar": tanggal.strftime("%Y-%m-%d"),
                  "lama_bulan": lama_benar, "jurusan": jurusan})

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(kotor).to_csv(OUT / "kotor_SINTETIS.csv", index=False)
pd.DataFrame(kunci).to_csv(OUT / "kotor_SINTETIS_kunci.csv", index=False)
baris_slang = [{"slang": s, "baku": baku} for baku, varian in SLANG.items() for s in varian]
pd.DataFrame(baris_slang).to_csv(OUT / "kamus_slang.csv", index=False)
print(pd.DataFrame(kotor).head(10).to_string())
