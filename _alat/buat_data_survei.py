"""Buat data latihan survei UKT SINTETIS (kotor, gaya soal final GEMASTIK 2024).

Pakai: python buat_data_survei.py
Hasil: data_latihan/tabular_extra/survei_kotor/ukt_SINTETIS_{train,test,test_labels}.csv
Target `kelas_ukt` 0-7 (ordinal), kelas 1 langka. Isi kotor yang sengaja dibuat:
angka sebagai teks ("Rp 1.500.000", "1,5 jt"), varian string ("S-1", "s1 ", "-"),
kolom teks bebas `pekerjaan_ayah`, missing, dan beberapa outlier salah ketik.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data_latihan" / "tabular_extra" / "survei_kotor"
N_TRAIN = 30380
N_TEST = 7595
rng = np.random.default_rng(42)
n = N_TRAIN + N_TEST
z = rng.normal(size=n)                      # tingkat ekonomi tersembunyi


def pilih_per_tingkat(nilai, varian, z_baris, sebar):
    """Pilih kategori ke-k (urut naik) berdasar z + derau, lalu pilih salah satu variannya."""
    posisi = (z_baris + rng.normal(scale=sebar, size=len(z_baris)) + 2.5) / 5 * len(nilai)
    idx = np.clip(posisi.astype(int), 0, len(nilai) - 1)
    teks = [rng.choice(varian[nilai[i]]) for i in idx]
    return np.array(teks, dtype=object), idx


def tulis_uang(rupiah):
    if np.isnan(rupiah):
        return np.nan
    if rupiah == 0:
        return rng.choice(["0", "-", "Tidak ada", "0"])
    juta = rupiah / 1e6
    bentuk = rng.integers(5)
    if bentuk == 0:
        return "Rp " + f"{rupiah:,.0f}".replace(",", ".")
    if bentuk == 1:
        return f"{rupiah:.0f}"
    if bentuk == 2:
        return f"{juta:.1f} jt".replace(".", ",")
    if bentuk == 3:
        return f"{juta:.1f} juta"
    return "Rp" + f"{rupiah:,.0f}".replace(",", ".") + ",-"


# --- pekerjaan ayah: teks bebas, urut dari skor ekonomi rendah ke tinggi
PEKERJAAN = {
    "tidak bekerja": ["tidak bekerja", "Tidak Bekerja", "-", "belum bekerja"],
    "buruh": ["buruh tani", "Buruh Tani", "buruh harian lepas", "buruh bangunan", "kuli angkut"],
    "petani": ["petani", "Petani", "petani sawah", "petani karet", "nelayan", "Nelayan"],
    "informal": ["ojek online", "ojol", "pedagang kecil", "wiraswasta warung", "pedagang keliling"],
    "karyawan": ["karyawan swasta", "Karyawan Swasta", "guru honorer", "sopir", "satpam"],
    "pns_rendah": ["PNS gol II", "pns golongan 2", "honorer pemda", "perangkat desa"],
    "pns": ["PNS gol III", "pns golongan III", "guru PNS", "TNI", "Polri", "pensiunan PNS"],
    "profesional": ["dokter", "Dokter Spesialis", "dosen", "notaris", "pengusaha", "Direktur"],
}
pekerjaan, skor_kerja = pilih_per_tingkat(list(PEKERJAAN), PEKERJAAN, z, 0.9)

PENDIDIKAN = {
    "SD": ["SD", "sd", "Tidak Tamat SD", "SD "], "SMP": ["SMP", "smp", "SLTP"],
    "SMA": ["SMA", "SMA/SMK", "SLTA", "sma ", "SMK"], "D3": ["D3", "D-3", "Diploma"],
    "S1": ["S1", "s1", "S-1", "S 1", "Sarjana"], "S2": ["S2", "S-2", "Magister"],
    "S3": ["S3", "S-3", "S- 3", "Doktor"],
}
pendidikan, _ = pilih_per_tingkat(list(PENDIDIKAN), PENDIDIKAN, z, 1.2)

RUMAH = {
    "menumpang": ["Menumpang", "menumpang", "numpang orang tua"],
    "sewa": ["Sewa", "Kontrak", "kontrak", "SEWA"],
    "milik": ["Milik Sendiri", "milik sendiri", "MILIK SENDIRI", "Milik sendiri "],
}
rumah, skor_rumah = pilih_per_tingkat(list(RUMAH), RUMAH, z, 1.5)

LISTRIK = {"450": ["450", "450 VA", "450va"], "900": ["900", "900 VA", "900va"],
           "1300": ["1300", "1.300 VA", "1300 VA"], "2200": ["2200", "2.200 VA", "2200VA"],
           "3500": ["3500", "3.500 VA"]}
listrik, skor_listrik = pilih_per_tingkat(list(LISTRIK), LISTRIK, z, 1.2)

# --- angka
pendapatan_ayah = np.exp(14.6 + 0.55 * z + 0.25 * skor_kerja + rng.normal(scale=0.45, size=n))
pendapatan_ayah[skor_kerja == 0] = 0
pendapatan_ibu = np.exp(14.0 + 0.5 * z + rng.normal(scale=0.5, size=n))
pendapatan_ibu[rng.random(n) < 0.45] = 0
salah_ketik = rng.random(n) < 0.004
pendapatan_ayah[salah_ketik] *= 100         # outlier: kelebihan dua nol
tanggungan = rng.poisson(2.0, size=n) + 1
kendaraan = rng.poisson(np.exp(0.2 + 0.45 * z)).astype(float)

# --- kategori lain
KOTA = ["Jakarta", "Bandung", "Surabaya", "Medan", "Makassar", "Semarang", "Yogyakarta",
        "Malang", "Palembang", "Denpasar", "Padang", "Pontianak", "Banjarmasin", "Manado",
        "Kupang", "Mataram", "Ambon", "Jayapura", "Kab. Garut", "Kab. Bogor", "Kab. Sleman",
        "Kab. Jember", "Kab. Banyumas", "Kab. Lombok Timur", "Kab. Deli Serdang"]
efek_kota = rng.normal(scale=0.25, size=len(KOTA))
kota_idx = rng.integers(len(KOTA), size=n)
kota = np.array(KOTA, dtype=object)[kota_idx]
kota = np.where(rng.random(n) < 0.15, [k.upper() for k in kota], kota)

PRODI = ["Kedokteran", "Farmasi", "Teknik Informatika", "Teknik Sipil", "Hukum", "Manajemen",
         "Akuntansi", "Pendidikan Matematika", "Pendidikan Bahasa Indonesia", "Agroteknologi",
         "Ilmu Komunikasi", "Psikologi", "Sastra Inggris", "Biologi", "Statistika"]
efek_prodi = np.linspace(0.8, -0.4, len(PRODI))     # prodi mahal punya UKT lebih tinggi
prodi_idx = rng.integers(len(PRODI), size=n)
jalur = rng.choice(["SNBP", "SNBT", "Mandiri"], size=n, p=[0.35, 0.4, 0.25])

kip = rng.random(n) < 0.06 + 0.10 * (z < -0.8)
beasiswa = np.where(kip, rng.choice(["KIP-K", "KIP", "kip-k", "KIP Kuliah"], size=n),
                    rng.choice(["Tidak", "Tidak Ada", "-", "tidak", "Belum Pernah"], size=n))

# --- target: skor tersembunyi -> 8 kelas ordinal (kelas 1 langka)
skor = (0.9 * np.log1p(pendapatan_ayah + pendapatan_ibu) / 3 + 0.35 * skor_kerja
        + 0.25 * skor_listrik + 0.3 * skor_rumah + 0.2 * kendaraan - 0.15 * tanggungan
        + efek_prodi[prodi_idx] + efek_kota[kota_idx] + 0.3 * (jalur == "Mandiri")
        + rng.normal(scale=0.55, size=n))
batas = np.quantile(skor, np.cumsum([0.11, 0.02, 0.15, 0.18, 0.18, 0.15, 0.11]))
kelas = np.searchsorted(batas, skor)
kelas[kip & (rng.random(n) < 0.9)] = 0

df = pd.DataFrame()
df["id"] = np.arange(1, n + 1)
df["penghasilan_ayah"] = [tulis_uang(v) for v in pendapatan_ayah]
df["penghasilan_ibu"] = [tulis_uang(v) for v in pendapatan_ibu]
df["pekerjaan_ayah"] = pekerjaan
df["pendidikan_ayah"] = pendidikan
df["jumlah_tanggungan"] = [f"{t} orang" if rng.random() < 0.15 else str(t) for t in tanggungan]
df["status_rumah"] = rumah
df["daya_listrik"] = listrik
df["jumlah_kendaraan"] = kendaraan
df["kota"] = kota
df["program_studi"] = np.array(PRODI, dtype=object)[prodi_idx]
df["jalur_masuk"] = jalur
df["riwayat_beasiswa"] = beasiswa
for col, persen in [("penghasilan_ibu", 0.04), ("jumlah_tanggungan", 0.08),
                    ("pendidikan_ayah", 0.05), ("jumlah_kendaraan", 0.05), ("daya_listrik", 0.03)]:
    df.loc[rng.random(n) < persen, col] = np.nan
df["kelas_ukt"] = kelas

OUT.mkdir(parents=True, exist_ok=True)
train = df.iloc[:N_TRAIN]
test = df.iloc[N_TRAIN:]
train.to_csv(OUT / "ukt_SINTETIS_train.csv", index=False)
test.drop(columns="kelas_ukt").to_csv(OUT / "ukt_SINTETIS_test.csv", index=False)
test[["id", "kelas_ukt"]].to_csv(OUT / "ukt_SINTETIS_test_labels.csv", index=False)
print(train.shape, test.shape)
print(train["kelas_ukt"].value_counts().sort_index().to_dict())
