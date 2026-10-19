"""Buat data demo iklan properti (mode baris) dengan kejanggalan yang sengaja disisipkan.
Pakai: python buat_properti.py  -> menulis properti.csv di folder ini."""
import os
import random

import pandas as pd

random.seed(42)
N_ROWS = 3000
HERE = os.path.dirname(os.path.abspath(__file__))

CITIES = ["Bekasi", "Depok", "Bogor", "Tangerang", "Bandung", "Surabaya", "Medan", "Semarang",
          "Yogyakarta", "Makassar", "Malang", "Jakarta Selatan"]
STREETS = ["Melati", "Mawar", "Merdeka", "Sudirman", "Diponegoro", "Ahmad Yani", "Kenanga",
           "Gatot Subroto", "Pahlawan", "Cempaka"]
# jenis: (judul, deskripsi, harga minimum, harga maksimum, luas minimum, luas maksimum)
KINDS = {
    "rumah": (["Rumah minimalis 2 lantai", "Rumah siap huni dekat tol", "Rumah hook asri",
               "Dijual rumah cluster"],
              ["3 kamar tidur, 2 kamar mandi, carport, SHM.", "Lingkungan aman, dekat sekolah.",
               "Bebas banjir, akses mobil 2 arah."], 400e6, 3e9, 60, 250),
    "apartemen": (["Apartemen studio furnished", "Unit apartemen 2BR view kota"],
                  ["Fasilitas kolam renang dan gym.", "Dekat stasiun MRT, siap huni."],
                  300e6, 2e9, 21, 80),
    "kost": (["Kost putri AC wifi", "Kost karyawan dekat kantor", "Kamar kost eksklusif"],
             ["Kamar mandi dalam, listrik token.", "Bulanan, sudah termasuk air dan wifi."],
             800e3, 4e6, 9, 20),
    "tanah": (["Tanah kavling siap bangun", "Tanah pinggir jalan raya"],
              ["SHM, cocok untuk investasi.", "Kontur rata, bebas banjir."], 100e6, 2e9, 100, 1000),
    "komersial": (["Ruko 3 lantai strategis", "Gudang akses kontainer", "Kantor siap pakai"],
                  ["Cocok untuk usaha, parkir luas.", "Pinggir jalan utama, ramai."],
                  800e6, 8e9, 80, 600),
}


def rupiah(value, kind):
    text = f"Rp {int(value):,}".replace(",", ".")
    if kind == "kost":
        text = text + "/bulan"
    return text


def make_row(number):
    kind = random.choice(list(KINDS))
    titles, descriptions, low_price, high_price, low_area, high_area = KINDS[kind]
    city = random.choice(CITIES)
    row = {"id": number, "judul": random.choice(titles), "deskripsi": random.choice(descriptions)}
    row["harga"] = rupiah(random.uniform(low_price, high_price), kind)
    row["alamat"] = f"Jl. {random.choice(STREETS)} No. {random.randint(1, 99)}, {city}"
    row["luas_m2"] = random.randint(low_area, high_area)
    row["jenis_asli"] = kind
    return row


def add_problem(row):
    # 10% baris diberi satu kejanggalan; kolom "masalah" = kunci jawaban demo
    problem = random.choice(["harga_mustahil", "harga_teks", "luas_negatif", "alamat_kabur",
                             "deskripsi_kosong", "salah_jenis"])
    row["masalah"] = problem
    if problem == "harga_mustahil":
        row["harga"] = rupiah(random.uniform(5e6, 30e6), "rumah")
    if problem == "harga_teks":
        row["harga"] = random.choice(["hubungi wa", "nego", "tanya admin", "-"])
    if problem == "luas_negatif":
        row["luas_m2"] = -row["luas_m2"]
    if problem == "alamat_kabur":
        row["alamat"] = random.choice(["dekat kampus", "pusat kota", "lokasi strategis", ""])
    if problem == "deskripsi_kosong":
        row["deskripsi"] = random.choice(["", "-", "info lebih lanjut dm", "aaaa"])
    if problem == "salah_jenis":
        row["jenis_asli"] = random.choice(list(KINDS))
    return row


rows = []
for number in range(N_ROWS):
    row = make_row(number)
    row["masalah"] = ""
    if random.random() < 0.10:
        row = add_problem(row)
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(f"{HERE}/properti.csv", index=False)
print(table.shape)
print(table["masalah"].value_counts())
