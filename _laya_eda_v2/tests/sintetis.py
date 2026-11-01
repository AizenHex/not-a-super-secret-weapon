"""Data SINTETIS yang meniru kekacauan data riil `Golongan UKT` (tabel audit RENCANA.md bagian 1).

Data riil (`../_laya_eda/data/train.csv`, 30.380 × 28) tidak ada di repo. Tabel ini dibuat dari aturan di bawah
dengan seed tetap, jadi uji bisa jalan di mana pun. Angka/porsi meniru audit, BUKAN salinan data riil:
nama kolom yang tidak tercatat di repo (mis. `Jalur Masuk`, `Fakultas`) adalah karangan untuk melengkapi 28 kolom.

Kekacauan yang ditiru: spasi di belakang, singkatan (Kab.), huruf besar/kecil, tanda baca di kode jenjang,
sinonim jenjang, penanda kosong palsu (`-`, `0`), kolom 1 nilai + kosong, isian bebas multi-bahan + salah ketik,
angka penjaga 2147483647 dan variannya, angka mustahil, nol bermakna vs nol kosong, satuan campur, logika antar
kolom, kolom kembar, kategori berurutan, label 0–7 timpang (kelas 1 sangat kecil), kosong tersebar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL = "Golongan UKT"
SENTINELS = [2147483647, 2156483647, 2347483647, 2597528647, 2000000000]
SENTINEL_COLUMNS = ["Pengeluaran Bulanan Keluarga", "Jumlah (Perkiraan Harga Pasar) Kendaraan yang Dimiliki",
                    "Luas Tanah Rumah (m2)", "NJOP per Meter", "Total Hutang", "Cicilan Hutang per Bulan",
                    "Rerata Pengeluaran Listrik & Air per Bulan"]
# porsi kelas 0..7: kelas 1 ±0,4% (riil 125 dari 30.380), kelas 4 terbesar (riil 6.617)
CLASS_SHARE = [0.08, 0.004, 0.10, 0.14, 0.218, 0.17, 0.15, 0.138]
INCOME = ["Kurang dari Rp 1.000.000", "Rp 1.000.000 - Rp 2.500.000", "Rp 2.500.001 - Rp 5.000.000",
          "Rp 5.000.001 - Rp 10.000.000", "Lebih dari Rp 10.000.000"]
CITY = ["Kota Semarang ", "Kota Semarang", "Kab. Semarang", "Kabupaten Semarang ", "Kab. Pati", "Kabupaten Pati",
        "Kota Salatiga", "Kab. Demak", "Kabupaten Kendal", "Kota Surakarta", "Kab. Grobogan", "Kota Magelang"]
EDUCATION = ["SLTA", "SMA", "MA", "SLTA Sederajat", "SD/MI Sederajat", "SMP", "S1", "S-1", "D-1", "D.1", " D1",
             "S-3", "S3 ", "S- 3", "S2", "D3", "Tidak Sekolah"]
EDUCATION_P = [0.13, 0.13, 0.12, 0.12, 0.1, 0.1, 0.08, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.04, 0.05]
# isian bebas: bahan dasar (bobot) + salah ketik jarang + pemisah/huruf/kata pengisi acak → ribuan variasi
ROOF = {"genteng": 0.5, "genting": 0.12, "seng": 0.15, "asbes": 0.12, "genteng beton": 0.03, "dak beton": 0.02,
        "sirap": 0.02, "genteng tanah liat": 0.04}
ROOF_TYPO = {"genteng": ["gentng", "geting", "gentig", "geneng", "gentheng"], "asbes": ["asbas", "esbes"]}
WALL = {"batu bata": 0.45, "bata merah": 0.08, "batako": 0.1, "semen": 0.08, "plester semen": 0.05, "kayu": 0.08,
        "papan": 0.06, "bambu": 0.03, "tembok bata": 0.04, "bata ringan": 0.03}
WALL_TYPO = {"batu bata": ["batu bat", "batu bara", "baru bata"], "semen": ["semn"]}
FLOOR = {"keramik": 0.5, "ubin": 0.12, "tegel": 0.08, "semen": 0.12, "tanah": 0.06, "plester": 0.04, "kayu": 0.04,
         "granit": 0.04}
FLOOR_TYPO = {"keramik": ["kermaik", "keremik", "keramk", "keramaik", "kramik", "dikeramik"],
              "tegel": ["tekhel", "teghel"]}
SEPARATOR = [", ", " & ", " dan ", "/", " + ", ","]
HISTORY = ["-", "Tidak", "Tidak Ada", "KIPK", "kipk", "Bidikmisi", "PIP"]
HISTORY_P = [0.21, 0.35, 0.1, 0.17, 0.04, 0.08, 0.05]
STUDY = ["Pendidikan Jasmani, Kesehatan, dan Rekreasi", "Teknik Informatika", "Akuntansi", "Kedokteran",
         "Ilmu Hukum", "Pendidikan Bahasa Inggris", "Manajemen", "Teknik Sipil", "Farmasi", "Sastra Indonesia",
         "Teknik Elektro", "Teknik Mesin", "Arsitektur", "Ekonomi Pembangunan", "Psikologi", "Ilmu Komunikasi",
         "Kesehatan Masyarakat", "Keperawatan", "Biologi", "Kimia", "Fisika", "Matematika", "Statistika",
         "Pendidikan Guru Sekolah Dasar", "Pendidikan Matematika", "Sejarah", "Geografi", "Agribisnis",
         "Agroteknologi", "Peternakan", "Ilmu Gizi", "Sosiologi", "Ilmu Pemerintahan", "Hubungan Internasional"]
JOB = ["Petani", "Wiraswasta", "PNS", "Buruh", "Karyawan Swasta", "TNI/Polri", "Pedagang", "Tidak Bekerja"]
HOUSE = ["Milik Sendiri", "Sewa", "Menumpang", "Milik Keluarga", "Dinas"]


def _pick(rng, options, share, n):
    share = np.asarray(share, dtype=float)
    return rng.choice(np.asarray(options, dtype=object), size=n, p=share / share.sum())


def _materials(rng, base: dict, typos: dict, filler: str, n: int) -> np.ndarray:
    """Isian bebas multi-bahan: 1–3 bahan, salah ketik ±1,5%, huruf acak, kata pengisi, titik, '-' / '0'."""
    words, weights = list(base), np.array(list(base.values()))
    cells = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.02:
            cells.append("-")
            continue
        if roll < 0.03:
            cells.append("0")
            continue
        count = rng.choice([1, 2, 3], p=[0.8, 0.17, 0.03])
        parts = list(rng.choice(words, size=count, replace=False, p=weights / weights.sum()))
        for i, part in enumerate(parts):
            if part in typos and rng.random() < 0.015:
                parts[i] = rng.choice(typos[part])
        text = parts[0]
        for part in parts[1:]:
            text += rng.choice(SEPARATOR) + part
        style = rng.random()
        text = text.title() if style < 0.3 else (text.upper() if style < 0.36 else
                                                 (text.capitalize() if style < 0.6 else text))
        if rng.random() < 0.05:
            text = filler + " " + text
        if rng.random() < 0.03:
            text += "."
        cells.append(text)
    return np.asarray(cells, dtype=object)


def _money(rng, center, n, spread=0.8):
    return np.round(np.exp(rng.normal(np.log(center), spread, n)), -3)


def buat(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Train sintetis berantakan: n baris × 28 kolom, label `Golongan UKT` 0–7."""
    rng = np.random.default_rng(seed)
    wealth = rng.normal(0, 1, n)                                         # kekayaan tersembunyi → label
    income = np.clip(np.round(wealth * 1.1 + 2 + rng.normal(0, 0.6, n)), 0, 4).astype(int)
    city = _pick(rng, CITY, np.ones(len(CITY)), n)
    typo = rng.random(n) < 0.01                                          # salah ketik kota (jarang)
    city[typo] = rng.choice(["Kota Semarng", "Kabupaten Kendl", "Kota Malang", "Kota Salatga"], typo.sum())
    owner = _pick(rng, HOUSE, [0.6, 0.15, 0.1, 0.12, 0.03], n)
    data = {
        "id": np.arange(1, n + 1),
        "Program Studi": _pick(rng, STUDY, np.ones(len(STUDY)), n),
        "Fakultas": _pick(rng, ["FT", "FE", "FK", "FH", "FIB", "FKIP"], np.ones(6), n),
        "Jalur Masuk": _pick(rng, ["SNBP", "SNBT", "Mandiri"], [0.3, 0.4, 0.3], n),
        "Jenis Kelamin": _pick(rng, ["L", "P"], [0.45, 0.55], n),
        "Kota": city,
        "Lokasi Tempat Tinggal": np.where(rng.random(n) < 0.992, city, _pick(rng, CITY, np.ones(len(CITY)), n)),
        "Pendidikan Orang Tua/Wali": _pick(rng, EDUCATION, EDUCATION_P, n),
        "Pekerjaan Orang Tua/Wali": _pick(rng, JOB, np.ones(len(JOB)), n),
        "Pendapatan Keluarga": np.asarray(INCOME, dtype=object)[income],
        "Pengeluaran Bulanan Keluarga": _money(rng, 3e6, n) * np.exp(0.3 * wealth),
        "Jumlah Tanggungan dalam Keluarga": rng.integers(0, 8, n).astype(float),
        "Status Kepemilikan Tempat Tinggal": owner,
        "Kepemilikan Aset": np.where(rng.random(n) < 0.993, owner, _pick(rng, HOUSE, np.ones(5), n)),
        "Luas Tanah Rumah (m2)": np.round(np.exp(rng.normal(4.8 + 0.3 * wealth, 0.6))),
        "NJOP per Meter": _money(rng, 1.5e6, n) * np.exp(0.4 * wealth),
        "Bahan Atap Rumah": _materials(rng, ROOF, ROOF_TYPO, "atap", n),
        "Bahan Tembok Rumah": _materials(rng, WALL, WALL_TYPO, "tembok", n),
        "Bahan Lantai Rumah": _materials(rng, FLOOR, FLOOR_TYPO, "lantai", n),
        "Daya Listrik": _pick(rng, [450, 900, 1300, 2200, 3500, 5500], [0.15, 0.3, 0.25, 0.15, 0.1, 0.05], n).astype(float),
        "Rerata Pengeluaran Listrik & Air per Bulan": _money(rng, 3e5, n, 0.6),
        "Jumlah (Perkiraan Harga Pasar) Kendaraan yang Dimiliki": _money(rng, 4e7, n) * np.exp(0.5 * wealth),
        "Total Hutang": np.where(rng.random(n) < 0.515, 0, _money(rng, 2e7, n)),
        "Riwayat Beasiswa atau Bantuan Pendidikan": _pick(rng, HISTORY, HISTORY_P, n),
        "Status Kepesertaan Program Bantuan Pemerintah": pd.Series(np.where(rng.random(n) < 0.16, "KIS/JKS/BPJS", "")).replace("", np.nan).to_numpy(object),
    }
    df = pd.DataFrame(data)
    df["Luas Bangunan Rumah (m2)"] = np.round(df["Luas Tanah Rumah (m2)"] * rng.uniform(0.4, 0.95, n))
    df["Cicilan Hutang per Bulan"] = np.round(df["Total Hutang"] / rng.uniform(12, 60, n), -3)
    _mess(df, rng, n)
    latent = wealth + 0.25 * rng.normal(0, 1, n)
    cuts = np.quantile(latent, np.cumsum(CLASS_SHARE)[:-1])
    df[LABEL] = np.searchsorted(cuts, latent)
    order = ["id", "Program Studi", "Fakultas", "Jalur Masuk", "Jenis Kelamin", "Kota", "Lokasi Tempat Tinggal",
             "Pendidikan Orang Tua/Wali", "Pekerjaan Orang Tua/Wali", "Pendapatan Keluarga",
             "Pengeluaran Bulanan Keluarga", "Jumlah Tanggungan dalam Keluarga", "Status Kepemilikan Tempat Tinggal",
             "Kepemilikan Aset", "Luas Tanah Rumah (m2)", "Luas Bangunan Rumah (m2)", "NJOP per Meter",
             "Bahan Atap Rumah", "Bahan Tembok Rumah", "Bahan Lantai Rumah", "Daya Listrik",
             "Rerata Pengeluaran Listrik & Air per Bulan", "Jumlah (Perkiraan Harga Pasar) Kendaraan yang Dimiliki",
             "Total Hutang", "Cicilan Hutang per Bulan", "Riwayat Beasiswa atau Bantuan Pendidikan",
             "Status Kepesertaan Program Bantuan Pemerintah", LABEL]
    return df[order]


def _mess(df: pd.DataFrame, rng, n: int) -> None:
    """Kekacauan angka: penjaga, mustahil, nol, satuan campur, logika antar kolom, kosong tersebar."""
    for column in SENTINEL_COLUMNS[:-1] + ["Rerata Pengeluaran Listrik & Air per Bulan"]:
        hit = rng.random(n) < 0.006
        df.loc[hit, column] = rng.choice(SENTINELS, hit.sum(), p=[0.8, 0.05, 0.05, 0.05, 0.05])
    small = rng.random(n) < 0.16                                         # satuan campur: unit, bukan rupiah
    df.loc[small, "Jumlah (Perkiraan Harga Pasar) Kendaraan yang Dimiliki"] = rng.integers(1, 6, small.sum())
    small = rng.random(n) < 0.16
    df.loc[small, "NJOP per Meter"] = rng.integers(100, 1000, small.sum())
    df.loc[rng.random(n) < 0.1, "NJOP per Meter"] = 0                    # nol mencurigakan (riil 3.023)
    df.loc[rng.random(n) < 0.009, "Daya Listrik"] = 0                   # riil 281 baris
    df.loc[rng.choice(n, 2, replace=False), "Daya Listrik"] = 2_200_000
    df.loc[rng.choice(n, 2, replace=False), "Jumlah Tanggungan dalam Keluarga"] = 10000
    bigger = (rng.random(n) < 0.029) & (df["Luas Tanah Rumah (m2)"] < 1e9)   # bangunan > tanah (riil 875)
    df.loc[bigger, "Luas Bangunan Rumah (m2)"] = df.loc[bigger, "Luas Tanah Rumah (m2)"] * 1.5
    debt = (df["Total Hutang"] > 0) & (rng.random(n) < 0.007)            # cicilan > hutang (riil 101)
    df.loc[debt, "Cicilan Hutang per Bulan"] = df.loc[debt, "Total Hutang"] * 2
    zero = (df["Total Hutang"] == 0) & (rng.random(n) < 0.01)            # hutang 0 tapi cicilan > 0 (riil 145)
    df.loc[zero, "Cicilan Hutang per Bulan"] = 500_000
    for column in df.columns.drop(["id"]):                               # ±0,7% kosong di hampir tiap kolom
        df.loc[rng.random(n) < 0.007, column] = np.nan
    broken = rng.choice(n, max(1, n // 300), replace=False)              # baris kosong >= 5 kolom
    df.loc[broken, list(df.columns[5:11])] = np.nan


def buat_test(train: pd.DataFrame, n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Test sintetis dari pola yang sama (id baru, tanpa label)."""
    test = buat(n, seed).drop(columns=[LABEL])
    test["id"] = np.arange(len(train) + 1, len(train) + n + 1)
    return test


def sample_submission(test: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    """Format lomba 2024: `id,Golongan UKT`, label bilangan bulat, urutan id diacak (bukan urutan test)."""
    ids = test["id"].sample(frac=1, random_state=seed).to_numpy()
    sample = pd.DataFrame()
    sample["id"] = ids
    sample[LABEL] = 0
    return sample


def tulis(folder, n: int = 3000) -> dict:
    """Simpan train.csv, test.csv, sample_submission.csv ke folder; kembalikan path-nya."""
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    train = buat(n)
    test = buat_test(train)
    paths = {"train": folder / "train.csv", "test": folder / "test.csv", "sample": folder / "sample_submission.csv"}
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    sample_submission(test).to_csv(paths["sample"], index=False)
    return paths


if __name__ == "__main__":
    frame = buat(30380)
    print(frame.shape)
    print(frame[LABEL].value_counts().sort_index().to_string())
