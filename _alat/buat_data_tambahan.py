"""Buat data demo SINTETIS untuk notebook paradigma baru (2026-10-17). Tanpa internet.
Pakai: python buat_data_tambahan.py [langkah ...]   (default: semua). Hasil di data_latihan/."""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]   # folder di atas repo ini
DL = ROOT / "data_latihan"
FONT = Path("C:/Windows/Fonts")
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)


def harga_pangan():
    """Panel harian 34 provinsi x 13 komoditas, gaya Datavidia 2024 (horizon 92 hari, MAPE)."""
    provinsi = ["Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi", "Sumatera Selatan",
                "Bengkulu", "Lampung", "Kep. Bangka Belitung", "Kep. Riau", "DKI Jakarta",
                "Jawa Barat", "Jawa Tengah", "DI Yogyakarta", "Jawa Timur", "Banten", "Bali",
                "NTB", "NTT", "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan",
                "Kalimantan Timur", "Kalimantan Utara", "Sulawesi Utara", "Sulawesi Tengah",
                "Sulawesi Selatan", "Sulawesi Tenggara", "Gorontalo", "Sulawesi Barat", "Maluku",
                "Maluku Utara", "Papua Barat", "Papua"]
    komoditas = {"beras_medium": (12500, 0.02), "beras_premium": (14500, 0.02),
                 "gula_pasir": (16000, 0.03), "minyak_goreng": (17500, 0.03),
                 "tepung_terigu": (12000, 0.02), "kedelai": (11500, 0.04),
                 "daging_sapi": (130000, 0.03), "daging_ayam": (36000, 0.08),
                 "telur_ayam": (28000, 0.07), "bawang_merah": (35000, 0.20),
                 "bawang_putih": (38000, 0.12), "cabai_merah": (45000, 0.35),
                 "cabai_rawit": (50000, 0.40)}
    tanggal = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    lebaran = pd.to_datetime(["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31"])
    hari_ke_lebaran = np.array([min(abs((t - lb).days) for lb in lebaran) for t in tanggal])
    efek_lebaran = np.exp(-hari_ke_lebaran / 12.0)
    musim = np.sin(2 * np.pi * tanggal.dayofyear / 365.25)
    rows = []
    for p_i, prov in enumerate(provinsi):
        jauh = 1.0 + 0.25 * (p_i >= 28) + 0.1 * (p_i in (8, 9, 23)) + rng.normal(0, 0.04)
        for kom, (dasar, vol) in komoditas.items():
            acak = np.cumsum(rng.normal(0, vol / 25, len(tanggal)))
            acak = acak - np.linspace(0, acak[-1], len(tanggal)) * 0.5
            tren = np.linspace(0, rng.uniform(0.03, 0.15), len(tanggal))
            harga = dasar * jauh * np.exp(tren + acak + vol * 0.6 * musim + vol * efek_lebaran)
            harga = np.round(harga * (1 + rng.normal(0, 0.005, len(tanggal))), -1)
            libur = (tanggal.dayofweek >= 5) & (rng.random(len(tanggal)) < 0.6)
            harga = np.where(libur | (rng.random(len(tanggal)) < 0.03), np.nan, harga)
            rows.append(pd.DataFrame({"date": tanggal, "provinsi": prov, "komoditas": kom,
                                      "harga": harga}))
    df = pd.concat(rows)
    folder = DL / "timeseries" / "global"
    folder.mkdir(parents=True, exist_ok=True)
    batas = pd.Timestamp("2024-10-01")
    train = df[df["date"] < batas]
    test = df[df["date"] >= batas].copy()
    test.insert(0, "id", np.arange(len(test)))
    train.to_csv(folder / "harga_pangan_SINTETIS_train.csv", index=False)
    test.drop(columns="harga").to_csv(folder / "harga_pangan_SINTETIS_test.csv", index=False)
    test[["id", "harga"]].to_csv(folder / "harga_pangan_SINTETIS_test_labels.csv", index=False)
    print("harga_pangan", train.shape, test.shape)


def label_noise():
    folder = DL / "tabular_extra" / "label_noise"
    folder.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DL / "tabular_extra" / "multiclass" / "covertype_train.csv")
    kunci = train[["id", "Cover_Type"]].rename(columns={"Cover_Type": "label_asli"})
    kelas = sorted(train["Cover_Type"].unique())
    acak = rng.random(len(train)) < 0.15
    salah = [random.choice([k for k in kelas if k != v]) for v in train["Cover_Type"]]
    train["Cover_Type"] = np.where(acak, salah, train["Cover_Type"])
    train.to_csv(folder / "covertype_noisy_SINTETIS_train.csv", index=False)
    kunci.to_csv(folder / "covertype_noisy_SINTETIS_kunci.csv", index=False)
    print("label_noise", acak.mean())


def ranking():
    """Pencarian produk e-commerce: query = pencarian, dokumen = produk, relevansi 0-3."""
    baris = []
    for q in range(600):
        n = int(rng.integers(20, 41))
        teks = rng.beta(2, 5, n)
        cocok = (rng.random(n) < 0.3 + 0.5 * teks).astype(int)
        harga = rng.lognormal(0, 0.45, n)
        rating = np.clip(rng.normal(4.3, 0.45, n), 1, 5).round(1)
        ulasan = rng.negative_binomial(1, 0.004, n)
        diskon = rng.choice([0, 0, 0, 5, 10, 20, 30, 50], n)
        jarak = rng.exponential(250, n).round(1)
        laten = (3 * teks + 1.2 * cocok + 0.8 * (rating - 4.3) * (ulasan > 20)
                 + 0.25 * np.log1p(ulasan) - 2.5 * np.maximum(harga - 1, 0)
                 + 0.015 * diskon * (harga > 0.8) - 0.001 * jarak + rng.normal(0, 0.35, n))
        for i in range(n):
            baris.append([q, f"P{q:03d}_{i:02d}", teks[i], cocok[i], harga[i], rating[i],
                          ulasan[i], diskon[i], jarak[i], laten[i]])
    df = pd.DataFrame(baris, columns=["query", "produk", "skor_teks", "kategori_cocok",
                                      "harga_per_anggaran", "rating_produk", "jumlah_ulasan",
                                      "diskon_persen", "jarak_km", "laten"])
    df["skor_teks"] = df["skor_teks"].round(3)
    df["harga_per_anggaran"] = df["harga_per_anggaran"].round(3)
    df["relevansi"] = np.digitize(df["laten"], df["laten"].quantile([0.55, 0.82, 0.95]))
    df = df.drop(columns="laten")
    df.insert(0, "id", np.arange(len(df)))
    folder = DL / "tabular_extra" / "ranking"
    folder.mkdir(parents=True, exist_ok=True)
    train = df[df["query"] < 480]
    test = df[df["query"] >= 480]
    train.to_csv(folder / "produk_rank_SINTETIS_train.csv", index=False)
    test.drop(columns="relevansi").to_csv(folder / "produk_rank_SINTETIS_test.csv", index=False)
    test[["id", "query", "relevansi"]].to_csv(folder / "produk_rank_SINTETIS_test_labels.csv",
                                              index=False)
    print("ranking", train.shape, test.shape, df["relevansi"].value_counts().to_dict())


def kota():
    """Ibu kota provinsi (koordinat perkiraan) untuk graf jarak / rute."""
    data = [("Banda Aceh", 5.55, 95.32), ("Medan", 3.59, 98.67), ("Padang", -0.95, 100.35),
            ("Pekanbaru", 0.51, 101.45), ("Jambi", -1.61, 103.61), ("Palembang", -2.98, 104.76),
            ("Bengkulu", -3.80, 102.27), ("Bandar Lampung", -5.43, 105.26),
            ("Pangkal Pinang", -2.13, 106.11), ("Tanjung Pinang", 0.92, 104.45),
            ("Jakarta", -6.20, 106.85), ("Bandung", -6.92, 107.61), ("Semarang", -6.97, 110.42),
            ("Yogyakarta", -7.80, 110.36), ("Surabaya", -7.25, 112.75), ("Serang", -6.12, 106.15),
            ("Denpasar", -8.65, 115.22), ("Mataram", -8.58, 116.12), ("Kupang", -10.18, 123.61),
            ("Pontianak", -0.03, 109.33), ("Palangka Raya", -2.21, 113.92),
            ("Banjarmasin", -3.32, 114.59), ("Samarinda", -0.50, 117.15),
            ("Tanjung Selor", 2.84, 117.37), ("Manado", 1.47, 124.84), ("Palu", -0.90, 119.87),
            ("Makassar", -5.15, 119.43), ("Kendari", -3.97, 122.51), ("Gorontalo", 0.54, 123.06),
            ("Mamuju", -2.68, 118.89), ("Ambon", -3.70, 128.18), ("Sofifi", 0.74, 127.56),
            ("Manokwari", -0.86, 134.06), ("Jayapura", -2.53, 140.72)]
    pd.DataFrame(data, columns=["kota", "lat", "lon"]).to_csv(DL / "graph" / "kota_indonesia.csv",
                                                              index=False)


def event_audio():
    """Klip 8 detik: noise + 2-4 kata speech_commands di waktu acak. Label = onset/offset/kata."""
    import librosa
    sumber = DL / "audio" / "classification" / "speech_commands_mini"
    daftar = pd.read_csv(sumber / "test.csv").assign(
        label=pd.read_csv(sumber / "test_labels.csv")["label"])
    folder = DL / "audio" / "event" / "klip"
    folder.mkdir(parents=True, exist_ok=True)
    sr = 16000
    rows = []
    for k in range(80):
        y = rng.normal(0, 0.01, sr * 8).astype(np.float32)
        posisi = 0.3
        for _ in range(rng.integers(2, 5)):
            r = daftar.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            kata, _ = librosa.load(sumber / r["path"], sr=sr)
            kata, (a, b) = librosa.effects.trim(kata, top_db=25)
            mulai = posisi + rng.uniform(0.2, 1.0)
            if mulai + len(kata) / sr > 7.8:
                break
            i = int(mulai * sr)
            y[i:i + len(kata)] += kata * rng.uniform(0.6, 1.0)
            rows.append([f"klip/klip{k:03d}.wav", round(mulai, 3), round(mulai + len(kata) / sr, 3),
                         r["label"]])
            posisi = mulai + len(kata) / sr
        soundfile.write(folder / f"klip{k:03d}.wav", y, sr)
    df = pd.DataFrame(rows, columns=["path", "onset", "offset", "label"])
    df.to_csv(DL / "audio" / "event" / "anotasi.csv", index=False)
    print("event_audio", df.shape)


KATA = ("selamat pagi gemastik data mining jakarta bandung surabaya medan makassar harga beras "
        "cabai merah bawang putih minyak goreng gula pasir telur ayam pembayaran tunai kartu "
        "debit nomor faktur pelanggan alamat jalan sudirman nomor rumah kode pos kecamatan "
        "kelurahan provinsi kabupaten kota pendidikan kesehatan layanan publik terima kasih").split()


def font_acak(ukuran):
    nama = random.choice(["arial.ttf", "calibri.ttf", "consola.ttf", "cour.ttf", "times.ttf",
                          "verdana.ttf", "tahoma.ttf", "georgia.ttf"])
    return ImageFont.truetype(str(FONT / nama), ukuran)


def ganggu(im, kuat):
    """Rotasi kecil, blur, noise, kontras rendah. kuat 0..1."""
    im = im.rotate(rng.uniform(-6, 6) * kuat, expand=True, fillcolor=255)
    if kuat > 0.3:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.6) * kuat))
    arr = np.asarray(im).astype(np.float32)
    arr = arr * (1 - 0.45 * kuat) + 255 * 0.45 * kuat * rng.uniform(0.3, 1.0)
    arr = arr + rng.normal(0, 30 * kuat, arr.shape)
    bayangan = np.linspace(0, rng.uniform(0, 70) * kuat, arr.shape[1])[None, :]
    return Image.fromarray(np.clip(arr - bayangan, 0, 255).astype(np.uint8))


def ocr_baris():
    folder = DL / "vision" / "ocr" / "baris"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(240):
        kata = random.sample(KATA, int(rng.integers(2, 5)))
        if rng.random() < 0.4:
            kata.append(f"{rng.integers(1, 31):02d}/{rng.integers(1, 13):02d}/20{rng.integers(20, 26)}")
        if rng.random() < 0.4:
            kata.append(f"Rp{rng.integers(1, 999)}.{rng.integers(0, 999):03d}")
        teks = " ".join(kata)
        font = font_acak(int(rng.integers(22, 40)))
        lebar = int(font.getlength(teks)) + 40
        im = Image.new("L", (lebar, 70), 255)
        ImageDraw.Draw(im).text((20, 15), teks, fill=0, font=font)
        kuat = [0.0, 0.4, 0.8][k % 3]
        ganggu(im, kuat).save(folder / f"baris{k:03d}.png")
        rows.append([f"baris/baris{k:03d}.png", teks, kuat])
    pd.DataFrame(rows, columns=["path", "teks", "gangguan"]).to_csv(
        DL / "vision" / "ocr" / "baris.csv", index=False)
    print("ocr_baris", len(rows))


TOKO = ["TOKO MAKMUR JAYA", "MINIMARKET SEJAHTERA", "WARUNG BU SITI", "SWALAYAN BAROKAH",
        "TOKO SUMBER REZEKI", "KOPERASI MAHASISWA"]
BARANG = [("Beras 5kg", 68000), ("Gula Pasir 1kg", 17500), ("Minyak Goreng 2L", 36000),
          ("Telur 1kg", 29000), ("Mie Instan", 3500), ("Kopi Sachet", 1500), ("Teh Celup", 8500),
          ("Sabun Mandi", 4500), ("Air Mineral", 4000), ("Roti Tawar", 15000)]


def gambar_teks(baris, lebar=520, font_size=22):
    font = font_acak(font_size)
    tinggi = 30 + len(baris) * (font_size + 10)
    im = Image.new("L", (lebar, tinggi), 255)
    draw = ImageDraw.Draw(im)
    for i, teks in enumerate(baris):
        draw.text((20, 15 + i * (font_size + 10)), teks, fill=0, font=font)
    return im


def isi_struk():
    toko = random.choice(TOKO)
    tanggal = f"{rng.integers(1, 29):02d}-{rng.integers(1, 13):02d}-2025"
    baris = [toko, f"Jl. {random.choice(['Merdeka', 'Sudirman', 'Diponegoro'])} No.{rng.integers(1, 99)}",
             f"Tanggal: {tanggal}", f"No: TRX{rng.integers(10000, 99999)}", "-" * 30]
    total = 0
    for nama, harga in random.sample(BARANG, int(rng.integers(2, 6))):
        qty = int(rng.integers(1, 4))
        total += qty * harga
        baris.append(f"{nama} x{qty}  {qty * harga:,}".replace(",", "."))
    baris += ["-" * 30, f"TOTAL  {total:,}".replace(",", "."), "Terima kasih"]
    return baris, {"toko": toko, "tanggal": tanggal, "total": total}


def ocr_struk():
    folder = DL / "vision" / "ocr" / "struk"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(80):
        baris, field = isi_struk()
        ganggu(gambar_teks(baris), rng.uniform(0, 0.5)).save(folder / f"struk{k:03d}.png")
        rows.append({"path": f"struk/struk{k:03d}.png", **field})
    pd.DataFrame(rows).to_csv(DL / "vision" / "ocr" / "struk.csv", index=False)
    print("ocr_struk", len(rows))


def isi_dokumen(jenis):
    nama = random.choice(["Budi Santoso", "Siti Aminah", "Andi Wijaya", "Dewi Lestari"])
    tanggal = f"{rng.integers(1, 29)} {random.choice(['Januari', 'Maret', 'Juli', 'Oktober'])} 2025"
    if jenis == "struk":
        return isi_struk()[0]
    if jenis == "surat_resmi":
        return [f"Nomor: {rng.integers(100, 999)}/UN/{rng.integers(1, 12)}/2025",
                "Perihal: Permohonan Data", f"Yth. Bapak/Ibu {nama}", "di tempat",
                "Dengan hormat, bersama surat ini kami", "mengajukan permohonan data penelitian.",
                "Atas perhatiannya kami ucapkan terima kasih.", "Hormat kami,", "Kepala Bagian"]
    if jenis == "undangan":
        return ["UNDANGAN", f"Kepada Yth. {nama}", "Kami mengundang Bapak/Ibu untuk hadir",
                f"pada acara {random.choice(['Seminar', 'Rapat', 'Syukuran'])} yang diadakan",
                f"Hari/tanggal: {tanggal}", f"Pukul: {rng.integers(8, 20)}.00 WIB",
                "Tempat: Aula Gedung Serbaguna", "Kehadiran Anda sangat kami harapkan."]
    return ["KWITANSI", f"No. {rng.integers(1000, 9999)}", f"Telah terima dari: {nama}",
            f"Uang sejumlah: Rp{rng.integers(1, 99) * 50000:,}".replace(",", "."),
            f"Untuk pembayaran: {random.choice(['sewa', 'iuran', 'servis'])}",
            f"Tanggal {tanggal}", "Penerima,", "( tanda tangan )"]


def ocr_dokumen():
    folder = DL / "vision" / "ocr" / "dokumen"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(200):
        jenis = ["struk", "surat_resmi", "undangan", "kwitansi"][k % 4]
        im = ganggu(gambar_teks(isi_dokumen(jenis), lebar=560, font_size=20), rng.uniform(0, 0.6))
        im.save(folder / f"dok{k:03d}.png")
        rows.append([f"dokumen/dok{k:03d}.png", jenis])
    df = pd.DataFrame(rows, columns=["path", "jenis"]).sample(frac=1, random_state=SEED)
    df.insert(0, "id", np.arange(len(df)))
    df.iloc[:150].to_csv(DL / "vision" / "ocr" / "dokumen_train.csv", index=False)
    df.iloc[150:][["id", "path"]].to_csv(DL / "vision" / "ocr" / "dokumen_test.csv", index=False)
    df.iloc[150:][["id", "jenis"]].to_csv(DL / "vision" / "ocr" / "dokumen_test_labels.csv",
                                          index=False)
    print("ocr_dokumen", len(df))


def ocr_tabel():
    folder = DL / "vision" / "ocr" / "tabel"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    kolom = ["Komoditas", "Satuan", "Harga", "Stok"]
    for k in range(40):
        font = font_acak(20)
        isi = [kolom]
        for nama, harga in random.sample(BARANG, int(rng.integers(3, 7))):
            isi.append([nama, random.choice(["kg", "pcs", "liter"]), f"{harga:,}".replace(",", "."),
                        str(rng.integers(0, 500))])
        lebar_kol = [230, 90, 110, 80]
        im = Image.new("L", (sum(lebar_kol) + 20, 20 + len(isi) * 40), 255)
        draw = ImageDraw.Draw(im)
        for r, baris in enumerate(isi):
            x = 10
            for c, teks in enumerate(baris):
                draw.rectangle([x, 10 + r * 40, x + lebar_kol[c], 50 + r * 40], outline=0)
                draw.text((x + 8, 18 + r * 40), teks, fill=0, font=font)
                rows.append([f"tabel/tabel{k:03d}.png", r, c, teks])
                x += lebar_kol[c]
        ganggu(im, rng.uniform(0, 0.3)).save(folder / f"tabel{k:03d}.png")
    pd.DataFrame(rows, columns=["path", "baris", "kolom", "teks"]).to_csv(
        DL / "vision" / "ocr" / "tabel.csv", index=False)
    print("ocr_tabel", len(rows))


def tulisan_tangan():
    folder = DL / "vision" / "ocr" / "tangan"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(60):
        teks = " ".join(random.sample(KATA, int(rng.integers(1, 4))))
        font = ImageFont.truetype(str(FONT / random.choice(["Inkfree.ttf", "segoesc.ttf",
                                                            "segoepr.ttf"])), 34)
        im = Image.new("L", (int(font.getlength(teks)) + 40, 80), 255)
        ImageDraw.Draw(im).text((20, 10), teks, fill=0, font=font)
        ganggu(im, 0.3).save(folder / f"tangan{k:03d}.png")
        rows.append([f"tangan/tangan{k:03d}.png", teks])
    pd.DataFrame(rows, columns=["path", "teks"]).to_csv(DL / "vision" / "ocr" / "tangan.csv",
                                                        index=False)


def mosaik():
    """2x2 mosaik CIFAR -> klasifikasi gambar multilabel."""
    cifar = DL / "vision" / "classification" / "cifar10_small"
    kelas = sorted(pd.read_csv(cifar / "train.csv")["label"].unique())
    folder = DL / "vision" / "multilabel" / "mosaik"
    folder.mkdir(parents=True, exist_ok=True)
    tabel = {}
    for split, n in [("train", 800), ("test", 200)]:
        src = pd.read_csv(cifar / f"{split}.csv")
        if split == "test":
            src["label"] = pd.read_csv(cifar / "test_labels.csv")["label"]
        rows = []
        for k in range(n):
            pilih = src.sample(4, random_state=int(rng.integers(1e9)))
            kanvas = Image.new("RGB", (64, 64))
            for j, r in enumerate(pilih.itertuples()):
                kanvas.paste(Image.open(cifar / r.path).convert("RGB"), ((j % 2) * 32, (j // 2) * 32))
            kanvas.save(folder / f"{split}{k:04d}.png")
            rows.append([f"mosaik/{split}{k:04d}.png"] + [int(c in set(pilih["label"])) for c in kelas])
        tabel[split] = pd.DataFrame(rows, columns=["path"] + kelas)
        tabel[split].insert(0, "id", np.arange(n) + (0 if split == "train" else 10000))
    tabel["train"].to_csv(DL / "vision" / "multilabel" / "train.csv", index=False)
    tabel["test"][["id", "path"]].to_csv(DL / "vision" / "multilabel" / "test.csv", index=False)
    tabel["test"].drop(columns="path").to_csv(DL / "vision" / "multilabel" / "test_labels.csv",
                                              index=False)
    print("mosaik", tabel["train"].shape)


def pennfudan_hitung():
    folder = DL / "vision" / "detection" / "PennFudanPed"
    rows = []
    for path in sorted((folder / "PedMasks").glob("*.png")):
        mask = np.asarray(Image.open(path))
        nama = path.name.replace("_mask", "")
        rows.append([f"PNGImages/{nama}", len(np.unique(mask)) - 1])
    df = pd.DataFrame(rows, columns=["path", "jumlah"])
    df.insert(0, "id", np.arange(len(df)))
    folder_out = DL / "vision" / "regresi"
    folder_out.mkdir(parents=True, exist_ok=True)
    df.iloc[:120].to_csv(folder_out / "pennfudan_train.csv", index=False)
    df.iloc[120:][["id", "path"]].to_csv(folder_out / "pennfudan_test.csv", index=False)
    df.iloc[120:][["id", "jumlah"]].to_csv(folder_out / "pennfudan_test_labels.csv", index=False)
    print("pennfudan_hitung", df["jumlah"].value_counts().to_dict())


def video():
    """Video 24 frame 96x96: bola bergerak. Kelas: kanan, kiri, atas, bawah, membesar, diam."""
    folder = DL / "video" / "gerak"
    folder.mkdir(parents=True, exist_ok=True)
    kelas = ["kanan", "kiri", "atas", "bawah", "membesar", "diam"]
    rows = []
    for k in range(240):
        nama = kelas[k % len(kelas)]
        latar = rng.integers(0, 120, (96, 96, 3)).astype(np.uint8)
        latar = cv2.GaussianBlur(latar, (0, 0), 6)
        warna = tuple(int(v) for v in rng.integers(130, 256, 3))
        x, y = rng.uniform(25, 70, 2)
        r = rng.uniform(6, 12)
        cepat = rng.uniform(1.0, 2.0)
        tulis = cv2.VideoWriter(str(folder / f"vid{k:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                12, (96, 96))
        for t in range(24):
            dx = {"kanan": cepat, "kiri": -cepat}.get(nama, 0.0) * (t - 12)
            dy = {"bawah": cepat, "atas": -cepat}.get(nama, 0.0) * (t - 12)
            rr = r * (1 + 0.06 * t) if nama == "membesar" else r
            frame = latar.copy()
            cv2.circle(frame, (int(x + dx), int(y + dy)), int(rr), warna, -1)
            frame = np.clip(frame + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
            tulis.write(frame)
        tulis.release()
        rows.append([f"gerak/vid{k:03d}.mp4", nama])
    df = pd.DataFrame(rows, columns=["path", "label"]).sample(frac=1, random_state=SEED)
    df.insert(0, "id", np.arange(len(df)))
    df.iloc[:180].to_csv(DL / "video" / "train.csv", index=False)
    df.iloc[180:][["id", "path"]].to_csv(DL / "video" / "test.csv", index=False)
    df.iloc[180:][["id", "label"]].to_csv(DL / "video" / "test_labels.csv", index=False)
    print("video", len(df))


def gambar_tabular():
    """CIFAR + fitur tabular berisik (berat, panjang, kecepatan) -> kelas. SINTETIS."""
    cifar = DL / "vision" / "classification" / "cifar10_small"
    profil = {"airplane": (40000, 30, 800), "automobile": (1200, 4.5, 180), "bird": (0.5, 0.3, 50),
              "cat": (4, 0.5, 40), "deer": (90, 1.8, 70), "dog": (20, 0.9, 45),
              "frog": (0.2, 0.1, 10), "horse": (500, 2.4, 70), "ship": (90000, 150, 40),
              "truck": (9000, 9, 120)}
    tabel = {}
    for split in ["train", "test"]:
        src = pd.read_csv(cifar / f"{split}.csv")
        if split == "test":
            src["label"] = pd.read_csv(cifar / "test_labels.csv")["label"]
        df = pd.DataFrame()
        df["id"] = np.arange(len(src)) + (0 if split == "train" else 10000)
        df["path"] = "../vision/classification/cifar10_small/" + src["path"]
        for j, kolom in enumerate(["berat_kg", "panjang_m", "kecepatan_kmj"]):
            dasar = src["label"].map(lambda c: profil[c][j]).to_numpy()
            df[kolom] = np.round(dasar * np.exp(rng.normal(0, 0.9, len(src))), 2)
        df["warna_dominan"] = rng.choice(["merah", "biru", "hijau", "abu"], len(src))
        df["label"] = src["label"]
        tabel[split] = df
    folder = DL / "multimodal" / "gambar_tabular"
    folder.mkdir(parents=True, exist_ok=True)
    tabel["train"].to_csv(folder / "train.csv", index=False)
    tabel["test"].drop(columns="label").to_csv(folder / "test.csv", index=False)
    tabel["test"][["id", "label"]].to_csv(folder / "test_labels.csv", index=False)
    print("gambar_tabular", tabel["train"].shape)


if __name__ == "__main__":
    langkah = sys.argv[1:] or ["harga_pangan", "label_noise", "ranking", "kota", "event_audio",
                               "ocr_baris", "ocr_struk", "ocr_dokumen", "ocr_tabel",
                               "tulisan_tangan", "mosaik", "pennfudan_hitung", "video",
                               "gambar_tabular"]
    for nama in langkah:
        globals()[nama]()
