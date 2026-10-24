"""Siapkan data demo vision yang "kotor" (mirip soal lomba), menggantikan data mainan.
Pakai: python unduh_data_vision_chaos.py vignette [pets funsd cord]
Mentah diunduh ke _unduhan_mentah/vision_chaos (di folder atas repo), hasil ke data_latihan/vision.
Sumber dan perubahan dicatat di data_latihan/SOURCES_vision.md."""
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # folder di atas repo
DL = (ROOT / "data_latihan").resolve() / "vision"     # resolve: data_latihan boleh berupa junction
MENTAH = DL.parent.parent / "_unduhan_mentah" / "vision_chaos"
MENTAH.mkdir(parents=True, exist_ok=True)
SEED = 42

# data soal latihan vignette (train = ekspor Roboflow COCO, test = foto tanpa anotasi)
VIGNETTE_SOURCE = ROOT / "data"


def unduh(url, nama):
    path = MENTAH / nama
    if not path.exists():
        print("unduh", url)
        urllib.request.urlretrieve(url, path)
    return path


def vignette():
    """Salin apa adanya: kotak ganda, kelas induk 'Images' tak terpakai, test tanpa kotak."""
    target = DL / "vignette"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ["train", "test"]:
        shutil.copytree(VIGNETTE_SOURCE / name, target / name)
    for name in ["train.csv", "test.csv", "sample_submission.csv"]:
        shutil.copy(VIGNETTE_SOURCE / name, target / name)
    print("vignette:", len(list((target / "train").glob("*.jpg"))), "train,",
          len(list((target / "test").glob("*.jpg"))), "test")


# 10 ras, jumlah sengaja tidak seimbang (200 → 30). File aneh (GIF/PNG berekstensi .jpg,
# grayscale, RGBA) dari ras ini selalu ikut.
PETS_COUNTS = {"Abyssinian": 200, "beagle": 180, "pug": 160, "Persian": 140, "Siamese": 110,
               "shiba_inu": 90, "Bengal": 70, "yorkshire_terrier": 55, "Egyptian_Mau": 40,
               "staffordshire_bull_terrier": 30}
PETS_ODD = ["Abyssinian_5", "Egyptian_Mau_139", "Egyptian_Mau_167", "Egyptian_Mau_177",
            "Egyptian_Mau_186", "Egyptian_Mau_191", "staffordshire_bull_terrier_2"]
# sisanya selalu masuk tanpa_label (untuk anotasi_deteksi): GIF, RGBA, grayscale
PETS_ODD_UNLABELED = ["Abyssinian_34", "Egyptian_Mau_14", "Egyptian_Mau_145", "Egyptian_Mau_129",
                      "staffordshire_bull_terrier_22"]
PETS_UNLABELED = 300


def pets():
    """Oxford-IIIT Pet: ukuran/rasio beragam, format campur, kotak kepala VOC hanya sebagian."""
    import tarfile

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    base = "https://thor.robots.ox.ac.uk/datasets/pets/"
    raw = MENTAH / "pets"
    if not (raw / "images").exists():
        for nama in ["images", "annotations"]:
            with tarfile.open(unduh(base + f"{nama}.tar.gz", f"pets_{nama}.tar.gz")) as tar:
                tar.extractall(raw)
    rng = np.random.default_rng(SEED)
    rows, unlabeled = [], []
    # dari folder gambar, bukan list.txt: list.txt melewatkan file GIF/grayscale yang justru dicari
    for path in sorted((raw / "images").glob("*.jpg")):
        name = path.stem
        breed = name.rsplit("_", 1)[0]
        if breed in PETS_COUNTS:
            rows.append([name, breed, "kucing" if breed[0].isupper() else "anjing"])
    table = pd.DataFrame(rows, columns=["name", "label", "spesies"])
    chosen = []
    for breed, count in PETS_COUNTS.items():
        names = table.loc[table["label"] == breed, "name"].tolist()
        odd = [n for n in names if n in PETS_ODD]
        rest = [n for n in rng.permutation(names) if n not in PETS_ODD + PETS_ODD_UNLABELED]
        chosen += odd + rest[:count - len(odd)]
        unlabeled += rest[count - len(odd):]
    table = table[table["name"].isin(chosen)].reset_index(drop=True)
    table["id"] = table["name"] + ".jpg"
    train, test = train_test_split(table, test_size=0.25, random_state=SEED,
                                   stratify=table["label"])

    target = DL / "pets"
    if target.exists():
        shutil.rmtree(target)
    for folder in ["train", "test", "tanpa_label", "kepala_voc", "trimap"]:
        (target / folder).mkdir(parents=True)
    for name in train["name"]:
        shutil.copy(raw / "images" / f"{name}.jpg", target / "train" / f"{name}.jpg")
    # nama file test disamarkan (nama asli memuat ras = bocoran label)
    test = test.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test["id"] = [f"test_{number:04d}.jpg" for number in range(1, len(test) + 1)]
    for name, new_id in zip(test["name"], test["id"]):
        shutil.copy(raw / "images" / f"{name}.jpg", target / "test" / new_id)
    for name in train["name"]:
        xml = raw / "annotations" / "xmls" / f"{name}.xml"
        if xml.exists():
            shutil.copy(xml, target / "kepala_voc" / xml.name)
        shutil.copy(raw / "annotations" / "trimaps" / f"{name}.png", target / "trimap" / f"{name}.png")
    unlabeled = PETS_ODD_UNLABELED + list(rng.permutation(unlabeled))
    for name in unlabeled[:PETS_UNLABELED]:
        shutil.copy(raw / "images" / f"{name}.jpg", target / "tanpa_label" / f"{name}.jpg")
    train[["id", "label", "spesies"]].to_csv(target / "train.csv", index=False)
    test[["id"]].to_csv(target / "test.csv", index=False)
    test[["id", "label", "spesies"]].to_csv(target / "test_labels.csv", index=False)
    sample = test[["id"]].copy()
    sample["label"] = train["label"].iloc[0]
    sample.to_csv(target / "sample_submission.csv", index=False)
    print("pets:", len(train), "train,", len(test), "test,",
          len(list((target / "kepala_voc").iterdir())), "xml kepala,",
          len(list((target / "tanpa_label").iterdir())), "tanpa label")
    pets_pasangan()


PETS_PAIRS = {"train": 2000, "test": 600}


def pets_pasangan():
    """Pasangan gambar pets: mirip = ras sama. Separuh pasangan negatif = spesies sama (sulit)."""
    import numpy as np
    import pandas as pd

    target = DL / "pets"
    rng = np.random.default_rng(SEED)
    tables = {"train": pd.read_csv(target / "train.csv"), "test": pd.read_csv(target / "test_labels.csv")}
    for part, count in PETS_PAIRS.items():
        table = tables[part]
        rows = []
        while len(rows) < count:
            a = table.iloc[rng.integers(len(table))]
            kind = rng.integers(3)          # 0 = ras sama, 1 = spesies sama ras beda, 2 = acak beda ras
            if kind == 0:
                pool = table[(table["label"] == a["label"]) & (table["id"] != a["id"])]
            elif kind == 1:
                pool = table[(table["spesies"] == a["spesies"]) & (table["label"] != a["label"])]
            else:
                pool = table[table["label"] != a["label"]]
            b = pool.iloc[rng.integers(len(pool))]
            rows.append([f"{part}/{a['id']}", f"{part}/{b['id']}", int(kind == 0)])
        pairs = pd.DataFrame(rows, columns=["gambar_a", "gambar_b", "mirip"])
        # buat 50:50 (kind 0 hanya 1/3): tambah positif sampai seimbang
        while pairs["mirip"].sum() < count // 2:
            a = table.iloc[rng.integers(len(table))]
            pool = table[(table["label"] == a["label"]) & (table["id"] != a["id"])]
            b = pool.iloc[rng.integers(len(pool))]
            negative = pairs.index[pairs["mirip"] == 0][0]
            pairs.loc[negative] = [f"{part}/{a['id']}", f"{part}/{b['id']}", 1]
        pairs = pairs.sample(frac=1, random_state=SEED).reset_index(drop=True)
        pairs.insert(0, "id", [f"p{part[:2]}_{number:04d}" for number in range(len(pairs))])
        if part == "train":
            pairs.to_csv(target / "pasangan_train.csv", index=False)
        else:
            pairs[["id", "gambar_a", "gambar_b"]].to_csv(target / "pasangan_test.csv", index=False)
            pairs[["id", "mirip"]].to_csv(target / "pasangan_test_labels.csv", index=False)
        print("pasangan", part, len(pairs), "| mirip:", pairs["mirip"].sum())


TUGAS = {"vignette": vignette, "pets": pets, "pets_pasangan": pets_pasangan}

if __name__ == "__main__":
    for nama in sys.argv[1:] or list(TUGAS):
        TUGAS[nama]()
