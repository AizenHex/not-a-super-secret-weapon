"""Unduh + rapikan data demo untuk notebook paradigma baru (2026-10-17).
Pakai: python unduh_data_tambahan.py   (butuh internet; hasil di data_latihan/, mentah di MENTAH)."""
import ast
import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_kddcup99, fetch_openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]   # folder di atas repo ini
DL = ROOT / "data_latihan"
MENTAH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_unduhan_mentah"
MENTAH.mkdir(parents=True, exist_ok=True)
SEED = 42
INDONLU = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/"
HF = "https://huggingface.co/api/datasets/"


def unduh(url, nama):
    path = MENTAH / nama
    if not path.exists():
        print("unduh", url)
        urllib.request.urlretrieve(url, path)
    return path


def simpan_split(df, folder, nama, target, test_size=0.25, stratify=True):
    """train (dengan target), test (tanpa target), test_labels (id + target)."""
    folder.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))
    strat = df[target] if stratify else None
    train, test = train_test_split(df, test_size=test_size, random_state=SEED, stratify=strat)
    train.to_csv(folder / f"{nama}_train.csv", index=False)
    test.drop(columns=target).to_csv(folder / f"{nama}_test.csv", index=False)
    test[["id"] + ([target] if isinstance(target, str) else list(target))].to_csv(
        folder / f"{nama}_test_labels.csv", index=False)
    print(folder / nama, train.shape, test.shape)


def ids():
    kdd = fetch_kddcup99(percent10=True, as_frame=True, data_home=str(MENTAH)).frame
    for col in kdd.columns:
        if kdd[col].dtype == object:
            kdd[col] = kdd[col].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    kdd["labels"] = kdd["labels"].str.rstrip(".")
    kategori = {"normal": "normal", "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
                "smurf": "dos", "teardrop": "dos", "ipsweep": "probe", "nmap": "probe",
                "portsweep": "probe", "satan": "probe"}
    kdd["kategori"] = kdd["labels"].map(kategori).fillna("r2l")
    u2r = ["buffer_overflow", "loadmodule", "perl", "rootkit"]
    kdd.loc[kdd["labels"].isin(u2r), "kategori"] = "u2r"
    kdd = kdd.drop_duplicates()
    langka = kdd[kdd["kategori"].isin(["r2l", "u2r"])]
    umum = kdd[~kdd["kategori"].isin(["r2l", "u2r"])].sample(20000, random_state=SEED)
    data = pd.concat([umum, langka]).sample(frac=1, random_state=SEED)
    data = data.drop(columns="labels")
    simpan_split(data, DL / "tabular_extra" / "ids", "kdd", "kategori")


def ordinal():
    wine = fetch_openml(data_id=40691, as_frame=True, data_home=str(MENTAH)).frame
    wine["quality"] = wine["class"].astype(int)
    simpan_split(wine.drop(columns="class"), DL / "tabular_extra" / "ordinal", "wine_quality",
                 "quality")


def geo():
    cal = fetch_california_housing(as_frame=True, data_home=str(MENTAH)).frame
    cal = cal.sample(8000, random_state=SEED)
    simpan_split(cal, DL / "tabular_extra" / "geo", "california", "MedHouseVal", stratify=False)


def satelit():
    sat = fetch_openml(data_id=182, as_frame=True, data_home=str(MENTAH)).frame
    sat = sat.rename(columns={"class": "kelas"})
    nama = {"1": "tanah_merah", "2": "kapas", "3": "tanah_abu", "4": "tanah_abu_lembap",
            "5": "tanah_vegetasi", "7": "tanah_abu_sangat_lembap"}
    sat["kelas"] = sat["kelas"].astype(str).str.rstrip(".").map(nama)
    simpan_split(sat, DL / "vision" / "satelit", "satimage", "kelas")


def uplift():
    path = unduh("https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
                 "hillstrom.csv.gz")
    df = pd.read_csv(gzip.open(path))
    df.insert(0, "id", np.arange(len(df)))
    (DL / "tabular_extra" / "uplift").mkdir(parents=True, exist_ok=True)
    df.to_csv(DL / "tabular_extra" / "uplift" / "hillstrom.csv", index=False)
    print("hillstrom", df.shape)


def absa():
    folder = DL / "text" / "absa"
    folder.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(unduh(INDONLU + "hoasa_absa-airy/train_preprocess.csv", "hoasa_train.csv"))
    valid = pd.read_csv(unduh(INDONLU + "hoasa_absa-airy/valid_preprocess.csv", "hoasa_valid.csv"))
    test = pd.read_csv(unduh(INDONLU + "hoasa_absa-airy/test_preprocess.csv", "hoasa_test.csv"))
    train = pd.concat([train, valid])
    for nama, df in [("train", train), ("test", test)]:
        df.insert(0, "id", np.arange(len(df)) + (0 if nama == "train" else 100000))
    aspek = [c for c in train.columns if c not in ("id", "review")]
    train.to_csv(folder / "hoasa_train.csv", index=False)
    test[["id", "review"]].to_csv(folder / "hoasa_test.csv", index=False)
    test[["id"] + aspek].to_csv(folder / "hoasa_test_labels.csv", index=False)
    print("hoasa", train.shape, test.shape)


def pasangan():
    folder = DL / "text" / "pair"
    folder.mkdir(parents=True, exist_ok=True)
    bagian = []
    for split in ["train", "valid", "test"]:
        url = INDONLU + f"wrete_entailment-ui/{split}_preprocess.csv"
        bagian.append(pd.read_csv(unduh(url, f"wrete_{split}.csv")))
    train = pd.concat(bagian[:2])
    test = bagian[2]
    train.insert(0, "id", np.arange(len(train)))
    test.insert(0, "id", np.arange(len(test)) + 100000)
    train.to_csv(folder / "wrete_train.csv", index=False)
    test.drop(columns="label").to_csv(folder / "wrete_test.csv", index=False)
    test[["id", "label"]].to_csv(folder / "wrete_test_labels.csv", index=False)
    print("wrete", train.shape, test.shape)


def tweet_inggris():
    url = HF + "cardiffnlp/tweet_eval/parquet/sentiment/{}/0.parquet"
    nama = {0: "negative", 1: "neutral", 2: "positive"}
    train = pd.read_parquet(unduh(url.format("train"), "tweet_sent_train.parquet"))
    test = pd.read_parquet(unduh(url.format("test"), "tweet_sent_test.parquet"))
    data = pd.concat([train.sample(8000, random_state=SEED), test.sample(2000, random_state=SEED)])
    data["label"] = data["label"].map(nama)
    simpan_split(data, DL / "text" / "english", "tweet_sentiment", "label", test_size=0.2)


def yelp():
    url = HF + "Yelp/yelp_review_full/parquet/yelp_review_full/test/0.parquet"
    df = pd.read_parquet(unduh(url, "yelp_test.parquet")).sample(8000, random_state=SEED)
    df["stars"] = df["label"] + 1
    df["text"] = df["text"].str.slice(0, 2000)
    simpan_split(df[["text", "stars"]], DL / "text" / "regression", "yelp", "stars")


def pos():
    folder = DL / "text" / "pos"
    folder.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        path = unduh(INDONLU + f"posp_pos-prosa/{split}_preprocess.txt", f"posp_{split}.txt")
        baris = []
        kalimat = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                kalimat += 1
                continue
            token, tag = line.split("\t")
            baris.append([kalimat, token, tag])
        df = pd.DataFrame(baris, columns=["kalimat_id", "token", "tag"])
        df.to_csv(folder / f"posp_{split}.csv", index=False)
        print("posp", split, df.shape, df["kalimat_id"].nunique())


def nusax():
    bahasa = ["indonesian", "javanese", "sundanese", "balinese", "minangkabau", "buginese",
              "acehnese", "banjarese", "madurese", "ngaju", "toba_batak", "english"]
    bagian = {"train": [], "test": []}
    for nama in bahasa:
        for split in ["train", "valid", "test"]:
            url = f"https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/{nama}/{split}.csv"
            df = pd.read_csv(unduh(url, f"nusax_{nama}_{split}.csv"))
            df["bahasa"] = nama
            bagian["test" if split == "test" else "train"].append(df)
    folder = DL / "text" / "nusax"
    folder.mkdir(parents=True, exist_ok=True)
    train = pd.concat(bagian["train"]).drop(columns="id")
    test = pd.concat(bagian["test"]).drop(columns="id")
    train.insert(0, "id", np.arange(len(train)))
    test.insert(0, "id", np.arange(len(test)) + 100000)
    train.to_csv(folder / "nusax_train.csv", index=False)
    test.drop(columns="label").to_csv(folder / "nusax_test.csv", index=False)
    test[["id", "label"]].to_csv(folder / "nusax_test_labels.csv", index=False)
    print("nusax", train.shape, test.shape)


def facqa():
    folder = DL / "text" / "qa"
    folder.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        df = pd.read_csv(unduh(INDONLU + f"facqa_qa-factoid-itb/{split}_preprocess.csv",
                               f"facqa_{split}.csv"))
        rows = []
        for r in df.itertuples():
            q = ast.literal_eval(r.question)
            p = ast.literal_eval(r.passage)
            lab = ast.literal_eval(r.seq_label)
            jawab = [tok for tok, tag in zip(p, lab) if tag != "O"]
            rows.append([" ".join(q), " ".join(p), " ".join(jawab)])
        out = pd.DataFrame(rows, columns=["pertanyaan", "konteks", "jawaban"])
        out.insert(0, "id", np.arange(len(out)))
        out.to_csv(folder / f"facqa_{split}.csv", index=False)
        print("facqa", split, out.shape)


def arxiv():
    url = HF + "ccdv/arxiv-classification/parquet/no_ref/validation/0.parquet"
    df = pd.read_parquet(unduh(url, "arxiv_valid.parquet"))
    nama = {0: "math.AC", 1: "cs.CV", 2: "cs.AI", 3: "cs.SY", 4: "math.GR", 5: "cs.CE",
            6: "cs.PL", 7: "cs.IT", 8: "cs.DS", 9: "cs.NE", 10: "math.ST"}
    df["label"] = df["label"].map(nama)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 1500)
    folder = DL / "text" / "arxiv"
    folder.mkdir(parents=True, exist_ok=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))
    test = df.iloc[:500]
    train = df.iloc[500:].copy()
    berlabel = train.groupby("label").head(10).index     # ±10 contoh berlabel per kelas
    train["label"] = train["label"].where(train.index.isin(berlabel))
    train.to_csv(folder / "arxiv_train.csv", index=False)
    test.drop(columns="label").to_csv(folder / "arxiv_test.csv", index=False)
    test[["id", "label"]].to_csv(folder / "arxiv_test_labels.csv", index=False)
    print("arxiv", train.shape, train["label"].notna().sum(), test.shape)


def tweet_topik():
    url = HF + "cardiffnlp/tweet_topic_multi/parquet/tweet_topic_multi/{}/0.parquet"
    df = pd.concat([pd.read_parquet(unduh(url.format(s), f"tweet_topic_{s}.parquet"))
                    for s in ["train_all", "test_2021"]])
    df = df[["date", "text", "label_name"]].drop_duplicates("text")
    df["label_name"] = df["label_name"].map(lambda v: "|".join(v))
    (DL / "text" / "topik_waktu").mkdir(parents=True, exist_ok=True)
    df.sort_values("date").to_csv(DL / "text" / "topik_waktu" / "tweet_topic.csv", index=False)
    print("tweet_topic", df.shape)


def cmapss():
    folder = DL / "timeseries" / "rul"
    folder.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"
    kolom = ["unit", "siklus", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
    for nama in ["train_FD001", "test_FD001"]:
        df = pd.read_csv(unduh(base + nama + ".txt", nama + ".txt"), sep=r"\s+", header=None)
        df.columns = kolom
        df.to_csv(folder / f"cmapss_{nama.split('_')[0]}.csv", index=False)
    rul = pd.read_csv(unduh(base + "RUL_FD001.txt", "RUL_FD001.txt"), header=None, names=["RUL"])
    rul.insert(0, "unit", np.arange(1, len(rul) + 1))
    rul.to_csv(folder / "cmapss_test_labels.csv", index=False)
    print("cmapss", rul.shape)


def nile():
    import statsmodels.api as sm
    df = sm.datasets.nile.load_pandas().data.astype(int)
    (DL / "timeseries" / "changepoint").mkdir(parents=True, exist_ok=True)
    df.to_csv(DL / "timeseries" / "changepoint" / "nile.csv", index=False)


def retail():
    path = unduh("https://archive.ics.uci.edu/static/public/352/online+retail.zip", "retail.zip")
    df = pd.read_excel(io.BytesIO(zipfile.ZipFile(path).read("Online Retail.xlsx")))
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    pelanggan = df["CustomerID"].drop_duplicates().sample(1500, random_state=SEED)
    sub = df[df["CustomerID"].isin(pelanggan)]
    sub.to_csv(DL / "unsupervised" / "online_retail.csv", index=False)
    print("online_retail", sub.shape)
    jual = df[df["Quantity"] > 0].copy()
    jual["date"] = jual["InvoiceDate"].dt.normalize()
    harian = jual.groupby(["StockCode", "date"])["Quantity"].sum().reset_index()
    hari_jual = harian.groupby("StockCode").size()
    produk = hari_jual[(hari_jual > 40) & (hari_jual < 120)].sample(30, random_state=SEED).index
    sub = harian[harian["StockCode"].isin(produk)].rename(columns={"StockCode": "item",
                                                                   "Quantity": "qty"})
    (DL / "timeseries" / "intermittent").mkdir(parents=True, exist_ok=True)
    sub.to_csv(DL / "timeseries" / "intermittent" / "retail_harian.csv", index=False)
    print("retail_harian", sub.shape)


def mutag():
    zipfile.ZipFile(DL / "graph" / "mutag.zip").extractall(DL / "graph")
    print("mutag diekstrak")


def ravdess():
    import librosa
    import soundfile
    path = unduh("https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip",
                 "ravdess.zip")
    emosi = {"01": "netral", "02": "tenang", "03": "senang", "04": "sedih", "05": "marah",
             "06": "takut", "07": "jijik", "08": "terkejut"}
    folder = DL / "audio" / "emosi" / "ravdess"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    arsip = zipfile.ZipFile(path)
    for nama in arsip.namelist():
        if not nama.endswith(".wav"):
            continue
        bagian = Path(nama).stem.split("-")
        aktor = int(bagian[6])
        if aktor > 12:
            continue
        y, sr = librosa.load(io.BytesIO(arsip.read(nama)), sr=16000)
        tujuan = folder / Path(nama).name
        soundfile.write(tujuan, y, 16000)
        rows.append([f"ravdess/{tujuan.name}", emosi[bagian[2]], int(bagian[3]), f"aktor{aktor:02d}"])
    df = pd.DataFrame(rows, columns=["path", "emosi", "intensitas", "speaker"])
    df.insert(0, "id", np.arange(len(df)))
    df.to_csv(DL / "audio" / "emosi" / "ravdess.csv", index=False)
    print("ravdess", df.shape)


if __name__ == "__main__":
    langkah = sys.argv[2:] or ["ids", "ordinal", "geo", "satelit", "uplift", "absa", "pasangan",
                               "tweet_inggris", "yelp", "pos", "nusax", "facqa", "arxiv",
                               "tweet_topik", "cmapss", "nile", "retail", "mutag", "ravdess"]
    for nama in langkah:
        globals()[nama]()
