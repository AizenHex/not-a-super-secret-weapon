"""Salah tebak: baris yang masih salah ditebak model baseline (TF-IDF + model linear).
Train = tebakan out-of-fold (tiap baris ditebak model yang tidak pernah melihatnya).
Test = model dilatih di seluruh train lalu menebak test. Tanpa streamlit."""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

import features

CONFIDENT = 0.7    # salah dengan peluang tebakan >= ini = "yakin salah" (curiga labelnya salah)
UNSURE = 0.5       # peluang tebakan < ini = model ragu


def text_spec(groups):
    return {"kata": "kata" in groups, "char": "char" in groups, "dense": {}, "te": []}


def train_rows(ctx, max_rows):
    y = ctx["data"][ctx["label_column"]]
    rows = features.pick_rows(y.dropna().index, max_rows, None)
    return features.drop_rare_classes(rows, y)


def own_probability(proba, classes, labels):
    # peluang yang diberikan model untuk label aslinya (label yang tidak dikenal model = 0)
    positions = pd.Series(np.arange(len(classes)), index=classes)
    own = positions.reindex(labels.to_numpy()).to_numpy()
    unknown = np.isnan(own)
    own = np.where(unknown, 0, own).astype(int)
    values = proba[np.arange(len(labels)), own]
    values[unknown] = 0.0
    return values


def mistake_kind(table):
    kinds = []
    for row in table.itertuples():
        if row.benar:
            kinds.append("benar")
        elif row.peluang_tebakan >= CONFIDENT:
            kinds.append("yakin salah (cek label)")
        elif row.peluang_tebakan < UNSURE:
            kinds.append("salah, model ragu")
        else:
            kinds.append("salah")
    return kinds


def class_table(labels, proba, classes):
    table = pd.DataFrame(index=labels.index)
    table["label"] = labels.to_numpy()
    table["tebakan"] = classes[proba.argmax(axis=1)]
    table["peluang_tebakan"] = proba.max(axis=1)
    table["peluang_label"] = own_probability(proba, classes, labels)
    table["benar"] = table["label"] == table["tebakan"]
    table["jenis"] = mistake_kind(table)
    return table


def number_table(labels, guesses):
    # label angka: salah = selisih tebakan; 5% selisih terbesar = meleset jauh
    table = pd.DataFrame(index=labels.index)
    table["label"] = labels.to_numpy()
    table["tebakan"] = guesses
    table["selisih"] = table["tebakan"] - table["label"]
    table["selisih_mutlak"] = table["selisih"].abs()
    far = table["selisih_mutlak"].quantile(0.95)
    table["jenis"] = "dekat"
    table.loc[table["selisih_mutlak"] >= far, "jenis"] = "meleset jauh"
    return table


def train_predictions(ctx, groups, max_rows):
    y = ctx["data"][ctx["label_column"]]
    task = features.task_of(y)
    rows = train_rows(ctx, max_rows)
    X = pd.DataFrame({"teks": ctx["texts"][rows]})
    model = features.make_model(text_spec(groups), task, y[rows].nunique())
    if task == "angka":
        splitter = KFold(features.FOLDS, shuffle=True, random_state=features.SEED)
        return number_table(y[rows], cross_val_predict(model, X, y[rows], cv=splitter))
    splitter = StratifiedKFold(features.FOLDS, shuffle=True, random_state=features.SEED)
    proba = cross_val_predict(model, X, y[rows], cv=splitter, method="predict_proba")
    return class_table(y[rows], proba, np.array(sorted(y[rows].unique())))


def test_predictions(ctx, test_texts, test_labels, groups, max_rows):
    # latih di train (baris sama dengan tabel train), tebak test
    y = ctx["data"][ctx["label_column"]]
    task = features.task_of(y)
    rows = train_rows(ctx, max_rows)
    model = features.make_model(text_spec(groups), task, y[rows].nunique())
    model.fit(pd.DataFrame({"teks": ctx["texts"][rows]}), y[rows])
    X_test = pd.DataFrame({"teks": test_texts})
    if task == "angka" and test_labels is None:
        return pd.DataFrame({"tebakan": model.predict(X_test)}, index=test_texts.index)
    if task == "angka":
        return number_table(test_labels, model.predict(X_test))
    proba = model.predict_proba(X_test)
    classes = model.classes_
    if test_labels is None:
        test_labels = pd.Series([None] * len(test_texts), index=test_texts.index)
        table = class_table(test_labels, proba, classes)
        return table.drop(columns=["label", "peluang_label", "benar", "jenis"])
    return class_table(test_labels, proba, classes)


def class_report(table):
    report = classification_report(table["label"].astype(str), table["tebakan"].astype(str),
                                   output_dict=True, zero_division=0)
    frame = pd.DataFrame(report).T
    frame = frame.rename(columns={"precision": "presisi", "f1-score": "f1",
                                  "support": "jumlah"})
    return frame.round(3)


def swapped_pairs(table, top=10):
    # pasangan label asli -> tebakan yang paling sering tertukar
    wrong = table[~table["benar"]]
    pairs = wrong.groupby(["label", "tebakan"]).size()
    pairs = pairs.sort_values(ascending=False).head(top)
    return pairs.reset_index(name="jumlah")


def error_by_length(table, texts):
    # apakah teks pendek / panjang lebih sering salah? (4 kelompok panjang yang sama banyak)
    lengths = texts[table.index].str.len()
    groups = pd.qcut(lengths, q=4, duplicates="drop")
    wrong = ~table["benar"]
    summary = wrong.groupby(groups, observed=True).agg(["mean", "size"])
    summary = summary.rename(columns={"mean": "persen_salah", "size": "jumlah"})
    summary["persen_salah"] = (100 * summary["persen_salah"]).round(1)
    summary.index = summary.index.astype(str)
    summary.index.name = "panjang teks (huruf)"
    return summary
