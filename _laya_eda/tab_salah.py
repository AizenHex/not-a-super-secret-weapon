"""Tab Salah tebak: tabel baris train / test yang masih salah ditebak model baseline."""
import io
import os

import pandas as pd
import streamlit as st

import eda
import features
import helpers
import kamus
import mistakes
from penjelasan import show_explanation

DEFAULT_TEST = "../../data_latihan/text/classification/emot_test.csv"
MODELS = {"TF-IDF kata": ("kata",), "TF-IDF kata + karakter": ("kata", "char")}


@st.cache_data(show_spinner="Model menebak tiap baris train (5 lipatan)...")
def cached_train(_ctx, signature, groups, max_rows):
    return mistakes.train_predictions(_ctx, groups, max_rows)


@st.cache_data(show_spinner="Model dilatih di train lalu menebak test...")
def cached_test(_ctx, signature, test_key, _test_texts, _test_labels, groups, max_rows):
    return mistakes.test_predictions(_ctx, _test_texts, _test_labels, groups, max_rows)


def remembered_button(key, label):
    # tombol yang 'diingat': hasil tetap tampil walau widget lain diklik
    if st.button(label, key=f"salah_tombol_{key}"):
        st.session_state[f"salah_flag_{key}"] = True
    return st.session_state.get(f"salah_flag_{key}", False)


def show_class_summary(ctx, table, where):
    wrong = table[~table["benar"]]
    accuracy = table["benar"].mean()
    confident = (table["jenis"] == "yakin salah (cek label)").sum()
    sentence = (f"Di {where}, model baseline salah menebak {len(wrong)} dari {len(table)} baris "
                f"({1 - accuracy:.1%}); {confident} di antaranya salah dengan yakin "
                f"(peluang ≥ {mistakes.CONFIDENT:.0%}), kandidat label salah.")
    st.success(sentence)
    helpers.add_finding(ctx, f"Salah tebak ({where})", sentence,
                        "Tebakan model TF-IDF + LogisticRegression.",
                        "Periksa pasangan kelas yang sering tertukar dan baris 'yakin salah'.",
                        min(1 - accuracy, 1.0))
    left, right = st.columns(2)
    left.write("**Pasangan label → tebakan yang paling sering tertukar**")
    left.dataframe(mistakes.swapped_pairs(table), hide_index=True)
    right.write("**Skor per kelas**")
    right.dataframe(mistakes.class_report(table))
    st.write("**Tabel silang label asli (baris) × tebakan (kolom)**")
    kamus.show_table(pd.crosstab(table["label"], table["tebakan"]))


def filter_class_rows(table, prefix):
    kinds = ["(semua salah)"] + sorted(table["jenis"].unique())
    left, middle, right = st.columns(3)
    kind = left.selectbox("Jenis", kinds, key=f"{prefix}_jenis")
    labels = ["(semua)"] + sorted(table["label"].astype(str).unique())
    label = middle.selectbox("Label asli", labels, key=f"{prefix}_label")
    guesses = ["(semua)"] + sorted(table["tebakan"].astype(str).unique())
    guess = right.selectbox("Tebakan", guesses, key=f"{prefix}_tebakan")
    shown = table[table["jenis"] == kind]
    if kind == "(semua salah)":
        shown = table[~table["benar"]]
    if label != "(semua)":
        shown = shown[shown["label"].astype(str) == label]
    if guess != "(semua)":
        shown = shown[shown["tebakan"].astype(str) == guess]
    return shown.sort_values("peluang_tebakan", ascending=False)


def show_rows(ctx, shown, texts, prefix, with_reasons):
    view = shown.copy()
    view.insert(0, "teks", texts[shown.index])
    if with_reasons:
        view["alasan_outlier"] = ctx["reasons"].reindex(shown.index)
        for name in ctx["lens_done"]:
            view[f"laya_{name}"] = ctx["table"][name].reindex(shown.index)
    st.write(f"{len(view)} baris (500 pertama ditampilkan)")
    kamus.show_table(view.head(500).round(3))
    st.download_button("Unduh tabel ini (.csv)", view.to_csv(index_label="row"),
                       f"{prefix}.csv", key=f"{prefix}_unduh")


def show_number_table(ctx, table, texts, prefix, with_reasons):
    error = table["selisih_mutlak"].mean()
    sentence = f"Rata-rata tebakan meleset {error:,.2f} dari label (MAE); 5% terburuk di bawah."
    st.success(sentence)
    helpers.add_finding(ctx, f"Salah tebak ({prefix})", sentence, "Tebakan Ridge + TF-IDF.",
                        "Periksa baris yang meleset jauh.", 0.5)
    show_rows(ctx, table.sort_values("selisih_mutlak", ascending=False), texts, prefix,
              with_reasons)


def section_train(ctx, groups, max_rows):
    st.markdown("### Train: tebakan out-of-fold")
    st.caption("Tiap baris ditebak model yang dilatih tanpa baris itu (5 lipatan), jadi "
               "kesalahannya mirip kesalahan di data baru.")
    if not remembered_button("train", "Cari salah tebak di train"):
        return
    table = cached_train(ctx, features.signature(ctx), groups, max_rows)
    if "benar" not in table.columns:
        show_number_table(ctx, table, ctx["texts"], "salah_train", True)
        return
    show_class_summary(ctx, table, "train")
    st.write("**Persen salah menurut panjang teks**")
    kamus.show_table(mistakes.error_by_length(table, ctx["texts"]))
    show_rows(ctx, filter_class_rows(table, "salah_train"), ctx["texts"], "salah_train", True)


def read_test():
    path = st.text_input("File CSV test (path relatif ke folder _laya_eda)", DEFAULT_TEST,
                         key="salah_test_path")
    uploaded = st.file_uploader("... atau unggah CSV test", type="csv", key="salah_unggah")
    if uploaded is not None:
        return pd.read_csv(io.BytesIO(uploaded.getvalue())), uploaded.name
    if not os.path.exists(path):
        st.info(f"File test tidak ditemukan: {path}")
        return None, path
    return pd.read_csv(path), f"{path}:{os.path.getmtime(path)}"


def test_inputs(ctx, test):
    missing = []
    for column in ctx["columns"]:
        if column not in test.columns:
            missing.append(column)
    if len(missing) > 0:
        st.error(f"Kolom ini tidak ada di CSV test: {', '.join(missing)}")
        return None, None
    labels = None
    if ctx["label_column"] in test.columns:
        labels = test[ctx["label_column"]]
    return helpers.row_texts(test, ctx["columns"]), labels


def show_unlabeled_test(ctx, table, texts):
    train_share = ctx["data"][ctx["label_column"]].value_counts(normalize=True)
    test_share = table["tebakan"].value_counts(normalize=True)
    compare = pd.DataFrame({"persen_train": 100 * train_share,
                            "persen_tebakan_test": 100 * test_share})
    unsure = (table["peluang_tebakan"] < mistakes.UNSURE).mean()
    st.success(f"Test tidak punya label, jadi yang bisa dilihat: {unsure:.1%} baris membuat model "
               f"ragu (peluang < {mistakes.UNSURE:.0%}). Baris paling ragu ada di atas.")
    st.write("**Sebaran label train vs sebaran tebakan di test (%)**")
    kamus.show_table(compare.fillna(0).round(1))
    show_rows(ctx, table.sort_values("peluang_tebakan"), texts, "ragu_test", False)


def section_test(ctx, groups, max_rows):
    st.markdown("### Test: model dilatih di seluruh train")
    test, test_key = read_test()
    if test is None:
        return
    texts, labels = test_inputs(ctx, test)
    if texts is None or not remembered_button("test", "Tebak data test"):
        return
    table = cached_test(ctx, features.signature(ctx), test_key, texts, labels, groups, max_rows)
    if labels is None and "peluang_tebakan" in table.columns:
        show_unlabeled_test(ctx, table, texts)
    elif labels is None:
        show_rows(ctx, table, texts, "tebakan_test", False)    # label angka tanpa jawaban
    elif "benar" not in table.columns:
        show_number_table(ctx, table, texts, "salah_test", False)
    else:
        show_class_summary(ctx, table, "test")
        show_rows(ctx, filter_class_rows(table, "salah_test"), texts, "salah_test", False)


def show(ctx):
    if ctx["label_column"] is None:
        st.info("Pilih kolom label di sidebar dulu.")
        return
    label = ctx["data"][ctx["label_column"]]
    if features.task_of(label) == "kelas" and label.nunique() > eda.MAX_LABEL_CLASSES:
        st.warning(f"Kolom label punya {label.nunique()} nilai berbeda, jadi bukan label kategori.")
        return
    st.write("Baris mana yang **masih salah ditebak** model baseline? Model: TF-IDF teks + "
             "LogisticRegression (label angka: Ridge). Model lombamu nanti bisa beda, tetapi baris "
             "yang sulit untuk baseline biasanya juga sulit untuk model lain.")
    left, right = st.columns(2)
    model_name = left.radio("Model", list(MODELS), horizontal=True, key="salah_model")
    max_rows = right.number_input("Maksimal baris train", 500, 50000, 5000, step=500,
                                  key="salah_max_rows")
    section_train(ctx, MODELS[model_name], int(max_rows))
    section_test(ctx, MODELS[model_name], int(max_rows))
    show_explanation("salah_tebak")
