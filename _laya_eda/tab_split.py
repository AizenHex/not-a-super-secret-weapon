"""Tab Split data: membagi train_siap (tab Ekspor) jadi train + validasi, dengan tiga cara
(acak / menurut waktu / menurut kelompok), lalu mengecek teks yang sama di kedua sisi (bocor)."""
import pandas as pd
import streamlit as st
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import eda
import helpers
from penjelasan import show_explanation

METHODS = ["acak", "waktu", "kelompok"]
SAME_TEXT = "(teks sama = satu kelompok)"


def simple_keys(ctx, rows):
    # kunci sederhana seperti cek duplikat_mirip: huruf kecil, hanya a-z dan 0-9
    keys = ctx["texts"].str.lower()
    keys = keys.str.replace(r"[^a-z0-9]", "", regex=True)
    return keys.reindex(rows)


def stratify_label(ctx, train):
    # stratify hanya untuk label kategori yang lengkap dan tiap kelasnya punya minimal 2 baris
    label = ctx["label_column"]
    if label is None or eda.is_regression_label(train[label]):
        return None
    if train[label].isna().any() or train[label].value_counts().min() < 2:
        return None
    return train[label]


def split_random(ctx, train, valid_share):
    stratify = stratify_label(ctx, train)
    train_rows, valid_rows = train_test_split(train.index, test_size=valid_share,
                                              random_state=helpers.SEED, stratify=stratify)
    return train_rows, valid_rows


def split_time(ctx, train, valid_share):
    # baris paling baru jadi validasi; tanggal yang tidak terbaca ikut train
    dates = pd.to_datetime(train[ctx["date_column"]], errors="coerce")
    order = dates.sort_values(na_position="first").index
    n_train = len(order) - int(len(order) * valid_share)
    return order[:n_train], order[n_train:]


def split_group(ctx, train, valid_share, group_column):
    # semua baris satu kelompok masuk ke sisi yang sama
    if group_column == SAME_TEXT:
        groups = simple_keys(ctx, train.index)
    else:
        groups = train[group_column].fillna("(kosong)")
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_share, random_state=helpers.SEED)
    train_numbers, valid_numbers = next(splitter.split(train, groups=groups))
    return train.index[train_numbers], train.index[valid_numbers]


def count_leaks(ctx, train_rows, valid_rows):
    # baris validasi yang teksnya (kunci sederhana) juga ada di train = bocor
    train_keys = simple_keys(ctx, train_rows)
    valid_keys = simple_keys(ctx, valid_rows)
    leaked = valid_keys.isin(train_keys) & (valid_keys != "")
    return int(leaked.sum())


def label_table(ctx, train, train_rows, valid_rows):
    # kategori: persen tiap kelas di kedua sisi; label angka: ringkasan describe
    label = ctx["label_column"]
    table = pd.DataFrame()
    if eda.is_regression_label(train[label]):
        table["train"] = train.loc[train_rows, label].describe()
        table["validasi"] = train.loc[valid_rows, label].describe()
        return table.round(2)
    table["train %"] = train.loc[train_rows, label].value_counts(normalize=True) * 100
    table["validasi %"] = train.loc[valid_rows, label].value_counts(normalize=True) * 100
    return table.fillna(0).round(1)


def show_summary(ctx, train, train_rows, valid_rows):
    leaks = count_leaks(ctx, train_rows, valid_rows)
    st.write(f"train: {len(train_rows)} baris · validasi: {len(valid_rows)} baris · "
             f"teks validasi yang juga ada di train: {leaks}")
    if leaks > 0:
        st.warning(f"{leaks} baris validasi punya teks yang sama dengan train, jadi skor "
                   f"validasi terlalu bagus. Pakai cara **kelompok** + {SAME_TEXT}.")
    if ctx["label_column"] is not None:
        st.dataframe(label_table(ctx, train, train_rows, valid_rows))
    return leaks


def pick_method(ctx):
    left, middle, right = st.columns(3)
    method = left.radio("Cara split", METHODS, horizontal=True, key="split_method")
    valid_share = middle.slider("Porsi validasi", 0.1, 0.4, 0.2, 0.05, key="split_share")
    group_column = None
    if method == "kelompok":
        # kolom asli yang masih ada sesudah buang kustom
        options = [SAME_TEXT]
        for column in ctx["data"].columns:
            if column in ctx["ready_train"].columns:
                options.append(column)
        group_column = right.selectbox("Kolom kelompok", options, key="split_group")
    return method, valid_share, group_column


def write_files(ctx, train_part, valid_part):
    folder = ctx["folder"]
    first, second, third = st.columns(3)
    if first.button("Tulis train_split.csv + valid_split.csv", key="split_tulis"):
        train_part.to_csv(f"{folder}/train_split.csv", index_label="row")
        valid_part.to_csv(f"{folder}/valid_split.csv", index_label="row")
        st.success(f"Tersimpan: _laya_eda/{folder}/train_split.csv dan valid_split.csv")
    second.download_button("Unduh train_split.csv", train_part.to_csv(index_label="row"),
                           "train_split.csv", key="split_unduh_train")
    third.download_button("Unduh valid_split.csv", valid_part.to_csv(index_label="row"),
                          "valid_split.csv", key="split_unduh_valid")


def group_code(ctx, group_column):
    # baris kode untuk kolom kelompok; teks sama = kunci sederhana dari kolom yang dibaca Laya
    if group_column != SAME_TEXT:
        return [f'groups = train["{group_column}"].fillna("(kosong)")']
    lines = ['texts = train["' + ctx["columns"][0] + '"].fillna("").astype(str)']
    for column in ctx["columns"][1:]:
        lines.append(f'texts = texts + " | " + train["{column}"].fillna("").astype(str)')
    lines.append("groups = texts.str.lower()")
    lines.append('groups = groups.str.replace(r"[^a-z0-9]", "", regex=True)')
    return lines


def time_code(ctx, valid_share):
    date = ctx["date_column"]
    return [f'train["{date}"] = pd.to_datetime(train["{date}"], errors="coerce")',
            f'train = train.sort_values("{date}", na_position="first")',
            f"n_train = len(train) - int(len(train) * {valid_share})",
            "train_part = train.iloc[:n_train]",
            "valid_part = train.iloc[n_train:]"]


def split_code(ctx, method, valid_share, group_column, stratified):
    # kode notebook yang memberi pembagian yang sama dari train_siap.csv
    lines = [f'train = pd.read_csv("../_laya_eda/{ctx["folder"]}/train_siap.csv")']
    if method == "acak":
        stratify = "None"
        if stratified:
            stratify = f'train["{ctx["label_column"]}"]'
        lines.append(f"train_part, valid_part = train_test_split(train, test_size={valid_share}, "
                     f"random_state=42, stratify={stratify})")
    elif method == "waktu":
        lines = lines + time_code(ctx, valid_share)
    else:
        lines = lines + group_code(ctx, group_column)
        lines.append(f"splitter = GroupShuffleSplit(n_splits=1, test_size={valid_share}, "
                     "random_state=42)")
        lines.append("train_numbers, valid_numbers = next(splitter.split(train, groups=groups))")
        lines.append("train_part = train.iloc[train_numbers]")
        lines.append("valid_part = train.iloc[valid_numbers]")
    return "\n".join(lines)


def split_rows(ctx, train, method, valid_share, group_column):
    if method == "acak":
        return split_random(ctx, train, valid_share)
    if method == "waktu":
        return split_time(ctx, train, valid_share)
    return split_group(ctx, train, valid_share, group_column)


def show(ctx):
    st.write("Membagi **train_siap** (tab Ekspor: catatan + fitur pilihan + buang kustom) jadi "
             "train dan validasi. Validasi dipakai mengukur model sebelum menebak test.")
    train = ctx["ready_train"]
    method, valid_share, group_column = pick_method(ctx)
    if method == "waktu" and ctx["date_column"] is None:
        st.warning("Pilih kolom tanggal di samping dulu untuk split menurut waktu.")
        show_explanation("split")
        return
    train_rows, valid_rows = split_rows(ctx, train, method, valid_share, group_column)
    leaks = show_summary(ctx, train, train_rows, valid_rows)
    write_files(ctx, train.loc[train_rows].sort_index(), train.loc[valid_rows].sort_index())
    stratified = stratify_label(ctx, train) is not None
    st.code(split_code(ctx, method, valid_share, group_column, stratified), language="python")
    show_explanation("split", f"{leaks} dari {len(valid_rows)} baris validasi punya teks yang "
                              "sama dengan train.")
