"""Tab Split data: membagi train_siap (hasil kode tab Kode EDA) jadi train + validasi, dengan
tiga cara (acak / menurut waktu / menurut kelompok), lalu mengecek teks yang sama di kedua sisi
(bocor). Filenya ditulis di tab Ekspor; kodenya ikut ke notebook tab Kode EDA."""
import pandas as pd
import streamlit as st
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import eda
import helpers
import kamus
from penjelasan import show_explanation

METHODS = ["acak", "waktu", "kelompok"]
SAME_TEXT = "(teks sama = satu kelompok)"


def simple_keys(ctx, rows):
    # kunci sederhana teks asli (helpers.simple_keys, dihitung sekali di context.py)
    return ctx["simple_keys"].reindex(rows)


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
        kamus.show_table(label_table(ctx, train, train_rows, valid_rows))
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


def same_text_code(ctx):
    # kunci teks sama dari CSV ASLI (sebelum kolom angka diubah / diisi), sama persis dengan
    # helpers.row_texts + helpers.simple_keys di aplikasi
    lines = ["raw = pd.read_csv(TRAIN_CSV)    # teks asli untuk kunci kelompok"]
    if len(ctx["columns"]) == 1:
        lines.append(f'texts = raw["{ctx["columns"][0]}"].fillna("").astype(str)')
    else:
        lines = lines + [f"READ_COLUMNS = {ctx['columns']}",
                         'texts = pd.Series("", index=raw.index)',
                         "for column in READ_COLUMNS:",
                         '    texts = texts + column + ": " + raw[column].fillna("").astype(str)'
                         ' + " | "']
    lines.append('groups = texts.str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)')
    lines.append("groups = groups.loc[train.index]")
    return lines


def group_code(ctx, group_column):
    # baris kode untuk kolom kelompok
    if group_column != SAME_TEXT:
        return [f'groups = train["{group_column}"].fillna("(kosong)")']
    return same_text_code(ctx)


def time_code(ctx, valid_share):
    # baris paling baru jadi validasi; train sendiri tidak diubah urutannya
    date = ctx["date_column"]
    return [f'dates = pd.to_datetime(train["{date}"], errors="coerce")',
            'order = dates.sort_values(na_position="first").index',
            f"n_train = len(order) - int(len(order) * {valid_share})",
            "train_part = train.loc[order[:n_train]].sort_index()",
            "valid_part = train.loc[order[n_train:]].sort_index()"]


def split_code(ctx, method, valid_share, group_column, stratified):
    # baris kode notebook (sesudah sel Kode EDA, memakai `train`) dengan pembagian yang sama
    lines = ["from sklearn.model_selection import GroupShuffleSplit, train_test_split", ""]
    if method == "acak":
        stratify = "None"
        if stratified:
            stratify = f'train["{ctx["label_column"]}"]'
        lines.append(f"train_part, valid_part = train_test_split(\n    train, test_size="
                     f"{valid_share}, random_state=42, stratify={stratify})")
        lines.append("train_part = train_part.sort_index()")
        lines.append("valid_part = valid_part.sort_index()")
    elif method == "waktu":
        lines = lines + time_code(ctx, valid_share)
    else:
        lines = lines + group_code(ctx, group_column)
        lines.append(f"splitter = GroupShuffleSplit(n_splits=1, test_size={valid_share}, "
                     "random_state=42)")
        lines.append("train_numbers, valid_numbers = next(splitter.split(train, groups=groups))")
        lines.append("train_part = train.iloc[train_numbers]")
        lines.append("valid_part = train.iloc[valid_numbers]")
    lines.append('print("train:", train_part.shape, "validasi:", valid_part.shape)')
    return lines


def split_rows(ctx, train, method, valid_share, group_column):
    if method == "acak":
        return split_random(ctx, train, valid_share)
    if method == "waktu":
        return split_time(ctx, train, valid_share)
    return split_group(ctx, train, valid_share, group_column)


def show(ctx):
    st.write("Membagi **train_siap** (hasil kode di tab **Kode EDA**) jadi train dan validasi. "
             "Validasi dipakai mengukur model sebelum menebak test. File ditulis di tab "
             "**Ekspor**; kode di bawah ikut masuk notebook tab Kode EDA.")
    train = ctx["ready_train"]
    method, valid_share, group_column = pick_method(ctx)
    if method == "waktu" and ctx["date_column"] is None:
        st.warning("Pilih kolom tanggal di samping dulu untuk split menurut waktu.")
        show_explanation("split")
        return
    train_rows, valid_rows = split_rows(ctx, train, method, valid_share, group_column)
    leaks = show_summary(ctx, train, train_rows, valid_rows)
    ctx["split_parts"] = (train.loc[train_rows].sort_index(), train.loc[valid_rows].sort_index())
    stratified = stratify_label(ctx, train) is not None
    ctx["split_lines"] = split_code(ctx, method, valid_share, group_column, stratified)
    st.code("\n".join(ctx["split_lines"]), language="python")
    show_explanation("split", f"{leaks} dari {len(valid_rows)} baris validasi punya teks yang "
                              "sama dengan train.")
