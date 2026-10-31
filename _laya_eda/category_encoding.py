"""Pilihan encoding kolom kategori di tab Kode EDA: satu cara untuk semua kolom, bisa diganti per
kolom. Kodenya ditulis codegen.py; nilai encoding (urutan, rata-rata label, sumbu makna) dihitung
dari train lalu dipakai sama untuk test."""
import pandas as pd
import streamlit as st

# cara -> keterangan; "tidak" = kolom tidak di-encode (tidak ikut ke kode)
METHODS = {
    "tidak": "tidak di-encode",
    "saran": "saran: ≤10 nilai → one-hot, lebih → kedekatan makna",
    "one_hot": "one-hot (satu kolom 0/1 per nilai, maks 30 nilai tersering)",
    "ordinal": "kode angka (urut dari yang tersering; nilai baru = -1)",
    "target": "target encoding (rata-rata label per nilai, dihaluskan)",
    "makna": "kedekatan makna (MiniLM + PCA: nilai yang artinya mirip → angka mirip)",
}
SAME = "(ikut pilihan umum)"
FEW_VALUES = 10


def candidates(ctx, table, skip):
    # kolom teks pendek (rata-rata ≤ 40 huruf) yang berulang (nilai berbeda < separuh baris)
    # = kategori, termasuk isian bebas yang berantakan ("genting", "Genteng", "genteng tanah")
    columns = []
    for column in table.columns:
        if column == ctx["label_column"] or column in skip:
            continue
        if pd.api.types.is_numeric_dtype(table[column]):
            continue
        unique = table[column].nunique()
        length = table[column].dropna().astype(str).str.len().mean()
        if 2 <= unique < len(table) / 2 and length <= 40:
            columns.append(column)
    return columns


def method_name(method):
    if method == SAME:
        return SAME
    return METHODS[method]


def method_list(ctx):
    # target encoding butuh label
    names = []
    for name in METHODS:
        if name == "target" and ctx["label_column"] is None:
            continue
        names.append(name)
    return names


def resolve(method, values):
    # "saran": sedikit nilai → one-hot; banyak nilai → kedekatan makna
    if method != "saran":
        return method
    if values.nunique() <= FEW_VALUES:
        return "one_hot"
    return "makna"


def pick_overrides(ctx, table, columns, general):
    chosen = {}
    with st.expander(f"Atur per kolom ({len(columns)} kolom kategori)"):
        for column in columns:
            label = f"{column} ({table[column].nunique()} nilai berbeda)"
            method = st.selectbox(label, [SAME] + method_list(ctx), format_func=method_name,
                                  key=f"enc_{column}__{ctx['folder']}")
            if method == SAME:
                method = general
            chosen[column] = resolve(method, table[column])
    return chosen


def pick_encoding(ctx, table, skip):
    # hasil: ({kolom: cara}, buang kolom asli?) hanya untuk kolom yang benar-benar di-encode
    columns = candidates(ctx, table, skip)
    if len(columns) == 0:
        st.caption("Tidak ada kolom kategori (teks pendek berulang).")
        return {}, False
    general = st.selectbox("Encoding kolom kategori (semua kolom)", method_list(ctx),
                           format_func=METHODS.get, key=f"enc_umum__{ctx['folder']}")
    chosen = pick_overrides(ctx, table, columns, general)
    encoding = {}
    for column in chosen:
        if chosen[column] != "tidak":
            encoding[column] = chosen[column]
    drop = st.checkbox("Buang kolom kategori asli sesudah di-encode (model butuh angka)",
                       value=True, key=f"enc_buang__{ctx['folder']}")
    return encoding, drop
