"""Tab Split: train → bagian latih + valid, SEBELUM langkah EDA. Kode split = blok pertama sesudah baca data.

Pratinjau memakai cache data yang sama dengan seluruh aplikasi (common.split_state), jadi setelah
'Pakai split ini' tidak ada yang dihitung dua kali.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import split as splitter
from ui.kamus import show_table
from ui import common

METHODS = {
    "tanpa split": "Semua baris train dipakai belajar; valid kosong. Cocok bila model dinilai dengan cross-validation "
                   "(mis. AutoGluon bagging) atau kamu hanya butuh data bersih.",
    "acak": "Baris diacak, sebagian ke valid. **Stratified** = proporsi tiap kelas label sama di train dan valid "
            "(penting bila kelasnya tidak seimbang). Pilihan baku bila tiap baris saling lepas.",
    "kelompok": "Semua baris dengan nilai kolom kelompok yang sama (id keluarga, sekolah, kabupaten) masuk ke sisi yang "
                "sama. Pakai bila baris satu kelompok mirip/berulang: kalau dipisah, model 'menghafal' kelompok dan "
                "skor valid terlalu bagus. **(baris kembar)** = baris yang isinya sama persis tidak dipisah.",
    "waktu": "Baris **sesudah tanggal batas** ke valid. Pakai bila test berasal dari periode sesudah train: valid harus "
             "meniru masa depan, bukan acak dari masa yang sama.",
}


@st.cache_data(max_entries=16, show_spinner=False)
def twins(key: str, _train: pd.DataFrame, _valid: pd.DataFrame, label: str | None) -> dict:
    return splitter.twins_across(_train, _valid, label)


def chosen_config(data: pd.DataFrame, label: str | None) -> dict | None:
    """Widget → config. Nilai awal widget = split yang sedang dipakai."""
    current = common.split_config() or {}
    st.session_state.setdefault("split_method", current.get("method", "tanpa split"))
    method = st.radio("Metode", list(METHODS), horizontal=True, key="split_method")
    st.caption(METHODS[method])
    if method == "tanpa split":
        return None
    left, middle, right = st.columns(3)
    st.session_state.setdefault("split_share", float(current.get("share", 0.2)))
    st.session_state.setdefault("split_seed", int(current.get("seed", 42)))
    share = left.slider("Porsi valid", 0.05, 0.5, step=0.05, key="split_share",
                        help="0,2 = 20% baris (atau 20% kelompok) ke valid. Umumnya 0,1–0,3.")
    seed = middle.number_input("random_state", 0, 10_000, step=1, key="split_seed",
                               help="Angka yang sama = baris valid yang sama setiap kali skrip dijalankan.")
    config = {"method": method, "share": round(float(share), 2), "seed": int(seed)}
    features = [c for c in data.columns if c != label]
    if method == "acak":
        problem = splitter.stratify_problem(data, label)
        st.session_state.setdefault("split_stratify", current.get("stratify", problem is None))
        stratify = right.checkbox("Stratified pada label", key="split_stratify", disabled=problem is not None,
                                  help="Proporsi tiap kelas label dijaga sama di train dan valid.")
        if problem:
            right.caption(f"Stratified tidak bisa: {problem}.")
        config["stratify"] = bool(stratify) and problem is None
    elif method == "kelompok":
        options = [splitter.ROW_GROUP, *features]
        if st.session_state.get("split_group") not in options:
            st.session_state["split_group"] = current.get("column") if current.get("column") in options else options[0]
        config["column"] = right.selectbox("Kolom kelompok", options, key="split_group")
    else:
        if st.session_state.get("split_time") not in features:
            st.session_state["split_time"] = current.get("column") if current.get("column") in features else features[0]
        column = right.selectbox("Kolom waktu", features, key="split_time")
        config["column"] = column
        config["numeric"] = bool(pd.api.types.is_numeric_dtype(data[column]))
    return config


def show_split() -> None:
    common.header("split",
                  "Bagi train jadi **train** (untuk belajar) dan **valid** (untuk menilai), **sebelum** langkah EDA. "
                  "Sesudah split, semua halaman EDA melihat bagian train saja; median, batas outlier, daftar kategori "
                  "dipelajari dari train lalu **dipakai** ke valid dan test. Baris valid dan test tidak pernah dibuang. "
                  "Kode split jadi blok pertama sesudah baca data di skrip ekspor.",
                  "Hampir selalu, sebelum menambah langkah: kamu butuh angka jujur untuk tahu langkahmu membantu atau "
                  "tidak (halaman AutoGluon menilai di valid ini).",
                  "Mengganti split sesudah banyak langkah = semua langkah dihitung ulang dari train baru (batas dan "
                  "pengisi bisa sedikit berubah). Split acak pada data berkelompok/berwaktu membuat skor terlalu bagus.")
    if not common.need_data():
        return
    data, label = common.raw_train(), common.label()
    if not label:
        st.info("Kolom label belum dipilih (sidebar **Data**): split tetap bisa, tapi tanpa stratified dan cek label.")
    config = chosen_config(data, label)
    current = common.split_config()
    try:
        result = common.split_state(config)
    except Exception as error:   # mis. stratified dengan kelas 1 baris: tampilkan apa adanya
        st.error(f"Split gagal: {type(error).__name__}: {error}")
        return
    train, valid, test = result["train"], result["valid"], result["test"]

    cols = st.columns(3)
    cols[0].metric("train", f"{len(train):,}", f"{len(train) / len(data):.0%} dari train asli", delta_color="off")
    cols[1].metric("valid", f"{len(valid):,}", f"{len(valid) / len(data):.0%}" if len(valid) else None,
                   delta_color="off")
    cols[2].metric("test (file)", f"{len(test):,}")
    if config and valid.empty:
        st.warning("Valid kosong dengan pilihan ini (mis. kolom waktu tidak terbaca sebagai tanggal). Ganti kolom/metode.")

    if config and not valid.empty:
        show_checks(config, data, train, valid, label)

    st.subheader("Kode split (blok pertama sesudah baca data)")
    st.code(splitter.split_code(config), language="python")
    if config == current:
        st.success(f"Split ini yang sedang dipakai: {splitter.describe(current)}.")
        return
    note = (" Semua langkah akan dihitung ulang dari train yang baru (langkah tidak dihapus)."
            if common.project()["steps"] else "")
    st.caption(f"Sedang dipakai: **{splitter.describe(current)}**.{note}")
    if st.button("Pakai tanpa split" if config is None else "Pakai split ini", type="primary", key="split_apply"):
        common.project()["split"] = config
        st.toast(f"Split: {splitter.describe(config)}")
        st.rerun()


def show_checks(config: dict, data: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame,
                label: str | None) -> None:
    """Cek buku teks sebelum memakai split: proporsi label, bocoran baris kembar, kelompok, rentang waktu."""
    if label:
        st.subheader("Label per bagian")
        if splitter.is_classification(data, label):
            table = pd.DataFrame({"train %": train[label].value_counts(normalize=True, dropna=False) * 100,
                                  "valid %": valid[label].value_counts(normalize=True, dropna=False) * 100}).fillna(0)
            table["selisih (poin %)"] = table["valid %"] - table["train %"]
            table["baris valid"] = valid[label].value_counts(dropna=False).reindex(table.index).fillna(0).astype(int)
            show_table(table.round(2).rename(index=str), {label: "Nilai label (kelas)."})
            gap = table["selisih (poin %)"].abs().max()
            if gap > 2:
                st.warning(f"Proporsi kelas berbeda sampai {gap:.1f} poin %. Untuk split acak, nyalakan **Stratified**.")
            small = table[table["baris valid"] < 5]
            if len(small):
                st.caption(f"Kelas dengan < 5 baris di valid: {', '.join(map(str, small.index))} — skor untuk kelas "
                           "itu tidak stabil.")
        else:
            show_table(pd.DataFrame({"train": train[label].describe(), "valid": valid[label].describe()}).T,
                         width="stretch")

    st.subheader("Bocoran: baris valid yang kembar dengan baris train")
    found = twins(common.split_key(config), train, valid, label)
    if found["kembar"]:
        st.warning(f"{found['kembar']:,} baris valid ({found['kembar'] / len(valid):.1%}) fiturnya sama persis dengan "
                   f"baris train ({found['beda label']:,} di antaranya labelnya berbeda). Model cukup menghafal untuk "
                   "menjawab baris itu, jadi skor valid terlalu bagus. Pakai metode **kelompok** → **(baris kembar)**.")
    else:
        st.success("Tidak ada baris valid yang fiturnya sama persis dengan baris train.")

    if config["method"] == "kelompok":
        if config["column"] == splitter.ROW_GROUP:
            st.caption("Kelompok = isi seluruh fitur: baris kembar selalu di sisi yang sama.")
        else:
            left, right = train[config["column"]].astype(str), valid[config["column"]].astype(str)
            shared = set(left) & set(right)
            st.write(f"Kelompok di train: **{left.nunique():,}** · di valid: **{right.nunique():,}** · di keduanya: "
                     f"**{len(shared)}** (harus 0)")
    if config["method"] == "waktu":
        column = config["column"]
        if config.get("numeric"):
            left, right = train[column], valid[column]
        else:
            left, right = (pd.to_datetime(part[column], errors="coerce") for part in (train, valid))
        st.write(f"Waktu train: **{left.min()} … {left.max()}** · valid: **{right.min()} … {right.max()}** · "
                 f"waktu kosong (tetap di train): **{int(left.isna().sum()):,}**")
