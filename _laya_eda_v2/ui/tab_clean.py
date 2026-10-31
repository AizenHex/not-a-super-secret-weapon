"""Tab Kosong & penanda, Kategori, Isian bebas, Outlier, Baris."""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import streamlit as st

from core import audit, outlier_tools, text_tools
from core import recipe as rcp
from ui import common


# ---------------------------------------------------------------- kosong & penanda

def show_missing() -> None:
    common.header("kosong",
                  "Cari sel yang **sebenarnya kosong tapi tidak terbaca kosong**: teks `'-'`/`'?'`, nilai penjaga "
                  "`2.147.483.647`, nol yang berarti 'tidak diisi', dan kolom checkbox (satu nilai + kosong).",
                  "Penanda teks dan nilai penjaga hampir selalu aman dijadikan kosong.",
                  "**Nol**: bisa nol sungguhan (tidak punya hutang). Jangan dijadikan kosong tanpa alasan. "
                  "Nilai bulat besar seperti 2.000.000.000 bisa saja asli: lihat 'maks wajar'.")
    if not common.need_data():
        return
    r, data = common.recipe(), common.features()

    st.subheader("1. Penanda kosong palsu di teks")
    st.caption("Dijadikan NaN oleh tidy_text untuk kolom teks yang dipilih di tab Tipe kolom.")
    st.dataframe(audit.fake_missing(data, r["missing_markers"]), width="stretch", hide_index=True)
    markers = st.text_input("Daftar penanda (pisahkan dengan |)", " | ".join(r["missing_markers"]))
    if st.button("Simpan penanda"):
        r["missing_markers"] = [m.strip().lower() for m in markers.split("|")] + [""]
        r["missing_markers"] = sorted(set(r["missing_markers"]))
        common.saved("Penanda disimpan")

    st.subheader("2. Nilai penjaga (sentinel)")
    found = audit.sentinels(data)
    if found.empty:
        st.write("Tidak ada nilai ≥ 2.000.000.000.")
    else:
        found.insert(0, "jadikan kosong", found["kolom"].isin(r["sentinels"]))
        edited = st.data_editor(found, width="stretch", hide_index=True, key="sentinel_editor",
                                disabled=[c for c in found.columns if c != "jadikan kosong"])
        floor = st.number_input("Batas bawah nilai penjaga", value=rcp.SENTINEL_FLOOR, step=1_000_000)
        if st.button("Simpan nilai penjaga"):
            r["sentinels"] = {c: int(floor) for c in edited.loc[edited["jadikan kosong"], "kolom"]}
            common.saved(f"{len(r['sentinels'])} kolom nilai penjaga disimpan")

    st.subheader("3. Nol yang mungkin berarti 'tidak diisi'")
    zeros = audit.zero_share(data)
    st.dataframe(zeros, width="stretch", hide_index=True)
    picked = st.multiselect("Nol → kosong (hanya bila nol mustahil, mis. luas tanah 0)", list(zeros["kolom"]),
                            [c for c in r["zero_as_missing"] if c in set(zeros["kolom"])])
    if st.button("Simpan pilihan nol"):
        r["zero_as_missing"] = picked
        common.saved("Pilihan nol disimpan")

    st.subheader("4. Kolom satu nilai + kosong (checkbox)")
    single = audit.single_value_columns(data)
    if single.empty:
        st.write("Tidak ada.")
    else:
        single.insert(0, "kosong berarti", [r["blank_as"].get(c, "") for c in single["kolom"]])
        st.caption("Isi 'kosong berarti' (mis. Tidak), kosongkan = biarkan kosong. Berubah hanya setelah disimpan.")
        edited = st.data_editor(single, width="stretch", hide_index=True, key="blank_editor",
                                disabled=["kolom", "satu-satunya nilai", "kosong"])
        if st.button("Simpan isi kosong checkbox"):
            r["blank_as"] = {c: v for c, v in zip(edited["kolom"], edited["kosong berarti"]) if v}
            common.saved(f"{len(r['blank_as'])} kolom checkbox disimpan")
            st.rerun()

    st.subheader("Sebelum → sesudah (jumlah kosong per kolom)")
    after = common.prepared()
    table = pd.DataFrame({"kosong sebelum": data.isna().sum(), "kosong sesudah": after.isna().sum()})
    table["selisih"] = table["kosong sesudah"] - table["kosong sebelum"]
    st.dataframe(table[table["selisih"] != 0], width="stretch")


# ---------------------------------------------------------------- kategori

def approval_table(table: pd.DataFrame, key: str, default_ok: bool) -> pd.DataFrame:
    table = table.copy()
    table.insert(0, "setuju", default_ok)
    return st.data_editor(table, width="stretch", hide_index=True, key=key,
                          column_config={"asli": st.column_config.TextColumn("asli (repr)")},
                          disabled=[c for c in table.columns if c not in ("setuju", "usulan baku")])


def show_categories() -> None:
    common.header("kategori",
                  "Satukan ejaan berbeda untuk **nilai yang sama** di kolom kategori "
                  "(`'Kab. Pati'` = `'Kabupaten Pati'`, `'Kota Semarang '` = `'Kota Semarang'`).",
                  "**Fingerprint** aman untuk beda huruf besar, spasi, tanda baca, urutan kata, singkatan.",
                  "**skrub.deduplicate** bisa menggabung nilai yang berbeda (`'Kab. Semarang'` → "
                  "`'Kota Semarang'`). Periksa baris berperingatan satu per satu.")
    if not common.need_data():
        return
    r = common.recipe()
    if not r["text_columns"]:
        st.info("Pilih kolom teks yang dirapikan di tab **Tipe kolom** dulu.")
        return
    raw = common.features()
    column = st.selectbox("Kolom", r["text_columns"], key="category_column")
    tidy = common.pipeline().tidy_text(raw[column])

    method = st.radio("Cara mencari", ["fingerprint (aman)", "skrub.deduplicate (opsional, periksa)"],
                      horizontal=True)
    if method.startswith("fingerprint"):
        proposals = text_tools.fingerprint_groups(tidy)
        st.caption("Kunci = huruf kecil, tanpa tanda baca, singkatan diperluas, kata diurutkan. "
                   "Kunci sama → usulan digabung ke ejaan tersering.")
        default_ok = True
    else:
        unique = tidy.nunique()
        st.caption(f"{unique} nilai unik. skrub.deduplicate mengklaster n-gram huruf seluruh nilai; "
                   "di data uji: 640 nilai ≈ 3 detik, 1.961 nilai ≈ 70 detik.")
        clusters = st.number_input("Jumlah klaster (0 = otomatis)", 0, max(unique, 1), 0)
        if st.button("Jalankan skrub.deduplicate"):
            st.session_state[f"dedup_{column}"] = text_tools.deduplicate_map(tidy, int(clusters) or None)
        proposals = st.session_state.get(f"dedup_{column}", pd.DataFrame(columns=["asli", "usulan baku", "jumlah"]))
        default_ok = False
    if proposals.empty:
        st.write("Tidak ada usulan.")
    else:
        proposals = proposals.assign(asli=proposals["asli"].map(repr))
        edited = approval_table(proposals, f"cat_{column}_{method[:5]}", default_ok)
        if st.button("Simpan peta yang dicentang", type="primary"):
            chosen = edited[edited["setuju"]]
            mapping = r["category_maps"].setdefault(column, {})
            mapping.update({ast.literal_eval(a): b for a, b in zip(chosen["asli"], chosen["usulan baku"])})
            common.saved(f"{len(chosen)} ejaan dipetakan")
            st.rerun()

    current = r["category_maps"].get(column, {})
    if current:
        with st.expander(f"Peta di resep untuk {column} ({len(current)})"):
            table = pd.DataFrame({"asli": [repr(k) for k in current], "baku": list(current.values())})
            st.dataframe(table, width="stretch", hide_index=True)
            if st.button("Hapus semua peta kolom ini"):
                r["category_maps"].pop(column)
                st.rerun()
    common.before_after(raw[column], common.prepared()[column])


# ---------------------------------------------------------------- isian bebas

def show_free_text() -> None:
    common.header("isian",
                  "Kolom isian bebas berisi **beberapa bahan** dalam satu sel dengan salah ketik "
                  "(`'Genteng & asbas.'`). Sel dipecah per bahan, salah ketik dibetulkan **per kata**, lalu "
                  "ditulis ulang sebagai daftar baku `'asbes, genteng'` (kolom tetap satu, nama tetap).",
                  "Kolom bahan atap/tembok/lantai, daftar fasilitas, dsb.",
                  "Kata sah yang mirip kata lain (`tanpa`/`tanah`, `batu`/`bata`, `dapur`/`kapur`). "
                  "Usulan skor < 85 tidak dicentang otomatis; kosakata dihitung juga dari kolom terkait.")
    if not common.need_data():
        return
    r, raw = common.recipe(), common.features()
    candidates = common.object_columns(raw)
    column = st.selectbox("Kolom isian bebas", candidates,
                          index=next((i for i, c in enumerate(candidates) if c.startswith("Bahan")), 0))
    related = st.multiselect("Kolom terkait (kosakata digabung)", [c for c in candidates if c != column],
                             [c for c in candidates if c != column and c.startswith("Bahan")])
    left, right = st.columns(2)
    min_count = left.slider("Kata baku = muncul minimal … kali", 5, 200, 30)
    cutoff = right.slider("Skor kemiripan minimal (fuzz.ratio)", 60, 95, 75)
    st.caption("Pemisah bahan: , & / + ; dan atau ( )")

    words = text_tools.word_map(raw[column], min_count, cutoff, [raw[c] for c in related])
    stored = r["free_text"].get(column, {})
    if not words.empty:
        words["setuju"] = [stored.get(w) == u if stored else ok
                           for w, u, ok in zip(words["kata asli"], words["usulan"], words["setuju"])]
    st.subheader(f"Usulan salah ketik ({len(words)})")
    edited = st.data_editor(words, width="stretch", hide_index=True, key=f"words_{column}",
                            disabled=["kata asli", "skor", "jumlah"])

    st.subheader("Kata baku yang mirip (kandidat sinonim)")
    st.caption("Tidak pernah otomatis. Centang bila 'kata 2' sama maknanya dengan 'kata 1' (mis. genteng = genting).")
    similar = text_tools.similar_vocabulary(raw[column], min_count)
    similar.insert(0, "samakan 2 → 1", [stored.get(b) == a for a, b in zip(similar["kata 1"], similar["kata 2"])])
    laya = st.session_state.get("laya_same", {})
    if laya:
        similar["Laya: sama?"] = [laya.get((a, b)) for a, b in zip(similar["kata 1"], similar["kata 2"])]
    synonyms = st.data_editor(similar, width="stretch", hide_index=True, key=f"syn_{column}",
                              disabled=[c for c in similar.columns if c != "samakan 2 → 1"])

    mapping = dict(zip(edited.loc[edited["setuju"], "kata asli"], edited.loc[edited["setuju"], "usulan"]))
    for first, second in synonyms.loc[synonyms["samakan 2 → 1"], ["kata 1", "kata 2"]].to_numpy():
        mapping[second] = first
    mapping = {w: mapping.get(u, u) for w, u in mapping.items()}          # gentng→genteng→genting
    left, right = st.columns(2)
    left.caption("Sebelum (nilai asli tersering)")
    left.dataframe(raw[column].value_counts().head(15), width="stretch")
    right.caption("Sesudah: bahan baku per baris (pratinjau dengan centang saat ini)")
    right.dataframe(text_tools.material_preview(raw[column], mapping, 15), width="stretch", hide_index=True)
    col1, col2 = st.columns(2)
    if col1.button("Simpan ke resep", type="primary", key=f"save_free_{column}"):
        r["free_text"][column] = mapping
        common.saved(f"{column}: {len(mapping)} kata dipetakan")
        st.rerun()
    if column in r["free_text"] and col2.button("Hapus dari resep", key=f"del_free_{column}"):
        r["free_text"].pop(column)
        st.rerun()


# ---------------------------------------------------------------- outlier

METHOD_TEXT = {
    "iqr": "**IQR (Tukey)**: di luar Q1 − k·IQR … Q3 + k·IQR. Cocok untuk data kira-kira simetris.",
    "iqr_log": "**IQR-log**: IQR di skala log10. Cocok untuk harga/luas/pendapatan (miring ke kanan). Nol tidak dinilai.",
    "mad": "**MAD**: |0,6745·(x − median)/MAD| > k. Lebih tahan outlier ekstrem daripada IQR. "
           "Gagal bila lebih dari separuh nilai sama (MAD = 0).",
    "mad_log": "**MAD-log**: MAD di skala log10. Nol tidak dinilai.",
}


def show_outliers() -> None:
    common.header("outlier",
                  "Bandingkan empat metode batas wajar untuk satu kolom, lihat baris yang kena, "
                  "lalu pilih **aksi**. Batas dihitung dari train dan dipakai sama di test.",
                  "Kolom angka kontinu (luas, harga, pengeluaran). Mulai dengan aksi **flag** (tidak mengubah).",
                  "**drop** membuang baris train; **clip** mengubah nilai. Kolom diskret (jumlah tanggungan) "
                  "dan kolom yang mayoritas nilainya sama sering salah ditandai.")
    if not common.need_data():
        return
    r, data = common.recipe(), common.prepared()
    numbers = common.number_columns(data)
    if not numbers:
        st.info("Tidak ada kolom angka.")
        return
    column = st.selectbox("Kolom angka (sudah melewati langkah penjaga → kosong)", numbers, key="outlier_column")
    values = data[column]
    st.caption(f"median {values.median():,.2f} · p99 {values.quantile(0.99):,.2f} · maks {values.max():,.2f} · "
               f"nol {int((values == 0).sum())} · kosong {int(values.isna().sum())}")
    for text in METHOD_TEXT.values():
        st.markdown(f"- {text}")
    st.dataframe(outlier_tools.compare_methods(values), width="stretch", hide_index=True)

    spec = r["outliers"].get(column, {"method": "iqr_log", "factor": 1.5, "action": "flag"})
    left, middle, right = st.columns(3)
    method = left.selectbox("Metode", list(rcp.OUTLIER_METHODS), list(rcp.OUTLIER_METHODS).index(spec["method"]),
                            format_func=lambda m: rcp.OUTLIER_METHODS[m])
    factor = middle.number_input("k", 0.5, 10.0, float(spec["factor"]), 0.5)
    action = right.selectbox("Aksi", list(rcp.OUTLIER_ACTIONS), list(rcp.OUTLIER_ACTIONS).index(spec["action"]),
                             format_func=lambda a: rcp.OUTLIER_ACTIONS[a])
    low, high = outlier_tools.outlier_bounds(values, method, factor)
    outside = outlier_tools.outside_rows(data, column, method, factor)
    st.write(f"Batas: **{low:,.2f} … {high:,.2f}** → {len(outside)} baris di luar batas")
    st.dataframe(outside.sort_values(column).iloc[np.r_[0:5, -5:0]] if len(outside) > 10 else outside,
                 width="stretch")

    groups = [c for c in [r["label"]] if c] + list(st.session_state.get("laya_answers", {}))
    if groups:
        group = st.selectbox("Bandingkan per kelompok (konteks)", [None, *groups])
        if group:
            source = data.assign(**{group: common.train()[group]} if group == r["label"]
                                 else {group: st.session_state["laya_answers"][group]})
            st.dataframe(outlier_tools.by_group(source, column, group, method, factor), width="stretch",
                         hide_index=True)
            st.caption("Nilai yang wajar di satu kelompok bisa aneh di kelompok lain. Jawaban Laya hanya "
                       "dipakai untuk melihat, tidak masuk pipeline.")
    col1, col2 = st.columns(2)
    if col1.button("Simpan ke resep", type="primary", key="save_outlier"):
        r["outliers"][column] = {"method": method, "factor": factor, "action": action}
        common.saved(f"Outlier {column}: {method}, {action}")
        st.rerun()
    if column in r["outliers"] and col2.button("Hapus dari resep", key="del_outlier"):
        r["outliers"].pop(column)
        st.rerun()

    with st.expander("Outlier multivariat (IsolationForest, hanya melihat)"):
        picked = st.multiselect("Kolom", numbers, numbers[:4], key="iforest_columns")
        share = st.slider("Perkiraan porsi aneh", 0.001, 0.1, 0.01, format="%.3f")
        if picked and st.button("Jalankan IsolationForest"):
            flags = outlier_tools.isolation_flags(data, picked, share)
            st.write(f"{int(flags.sum())} baris ditandai (tidak masuk resep).")
            st.dataframe(data.loc[flags, picked].head(50), width="stretch")
    if r["outliers"]:
        st.caption(f"Di resep: {r['outliers']}")


# ---------------------------------------------------------------- baris

def show_rows() -> None:
    common.header("baris",
                  "Aturan **membuang baris train**: duplikat persis, terlalu banyak kosong, atau aturan logika "
                  "buatanmu. Baris test **tidak pernah** dibuang.",
                  "Baris rusak total (20+ kolom kosong), data mustahil (luas bangunan 100.000 m²).",
                  "Setiap baris yang dibuang mengubah distribusi. Aturan logika yang terlalu luas membuang data sah.")
    if not common.need_data():
        return
    r, data = common.recipe(), common.prepared()
    findings = audit.row_findings(common.train(), r["label"])
    r["drop_duplicates"] = st.checkbox(f"Buang duplikat persis ({findings['duplikat persis']} baris)",
                                       r["drop_duplicates"])
    missing = data.isna().sum(axis=1)
    use_limit = st.checkbox("Buang baris dengan terlalu banyak kosong", r["max_missing_per_row"] is not None)
    if use_limit:
        limit = st.slider("Kosong per baris lebih dari …", 1, data.shape[1], r["max_missing_per_row"] or 10)
        st.caption(f"→ {int((missing > limit).sum())} baris dibuang")
        r["max_missing_per_row"] = limit
    else:
        r["max_missing_per_row"] = None

    st.subheader("Aturan logika (pandas eval)")
    st.caption("Nama kolom dengan spasi ditulis di antara backtick, mis. `` `Luas Bangunan Rumah (m2)` > 100000 ``")
    rule = st.text_input("Aturan: baris yang COCOK dibuang", key="rule_input")
    if rule:
        try:
            hits = audit.check_rule(data, rule)
            st.write(f"{len(hits)} baris cocok")
            st.dataframe(hits.head(20), width="stretch")
            if st.button("Tambahkan aturan"):
                r["drop_queries"].append(rule)
                st.rerun()
        except Exception as error:   # aturan buatan pengguna: tampilkan pesan, jangan crash
            st.error(f"Aturan tidak bisa dibaca: {error}")
    for i, saved_rule in enumerate(r["drop_queries"]):
        left, right = st.columns([5, 1])
        left.code(saved_rule)
        if right.button("hapus", key=f"del_rule_{i}"):
            r["drop_queries"].pop(i)
            st.rerun()
