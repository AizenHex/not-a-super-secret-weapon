"""Tab Operasi (panel operasi + ubah langkah), Ejaan (clustering), Isian bebas, Outlier."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from core import outlier_tools, text_tools
from core.ops import MISSING_MARKERS, OPS, SENTINEL_FLOOR, summary
from ui import common

METHODS = {"iqr": "IQR (Tukey)", "iqr_log": "IQR skala log", "mad": "MAD", "mad_log": "MAD skala log"}
ACTIONS = {"clip": "clip: potong ke batas (train + test)", "nan": "nan: jadikan kosong (train + test)",
           "drop": "drop: buang baris (train saja)"}


# ---------------------------------------------------------------- panel operasi

def form(op: str, data: pd.DataFrame, old: dict, key: str) -> dict | None:
    """Widget parameter untuk satu operasi. Kembalikan params, atau None bila belum lengkap."""
    columns, texts, numbers = list(data.columns), common.object_columns(data), common.number_columns(data)

    def many(title, options, default=()):
        return st.multiselect(title, options, [c for c in old.get("columns", default) if c in options], key=f"{key}_cols")

    def one(title, options):
        if not options:
            st.info("Tidak ada kolom yang cocok.")
            return None
        index = options.index(old["column"]) if old.get("column") in options else 0
        return st.selectbox(title, options, index, key=f"{key}_col")

    if op in ("drop_columns",):
        picked = many("Kolom", columns)
        return {"columns": picked} if picked else None
    if op == "rename_column":
        column = one("Kolom", columns)
        new = st.text_input("Nama baru", old.get("new", ""), key=f"{key}_new")
        return {"column": column, "new": new} if column and new else None
    if op == "to_number":
        picked = many("Kolom teks berisi angka", texts)
        indonesian = st.checkbox("Format Indonesia (titik = ribuan, koma = desimal)", old.get("indonesian", True),
                                 key=f"{key}_id")
        return {"columns": picked, "indonesian": indonesian} if picked else None
    if op == "tidy_text":
        picked = many("Kolom teks", texts, texts)
        markers = st.text_input("Penanda kosong (pisah |)", " | ".join(old.get("markers", MISSING_MARKERS)),
                                key=f"{key}_mk")
        marks = sorted({m.strip().lower() for m in markers.split("|")} | {""})
        return {"columns": picked, "markers": marks} if picked else None
    if op == "change_case":
        picked = many("Kolom teks", texts)
        cases = ["lower", "upper", "title"]
        case = st.radio("Jadi", cases, cases.index(old.get("case", "lower")), horizontal=True, key=f"{key}_case",
                        format_func={"lower": "huruf kecil", "upper": "HURUF BESAR", "title": "Huruf Judul"}.get)
        return {"columns": picked, "case": case} if picked else None
    if op == "map_values":
        column = one("Kolom", texts)
        if column is None:
            return None
        st.caption("Satu baris per pasangan: `nilai asli => nilai baku`")
        text = "\n".join(f"{a} => {b}" for a, b in old.get("mapping", {}).items())
        text = st.text_area("Peta", text, height=150, key=f"{key}_map")
        mapping = {}
        for line in text.splitlines():
            if "=>" in line:
                left, right = line.split("=>", 1)
                mapping[left.strip()] = right.strip()
        return {"column": column, "mapping": mapping} if mapping else None
    if op == "free_text":
        column = one("Kolom isian bebas", texts)
        st.caption("Peta kata disarankan dari tab **Isian bebas**. Di sini: `kata salah => kata baku` per baris.")
        text = st.text_area("Peta kata", "\n".join(f"{a} => {b}" for a, b in old.get("words", {}).items()),
                            key=f"{key}_words")
        words = {a.strip(): b.strip() for a, b in (l.split("=>", 1) for l in text.splitlines() if "=>" in l)}
        return {"column": column, "words": words} if column else None
    if op in ("sentinel_to_nan", "zero_to_nan", "log_transform"):
        picked = many("Kolom angka", numbers)
        params = {"columns": picked}
        if op == "sentinel_to_nan":
            params["floor"] = int(st.number_input("Batas bawah nilai penjaga", value=int(old.get("floor", SENTINEL_FLOOR)),
                                                  step=1_000_000, key=f"{key}_floor"))
        return params if picked else None
    if op == "fill_constant":
        picked = many("Kolom", columns)
        value = st.text_input("Nilai pengisi", str(old.get("value", "Tidak")), key=f"{key}_val")
        if picked and all(c in numbers for c in picked):
            value = pd.to_numeric(value, errors="coerce")
            value = None if pd.isna(value) else float(value)
        return {"columns": picked, "value": value} if picked and value is not None else None
    if op == "impute":
        strategy = st.radio("Strategi", ["median", "mean", "mode"], ["median", "mean", "mode"].index(old.get("strategy", "median")),
                            horizontal=True, key=f"{key}_st")
        options = [c for c in (numbers if strategy != "mode" else columns) if data[c].isna().any() and data[c].notna().any()]
        picked = many("Kolom yang masih kosong", options)
        return {"columns": picked, "strategy": strategy} if picked else None
    if op == "drop_duplicates":
        return {}
    if op == "drop_missing_rows":
        limit = st.slider("Kosong per baris maksimal", 1, data.shape[1], int(old.get("max_missing", 10)), key=f"{key}_max")
        return {"max_missing": limit}
    if op == "drop_rows_query":
        st.caption("Nama kolom berspasi diapit backtick: `` `Luas Bangunan Rumah (m2)` > 100000 ``")
        rule = st.text_input("Baris yang COCOK dibuang", old.get("rule", ""), key=f"{key}_rule")
        return {"rule": rule} if rule else None
    if op == "outlier":
        column = one("Kolom angka", numbers)
        left, middle, right = st.columns(3)
        method = left.selectbox("Metode", list(METHODS), list(METHODS).index(old.get("method", "iqr_log")),
                                format_func=METHODS.get, key=f"{key}_m")
        factor = middle.number_input("k", 0.5, 10.0, float(old.get("factor", 1.5)), 0.5, key=f"{key}_k")
        action = right.selectbox("Aksi", list(ACTIONS), list(ACTIONS).index(old.get("action", "clip")),
                                 format_func=ACTIONS.get, key=f"{key}_a")
        return {"column": column, "method": method, "factor": factor, "action": action} if column else None
    if op == "scale":
        picked = many("Kolom angka", numbers)
        method = st.radio("Cara", ["standard", "minmax"], ["standard", "minmax"].index(old.get("method", "standard")),
                          horizontal=True, key=f"{key}_sc")
        return {"columns": picked, "method": method} if picked else None
    if op in ("onehot", "materials_onehot"):
        column = one("Kolom", texts)
        share = st.slider("Porsi minimal (lebih jarang → _lainnya)", 0.0, 0.1, float(old.get("min_share", 0.01)), 0.005,
                          format="%.3f", key=f"{key}_share")
        if column:
            st.caption(f"{data[column].nunique()} nilai unik")
        return {"column": column, "min_share": share} if column else None
    if op == "ordinal":
        column = one("Kolom", texts)
        if column is None:
            return None
        values = sorted(data[column].dropna().unique(), key=str)
        text = st.text_area("Urutan dari terendah (kosongkan = abjad)", "\n".join(map(str, old.get("order", []))),
                            height=150, key=f"{key}_order", help="Nilai yang ada: " + ", ".join(map(str, values[:40])))
        order = [line.strip() for line in text.splitlines() if line.strip()]
        missing = set(values) - set(order)
        if order and missing:
            st.warning(f"Tidak ada di urutan (jadi −1): {sorted(missing, key=str)[:10]}")
        return {"column": column, "order": order}
    if op == "skrub_encoder":
        column = one("Kolom teks", texts)
        kinds = ["string", "minhash", "gap"]
        kind = st.radio("Encoder", kinds, kinds.index(old.get("kind", "string")), horizontal=True, key=f"{key}_kind")
        n = st.slider("Jumlah kolom hasil", 2, 30, int(old.get("n_components", 10)), key=f"{key}_n")
        return {"column": column, "kind": kind, "n_components": n} if column else None
    if op == "autogluon_features":
        if not common.label():
            st.error("Butuh kolom label (tab Data).")
            return None
        return {}
    return None


def show_operations() -> None:
    common.header("operasi",
                  "Panel operasi ala **Data Wrangler**: pilih operasi, isi parameter, lihat **sebelum → sesudah**, "
                  "lalu tambahkan sebagai langkah. Setiap langkah = satu blok kode pandas lurus (tanpa fungsi).",
                  "Semua perbaikan yang tidak diusulkan otomatis di tab Temuan.",
                  "Langkah yang membuang baris/kolom. Baca 'kapan berbahaya' setiap operasi.")
    if not common.need_data():
        return
    data = common.current()
    groups = sorted({spec["group"] for spec in OPS.values()}, key=["Kolom", "Teks", "Kosong", "Baris", "Angka", "Encoding"].index)
    group = st.radio("Kelompok", groups, horizontal=True, key="op_group")
    names = [name for name, spec in OPS.items() if spec["group"] == group]
    op = st.selectbox("Operasi", names, format_func=lambda n: OPS[n]["title"], key="op_name")
    common.op_help(op)
    params = form(op, data, {}, f"new_{op}")
    if params is not None:
        common.preview_step(op, params, f"new_{op}")

    steps = common.project()["steps"]
    if steps:
        st.divider()
        st.subheader("Ubah langkah yang sudah ada")
        index = st.selectbox("Langkah", range(len(steps)), format_func=lambda i: f"{i + 1}. {summary(steps[i])}",
                             key="edit_index")
        step = steps[index]
        before = common.state(index)["train"]
        before = before.drop(columns=[common.label()]) if common.label() in before else before
        st.caption("Pratinjau dihitung di data sesudah langkah sebelumnya.")
        params = form(step["op"], before, step["params"], f"edit_{index}_{step['op']}")
        if params is not None:
            common.preview_step(step["op"], params, f"edit_{index}", replace=index)
        with st.expander("Parameter mentah (JSON)"):
            st.code(json.dumps(step["params"], ensure_ascii=False, indent=2), language="json")


# ---------------------------------------------------------------- ejaan

def approval(table: pd.DataFrame, key: str, default_ok: bool) -> dict:
    table = table.copy()
    table.insert(0, "setuju", default_ok if "peringatan" not in table else default_ok & (table["peringatan"] == ""))
    edited = st.data_editor(table, width="stretch", hide_index=True, key=key,
                            disabled=[c for c in table.columns if c not in ("setuju", "usulan baku")])
    chosen = edited[edited["setuju"]]
    return dict(zip(chosen["asli"], chosen["usulan baku"]))


def show_spelling() -> None:
    common.header("ejaan",
                  "Clustering ala **OpenRefine** untuk kolom kategori: satukan ejaan berbeda dari nilai yang sama "
                  "(`'Kab. Pati'` = `'Kabupaten Pati'`). Hasilnya satu langkah **Satukan ejaan (peta nilai)**.",
                  "**Key collision (fingerprint)** aman: beda huruf besar, spasi, tanda baca, urutan kata, singkatan.",
                  "**Nearest neighbor** dan **skrub.deduplicate** bisa menggabung nilai berbeda (`Kota Malang` → "
                  "`Kota Magelang`). Tidak dicentang otomatis; periksa satu per satu.")
    if not common.need_data():
        return
    data = common.current()
    texts = common.object_columns(data)
    if not texts:
        st.info("Tidak ada kolom teks.")
        return
    column = st.selectbox("Kolom", texts, key="spell_column")
    st.caption(f"{data[column].nunique():,} nilai unik. Rapikan teks dulu (tab Temuan/Operasi) agar spasi tidak "
               "membuat nilai kembar.")
    method = st.radio("Metode", ["key collision: fingerprint (aman)", "nearest neighbor: rapidfuzz (periksa)",
                                 "skrub.deduplicate (opsional, periksa)"], horizontal=True, key="spell_method")
    if method.startswith("key"):
        proposals, default_ok = text_tools.fingerprint_groups(data[column]), True
    elif method.startswith("nearest"):
        left, right = st.columns(2)
        min_count = left.slider("Nilai 'sering' = muncul minimal … kali", 2, 200, 20, key="nn_min")
        cutoff = right.slider("Skor minimal (token_sort_ratio)", 70, 100, 88, key="nn_cut")
        proposals, default_ok = text_tools.neighbor_groups(data[column], min_count, cutoff), False
    else:
        clusters = st.number_input("Jumlah klaster (0 = otomatis)", 0, max(data[column].nunique(), 1), 0)
        st.caption("Data uji: 640 nilai ≈ 3 detik, 1.961 nilai ≈ 70 detik.")
        if st.button("Jalankan skrub.deduplicate"):
            st.session_state[f"dedup_{column}"] = text_tools.deduplicate_map(data[column], int(clusters) or None)
        proposals = st.session_state.get(f"dedup_{column}", pd.DataFrame(columns=["asli", "usulan baku", "jumlah"]))
        default_ok = False
    if proposals.empty:
        st.write("Tidak ada usulan.")
        return
    mapping = approval(proposals, f"spell_{column}_{method[:4]}", default_ok)
    st.write(f"{len(mapping)} ejaan dicentang.")
    if mapping:
        common.preview_step("map_values", {"column": column, "mapping": mapping}, f"spell_{column}")


# ---------------------------------------------------------------- isian bebas

def show_free_text() -> None:
    common.header("isian",
                  "Kolom isian bebas berisi **beberapa bahan** dalam satu sel dengan salah ketik "
                  "(`'Genteng & asbas.'`). Sel dipecah per bahan, salah ketik dibetulkan **per kata**, lalu "
                  "ditulis ulang sebagai daftar baku `'asbes, genteng'` (kolom tetap satu, nama tetap).",
                  "Kolom bahan atap/tembok/lantai, daftar fasilitas.",
                  "Kata sah yang mirip kata lain (`tanpa`/`tanah`, `batu`/`bata`). Usulan skor < 85 tidak dicentang "
                  "otomatis; kosakata dihitung juga dari kolom terkait.")
    if not common.need_data():
        return
    data = common.current()
    texts = common.object_columns(data)
    if not texts:
        st.info("Tidak ada kolom teks.")
        return
    column = st.selectbox("Kolom isian bebas", texts,
                          index=next((i for i, c in enumerate(texts) if c.startswith("Bahan")), 0), key="free_column")
    related = st.multiselect("Kolom terkait (kosakata digabung)", [c for c in texts if c != column],
                             [c for c in texts if c != column and c.startswith("Bahan")], key="free_related")
    left, right = st.columns(2)
    min_count = left.slider("Kata baku = muncul minimal … kali", 5, 200, 30, key="free_min")
    cutoff = right.slider("Skor kemiripan minimal (fuzz.ratio)", 60, 95, 75, key="free_cut")

    words = text_tools.word_map(data[column], min_count, cutoff, [data[c] for c in related])
    st.subheader(f"Usulan salah ketik ({len(words)})")
    edited = st.data_editor(words, width="stretch", hide_index=True, key=f"words_{column}",
                            disabled=["kata asli", "skor", "jumlah"])
    st.subheader("Kata baku yang mirip (kandidat sinonim)")
    st.caption("Tidak pernah otomatis. Centang bila 'kata 2' sama maknanya dengan 'kata 1' (genteng = genting).")
    similar = text_tools.similar_vocabulary(data[column], min_count)
    similar.insert(0, "samakan 2 → 1", False)
    laya = st.session_state.get("laya_same", {})
    if laya:
        similar["Laya: sama?"] = [laya.get((a, b)) for a, b in zip(similar["kata 1"], similar["kata 2"])]
    synonyms = st.data_editor(similar, width="stretch", hide_index=True, key=f"syn_{column}",
                              disabled=[c for c in similar.columns if c != "samakan 2 → 1"])
    mapping = dict(zip(edited.loc[edited["setuju"], "kata asli"], edited.loc[edited["setuju"], "usulan"]))
    for first, second in synonyms.loc[synonyms["samakan 2 → 1"], ["kata 1", "kata 2"]].to_numpy():
        mapping[second] = first
    mapping = {w: mapping.get(u, u) for w, u in mapping.items()}          # gentng→genteng→genting
    st.subheader("Pratinjau langkah")
    common.preview_step("free_text", {"column": column, "words": mapping}, f"free_{column}")


# ---------------------------------------------------------------- outlier

METHOD_TEXT = {
    "iqr": "**IQR (Tukey)**: di luar Q1 − k·IQR … Q3 + k·IQR. Cocok untuk data kira-kira simetris.",
    "iqr_log": "**IQR-log**: IQR di skala log10. Cocok untuk harga/luas/pendapatan (miring ke kanan). Nol tidak dinilai.",
    "mad": "**MAD**: |x − median| / (MAD/0,6745) > k. Lebih tahan outlier ekstrem. Gagal bila separuh nilai sama.",
    "mad_log": "**MAD-log**: MAD di skala log10. Nol tidak dinilai.",
}


def show_outliers() -> None:
    common.header("outlier",
                  "Bandingkan empat metode batas wajar untuk satu kolom, lihat baris yang kena, pilih **aksi**, "
                  "lalu tambahkan sebagai langkah. Batas dihitung dari train dan dipakai sama di test.",
                  "Kolom angka kontinu (luas, harga, pengeluaran), setelah nilai penjaga dijadikan kosong.",
                  "**drop** membuang baris train; **clip** mengubah nilai. Kolom diskret dan kolom yang mayoritas "
                  "nilainya sama sering salah ditandai.")
    if not common.need_data():
        return
    data = common.current()
    numbers = common.number_columns(data)
    if not numbers:
        st.info("Tidak ada kolom angka.")
        return
    column = st.selectbox("Kolom angka", numbers, key="outlier_column")
    values = data[column]
    st.caption(f"median {values.median():,.2f} · p99 {values.quantile(0.99):,.2f} · maks {values.max():,.2f} · "
               f"nol {int((values == 0).sum())} · kosong {int(values.isna().sum())}")
    for text in METHOD_TEXT.values():
        st.markdown(f"- {text}")
    st.dataframe(outlier_tools.compare_methods(values), width="stretch", hide_index=True)

    left, middle, right = st.columns(3)
    method = left.selectbox("Metode", list(METHODS), 1, format_func=METHODS.get, key="out_m")
    factor = middle.number_input("k", 0.5, 10.0, 1.5 if method.startswith("iqr") else 3.5, 0.5, key=f"out_k_{method}")
    action = right.selectbox("Aksi", list(ACTIONS), format_func=ACTIONS.get, key="out_a")
    low, high = outlier_tools.outlier_bounds(values, method, factor)
    outside = outlier_tools.outside_rows(data, column, method, factor)
    st.write(f"Batas: **{low:,.2f} … {high:,.2f}** → {len(outside)} baris di luar batas")
    st.dataframe(outside.sort_values(column).iloc[np.r_[0:5, -5:0]] if len(outside) > 10 else outside,
                 width="stretch")

    label = common.label()
    groups = [c for c in [label] if c] + list(st.session_state.get("laya_answers", {}))
    if groups:
        group = st.selectbox("Bandingkan per kelompok (konteks)", [None, *groups], key="out_group")
        if group:
            source = (data.assign(**{group: common.state()["train"][group]}) if group == label
                      else data.assign(**{group: st.session_state["laya_answers"][group]}))
            st.dataframe(outlier_tools.by_group(source, column, group, method, factor), width="stretch", hide_index=True)
            st.caption("Nilai wajar di satu kelompok bisa aneh di kelompok lain. Jawaban Laya hanya untuk melihat.")
    st.subheader("Pratinjau langkah")
    common.preview_step("outlier", {"column": column, "method": method, "factor": factor, "action": action},
                        f"out_{column}")

    with st.expander("Outlier multivariat (IsolationForest, hanya melihat)"):
        picked = st.multiselect("Kolom", numbers, numbers[:4], key="iforest_columns")
        share = st.slider("Perkiraan porsi aneh", 0.001, 0.1, 0.01, format="%.3f")
        if picked and st.button("Jalankan IsolationForest"):
            flags = outlier_tools.isolation_flags(data, picked, share)
            st.write(f"{int(flags.sum())} baris ditandai (tidak jadi langkah).")
            st.dataframe(data.loc[flags, picked].head(50), width="stretch")
