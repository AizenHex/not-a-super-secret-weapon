"""Tab Isi kosong, Label, Train vs test, Encoding, Split."""
from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from core import audit
from core import recipe as rcp
from core.codegen import build_script
from core.runner import load_script
from ui import common


# ---------------------------------------------------------------- isi kosong

def show_impute() -> None:
    common.header("isi",
                  "Pilih cara mengisi sel kosong **per kolom** (sklearn `SimpleImputer`). Median/modus dipelajari "
                  "dari train, lalu dipakai sama di test.",
                  "Kolom dengan sedikit kosong. Angka miring → median; kategori → modus atau konstanta 'Tidak diketahui'.",
                  "Kolom yang kosongnya banyak: mengisi menyembunyikan informasi. Pertimbangkan biarkan kosong "
                  "(banyak model, mis. HistGradientBoosting/LightGBM, menerima NaN).")
    if not common.need_data():
        return
    r, data = common.recipe(), common.prepared()
    missing = data.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("Tidak ada sel kosong setelah langkah sebelumnya.")
        return
    table = pd.DataFrame({
        "kolom": missing.index,
        "tipe": ["angka" if pd.api.types.is_numeric_dtype(data[c]) else "teks" for c in missing.index],
        "kosong": missing.to_numpy(),
        "kosong %": (missing / len(data) * 100).round(1).to_numpy(),
        "strategi": [r["impute"].get(c, {}).get("strategy", "none") for c in missing.index],
        "nilai konstanta": [str(r["impute"].get(c, {}).get("value", "")) for c in missing.index],
    })
    st.caption("median/mean hanya untuk angka. Untuk teks pakai most_frequent atau constant.")
    edited = st.data_editor(table, width="stretch", hide_index=True, key="impute_editor",
                            disabled=["kolom", "tipe", "kosong", "kosong %"],
                            column_config={"strategi": st.column_config.SelectboxColumn(
                                options=list(rcp.IMPUTE_STRATEGIES), required=True)})
    wrong = edited[(edited["tipe"] == "teks") & edited["strategi"].isin(["median", "mean"])]
    if not wrong.empty:
        st.error(f"median/mean tidak bisa untuk kolom teks: {list(wrong['kolom'])}")
    elif st.button("Simpan ke resep", type="primary", key="save_impute"):
        r["impute"] = {}
        for row in edited.itertuples(index=False):
            if row.strategi == "none":
                continue
            spec = {"strategy": row.strategi}
            if row.strategi == "constant":
                value = row._5
                spec["value"] = float(value or 0) if row.tipe == "angka" else (value or "Tidak diketahui")
            r["impute"][row.kolom] = spec
        common.saved(f"{len(r['impute'])} kolom akan diisi")
        st.rerun()


# ---------------------------------------------------------------- label

def show_label() -> None:
    common.header("label",
                  "Periksa kolom label: distribusi, duplikat fitur dengan label berbeda, dan (opsional) "
                  "label yang mungkin salah menurut **cleanlab**.",
                  "Klasifikasi dengan label dari isian manusia.",
                  "cleanlab hanya menebak dari model; jangan ubah label otomatis. Tab ini hanya menandai.")
    if not common.need_data():
        return
    r, data = common.recipe(), common.train()
    if not r["label"]:
        st.info("Pilih kolom label di tab **Data**.")
        return
    counts = data[r["label"]].value_counts(dropna=False).sort_index()
    st.bar_chart(counts)
    st.caption(f"kosong: {int(data[r['label']].isna().sum())} · kelas terkecil: {counts.min()} baris")
    findings = audit.row_findings(data, r["label"])
    st.metric("grup duplikat fitur dengan label berbeda", findings["grup duplikat dengan label berbeda"])

    with st.expander("cleanlab Datalab (opsional, ±1 menit)"):
        st.markdown("Model `HistGradientBoostingClassifier` dilatih 5-fold di kolom angka hasil pipeline, "
                    "lalu cleanlab mencari baris yang labelnya paling tidak cocok dengan prediksi.")
        if st.button("Jalankan cleanlab"):
            from cleanlab import Datalab
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.model_selection import cross_val_predict

            clean = common.run_pipeline()["train"]
            features = clean.drop(columns=[r["label"]]).select_dtypes("number")
            labels = clean[r["label"]]
            probabilities = cross_val_predict(HistGradientBoostingClassifier(random_state=0), features, labels,
                                              cv=5, method="predict_proba")
            lab = Datalab(data=clean[[r["label"]]], label_name=r["label"], verbosity=0)
            lab.find_issues(pred_probs=probabilities, issue_types={"label": {}})
            issues = lab.get_issues("label")
            flagged = issues[issues["is_label_issue"]].sort_values("label_score")
            st.write(f"{len(flagged)} baris ditandai ({len(flagged) / len(clean):.1%}). Label tidak diubah.")
            st.dataframe(flagged.join(clean).head(100), width="stretch")


# ---------------------------------------------------------------- train vs test

def show_test() -> None:
    common.header("test",
                  "Bandingkan train dan test sebelum pipeline dipakai untuk test.",
                  "Selalu, bila punya file test.",
                  "Kategori baru di test → di one-hot jadi 'infrequent'/semua 0; di kode angka jadi −1.")
    if not common.need_data():
        return
    test = common.test()
    if test is None:
        st.info("Muat file test di tab **Data** untuk membandingkan.")
        return
    r, train = common.recipe(), common.features()
    missing = sorted(set(train.columns) - set(test.columns))
    extra = sorted(set(test.columns) - set(train.columns) - {r["label"]})
    st.write(f"Kolom tidak ada di test: **{missing or '-'}** · kolom lebih di test: **{extra or '-'}**")
    module = common.pipeline()
    train_ready, test_ready = common.prepared(), module.prepare(test.reindex(columns=train.columns))

    rows = []
    for column in common.object_columns(train_ready):
        new = set(test_ready[column].dropna()) - set(train_ready[column].dropna())
        if new:
            rows.append({"kolom": column, "nilai baru": len(new),
                         "baris test terdampak": int(test_ready[column].isin(new).sum()),
                         "contoh": ", ".join(map(repr, sorted(new)[:5]))})
    st.subheader("Kategori di test yang tidak ada di train")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    from scipy.stats import ks_2samp
    shift = []
    for column in common.number_columns(train_ready):
        a, b = train_ready[column].dropna(), test_ready[column].dropna()
        if len(a) and len(b):
            result = ks_2samp(a, b)
            shift.append({"kolom": column, "KS": round(result.statistic, 3), "p": result.pvalue,
                          "median train": a.median(), "median test": b.median()})
    st.subheader("Pergeseran distribusi angka (scipy ks_2samp)")
    st.caption("KS mendekati 0 = mirip. p kecil dengan KS kecil biasanya hanya efek jumlah data besar.")
    st.dataframe(pd.DataFrame(shift).sort_values("KS", ascending=False), width="stretch", hide_index=True)


# ---------------------------------------------------------------- encoding

ENCODER_GUIDE = pd.DataFrame([
    ("none", "Biarkan teks", "Model yang menerima kategori langsung (CatBoost, LightGBM categorical)", "Model sklearn biasa menolak teks"),
    ("onehot", "Satu kolom 0/1 per nilai (OneHotEncoder)", "Kategori sedikit (≤ ±20) tanpa urutan", "Ratusan nilai → ratusan kolom"),
    ("ordinal", "Kode angka abjad (OrdinalEncoder)", "Model pohon; nilai banyak", "Model linear membaca kode sebagai urutan"),
    ("ordered", "Kode angka dengan urutanmu", "Tingkat pendidikan, rentang pendapatan", "Urutan salah = informasi salah"),
    ("materials", "Satu kolom 0/1 per bahan + _lainnya", "Isian bebas yang sudah dibakukan di tab Isian bebas", "Tanpa pembakuan → kolom untuk tiap salah ketik"),
    ("string", "skrub StringEncoder: TF-IDF n-gram + SVD", "Teks pendek bervariasi banyak (nama prodi, kota)", "Kolom hasil = angka tanpa nama yang bisa dibaca"),
    ("minhash", "skrub MinHashEncoder", "Sangat banyak nilai unik, model pohon", "Kolom tanpa makna; tidak cocok untuk model linear"),
    ("gap", "skrub GapEncoder: topik laten", "Ingin kolom yang bisa ditafsir sebagai topik", "Lambat; hasil tergantung random_state"),
], columns=["encoder", "apa", "cocok", "berbahaya"])


def preview_columns(column: str, spec: dict) -> list[str]:
    """Nama kolom baru dari kode pipeline sendiri (fit di data train setelah prepare)."""
    r = copy.deepcopy(common.recipe())
    r["encoding"] = {column: spec}
    r["impute"] = {}
    module = load_script(build_script(r, {}))
    data = module.prepare(common.features())
    data = module.encode_materials(data, module.fit_materials(data))
    out = module.build_preprocessor().fit(data).get_feature_names_out()
    return [name for name in out if name not in data.columns or name == column]


def show_encoding() -> None:
    common.header("encoding",
                  "Ubah kolom teks jadi angka untuk model. **Default: tidak di-encode.** Nama kolom baru "
                  "ditampilkan dulu sebelum disimpan; kolom asli lain tidak pernah diganti nama.",
                  "Sebelum melatih model sklearn yang butuh angka.",
                  "Encoder skrub menghasilkan kolom angka tanpa makna yang bisa dibaca. Pakai hanya bila sadar.")
    if not common.need_data():
        return
    r, data = common.recipe(), common.prepared()
    st.dataframe(ENCODER_GUIDE, width="stretch", hide_index=True)
    texts = common.object_columns(data)
    if not texts:
        st.info("Tidak ada kolom teks.")
        return
    column = st.selectbox("Kolom", texts, key="encoding_column")
    st.caption(f"{data[column].nunique()} nilai unik · {int(data[column].isna().sum())} kosong")
    spec = r["encoding"].get(column, {"method": "none"})
    methods = ["none", *rcp.ENCODERS]
    method = st.selectbox("Encoder", methods, methods.index(spec["method"]),
                          format_func=lambda m: "tidak di-encode" if m == "none" else rcp.ENCODERS[m])
    new_spec = {"method": method}
    if method in ("onehot", "materials"):
        new_spec["min_share"] = st.slider("Nilai yang muncul di < … baris jadi 'jarang'", 0.0, 0.1,
                                          float(spec.get("min_share", 0.01)), 0.005, format="%.3f")
    if method == "materials" and column not in r["free_text"]:
        st.warning("Kolom ini belum dibakukan di tab **Isian bebas**: setiap ejaan akan jadi bahan sendiri.")
    if method == "ordered":
        values = sorted(data[column].dropna().unique(), key=str)
        order = spec.get("order") or values
        text = st.text_area("Urutan (satu nilai per baris, dari terendah)", "\n".join(map(str, order)), height=200)
        new_spec["order"] = [line for line in text.splitlines() if line.strip()]
        unknown = set(values) - set(new_spec["order"])
        if unknown:
            st.warning(f"Tidak ada di urutan (akan jadi −1): {sorted(unknown, key=str)[:10]}")
    if method in ("string", "minhash", "gap"):
        new_spec["n_components"] = st.slider("Jumlah kolom hasil", 2, 30, int(spec.get("n_components", 10)))

    if method != "none":
        if st.button("Pratinjau kolom baru"):
            with st.spinner("Fit encoder di train…"):
                st.session_state["encoding_preview"] = (column, new_spec, preview_columns(column, new_spec))
        preview = st.session_state.get("encoding_preview")
        if preview and preview[0] == column and preview[1] == new_spec:
            names = preview[2]
            st.write(f"Kolom **{column}** diganti oleh **{len(names)} kolom**:")
            st.code(", ".join(names))
            if st.button("Simpan ke resep", type="primary", key="save_encoding"):
                r["encoding"][column] = new_spec
                common.saved(f"{column}: {method}")
                st.rerun()
        else:
            st.caption("Klik **Pratinjau kolom baru** dulu; tombol simpan muncul setelah pratinjau.")
    elif column in r["encoding"] and st.button("Kembalikan ke tidak di-encode"):
        r["encoding"].pop(column)
        st.rerun()
    if r["encoding"]:
        st.caption("Di resep: " + "; ".join(f"{c} → {s['method']}" for c, s in r["encoding"].items()))


# ---------------------------------------------------------------- split

def show_split() -> None:
    common.header("split",
                  "Bagi train bersih untuk validasi. Split dilakukan **setelah** pembersihan train, dan file "
                  "test tidak ikut di-split.",
                  "Stratified untuk klasifikasi; k-fold bila data sedikit.",
                  "Split acak untuk data deret waktu atau data berkelompok (satu orang beberapa baris) → bocor.")
    if not common.need_data():
        return
    r = common.recipe()
    split = r["split"]
    methods = list(rcp.SPLITS)
    split["method"] = st.radio("Cara split", methods, methods.index(split["method"]),
                               format_func=lambda m: rcp.SPLITS[m])
    if split["method"] in ("random", "stratified"):
        split["valid_size"] = st.slider("Porsi validasi", 0.05, 0.5, float(split["valid_size"]), 0.05)
    if split["method"] == "kfold":
        split["folds"] = st.slider("Jumlah fold", 2, 10, int(split["folds"]))
    if split["method"] == "stratified" and not r["label"]:
        st.error("Stratified butuh kolom label (tab Data).")
    split["seed"] = int(st.number_input("random_state", value=int(split["seed"]), step=1))
