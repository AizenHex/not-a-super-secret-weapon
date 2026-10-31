"""Tab Label (cleanlab), Train vs test, AutoGluon (skor sebelum vs sesudah, kolom penting)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import audit
from ui import common


# ---------------------------------------------------------------- label

def show_label() -> None:
    common.header("label",
                  "Periksa kolom label: distribusi, duplikat fitur dengan label berbeda, dan (opsional) label yang "
                  "mungkin salah menurut **cleanlab** dengan probabilitas out-of-fold dari model cepat atau AutoGluon.",
                  "Klasifikasi dengan label dari isian manusia.",
                  "cleanlab hanya menebak dari model; model lemah menandai terlalu banyak. Label tidak diubah.")
    if not common.need_data():
        return
    label, data = common.label(), common.state()["train"]
    if not label:
        st.info("Pilih kolom label di tab **Data**.")
        return
    counts = data[label].value_counts(dropna=False).sort_index()
    st.bar_chart(counts.rename(index=str))
    st.caption(f"kosong: {int(data[label].isna().sum())} · kelas terkecil: {counts.min()} baris")
    findings = audit.row_findings(data, label)
    st.metric("grup duplikat fitur dengan label berbeda", findings["grup duplikat dengan label berbeda"])

    st.subheader("cleanlab: label yang mungkin salah (opsional)")
    model = st.radio("Model untuk probabilitas out-of-fold", ["cepat: HistGradientBoosting 5-fold (kolom angka, ±30 detik)",
                                                              "kuat: AutoGluon bagging 5-fold (semua kolom)"],
                     key="label_model")
    seconds = st.slider("Batas waktu AutoGluon (detik)", 60, 1800, 300, 60, key="label_time") if "AutoGluon" in model else 0
    if st.button("Jalankan cleanlab"):
        with st.spinner("Melatih model out-of-fold…"):
            oof = None
            if seconds:
                from core import ag
                oof, _ = ag.oof_probabilities(data, label, seconds)
            st.session_state["label_flags"] = audit.label_issues(data, label, oof=oof)
    flagged = st.session_state.get("label_flags")
    if flagged is not None:
        accuracy, baseline = flagged.attrs["akurasi model"], flagged.attrs["akurasi tebak terbanyak"]
        share = len(flagged) / len(data)
        st.write(f"Akurasi model (out-of-fold): **{accuracy:.1%}** · tebak kelas terbanyak saja: {baseline:.1%}")
        st.write(f"{len(flagged):,} baris ditandai ({share:.1%}). Label tidak diubah.")
        if share > 0.2 or accuracy < 0.6:
            st.warning("Model terlalu lemah untuk label ini, jadi cleanlab menandai terlalu banyak baris. Ini tanda "
                       "label sulit ditebak dari fitur, BUKAN bukti label salah. Lihat hanya baris dengan skor label "
                       "paling rendah dan 'yakin tebakan' tinggi.")
        st.dataframe(flagged.head(200), width="stretch")
        st.download_button("Unduh baris ditandai (.csv)", flagged.to_csv(), "label_meragukan.csv", "text/csv")


# ---------------------------------------------------------------- train vs test

def show_test() -> None:
    common.header("test",
                  "Bandingkan train dan test **sesudah semua langkah**.",
                  "Selalu, bila punya file test.",
                  "Kategori baru di test → di one-hot jadi `_lainnya`; di kode angka jadi −1.")
    if not common.need_data():
        return
    if common.raw_test() is None:
        st.info("Muat file test di tab **Data** untuk membandingkan.")
        return
    result = common.state()
    train, test, label = common.current(), result["test"], common.label()
    missing = sorted(set(train.columns) - set(test.columns))
    extra = sorted(set(test.columns) - set(train.columns) - {label})
    st.write(f"Kolom tidak ada di test: **{missing or '-'}** · kolom lebih di test: **{extra or '-'}**")
    rows = []
    for column in common.object_columns(train):
        if column in test:
            new = set(test[column].dropna()) - set(train[column].dropna())
            if new:
                rows.append({"kolom": column, "nilai baru": len(new),
                             "baris test terdampak": int(test[column].isin(new).sum()),
                             "contoh": ", ".join(map(repr, sorted(new, key=str)[:5]))})
    st.subheader("Kategori di test yang tidak ada di train")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    from scipy.stats import ks_2samp
    shift = []
    for column in common.number_columns(train):
        if column in test:
            a, b = train[column].dropna(), pd.to_numeric(test[column], errors="coerce").dropna()
            if len(a) and len(b):
                ks = ks_2samp(a, b)
                shift.append({"kolom": column, "KS": round(ks.statistic, 3), "p": ks.pvalue,
                              "median train": a.median(), "median test": b.median()})
    st.subheader("Pergeseran distribusi angka (scipy ks_2samp)")
    st.caption("KS mendekati 0 = mirip. p kecil dengan KS kecil biasanya hanya efek jumlah data besar.")
    if shift:
        st.dataframe(pd.DataFrame(shift).sort_values("KS", ascending=False), width="stretch", hide_index=True)


# ---------------------------------------------------------------- autogluon

def show_autogluon() -> None:
    common.header("autogluon",
                  "**AutoGluon** (`TabularPredictor`) sebagai juri: apakah daftar langkahmu membuat model lebih baik? "
                  "Train dibagi sekali (80% latih, 20% uji); data **mentah** dan data **sesudah langkah** dinilai di "
                  "baris uji yang sama. Lalu **kolom penting** (permutation importance).",
                  "Setelah beberapa langkah: untuk tahu langkah mana yang membantu, dan kolom mana yang mencurigakan.",
                  "Skor naik-turun sedikit bisa karena acak. Langkah yang membuang baris uji tidak berlaku (bagian uji "
                  "diperlakukan seperti test). Kolom id yang penting = tanda kebocoran urutan data.")
    if not common.need_data():
        return
    label = common.label()
    if not label:
        st.info("Pilih kolom label di tab **Data**.")
        return
    left, right = st.columns(2)
    seconds = left.slider("Batas waktu per model (detik)", 30, 1800, 120, 30, key="ag_time")
    presets = right.selectbox("Preset", ["medium_quality", "good_quality", "high_quality", "best_quality"],
                              key="ag_presets", help="Lebih tinggi = lebih lama, lebih akurat.")
    steps = common.project()["steps"]
    st.caption(f"Dua model dilatih (mentah dan sesudah {sum(s.get('on', True) for s in steps)} langkah aktif): "
               f"± {2 * seconds // 60 + 1} menit.")
    if st.button("Bandingkan mentah vs sesudah langkah", type="primary"):
        from core import ag
        with st.spinner("AutoGluon melatih dua model…"):
            st.session_state["ag_result"] = ag.compare(common.raw_train(), label, steps, seconds, presets)
    result = st.session_state.get("ag_result")
    if result is None:
        return
    if result["galat"]:
        st.error(f"Langkah {result['galat']['langkah']} gagal di bagian latih: {result['galat']['galat']}")
        return
    raw, clean = result["mentah"], result["bersih"]
    metric = raw["metrik"]
    cols = st.columns(3)
    cols[0].metric(f"{metric} mentah", f"{raw['skor'][metric]:.4f}")
    cols[1].metric(f"{metric} sesudah langkah", f"{clean['skor'][metric]:.4f}",
                   f"{clean['skor'][metric] - raw['skor'][metric]:+.4f}")
    cols[2].metric("baris latih", f"{result['baris latih'][1]:,}", result["baris latih"][1] - result["baris latih"][0] or None)
    st.caption(f"Baris uji yang sama: {result['baris uji']:,}. Metrik lain:")
    st.dataframe(pd.DataFrame({"mentah": raw["skor"], "sesudah langkah": clean["skor"]}), width="stretch")
    with st.expander("Papan model (leaderboard)"):
        st.dataframe(clean["papan"], width="stretch", hide_index=True)

    st.subheader("Kolom penting (permutation importance)")
    which = st.radio("Model", ["sesudah langkah", "mentah"], horizontal=True, key="ag_which")
    if st.button("Hitung kolom penting"):
        from core import ag
        from core import steps as engine
        fit_part, hold_part = ag.split(common.raw_train(), label)
        if which == "mentah":
            data = hold_part
        else:
            run = engine.run(steps, label, fit_part, hold_part.drop(columns=[label]))
            data = run["test"].assign(**{label: hold_part[label]})
        model = (raw if which == "mentah" else clean)["model"]
        with st.spinner("Mengacak kolom satu per satu…"):
            st.session_state["ag_importance"] = (which, ag.importance(model, data))
    stored = st.session_state.get("ag_importance")
    if stored:
        which, table = stored
        st.caption(f"Model {which}. importance = turunnya {metric} bila isi kolom diacak; ≤ 0 = tidak dipakai.")
        st.bar_chart(table.set_index("kolom")["importance"].head(30), horizontal=True)
        st.dataframe(table, width="stretch", hide_index=True)
        suspicious = [k for k in table["kolom"].head(10) if str(k).lower() in ("id", "index", "no", "nomor")]
        if suspicious:
            st.warning(f"Kolom {suspicious} termasuk 10 terpenting. Kolom nomor urut seharusnya tidak membawa "
                       "informasi: kemungkinan urutan data berkaitan dengan label (bocor). Pertimbangkan membuangnya.")
