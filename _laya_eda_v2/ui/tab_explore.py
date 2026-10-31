"""Tab Data, Ringkasan, Tipe kolom."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import audit
from core import recipe as rcp
from ui import common

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT.parent / "_laya_eda" / "data" / "train.csv"


@st.cache_data(show_spinner="Membaca CSV…")
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


def load_frame(title: str, default: str, key: str) -> tuple[pd.DataFrame | None, str]:
    """Kembalikan (data, kunci sumber). Kunci dipakai cache pipeline (bukan id objek)."""
    path = st.text_input(f"{title}: path CSV", default, key=f"{key}_path")
    upload = st.file_uploader(f"… atau unggah {title}", type="csv", key=f"{key}_upload")
    if upload is not None:
        return read_csv(upload), f"upload:{upload.name}:{upload.size}"
    if path and Path(path).exists():
        return read_csv(path), f"path:{path}:{Path(path).stat().st_mtime}"
    if path:
        st.error(f"File tidak ditemukan: {path}")
    return None, ""


def show_data() -> None:
    common.header("data",
                  "Muat data **train** (wajib) dan **test** (opsional). Semua statistik nanti dipelajari "
                  "dari train saja, lalu dipakai sama di test.",
                  "Selalu: ini langkah pertama.",
                  "Test yang ikut dipakai menghitung statistik = kebocoran data. Di sini itu tidak terjadi.")
    left, right = st.columns(2)
    with left:
        data, train_key = load_frame("train", str(DEFAULT_TRAIN), "train")
    with right:
        test, test_key = load_frame("test", "", "test")
    if data is None:
        return
    st.session_state["train_id"] = f"{train_key}|{test_key}"
    st.session_state["train"], st.session_state["test"] = data, test

    r = common.recipe()
    options = [None, *data.columns]
    current = options.index(r["label"]) if r["label"] in data else 0
    r["label"] = st.selectbox("Kolom label (target)", options, index=current,
                              help="Label tidak pernah diubah, di-encode, atau diisi.")
    st.write(f"train: **{data.shape[0]:,} baris × {data.shape[1]} kolom**"
             + (f" · test: **{test.shape[0]:,} × {test.shape[1]}**" if test is not None else ""))
    st.dataframe(data.head(20), width="stretch")

    st.subheader("Resep")
    st.caption("Resep = semua keputusanmu. Simpan untuk dilanjutkan nanti, atau muat resep lama.")
    left, right = st.columns(2)
    left.download_button("Unduh resep (.json)", json.dumps(r, ensure_ascii=False, indent=2),
                         "resep_eda.json", "application/json")
    upload = right.file_uploader("Muat resep", type="json", key="recipe_upload")
    if upload is not None and right.button("Pakai resep ini"):
        st.session_state["recipe"] = rcp.merge_defaults(json.load(upload))
        st.rerun()
    if st.button("Mulai resep baru (kosong)"):
        st.session_state["recipe"] = rcp.new_recipe(r["label"])
        st.rerun()


def show_summary() -> None:
    common.header("ringkasan",
                  "Gambaran data apa adanya. Tab ini **hanya membaca**.",
                  "Untuk tahu masalah apa yang ada sebelum memilih langkah.",
                  "Tidak ada: tab ini tidak mengubah data.")
    if not common.need_data():
        return
    data, label = common.train(), common.recipe()["label"]
    st.dataframe(audit.column_profile(data), width="stretch", hide_index=True)
    findings = audit.row_findings(data, label)
    cols = st.columns(3)
    cols[0].metric("duplikat persis", findings["duplikat persis"])
    cols[1].metric("duplikat beda label", findings["grup duplikat dengan label berbeda"])
    cols[2].metric("kosong terbanyak dalam 1 baris", findings["kosong per baris (maks)"])
    st.caption("Banyak kolom kosong per baris → jumlah baris")
    st.bar_chart(pd.Series(findings["distribusi kosong per baris"]))
    if st.button("Buat laporan skrub.TableReport (bisa beberapa detik)"):
        import skrub
        st.components.v1.html(skrub.TableReport(data, max_plot_columns=40).html_snippet(),
                              height=900, scrolling=True)


def show_types() -> None:
    common.header("tipe",
                  "Tentukan peran tiap kolom: dibuang, **angka yang tersimpan sebagai teks** "
                  "(`'Rp 1.500.000'` → 1500000), atau **teks yang dirapikan** (spasi, penanda kosong palsu).",
                  "Kolom id, kolom kembar (isi hampir sama dengan kolom lain), angka berformat teks.",
                  "Membuang kolom yang ternyata berguna. Kolom kembar: buang salah satu saja, bukan dua-duanya.")
    if not common.need_data():
        return
    r, data = common.recipe(), common.features()
    profile = audit.column_profile(data)
    st.dataframe(profile[["kolom", "tipe tebakan", "dtype pandas", "unik", "contoh tersering"]],
                 width="stretch", hide_index=True)

    st.subheader("Kolom kembar")
    st.caption("Cramér's V (skrub.column_associations): 1 = isi dua kolom saling menentukan penuh.")
    if st.button("Cari kolom kembar"):
        st.session_state["twins"] = audit.twin_columns(data)
    if "twins" in st.session_state:
        st.dataframe(st.session_state["twins"], width="stretch", hide_index=True)

    kinds = profile.set_index("kolom")["tipe tebakan"]
    columns = list(data.columns)
    with st.form("types"):
        drop = st.multiselect("Kolom dibuang", columns, [c for c in r["drop_columns"] if c in columns],
                              help=f"Saran: {', '.join(kinds[kinds.isin(['id', 'kosong'])].index) or '-'}")
        numbers = st.multiselect("Angka tersimpan sebagai teks → to_number", columns,
                                 [c for c in r["number_columns"] if c in columns],
                                 help=f"Saran: {', '.join(kinds[kinds == 'angka-teks'].index) or '-'}")
        indonesian = st.checkbox("Format Indonesia (titik = ribuan, koma = desimal)", r["indonesian_numbers"])
        suggestion = [c for c in common.object_columns(data) if kinds[c] in ("kategori", "isian bebas")]
        texts = st.multiselect("Kolom teks yang dirapikan (tidy_text)", columns,
                               [c for c in r["text_columns"] if c in columns],
                               help="Rapikan spasi + penanda kosong palsu → NaN. Wajib untuk tab Kategori.")
        use_suggestion = st.checkbox(f"Pakai saran: semua kolom teks ({len(suggestion)})", not r["text_columns"])
        if st.form_submit_button("Simpan ke resep", type="primary"):
            r["drop_columns"], r["number_columns"], r["indonesian_numbers"] = drop, numbers, indonesian
            r["text_columns"] = sorted(set(texts) | set(suggestion)) if use_suggestion else texts
            common.saved("Tipe kolom disimpan")
            st.rerun()
    st.caption(f"Di resep: dibuang {r['drop_columns'] or '-'} · angka-teks {r['number_columns'] or '-'} · "
               f"teks dirapikan {len(r['text_columns'])} kolom")
