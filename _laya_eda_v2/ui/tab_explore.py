"""Tab Data (grid + wawasan kolom) dan Temuan (masalah → usulan langkah)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import audit
from core import steps as engine
from core.ops import MISSING_MARKERS, OPS, SENTINEL_FLOOR
from ui import common

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT.parent / "_laya_eda" / "data" / "train.csv"


@st.cache_data(show_spinner="Membaca CSV…")
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


def load_frame(title: str, default: str, key: str) -> tuple[pd.DataFrame | None, str]:
    """Kembalikan (data, kunci sumber). Kunci dipakai cache langkah (bukan id objek)."""
    path = st.text_input(f"{title}: path CSV", default, key=f"{key}_path")
    upload = st.file_uploader(f"… atau unggah {title}", type="csv", key=f"{key}_upload")
    if upload is not None:
        return read_csv(upload), f"upload:{upload.name}:{upload.size}"
    if path and Path(path).exists():
        return read_csv(path), f"path:{path}:{Path(path).stat().st_mtime}"
    if path:
        st.error(f"File tidak ditemukan: {path}")
    return None, ""


def insight(series: pd.Series) -> None:
    """Wawasan satu kolom: angka → ringkasan + histogram; teks → nilai tersering."""
    st.caption(f"{series.dtype} · kosong {int(series.isna().sum()):,} ({series.isna().mean():.1%}) · "
               f"unik {series.nunique():,} · tebakan tipe: {audit.guess_type(series)}")
    if pd.api.types.is_numeric_dtype(series):
        values = series.dropna()
        st.dataframe(values.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_frame().T, width="stretch")
        if len(values):
            st.bar_chart(pd.cut(values.clip(upper=values.quantile(0.99)), 30).value_counts(sort=False)
                         .rename(index=str))
    else:
        st.bar_chart(series.value_counts(dropna=False).head(25).rename(index=str), horizontal=True)


def show_data() -> None:
    common.header("data",
                  "Muat data **train** (wajib) dan **test** (opsional), pilih label, lalu lihat data di "
                  "**langkah mana pun** (seperti grid Data Wrangler). Semua statistik dipelajari dari train saja.",
                  "Selalu: ini langkah pertama.",
                  "Test yang ikut dipakai menghitung statistik = kebocoran data. Di sini itu tidak terjadi.")
    left, right = st.columns(2)
    with left:
        data, train_key = load_frame("train", str(DEFAULT_TRAIN), "train")
    with right:
        test, test_key = load_frame("test", "", "test")
    if data is None:
        return
    st.session_state["data_id"] = f"{train_key}|{test_key}"
    st.session_state["train"], st.session_state["test"] = data, test

    p = common.project()
    options = [None, *data.columns]
    chosen = st.selectbox("Kolom label (target)", options, options.index(p["label"]) if p["label"] in data else 0,
                          help="Label tidak pernah diubah, di-encode, atau diisi oleh langkah mana pun.")
    if chosen != p["label"]:
        p["label"] = chosen
        st.rerun()

    steps = p["steps"]
    at = st.select_slider("Lihat data sesudah langkah", list(range(len(steps) + 1)), len(steps),
                          format_func=lambda n: "mentah (0)" if n == 0 else f"{n}. {engine.summary(steps[n - 1])}")
    result = common.state(at)
    view = result["train"]
    if result["error"]:
        st.error(f"Langkah {result['error']['langkah']} gagal: {result['error']['galat']}")
    st.write(f"train: **{view.shape[0]:,} baris × {view.shape[1]} kolom** · test: **{result['test'].shape[0]:,} × "
             f"{result['test'].shape[1]}**")
    st.dataframe(view.head(500), width="stretch", height=320)

    st.subheader("Wawasan kolom")
    profile = audit.column_profile(view)
    st.dataframe(profile, width="stretch", hide_index=True,
                 column_config={"kosong %": st.column_config.ProgressColumn("kosong %", min_value=0, max_value=100,
                                                                           format="%.1f")})
    column = st.selectbox("Detail kolom", list(view.columns), key="insight_column")
    insight(view[column])
    if st.button("Laporan lengkap skrub.TableReport (bisa beberapa detik)"):
        import skrub
        st.components.v1.html(skrub.TableReport(view, max_plot_columns=40).html_snippet(), height=900,
                              scrolling=True)

    st.subheader("Simpan / muat daftar langkah")
    left, right = st.columns(2)
    left.download_button("Unduh langkah (.json)", json.dumps(p, ensure_ascii=False, indent=2), "langkah_eda.json",
                         "application/json")
    upload = right.file_uploader("Muat langkah", type="json", key="project_upload")
    if upload is not None and right.button("Pakai langkah ini"):
        loaded = json.load(upload)
        st.session_state["project"] = {"label": loaded.get("label"), "steps": loaded.get("steps", [])}
        st.rerun()


# ---------------------------------------------------------------- temuan

def suggestion(title: str, detail: pd.DataFrame | str, op: str, params: dict, key: str) -> None:
    """Satu temuan: penjelasan operasi, rincian, pratinjau, tombol tambah langkah."""
    with st.expander(title):
        common.op_help(op)
        if isinstance(detail, str):
            st.write(detail)
        else:
            st.dataframe(detail, width="stretch", hide_index=True)
        if st.toggle("Pratinjau sebelum → sesudah", key=f"pv_{key}"):
            common.preview_step(op, params, key)


def show_findings() -> None:
    common.header("temuan",
                  "Pindai data **sesudah langkah terakhir** dan usulkan langkah untuk tiap masalah. Tidak ada yang "
                  "diterapkan otomatis: buka temuan, baca penjelasan, pratinjau, lalu klik **Tambah langkah**.",
                  "Setelah memuat data, dan lagi setiap menambah beberapa langkah.",
                  "Usulan hanyalah tebakan dari pola data; konteks kolom (apa arti '0' atau '-') kamu yang tahu.")
    if not common.need_data():
        return
    data, label = common.current(), common.label()
    found = 0
    profile = audit.column_profile(data).set_index("kolom")
    kinds = profile["tipe tebakan"]

    fake = audit.fake_missing(data)
    if len(fake):
        found += 1
        columns = sorted(fake["kolom"].unique())
        suggestion(f"Penanda kosong palsu di {len(columns)} kolom teks ({fake['jumlah'].sum():,} sel)", fake,
                   "tidy_text", {"columns": columns, "markers": MISSING_MARKERS}, "fake")
    texts = [c for c in common.object_columns(data) if kinds[c] in ("kategori", "isian bebas")]
    if texts and not len(fake):
        suggestion(f"Rapikan spasi di {len(texts)} kolom teks", "Spasi di awal/akhir/ganda membuat nilai sama "
                   "terhitung berbeda.", "tidy_text", {"columns": texts, "markers": MISSING_MARKERS}, "tidy")

    sentinel = audit.sentinels(data)
    if len(sentinel):
        found += 1
        suggestion(f"Nilai penjaga ≥ 2.000.000.000 di {len(sentinel)} kolom", sentinel, "sentinel_to_nan",
                   {"columns": list(sentinel["kolom"]), "floor": SENTINEL_FLOOR}, "sentinel")

    numeric_text = list(kinds[kinds == "angka-teks"].index)
    if numeric_text:
        found += 1
        suggestion(f"Angka tersimpan sebagai teks: {len(numeric_text)} kolom", profile.loc[numeric_text].reset_index(),
                   "to_number", {"columns": numeric_text, "indonesian": True}, "number")

    single = audit.single_value_columns(data)
    for row in single.itertuples(index=False):
        found += 1
        suggestion(f"Kolom checkbox '{row.kolom}': hanya '{row[1]}' + {row.kosong:,} kosong",
                   "Biasanya kosong = 'tidak'. Ubah nilai pengisi di tab Operasi bila artinya lain.",
                   "fill_constant", {"columns": [row.kolom], "value": "Tidak"}, f"single_{row.kolom}")

    ids = list(kinds[kinds.isin(["id", "kosong"])].index)
    if ids:
        found += 1
        suggestion(f"Kolom id / kosong total: {', '.join(ids)}", "Kolom id unik per baris tidak membawa informasi; "
                   "bila model menganggapnya penting (tab AutoGluon), itu tanda kebocoran urutan data.",
                   "drop_columns", {"columns": ids}, "ids")

    findings = audit.row_findings(common.state()["train"], label)
    if findings["duplikat persis"]:
        found += 1
        suggestion(f"{findings['duplikat persis']:,} baris duplikat persis di train",
                   f"Grup duplikat dengan label berbeda: {findings['grup duplikat dengan label berbeda']}",
                   "drop_duplicates", {}, "dups")
    if findings["kosong per baris (maks)"] >= max(10, data.shape[1] // 2):
        found += 1
        suggestion(f"Ada baris dengan {findings['kosong per baris (maks)']} kolom kosong",
                   pd.DataFrame({"kosong per baris": list(findings["distribusi kosong per baris"]),
                                 "jumlah baris": list(findings["distribusi kosong per baris"].values())}),
                   "drop_missing_rows", {"max_missing": max(10, data.shape[1] // 2)}, "rowsmissing")

    st.subheader("Kolom kembar (opsional)")
    st.caption("Cramér's V (skrub.column_associations) ≥ 0,85: isi dua kolom saling menentukan. Buang salah satu saja.")
    if st.button("Cari kolom kembar"):
        st.session_state["twins"] = audit.twin_columns(data)
    twins = st.session_state.get("twins")
    if twins is not None and len(twins):
        st.dataframe(twins, width="stretch", hide_index=True)
        pick = st.multiselect("Kolom kembar yang dibuang", sorted(set(twins["kolom 2"])), key="twin_pick")
        if pick:
            suggestion(f"Buang {len(pick)} kolom kembar", "", "drop_columns", {"columns": pick}, "twins")
    if not found:
        st.success("Tidak ada temuan otomatis di data sesudah langkah terakhir.")
    st.caption(f"Semua operasi lain ada di tab **Operasi** ({len(OPS)} operasi).")
