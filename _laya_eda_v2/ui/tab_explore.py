"""Tab Data (grid + wawasan kolom) dan Temuan (masalah → usulan langkah)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core import audit
from core import steps as engine
from core.ops import MISSING_MARKERS, OPS, SENTINEL_FLOOR
from ui import common, jobs

ROOT = Path(__file__).resolve().parents[1]
# LAYA_TRAIN / LAYA_TEST: path awal (uji memakai data sintetis); kalau tidak di-set, data riil di _laya_eda/data
DEFAULT_TRAIN = os.environ.get("LAYA_TRAIN", str(ROOT.parent / "_laya_eda" / "data" / "train.csv"))
DEFAULT_TEST = os.environ.get("LAYA_TEST", "")


@st.cache_data(show_spinner="Membaca CSV…")
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


def load_frame(title: str, default: str, key: str) -> tuple[pd.DataFrame | None, str]:
    """Kembalikan (data, kunci sumber). Kunci dipakai cache langkah (bukan id objek)."""
    st.session_state.setdefault(f"{key}_path", default)  # path tersimpan (common.restore_session) menang
    path = st.text_input(f"{title}: path CSV", key=f"{key}_path")
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


def sidebar_data() -> None:
    """Muat data di sidebar: selalu tampil, jadi pindah halaman tidak menghapus file yang diunggah."""
    with st.sidebar.expander("Data", expanded=common.raw_train() is None):
        data, train_key = load_frame("train", DEFAULT_TRAIN, "train")
        test, test_key = load_frame("test", DEFAULT_TEST, "test")
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
        st.caption(f"train {data.shape[0]:,} × {data.shape[1]}"
                   + (f" · test {test.shape[0]:,} × {test.shape[1]}" if test is not None else " · tanpa test"))


def table_report_job(view: pd.DataFrame) -> str:
    """Jalan di thread (tanpa st.*)."""
    import skrub
    return skrub.TableReport(view, max_plot_columns=40).html_snippet()


def show_data() -> None:
    common.header("data",
                  "Lihat data di **langkah mana pun** (seperti grid Data Wrangler) dan wawasan tiap kolom. "
                  "File train (wajib), test (opsional), dan label dipilih di sidebar **Data**. "
                  "Semua statistik dipelajari dari train saja.",
                  "Selalu: ini langkah pertama.",
                  "Test yang ikut dipakai menghitung statistik = kebocoran data. Di sini itu tidak terjadi.")
    if not common.need_data():
        return
    p = common.project()
    steps = p["steps"]
    start = "mentah (0)" if not common.split_config() else "sesudah split (0)"
    at = st.select_slider("Lihat data sesudah langkah", list(range(len(steps) + 1)), len(steps),
                          format_func=lambda n: start if n == 0 else f"{n}. {engine.summary(steps[n - 1])}")
    result = common.state(at)
    view = result["train"]
    if result["error"]:
        st.error(f"Langkah {result['error']['langkah']} gagal: {result['error']['galat']}")
    st.write(f"train: **{view.shape[0]:,} baris × {view.shape[1]} kolom** · valid: **{len(result['valid']):,}** · "
             f"test: **{result['test'].shape[0]:,} × {result['test'].shape[1]}**")
    if common.split_config():
        st.caption("Grid, wawasan kolom, dan semua halaman EDA melihat bagian **train** saja: valid disimpan untuk "
                   "menilai langkahmu dengan jujur (halaman Split).")
    focus = st.session_state.get("fokus_baris")
    if focus:                                          # dari tab Latih: baris dengan salah terbesar (E11)
        rows = view.index.intersection(focus)
        st.warning(f"Menampilkan {len(rows):,} baris dari tab **Latih** (salah terbesar). Cari sebabnya di sini.")
        if st.button("Tampilkan semua baris lagi", key="clear_focus"):
            st.session_state["fokus_baris"] = None
            st.rerun()
        view = view.loc[rows]
    st.dataframe(view.head(500), width="stretch", height=320)

    st.subheader("Wawasan kolom")
    profile = common.analysis("column_profile", common.state_key(at), view)
    st.dataframe(profile, width="stretch", hide_index=True,
                 column_config={"kosong %": st.column_config.ProgressColumn("kosong %", min_value=0, max_value=100,
                                                                           format="%.1f")})
    column = st.selectbox("Detail kolom", list(view.columns), key="insight_column")
    insight(view[column])
    if st.button("Laporan lengkap skrub.TableReport (bisa beberapa detik)", disabled=jobs.running("table_report")):
        jobs.start("table_report", "skrub.TableReport", table_report_job, view.copy(),
                   note="mentah" if at == 0 else f"sesudah langkah {at}")
        st.rerun()
    jobs.status("table_report")
    report = jobs.result("table_report")
    if report is not None:
        st.components.v1.html(report, height=900, scrolling=True)

    st.subheader("Simpan / muat daftar langkah")
    left, right = st.columns(2)
    left.download_button("Unduh langkah (.json)", json.dumps(p, ensure_ascii=False, indent=2), "langkah_eda.json",
                         "application/json")
    upload = right.file_uploader("Muat langkah", type="json", key="project_upload")
    if upload is not None and right.button("Pakai langkah ini"):
        loaded = json.load(upload)
        st.session_state["project"] = {"label": loaded.get("label"), "steps": loaded.get("steps", []),
                                       "split": loaded.get("split")}
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
    data, label, key = common.current(), common.label(), common.state_key() + "|fitur"
    found = 0
    profile = common.analysis("column_profile", key, data).set_index("kolom")
    kinds = profile["tipe tebakan"]

    fake = common.analysis("fake_missing", key, data)
    messy = common.analysis("messy_text", key, data)
    answers = fake[fake["mungkin kategori"].astype(bool)]
    for row in answers.itertuples(index=False):
        found += 1
        st.warning(f"**{row.penanda} di '{row.kolom}' = {row[3]}% isi kolom.** Penanda kosong yang sesering jawaban "
                   "lain biasanya sebuah jawaban ('tidak ada'), bukan sel yang lupa diisi. Tidak diusulkan jadi kosong; "
                   "bila memang berarti 'tidak', satukan di tab **Ejaan** (kamus 'tidak').")
    share = data.isna().mean()
    usual = share.median()
    made = {}                                         # kolom → [(nomor langkah, sel kosong yang dibuatnya, nama)]
    for entry in common.state()["log"]:
        for column, count in (entry.get("kosong baru") or {}).items():
            made.setdefault(column, []).append((entry["langkah"], count, entry["operasi"]))
    for column in share[(share >= 0.05) & (share >= 5 * usual)].index:
        found += 1
        causes = [f"langkah {number} ({name.split(' · ')[0]}) membuat {count:,}"
                  for number, count, name in made.get(column, []) if count >= 0.01 * len(data)]
        st.warning(f"**'{column}' kosong {share[column]:.0%}, kolom lain umumnya {usual:.1%}.** "
                   + (f"Dibuat oleh langkahmu: {'; '.join(causes)}. Kosong sebanyak ini yang muncul karena langkah "
                      "biasanya jawaban yang terhapus (mis. '-' = 'tidak ada'), bukan data hilang: keluarkan kolom ini "
                      "dari langkah tersebut, lalu satukan '-' di tab **Ejaan**."
                      if causes else
                      "Sudah kosong sejak data mentah. Jangan buang barisnya; pertimbangkan kategori sendiri "
                      "('tidak diisi') atau penanda kosong, karena kosong di sini bisa membawa informasi."))
    plain = fake[~fake["mungkin kategori"].astype(bool)]
    columns = sorted((set(plain["kolom"]) | set(messy["kolom"])) - set(answers["kolom"]))
    if columns:
        found += 1
        detail = messy.merge(plain.groupby("kolom")["jumlah"].sum().rename("penanda kosong").reset_index(),
                             on="kolom", how="outer").fillna(0)
        suggestion(f"Rapikan teks di {len(columns)} kolom: {int(detail['sel berspasi lebih'].sum()):,} sel berspasi lebih, "
                   f"{int(detail['penanda kosong'].sum()):,} penanda kosong",
                   detail[detail["kolom"].isin(columns)], "tidy_text", {"columns": columns, "markers": MISSING_MARKERS},
                   "tidy")

    spelling = common.analysis("spelling_findings", key, data)
    if len(spelling):
        found += 1
        suggestion(f"Ejaan berbeda untuk nilai yang sama di {len(spelling)} kolom kategori",
                   spelling[["kolom", "nilai unik", "sesudah", "contoh"]], "spelling",
                   {"groups": dict(zip(spelling["kolom"], spelling["groups"])),
                    "examples": dict(zip(spelling["kolom"], spelling["examples"]))}, "spelling")

    sentinel = common.analysis("sentinels", key, data)
    if len(sentinel):
        found += 1
        suggestion(f"Nilai penjaga ≥ 2.000.000.000 di {len(sentinel)} kolom", sentinel, "sentinel_to_nan",
                   {"columns": list(sentinel["kolom"]), "floor": SENTINEL_FLOOR}, "sentinel")

    numeric_text = list(kinds[kinds == "angka-teks"].index)
    if numeric_text:
        found += 1
        suggestion(f"Angka tersimpan sebagai teks: {len(numeric_text)} kolom", profile.loc[numeric_text].reset_index(),
                   "to_number", {"columns": numeric_text, "indonesian": True}, "number")

    single = common.analysis("single_value_columns", key, data)
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

    findings = common.analysis("row_findings", common.state_key(), common.state()["train"], label)
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

    ordinal = common.analysis("ordinal_findings", key, data)
    if len(ordinal):
        st.subheader("Kolom berurutan (encoding opsional)")
        st.caption("Model pohon memakai urutan ini lebih baik daripada kategori acak. Tidak wajib: AutoGluon menerima "
                   "teks kategori apa adanya.")
        for row in ordinal.itertuples(index=False):
            if row.urutan is None:
                st.caption(f"'{row.kolom}': {row.alasan}")
                continue
            counts = data[row.kolom].value_counts()
            detail = pd.DataFrame({"kode": range(len(row.urutan)), "nilai": row.urutan,
                                   "jumlah": [int(counts.get(v, 0)) for v in row.urutan]})
            suggestion(f"'{row.kolom}' berurutan: {row.alasan}", detail, "ordinal",
                       {"column": row.kolom, "order": row.urutan}, f"ordinal_{row.kolom}")

    st.subheader("Kolom kembar (opsional)")
    st.caption("Cramér's V (skrub.column_associations) ≥ 0,85: isi dua kolom saling menentukan. Buang salah satu saja.")
    if st.button("Cari kolom kembar", disabled=jobs.running("twins")):
        jobs.start("twins", "Cari kolom kembar", audit.twin_columns, data.copy(),
                   note=f"{len(common.project()['steps'])} langkah saat dimulai")
        st.rerun()
    jobs.status("twins")
    twins = jobs.result("twins")
    if twins is not None and len(twins):
        st.dataframe(twins, width="stretch", hide_index=True)
        pick = st.multiselect("Kolom kembar yang dibuang", sorted(set(twins["kolom 2"])), key="twin_pick")
        if pick:
            suggestion(f"Buang {len(pick)} kolom kembar", "", "drop_columns", {"columns": pick}, "twins")
    if not found:
        st.success("Tidak ada temuan otomatis di data sesudah langkah terakhir.")
    st.caption(f"Semua operasi lain ada di tab **Operasi** ({len(OPS)} operasi).")
