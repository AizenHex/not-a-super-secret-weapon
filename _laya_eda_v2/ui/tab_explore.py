"""Tab Data (grid + wawasan kolom) dan Temuan (masalah → usulan langkah)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core import audit, findings, reader, vocab
from core import steps as engine
from core.ops import OPS
from ui.kamus import show_table
from ui import common, jobs

ROOT = Path(__file__).resolve().parents[1]
# LAYA_TRAIN / LAYA_TEST: path awal (uji memakai data sintetis); kalau tidak di-set, data riil di _laya_eda/data
DEFAULT_TRAIN = os.environ.get("LAYA_TRAIN", str(ROOT.parent / "_laya_eda" / "data" / "train.csv"))
DEFAULT_TEST = os.environ.get("LAYA_TEST", "")


@st.cache_data(show_spinner="Membaca CSV…", max_entries=8)
def read_csv(source, options_json: str) -> pd.DataFrame:
    """Baca dengan opsi baca proyek (core/reader.py): kode yang sama dengan blok baca di skrip (G5)."""
    return reader.read(source, json.loads(options_json))


@st.cache_data(show_spinner=False, max_entries=8)
def sniff(source) -> tuple[dict, list[dict]]:
    return reader.sniff(source)


def load_frame(title: str, default: str, key: str) -> tuple[pd.DataFrame | None, str]:
    """Kembalikan (data, kunci sumber). Kunci dipakai cache langkah (bukan id objek). Opsi baca = milik proyek."""
    st.session_state.setdefault(f"{key}_path", default)  # path tersimpan (common.restore_session) menang
    path = st.text_input(f"{title}: path CSV", key=f"{key}_path")
    upload = st.file_uploader(f"… atau unggah {title}", type="csv", key=f"{key}_upload")
    source, name = None, ""
    if upload is not None:
        source, name = upload.getvalue(), f"upload:{upload.name}:{upload.size}"
    elif path and Path(path).exists():
        source, name = path, f"path:{path}:{Path(path).stat().st_mtime}"
    elif path:
        st.error(f"File tidak ditemukan: {path}")
    if source is None:
        return None, ""
    p = common.project()
    if key == "train" and (p.get("baca") is None or st.session_state.get("baca_sumber") != name):
        options, notes = sniff(source)
        if p.get("baca") is None or st.session_state.get("baca_sumber") is not None:
            p["baca"] = options                         # file baru: tebak ulang opsi baca
        st.session_state["baca_sumber"], st.session_state["baca_temuan"] = name, notes
    options = p.get("baca") or reader.DEFAULT
    return read_csv(source, json.dumps(options, sort_keys=True)), f"{name}|{json.dumps(options, sort_keys=True)}"


def read_options(columns: list[str]) -> None:
    """Opsi baca yang bisa diedit (G5); berlaku untuk train dan test, ikut tertulis di skrip."""
    p = common.project()
    options = dict(p.get("baca") or reader.DEFAULT)
    with st.expander("Opsi baca file"):
        seps = [",", ";", "\t", "|"]
        sep = st.selectbox("Pemisah kolom", seps, seps.index(options["sep"]) if options["sep"] in seps else 0,
                           key="baca_sep", format_func=lambda s: {"\t": "tab"}.get(s, s))
        decimal = st.radio("Desimal", [".", ","], [".", ","].index(options.get("desimal", ".")), horizontal=True,
                           key="baca_desimal")
        encodings = ["utf-8-sig", "cp1252", "latin-1"]
        encoding = st.selectbox("Encoding", encodings, encodings.index(options["encoding"])
                                if options["encoding"] in encodings else 0, key="baca_encoding")
        header = st.checkbox("Baris pertama = nama kolom", options["judul"], key="baca_judul")
        na_text = st.multiselect("Kolom yang 'NA'/'None'-nya berarti 'tidak punya' (tetap teks)", columns,
                                 [c for c in options.get("na_teks", []) if c in columns], key="baca_na")
        new = {**options, "sep": sep, "desimal": decimal, "encoding": encoding, "judul": header, "na_teks": na_text}
        if new != options:
            p["baca"] = new
            st.rerun()


def insight(series: pd.Series) -> None:
    """Wawasan satu kolom: angka → ringkasan + histogram; teks → nilai tersering."""
    st.caption(f"{series.dtype} · kosong {int(series.isna().sum()):,} ({series.isna().mean():.1%}) · "
               f"unik {series.nunique():,} · tebakan tipe: {audit.guess_type(series)}")
    if pd.api.types.is_numeric_dtype(series):
        values = series.dropna()
        show_table(values.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_frame().T)
        if len(values):
            st.bar_chart(pd.cut(values.clip(upper=values.quantile(0.99)), 30).value_counts(sort=False)
                         .rename(index=str).rename("jumlah").rename_axis("rentang"))
    else:
        st.bar_chart(series.value_counts(dropna=False).head(25).rename(index=str).rename("jumlah")
                     .rename_axis("nilai"), horizontal=True)      # nama kolom ber-':' merusak grafik Vega


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
        read_options(list(data.columns))
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
    show_table(view.head(500), data_columns=True, height=320)

    st.subheader("Wawasan kolom")
    profile = common.analysis("column_profile", common.state_key(at), view)
    show_table(profile, hide_index=True,
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

    vocabulary_editor()

    st.subheader("Simpan / muat daftar langkah")
    left, right = st.columns(2)
    left.download_button("Unduh langkah (.json)", json.dumps(p, ensure_ascii=False, indent=2), "langkah_eda.json",
                         "application/json")
    upload = right.file_uploader("Muat langkah", type="json", key="project_upload")
    if upload is not None and right.button("Pakai langkah ini"):
        loaded = json.load(upload)
        current = common.project()
        st.session_state["project"] = {"label": loaded.get("label"), "steps": loaded.get("steps", []),
                                       "split": loaded.get("split"),
                                       "kosakata": loaded.get("kosakata") or current.get("kosakata"),
                                       "baca": loaded.get("baca") or current.get("baca")}
        st.rerun()


LISTS = {"penanda_kosong": "Penanda kosong palsu (isi sel yang berarti kosong)",
         "kata_tidak": "Jawaban 'tidak' (kunci tanpa spasi/tanda baca) yang digabung di tab Ejaan",
         "pemisah": "Pemisah isian bebas multi-nilai (simbol atau kata)",
         "kata_umum": "Kata yang tidak pernah dianggap salah ketik",
         "kata_pengisi": "Kata pengisi yang dibuang dari satu bagian isian bebas"}
MAPS = {"singkatan": "Singkatan (regex kata → kata panjang), dipakai kunci ejaan",
        "kata_baku": "Kata varian → kata baku (isian bebas)", "frasa_baku": "Frasa → frasa baku (isian bebas)"}


def vocabulary_editor() -> None:
    """G4: kata domain = data proyek yang bisa diedit (bukan kode). Langkah yang sudah ada tidak berubah."""
    p = common.project()
    with st.expander("Kamus proyek (kata domain, bisa diedit)"):
        st.caption("Dipakai Temuan, Ejaan, dan Isian bebas. Proyek baru = umum (tanpa kata domain). Langkah yang sudah "
                   "dibuat menyimpan salinan kamusnya sendiri, jadi kodenya tidak berubah saat kamus diedit.")
        left, right = st.columns([3, 1])
        preset = left.selectbox("Preset", list(vocab.PRESETS), key="kamus_preset")
        if right.button("Muat preset", key="kamus_muat"):
            p["kosakata"] = json.loads(json.dumps(vocab.PRESETS[preset]))
            st.rerun()
        current = {**vocab.UMUM, **(p.get("kosakata") or {})}
        new = dict(current)
        for name, title in LISTS.items():
            text = st.text_area(title + " — satu per baris", "\n".join(current[name]), key=f"kamus_{name}", height=110)
            new[name] = [line.strip() if name != "penanda_kosong" else line.strip().lower()
                         for line in text.splitlines() if line.strip() or name == "penanda_kosong"]
        for name, title in MAPS.items():
            text = st.text_area(title + " — satu per baris: `asli => baku`",
                                "\n".join(f"{a} => {b}" for a, b in current[name].items()), key=f"kamus_{name}", height=110)
            new[name] = {a.strip(): b.strip() for a, b in (line.split("=>", 1) for line in text.splitlines() if "=>" in line)}
        levels = current["jenjang"]
        st.caption(f"Kelompok berurutan (jenjang): {len(levels['urutan'])} tingkat, {len(levels['kunci'])} kunci "
                   "(isi lewat preset; ubah di file langkah .json bila perlu).")
        if new != current and st.button("Simpan kamus", type="primary", key="kamus_simpan"):
            p["kosakata"] = new
            st.rerun()


# ---------------------------------------------------------------- temuan

def suggestion(title: str, detail: pd.DataFrame | str, op: str, params: dict, key: str) -> None:
    """Satu usulan (ordinal, kolom kembar): penjelasan, rincian, parameter yang bisa diedit, pratinjau, tambah."""
    finding_box({"judul": title, "alasan": detail if isinstance(detail, str) else "",
                 "detail": None if isinstance(detail, str) else detail, "penting": 3,
                 "pilihan": [{"judul": OPS[op]["title"], "op": op, "params": params}]}, key)


@st.cache_data(max_entries=16, show_spinner="Memindai temuan…")
def scan(key: str, _data: pd.DataFrame, label: str | None, markers: tuple) -> list[dict]:
    return findings.scan(_data, label, list(markers))


def finding_box(item: dict, key: str) -> None:
    """Satu temuan (G1) + pilihan aksi + parameter yang bisa diedit (G2) + pratinjau sebelum → sesudah."""
    icon = {1: "🔴", 2: "🟡", 3: "⚪"}.get(item["penting"], "⚪")
    with st.expander(f"{icon} {item['judul']}"):
        if item["alasan"]:
            st.markdown(item["alasan"])
        if item.get("detail") is not None and len(item["detail"]):
            show_table(item["detail"], hide_index=True)
        choices = item["pilihan"]
        pick = 0
        if len(choices) > 1:
            pick = st.radio("Pilihan", list(range(len(choices))), key=f"{key}_pick",
                            format_func=lambda i: choices[i]["judul"])
        choice = choices[pick]
        if choice["op"] is None:
            st.caption(f"Tanpa langkah: {choice['judul']}.")
            return
        common.op_help(choice["op"])
        if st.toggle("Ubah parameter + pratinjau sebelum → sesudah", key=f"pv_{key}"):
            from ui.tab_clean import form

            params = form(choice["op"], common.current(), choice["params"], f"{key}_{pick}_form")
            if params is None:
                st.info("Parameter belum lengkap.")
                return
            common.preview_step(choice["op"], params, f"{key}_{pick}")


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
    fake = common.analysis("fake_missing", key, data)
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
        causes = [f"langkah {number} ({name.split(' · ')[0]}) membuat {count:,}"
                  for number, count, name in made.get(column, []) if count >= 0.01 * len(data)]
        if causes:
            found += 1
            st.warning(f"**'{column}' kosong {share[column]:.0%}.** Dibuat oleh langkahmu: {'; '.join(causes)}. Kosong "
                       "yang muncul karena langkah biasanya jawaban yang terhapus (mis. '-' = 'tidak ada'), bukan data "
                       "hilang: keluarkan kolom ini dari langkah tersebut.")
    full = common.state()["train"]
    items = list(st.session_state.get("baca_temuan") or []) if not common.project()["steps"] else []
    items += scan(common.state_key() + f"|temuan|{label}", full, label, tuple(vocab.get("penanda_kosong")))
    st.caption(f"{len(items)} temuan · 🔴 penting · 🟡 sebaiknya · ⚪ info. Pilih aksi, ubah parameternya bila perlu, "
               "lihat pratinjau, lalu tambah langkah. Kamus kata domain: halaman **Data** → Kamus proyek.")
    for number, item in enumerate(items):
        found += 1
        finding_box(item, f"f{number}_{item['jenis']}")

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
        show_table(twins, hide_index=True)
        pick = st.multiselect("Kolom kembar yang dibuang", sorted(set(twins["kolom 2"])), key="twin_pick")
        if pick:
            suggestion(f"Buang {len(pick)} kolom kembar", "", "drop_columns",
                       {"columns": pick, "reason": "kolom kembar: isinya hampir sama dengan kolom lain"}, "twins")
    if not found:
        st.success("Tidak ada temuan otomatis di data sesudah langkah terakhir.")
    st.caption(f"Semua operasi lain ada di tab **Operasi** ({len(OPS)} operasi).")
