"""Tab Laya: konteks dari jawaban Laya. Jawaban TIDAK PERNAH masuk data bersih atau pipeline.

Dua pemakaian:
1. lensa per baris (pertanyaan + pilihan) → jawaban jadi kelompok pembanding di tab Outlier;
2. "apakah dua ejaan ini bermakna sama?" untuk pasangan kandidat sinonim di tab Isian bebas.
Semua tab lain tetap jalan tanpa Laya.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui import common, jobs

ROOT = Path(__file__).resolve().parents[1]
LAYA_FOLDER = ROOT.parent.parent / "pretrained" / "laya"
BATCH_SIZE = 32


@st.cache_resource(show_spinner="Memuat Laya…")
def load_laya():
    import laya
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return laya.load(str(LAYA_FOLDER), subfolder="multilingual", device=device)


def laya_available() -> str | None:
    try:
        import laya  # noqa: F401
    except ImportError:
        return "paket `laya` tidak terpasang"
    if not LAYA_FOLDER.exists():
        return f"model tidak ditemukan di {LAYA_FOLDER}"
    return None


def ask_rows(judge, data: pd.DataFrame, columns: list[str], name: str, question: str,
             choices: dict[str, str]) -> pd.DataFrame:
    states = [{c: str(v) for c, v in row.items()} for row in data[columns].to_dict("records")]
    lens = {name: {"type": "choice", "instructions": question, "criteria": choices}}
    answers = judge.predict_batch(states, lens, batch_size=BATCH_SIZE)
    return pd.DataFrame({"jawaban": [a["answers"][name]["choice"] for a in answers],
                         "yakin": [a["answers"][name]["answer_confidence"] for a in answers]},
                        index=data.index)


def ask_same(judge, pairs: list[tuple[str, str]], context: str) -> dict:
    states = [{"konteks": context, "kata 1": a, "kata 2": b} for a, b in pairs]
    lens = {"sama": {"type": "noul", "instructions": "Apakah kata 1 dan kata 2 bermakna sama (hanya beda ejaan "
                                                     "atau sinonim) dalam konteks ini?"}}
    answers = judge.predict_batch(states, lens, batch_size=BATCH_SIZE)
    return {pair: round(a["answers"]["sama"]["noul"], 2) for pair, a in zip(pairs, answers)}


def show_laya() -> None:
    common.header("laya",
                  "Laya membaca baris dan menjawab pertanyaanmu. Jawabannya dipakai **sebagai konteks untuk "
                  "melihat data**: kelompok pembanding di tab Outlier, dan pendapat kedua untuk kandidat sinonim. "
                  "Jawaban Laya tidak pernah ditulis ke data bersih atau ke kode pipeline.",
                  "Data dengan kolom teks yang maknanya perlu dibaca (bahan rumah, pekerjaan, alasan).",
                  "Jawaban model bisa salah; lihat kolom 'yakin'. Menjalankan banyak baris butuh GPU.")
    if not common.need_data():
        return
    problem = laya_available()
    if problem:
        st.warning(f"Laya tidak tersedia: {problem}. Semua tab lain tetap bisa dipakai.")
        return
    data = common.raw_train()

    st.subheader("1. Lensa per baris → kelompok pembanding")
    columns = st.multiselect("Kolom yang dibaca Laya", list(data.columns), key="laya_columns")
    name = st.text_input("Nama lensa", "kelas_rumah")
    question = st.text_input("Pertanyaan", "Seberapa layak kondisi rumah ini?")
    choices_text = st.text_area("Pilihan (satu per baris, 'nama: penjelasan')",
                                "sederhana: bahan murah, lantai tanah/semen\n"
                                "sedang: bahan umum, lantai keramik biasa\n"
                                "mewah: bahan mahal, lantai granit/marmer")
    rows = st.slider("Jumlah baris (sampel acak tetap)", 100, len(data), min(2000, len(data)), 100)
    if columns and st.button("Jalankan lensa", type="primary", disabled=jobs.running("laya_lens")):
        choices = {}
        for line in choices_text.splitlines():
            key, _, text = line.partition(":")
            if key.strip():
                choices[key.strip()] = text.strip() or key.strip()
        sample = data.sample(rows, random_state=0) if rows < len(data) else data
        jobs.start("laya_lens", f"Laya: lensa {name}", lens_job, sample[columns].copy(), columns, name, question,
                   choices, data.index, note=f"{len(sample):,} baris · hasil/laya_{name}.csv")
        st.rerun()
    jobs.status("laya_lens")
    collect()
    for key, answers in st.session_state.get("laya_answers", {}).items():
        st.write(f"**{key}**: {int(answers.notna().sum())} baris dijawab")
        st.bar_chart(answers.value_counts())

    st.subheader("2. Pendapat kedua untuk kandidat sinonim")
    from core import text_tools
    texts = common.object_columns(common.current())
    column = st.selectbox("Kolom isian bebas", texts, key="laya_same_column")
    pairs = text_tools.similar_vocabulary(data[column])
    st.dataframe(pairs, width="stretch", hide_index=True)
    if not pairs.empty and st.button("Tanya Laya: pasangan ini sama?", disabled=jobs.running("laya_same")):
        tuples = list(zip(pairs["kata 1"], pairs["kata 2"]))
        jobs.start("laya_same", "Laya: pasangan sama?", same_job, tuples, column, note=column)
        st.rerun()
    if jobs.status("laya_same") is not None and jobs.result("laya_same") is not None:
        st.success("Jawaban tampil sebagai kolom 'Laya: sama?' di tab Isian bebas (peluang 'ya').")


# ---------------------------------------------------------------- tugas latar (tanpa st.*: jalan di thread)

def lens_job(sample: pd.DataFrame, columns: list[str], name: str, question: str, choices: dict,
             index: pd.Index) -> tuple[str, pd.Series]:
    answers = ask_rows(load_laya(), sample, columns, name, question, choices)
    folder = ROOT / "hasil"
    folder.mkdir(exist_ok=True)
    answers.to_csv(folder / f"laya_{name}.csv")
    return f"laya: {name}", answers["jawaban"].reindex(index)


def same_job(pairs: list[tuple[str, str]], column: str) -> dict:
    return ask_same(load_laya(), pairs, column)


def collect() -> None:
    """Jawaban Laya dari tugas latar yang sudah selesai → session_state (dibaca tab Outlier dan Isian bebas).

    Dipanggil di setiap run (app.py), jadi jawaban masuk walau halaman Laya tidak sedang dibuka.
    """
    for name in ("laya_lens", "laya_same"):
        job = jobs.get(name)
        if job is None or jobs.result(name) is None or st.session_state.get(f"_{name}_at") == job.finished:
            continue
        st.session_state[f"_{name}_at"] = job.finished
        if name == "laya_lens":
            key, answers = job.result
            st.session_state.setdefault("laya_answers", {})[key] = answers
        else:
            st.session_state.setdefault("laya_same", {}).update(job.result)
