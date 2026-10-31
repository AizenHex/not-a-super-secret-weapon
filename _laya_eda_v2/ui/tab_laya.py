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

from ui import common

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
    data = common.train()

    st.subheader("1. Lensa per baris → kelompok pembanding")
    columns = st.multiselect("Kolom yang dibaca Laya", list(data.columns), key="laya_columns")
    name = st.text_input("Nama lensa", "kelas_rumah")
    question = st.text_input("Pertanyaan", "Seberapa layak kondisi rumah ini?")
    choices_text = st.text_area("Pilihan (satu per baris, 'nama: penjelasan')",
                                "sederhana: bahan murah, lantai tanah/semen\n"
                                "sedang: bahan umum, lantai keramik biasa\n"
                                "mewah: bahan mahal, lantai granit/marmer")
    rows = st.slider("Jumlah baris (sampel acak tetap)", 100, len(data), min(2000, len(data)), 100)
    if columns and st.button("Jalankan lensa", type="primary"):
        choices = {}
        for line in choices_text.splitlines():
            key, _, text = line.partition(":")
            if key.strip():
                choices[key.strip()] = text.strip() or key.strip()
        sample = data.sample(rows, random_state=0) if rows < len(data) else data
        with st.spinner(f"Laya menjawab {len(sample)} baris…"):
            answers = ask_rows(load_laya(), sample, columns, name, question, choices)
        st.session_state.setdefault("laya_answers", {})[f"laya: {name}"] = answers["jawaban"].reindex(data.index)
        folder = ROOT / "hasil"
        folder.mkdir(exist_ok=True)
        answers.to_csv(folder / f"laya_{name}.csv")
        st.success(f"Tersimpan di hasil/laya_{name}.csv · dipakai sebagai kelompok di tab Outlier.")
    for key, answers in st.session_state.get("laya_answers", {}).items():
        st.write(f"**{key}**: {int(answers.notna().sum())} baris dijawab")
        st.bar_chart(answers.value_counts())

    st.subheader("2. Pendapat kedua untuk kandidat sinonim")
    from core import text_tools
    texts = common.object_columns(common.features())
    column = st.selectbox("Kolom isian bebas", texts, key="laya_same_column")
    pairs = text_tools.similar_vocabulary(data[column])
    st.dataframe(pairs, width="stretch", hide_index=True)
    if not pairs.empty and st.button("Tanya Laya: pasangan ini sama?"):
        tuples = list(zip(pairs["kata 1"], pairs["kata 2"]))
        with st.spinner("Laya membandingkan…"):
            st.session_state.setdefault("laya_same", {}).update(ask_same(load_laya(), tuples, column))
        st.success("Jawaban tampil sebagai kolom 'Laya: sama?' di tab Isian bebas (peluang 'ya').")
