"""Tab Temuan utama: semua temuan dari tab lain (ctx["findings"]) diurutkan + laporan Markdown."""
import datetime

import pandas as pd
import streamlit as st

from penjelasan import show_explanation


def findings_table(ctx):
    table = pd.DataFrame(ctx["findings"], columns=["bagian", "temuan", "alasan", "saran",
                                                   "penting"])
    table = table.drop_duplicates(subset=["bagian", "temuan"])
    return table.sort_values("penting", ascending=False)


def report_header(ctx):
    data = ctx["data"]
    lines = [f"# Laporan EDA — {ctx['csv_path']}",
             f"Dibuat {datetime.date.today().isoformat()} dengan Laya EDA.", "",
             "## Data",
             f"- {len(data)} baris × {len(data.columns)} kolom",
             f"- kolom yang dibaca: {', '.join(ctx['columns'])}"]
    if ctx["label_column"] is not None:
        counts = data[ctx["label_column"]].value_counts()
        lines.append(f"- label `{ctx['label_column']}`: {len(counts)} kelas, terbesar "
                     f"{counts.iloc[0]} baris, terkecil {counts.iloc[-1]} baris")
    if len(ctx["lens_done"]) > 0:
        lines.append(f"- lensa Laya yang dijalankan: {', '.join(ctx['lens_done'])}")
    return lines


def build_report(ctx, table):
    lines = report_header(ctx)
    lines = lines + ["", "## Temuan utama (urut dari paling penting)", ""]
    number = 1
    for row in table.itertuples():
        lines.append(f"### {number}. [{row.bagian}] {row.temuan}")
        lines.append(f"- **Alasan:** {row.alasan}")
        lines.append(f"- **Saran:** {row.saran}")
        lines.append(f"- penting: {row.penting:.2f}")
        lines.append("")
        number = number + 1
    lines.append("## Metode")
    lines.append("Rumus, alasan pemilihan, batasan, dan angka uji tiap metode: lihat "
                 "`_laya_eda/METODE.md`.")
    return "\n".join(lines) + "\n"


def show(ctx):
    table = findings_table(ctx)
    st.write(f"**{len(table)} temuan** dari semua tab, diurutkan dari paling penting. Temuan "
             "bertambah setelah kamu menjalankan Laya, Kelompokkan, atau memilih label.")
    if len(table) == 0:
        st.info("Belum ada temuan. Pilih data dan kolom label dulu.")
        return
    sections = ["(semua)"] + sorted(table["bagian"].unique())
    section = st.selectbox("Bagian", sections, key="temuan_bagian")
    shown = table
    if section != "(semua)":
        shown = table[table["bagian"] == section]
    st.dataframe(shown, hide_index=True)
    report = build_report(ctx, table)
    first, second = st.columns(2)
    first.download_button("Unduh laporan (.md)", report.encode("utf-8"), "laporan_eda.md",
                          key="temuan_unduh")
    if second.button("Tulis laporan ke folder hasil", key="temuan_tulis"):
        with open(f"{ctx['folder']}/laporan_eda.md", "w", encoding="utf-8") as file:
            file.write(report)
        st.success(f"Tersimpan: _laya_eda/{ctx['folder']}/laporan_eda.md")
    with st.expander("Pratinjau laporan"):
        st.markdown(report)
    show_explanation("temuan")
