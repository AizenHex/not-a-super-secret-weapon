"""Tab Temukan kategori: MiniLM mengelompokkan teks yang mirip, lalu kelompok dijadikan lensa."""
import os

import numpy as np
import pandas as pd
import streamlit as st

import helpers
from penjelasan import show_explanation


def run_grouping(ctx, n_groups, max_rows):
    folder = ctx["folder"]
    sample = ctx["data"].sample(n=int(max_rows), random_state=helpers.SEED).index
    with st.spinner("MiniLM membaca teks ..."):
        vectors = helpers.encode(ctx["load_minilm"](), ctx["texts"][sample])
    groups = helpers.find_groups(vectors, n_groups)
    keywords = helpers.group_keywords(ctx["texts"][sample], groups)
    np.save(f"{folder}/vektor.npy", vectors)
    group_rows = pd.DataFrame({"row": sample, "kelompok": groups})
    group_rows.to_csv(f"{folder}/kelompok.csv", index=False)
    info = pd.DataFrame({"kelompok": range(n_groups)})
    info["kata_kunci"] = info["kelompok"].map(keywords)
    info.to_csv(f"{folder}/kelompok_info.csv", index=False)


def show_groups(ctx):
    folder = ctx["folder"]
    group_rows = pd.read_csv(f"{folder}/kelompok.csv")
    info = pd.read_csv(f"{folder}/kelompok_info.csv")
    sizes = group_rows["kelompok"].value_counts()
    for group, keywords in zip(info["kelompok"], info["kata_kunci"]):
        members = group_rows.loc[group_rows["kelompok"] == group, "row"]
        with st.expander(f"Kelompok {group} · {sizes[group]} baris · {keywords}"):
            st.dataframe(ctx["data"].loc[members.head(10), ctx["columns"]])
    return info


def merge_names(named):
    # nama sama = kelompok digabung; nama kosong = kelompok tidak dipakai
    criteria = {}
    for group_name, description in zip(named["nama"], named["penjelasan"]):
        group_name = str(group_name).strip()
        if group_name == "" or group_name == "None":
            continue
        if group_name in criteria:
            criteria[group_name] = criteria[group_name] + ", " + description
        else:
            criteria[group_name] = description
    return criteria


def show_make_lens(ctx, info):
    st.write("**Jadikan lensa.** Beri nama tiap kelompok. Nama sama = kelompok digabung. "
             "Nama dikosongkan = kelompok tidak dipakai. Penjelasan boleh diperbaiki.")
    naming = pd.DataFrame({"kelompok": info["kelompok"]})
    naming["nama"] = "kelompok_" + info["kelompok"].astype(str)
    naming["penjelasan"] = info["kata_kunci"]
    named = st.data_editor(naming, disabled=["kelompok"], hide_index=True, key="naming")
    lens_name = st.text_input("Nama lensa baru", "topik")
    if st.button("Jadikan lensa"):
        criteria = merge_names(named)
        ctx["lenses"][lens_name] = {"type": "choice", "criteria": criteria,
                                    "instructions": "Teks ini termasuk kategori apa?"}
        helpers.save_lenses(ctx["folder"], ctx["lenses"])
        if os.path.exists(helpers.lens_file(ctx["folder"], lens_name)):
            os.remove(helpers.lens_file(ctx["folder"], lens_name))
        st.success(f"Lensa `{lens_name}` ({len(criteria)} pilihan) dibuat. Jalankan di tab "
                   "Lensa Laya.")


def add_findings(ctx):
    group_rows = pd.read_csv(f"{ctx['folder']}/kelompok.csv")
    info = pd.read_csv(f"{ctx['folder']}/kelompok_info.csv")
    sizes = group_rows["kelompok"].value_counts()
    biggest = sizes.index[0]
    share = sizes.iloc[0] / len(group_rows)
    keywords = info.loc[info["kelompok"] == biggest, "kata_kunci"].iloc[0]
    helpers.add_finding(
        ctx, "Kelompok makna", f"{len(sizes)} kelompok; yang terbesar (nomor {biggest}) berisi "
        f"{100 * share:.1f}% baris, kata khasnya: {keywords}.",
        "KMeans pada vektor MiniLM mengelompokkan teks yang maknanya dekat.",
        "Beri nama tiap kelompok lalu jadikan lensa untuk memeriksa kategori.", 0.3)


def show(ctx):
    st.write("Laya hanya bisa **memilih** dari daftar kategori yang kita tulis. Di sini MiniLM "
             "mengelompokkan teks yang mirip, supaya kamu tahu kategori apa yang sebenarnya ada.")
    data = ctx["data"]
    n_groups = st.slider("Jumlah kelompok", 3, 40, 12)
    max_rows = st.number_input("Maksimal baris (sampel acak)", 50, len(data), min(20000, len(data)))
    st.caption("Vektor MiniLM ikut disimpan dan dipakai lagi di tab Outlier (teks jauh/nyasar, "
               "teks yang maknanya sama) dan tab Peta data (peta 2D).")
    if st.button("Kelompokkan"):
        run_grouping(ctx, n_groups, max_rows)
    if os.path.exists(f"{ctx['folder']}/kelompok_info.csv"):
        info = show_groups(ctx)
        show_make_lens(ctx, info)
        add_findings(ctx)
    show_explanation("kelompok")
