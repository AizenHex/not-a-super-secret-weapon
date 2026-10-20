"""Tab Lensa Laya: daftar lensa, lensa siap pakai, buat lensa sendiri, menjalankan Laya."""
import os

import pandas as pd
import streamlit as st

import helpers
from lenses import PRESETS
from penjelasan import show_explanation

TYPE_NAMES = {"pilih satu": "choice", "ya / tidak": "noul", "skala": "score"}
SPEED = 300    # perkiraan jawaban Laya per detik di GPU (hasil uji, teks ±180 huruf)


def show_overview(ctx):
    overview = []
    for name in ctx["lenses"]:
        lens = ctx["lenses"][name]
        options = ""
        if "criteria" in lens:
            options = ", ".join(lens["criteria"])
        overview.append({"lensa": name, "jenis": lens["type"], "pertanyaan": lens["instructions"],
                         "pilihan": options,
                         "sudah_dihitung": len(helpers.done_rows(ctx["folder"], name))})
    st.write("**Lensa aktif** (satu lensa = satu pertanyaan yang dijawab Laya untuk tiap baris)")
    if len(overview) == 0:
        st.info("Belum ada lensa. Tambah lensa siap pakai atau buat sendiri di bawah.")
    else:
        st.dataframe(pd.DataFrame(overview), hide_index=True)


def remove_result_file(ctx, name):
    if os.path.exists(helpers.lens_file(ctx["folder"], name)):
        os.remove(helpers.lens_file(ctx["folder"], name))


def show_presets(ctx):
    preset = st.selectbox("Lensa siap pakai (isi lengkap ada di lenses.py)", list(PRESETS))
    st.json(PRESETS[preset], expanded=False)
    if st.button("Tambah lensa ini"):
        ctx["lenses"][preset] = PRESETS[preset]
        helpers.save_lenses(ctx["folder"], ctx["lenses"])
        st.rerun()
    removed = st.selectbox("Hapus lensa", ["-"] + list(ctx["lenses"]))
    if st.button("Hapus") and removed != "-":
        del ctx["lenses"][removed]
        helpers.save_lenses(ctx["folder"], ctx["lenses"])
        remove_result_file(ctx, removed)
        st.rerun()


def show_new_lens_form(ctx):
    with st.form("lensa_baru"):
        st.write("**Buat / ubah lensa** (nama sama = ganti, hasil lamanya dihapus)")
        name = st.text_input("Nama lensa (tanpa spasi), misal: topik")
        type_name = st.radio("Jenis", list(TYPE_NAMES), horizontal=True)
        question = st.text_input("Pertanyaan untuk Laya", "Apa topik utama teks ini?")
        options_text = st.text_area("Pilihan, satu per baris. Pilih satu: `nama: penjelasan`. "
                                    "Skala: urut dari terendah. Ya/tidak: kosongkan.")
        with_label = st.checkbox("Laya juga melihat kolom label (untuk mencari label salah)")
        if st.form_submit_button("Simpan lensa") and name.strip() != "":
            lens = {"type": TYPE_NAMES[type_name], "instructions": question}
            if lens["type"] != "noul":
                lens["criteria"] = helpers.parse_options(options_text, lens["type"])
            if with_label:
                lens["with_label"] = True
            ctx["lenses"][name.strip()] = lens
            helpers.save_lenses(ctx["folder"], ctx["lenses"])
            remove_result_file(ctx, name.strip())
            st.rerun()


def rows_to_do(ctx, target):
    # per lensa: baris di target yang belum dijawab Laya
    todo = {}
    for name in ctx["lenses"]:
        done = helpers.done_rows(ctx["folder"], name)
        rows = []
        for row in target:
            if row not in done:
                rows.append(row)
        todo[name] = rows
    return todo


def run_all(ctx, todo):
    judge = ctx["load_laya"]()
    for name in ctx["lenses"]:
        if len(todo[name]) == 0:
            continue
        lens_label = None
        if ctx["lenses"][name].get("with_label"):
            if ctx["label_column"] is None:
                st.warning(f"Lensa {name} butuh kolom label; pilih di samping. Dilewati.")
                continue
            lens_label = ctx["label_column"]
        bar = st.progress(0.0, text=f"lensa {name} ...")
        helpers.run_lens(judge, ctx["data"], todo[name], ctx["columns"], lens_label, name,
                         ctx["lenses"][name], ctx["folder"], bar)
    st.success("Selesai. Lihat tab Peta data.")


def show_run(ctx):
    data = ctx["data"]
    st.write("**Jalankan Laya.** Urutan acak selalu sama, jadi menambah jumlah baris nanti hanya "
             "menghitung baris baru. Boleh dihentikan (tombol Stop kanan atas); hasil tersimpan.")
    use_all = st.checkbox(f"Semua baris ({len(data)})")
    n_rows = st.number_input("Jumlah baris", 10, len(data), min(1000, len(data)))
    if use_all:
        n_rows = len(data)
    order = data.sample(frac=1, random_state=helpers.SEED).index
    todo = rows_to_do(ctx, order[:int(n_rows)])
    total = 0
    for name in todo:
        total = total + len(todo[name])
    st.write(f"Perlu dihitung: **{total}** jawaban · perkiraan **{total / SPEED / 60:.1f} menit**")
    if st.button("Jalankan Laya", type="primary", disabled=total == 0):
        run_all(ctx, todo)
    show_explanation("jalankan_laya")


def show(ctx):
    show_overview(ctx)
    left, right = st.columns(2)
    with left:
        show_presets(ctx)
    with right:
        show_new_lens_form(ctx)
    show_explanation("lensa")
    st.divider()
    show_run(ctx)
