"""Tab Ekspor (paling akhir): tulis / unduh file.
(1) train_siap.csv (+ test_siap.csv) = hasil kode tab Kode EDA,
(2) train_split.csv + valid_split.csv = pembagian tab Split data,
(3) data_berlensa.csv = data + jawaban Laya + cek teks + alasan + catatan, untuk ditelusuri.
Nama kolom asli tidak diubah: kolom angka berisi angka hasil baca di kolom yang sama."""
import functools

import streamlit as st

import helpers
import kamus
from penjelasan import show_explanation

NOTE_COLUMNS = helpers.NOTE_COLUMNS


def build_export(ctx):
    export = ctx["table"].copy()
    for column in ctx["numbers"].columns:
        export[column] = ctx["numbers"][column]    # kolom yang sama, isinya angka
    for column in ["panjang", "jumlah_kata", "rasio_bukan_huruf", "duplikat", "duplikat_mirip"]:
        export[f"teks__{column}"] = ctx["stats"][column]
    export["alasan"] = ctx["reasons"]
    notes = helpers.load_notes(ctx["folder"])
    for column in NOTE_COLUMNS:
        export[column] = notes[column].reindex(export.index).fillna("")
    return export


def csv_text(table, with_row):
    # with_row: kolom `row` = nomor baris di CSV asli (train); test tanpa kolom itu
    if with_row:
        return table.to_csv(index_label="row")
    return table.to_csv(index=False)


def file_buttons(ctx, key, files):
    # files: {nama file: (tabel, with_row)}. Satu tombol tulis untuk semua file + satu tombol
    # unduh per file. Isi unduhan baru dibuat saat tombol diklik (callable), jadi klik widget
    # lain tidak menulis CSV besar ke memori tiap kali
    columns = st.columns(len(files) + 1)
    if columns[0].button("Tulis ke folder hasil", key=f"ekspor_tulis_{key}"):
        for name in files:
            table, with_row = files[name]
            with open(f"{ctx['folder']}/{name}", "w", encoding="utf-8", newline="") as file:
                file.write(csv_text(table, with_row))
        st.success(f"Tersimpan di _laya_eda/{ctx['folder']}: {', '.join(files)}")
    position = 1
    for name in files:
        table, with_row = files[name]
        columns[position].download_button(f"Unduh {name}",
                                          functools.partial(csv_text, table, with_row), name,
                                          mime="text/csv", key=f"ekspor_unduh_{name}")
        position = position + 1


def show_ready(ctx):
    st.subheader("1. train_siap.csv (+ test_siap.csv)")
    train = ctx["ready_train"]
    files = {"train_siap.csv": (train, True)}
    text = f"Hasil kode tab Kode EDA: train {len(train)} baris × {len(train.columns)} kolom"
    if ctx["ready_test"] is not None:
        files["test_siap.csv"] = (ctx["ready_test"], False)
        text = text + f" · test {len(ctx['ready_test'])} baris"
    st.write(text)
    file_buttons(ctx, "siap", files)


def show_split(ctx):
    st.subheader("2. train_split.csv + valid_split.csv")
    if ctx["split_parts"] is None:
        st.info("Belum ada pembagian: atur di tab Split data.")
        return
    train_part, valid_part = ctx["split_parts"]
    st.write(f"Dari tab Split data: train {len(train_part)} baris · validasi {len(valid_part)} "
             "baris")
    file_buttons(ctx, "split", {"train_split.csv": (train_part, True),
                                "valid_split.csv": (valid_part, True)})


def show_inspection(ctx):
    st.subheader("3. data_berlensa.csv (semua hasil, untuk ditelusuri)")
    st.write("Data asli (kolom angka sudah angka, nama tetap) + jawaban Laya (+ yakin) + "
             "kelompok + cek teks + alasan outlier + catatanmu.")
    export = build_export(ctx)
    file_buttons(ctx, "berlensa", {"data_berlensa.csv": (export, True)})
    st.code(f'data = pd.read_csv("../_laya_eda/{ctx["folder"]}/data_berlensa.csv")\n'
            'data = data[data["keputusan"] != "buang"]', language="python")
    kamus.show_table(export.head(20))


def show(ctx):
    show_ready(ctx)
    show_split(ctx)
    show_inspection(ctx)
    show_explanation("ekspor")
