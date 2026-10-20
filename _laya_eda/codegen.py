"""Pembuat kode EDA otomatis: dari pilihan dan temuan di Laya EDA, tulis notebook bergaya
ATURAN_KODE (sel # %%) yang bisa langsung ditempel ke notebook lomba. Tidak memakai helpers.

config (dict):
    csv_path, test_csv (atau None), columns, label_column (atau None),
    number_columns, indonesian_format, use_log, ratio (pasangan [atas, bawah] atau None),
    outlier_number, outlier_category (atau None), lenses (dict lensa yang dipakai),
    drop_rows (list nomor baris), new_labels (dict nomor baris -> label baru), drop_duplicates
"""
import datetime
import pprint
import textwrap

import nbformat


def literal(value):
    # tulis nilai Python sebagai teks kode yang valid (dict/list dirapikan, lebar ≤ 90)
    return pprint.pformat(value, width=90, sort_dicts=False)


def markdown(text):
    return ("markdown", text)


def code(text):
    return ("code", text)


def intro_cells(config):
    today = datetime.date.today().isoformat()
    steps = ["baca data", "catatan manual (buang / ganti label)"]
    if config["drop_duplicates"]:
        steps.append("buang duplikat")
    if len(config["number_columns"]) > 0:
        steps.append("ubah kolom angka")
    if len(config["lenses"]) > 0:
        steps.append("lensa Laya sebagai kolom baru")
    if config["outlier_number"] is not None and config["outlier_category"] is not None:
        steps.append("tanda outlier per kategori")
    steps.append("simpan")
    title = (f"# EDA dari Laya EDA ({today})\n"
             f"Dibuat otomatis dari temuan di `_laya_eda`. Langkah: {', '.join(steps)}.\n\n"
             "Jalankan dari atas. Ubah nilai di sel **Konfigurasi** bila path atau kolom beda.")
    imports = "import numpy as np\nimport pandas as pd"
    if len(config["lenses"]) > 0:
        imports = imports + "\nimport torch\nimport laya"
    return [markdown(title), code(imports)]


def config_cells(config):
    lines = [f'TRAIN_CSV = "{config["csv_path"]}"   # GANTI: path relatif dari folder notebook']
    if config["test_csv"] is not None:
        lines.append(f'TEST_CSV = "{config["test_csv"]}"')
    lines.append(f"TEXT_COLUMNS = {literal(config['columns'])}")
    if config["label_column"] is not None:
        lines.append(f'LABEL = "{config["label_column"]}"')
    if len(config["number_columns"]) > 0:
        lines.append(f"NUMBER_COLUMNS = {literal(config['number_columns'])}")
        lines.append(f"INDONESIAN_FORMAT = {config['indonesian_format']}   # titik = ribuan")
    if len(config["lenses"]) > 0:
        lines.append('LAYA_FOLDER = "../../pretrained/laya"')
        lines.append("BATCH_SIZE = 32")
        lines.append('DEVICE = "cuda" if torch.cuda.is_available() else "cpu"')
    lines.append('OUTPUT_TRAIN = "train_bersih.csv"')
    if config["test_csv"] is not None:
        lines.append('OUTPUT_TEST = "test_bersih.csv"')
    return [markdown("## Konfigurasi"), code("\n".join(lines))]


def read_cells(config):
    text = "train = pd.read_csv(TRAIN_CSV)\nprint(\"train:\", train.shape)"
    if config["test_csv"] is not None:
        text = text + "\ntest = pd.read_csv(TEST_CSV)\nprint(\"test:\", test.shape)"
    text = text + "\ntrain.head()"
    return [markdown("## Baca data"), code(text)]


def notes_cells(config):
    explain = ("## Catatan manual\n"
               "Hasil keputusanmu di tab *Telusuri + catatan*. Nomor baris = urutan baris di CSV "
               "asli, jadi sel ini harus dijalankan **sebelum** baris lain dibuang.")
    lines = [f"DROP_ROWS = {literal(config['drop_rows'])}",
             f"NEW_LABELS = {literal(config['new_labels'])}",
             ""]
    if config["label_column"] is not None:
        lines.append("for row in NEW_LABELS:")
        lines.append("    train.loc[row, LABEL] = NEW_LABELS[row]")
    lines.append("train = train.drop(index=DROP_ROWS)")
    lines.append('print("dibuang:", len(DROP_ROWS), "| label diganti:", len(NEW_LABELS))')
    lines.append('print("train sekarang:", train.shape)')
    return [markdown(explain), code("\n".join(lines))]


def duplicate_cells(config):
    if not config["drop_duplicates"]:
        return []
    explain = ("## Buang duplikat\n"
               "Baris dengan isi kolom teks yang sama persis hanya disimpan sekali (yang pertama). "
               "Duplikat membuat model terlalu percaya diri dan skor validasi terlalu tinggi.")
    text = ("duplicate = train.duplicated(subset=TEXT_COLUMNS, keep=\"first\")\n"
            "print(\"duplikat dibuang:\", duplicate.sum())\n"
            "train = train[~duplicate]")
    return [markdown(explain), code(text)]


def number_cells(config):
    if len(config["number_columns"]) == 0:
        return []
    explain = ("## Ubah kolom angka\n"
               "Teks seperti `Rp 1.250.000` atau `1.500.000/bulan` diubah jadi angka. Yang tidak "
               "berisi angka (`hubungi wa`) menjadi kosong (NaN).")
    text = '''def to_number(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    text = series.astype(str).str.lower()
    text = text.str.replace("rp", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    if INDONESIAN_FORMAT:
        text = text.str.replace(".", "", regex=False)    # titik = pemisah ribuan
        text = text.str.replace(",", ".", regex=False)   # koma = desimal
    else:
        text = text.str.replace(",", "", regex=False)
    number_text = text.str.extract(r"^(-?\\d+(?:\\.\\d+)?)")[0]
    return pd.to_numeric(number_text, errors="coerce")


# coba: sekitar 1250000 dan NaN
print(to_number(pd.Series(["Rp 1.250.000", "hubungi wa"])).tolist())

for column in NUMBER_COLUMNS:
    train[f"{column}_angka"] = to_number(train[column])
    print(column, "tidak terbaca:", train[f"{column}_angka"].isna().sum())'''
    if config["test_csv"] is not None:
        text = text + "\n    test[f\"{column}_angka\"] = to_number(test[column])"
    ratio = config["ratio"]
    if ratio is not None:
        name = f"{ratio[0]}_per_{ratio[1]}"
        tables = ["train"]
        if config["test_csv"] is not None:
            tables.append("test")
        text = text + "\n"
        for table in tables:
            top = f'{table}["{ratio[0]}_angka"]'
            text = text + f'\nbottom = {table}["{ratio[1]}_angka"]'
            text = text + f'\n{table}["{name}"] = {top} / bottom.where(bottom > 0)'
    return [markdown(explain), code(text)]


LENS_CODE = '''judge = laya.load(LAYA_FOLDER, subfolder="multilingual", device=DEVICE)


def make_states(table):
    # satu kolom: kirim teksnya; beberapa kolom: kirim dict kolom -> nilai
    states = []
    for row in table.index:
        if len(TEXT_COLUMNS) == 1:
            states.append(str(table.at[row, TEXT_COLUMNS[0]]))
            continue
        state = {}
        for column in TEXT_COLUMNS:
            state[column] = str(table.at[row, column])
        states.append(state)
    return states


def add_lens_columns(table):
    states = make_states(table)
    for name in LENSES:
        question = LENSES[name]
        results = judge.predict_batch(states, {name: question}, batch_size=BATCH_SIZE)
        answers = []
        values = []
        for result in results:
            answer = result["answers"][name]
            if answer["type"] == "choice":
                answers.append(answer["choice"])
                values.append(answer["answer_confidence"])
            elif answer["type"] == "score":
                answers.append(round(answer["score"]))
                values.append(answer["score"])
            else:
                answers.append(int(answer["noul"] >= 0.5))
                values.append(answer["noul"])
        table[name] = answers
        table[f"{name}__nilai"] = values
    return table


# coba pada 3 baris dulu
add_lens_columns(train.head(3).copy())[list(LENSES)]'''


def lens_cells(config):
    lenses = {}
    skipped = []
    for name in config["lenses"]:
        lens = config["lenses"][name]
        if lens.get("with_label"):
            skipped.append(name)
            continue
        question = {"type": lens["type"], "instructions": lens["instructions"]}
        if "criteria" in lens:
            question["criteria"] = lens["criteria"]
        lenses[name] = question
    if len(lenses) == 0:
        return []
    explain = ("## Lensa Laya → kolom baru\n"
               "Pertanyaan yang sama dengan di Laya EDA, dijalankan ke train **dan** test (fitur "
               "yang hanya ada di train tidak berguna). Kolom `<lensa>` = jawaban, "
               "`<lensa>__nilai` = peluang/skor. Kecepatan ±75–300 baris/detik per lensa di GPU.")
    if len(skipped) > 0:
        explain = explain + (f"\n\nLensa {', '.join(skipped)} tidak ditulis: lensa itu melihat "
                             "label, jadi hanya untuk mencari label salah, bukan fitur.")
    cells = [markdown(explain), code(f"LENSES = {literal(lenses)}"), code(LENS_CODE)]
    run = "train = add_lens_columns(train)"
    if config["test_csv"] is not None:
        run = run + "\ntest = add_lens_columns(test)"
    run = run + f"\ntrain[{literal(list(lenses))}].describe(include=\"all\")"
    cells.append(code(run))
    return cells


def outlier_cells(config):
    number = config["outlier_number"]
    category = config["outlier_category"]
    if number is None or category is None:
        return []
    if category == "kelompok":
        return [markdown("## Outlier per kategori\nKategori `kelompok` (KMeans MiniLM) tidak "
                         "ditulis ulang di sini; pilih lensa Laya atau label sebagai kategori.")]
    column = f"{number}_angka"
    if number not in config["number_columns"]:
        column = number
    scale = ""
    if config["use_log"]:
        scale = ", pada skala log10 karena angka seperti harga menyebar secara perkalian"
    explain = (f"## Outlier `{number}` per `{category}`\n"
               "Batas wajar dihitung **di dalam tiap kategori** (Q1 − 1,5×IQR sampai "
               f"Q3 + 1,5×IQR{scale}). Nilai yang wajar secara keseluruhan bisa aneh untuk "
               "jenisnya sendiri.")
    lines = [f'NUMBER = "{column}"', f'CATEGORY = "{category}"', "",
             "values = train[NUMBER]"]
    if config["use_log"]:
        lines.append("values = np.log10(values.where(values > 0))")
    lines = lines + ["grouped = values.groupby(train[CATEGORY])",
                     "q1 = grouped.transform(\"quantile\", 0.25)",
                     "q3 = grouped.transform(\"quantile\", 0.75)",
                     "spread = 1.5 * (q3 - q1)",
                     f'train["outlier_{number}"] = (values < q1 - spread) | '
                     "(values > q3 + spread)",
                     f"print(\"outlier:\", train[\"outlier_{number}\"].sum())",
                     f"train[train[\"outlier_{number}\"]].head(10)"]
    return [markdown(explain), code("\n".join(lines))]


def save_cells(config):
    text = 'train.to_csv(OUTPUT_TRAIN, index=False)\nprint("disimpan:", OUTPUT_TRAIN, train.shape)'
    if config["test_csv"] is not None:
        text = text + '\ntest.to_csv(OUTPUT_TEST, index=False)\nprint("disimpan:", OUTPUT_TEST)'
    return [markdown("## Simpan"), code(text)]


def build_cells(config):
    cells = intro_cells(config)
    cells = cells + config_cells(config)
    cells = cells + read_cells(config)
    cells = cells + notes_cells(config)
    cells = cells + duplicate_cells(config)
    cells = cells + number_cells(config)
    cells = cells + lens_cells(config)
    cells = cells + outlier_cells(config)
    cells = cells + save_cells(config)
    return cells


def to_script(cells):
    # format # %% yang dipakai seluruh toolkit (_alat/nb/*.py)
    parts = []
    for kind, text in cells:
        if kind == "markdown":
            commented = []
            for line in text.splitlines():
                pieces = textwrap.wrap(line, width=96)    # baris markdown panjang dipotong
                if len(pieces) == 0:
                    pieces = [""]
                for piece in pieces:
                    commented.append(("# " + piece).rstrip())
            parts.append("# %% [markdown]\n" + "\n".join(commented))
        else:
            parts.append("# %%\n" + text)
    return "\n\n".join(parts) + "\n"


def to_notebook(cells):
    notebook = nbformat.v4.new_notebook()
    for kind, text in cells:
        if kind == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            notebook.cells.append(nbformat.v4.new_code_cell(text))
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                       "name": "python3"}
    return nbformat.writes(notebook)
