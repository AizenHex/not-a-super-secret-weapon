"""Outlier angka dan teks + ALASANNYA dalam kalimat biasa yang memuat angka bukti.
Tiap baris yang ditandai mendapat satu atau beberapa alasan (angka, teks, Laya, duplikat)."""
import re

import numpy as np
import pandas as pd

MONEY_WORDS = ["harga", "price", "gaji", "biaya", "tarif"]    # kolom bernama ini diberi "Rp"
MAX_LISTED = 3        # paling banyak sekian nomor baris disebut dalam satu alasan
MAX_GROUP = 200       # kelompok duplikat yang lebih besar hanya dicek sebagian
SEPARATOR = " | "     # pemisah antar alasan di kolom `alasan`
# status yang dijadikan alasan di kolom `alasan`; "hanya di keseluruhan" = wajar untuk kategorinya
LOCAL_STATUSES = ["janggal hanya di kategorinya", "janggal di keduanya"]
CLOSE_GAP = 0.10      # selisih peluang di bawah ini = "hampir seri"
STATUS_TEXT = {
    "janggal hanya di kategorinya": "janggal hanya di kategorinya (di seluruh data masih wajar)",
    "janggal di keduanya": "janggal di kategorinya dan di seluruh data",
    "janggal hanya di keseluruhan": "janggal di seluruh data, tetapi wajar untuk kategorinya",
}


# ---------------- format angka agar mudah dibaca ----------------

def decimal_text(value):
    # tiga angka penting, koma desimal: 11.5 -> "11,5", 0.5234 -> "0,523", 250.0 -> "250"
    digits = 3
    if value >= 100:
        digits = 0
    elif value >= 10:
        digits = 1
    elif value >= 1:
        digits = 2
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0")
        text = text.rstrip(".")
    return text.replace(".", ",")


def format_number(value, money=False):
    # 11500000 -> "11,5 jt"; 1150000000 -> "1,15 M"; 250000 -> "250.000"
    sign = ""
    if value < 0:
        sign = "-"
    value = abs(value)
    if value >= 1e9:
        text = decimal_text(value / 1e9) + " M"
    elif value >= 1e6:
        text = decimal_text(value / 1e6) + " jt"
    elif value >= 1000:
        text = f"{value:,.0f}".replace(",", ".")
    else:
        text = decimal_text(value)
    text = sign + text
    if money:
        text = "Rp " + text
    return text


def column_money(column):
    for word in MONEY_WORDS:
        if word in column.lower():
            return True
    return False


def percent(value):
    return f"{100 * value:.0f}%"


def rows_text(rows, total):
    # "baris 12, 45, 98 dan 4 baris lain"
    shown = rows[:MAX_LISTED]
    names = []
    for row in shown:
        names.append(str(row))
    text = "baris " + ", ".join(names)
    if total > len(shown):
        text = text + f" dan {total - len(shown)} baris lain"
    return text


# ---------------- outlier angka ----------------

def scale_values(values, use_log):
    # skala log cocok untuk harga/luas: 15 juta vs 1,5 miliar terlihat jauh
    if use_log:
        return np.log10(values.where(values > 0))
    return values


def unscale(value, use_log):
    if use_log:
        return 10 ** value
    return value


def iqr_outliers(values, use_log):
    scaled = scale_values(values, use_log)
    q1 = scaled.quantile(0.25)
    q3 = scaled.quantile(0.75)
    spread = 1.5 * (q3 - q1)
    return (scaled < q1 - spread) | (scaled > q3 + spread)


def iqr_bounds(values, use_log):
    # batas wajar IQR Tukey dalam satuan ASLI (kalau skala log, dibalikkan dengan 10^batas)
    scaled = scale_values(values, use_log)
    q1 = scaled.quantile(0.25)
    q3 = scaled.quantile(0.75)
    spread = 1.5 * (q3 - q1)
    return unscale(q1 - spread, use_log), unscale(q3 + spread, use_log)


def group_outliers(values, groups, use_log):
    # outlier IQR di dalam tiap kelompok: dibandingkan dengan baris sejenis saja
    table = pd.DataFrame({"nilai": values, "kelompok": groups})
    table = table.dropna()
    scaled = scale_values(table["nilai"], use_log)
    grouped = scaled.groupby(table["kelompok"])
    q1 = grouped.transform("quantile", 0.25)
    q3 = grouped.transform("quantile", 0.75)
    spread = (q3 - q1) * 1.5
    table["median_kelompok"] = table.groupby("kelompok")["nilai"].transform("median")
    table["kali_median"] = table["nilai"] / table["median_kelompok"]
    table["batas_bawah"] = unscale(q1 - spread, use_log)
    table["batas_atas"] = unscale(q3 + spread, use_log)
    table["outlier"] = (scaled < q1 - spread) | (scaled > q3 + spread)
    return table


def compare_outliers(values, groups, use_log):
    # dua pembanding: semua baris vs baris sekategori saja
    table = group_outliers(values, groups, use_log)
    table["outlier_keseluruhan"] = iqr_outliers(values, use_log)[table.index]
    table["persentil_keseluruhan"] = (100 * values.rank(pct=True)[table.index]).round()
    table["persentil_kategori"] = (100 * table.groupby("kelompok")["nilai"].rank(pct=True)).round()
    in_group = table["outlier"]
    overall = table["outlier_keseluruhan"]
    table["status"] = "normal"
    table.loc[in_group & overall, "status"] = "janggal di keduanya"
    table.loc[in_group & ~overall, "status"] = "janggal hanya di kategorinya"
    table.loc[~in_group & overall, "status"] = "janggal hanya di keseluruhan"
    return table


def describe_value(values, row, use_log):
    # posisi satu nilai di antara values: persentil, median, outlier atau bukan
    values = values.dropna()
    value = values[row]
    position = {"persentil": round(100 * (values < value).mean()), "median": values.median()}
    position["outlier"] = bool(iqr_outliers(values, use_log)[row])
    return position


# ---------------- outlier teks (jarak makna) ----------------

def category_centers(vectors, labels):
    # pusat makna tiap kategori = rata-rata vektor, dinormalkan jadi panjang 1
    names = list(pd.unique(labels))
    centers = []
    for name in names:
        center = vectors[labels == name].mean(axis=0)
        centers.append(center / np.linalg.norm(center))
    return names, np.array(centers)


def far_from_category(table):
    # "jauh" = jarak ke pusat kategorinya di atas batas IQR kategori itu
    checked = group_outliers(table["jarak_ke_kategorinya"], table["kategori"], False)
    table["jarak_median"] = checked["median_kelompok"]
    table["batas_jarak"] = checked["batas_atas"]
    table["jauh"] = checked["outlier"] & (checked["kali_median"] > 1)
    return table


def text_outliers(vectors, rows, categories):
    # jarak makna tiap teks ke pusat kategorinya; "nyasar" = lebih dekat ke kategori lain
    labels = categories[rows].to_numpy()
    keep = pd.notna(labels)
    vectors = vectors[keep]
    labels = labels[keep]
    names, centers = category_centers(vectors, labels)
    name_array = np.array(names, dtype=object)
    similarity = vectors @ centers.T
    position = np.arange(len(labels))
    own = pd.Index(names).get_indexer(labels)
    table = pd.DataFrame({"kategori": labels}, index=np.asarray(rows)[keep])
    table["jarak_ke_kategorinya"] = 1 - similarity[position, own]
    table["kategori_terdekat"] = name_array[similarity.argmax(axis=1)]
    table["nyasar"] = table["kategori_terdekat"] != table["kategori"]
    similarity[position, own] = -np.inf    # sisakan kategori LAIN saja
    table["kategori_lain"] = name_array[similarity.argmax(axis=1)]
    table["jarak_lain"] = 1 - similarity.max(axis=1)
    return far_from_category(table)


# ---------------- duplikat dan konflik label ----------------

def simple_text(text):
    # huruf kecil, hanya huruf dan angka: "Halo, Dunia!" sama dengan "halo dunia"
    return re.sub(r"[^a-z0-9]", "", text.lower())


def same_text_groups(texts, simplify):
    # {kunci teks: [nomor baris, ...]}, hanya teks yang muncul lebih dari sekali
    groups = {}
    for row, text in texts.items():
        key = text
        if simplify:
            key = simple_text(text)
        if key == "":
            continue
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    twins = {}
    for key in groups:
        if len(groups[key]) > 1:
            twins[key] = groups[key]
    return twins


def first_row_by_label(members, labels):
    first = {}
    for row in members:
        if labels[row] not in first:
            first[labels[row]] = row
    return first


def conflict_text(own_label, first_by_label):
    parts = []
    for label in first_by_label:
        if label != own_label and len(parts) < MAX_LISTED:
            parts.append(f"baris {first_by_label[label]} berlabel '{label}'")
    if len(parts) == 0:
        return ""
    return (f"KONFLIK label: baris ini berlabel '{own_label}', tetapi " + ", ".join(parts)
            + " padahal teksnya sama/hampir sama.")


def twin_reason(row, members, texts, labels, first_by_label):
    exact = []
    near = []
    for other in members[:MAX_GROUP]:
        if other == row:
            continue
        if texts[other] == texts[row]:
            exact.append(other)
        else:
            near.append(other)
    sentences = []
    if len(exact) > 0:
        sentences.append(f"Teks sama persis dengan {rows_text(exact, len(members) - 1)}.")
    if len(near) > 0:
        sentences.append(f"Teks hampir sama (beda huruf besar/spasi/tanda baca) dengan "
                         f"{rows_text(near, len(near))}.")
    if labels is not None:
        sentences.append(conflict_text(labels[row], first_by_label))
    return " ".join(sentences).strip()


def duplicate_reasons(texts, labels):
    # {nomor baris: kalimat} untuk duplikat, hampir duplikat, dan konflik label
    reasons = {}
    twins = same_text_groups(texts, True)
    for members in twins.values():
        first_by_label = {}
        if labels is not None:
            first_by_label = first_row_by_label(members, labels)
        for row in members:
            reasons[row] = twin_reason(row, members, texts, labels, first_by_label)
    return reasons


# ---------------- kalimat alasan ----------------

def times_text(value):
    if pd.isna(value) or np.isinf(value):
        return "?"
    return decimal_text(value)


def number_reason(column, line, dim):
    # angka per kategori: nilai, kali median, batas wajar, persentil, status
    money = column_money(column)
    category = f"{dim} '{line['kelompok']}'"
    value = format_number(line["nilai"], money)
    median = format_number(line["median_kelompok"], money)
    low = format_number(line["batas_bawah"], money)
    high = format_number(line["batas_atas"], money)
    text = f"{column} {value} adalah {times_text(line['kali_median'])} kali median {category} "
    text = text + f"(median {median}). Batas wajar {category}: {low} sampai {high}. "
    text = text + f"Persentil {line['persentil_kategori']:.0f} di {category}, "
    text = text + f"{line['persentil_keseluruhan']:.0f} di seluruh data. "
    return text + "Status: " + STATUS_TEXT[line["status"]] + "."


def overall_reason(column, value, ranks, bounds, median):
    # angka di luar batas wajar seluruh data (baris belum punya kategori)
    money = column_money(column)
    low, high = bounds
    text = f"{column} {format_number(value, money)} di luar batas wajar seluruh data "
    text = text + f"({format_number(low, money)} sampai {format_number(high, money)}; "
    text = text + f"median {format_number(median, money)}). "
    return text + f"Persentil {100 * ranks:.0f} di seluruh data."


def text_reason(dim, line):
    # teks jauh dari pusat kategorinya dan/atau lebih dekat ke kategori lain
    own = f"{dim} '{line['kategori']}'"
    other = f"'{line['kategori_lain']}' (jarak {decimal_text(line['jarak_lain'])})"
    distance = decimal_text(line["jarak_ke_kategorinya"])
    sentences = []
    if line["jauh"]:
        sentences.append(f"Teks jauh dari makna {own}: jarak {distance}, padahal median jarak "
                         f"kategori ini {decimal_text(line['jarak_median'])} dan batas wajar "
                         f"{decimal_text(line['batas_jarak'])}.")
    if line["nyasar"]:
        sentences.append(f"Teks lebih dekat ke {other} daripada ke kategorinya sendiri "
                         f"(jarak {distance}).")
    else:
        sentences.append(f"Kategori lain terdekat: {other}.")
    return " ".join(sentences)


def closeness_text(gap):
    if gap < CLOSE_GAP:
        return "hampir seri"
    return f"selisih {100 * gap:.0f} poin"


def laya_reason(name, kind, line, doubt_limit):
    # Laya ragu: pilihan pertama dan kedua beserta peluangnya
    answer = line[name]
    sure = decimal_text(line[f"{name}__yakin"])
    limit = decimal_text(doubt_limit)
    first = line[f"{name}__pertama_nilai"]
    second = line[f"{name}__kedua"]
    second_value = line[f"{name}__kedua_nilai"]
    tail = f"; keyakinan {sure} di bawah batas {limit}."
    if pd.isna(first) or pd.isna(second_value):
        return f"Laya (lensa '{name}') memilih '{answer}' dengan keyakinan {sure} < batas {limit}."
    gap = closeness_text(first - second_value)
    if kind == "noul":
        yes = line[f"{name}__nilai"]
        return (f"Laya (lensa '{name}') menjawab '{answer}': peluang ya {percent(yes)}, "
                f"tidak {percent(1 - yes)} - {gap}" + tail)
    return (f"Laya (lensa '{name}') memilih '{answer}' {percent(first)}, pilihan kedua "
            f"'{second}' {percent(second_value)} - {gap}" + tail)


# ---------------- mengumpulkan semua alasan ----------------

def add_reason(parts, row, text):
    if row not in parts:
        parts[row] = []
    parts[row].append(text)


def add_unreadable(parts, column, raw, values):
    unreadable = raw.notna() & values.isna()
    for row in raw[unreadable].index:
        add_reason(parts, row, f"{column} berisi '{raw[row]}' yang tidak terbaca sebagai angka "
                               "(isiannya tidak diawali angka).")


def add_non_positive(parts, column, values):
    money = column_money(column)
    for row in values[values < 0].index:
        add_reason(parts, row, f"{column} bernilai negatif ({format_number(values[row], money)}); "
                               "ukuran, harga, atau jumlah seharusnya positif.")
    for row in values[values == 0].index:
        add_reason(parts, row, f"{column} bernilai nol; periksa apakah memang seharusnya nol.")


def add_overall_reasons(parts, column, values, use_log, covered):
    flagged = iqr_outliers(values, use_log)
    rows = flagged[flagged].index.difference(covered)
    ranks = values.rank(pct=True)
    # batas dan median sama untuk semua baris: hitung sekali, bukan sekali per baris
    bounds = iqr_bounds(values, use_log)
    median = values.median()
    for row in rows:
        add_reason(parts, row, overall_reason(column, values[row], ranks[row], bounds, median))


def add_outlier_reasons(parts, column, values, categories, dim, use_log):
    covered = []
    if categories is not None:
        checked = compare_outliers(values, categories, use_log)
        covered = checked.index
        flagged = checked[checked["status"].isin(LOCAL_STATUSES)]
        for row, line in flagged.iterrows():
            add_reason(parts, row, number_reason(column, line, dim))
    add_overall_reasons(parts, column, values, use_log, covered)


def add_number_reasons(parts, ctx):
    dim = ctx["reason_dim"]
    categories = None
    if dim is not None:
        categories = ctx["table"][dim]
    for column in ctx["numbers"].columns:
        values = ctx["numbers"][column]
        if column in ctx["data"].columns:
            add_unreadable(parts, column, ctx["data"][column], values)
        add_non_positive(parts, column, values)
        add_outlier_reasons(parts, column, values, categories, dim, ctx["use_log"])


def add_text_reasons(parts, ctx):
    distances = ctx["distances"]
    if distances is None:
        return
    flagged = distances[distances["jauh"] | distances["nyasar"]]
    for row, line in flagged.iterrows():
        add_reason(parts, row, text_reason(ctx["reason_dim"], line))


def add_laya_reasons(parts, ctx):
    table = ctx["table"]
    for name in ctx["lens_done"]:
        kind = ctx["lenses"][name]["type"]
        doubtful = table[table[f"{name}__yakin"] < ctx["doubt_limit"]]
        for row, line in doubtful.iterrows():
            add_reason(parts, row, laya_reason(name, kind, line, ctx["doubt_limit"]))


def add_duplicate_reasons(parts, ctx):
    labels = None
    if ctx["label_column"] is not None:
        labels = ctx["data"][ctx["label_column"]].fillna("(kosong)").astype(str)
    reasons = duplicate_reasons(ctx["texts"], labels)
    for row in reasons:
        if reasons[row] != "":
            add_reason(parts, row, reasons[row])


def collect_reasons(ctx):
    # {nomor baris: [alasan, ...]}; urutan alasan: angka, teks, Laya, duplikat
    parts = {}
    add_number_reasons(parts, ctx)
    add_text_reasons(parts, ctx)
    add_laya_reasons(parts, ctx)
    add_duplicate_reasons(parts, ctx)
    return parts


def join_reasons(parts, index):
    # satu kolom `alasan` (teks) untuk semua baris; kosong = tidak ada yang ditandai
    reasons = pd.Series("", index=index)
    for row in parts:
        reasons[row] = SEPARATOR.join(parts[row])
    return reasons
