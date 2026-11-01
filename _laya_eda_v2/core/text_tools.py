"""Alat teks: usulan ejaan. Semua fungsi hanya MENGUSULKAN; pengguna menyetujui per baris.

Kolom kategori (tab Ejaan) → langkah 'Satukan ejaan' = SATU aturan kunci + daftar GABUNG pendek:
1. kunci (key collision OpenRefine, varian n-gram): huruf kecil, singkatan dipanjangkan, spasi & tanda baca
   dibuang. 'S- 2', 'S-2', 'S2' berkunci 's2'; 'Kab. Pati' dan 'Kabupaten Pati' berkunci 'kabupatenpati'.
   Aturan ini menyatukan semua varian semacam itu tanpa daftar, juga varian baru di test.
2. gabung kunci berbeda yang bermakna sama, selalu dengan alasan:
   - urutan kata (fingerprint OpenRefine: kata unik terurut): 'Semarang Kota' = 'Kota Semarang';
   - kamus jenjang pendidikan: SD = MI, SMP = MTs = SLTP, SMA = MA = SLTA, S1 = D4 bila ada label 'S1/D4';
   - kamus 'tidak': Tidak = Tidak Ada = Tidak Dapat, dan '-' bila jumlahnya setara (bukan kosong);
   - salah ketik (rapidfuzz, opsional): kunci → kunci mirip yang jauh lebih sering;
   - skrub.deduplicate (opsional).
Kolom isian bebas (tab Isian bebas) → peta KATA (salah ketik per kata + kamus kecil bahan bangunan), kata
pengisi (satu regex: kata nama kolom, 'dari', 'biasa'), lalu peta FRASA pendek (bata merah = batu bata).
"""
from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

from core.ops import ABBREVIATIONS, KEY_DIGITS, KEY_DROP
from core.ops import SEPARATORS as MATERIAL_SEPARATORS

# kata pendek/umum yang mirip ejaan dengan kata lain tapi beda makna: jangan pernah dipetakan
STOPWORDS = {"tanpa", "tanah", "tidak", "dak", "bagian", "sebagian", "dan", "atau", "dengan", "yang",
             "ada", "belum", "sudah", "lain", "lainnya", "sengon", "seng", "dari", "di", "ke", "pakai"}
# kata mirip yang 20× lebih jarang dianggap salah ketik, walau sering muncul (gendeng 52 vs genteng 10.953)
TYPO_RATIO = 20
# kamus kecil bahan bangunan (KBBI dan kategori bahan atap/dinding/lantai BPS): varian → kata baku
MATERIAL_WORDS = {"genting": "genteng", "gendeng": "genteng", "kramik": "keramik", "plaster": "plester",
                  "plesteran": "plester", "tegel": "ubin", "tekel": "ubin", "tehel": "ubin", "jubin": "ubin",
                  "herbel": "hebel", "batubata": "batu bata", "esbes": "asbes"}
# frasa bahan yang sama makna menurut KBBI, berlaku per bahan sesudah peta kata:
# bata = batu bata (bata merah); genting = "atap dari tanah liat yang dibakar" (genteng tanah liat = genteng)
MATERIAL_PHRASES = {"bata": "batu bata", "bata merah": "batu bata", "batu bata merah": "batu bata",
                    "genteng tanah liat": "genteng", "genteng tanah": "genteng"}
# kata pengisi di dalam satu bahan ('genteng dari tanah liat', 'keramik biasa'); kata nama kolom ditambahkan
FILLER_WORDS = {"dari", "biasa", "bahan", "rumah"}

# jenjang pendidikan, urut naik. Kunci → nomor jenjang. Label gabungan ('S1/D-4', 'D-3/D2') dibaca per bagian.
EDUCATION_LEVELS = ["tidak sekolah", "tidak lulus SD", "SD sederajat", "SMP sederajat", "SMA sederajat",
                    "D1", "D2", "D3", "D4/S1", "S2", "S3"]
EDUCATION = {
    "tidaksekolah": 0, "belumsekolah": 0, "tidaklulussd": 1, "tidaktamatsd": 1, "belumtamatsd": 1,
    "sd": 2, "mi": 2, "sdmi": 2, "sdsederajat": 2, "sdmisederajat": 2, "paketa": 2,
    "smp": 3, "mts": 3, "sltp": 3, "smpmts": 3, "smpsederajat": 3, "sltpsederajat": 3, "paketb": 3,
    "sma": 4, "ma": 4, "smk": 4, "slta": 4, "smama": 4, "smasmk": 4, "smasederajat": 4, "sltasederajat": 4,
    "paketc": 4, "stm": 4, "smea": 4,
    "d1": 5, "d2": 6, "d3": 7, "d4": 8, "s1": 8, "s2": 9, "s3": 10,
}
# jawaban 'tidak' (kunci). Kunci '' = isi hanya tanda baca ('-'): digabung hanya bila jumlahnya setara.
NEGATIVE = {"tidak", "tidakada", "tidakdapat", "tidakpunya", "tidakpernah", "tidakmenerima", "belum", "belumada",
            "belumpernah", "tak", "takada", "no", "none", "nihil"}
MULTI_PART = r",|&|\+|;|\bdan\b"


# ---------------------------------------------------------------- kunci

def keys(series: pd.Series) -> pd.Series:
    """Kunci ejaan tiap sel, dengan ekspresi pandas yang sama persis dengan kode langkah 'Satukan ejaan'."""
    key = series.astype("string").str.lower().replace(ABBREVIATIONS, regex=True)   # astype: kolom test kosong semua
    return key.str.replace(KEY_DIGITS, "_", regex=True).str.replace(KEY_DROP, "", regex=True)


def key_of(value) -> str:
    text = str(value).lower()
    for pattern, full in ABBREVIATIONS.items():
        text = re.sub(pattern, full, text)
    return re.sub(KEY_DROP, "", re.sub(KEY_DIGITS, "_", text))


def fingerprint(value) -> str:
    """OpenRefine fingerprint: huruf kecil, singkatan dipanjangkan, tanda baca DIBUANG, kata unik terurut."""
    text = str(value).lower()
    for pattern, full in ABBREVIATIONS.items():
        text = re.sub(pattern, full, text)
    return " ".join(sorted(set(re.sub(r"[^\w\s]", "", text).split())))


def key_table(series: pd.Series) -> pd.DataFrame:
    """Satu baris per ejaan: asli, jumlah, kunci, baku (ejaan rapi tersering di kuncinya, seperti kode langkah)."""
    counts = series.dropna().astype(str).value_counts()
    table = pd.DataFrame({"asli": counts.index, "jumlah": counts.to_numpy()})
    table["kunci"] = keys(table["asli"])
    table["rapi"] = table["asli"].str.strip().str.replace(r"\s+", " ", regex=True)
    tidy = table.groupby(["kunci", "rapi"], sort=False)["jumlah"].sum().reset_index()
    tidy = tidy.sort_values("jumlah", ascending=False, kind="stable").drop_duplicates("kunci")
    table["baku"] = table["kunci"].map(tidy.set_index("kunci")["rapi"])
    return table


def is_free_text(series: pd.Series) -> bool:
    """Isian bebas: > 100 kunci berbeda dan ≥ 3% sel berisi beberapa bagian (',', '&', 'dan')."""
    text = series.dropna().astype(str)
    return keys(text).nunique() > 100 and text.str.contains(MULTI_PART, regex=True).mean() >= 0.03


# ---------------------------------------------------------------- gabung kunci

def education_levels(value) -> set[int]:
    """Jenjang yang disebut satu nilai: 'S1/D-4' → {8}, 'D-3/D2' → {6, 7}, tidak dikenal → set()."""
    key = key_of(value)
    if key in EDUCATION:
        return {EDUCATION[key]}
    parts = [key_of(part).removesuffix("sederajat") for part in re.split(r"/|,|\batau\b", str(value))]
    if len(parts) > 1 and all(part in EDUCATION for part in parts):
        return {EDUCATION[part] for part in parts}
    return set()


def _education_edges(total: pd.Series, spelled: pd.Series) -> list[tuple]:
    levels = {key: education_levels(spelled[key]) for key in total.index}
    known = [key for key in total.index if levels[key]]
    if len(known) < 3 or total[known].sum() < 0.8 * total.sum():
        return []
    root = list(range(len(EDUCATION_LEVELS)))          # jenjang yang muncul di satu label gabungan = satu kelompok
    for key in known:
        first = root[min(levels[key])]
        for level in levels[key]:
            old = root[level]
            root = [first if r == old else r for r in root]
    groups = {}
    for key in known:                                   # total urut menurun: anggota pertama = kunci tersering
        groups.setdefault(root[min(levels[key])], []).append(key)
    edges = []
    for members in groups.values():
        names = sorted({EDUCATION_LEVELS[level] for key in members for level in levels[key]}, key=EDUCATION_LEVELS.index)
        for key in members[1:]:
            edges.append((key, members[0], f"kamus jenjang: {' = '.join(names)} ('{spelled[key]}' = "
                                           f"'{spelled[members[0]]}')", True))
    return edges


def _negative_edges(total: pd.Series, spelled: pd.Series, table: pd.DataFrame) -> list[tuple]:
    negative = [key for key in total.index if key in NEGATIVE]
    if not negative:
        return []
    target = negative[0]
    edges = [(key, target, f"kamus 'tidak': '{spelled[key]}' = '{spelled[target]}'", True) for key in negative[1:]]
    if "" in total.index and (len(negative) >= 2 or total[""] >= 0.5 * total[target]):
        marks = ", ".join(repr(v) for v in table.loc[table["kunci"] == "", "asli"].head(3))
        share = total[""] / total[target]
        edges.append(("", target, f"{marks} biasanya penanda kosong, TAPI di sini jumlahnya {share:.0%} dari "
                                  f"'{spelled[target]}' dan kolom ini berisi jawaban 'tidak' → kemungkinan varian "
                                  "'tidak', bukan kosong. Periksa arti formulirnya.", share >= 0.5))
    return edges


def _typo_edges(total: pd.Series, spelled: pd.Series, cutoff: float, min_ratio: float) -> list[tuple]:
    edges = []
    names = {key: spelled[key].lower() for key in total.index}
    for key, count in total.items():
        bigger = {other: names[other] for other in total.index if other != key and total[other] >= min_ratio * count}
        match = process.extractOne(names[key], bigger, scorer=fuzz.token_sort_ratio, score_cutoff=cutoff) if bigger else None
        if not match or re.findall(r"\d+", names[key]) != re.findall(r"\d+", match[0]):   # S2 ≠ S3, D1 ≠ D3
            continue
        warning = ("beda kata depan, periksa" if fingerprint(names[key]).split()[:1] != fingerprint(match[0]).split()[:1]
                   else "skor < 95: bisa nilai berbeda (Malang ≠ Magelang)" if match[1] < 95 else "")
        edges.append((key, match[2], f"salah ketik? token_sort_ratio {match[1]:.0f}, '{spelled[match[2]]}' "
                                     f"{total[match[2]] / count:.0f}× lebih sering" + (f"; {warning}" if warning else ""),
                      False))
    return edges


def merge_proposals(series: pd.Series, typos: bool = False, cutoff: float = 88, min_ratio: float = 3,
                    extra: dict[str, str] | None = None) -> pd.DataFrame:
    """Usulan GABUNG: satu baris per kunci yang digabung ke kunci lain (kunci tersering di kelompoknya).

    Sumber: urutan kata, kamus jenjang, kamus 'tidak', (opsional) salah ketik rapidfuzz, (opsional) peta
    `extra` {asli: usulan} dari skrub.deduplicate. Kolom 'setuju' = usulan centang awal.
    """
    columns = ["kunci", "ejaan", "jumlah", "gabung ke", "jadi", "alasan", "setuju"]
    table = key_table(series)
    if table.empty:
        return pd.DataFrame(columns=columns)
    total = table.groupby("kunci")["jumlah"].sum().sort_values(ascending=False, kind="stable")
    spelled = table.drop_duplicates("kunci").set_index("kunci")["baku"]
    prints = pd.Series({key: fingerprint(spelled[key]) for key in total.index})
    edges = []
    for _, members in prints.groupby(prints, sort=False):
        edges += [(key, members.index[0], f"urutan kata sama: '{spelled[key]}' = '{spelled[members.index[0]]}'", True)
                  for key in members.index[1:]]
    edges += _education_edges(total, spelled) + _negative_edges(total, spelled, table)
    if typos:
        edges += _typo_edges(total, spelled, cutoff, min_ratio)
    for value, target in (extra or {}).items():
        if key_of(value) in total.index and key_of(target) in total.index and key_of(value) != key_of(target):
            edges.append((key_of(value), key_of(target), f"skrub.deduplicate: '{value}' → '{target}'", False))

    parent = {key: key for key in total.index}
    link = {}           # akar lama → (alasan, setuju) sisi yang menyambungkannya ke kelompok lain
    for a, b, why, ok in edges:
        while parent[a] != a:
            a = parent[a]
        while parent[b] != b:
            b = parent[b]
        if a != b:          # baku kelompok = kunci tersering, tapi tidak pernah kunci '' (isi hanya tanda baca)
            keep, other = (a, b) if (a != "", total[a]) >= (b != "", total[b]) else (b, a)
            parent[other] = keep
            link[other] = (why, ok)
    rows = []
    for key in total.index:
        top, path = key, []
        while parent[top] != top:
            path.append(link[top])
            top = parent[top]
        if top == key:
            continue
        # satu sambungan tak dicentang (mis. salah ketik) di jalur → baris ini juga tak dicentang (tidak ikut diam-diam)
        spellings = table.loc[table["kunci"] == key]
        rows.append({"kunci": key, "ejaan": ", ".join(f"{a!r}" for a in spellings["asli"].head(3)),
                     "jumlah": int(total[key]), "gabung ke": top, "jadi": spelled[top],
                     "alasan": " → lalu ".join(dict.fromkeys(why for why, _ in path)),
                     "setuju": all(ok for _, ok in path)})
    return pd.DataFrame(rows, columns=columns)


def same_key_table(series: pd.Series) -> pd.DataFrame:
    """Ejaan yang disatukan oleh aturan kunci saja (beda huruf besar/spasi/tanda baca/singkatan)."""
    table = key_table(series)
    table = table[table["asli"] != table["baku"]]
    return table[["asli", "baku", "jumlah", "kunci"]].rename(columns={"baku": "jadi"}).reset_index(drop=True)


def plan_groups(proposals: pd.DataFrame) -> tuple[dict, dict]:
    """Baris usulan yang dicentang → GABUNG {kunci baku: [kunci lain]} + komentar contoh ejaan per kelompok."""
    groups, examples = {}, {}
    chosen = proposals[proposals["setuju"].astype(bool)]
    for target, part in chosen.groupby("gabung ke", sort=False):
        groups[target] = sorted(part["kunci"])
        names = [str(text).split(", ")[0].strip("'\"") for text in part["ejaan"]]
        examples[target] = f"{part['jadi'].iloc[0]} ← " + ", ".join(names[:6]) + ("…" if len(names) > 6 else "")
    return groups, examples


# ---------------------------------------------------------------- urutan (ordinal)

def _number(text: str) -> float:
    return float(text.replace(".", "").replace(",", "."))


def ordinal_order(series: pd.Series) -> dict | None:
    """Usulan urutan untuk kolom kategori berurutan: jenjang pendidikan, atau rentang angka di dalam teks
    ('kurang dari Rp 1.000.000' < 'Rp 1.000.000 s.d. Rp 2.000.000' < 'lebih dari Rp 5.000.000').

    Kembalikan {"urutan": [...], "alasan": str}; urutan None bila satu jenjang masih punya beberapa ejaan.
    """
    values = list(series.dropna().unique())
    if not 3 <= len(values) <= 30 or not all(isinstance(v, str) for v in values):
        return None
    levels = {value: education_levels(value) for value in values}
    if all(levels.values()):
        rank = {value: min(level) for value, level in levels.items()}
        if len(set(rank.values())) < len(values):
            return {"urutan": None, "alasan": "jenjang pendidikan, tapi satu jenjang masih punya beberapa ejaan "
                                              "(mis. SMA, MA, SLTA): satukan dulu di tab Ejaan."}
        return {"urutan": sorted(values, key=rank.get), "alasan": "urutan jenjang pendidikan (kamus jenjang)"}
    numbers = {value: [_number(n) for n in re.findall(r"\d[\d.,]*\d|\d", value)] for value in values}
    if not all(numbers.values()):
        return None
    position = {}
    for value, found in numbers.items():
        if re.search(r"kurang|<|bawah|maks", value.lower()):
            position[value] = min(found) * 0.5
        elif re.search(r"lebih|>|atas|min", value.lower()):
            position[value] = max(found) * 1.5
        else:
            position[value] = sum(found) / len(found)
    return {"urutan": sorted(values, key=position.get), "alasan": "angka di dalam teks (rentang); "
                                                                  "'kurang dari' paling bawah, 'lebih dari' paling atas"}


# ---------------------------------------------------------------- isian bebas (per kata)

def words_of(series: pd.Series) -> pd.Series:
    parts = series.dropna().astype(str).str.lower().str.replace(r"[.\-]", " ", regex=True)
    parts = parts.str.split(MATERIAL_SEPARATORS, regex=True).explode().str.strip()
    return parts[parts != ""].str.split().explode()


def word_map(series: pd.Series, min_count: int = 30, cutoff: float = 75,
             related: list[pd.Series] | None = None) -> pd.DataFrame:
    """Kata jarang -> kata baku terdekat (rapidfuzz fuzz.ratio), ditambah kamus bahan (MATERIAL_WORDS).

    Kosakata baku = kata yang muncul >= min_count kali di kolom ini DAN kolom terkait (related),
    supaya kata sah di kolom lain ('batu' di Bahan Tembok) tidak dianggap salah ketik ('bata').
    Kata pendek (< 4 huruf) butuh skor >= 80. Kata di STOPWORDS tidak pernah dipetakan.
    """
    counts = words_of(series).value_counts()
    total = pd.concat([words_of(s) for s in [series, *(related or [])]]).value_counts()
    vocabulary = [w for w in total[total >= min_count].index if w not in STOPWORDS and re.fullmatch(r"[a-z]+", w)]
    rows = {}
    for word, count in counts.items():
        if word in STOPWORDS or not re.fullmatch(r"[a-z]{3,}", word):    # huruf saja: "kayu''" / '2x3' bukan kata
            continue
        limit = cutoff if len(word) >= 4 else max(cutoff, 80)
        note = ""
        if total[word] >= min_count:
            # kata sering tetap salah ketik bila ada kata mirip yang JAUH lebih sering DI KOLOM INI
            # (esbes 92 vs asbes 3.423). Dibanding di kolom ini saja: 'bata' sering di tembok, bukan di atap.
            bigger = [w for w in vocabulary if counts.get(w, 0) >= TYPO_RATIO * count]
            best = (process.extractOne(word, bigger, scorer=fuzz.ratio, score_cutoff=max(limit, 80))
                    if bigger else None)
            if best:
                note = f"sering ({int(count)}×), tapi '{best[0]}' {int(counts[best[0]] / count)}× lebih sering"
        else:
            best = process.extractOne(word, vocabulary, scorer=fuzz.ratio, score_cutoff=limit)
        if best:
            rows[word] = {"kata asli": word, "usulan": best[0], "skor": round(best[1]), "jumlah": int(count),
                          "setuju": best[1] >= 85 and len(word) >= 4,     # 3 huruf: beda 1 huruf = kata lain (ubi)
                          "catatan": note}
    for word, right in MATERIAL_WORDS.items():
        if word in counts and all(part in counts for part in right.split()):
            rows[word] = {"kata asli": word, "usulan": right, "skor": None, "jumlah": int(counts[word]),
                          "setuju": True, "catatan": "kamus bahan (KBBI / kategori BPS): sinonim, bukan salah ketik"
                          if fuzz.ratio(word, right) < 75 else "kamus bahan (KBBI / kategori BPS)"}
    final = resolve_chains({word: row["usulan"] for word, row in rows.items() if row["setuju"]})   # hanya lewat yang dicentang
    for word, row in rows.items():
        row["usulan"] = final.get(word, row["usulan"])
    table = pd.DataFrame(list(rows.values()), columns=["kata asli", "usulan", "skor", "jumlah", "setuju", "catatan"])
    return table.sort_values("jumlah", ascending=False, ignore_index=True)


def resolve_chains(words: dict[str, str]) -> dict[str, str]:
    """dikramik → kramik → keramik jadi dikramik → keramik (peta kata dijalankan sekali jalan, bukan berantai)."""
    resolved = {}
    for word, target in words.items():
        seen = {word}
        while target in words and target not in seen:
            seen.add(target)
            target = words[target]
        if target != word:
            resolved[word] = target
    return resolved


def similar_vocabulary(series: pd.Series, min_count: int = 30, cutoff: float = 80) -> pd.DataFrame:
    """Pasangan kata baku yang mirip: kandidat sinonim, tidak pernah otomatis (yang ada di kamus dilewati)."""
    counts = words_of(series).value_counts()
    vocabulary = [w for w in counts[counts >= min_count].index if w not in STOPWORDS]
    rows = []
    for i, first in enumerate(vocabulary):
        for second in vocabulary[i + 1:]:
            if MATERIAL_WORDS.get(first) == second or MATERIAL_WORDS.get(second) == first:
                continue
            score = fuzz.ratio(first, second)
            if score >= cutoff and counts[first] < TYPO_RATIO * counts[second]:   # yang timpang = salah ketik (word_map)
                rows.append({"kata 1": first, "jumlah 1": int(counts[first]), "kata 2": second,
                             "jumlah 2": int(counts[second]), "skor": round(score)})
    return pd.DataFrame(rows, columns=["kata 1", "jumlah 1", "kata 2", "jumlah 2", "skor"])


def fillers_for(column: str) -> list[str]:
    """Kata pengisi kolom ini: FILLER_WORDS + kata nama kolom ('Bahan Lantai Rumah': 'lantai keramik' = 'keramik')."""
    return sorted(FILLER_WORDS | set(re.findall(r"[a-z]{3,}", column.lower())))


def phrases_of(series: pd.Series, words: dict[str, str], fillers=()) -> pd.Series:
    """Bahan per bagian sel sesudah peta kata dan buang kata pengisi (sama dengan kode langkah); indeks = baris."""
    items = series.dropna().astype(str).str.lower().str.replace(r"[.\-]", " ", regex=True)
    items = items.str.split(MATERIAL_SEPARATORS, regex=True).explode().str.strip()
    items = items[items != ""]
    drop, fixed = set(fillers), []
    for item in items:
        tokens = " ".join(words.get(word, word) for word in item.split()).split()
        tokens = [word for i, word in enumerate(tokens) if i == 0 or word != tokens[i - 1]]   # 'ubin ubin' → 'ubin'
        fixed.append(" ".join([word for word in tokens if word not in drop] or tokens))   # 'tembok' saja tetap
    return pd.Series(fixed, index=items.index)


def phrase_map(series: pd.Series, words: dict[str, str], fillers=(), min_count: int = 5,
               min_ratio: float = 2) -> pd.DataFrame:
    """Usulan PETA_FRASA {bahan: bahan baku}, hanya untuk bahan yang muncul di ≥ min_count baris (ekor panjang
    tidak mengubah model, dan peta tetap pendek).

    - kamus MATERIAL_PHRASES ('bata merah' = 'batu bata'): dicentang;
    - urutan kata beda ('plester semen' = 'semen plester'): ke urutan tersering, dicentang;
    - memuat tepat satu bahan yang ≥ min_ratio× lebih sering ('genteng beton' memuat 'genteng'): TIDAK dicentang,
      karena kata tambahannya bisa rincian yang penting (beton ≠ tanah liat).
    """
    columns = ["setuju", "frasa", "jumlah", "usulan", "alasan"]
    parts = phrases_of(series, words, fillers)
    counts = parts.rename_axis("baris").reset_index(name="frasa").drop_duplicates()["frasa"].value_counts()
    frequent = counts[counts >= min_count]
    order = {}
    for phrase in counts.index:                       # tersering dulu: urutan kata tersering jadi baku
        order.setdefault(" ".join(sorted(phrase.split())), phrase)
    rows = []
    for phrase, count in frequent.items():
        tokens = set(phrase.split())
        same = order[" ".join(sorted(tokens))] if len(tokens) == len(phrase.split()) else phrase
        if phrase in MATERIAL_PHRASES:
            rows.append([True, phrase, count, MATERIAL_PHRASES[phrase], "kamus KBBI: sama makna"])
        elif same != phrase:
            rows.append([True, phrase, count, same, f"urutan kata beda dari '{same}' ({counts[same]:,} baris)"])
        else:
            inside = [other for other in frequent.index
                      if counts[other] >= min_ratio * count and set(other.split()) < tokens]
            longest = max((len(other.split()) for other in inside), default=0)
            inside = [other for other in inside if len(other.split()) == longest]
            if len(inside) == 1:                      # dua kandidat ('semen plester': semen? plester?) → lewati
                extra = " ".join(word for word in phrase.split() if word not in set(inside[0].split()))
                rows.append([False, phrase, count, inside[0],
                             f"memuat '{inside[0]}' ({counts[inside[0]] / count:.0f}× lebih sering); "
                             f"rincian '{extra}' hilang, periksa"])
    table = pd.DataFrame(rows, columns=columns)
    checked = table[table["setuju"].astype(bool)]           # rantai hanya lewat usulan yang dicentang
    final = resolve_chains(dict(zip(checked["frasa"], checked["usulan"])))   # 'merah bata' → 'bata merah' → 'batu bata'
    table["usulan"] = [final.get(target, target) for target in table["usulan"]]
    return table[table["usulan"] != table["frasa"]].reset_index(drop=True)


def material_preview(series: pd.Series, words: dict[str, str], top: int = 20, fillers=(),
                     phrases: dict[str, str] | None = None) -> pd.DataFrame:
    """Hasil pecah + peta kata (+ kata pengisi, peta frasa), untuk dilihat sebelum disetujui: bahan dan barisnya."""
    items = phrases_of(series, words, fillers).replace(phrases or {})
    items = items.rename_axis("baris").reset_index(name="bahan").drop_duplicates()
    share = items["bahan"].value_counts().head(top)
    return pd.DataFrame({"bahan": share.index, "baris": share.to_numpy(),
                         "% baris": (share / series.notna().sum() * 100).round(1).to_numpy()})


def deduplicate_map(series: pd.Series, n_clusters: int | None = None) -> pd.DataFrame:
    """skrub.deduplicate (opsional, bisa lambat untuk ribuan nilai unik). Hanya baris yang berubah."""
    import skrub

    values = series.dropna().astype(str)
    unique = pd.Series(values.unique())
    counts = values.value_counts()
    mapped = skrub.deduplicate(unique, n_clusters=n_clusters)
    table = pd.DataFrame({"asli": unique.to_numpy(), "usulan baku": list(mapped)})
    table["jumlah"] = table["asli"].map(counts)
    table = table[table["asli"] != table["usulan baku"]].sort_values("jumlah", ascending=False)
    return table.reset_index(drop=True)
