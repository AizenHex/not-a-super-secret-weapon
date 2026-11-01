"""Katalog operasi (ala Data Wrangler). Satu operasi = satu langkah = satu blok kode pandas LURUS.

Aturan kode yang dihasilkan:
- tanpa `def`: hanya baris pandas/numpy biasa, `for` dan `if` seperlunya;
- `train`, `valid`, dan `test` selalu ada (valid kosong bila belum di-split, test kosong bila tidak punya
  file test), jadi tidak perlu cabang;
- statistik (median, batas outlier, daftar kategori) dihitung dari `train` SAJA, lalu dipakai ke ketiganya:
  valid menilai langkahmu dengan jujur karena tidak pernah dipelajari;
- baris hanya dibuang dari `train`; `valid` dan `test` tidak pernah dibuang.

Setiap operasi: group, title, what (apa), fits (kapan cocok), danger (kapan berbahaya),
code(params) -> str, summary(params) -> str.
"""
from __future__ import annotations

import pprint
import re

MISSING_MARKERS = ["", "-", "--", "?", "nan", "null", "none", "n/a", "na", "tidak tahu"]
SEPARATORS = r"\s*(?:,|&|/|\+|;|\bdan\b|\batau\b|\(|\))\s*"
SENTINEL_FLOOR = 2_000_000_000
# kunci ejaan: huruf kecil, singkatan dipanjangkan, tanda baca & spasi dibuang ('S-1/D4' = 's1 / d-4' = 's1d4');
# tanda baca di antara dua angka diganti '_' supaya '1-2' tidak sama dengan '12'
ABBREVIATIONS = {r"\bkab\b": "kabupaten", r"\bkec\b": "kecamatan", r"\bkel\b": "kelurahan", r"\bprov\b": "provinsi"}
KEY_DIGITS = r"(?<=\d)\W+(?=\d)"
KEY_DROP = r"\W"


def lit(value) -> str:
    return pprint.pformat(value, width=92, sort_dicts=False)


def word_regex(words: dict[str, str]) -> dict[str, str]:
    """{kata salah: kata baku} -> {regex: kata baku}, satu regex per kata baku: '\\b(?:kramik|keramk)\\b'."""
    wrong_by_right = {}
    for wrong, right in words.items():
        wrong_by_right.setdefault(right, []).append(re.escape(wrong))
    return {rf"\b(?:{'|'.join(sorted(wrong))})\b": right for right, wrong in wrong_by_right.items()}


# ---------------------------------------------------------------- kolom

def drop_columns(p):
    return (f"KOLOM_DIBUANG = {lit(p['columns'])}\n"
            "train = train.drop(columns=KOLOM_DIBUANG)\n"
            "valid = valid.drop(columns=KOLOM_DIBUANG)\n"
            'test = test.drop(columns=KOLOM_DIBUANG, errors="ignore")')


def to_number(p):
    lines = [f"KOLOM_ANGKA_TEKS = {lit(p['columns'])}",
             "for df in (train, valid, test):",
             "    for kolom in KOLOM_ANGKA_TEKS:",
             '        teks = df[kolom].astype(str).str.replace(r"[^\\d,.\\-]", "", regex=True)']
    if p.get("indonesian", True):
        lines.append('        teks = teks.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)'
                     "  # 1.500.000,5 -> 1500000.5")
    else:
        lines.append('        teks = teks.str.replace(",", "", regex=False)')
    lines.append('        df[kolom] = pd.to_numeric(teks, errors="coerce")')
    return "\n".join(lines)


# ---------------------------------------------------------------- teks

def tidy_text(p):
    return (f"KOLOM_TEKS = {lit(p['columns'])}\n"
            f"PENANDA_KOSONG = {lit(p.get('markers', MISSING_MARKERS))}\n"
            "for df in (train, valid, test):\n"
            "    for kolom in KOLOM_TEKS:\n"
            '        teks = df[kolom].astype(str).str.strip().str.replace(r"\\s+", " ", regex=True)\n'
            "        df[kolom] = teks.where(df[kolom].notna() & ~teks.str.lower().isin(PENANDA_KOSONG))")


def change_case(p):
    return (f"KOLOM_HURUF = {lit(p['columns'])}\n"
            "for df in (train, valid, test):\n"
            "    for kolom in KOLOM_HURUF:\n"
            f"        df[kolom] = df[kolom].str.{p['case']}()")


def map_values(p):
    name = "PETA_" + re.sub(r"\W+", "_", p["column"].upper()).strip("_")[:24]
    return (f"{name} = {lit(p['mapping'])}\n"
            "for df in (train, valid, test):\n"
            f"    ganti = df[{p['column']!r}].isin({name}.keys())\n"
            f"    df.loc[ganti, {p['column']!r}] = df.loc[ganti, {p['column']!r}].map({name})")


def spelling(p):
    """p['groups'] = {kolom: {kunci baku: [kunci lain, ...]}}; contoh ejaan (p['examples']) hanya untuk komentar."""
    examples = p.get("examples", {})
    lines = ["# satukan ejaan: nilai berkunci sama = nilai yang sama ('Kab. Pati' = 'Kabupaten Pati', 'S- 2' = 'S2');",
             "# ejaan baku tiap kunci = ejaan tersering di train; GABUNG menyatukan kunci berbeda yang bermakna sama",
             "SINGKATAN = {" + ", ".join(f'r"{k}": "{v}"' for k, v in ABBREVIATIONS.items()) + "}",
             "GABUNG = {"]
    for column, groups in p["groups"].items():
        if not groups:
            lines.append(f"    {column!r}: {{}},")
            continue
        lines.append(f"    {column!r}: {{")
        for target, others in groups.items():
            note = examples.get(column, {}).get(target)
            lines.append(f"        {target!r}: {others!r}," + (f"   # {note}" if note else ""))
        lines.append("    },")
    lines += ["}",
              "for kolom, kelompok in GABUNG.items():",
              "    ke_baku = {}",
              "    for baku, lain in kelompok.items():",
              "        for kunci in lain:",
              "            ke_baku[kunci] = baku",
              "    kunci_tiap = []",
              "    for df in (train, valid, test):",
              "        kunci = df[kolom].astype(\"string\").str.lower().replace(SINGKATAN, regex=True)",
              f'        kunci_tiap.append(kunci.str.replace(r"{KEY_DIGITS}", "_", regex=True)'
              f'.str.replace(r"{KEY_DROP}", "", regex=True))',
              '    ejaan = train[kolom].str.strip().str.replace(r"\\s+", " ", regex=True)',
              '    ejaan_baku = pd.DataFrame({"kunci": kunci_tiap[0], "ejaan": ejaan}).value_counts().reset_index()',
              '    ejaan_baku = ejaan_baku.drop_duplicates("kunci").set_index("kunci")["ejaan"]   # tersering per kunci',
              "    sebelum = train[kolom].nunique()",
              "    for df, kunci in zip((train, valid, test), kunci_tiap):",
              "        df[kolom] = kunci.replace(ke_baku).map(ejaan_baku).fillna(df[kolom])",
              '    print(kolom, ":", sebelum, "->", train[kolom].nunique(), "nilai unik di train")']
    return "\n".join(lines)


def free_text(p):
    """p: column, words {kata salah: kata baku}; opsional fillers [kata pengisi] dan phrases {bahan: bahan baku}."""
    column, fillers, phrases = p["column"], p.get("fillers") or [], p.get("phrases") or {}
    words = "".join(f'\n    r"{pattern}": "{right}",' for pattern, right in word_regex(p["words"]).items())
    lines = [f"# isian bebas {column!r}: pecah per bahan, betulkan salah ketik per kata, urutkan",
             f'PEMISAH = r"{SEPARATORS}"',
             f"PETA_KATA = {{{words}\n}}   # regex kata salah: kata baku"]
    if fillers:
        pattern = "|".join(re.escape(word) for word in fillers)
        lines.append(f'PENGISI = r"\\b(?:{pattern})\\b"   # kata pengisi: \'lantai keramik\' = \'keramik\'')
    if phrases:
        lines.append("PETA_FRASA = {" + "".join(f"\n    {a!r}: {b!r}," for a, b in phrases.items()) + "\n}   # bahan: bahan baku")
    lines += ["for df in (train, valid, test):",
              f"    bagian = df[{column!r}].astype(\"string\").str.lower().str.replace(r\"[.\\-]\", \" \", regex=True)"
              "   # 'batu-bata.' → 'batu bata '",
              "    bagian = bagian.str.split(PEMISAH, regex=True).explode().dropna()",
              '    bagian = bagian.replace(PETA_KATA, regex=True).str.split().str.join(" ")   # spasi dirapikan',
              '    bagian = bagian.str.replace(r"\\b(\\w+)(?: \\1\\b)+", r"\\1", regex=True)   # \'tegel ubin\' → \'ubin ubin\' → \'ubin\'']
    if fillers:
        lines += ['    bersih = bagian.str.replace(PENGISI, "", regex=True).str.split().str.join(" ")',
                  '    bagian = bersih.where(bersih != "", bagian)   # isi hanya kata pengisi (\'tembok\') tetap']
    if phrases:
        lines.append("    bagian = bagian.replace(PETA_FRASA)")
    lines += ['    bagian = bagian[bagian != ""].rename_axis("baris").reset_index(name="bahan")',
              '    bagian = bagian.drop_duplicates().sort_values("bahan")',
              f"    df[{column!r}] = bagian.groupby(\"baris\")[\"bahan\"].agg(\", \".join)"]
    return "\n".join(lines)


# ---------------------------------------------------------------- kosong

def sentinel_to_nan(p):
    return (f"KOLOM_PENJAGA = {lit(p['columns'])}\n"
            f"BATAS_PENJAGA = {int(p.get('floor', SENTINEL_FLOOR)):_}  # 2.147.483.647 (2^31 - 1) dan variannya\n"
            "for df in (train, valid, test):\n"
            "    df[KOLOM_PENJAGA] = df[KOLOM_PENJAGA].mask(df[KOLOM_PENJAGA] >= BATAS_PENJAGA)")


def zero_to_nan(p):
    return (f"KOLOM_NOL_KOSONG = {lit(p['columns'])}\n"
            "for df in (train, valid, test):\n"
            "    df[KOLOM_NOL_KOSONG] = df[KOLOM_NOL_KOSONG].replace(0, np.nan)")


def fill_constant(p):
    return (f"ISI_TETAP = {lit({c: p['value'] for c in p['columns']})}\n"
            "for df in (train, valid, test):\n"
            "    df.fillna(ISI_TETAP, inplace=True)")


def impute(p):
    stat = {"median": ".median()", "mean": ".mean()", "mode": ".mode().iloc[0]"}[p["strategy"]]
    return (f"KOLOM_ISI = {lit(p['columns'])}\n"
            f"nilai_isi = {{kolom: train[kolom]{stat} for kolom in KOLOM_ISI}}   # dipelajari dari train\n"
            'print("nilai pengisi:", nilai_isi)\n'
            "for df in (train, valid, test):\n"
            "    df.fillna(nilai_isi, inplace=True)")


# ---------------------------------------------------------------- baris (train saja)

def drop_duplicates(p):
    return ("sebelum = len(train)\n"
            "train = train.drop_duplicates()\n"
            'print("duplikat dibuang:", sebelum - len(train))')


def drop_missing_rows(p):
    return (f"MAKS_KOSONG_PER_BARIS = {int(p['max_missing'])}\n"
            "sebelum = len(train)\n"
            "train = train[train.isna().sum(axis=1) <= MAKS_KOSONG_PER_BARIS].copy()\n"
            'print("baris terlalu banyak kosong dibuang:", sebelum - len(train))')


def drop_rows_query(p):
    return (f"ATURAN_BUANG = {p['rule']!r}\n"
            "sebelum = len(train)\n"
            "train = train[~train.eval(ATURAN_BUANG).fillna(False).astype(bool)].copy()\n"
            'print("dibuang oleh aturan:", sebelum - len(train))')


# ---------------------------------------------------------------- angka

def outlier(p):
    column, method, factor, action = p["column"], p["method"], p.get("factor", 0), p["action"]
    log = method.endswith("_log")
    if method == "manual":
        # batas dari pengetahuan tentang datanya (mis. tanggungan > 30 orang mustahil), bukan dari rumus
        low, high = p.get("low"), p.get("high")
        lines = [f"# outlier {column!r}: batas manual, aksi = {action}",
                 f"bawah, atas = {'-np.inf' if low is None else low}, {'np.inf' if high is None else high}"]
    else:
        lines = [f"# outlier {column!r}: {method}, k = {factor}, aksi = {action}",
                 f"nilai = train[{column!r}]" + (f"[train[{column!r}] > 0]" if log else ".dropna()")]
    if log:
        lines.append("nilai = np.log10(nilai)               # skala log: nol tidak dinilai")
    if method.startswith("iqr"):
        lines += ["q1, q3 = nilai.quantile([0.25, 0.75])",
                  "sebaran = q3 - q1",
                  f"bawah, atas = q1 - {factor} * sebaran, q3 + {factor} * sebaran"]
    elif method.startswith("mad"):
        lines += ["tengah = nilai.median()",
                  "sebaran = (nilai - tengah).abs().median() / 0.6745   # MAD (Iglewicz & Hoaglin)",
                  f"bawah, atas = tengah - {factor} * sebaran, tengah + {factor} * sebaran"]
    if log:
        lines.append("bawah, atas = 10 ** bawah, 10 ** atas")
    if method != "manual":
        lines += ["if not sebaran > 0:                    # mayoritas nilai sama -> metode tidak bisa membedakan",
                  "    bawah, atas = -np.inf, np.inf"]
    lines.append(f'print("batas {column}:", bawah, atas)')
    inside_low = f"(df[{column!r}] > 0) & " if log else ""
    if action == "clip":
        lines += ["for df in (train, valid, test):",
                  f"    df[{column!r}] = df[{column!r}].mask({inside_low}(df[{column!r}] < bawah), bawah)",
                  f"    df[{column!r}] = df[{column!r}].mask(df[{column!r}] > atas, atas)"]
    elif action == "nan":
        lines += ["for df in (train, valid, test):",
                  f"    df[{column!r}] = df[{column!r}].mask({inside_low}(df[{column!r}] < bawah) | (df[{column!r}] > atas))"]
    else:
        low = f"(train[{column!r}] > 0) & " if log else ""
        lines += ["sebelum = len(train)",
                  f"train = train[~({low}(train[{column!r}] < bawah) | (train[{column!r}] > atas))].copy()",
                  f'print("outlier {column} dibuang dari train:", sebelum - len(train))']
    return "\n".join(lines)


def log_transform(p):
    return (f"KOLOM_LOG = {lit(p['columns'])}\n"
            "for df in (train, valid, test):\n"
            "    df[KOLOM_LOG] = np.log1p(df[KOLOM_LOG].clip(lower=0))")


def scale(p):
    if p["method"] == "standard":
        stats = ("pusat = train[KOLOM_SKALA].mean()\n"
                 "lebar = train[KOLOM_SKALA].std(ddof=0).replace(0, 1)   # ddof=0 = StandardScaler")
    else:
        stats = ("pusat = train[KOLOM_SKALA].min()\n"
                 "lebar = (train[KOLOM_SKALA].max() - pusat).replace(0, 1)")
    return (f"KOLOM_SKALA = {lit(p['columns'])}\n{stats}   # dari train\n"
            "for df in (train, valid, test):\n"
            "    df[KOLOM_SKALA] = (df[KOLOM_SKALA] - pusat) / lebar")


# ---------------------------------------------------------------- encoding

def onehot(p):
    column, share = p["column"], p.get("min_share", 0.01)
    return (f"# one-hot {column!r}: nilai dengan porsi < {share} dan nilai baru di test -> {column}_lainnya\n"
            f"porsi = train[{column!r}].value_counts(normalize=True)\n"
            f"kategori = porsi[porsi >= {share}].index.tolist()\n"
            "hasil = []\n"
            "for df in (train, valid, test):\n"
            f"    dummies = pd.get_dummies(pd.Categorical(df[{column!r}], categories=kategori), "
            f"prefix={column!r}, prefix_sep=\"_\", dtype=int)\n"
            "    dummies.index = df.index\n"
            f"    dummies[{column + '_lainnya'!r}] = (df[{column!r}].notna() & ~df[{column!r}].isin(kategori)).astype(int)\n"
            f"    hasil.append(pd.concat([df.drop(columns={column!r}), dummies], axis=1))\n"
            "train, valid, test = hasil")


def ordinal(p):
    column = p["column"]
    if p.get("order"):
        source = f"URUTAN = {lit(p['order'])}\n"
    else:
        source = f"URUTAN = sorted(train[{column!r}].dropna().unique())   # abjad, dari train\n"
    return (f"# kode angka {column!r}: posisi di URUTAN; kosong / nilai baru = -1\n{source}"
            "kode = {nilai: nomor for nomor, nilai in enumerate(URUTAN)}\n"
            "for df in (train, valid, test):\n"
            f"    df[{column!r}] = df[{column!r}].map(kode).fillna(-1).astype(int)")


def materials_onehot(p):
    column, share = p["column"], p.get("min_share", 0.01)
    return (f"# satu kolom 0/1 per bahan di {column!r} (pisah ', '); bahan jarang/baru -> {column}_lainnya\n"
            "# (explode + crosstab: jauh lebih cepat dari str.get_dummies bila bahannya ratusan)\n"
            f"bagian = train[{column!r}].str.split(\", \").explode().dropna()\n"
            "bagian = bagian.rename_axis(\"baris\").reset_index(name=\"bahan\").drop_duplicates()\n"
            "porsi = bagian[\"bahan\"].value_counts() / len(train)   # porsi baris train yang punya bahan itu\n"
            f"bahan = porsi[porsi >= {share}].index.tolist()\n"
            "hasil = []\n"
            "for df in (train, valid, test):\n"
            f"    bagian = df[{column!r}].str.split(\", \").explode().dropna()\n"
            "    bagian = bagian.rename_axis(\"baris\").reset_index(name=\"bahan\").drop_duplicates()\n"
            "    dikenal = bagian[bagian[\"bahan\"].isin(bahan)]\n"
            "    dummies = pd.crosstab(dikenal[\"baris\"], dikenal[\"bahan\"])\n"
            f"    dummies = dummies.reindex(index=df.index, columns=bahan, fill_value=0).add_prefix({column + '_'!r})\n"
            "    lain = bagian.loc[~bagian[\"bahan\"].isin(bahan), \"baris\"]\n"
            f"    dummies[{column + '_lainnya'!r}] = df.index.isin(lain).astype(int)\n"
            "    dummies.index.name, dummies.columns.name = df.index.name, None\n"
            f"    hasil.append(pd.concat([df.drop(columns={column!r}), dummies], axis=1))\n"
            "train, valid, test = hasil")


SKRUB = {"string": "StringEncoder", "minhash": "MinHashEncoder", "gap": "GapEncoder"}


def skrub_encoder(p):
    column, kind, n = p["column"], p["kind"], int(p.get("n_components", 10))
    extra = ", random_state=0" if kind == "gap" else ""
    return (f"# skrub {SKRUB[kind]} {column!r}: kolom angka hasil tidak punya makna langsung\n"
            f"encoder = skrub.{SKRUB[kind]}(n_components={n}{extra})\n"
            f"angka_train = encoder.fit_transform(train[{column!r}])\n"
            "angka_train.index = train.index\n"
            f"train = pd.concat([train.drop(columns={column!r}), angka_train], axis=1)\n"
            "if len(valid):\n"
            f"    angka_valid = encoder.transform(valid[{column!r}])\n"
            "    angka_valid.index = valid.index\n"
            f"    valid = pd.concat([valid.drop(columns={column!r}), angka_valid], axis=1)\n"
            "else:\n"
            "    valid = train.iloc[:0]\n"
            "if len(test):\n"
            f"    angka_test = encoder.transform(test[{column!r}])\n"
            "    angka_test.index = test.index\n"
            f"    test = pd.concat([test.drop(columns={column!r}), angka_test], axis=1)\n"
            "else:\n"
            "    test = train.iloc[:0].drop(columns=[LABEL], errors=\"ignore\")")


def autogluon_features(p):
    return ("# AutoGluon AutoMLPipelineFeatureGenerator: tipe otomatis, tanggal, teks, kategori -> angka\n"
            "generator = AutoMLPipelineFeatureGenerator(verbosity=0)\n"
            "fitur = generator.fit_transform(train.drop(columns=[LABEL]))   # dipelajari dari train\n"
            "fitur.index = train.index\n"
            "train = pd.concat([fitur, train[[LABEL]]], axis=1)\n"
            "if len(valid):\n"
            "    fitur_valid = generator.transform(valid.drop(columns=[LABEL]))\n"
            "    fitur_valid.index = valid.index\n"
            "    valid = pd.concat([fitur_valid, valid[[LABEL]]], axis=1)\n"
            "else:\n"
            "    valid = train.iloc[:0]\n"
            "if len(test):\n"
            "    fitur_test = generator.transform(test.drop(columns=[LABEL], errors=\"ignore\"))\n"
            "    fitur_test.index = test.index\n"
            "    test = pd.concat([fitur_test, test.filter([LABEL])], axis=1)\n"
            "else:\n"
            "    test = fitur.iloc[:0]")


# ---------------------------------------------------------------- transformer sklearn / feature_engine (E1)
# Satu konstruktor dipakai DUA tempat: blok EDA (fit di train, transform valid & test) dan Pipeline sklearn
# (core/pipeline.py). Untuk op pandas lama (impute, scale, ...) konstruktornya dibuktikan memberi angka yang
# sama (tests/test_extra.py); op baru di bawah memakai transformer itu langsung di blok EDA.

TRANSFORMER_IMPORTS = {
    "SimpleImputer": "from sklearn.impute import SimpleImputer",
    "StandardScaler": "from sklearn.preprocessing import StandardScaler",
    "MinMaxScaler": "from sklearn.preprocessing import MinMaxScaler",
    "PowerTransformer": "from sklearn.preprocessing import PowerTransformer",
    "OrdinalEncoder": "from sklearn.preprocessing import OrdinalEncoder",
    "OneHotEncoder": "from sklearn.preprocessing import OneHotEncoder",
    "TargetEncoder": "from sklearn.preprocessing import TargetEncoder",
    "FunctionTransformer": "from sklearn.preprocessing import FunctionTransformer",
    "make_pipeline": "from sklearn.pipeline import make_pipeline",
    "StratifiedKFold": "from sklearn.model_selection import StratifiedKFold",
    "KFold": "from sklearn.model_selection import KFold",
    "Winsorizer": "from feature_engine.outliers import Winsorizer",
    "ArbitraryOutlierCapper": "from feature_engine.outliers import ArbitraryOutlierCapper",
    "RareLabelEncoder": "from feature_engine.encoding import RareLabelEncoder",
}


def _spec(make: str, columns: list[str], same: bool = True, target: bool = False) -> dict:
    """make = teks konstruktor; same = nama kolom keluaran = kolom masuk; target = fit butuh label."""
    names = [name for name in TRANSFORMER_IMPORTS if re.search(rf"\b{name}\(", make)]
    return {"make": make, "columns": list(columns), "same": same, "target": target,
            "imports": [TRANSFORMER_IMPORTS[name] for name in names]}


def transformer(step: dict) -> dict | None:
    """Spesifikasi transformer sklearn/feature_engine untuk satu langkah, None bila langkah itu hanya pandas."""
    op, p = step["op"], step["params"]
    if op == "impute":
        strategy = {"median": "median", "mean": "mean", "mode": "most_frequent"}[p["strategy"]]
        return _spec(f"SimpleImputer(strategy={strategy!r}, keep_empty_features=True)", p["columns"])
    if op == "fill_constant":
        return _spec(f"SimpleImputer(strategy=\"constant\", fill_value={p['value']!r}, keep_empty_features=True)",
                     p["columns"])
    if op == "scale":
        return _spec("StandardScaler()" if p["method"] == "standard" else "MinMaxScaler()", p["columns"])
    if op == "log_transform":
        return _spec("make_pipeline(FunctionTransformer(pd.DataFrame.clip, kw_args={\"lower\": 0}, "
                     "feature_names_out=\"one-to-one\"), FunctionTransformer(np.log1p, feature_names_out=\"one-to-one\"))",
                     p["columns"])
    if op == "ordinal":
        categories = f"[{lit(list(p['order']))}]" if p.get("order") else '"auto"'
        return _spec(f"OrdinalEncoder(categories={categories}, handle_unknown=\"use_encoded_value\", unknown_value=-1, "
                     "encoded_missing_value=-1)", [p["column"]])
    if op == "outlier" and p["method"] == "manual" and p["action"] == "clip":
        caps = []
        if p.get("high") is not None:
            caps.append(f"max_capping_dict={ {p['column']: p['high']} !r}")
        if p.get("low") is not None:
            caps.append(f"min_capping_dict={ {p['column']: p['low']} !r}")
        return _spec(f"ArbitraryOutlierCapper({', '.join(caps)}, missing_values=\"ignore\")", [p["column"]]) if caps else None
    if op in SKLEARN_OPS:
        return SKLEARN_OPS[op](p)
    return None


def _target_spec(p):
    if p.get("kind", "kelas") == "angka":
        make = "TargetEncoder(target_type=\"continuous\", cv=KFold(5, shuffle=True, random_state=0))"
    else:
        make = "TargetEncoder(target_type=\"auto\", cv=StratifiedKFold(5, shuffle=True, random_state=0))"
    return _spec(make, p["columns"], same=False, target=True)


SKLEARN_OPS = {
    "target_encode": _target_spec,
    "onehot_sklearn": lambda p: _spec(  # noqa: E731 (kode Laya, bukan kode hasil)
        f"OneHotEncoder(handle_unknown=\"infrequent_if_exist\", min_frequency={float(p.get('min_share', 0.01))}, "
        "sparse_output=False)", p["columns"], same=False),
    "rare_label": lambda p: _spec(
        f"RareLabelEncoder(tol={float(p.get('tol', 0.01))}, n_categories=1, replace_with=\"Lainnya\", "
        "missing_values=\"ignore\")", p["columns"]),
    "winsorize": lambda p: _spec(
        f"Winsorizer(capping_method={p.get('method', 'iqr')!r}, tail=\"both\", fold={float(p.get('fold', 1.5))}, "
        "missing_values=\"ignore\")", p["columns"]),
    "power": lambda p: _spec("PowerTransformer(method=\"yeo-johnson\")", p["columns"]),
}


def transformer_code(p, op: str) -> str:
    """Blok EDA lurus untuk op sklearn/feature_engine: fit di train (label-nya bila perlu), transform semua."""
    spec = SKLEARN_OPS[op](p)
    lines = [*spec["imports"],
             f"KOLOM_UBAH = {lit(spec['columns'])}",
             f"pengubah = {spec['make']}.set_output(transform=\"pandas\")"]
    if spec["target"]:
        lines += ["berlabel = train[LABEL].notna()",
                  "# fit_transform = cross-fitting: baris train tidak melihat labelnya sendiri (fit lalu transform = bocor)",
                  "angka_train = pengubah.fit_transform(train.loc[berlabel, KOLOM_UBAH], train.loc[berlabel, LABEL])",
                  "angka_train = angka_train.reindex(train.index)",
                  "if not berlabel.all():",
                  "    angka_train.loc[~berlabel] = pengubah.transform(train.loc[~berlabel, KOLOM_UBAH]).to_numpy()"]
    else:
        lines.append("pengubah.fit(train[KOLOM_UBAH])   # dipelajari dari train saja")
    if spec["same"] and not spec["target"]:
        lines += ["for df in (train, valid, test):",
                  "    if len(df):                     # frame 0 baris dilewati (transformer menolak 0 sampel)",
                  "        df[KOLOM_UBAH] = pengubah.transform(df[KOLOM_UBAH])"]
        return "\n".join(lines)
    train_part = "angka_train" if spec["target"] else "pengubah.transform(train[KOLOM_UBAH])"
    lines += ["hasil = []",
              "for df in (train, valid, test):",
              "    if df is train:",
              f"        angka = {train_part}",
              "    elif len(df):",
              "        angka = pengubah.transform(df[KOLOM_UBAH])",
              "    else:",
              "        angka = pd.DataFrame(columns=pengubah.get_feature_names_out(), dtype=float)",
              "    angka.index = df.index",
              "    hasil.append(pd.concat([df.drop(columns=KOLOM_UBAH), angka], axis=1))",
              "train, valid, test = hasil"]
    return "\n".join(lines)


# ---------------------------------------------------------------- katalog

OPS = {
    "drop_columns": dict(group="Kolom", title="Buang kolom", code=drop_columns,
                         what="Hapus kolom dari train, valid, dan test.",
                         fits="Kolom id, kolom kembar (isi sama dengan kolom lain), kolom bocoran label.",
                         danger="Membuang kolom yang ternyata berguna. Lihat tab AutoGluon → kolom penting."),
    "to_number": dict(group="Kolom", title="Teks → angka", code=to_number,
                      what="Baca angka yang tersimpan sebagai teks: 'Rp 1.500.000,5' → 1500000.5.",
                      fits="Kolom harga/luas yang terbaca sebagai teks.",
                      danger="Teks yang bukan angka jadi kosong (NaN). Cek jumlah kosong sesudahnya."),
    "tidy_text": dict(group="Teks", title="Rapikan teks + penanda kosong", code=tidy_text,
                      what="Buang spasi berlebih; '-', '?', 'null', … jadi kosong (NaN).",
                      fits="Semua kolom teks/kategori, sebelum menyatukan ejaan.",
                      danger="Jika '-' berarti 'tidak ada' (bukan 'tidak diisi'), isi kembali dengan Isi nilai tetap."),
    "change_case": dict(group="Teks", title="Ubah huruf besar/kecil", code=change_case,
                        what="Semua huruf kecil/besar/judul.",
                        fits="'Genteng' dan 'genteng' harus dianggap sama.",
                        danger="Singkatan (SMA, KIPK) ikut berubah."),
    "spelling": dict(group="Teks", title="Satukan ejaan (kunci + gabung)", code=spelling,
                     what="Nilai yang hanya beda huruf besar/spasi/tanda baca/singkatan disatukan oleh satu aturan "
                          "('S- 2' = 'S2', 'Kab. Pati' = 'Kabupaten Pati'); kunci berbeda yang bermakna sama "
                          "(SMA = MA = SLTA) digabung lewat daftar pendek GABUNG.",
                     fits="Kolom kategori. Usulan dari tab Ejaan. Kode tetap pendek berapa pun variasinya.",
                     danger="GABUNG yang salah (Kota Malang ≠ Kota Magelang, D1 ≠ D3). Periksa alasan tiap gabungan."),
    "map_values": dict(group="Teks", title="Ganti nilai (peta nilai)", code=map_values,
                       what="Ganti nilai tertentu persis: {'Lainnya ': 'Lainnya'}. Untuk ejaan, pakai Satukan ejaan.",
                       fits="Cari & ganti manual beberapa nilai.",
                       danger="Peta ratusan pasangan = kode panjang yang tidak berlaku untuk ejaan baru di test."),
    "free_text": dict(group="Teks", title="Bakukan isian bebas (per bahan)", code=free_text,
                      what="'Genteng & asbas.' → 'asbes, genteng': pecah per bahan, betulkan kata, urutkan.",
                      fits="Kolom bahan atap/tembok/lantai, daftar fasilitas.",
                      danger="Peta kata yang salah (batu → bata). Hanya pakai kata yang sudah dicentang."),
    "sentinel_to_nan": dict(group="Kosong", title="Nilai penjaga → kosong", code=sentinel_to_nan,
                            what="Nilai ≥ 2.000.000.000 (2^31 − 1 dan varian) dijadikan kosong.",
                            fits="Angka 2.147.483.647 yang dipakai sistem sebagai 'tidak diisi'.",
                            danger="Nilai besar yang asli (hutang 2 miliar). Lihat 'maks wajar' di tab Temuan."),
    "zero_to_nan": dict(group="Kosong", title="Nol → kosong", code=zero_to_nan,
                        what="0 dijadikan kosong.",
                        fits="Nol yang mustahil (luas tanah 0, daya listrik 0).",
                        danger="Nol sungguhan (tidak punya hutang) jadi hilang."),
    "fill_constant": dict(group="Kosong", title="Isi nilai tetap", code=fill_constant,
                          what="Sel kosong diisi nilai pilihan, mis. 'Tidak'.",
                          fits="Kolom checkbox: satu nilai + kosong (kosong = tidak).",
                          danger="Kosong yang artinya 'tidak tahu' dianggap 'tidak'."),
    "impute": dict(group="Kosong", title="Isi kosong (median/rata-rata/modus)", code=impute,
                   what="Isi kosong dengan statistik dari train; nilai yang sama dipakai untuk valid dan test.",
                   fits="Kolom dengan sedikit kosong. Angka miring → median.",
                   danger="Kosong banyak → info hilang. Banyak model (AutoGluon, LightGBM) menerima kosong."),
    "drop_duplicates": dict(group="Baris", title="Buang duplikat (train)", code=drop_duplicates,
                            what="Hapus baris train yang persis sama.",
                            fits="Data hasil gabungan beberapa sumber.",
                            danger="Dua orang berbeda dengan isian sama ikut terbuang."),
    "drop_missing_rows": dict(group="Baris", title="Buang baris banyak kosong (train)", code=drop_missing_rows,
                              what="Hapus baris train dengan kosong lebih dari batas.",
                              fits="Baris rusak (20+ kolom kosong).",
                              danger="Batas terlalu kecil membuang banyak data."),
    "drop_rows_query": dict(group="Baris", title="Buang baris dengan aturan (train)", code=drop_rows_query,
                            what="Hapus baris train yang cocok dengan aturan pandas, mis. `Luas` > 100000.",
                            fits="Data mustahil.",
                            danger="Aturan terlalu luas membuang data sah."),
    "outlier": dict(group="Angka", title="Tangani outlier", code=outlier,
                    what="Batas wajar dari train (IQR/MAD, opsional skala log), lalu potong, kosongkan, atau buang.",
                    fits="Angka kontinu (luas, harga). Bandingkan metode di tab Outlier dulu.",
                    danger="Buang/potong nilai ekstrem yang asli. Kolom diskret sering salah ditandai."),
    "log_transform": dict(group="Angka", title="Transformasi log1p", code=log_transform,
                          what="x → log(1 + x). Membuat angka miring lebih simetris.",
                          fits="Model linear; harga/luas.",
                          danger="Model pohon tidak butuh; nilai negatif dipotong ke 0."),
    "scale": dict(group="Angka", title="Skala (standar / min-max)", code=scale,
                  what="Ubah skala angka memakai rata-rata/simpangan atau min/maks train.",
                  fits="Model linear, KNN, jaringan saraf.",
                  danger="Model pohon dan AutoGluon tidak butuh."),
    "onehot": dict(group="Encoding", title="One-hot", code=onehot,
                   what="Satu kolom 0/1 per nilai; nilai jarang/baru → <kolom>_lainnya.",
                   fits="Kategori sedikit (≤ ±20) tanpa urutan.",
                   danger="Ratusan nilai → ratusan kolom. Satukan ejaan dulu."),
    "ordinal": dict(group="Encoding", title="Kode angka / berurutan", code=ordinal,
                    what="Nilai → nomor urut (abjad, atau urutan yang kamu tulis). Kosong/baru = −1.",
                    fits="Tingkat pendidikan, rentang pendapatan; model pohon.",
                    danger="Model linear membaca nomor sebagai jarak."),
    "materials_onehot": dict(group="Encoding", title="One-hot per bahan", code=materials_onehot,
                             what="'asbes, genteng' → kolom bahan_asbes = 1, bahan_genteng = 1.",
                             fits="Setelah Bakukan isian bebas.",
                             danger="Tanpa pembakuan, setiap salah ketik jadi kolom."),
    "skrub_encoder": dict(group="Encoding", title="skrub String/MinHash/Gap", code=skrub_encoder,
                          what="Teks bervariasi banyak → beberapa kolom angka (n-gram huruf).",
                          fits="Nama prodi, kota dengan ratusan nilai, untuk model pohon.",
                          danger="Kolom hasil tidak bisa dibaca maknanya."),
    "autogluon_features": dict(group="Encoding", title="AutoGluon feature generator", code=autogluon_features,
                               what="AutoGluon mengubah seluruh tabel jadi fitur siap model (kategori, teks, tanggal).",
                               fits="Langkah terakhir bila model yang dipakai bukan AutoGluon.",
                               danger="Kolom bisa dibuang/diubah otomatis; lihat sebelum → sesudah. "
                                      "TabularPredictor sudah melakukan ini sendiri."),
    "target_encode": dict(group="Encoding", title="Target encoding (sklearn TargetEncoder)",
                          code=lambda p: transformer_code(p, "target_encode"),
                          what="Kategori → rata-rata label per kategori (per kelas bila label kelas), dihaluskan. "
                               "Train memakai fit_transform (cross-fitting 5 fold), valid/test memakai transform.",
                          fits="Kategori dengan banyak nilai (kota, prodi) untuk model pohon.",
                          danger="Fit lalu transform di train = bocor (skor terlalu bagus). Kategori sangat jarang "
                                 "tetap berisik walau dihaluskan."),
    "onehot_sklearn": dict(group="Encoding", title="One-hot (sklearn OneHotEncoder)",
                           code=lambda p: transformer_code(p, "onehot_sklearn"),
                           what="Satu kolom 0/1 per nilai; nilai dengan porsi < min_frequency dan nilai baru di "
                                "test → kolom <kolom>_infrequent_sklearn. Kosong = kategori sendiri (<kolom>_nan).",
                           fits="Kategori sedikit tanpa urutan, bila ingin langkah ini ikut Pipeline sklearn.",
                           danger="Ratusan nilai → ratusan kolom. Satukan ejaan dulu."),
    "rare_label": dict(group="Encoding", title="Kategori langka → Lainnya (feature_engine)",
                       code=lambda p: transformer_code(p, "rare_label"),
                       what="Kategori dengan porsi < tol di train (dan kategori baru di test) diganti 'Lainnya'.",
                       fits="Sebelum one-hot/target encoding pada kolom berekor panjang.",
                       danger="Kategori langka yang penting (kelas minoritas) ikut tergabung."),
    "winsorize": dict(group="Angka", title="Winsorize (feature_engine Winsorizer)",
                      code=lambda p: transformer_code(p, "winsorize"),
                      what="Potong nilai di luar batas IQR/MAD/gaussian/quantile dari train ke batasnya.",
                      fits="Angka kontinu dengan ekor panjang; masuk Pipeline sklearn.",
                      danger="Sebaran 0 (mayoritas nilai sama) → batas = nilai itu, semua nilai lain terpotong. "
                             "Tanpa skala log; untuk IQR-log pakai Tangani outlier."),
    "power": dict(group="Angka", title="PowerTransformer (Yeo-Johnson)",
                  code=lambda p: transformer_code(p, "power"),
                  what="Ubah angka miring jadi mendekati normal lalu distandarkan (parameter dari train).",
                  fits="Model linear/KNN pada angka sangat miring.",
                  danger="Model pohon tidak butuh; nilai hasil sulit dibaca."),
}


def summary(step: dict) -> str:
    """Teks pendek untuk daftar langkah."""
    if step["op"] not in OPS:   # mis. 'rename_column' (dihapus 2026-11-02, R8): langkah lama tetap terlihat
        return f"(operasi tidak dikenal: {step['op']})"
    p, op = step["params"], OPS[step["op"]]["title"]
    columns = p.get("columns") or list(p.get("groups", {}))
    target = p.get("column") or ", ".join(columns[:3]) + ("…" if len(columns) > 3 else "")
    extra = {"outlier": f" ({p.get('method')}, {p.get('action')})", "impute": f" ({p.get('strategy')})",
             "spelling": f" ({sum(len(o) for g in p.get('groups', {}).values() for o in g.values())} gabung)",
             "map_values": f" ({len(p.get('mapping', {}))} nilai)", "free_text": f" ({len(p.get('words', {}))} kata, {len(p.get('phrases') or {})} frasa)",
             "drop_rows_query": f": {p.get('rule')}", "skrub_encoder": f" ({p.get('kind')})"}.get(step["op"], "")
    return f"{op}" + (f" · {target}" if target else "") + extra
