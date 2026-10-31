"""Resep → kode pipeline Python (pandas, numpy, scikit-learn, pandera, skrub, joblib).

Kode yang dihasilkan berdiri sendiri: tidak mengimpor apa pun dari Laya EDA. Aplikasi
menjalankan kode yang sama persis (core/runner.py), jadi data di aplikasi = data dari kode.
Bagian dipisah dengan penanda `# %%` supaya bisa dijalankan per sel (VS Code / Jupyter).
"""
from __future__ import annotations

import json
import pprint
import re

SKRUB_ENCODERS = {"string": "StringEncoder", "minhash": "MinHashEncoder", "gap": "GapEncoder"}


def literal(value) -> str:
    return pprint.pformat(value, width=96, sort_dicts=False)


def slug(name: str) -> str:
    return re.sub(r"\W+", "_", name.lower()).strip("_")[:30]


# ---------------------------------------------------------------- potongan kode tetap

TO_NUMBER = '''
def to_number(series: pd.Series) -> pd.Series:
    """'Rp 1.500.000,5' -> 1500000.5 (format Indonesia); teks yang bukan angka -> NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("string").str.replace(r"[^\\d,.\\-]", "", regex=True)
    if INDONESIAN_NUMBERS:
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        text = text.str.replace(",", "", regex=False)
    return pd.to_numeric(text.astype(object), errors="coerce").astype(float)
'''

TIDY_TEXT = '''
def tidy_text(series: pd.Series) -> pd.Series:
    """Rapikan spasi; penanda kosong palsu ('-', '?', 'null', ...) -> NaN."""
    text = series.astype("string").str.strip().str.replace(r"\\s+", " ", regex=True)
    text = text.mask(text.str.lower().isin(MISSING_MARKERS))
    return text.astype(object).where(text.notna(), np.nan)
'''

CANONICAL_MATERIALS = '''
def canonical_materials(series: pd.Series, words: dict[str, str]) -> pd.Series:
    """Isian bebas -> daftar bahan baku: 'Genteng & asbas.' -> 'asbes, genteng'.

    Huruf kecil, pecah di pemisah (, & / + ; dan atau), betulkan salah ketik per kata
    memakai peta kata, buang duplikat, urutkan supaya urutan tulisan tidak berpengaruh.
    """
    def fix(parts):
        if not isinstance(parts, list):
            return np.nan
        items = {" ".join(words.get(w, w) for w in part.split()) for part in parts}
        items.discard("")
        return ", ".join(sorted(items)) if items else np.nan

    pieces = series.str.lower().str.replace(r"[.]", " ", regex=True).str.split(MATERIAL_SEPARATORS)
    return pieces.map(fix)
'''

OUTLIER_BOUNDS = '''
def outlier_bounds(values: pd.Series, method: str, factor: float) -> tuple[float, float]:
    """Batas wajar dari train. iqr: Tukey; mad: modified z-score (Iglewicz & Hoaglin).

    Akhiran _log = dihitung di skala log10 (hanya nilai > 0; nol tidak pernah ditandai).
    Sebaran nol (mis. mayoritas nilai sama) -> metode tidak bisa membedakan -> tanpa batas.
    """
    log = method.endswith("_log")
    x = np.log10(values[values > 0]) if log else values.dropna()
    if method.startswith("iqr"):
        q1, q3 = x.quantile([0.25, 0.75])
        spread = q3 - q1
        low, high = q1 - factor * spread, q3 + factor * spread
    else:
        center = x.median()
        spread = (x - center).abs().median() / 0.6745
        low, high = center - factor * spread, center + factor * spread
    if not spread > 0:
        return -np.inf, np.inf
    return (10**low, 10**high) if log else (float(low), float(high))


def is_outside(values: pd.Series, bounds: tuple[float, float], method: str) -> pd.Series:
    low, high = bounds
    outside = (values < low) | (values > high)
    return outside & (values > 0) if method.endswith("_log") else outside


def fit_outlier_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        column: outlier_bounds(df[column], spec["method"], spec["factor"])
        for column, spec in OUTLIERS.items()
        if column in df
    }


def apply_outliers(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Aksi 'clip' dan 'nan' (train + test). 'flag' dan 'drop' tidak mengubah nilai di sini."""
    df = df.copy()
    for column, (low, high) in bounds.items():
        spec = OUTLIERS[column]
        outside = is_outside(df[column], (low, high), spec["method"])
        if spec["action"] == "clip":
            df[column] = df[column].where(~outside, df[column].clip(low, high))
        elif spec["action"] == "nan":
            df[column] = df[column].mask(outside)
    return df
'''

NO_OUTLIERS = '''
def fit_outlier_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {}


def apply_outliers(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return df
'''

MATERIALS = '''
def fit_materials(df: pd.DataFrame) -> dict[str, list[str]]:
    """Bahan yang muncul di >= min_share baris train jadi kolom; sisanya ke '<kolom>_lainnya'."""
    vocabulary = {}
    for column, min_share in MATERIAL_COLUMNS.items():
        share = df[column].str.get_dummies(sep=", ").mean()
        vocabulary[column] = share[share >= min_share].sort_values(ascending=False).index.tolist()
    return vocabulary


def encode_materials(df: pd.DataFrame, vocabulary: dict[str, list[str]]) -> pd.DataFrame:
    for column, keep in vocabulary.items():
        dummies = df[column].str.get_dummies(sep=", ")
        block = dummies.reindex(columns=keep, fill_value=0).add_prefix(f"{column}_")
        block[f"{column}_lainnya"] = dummies.drop(columns=keep, errors="ignore").any(axis=1).astype(int)
        df = pd.concat([df.drop(columns=column), block], axis=1)
    return df
'''

NO_MATERIALS = '''
def fit_materials(df: pd.DataFrame) -> dict[str, list[str]]:
    return {}


def encode_materials(df: pd.DataFrame, vocabulary: dict[str, list[str]]) -> pd.DataFrame:
    return df
'''

FINISH = '''
def restore_order(frame: pd.DataFrame, original: pd.Index) -> pd.DataFrame:
    """ColumnTransformer mengubah urutan kolom; kembalikan urutan asli, kolom baru di belakang."""
    first = [column for column in original if column in frame.columns]
    return frame[first + [column for column in frame.columns if column not in first]]


def make_schema(frame: pd.DataFrame) -> pa.DataFrameSchema:
    """Skema dari train: nama, urutan, dan tipe kolom. Test yang berbeda -> error yang jelas."""
    columns = {name: pa.Column(str(dtype), nullable=True) for name, dtype in frame.dtypes.items()}
    return pa.DataFrameSchema(columns, strict=True, ordered=True, coerce=True)


def transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Proses data baru (train, test, inferensi) memakai statistik yang dipelajari dari train."""
    df = df.reindex(columns=state["input_columns"])     # kolom hilang -> NaN, kolom lebih dibuang
    df = apply_outliers(prepare(df), state["bounds"])
    df = encode_materials(df, state["materials"])
    return restore_order(state["preprocessor"].transform(df), df.columns)
'''

RUN = '''
def run(train: pd.DataFrame, test: pd.DataFrame | None = None) -> dict:
    """Pelajari statistik dari train, lalu proses train dan test dengan jalur yang sama."""
    label = train[LABEL] if LABEL else None
    features = train.drop(columns=[LABEL]) if LABEL else train
    prepared = prepare(features)
    bounds = fit_outlier_bounds(prepared)
    reasons = rows_to_drop(prepared if label is None else prepared.assign(**{LABEL: label}), bounds)
    keep = ~reasons.any(axis=1)

    kept = apply_outliers(prepared.loc[keep], bounds)
    materials = fit_materials(kept)
    preprocessor = build_preprocessor().fit(encode_materials(kept, materials))
    state = {
        "input_columns": features.columns.tolist(),
        "bounds": bounds,
        "materials": materials,
        "preprocessor": preprocessor,
    }

    train_clean = transform(features.loc[keep], state)
    state["schema"] = make_schema(train_clean)
    if label is not None:
        train_clean[LABEL] = label.loc[keep].to_numpy()

    test_clean, warnings = None, []
    if empty := kept.columns[kept.isna().all()].tolist():
        warnings.append(f"kolom kosong semua di train (imputer mengisi 0, sebaiknya dibuang): {empty}")
    if test is not None:
        expected = set(state["input_columns"])
        if missing := sorted(expected - set(test.columns)):
            warnings.append(f"kolom tidak ada di test (diisi kosong): {missing}")
        if extra := sorted(set(test.columns) - expected):
            warnings.append(f"kolom test yang tidak ada di train (diabaikan): {extra}")
        test_clean = state["schema"].validate(transform(test, state), lazy=True)
    return {
        "train": train_clean,
        "test": test_clean,
        "state": state,
        "prepared": prepared,
        "drop_reasons": reasons,
        "warnings": warnings,
    }
'''


# ---------------------------------------------------------------- bagian yang tergantung resep

def uses(recipe: dict, method: str) -> bool:
    return any(spec["method"] == method for spec in recipe["encoding"].values())


def skrub_used(recipe: dict) -> bool:
    return any(spec["method"] in SKRUB_ENCODERS for spec in recipe["encoding"].values())


def imports(recipe: dict) -> str:
    lines = [
        "from pathlib import Path",
        "",
        "import joblib",
        "import numpy as np",
        "import pandas as pd",
        "import pandera.pandas as pa",
    ]
    if skrub_used(recipe):
        lines.append("import skrub")
    lines += ["from sklearn.compose import ColumnTransformer", "from sklearn.impute import SimpleImputer"]
    if recipe["split"]["method"] in ("random", "stratified"):
        lines.append("from sklearn.model_selection import train_test_split")
    if recipe["split"]["method"] == "kfold":
        lines.append("from sklearn.model_selection import KFold, StratifiedKFold")
    if any(spec["method"] != "materials" for spec in recipe["encoding"].values()):
        lines.append("from sklearn.pipeline import make_pipeline")
    encoders = [name for name, method in (("OneHotEncoder", "onehot"), ("OrdinalEncoder", "ordinal"))
                if uses(recipe, method) or (method == "ordinal" and uses(recipe, "ordered"))]
    if encoders:
        lines.append(f"from sklearn.preprocessing import {', '.join(encoders)}")
    return "\n".join(lines)


def constants(recipe: dict, paths: dict) -> str:
    material_columns = {column: spec.get("min_share", 0.01)
                        for column, spec in recipe["encoding"].items() if spec["method"] == "materials"}
    order = {column: spec["order"] for column, spec in recipe["encoding"].items()
             if spec["method"] == "ordered"}
    values = {
        "TRAIN_CSV": paths.get("train"),
        "TEST_CSV": paths.get("test"),
        "OUTPUT_DIR": paths.get("output", "hasil"),
        "LABEL": recipe["label"],
        "DROP_COLUMNS": recipe["drop_columns"],
        "NUMBER_COLUMNS": recipe["number_columns"],
        "INDONESIAN_NUMBERS": recipe["indonesian_numbers"],
        "TEXT_COLUMNS": recipe["text_columns"],
        "MISSING_MARKERS": recipe["missing_markers"],
        "SENTINELS": recipe["sentinels"],
        "ZERO_AS_MISSING": recipe["zero_as_missing"],
        "BLANK_AS": recipe["blank_as"],
        "CATEGORY_MAPS": recipe["category_maps"],
        "FREE_TEXT": recipe["free_text"],
        "MATERIAL_SEPARATORS": r"\s*(?:,|&|/|\+|;|\bdan\b|\batau\b|\(|\))\s*",
        "OUTLIERS": recipe["outliers"],
        "DROP_DUPLICATES": recipe["drop_duplicates"],
        "MAX_MISSING_PER_ROW": recipe["max_missing_per_row"],
        "DROP_QUERIES": recipe["drop_queries"],
        "MATERIAL_COLUMNS": material_columns,
        "ORDER": order,
        "SPLIT": recipe["split"],
    }
    skip = set()
    if not recipe["free_text"]:
        skip.add("MATERIAL_SEPARATORS")
    if not material_columns:
        skip.add("MATERIAL_COLUMNS")
    if not order:
        skip.add("ORDER")
    if not recipe["number_columns"]:
        skip.add("INDONESIAN_NUMBERS")
    if not recipe["text_columns"]:
        skip.add("MISSING_MARKERS")
    lines = [f"{name} = {literal(value)}" for name, value in values.items() if name not in skip]
    return "\n".join(lines)


def prepare_function(recipe: dict) -> str:
    body = ['    df = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()']
    if recipe["number_columns"]:
        body += ["    numbers = [column for column in NUMBER_COLUMNS if column in df]",
                 "    df[numbers] = df[numbers].apply(to_number)"]
    if recipe["sentinels"]:
        body += ["    for column, floor in SENTINELS.items():",
                 "        df[column] = df[column].mask(df[column] >= floor)   # 2147483647 dkk"]
    if recipe["zero_as_missing"]:
        body.append("    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)")
    if recipe["text_columns"]:
        body += ["    texts = [column for column in TEXT_COLUMNS if column in df]",
                 "    df[texts] = df[texts].apply(tidy_text)"]
    if recipe["blank_as"]:
        body.append("    df = df.fillna(BLANK_AS)")
    if recipe["category_maps"]:
        body += ["    for column, mapping in CATEGORY_MAPS.items():     # {ejaan asli: ejaan baku}",
                 "        df[column] = df[column].map(lambda value: mapping.get(value, value))"]
    if recipe["free_text"]:
        body += ["    for column, words in FREE_TEXT.items():",
                 "        df[column] = canonical_materials(df[column], words)"]
    body.append("    return df")
    head = ('def prepare(df: pd.DataFrame) -> pd.DataFrame:\n'
            '    """Langkah tanpa statistik train (tipe, penanda kosong, peta ejaan): sama untuk semua data."""')
    return head + "\n" + "\n".join(body)


def drop_function(recipe: dict) -> str:
    body = ["    reasons = pd.DataFrame(index=df.index)"]
    if recipe["drop_duplicates"]:
        body.append('    reasons["duplikat"] = df.duplicated()')
    if recipe["max_missing_per_row"] is not None:
        body.append('    reasons["banyak_kosong"] = df.isna().sum(axis=1) > MAX_MISSING_PER_ROW')
    if any(spec["action"] == "drop" for spec in recipe["outliers"].values()):
        body += ["    for column, bounds in bounds.items():",
                 '        if OUTLIERS[column]["action"] == "drop":',
                 '            reasons[f"outlier: {column}"] = is_outside(df[column], bounds, OUTLIERS[column]["method"])']
    if recipe["drop_queries"]:
        body += ["    for rule in DROP_QUERIES:",
                 '        reasons[f"aturan: {rule}"] = df.eval(rule).fillna(False).astype(bool)']
    body.append("    return reasons.astype(bool)")
    head = ('def rows_to_drop(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:\n'
            '    """Satu kolom True/False per alasan membuang baris. Hanya untuk train; test tidak dibuang."""')
    return head + "\n" + "\n".join(body)


def imputer_expression(spec: dict) -> str:
    if spec["strategy"] == "constant":
        return f'SimpleImputer(strategy="constant", fill_value={spec.get("value")!r}, keep_empty_features=True)'
    return f'SimpleImputer(strategy="{spec["strategy"]}", keep_empty_features=True)'


def encoder_expression(column: str, spec: dict) -> str:
    method = spec["method"]
    if method == "onehot":
        share = spec.get("min_share", 0.01) or None          # 0 → tanpa batas jarang (sklearn menolak 0.0)
        return (f'OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency={share}, '
                'sparse_output=False)')
    if method == "ordinal":
        return 'OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)'
    if method == "ordered":
        return (f'OrdinalEncoder(categories=[ORDER[{column!r}]], handle_unknown="use_encoded_value", '
                'unknown_value=-1, encoded_missing_value=-1)')
    extra = ", random_state=0" if method == "gap" else ""
    return f'skrub.{SKRUB_ENCODERS[method]}(n_components={spec.get("n_components", 10)}{extra})'


def transformer_entries(recipe: dict) -> list[tuple[str, str, str]]:
    encoded = {column: spec for column, spec in recipe["encoding"].items() if spec["method"] != "materials"}
    entries, groups = [], {}
    for column, spec in recipe["impute"].items():
        if column in encoded or spec["strategy"] == "none":
            continue
        if spec["strategy"] == "constant":
            entries.append((f"isi_{slug(column)}", imputer_expression(spec), literal([column])))
        else:
            groups.setdefault(spec["strategy"], []).append(column)
    for strategy, columns in groups.items():
        entries.insert(0, (f"isi_{strategy}", imputer_expression({"strategy": strategy}), literal(columns)))
    for column, spec in encoded.items():
        encoder = encoder_expression(column, spec)
        impute = recipe["impute"].get(column, {"strategy": "none"})
        if spec["method"] in SKRUB_ENCODERS:          # encoder skrub = satu kolom (Series), terima NaN
            entries.append((f"{spec['method']}_{slug(column)}", encoder, repr(column)))
        elif impute["strategy"] == "none":
            entries.append((f"{spec['method']}_{slug(column)}", encoder, literal([column])))
        else:
            pipeline = f"make_pipeline({imputer_expression(impute)}, {encoder})"
            entries.append((f"{spec['method']}_{slug(column)}", pipeline, literal([column])))
    return entries


def preprocessor_function(recipe: dict) -> str:
    entries = transformer_entries(recipe)
    rows = "\n".join(f'            ("{name}", {transformer}, {columns}),' for name, transformer, columns in entries)
    listing = f"[\n{rows}\n        ]" if entries else "[]"
    return (
        "def build_preprocessor() -> ColumnTransformer:\n"
        '    """Isi kosong + encoding. Median, modus, dan daftar kategori dipelajari saat fit (train)."""\n'
        "    return ColumnTransformer(\n"
        f"        {listing},\n"
        '        remainder="passthrough",\n'
        "        verbose_feature_names_out=False,\n"
        '    ).set_output(transform="pandas")'
    )


def split_function(recipe: dict) -> str:
    method = recipe["split"]["method"]
    if method == "none":
        return ""
    if method in ("random", "stratified"):
        stratify = "train_clean[LABEL]" if method == "stratified" else "None"
        return (
            "def split(train_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:\n"
            '    """Pisahkan data validasi dari train bersih."""\n'
            "    return train_test_split(\n"
            '        train_clean, test_size=SPLIT["valid_size"], random_state=SPLIT["seed"],\n'
            f"        stratify={stratify},\n"
            "    )"
        )
    return (
        "def split(train_clean: pd.DataFrame) -> pd.Series:\n"
        '    """Nomor fold per baris (disimpan di file terpisah, bukan kolom baru)."""\n'
        "    classify = LABEL is not None and train_clean[LABEL].nunique() <= 20\n"
        "    splitter = (StratifiedKFold if classify else KFold)(\n"
        '        n_splits=SPLIT["folds"], shuffle=True, random_state=SPLIT["seed"])\n'
        '    folds = pd.Series(-1, index=train_clean.index, name="fold")\n'
        "    target = train_clean[LABEL] if classify else None\n"
        "    for fold, (_, rows) in enumerate(splitter.split(train_clean, target)):\n"
        "        folds.iloc[rows] = fold\n"
        "    return folds"
    )


def main_function(recipe: dict) -> str:
    lines = [
        "def main() -> None:",
        "    train = pd.read_csv(TRAIN_CSV)",
        "    test = pd.read_csv(TEST_CSV) if TEST_CSV else None",
        "    result = run(train, test)",
        "    output = Path(OUTPUT_DIR)",
        "    output.mkdir(parents=True, exist_ok=True)",
        '    result["train"].to_csv(output / "train_bersih.csv", index=False)',
        '    if result["test"] is not None:',
        '        result["test"].to_csv(output / "test_bersih.csv", index=False)',
        '    joblib.dump(result["state"], output / "pipeline_eda.joblib")',
    ]
    method = recipe["split"]["method"]
    if method in ("random", "stratified"):
        lines += ['    train_part, valid_part = split(result["train"])',
                  '    train_part.to_csv(output / "train_part.csv", index=False)',
                  '    valid_part.to_csv(output / "valid_part.csv", index=False)']
    elif method == "kfold":
        lines.append('    split(result["train"]).to_csv(output / "folds.csv", index=False)')
    lines += [
        '    dropped = result["drop_reasons"].any(axis=1).sum()',
        '    print(f"train: {len(train)} -> {len(result[\'train\'])} baris ({dropped} dibuang), "',
        '          f"{result[\'train\'].shape[1]} kolom")',
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
    ]
    return "\n".join(lines)


def header(recipe: dict) -> str:
    return (
        '"""Pipeline EDA dari Laya EDA.\n\n'
        "Statistik (median, batas outlier, daftar kategori) dipelajari dari train saja, lalu\n"
        "dipakai sama untuk test dan data baru. Jalankan: python pipeline_eda.py\n\n"
        "Inferensi nanti:\n"
        "    from pipeline_eda import transform\n"
        '    state = joblib.load("hasil/pipeline_eda.joblib")\n'
        "    data_baru_bersih = transform(data_baru, state)\n"
        '"""'
    )


def build_script(recipe: dict, paths: dict) -> str:
    sections = [
        header(recipe),
        imports(recipe),
        "# %% Keputusan dari Laya EDA\n" + constants(recipe, paths),
    ]
    helpers = []
    if recipe["number_columns"]:
        helpers.append(TO_NUMBER.strip())
    if recipe["text_columns"]:
        helpers.append(TIDY_TEXT.strip())
    if recipe["free_text"]:
        helpers.append(CANONICAL_MATERIALS.strip())
    helpers.append(prepare_function(recipe))
    sections.append("# %% Langkah tanpa statistik (tipe, penanda kosong, ejaan)\n" + "\n\n\n".join(helpers))
    outliers = OUTLIER_BOUNDS if recipe["outliers"] else NO_OUTLIERS
    sections.append("# %% Outlier dan baris yang dibuang (train)\n" + outliers.strip() + "\n\n\n" + drop_function(recipe))
    materials = MATERIALS if uses(recipe, "materials") else NO_MATERIALS
    sections.append("# %% Isi kosong dan encoding\n" + materials.strip() + "\n\n\n" + preprocessor_function(recipe))
    sections.append("# %% Jalur train dan test\n" + FINISH.strip() + "\n\n\n" + RUN.strip())
    split = split_function(recipe)
    sections.append("# %% Split dan simpan\n" + (split + "\n\n\n" if split else "") + main_function(recipe))
    return "\n\n\n".join(sections) + "\n"


def to_notebook(script: str) -> str:
    """Script → .ipynb: setiap blok `# %%` jadi satu sel kode, docstring pembuka jadi markdown."""
    blocks = re.split(r"\n(?=# %%)", script)
    cells = []
    intro = blocks[0].split('"""')
    if len(intro) >= 3:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": intro[1].strip()})
        first_code = intro[2].strip()
        if first_code:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": first_code})
    for block in blocks[1:]:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": block.strip()})
    notebook = {"cells": cells, "metadata": {"language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(notebook, ensure_ascii=False, indent=1)
