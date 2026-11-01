"""Fitur extra E1–E2 (VISI.md bagian 0): langkah → Pipeline sklearn (preprocess → model), simpan/muat joblib.

Pembagian langkah aktif:
- **blok EDA** (pandas, dijalankan di train dan test sebelum pipeline): langkah tanpa transformer sklearn
  (ejaan, isian bebas, penanda kosong, buang baris, ...) dan setiap langkah yang datang SEBELUM langkah pandas;
- **pipeline**: akhiran terpanjang daftar langkah yang semuanya punya transformer (`ops.transformer`).
Urutan langkah tidak pernah diubah, jadi hasil = hasil blok kode EDA (diuji di tests/test_extra.py).

Pipeline dibuat dengan MENJALANKAN kode teks yang juga ditulis ke template (satu sumber, seperti steps.py).
Kode teks itu lurus: tanpa def/lambda/class (R4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import ops
from core.steps import active

TEXT_ENCODER = ('ColumnTransformer([("kode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, '
                'encoded_missing_value=-1), make_column_selector(dtype_exclude="number"))], '
                'remainder="passthrough", verbose_feature_names_out=False)')
BASE_IMPORTS = ["import numpy as np", "import pandas as pd",
                "from sklearn.compose import ColumnTransformer, make_column_selector",
                "from sklearn.pipeline import Pipeline",
                "from sklearn.preprocessing import OrdinalEncoder"]


def plan(steps: list[dict]) -> dict:
    """{'eda': langkah blok EDA, 'pipe': [(nomor langkah, langkah, spesifikasi transformer)]}."""
    chosen = active(steps)
    numbered = list(enumerate(chosen, 1))
    cut = len(numbered)
    while cut > 0 and ops.transformer(numbered[cut - 1][1]) is not None:
        cut -= 1
    return {"eda": [step for _, step in numbered[:cut]],
            "pipe": [(number, step, ops.transformer(step)) for number, step in numbered[cut:]]}


def preprocess_code(pipe_steps: list[tuple]) -> str:
    """Kode lurus yang membuat list `tahap` (satu ColumnTransformer per langkah + kode teks sisa)."""
    imports = list(BASE_IMPORTS)
    for _, _, spec in pipe_steps:
        imports += [line for line in spec["imports"] if line not in imports]
    lines = [*imports, "tahap = []"]
    for number, step, spec in pipe_steps:
        lines.append(f"# langkah {number}: {ops.summary(step)}")
        lines.append(f'tahap.append(("langkah_{number}", ColumnTransformer([("ubah", {spec["make"]}, '
                     f'{ops.lit(spec["columns"])})], remainder="passthrough", verbose_feature_names_out=False)'
                     '.set_output(transform="pandas")))')
    lines.append("# kolom teks yang tersisa -> kode angka (nilai baru/kosong = -1) agar semua model bisa membaca")
    lines.append(f'tahap.append(("teks_sisa", {TEXT_ENCODER}))')
    return "\n".join(lines)


def pipeline_code(steps: list[dict], model_code: str, model_imports: list[str], sampler: str | None = None) -> str:
    """Kode lurus: tahap preprocessing + (opsional) oversampling di dalam fold + model → `pipe`."""
    lines = [preprocess_code(plan(steps)["pipe"]), *model_imports]
    if sampler:
        lines += ["# oversampling hanya saat fit (fold latih); predict / fold validasi / test tidak disentuh",
                  "from imblearn.pipeline import Pipeline as PipelineImblearn",
                  "from sklearn.impute import SimpleImputer",
                  'tahap.append(("isi_kosong_smote", SimpleImputer(strategy="median", keep_empty_features=True)))',
                  f'tahap.append(("oversampling", {sampler}))',
                  f'tahap.append(("model", {model_code}))',
                  "pipe = PipelineImblearn(tahap)"]
    else:
        lines += [f'tahap.append(("model", {model_code}))', "pipe = Pipeline(tahap)"]
    return "\n".join(lines)


def build(steps: list[dict], model_code: str, model_imports: list[str], sampler: str | None = None):
    """Jalankan pipeline_code apa adanya → objek Pipeline (belum di-fit)."""
    namespace: dict = {}
    exec(pipeline_code(steps, model_code, model_imports, sampler), namespace)
    return namespace["pipe"]


def features(data: pd.DataFrame, label: str | None, exclude: list[str]) -> list[str]:
    """Kolom fitur = semua kolom kecuali label dan kolom yang dikecualikan (id)."""
    return [c for c in data.columns if c != label and c not in exclude]


def save(pipe, path: str | Path, info: dict) -> dict:
    """joblib.dump + `<path>.json` berisi versi library (dicetak saat muat) dan info prediksi (kelas, ambang)."""
    import joblib
    import sklearn

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    meta = {**info, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    Path(f"{path}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return meta


def load(path: str | Path) -> tuple:
    """(pipe, info). Versi sklearn saat simpan vs sekarang ada di info['sklearn'] / info['sklearn_sekarang']."""
    import joblib
    import sklearn

    info = json.loads(Path(f"{path}.json").read_text(encoding="utf-8"))
    info["sklearn_sekarang"] = sklearn.__version__
    return joblib.load(path), info


__all__ = ["plan", "preprocess_code", "pipeline_code", "build", "features", "save", "load"]
