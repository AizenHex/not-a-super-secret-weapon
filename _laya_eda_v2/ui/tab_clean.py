"""Tab Operasi (panel operasi + ubah langkah), Ejaan (clustering), Isian bebas, Outlier."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from core import outlier_tools, text_tools
from core import steps as engine
from core.ops import MISSING_MARKERS, OPS, SENTINEL_FLOOR, summary
from ui import common, jobs

METHODS = {"iqr": "IQR (Tukey)", "iqr_log": "IQR skala log", "mad": "MAD", "mad_log": "MAD skala log",
           "manual": "batas manual (dari pengetahuanmu)"}
ACTIONS = {"clip": "clip: potong ke batas (train + test)", "nan": "nan: jadikan kosong (train + test)",
           "drop": "drop: buang baris (train saja)"}


# ---------------------------------------------------------------- panel operasi

def is_discrete(values: pd.Series) -> bool:
    """Bilangan bulat dengan sedikit nilai berbeda (jumlah tanggungan, jumlah kamar)."""
    values = values.dropna()
    return len(values) > 0 and bool((values == values.round()).all()) and values.nunique() <= 30


def bound_widgets(values: pd.Series, key: str, old: dict) -> dict:
    """Metode + k, atau batas manual. Kembalikan bagian params: method, factor, low, high."""
    left, middle, right = st.columns(3)
    names = list(METHODS)
    method = left.selectbox("Metode", names, names.index(old.get("method", "iqr_log")), format_func=METHODS.get,
                            key=f"{key}_m")
    if method == "manual":
        st.caption("Isi batas yang MUSTAHIL menurut pengetahuanmu (mis. tanggungan > 30 orang). Kosongkan sisi yang "
                   "tidak dibatasi. Nilai jarang tapi mungkin (keluarga besar 12 orang) tetap aman.")
        low = middle.number_input("Batas bawah (kosong = tanpa)", value=old.get("low"), key=f"{key}_low",
                                  placeholder=f"min data {values.min():,.0f}")
        high = right.number_input("Batas atas (kosong = tanpa)", value=old.get("high"), key=f"{key}_high",
                                  placeholder=f"maks data {values.max():,.0f}")
        return {"method": method, "factor": 0, "low": low, "high": high}
    default = float(old.get("factor") or outlier_tools.FACTORS[method])
    factor = middle.number_input("k", 0.5, 10.0, default, 0.5, key=f"{key}_k_{method}")
    return {"method": method, "factor": factor, "low": None, "high": None}


def form(op: str, data: pd.DataFrame, old: dict, key: str) -> dict | None:
    """Widget parameter untuk satu operasi. Kembalikan params, atau None bila belum lengkap."""
    columns, texts, numbers = list(data.columns), common.object_columns(data), common.number_columns(data)

    def many(title, options, default=()):
        return st.multiselect(title, options, [c for c in old.get("columns", default) if c in options], key=f"{key}_cols")

    def one(title, options):
        if not options:
            st.info("Tidak ada kolom yang cocok.")
            return None
        index = options.index(old["column"]) if old.get("column") in options else 0
        return st.selectbox(title, options, index, key=f"{key}_col")

    if op in ("drop_columns",):
        picked = many("Kolom", columns)
        return {"columns": picked} if picked else None
    if op == "to_number":
        picked = many("Kolom teks berisi angka", texts)
        indonesian = st.checkbox("Format Indonesia (titik = ribuan, koma = desimal)", old.get("indonesian", True),
                                 key=f"{key}_id")
        return {"columns": picked, "indonesian": indonesian} if picked else None
    if op == "tidy_text":
        picked = many("Kolom teks", texts, texts)
        markers = st.text_input("Penanda kosong (pisah |)", " | ".join(old.get("markers", MISSING_MARKERS)),
                                key=f"{key}_mk")
        marks = sorted({m.strip().lower() for m in markers.split("|")} | {""})
        return {"columns": picked, "markers": marks} if picked else None
    if op == "change_case":
        picked = many("Kolom teks", texts)
        cases = ["lower", "upper", "title"]
        case = st.radio("Jadi", cases, cases.index(old.get("case", "lower")), horizontal=True, key=f"{key}_case",
                        format_func={"lower": "huruf kecil", "upper": "HURUF BESAR", "title": "Huruf Judul"}.get)
        return {"columns": picked, "case": case} if picked else None
    if op == "spelling":
        groups = old.get("groups", {})
        st.caption("Usulan GABUNG (SMA = MA = SLTA, Tidak = Tidak Ada, …) dibuat di tab **Ejaan**. Di sini: pilih kolom "
                   "untuk aturan kunci; gabungan yang sudah ada di langkah ini dipertahankan.")
        picked = st.multiselect("Kolom kategori", texts, [c for c in groups if c in texts], key=f"{key}_cols")
        return ({"groups": {c: groups.get(c, {}) for c in picked}, "examples": old.get("examples", {})}
                if picked else None)
    if op == "map_values":
        column = one("Kolom", texts)
        if column is None:
            return None
        st.caption("Satu baris per pasangan: `nilai asli => nilai baku`")
        text = "\n".join(f"{a} => {b}" for a, b in old.get("mapping", {}).items())
        text = st.text_area("Peta", text, height=150, key=f"{key}_map")
        mapping = {}
        for line in text.splitlines():
            if "=>" in line:
                left, right = line.split("=>", 1)
                mapping[left.strip()] = right.strip()
        return {"column": column, "mapping": mapping} if mapping else None
    if op == "free_text":
        column = one("Kolom isian bebas", texts)
        st.caption("Peta kata disarankan dari tab **Isian bebas**. Di sini: `kata salah => kata baku` per baris.")
        text = st.text_area("Peta kata", "\n".join(f"{a} => {b}" for a, b in old.get("words", {}).items()),
                            key=f"{key}_words")
        words = {a.strip(): b.strip() for a, b in (l.split("=>", 1) for l in text.splitlines() if "=>" in l)}
        return {"column": column, "words": words} if column else None
    if op in ("sentinel_to_nan", "zero_to_nan", "log_transform"):
        picked = many("Kolom angka", numbers)
        params = {"columns": picked}
        if op == "sentinel_to_nan":
            params["floor"] = int(st.number_input("Batas bawah nilai penjaga", value=int(old.get("floor", SENTINEL_FLOOR)),
                                                  step=1_000_000, key=f"{key}_floor"))
        return params if picked else None
    if op == "fill_constant":
        picked = many("Kolom", columns)
        value = st.text_input("Nilai pengisi", str(old.get("value", "Tidak")), key=f"{key}_val")
        if picked and all(c in numbers for c in picked):
            value = pd.to_numeric(value, errors="coerce")
            value = None if pd.isna(value) else float(value)
        return {"columns": picked, "value": value} if picked and value is not None else None
    if op == "impute":
        strategy = st.radio("Strategi", ["median", "mean", "mode"], ["median", "mean", "mode"].index(old.get("strategy", "median")),
                            horizontal=True, key=f"{key}_st")
        options = [c for c in (numbers if strategy != "mode" else columns) if data[c].isna().any() and data[c].notna().any()]
        picked = many("Kolom yang masih kosong", options)
        return {"columns": picked, "strategy": strategy} if picked else None
    if op == "drop_duplicates":
        return {}
    if op == "drop_missing_rows":
        limit = st.slider("Kosong per baris maksimal", 1, data.shape[1], int(old.get("max_missing", 10)), key=f"{key}_max")
        return {"max_missing": limit}
    if op == "drop_rows_query":
        st.caption("Nama kolom berspasi diapit backtick: `` `Luas Bangunan Rumah (m2)` > 100000 ``")
        rule = st.text_input("Baris yang COCOK dibuang", old.get("rule", ""), key=f"{key}_rule")
        return {"rule": rule} if rule else None
    if op == "outlier":
        column = one("Kolom angka", numbers)
        if column is None:
            return None
        bounds = bound_widgets(data[column], key, old)
        action = st.selectbox("Aksi", list(ACTIONS), list(ACTIONS).index(old.get("action", "clip")),
                              format_func=ACTIONS.get, key=f"{key}_a")
        return {"column": column, **bounds, "action": action}
    if op == "scale":
        picked = many("Kolom angka", numbers)
        method = st.radio("Cara", ["standard", "minmax"], ["standard", "minmax"].index(old.get("method", "standard")),
                          horizontal=True, key=f"{key}_sc")
        return {"columns": picked, "method": method} if picked else None
    if op in ("onehot", "materials_onehot"):
        column = one("Kolom", texts)
        share = st.slider("Porsi minimal (lebih jarang → _lainnya)", 0.0, 0.1, float(old.get("min_share", 0.01)), 0.005,
                          format="%.3f", key=f"{key}_share")
        if column:
            st.caption(f"{data[column].nunique()} nilai unik")
        return {"column": column, "min_share": share} if column else None
    if op == "ordinal":
        column = one("Kolom", texts)
        if column is None:
            return None
        values = sorted(data[column].dropna().unique(), key=str)
        guess = text_tools.ordinal_order(data[column]) or {}
        if guess and not old.get("order"):
            st.caption(f"Usulan urutan: {guess['alasan']}")
        order = old.get("order") or guess.get("urutan") or []
        text = st.text_area("Urutan dari terendah (kosongkan = abjad)", "\n".join(map(str, order)),
                            height=150, key=f"{key}_order_{column}",
                            help="Nilai yang ada: " + ", ".join(map(str, values[:40])))
        order = [line.strip() for line in text.splitlines() if line.strip()]
        missing = set(values) - set(order)
        if order and missing:
            st.warning(f"Tidak ada di urutan (jadi −1): {sorted(missing, key=str)[:10]}")
        return {"column": column, "order": order}
    if op == "skrub_encoder":
        column = one("Kolom teks", texts)
        kinds = ["string", "minhash", "gap"]
        kind = st.radio("Encoder", kinds, kinds.index(old.get("kind", "string")), horizontal=True, key=f"{key}_kind")
        n = st.slider("Jumlah kolom hasil", 2, 30, int(old.get("n_components", 10)), key=f"{key}_n")
        return {"column": column, "kind": kind, "n_components": n} if column else None
    if op in ("target_encode", "onehot_sklearn", "rare_label"):
        picked = many("Kolom kategori", texts)
        params = {"columns": picked}
        if op == "target_encode":
            kinds = ["kelas", "angka"]
            params["kind"] = st.radio("Label dibaca sebagai", kinds, kinds.index(old.get("kind", "kelas")),
                                      horizontal=True, key=f"{key}_kind",
                                      help="kelas: satu kolom per kelas (rata-rata 0/1); angka: satu kolom rata-rata label.")
            if not common.label():
                st.error("Butuh kolom label (sidebar Data).")
                return None
        if op == "onehot_sklearn":
            params["min_share"] = st.slider("min_frequency (porsi)", 0.0, 0.1, float(old.get("min_share", 0.01)), 0.005,
                                            format="%.3f", key=f"{key}_share")
        if op == "rare_label":
            params["tol"] = st.slider("Porsi minimal (tol)", 0.001, 0.1, float(old.get("tol", 0.01)), 0.001,
                                      format="%.3f", key=f"{key}_tol")
        return params if picked else None
    if op in ("winsorize", "power"):
        picked = many("Kolom angka", numbers)
        params = {"columns": picked}
        if op == "winsorize":
            methods = ["iqr", "mad", "gaussian", "quantiles"]
            params["method"] = st.radio("Metode batas", methods, methods.index(old.get("method", "iqr")),
                                        horizontal=True, key=f"{key}_wm")
            params["fold"] = st.number_input("fold (k; quantiles: porsi ekor, mis. 0.05)", 0.0, 10.0,
                                             float(old.get("fold", 1.5)), 0.05, key=f"{key}_wf")
        return params if picked else None
    if op == "autogluon_features":
        if not common.label():
            st.error("Butuh kolom label (sidebar Data).")
            return None
        return {}
    return None


def show_operations() -> None:
    common.header("operasi",
                  "Panel operasi ala **Data Wrangler**: pilih operasi, isi parameter, lihat **sebelum → sesudah**, "
                  "lalu tambahkan sebagai langkah. Setiap langkah = satu blok kode pandas lurus (tanpa fungsi).",
                  "Semua perbaikan yang tidak diusulkan otomatis di tab Temuan.",
                  "Langkah yang membuang baris/kolom. Baca 'kapan berbahaya' setiap operasi.")
    if not common.need_data():
        return
    data = common.current()
    groups = sorted({spec["group"] for spec in OPS.values()}, key=["Kolom", "Teks", "Kosong", "Baris", "Angka", "Encoding"].index)
    group = st.radio("Kelompok", groups, horizontal=True, key="op_group")
    names = [name for name, spec in OPS.items() if spec["group"] == group]
    op = st.selectbox("Operasi", names, format_func=lambda n: OPS[n]["title"], key="op_name")
    common.op_help(op)
    params = form(op, data, {}, f"new_{op}")
    if params is not None:
        common.preview_step(op, params, f"new_{op}")

    steps = common.project()["steps"]
    if steps:
        st.divider()
        st.subheader("Ubah langkah yang sudah ada")
        index = st.selectbox("Langkah", range(len(steps)), format_func=lambda i: f"{i + 1}. {summary(steps[i])}",
                             key="edit_index")
        step = steps[index]
        before = common.state(index)["train"]
        before = before.drop(columns=[common.label()]) if common.label() in before else before
        st.caption("Pratinjau dihitung di data sesudah langkah sebelumnya.")
        params = form(step["op"], before, step["params"], f"edit_{index}_{step['op']}")
        if params is not None:
            common.preview_step(step["op"], params, f"edit_{index}", replace=index)
        with st.expander("Parameter mentah (JSON)"):
            st.code(json.dumps(step["params"], ensure_ascii=False, indent=2), language="json")


# ---------------------------------------------------------------- ejaan

SPELL_METHODS = ["aturan kunci + kamus (aman)", "+ salah ketik rapidfuzz (periksa)",
                 "+ skrub.deduplicate (opsional, periksa)"]


def merge_editor(proposals: pd.DataFrame, key: str) -> pd.DataFrame:
    """Tabel usulan GABUNG: hanya kolom 'setuju' yang bisa diubah."""
    first = [c for c in ["setuju", "kolom", "ejaan", "jadi", "jumlah", "alasan"] if c in proposals]   # 'MA' → 'SLTA' terbaca dulu
    table = proposals[first + [c for c in proposals.columns if c not in first]]
    return st.data_editor(table, width="stretch", hide_index=True, key=key,
                          disabled=[c for c in table.columns if c != "setuju"],
                          column_config={"alasan": st.column_config.TextColumn("alasan", width="large")})


def show_spelling() -> None:
    common.header("ejaan",
                  "Satukan ejaan berbeda dari nilai yang sama di kolom **kategori**. Hasilnya satu langkah "
                  "**Satukan ejaan**: satu aturan kunci (huruf kecil, tanpa spasi/tanda baca, `Kab.` = `Kabupaten`) "
                  "+ daftar pendek **GABUNG** untuk kunci berbeda yang bermakna sama. Kode tetap pendek berapa pun "
                  "banyaknya variasi, dan ejaan baru di test ikut dibakukan.",
                  "**Aturan kunci** (key collision OpenRefine) selalu aman: `S- 2` = `S-2` = `S2`. **Kamus** jenjang "
                  "pendidikan (SMA = MA = SLTA) dan jawaban 'tidak' (Tidak = Tidak Ada) dicentang dengan alasannya.",
                  "**Salah ketik rapidfuzz** dan **skrub.deduplicate** bisa menggabung nilai berbeda (`Kota Malang` → "
                  "`Kota Magelang`): tidak dicentang otomatis. Kolom isian bebas (Bahan …) dibakukan per kata di tab "
                  "**Isian bebas**, bukan di sini.")
    if not common.need_data():
        return
    data = common.current()
    texts = common.object_columns(data)
    if not texts:
        st.info("Tidak ada kolom teks.")
        return
    mode = st.radio("Cara", ["semua kolom sekaligus (tabel)", "satu kolom (rinci)"], horizontal=True, key="spell_mode")
    if mode.startswith("semua"):
        show_spelling_all(data, texts)
    else:
        show_spelling_column(data, texts)


@st.cache_data(max_entries=8, show_spinner="Mencari ejaan berbeda di kolom kategori…")
def spelling_overview(key: str, _data: pd.DataFrame, columns: list[str], typos: bool, cutoff: int,
                      min_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Ringkasan per kolom + usulan GABUNG semua kolom dalam satu tabel. Kolom isian bebas dilewati."""
    summary, parts, free = [], [], []
    for column in columns:
        series = _data[column]
        if text_tools.is_free_text(series):
            free.append(column)
            continue
        proposals = text_tools.merge_proposals(series, typos, cutoff, min_ratio)
        after_keys = text_tools.key_table(series)["kunci"].nunique()
        summary.append({"kolom": column, "nilai unik": int(series.nunique()), "sesudah aturan kunci": int(after_keys),
                        "usulan gabung": len(proposals)})
        parts.append(proposals.assign(kolom=column))
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["kolom", "setuju"])
    table = table[["kolom", *[c for c in table.columns if c != "kolom"]]]
    return pd.DataFrame(summary, columns=["kolom", "nilai unik", "sesudah aturan kunci", "usulan gabung"]), table, free


def spelling_params(columns: list[str], proposals: pd.DataFrame) -> dict:
    """Kolom terpilih + baris usulan yang dicentang → params langkah 'spelling'."""
    groups, examples = {}, {}
    for column in columns:
        part = proposals[proposals["kolom"] == column] if "kolom" in proposals else proposals
        groups[column], examples[column] = text_tools.plan_groups(part)
    return {"groups": groups, "examples": examples}


def show_spelling_all(data: pd.DataFrame, texts: list[str]) -> None:
    """Semua kolom kategori sekaligus: ringkasan per kolom, usulan gabung, lalu SATU langkah untuk semuanya."""
    method = st.radio("Metode", SPELL_METHODS[:2], horizontal=True, key="spell_all_method")
    typos, cutoff, ratio = method != SPELL_METHODS[0], 88, 3.0
    if typos:
        left, right = st.columns(2)
        ratio = left.slider("Kunci tujuan minimal … kali lebih sering", 1.5, 50.0, 3.0, 0.5, key="nn_all_ratio")
        cutoff = right.slider("Skor minimal (token_sort_ratio)", 70, 100, 88, key="nn_all_cut")
    summary, proposals, free = spelling_overview(common.state_key(), data, texts, typos, cutoff, ratio)
    if free:
        st.caption("Dilewati, isian bebas (dibakukan per kata di tab **Isian bebas**): "
                   + ", ".join(f"{c} ({data[c].nunique():,} nilai)" for c in free))
    summary.insert(0, "pakai", (summary["sesudah aturan kunci"] < summary["nilai unik"]) | (summary["usulan gabung"] > 0))
    if not summary["pakai"].any():
        st.success("Tidak ada ejaan berbeda di kolom kategori mana pun: tidak perlu langkah.")
        return
    st.subheader("1. Kolom")
    st.caption("'sesudah aturan kunci' = nilai unik bila beda huruf besar/spasi/tanda baca/singkatan dianggap sama. "
               "Kolom tanpa centang tidak disentuh.")
    summary = st.data_editor(summary, width="stretch", hide_index=True, key="spell_all_columns",
                             disabled=[c for c in summary.columns if c != "pakai"])
    columns = list(summary.loc[summary["pakai"], "kolom"])
    st.subheader("2. Gabung kunci berbeda yang bermakna sama")
    shown = proposals[proposals["kolom"].isin(columns)]
    if shown.empty:
        st.caption("Tidak ada usulan gabung: langkah hanya memakai aturan kunci.")
        chosen = shown
    else:
        st.caption("Baca alasannya. Hapus centang bila dua nilai itu memang berbeda di datamu.")
        chosen = merge_editor(shown, f"spell_all_merge_{method[:3]}")
    if not columns:
        st.info("Belum ada kolom dicentang: tidak ada langkah, data tidak berubah.")
        return
    st.subheader("3. Pratinjau langkah")
    common.preview_step("spelling", spelling_params(columns, chosen), "spell_all")


def show_spelling_column(data: pd.DataFrame, texts: list[str]) -> None:
    column = st.selectbox("Kolom", texts, key="spell_column")
    series = data[column]
    if text_tools.is_free_text(series):
        st.warning(f"'{column}' terlihat seperti isian bebas (beberapa bahan per sel). Pembakuan per kata di tab "
                   "**Isian bebas** biasanya lebih tepat; di sini setiap kombinasi dianggap nilai sendiri.")
    same = text_tools.same_key_table(series)
    after = text_tools.key_table(series)["kunci"].nunique()
    st.caption(f"{series.nunique():,} nilai unik → {after:,} sesudah aturan kunci ({len(same):,} ejaan disatukan "
               "tanpa daftar).")
    if len(same):
        with st.expander(f"Ejaan yang disatukan aturan kunci ({len(same):,})"):
            st.dataframe(same.head(300), width="stretch", hide_index=True)
    method = st.radio("Metode", SPELL_METHODS, horizontal=True, key="spell_method")
    typos, cutoff, ratio, extra = method == SPELL_METHODS[1], 88, 3.0, None
    if typos:
        left, right = st.columns(2)
        ratio = left.slider("Kunci tujuan minimal … kali lebih sering", 1.5, 50.0, 3.0, 0.5, key="nn_ratio")
        cutoff = right.slider("Skor minimal (token_sort_ratio)", 70, 100, 88, key="nn_cut")
    elif method == SPELL_METHODS[2]:
        clusters = st.number_input("Jumlah klaster (0 = otomatis)", 0, max(series.nunique(), 1), 0)
        st.caption("Data uji: 640 nilai ≈ 3 detik, 1.961 nilai ≈ 70 detik.")
        name = f"dedup_{column}"
        if st.button("Jalankan skrub.deduplicate", disabled=jobs.running(name)):
            jobs.start(name, f"skrub.deduplicate {column}", text_tools.deduplicate_map, series.copy(),
                       int(clusters) or None)
            st.rerun()
        jobs.status(name)
        found = jobs.result(name)
        if found is not None:
            extra = dict(zip(found["asli"], found["usulan baku"]))
    proposals = text_tools.merge_proposals(series, typos, cutoff, ratio, extra)
    if proposals.empty:
        st.caption("Tidak ada usulan gabung.")
        chosen = proposals
    else:
        st.subheader(f"Gabung kunci berbeda ({len(proposals)})")
        chosen = merge_editor(proposals, f"spell_{column}_{method[:3]}")
    if not len(same) and not chosen["setuju"].astype(bool).any():
        st.info("Tidak ada yang perlu disatukan di kolom ini.")
        return
    common.preview_step("spelling", spelling_params([column], chosen), f"spell_{column}")


# ---------------------------------------------------------------- isian bebas

@st.cache_data(max_entries=16, show_spinner="Menghitung hasil peta kata…")
def free_text_effect(key: str, params_json: str, _before: pd.Series) -> dict:
    """Langkah isian bebas selalu menulis ulang sel (huruf kecil, urut, pemisah ', '), jadi hampir semua sel
    'berubah'. Pisahkan yang hanya berubah format dari yang ISINYA dibetulkan (kata, kata pengisi, frasa).

    ±4 detik di 30 ribu baris, jadi di-cache per (data, params): klik lain tidak menghitung ulang.
    """
    params = json.loads(params_json)
    column, frame = params["column"], _before.to_frame()
    plain = engine.run([{"op": "free_text", "params": {"column": column, "words": {}}, "on": True}],
                       None, frame, None)["train"][column]
    fixed = engine.run([{"op": "free_text", "params": params, "on": True}], None, frame, None)["train"][column]
    words_fixed = (plain != fixed) & plain.notna()
    return {"hanya format": int(((_before != plain) & _before.notna()).sum()),
            "kata dibetulkan": int(words_fixed.sum()),
            "contoh": pd.DataFrame({"asli": _before[words_fixed], "sesudah": fixed[words_fixed]}).drop_duplicates().head(30),
            "bahan tanpa peta": text_tools.material_preview(_before, {}),
            "bahan dengan peta": text_tools.material_preview(_before, params["words"], 20, params["fillers"],
                                                             params["phrases"])}


def show_free_text() -> None:
    common.header("isian",
                  "Kolom isian bebas berisi **beberapa bahan** dalam satu sel dengan salah ketik "
                  "(`'Genteng & asbas.'`). Sel dipecah per bahan, salah ketik dibetulkan **per kata**, lalu "
                  "ditulis ulang sebagai daftar baku `'asbes, genteng'` (kolom tetap satu, nama tetap).",
                  "Kolom bahan atap/tembok/lantai, daftar fasilitas.",
                  "Kata sah yang mirip kata lain (`tanpa`/`tanah`, `batu`/`bata`). Usulan skor < 85 tidak dicentang "
                  "otomatis; kosakata dihitung juga dari kolom terkait.")
    if not common.need_data():
        return
    data = common.current()
    texts = common.object_columns(data)
    if not texts:
        st.info("Tidak ada kolom teks.")
        return
    column = st.selectbox("Kolom isian bebas", texts,
                          index=next((i for i, c in enumerate(texts) if c.startswith("Bahan")), 0), key="free_column")
    related = st.multiselect("Kolom terkait (kosakata digabung)", [c for c in texts if c != column],
                             [c for c in texts if c != column and c.startswith("Bahan")], key="free_related")
    left, right = st.columns(2)
    min_count = left.slider("Kata baku = muncul minimal … kali", 5, 200, 30, key="free_min")
    cutoff = right.slider("Skor kemiripan minimal (fuzz.ratio)", 60, 95, 75, key="free_cut")

    words = text_tools.word_map(data[column], min_count, cutoff, [data[c] for c in related])
    st.subheader(f"Usulan salah ketik ({len(words)})")
    edited = st.data_editor(words, width="stretch", hide_index=True, key=f"words_{column}",
                            disabled=["kata asli", "skor", "jumlah", "catatan"])
    st.subheader("Kata baku yang mirip (kandidat sinonim)")
    st.caption("Tidak pernah otomatis. Centang bila 'kata 2' sama maknanya dengan 'kata 1' (genteng = genting).")
    similar = text_tools.similar_vocabulary(data[column], min_count)
    similar.insert(0, "samakan 2 → 1", False)
    laya = st.session_state.get("laya_same", {})
    if laya:
        similar["Laya: sama?"] = [laya.get((a, b)) for a, b in zip(similar["kata 1"], similar["kata 2"])]
    synonyms = st.data_editor(similar, width="stretch", hide_index=True, key=f"syn_{column}",
                              disabled=[c for c in similar.columns if c != "samakan 2 → 1"])
    mapping = dict(zip(edited.loc[edited["setuju"], "kata asli"], edited.loc[edited["setuju"], "usulan"]))
    for first, second in synonyms.loc[synonyms["samakan 2 → 1"], ["kata 1", "kata 2"]].to_numpy():
        mapping[second] = first
    mapping = text_tools.resolve_chains(mapping)          # dikramik → kramik → keramik = dikramik → keramik

    st.subheader("Kata pengisi dan frasa")
    st.caption("Peta kata tidak bisa menyatukan **frasa**: 'lantai keramik' = 'keramik', 'bata merah' = 'batu bata'. "
               "Kata pengisi dibuang dari tiap bahan dengan satu regex (bahan yang isinya hanya kata pengisi tetap). "
               "Peta frasa hanya untuk bahan yang cukup sering, jadi tetap pendek.")
    fillers = st.multiselect("Kata pengisi (kata nama kolom + 'dari', 'biasa')",
                             sorted(set(text_tools.fillers_for(column)) | set(text_tools.FILLER_WORDS)),
                             text_tools.fillers_for(column), key=f"fill_{column}")
    left, right = st.columns(2)
    phrase_min = left.slider("Frasa diperiksa bila muncul di ≥ … baris", 2, 100, 5, key="phrase_min")
    phrase_ratio = right.slider("'Memuat bahan lain' bila bahan itu ≥ … kali lebih sering", 1.5, 20.0, 2.0, 0.5,
                                key="phrase_ratio")
    phrases = text_tools.phrase_map(data[column], mapping, fillers, phrase_min, phrase_ratio)
    st.caption(f"{len(phrases)} usulan. Dicentang awal: kamus (KBBI) dan urutan kata. 'Memuat bahan lain' tidak "
               "dicentang: kata tambahannya bisa rincian penting ('genteng beton' ≠ 'genteng').")
    chosen = st.data_editor(phrases, width="stretch", hide_index=True, key=f"phrases_{column}",
                            disabled=[c for c in phrases.columns if c != "setuju"])
    phrase_mapping = dict(zip(chosen.loc[chosen["setuju"].astype(bool), "frasa"],
                              chosen.loc[chosen["setuju"].astype(bool), "usulan"]))
    params = {"column": column, "words": mapping, "fillers": fillers, "phrases": phrase_mapping}

    effect = free_text_effect(common.state_key(), json.dumps(params, sort_keys=True, ensure_ascii=False),
                              common.state()["train"][column])
    st.subheader("Hasil peta kata + frasa")
    cols = st.columns(3)
    cols[0].metric("kata · frasa dipetakan", f"{len(mapping)} · {len(phrase_mapping)}")
    cols[1].metric("sel yang isinya dibetulkan", f"{effect['kata dibetulkan']:,}")
    cols[2].metric("sel yang hanya berubah format", f"{effect['hanya format']:,}",
                   help="huruf kecil, urutan bahan, pemisah ', ' — tidak mengubah isi")
    if len(effect["contoh"]):
        st.dataframe(effect["contoh"], width="stretch")
    else:
        st.info("Belum ada kata yang dibetulkan: centang usulan di tabel atas.")
    left, right = st.columns(2)
    left.caption("Bahan terbanyak TANPA peta kata (hanya format)")
    left.dataframe(effect["bahan tanpa peta"], width="stretch", hide_index=True)
    right.caption("Bahan terbanyak DENGAN peta kata, kata pengisi, frasa")
    right.dataframe(effect["bahan dengan peta"], width="stretch", hide_index=True)
    st.subheader("Pratinjau langkah")
    common.preview_step("free_text", params, f"free_{column}")


# ---------------------------------------------------------------- outlier

METHOD_TEXT = {
    "iqr": "**IQR (Tukey)**: di luar Q1 − k·IQR … Q3 + k·IQR. Cocok untuk data kira-kira simetris.",
    "iqr_log": "**IQR-log**: IQR di skala log10. Cocok untuk harga/luas/pendapatan (miring ke kanan). Nol tidak dinilai.",
    "mad": "**MAD**: |x − median| / (MAD/0,6745) > k. Lebih tahan outlier ekstrem. Gagal bila separuh nilai sama.",
    "mad_log": "**MAD-log**: MAD di skala log10. Nol tidak dinilai.",
}


def show_outliers() -> None:
    common.header("outlier",
                  "Bandingkan empat metode batas wajar untuk satu kolom, lihat baris yang kena, pilih **aksi**, "
                  "lalu tambahkan sebagai langkah. Batas dihitung dari train dan dipakai sama di test.",
                  "Kolom angka kontinu (luas, harga, pengeluaran), setelah nilai penjaga dijadikan kosong.",
                  "**drop** membuang baris train; **clip** mengubah nilai. Kolom diskret dan kolom yang mayoritas "
                  "nilainya sama sering salah ditandai.")
    if not common.need_data():
        return
    data = common.current()
    numbers = common.number_columns(data)
    if not numbers:
        st.info("Tidak ada kolom angka.")
        return
    for text in METHOD_TEXT.values():
        st.markdown(f"- {text}")
    mode = st.radio("Cara", ["banyak kolom sekaligus (tabel)", "satu kolom (rinci)"], horizontal=True, key="out_mode")
    if mode.startswith("banyak"):
        show_outlier_table(data, numbers)
    else:
        show_outlier_column(data, numbers)

    with st.expander("Outlier multivariat (IsolationForest, hanya melihat)"):
        picked = st.multiselect("Kolom", numbers, numbers[:4], key="iforest_columns")
        share = st.slider("Perkiraan porsi aneh", 0.001, 0.1, 0.01, format="%.3f")
        if picked and st.button("Jalankan IsolationForest", disabled=jobs.running("iforest")):
            jobs.start("iforest", "IsolationForest", isolation_job, data[picked].copy(), share,
                       note=f"{len(picked)} kolom, porsi {share:.3f}")
            st.rerun()
        jobs.status("iforest")
        flagged = jobs.result("iforest")
        if flagged is not None:
            st.write(f"{len(flagged)} baris ditandai (tidak jadi langkah).")
            st.dataframe(flagged.head(50), width="stretch")


def isolation_job(data: pd.DataFrame, share: float) -> pd.DataFrame:
    """Jalan di thread (tanpa st.*): baris yang ditandai, disimpan utuh agar tetap benar walau langkah berubah."""
    return data[outlier_tools.isolation_flags(data, list(data.columns), share)]


def show_outlier_table(data: pd.DataFrame, numbers: list[str]) -> None:
    """Semua kolom angka dalam satu tabel: pilih 'perbaiki' di baris yang mau, lalu tambah semua langkahnya sekaligus."""
    done = {s["params"].get("column") for s in common.project()["steps"] if s["op"] == "outlier"}
    rows = []
    for column in numbers:
        discrete = is_discrete(data[column])
        rows.append({"kolom": column, "jenis": "diskret" if discrete else "kontinu", "median": data[column].median(),
                     "maks": data[column].max(), "sudah ada langkah": column in done, "perlakuan": "biarkan",
                     "metode": "manual" if discrete else "iqr_log", "k": np.nan, "batas bawah": np.nan,
                     "batas atas": np.nan, "aksi": "clip"})
    st.caption("Semua kolom mulai **biarkan** (data tidak berubah). Ubah jadi **perbaiki** hanya di kolom yang mau "
               "diperbaiki. Kolom **diskret** (mis. jumlah tanggungan) diawali metode **manual**: isi batas atas hanya "
               "untuk nilai yang mustahil, karena keluarga besar 12 orang itu wajar.")
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True, width="stretch", key="out_table",
        disabled=["kolom", "jenis", "median", "maks", "sudah ada langkah"],
        column_config={
            "perlakuan": st.column_config.SelectboxColumn(options=["biarkan", "perbaiki"], required=True),
            "metode": st.column_config.SelectboxColumn(options=list(METHODS), required=True),
            "k": st.column_config.NumberColumn(min_value=0.5, max_value=10.0, step=0.5,
                                               help="kosong = bawaan metode (IQR 1,5 · MAD 3,5); diabaikan untuk manual"),
            "batas bawah": st.column_config.NumberColumn(help="hanya untuk metode manual; kosong = tanpa batas"),
            "batas atas": st.column_config.NumberColumn(help="hanya untuk metode manual; kosong = tanpa batas"),
            "aksi": st.column_config.SelectboxColumn(options=list(ACTIONS), required=True,
                                                     help="clip = potong ke batas, nan = kosongkan, drop = buang baris train"),
        })

    plan, result_rows = [], []
    for row in edited.to_dict("records"):
        method = row["metode"]
        factor = 0.0 if method == "manual" else float(row["k"] if pd.notna(row["k"]) else outlier_tools.FACTORS[method])
        low = None if method != "manual" or pd.isna(row["batas bawah"]) else float(row["batas bawah"])
        high = None if method != "manual" or pd.isna(row["batas atas"]) else float(row["batas atas"])
        values = data[row["kolom"]]
        bounds = outlier_tools.outlier_bounds(values, method, factor, low, high)
        outside = outlier_tools.is_outside(values, bounds, method)
        result_rows.append({"kolom": row["kolom"], "perlakuan": row["perlakuan"], "batas bawah": bounds[0],
                            "batas atas": bounds[1], "di luar batas": int(outside.sum()),
                            "% data": round(outside.mean() * 100, 2)})
        if row["perlakuan"] == "perbaiki":
            if method == "manual" and low is None and high is None:
                st.warning(f"**{row['kolom']}**: metode manual tanpa batas apa pun → dilewati.")
                continue
            plan.append({"column": row["kolom"], "method": method, "factor": factor, "low": low, "high": high,
                         "action": row["aksi"]})
    st.caption("Hasil dengan pilihan di atas (dihitung dari train sesudah semua langkah):")
    st.dataframe(pd.DataFrame(result_rows), width="stretch", hide_index=True)

    if not plan:
        st.info("Belum ada kolom bertanda **perbaiki**: tidak ada langkah yang ditambahkan, data tidak berubah.")
        return
    common.preview_steps("outlier", plan, "out_table")


def show_outlier_column(data: pd.DataFrame, numbers: list[str]) -> None:
    """Satu kolom secara rinci: banding metode, baris yang kena, per kelompok, lalu biarkan atau perbaiki."""
    column = st.selectbox("Kolom angka", numbers, key="outlier_column")
    values = data[column]
    st.caption(f"median {values.median():,.2f} · p99 {values.quantile(0.99):,.2f} · maks {values.max():,.2f} · "
               f"nol {int((values == 0).sum())} · kosong {int(values.isna().sum())}")
    st.dataframe(outlier_tools.compare_methods(values), width="stretch", hide_index=True)
    if is_discrete(values):
        st.warning(f"**{column}** berisi bilangan bulat dengan {values.nunique()} nilai berbeda (diskret). IQR/MAD "
                   "menandai nilai jarang yang wajar (mis. keluarga besar 12 orang) sebagai outlier. Pilihan aman: "
                   "**biarkan**, atau **batas manual** hanya untuk nilai yang mustahil.")
    choice = st.radio("Perlakuan kolom ini", ["biarkan: tidak diperbaiki (hanya dilihat)", "perbaiki dengan langkah"],
                      horizontal=True, key=f"out_choice_{column}")
    params = bound_widgets(values, "out", {"method": "manual" if is_discrete(values) else "iqr_log"})
    method, factor = params["method"], params["factor"]
    low, high = outlier_tools.outlier_bounds(values, method, factor, params["low"], params["high"])
    outside = outlier_tools.outside_rows(data, column, method, factor, params["low"], params["high"])
    st.write(f"Batas: **{low:,.2f} … {high:,.2f}** → {len(outside)} baris di luar batas")
    st.dataframe(outside.sort_values(column).iloc[np.r_[0:5, -5:0]] if len(outside) > 10 else outside,
                 width="stretch")

    label = common.label()
    groups = [c for c in [label] if c] + list(st.session_state.get("laya_answers", {}))
    if groups:
        group = st.selectbox("Bandingkan per kelompok (konteks)", [None, *groups], key="out_group")
        if group:
            source = (data.assign(**{group: common.state()["train"][group]}) if group == label
                      else data.assign(**{group: st.session_state["laya_answers"][group]}))
            st.dataframe(outlier_tools.by_group(source, column, group, method, factor, params["low"], params["high"]),
                         width="stretch", hide_index=True)
            st.caption("Nilai wajar di satu kelompok bisa aneh di kelompok lain. Jawaban Laya hanya untuk melihat.")
    if choice.startswith("biarkan"):
        st.info(f"**{column}** dibiarkan apa adanya: tidak ada langkah yang ditambahkan, data tidak berubah. "
                "Tabel di atas hanya untuk dilihat.")
        return
    action = st.selectbox("Aksi", list(ACTIONS), format_func=ACTIONS.get, key="out_a")
    st.subheader("Pratinjau langkah")
    common.preview_step("outlier", {"column": column, **params, "action": action}, f"out_{column}")
