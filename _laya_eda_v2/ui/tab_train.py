"""Tab Latih (fitur extra E1, E3, E4, E5, E7–E12): langkah → Pipeline sklearn → validasi silang beberapa model
dengan metrik lomba → papan skor + ensemble OOF → analisis kesalahan + figur → simpan pipeline.joblib.

Latih berat jalan di thread (ui/jobs.py): pindah halaman tidak membatalkan.
"""
from __future__ import annotations

import copy
import importlib.util

import pandas as pd
import streamlit as st

from core import audit, diagnose, ensemble, schema, submission, training
from core import pipeline as pipes
from core import split as splitter
from ui import common, jobs
from ui.kamus import show_table


def default_task(data: pd.DataFrame, label: str) -> str:
    """Label bilangan bulat berurutan (mis. Golongan UKT 0–7) → 'berurutan'; teks/sedikit nilai → 'kelas'."""
    values = data[label].dropna()
    if not splitter.is_classification(data, label):
        return "angka"
    if pd.api.types.is_numeric_dtype(values) and len(values) and (values % 1 == 0).all() and values.nunique() > 2:
        return "berurutan"
    return "kelas"


def train_job(setelan: dict, train: pd.DataFrame, test: pd.DataFrame | None, models: list[str], budget: float) -> dict:
    """Jalan di thread (tanpa st.*)."""
    board = training.leaderboard(setelan, train, test, models, budget=budget)
    finished = {name: r for name, r in board["hasil"].items() if r["cv"]["baris_oof"].any()}
    if len(finished) > 1:
        rows = next(iter(finished.values()))["cv"]["baris_oof"]
        oofs = {name: r["cv"]["oof"][rows] for name, r in finished.items()}
        y = next(iter(finished.values()))["y"][rows]
        board["ensemble"] = ensemble.choose(oofs, y, setelan["tugas"], setelan["metrik"])
    return board


def plan_table(steps: list[dict]) -> pd.DataFrame:
    chosen = pipes.plan(steps)
    rows = [{"langkah": i + 1, "operasi": engine_summary(step), "bagian": "blok EDA (pandas, sebelum pipeline)",
             "transformer": "-"} for i, step in enumerate(chosen["eda"])]
    rows += [{"langkah": number, "operasi": engine_summary(step), "bagian": "Pipeline sklearn",
              "transformer": spec["make"].split("(")[0]} for number, step, spec in chosen["pipe"]]
    return pd.DataFrame(rows)


def engine_summary(step: dict) -> str:
    from core.ops import summary
    return summary(step)


def settings(data: pd.DataFrame, label: str) -> tuple[dict, list[str]]:
    left, middle, right = st.columns(3)
    tasks = list(training.TASKS)
    st.session_state.setdefault("train_task", default_task(data, label))
    task = left.radio("Tugas", tasks, key="train_task", format_func=training.TASKS.get)
    allowed = [m for m, spec in training.METRICS.items()
               if (spec["kind"] == "value") == (task == "angka") and not (task == "berurutan" and spec["kind"] == "proba")]
    if st.session_state.get("train_metric") not in allowed:
        st.session_state["train_metric"] = "qwk" if task == "berurutan" else allowed[0]
    metric = middle.selectbox("Metrik lomba (E7)", allowed, key="train_metric",
                              format_func=lambda m: training.METRICS[m]["title"])
    mode = right.radio("Mode waktu (E10)", list(training.MODES), horizontal=True, key="train_mode")
    right.caption(training.MODES[mode]["what"] + " Default CPU, n_jobs=2 (laptop pernah mati saat beban GPU berat).")
    available = [m for m in training.MODELS if importlib.util.find_spec(training.MODELS[m]["lib"].split(".")[0])]
    st.session_state.setdefault("train_models", [m for m in ("lightgbm", "catboost", "xgboost") if m in available])
    models = st.multiselect("Model baseline (E5)", list(training.MODELS), key="train_models",
                            format_func=lambda m: training.MODELS[m]["title"] + ("" if m in available else " (belum terpasang)"))
    with st.expander("Apa / kapan cocok / kapan berbahaya: model"):
        show_table(pd.DataFrame([{"model": spec["title"], "apa": spec["what"], "kapan cocok": spec["fits"],
                                  "kapan berbahaya": spec["danger"]} for spec in training.MODELS.values()]),
                   {"apa": "Cara kerja.", "kapan cocok": "Situasi yang cocok.", "kapan berbahaya": "Risiko."},
                   hide_index=True)
    left, middle, right = st.columns(3)
    kinds = list(training.FOLDS) if task != "angka" else [k for k in training.FOLDS if k != "stratified"]
    if st.session_state.get("train_fold") not in kinds:
        st.session_state["train_fold"] = kinds[0]
    fold = left.selectbox("Validasi silang (E3)", kinds, key="train_fold", format_func=training.FOLDS.get)
    features = [c for c in data.columns if c != label]
    group = time_column = None
    if fold == "group":
        group = middle.selectbox("Kolom kelompok", features, key="train_group")
    if fold == "time":
        time_column = middle.selectbox("Kolom waktu (diurutkan)", features, key="train_time")
    guess_id = [c for c in features if audit.guess_type(data[c]) == "id"]
    exclude = right.multiselect("Bukan fitur (id, bocoran)", features, guess_id, key="train_exclude")
    imbalance = "tanpa"
    if task != "angka":
        report = training.imbalance_report(data[label].dropna())
        options = list(training.IMBALANCE) if task == "kelas" else ["tanpa", "bobot"]
        st.caption(f"Kelas terkecil {report['terkecil']!r} = {report['jumlah'].min():,} baris, terbesar "
                   f"{report['terbesar']!r} = {report['jumlah'].max():,} → rasio {report['rasio']:.3f}"
                   + (" (**timpang**, E12)" if report["timpang"] else ""))
        imbalance = st.radio("Kelas timpang (E12)", options, horizontal=True, key="train_imbalance")
        spec = training.IMBALANCE[imbalance]
        st.caption(f"**Apa.** {spec['what']} **Kapan cocok.** {spec['fits']} **Kapan berbahaya.** {spec['danger']}")
    config = training.MODES[mode]
    return dict(label=label, langkah=copy.deepcopy(common.project()["steps"]), split=copy.deepcopy(common.split_config()),
                tugas=task, metrik=metric, n_estimators=config["n"], fold=fold, n_fold=config["folds"], seed=0,
                timpang=imbalance, kecuali=exclude, kolom_grup=group, kolom_waktu=time_column, mode=mode,
                anggaran=config["budget"]), models


def show_train() -> None:
    common.header("latih",
                  "Fitur **extra**: langkah yang punya transformer sklearn/feature_engine disusun jadi **Pipeline** "
                  "(preprocess → model) dan dinilai dengan **validasi silang**: preprocessing di-fit ulang di setiap "
                  "fold (tanpa bocor). Beberapa model baseline masuk satu **papan skor** dengan metrik lomba, lalu "
                  "ensemble dari prediksi out-of-fold, analisis kesalahan, dan `pipeline.joblib` untuk tab Submission.",
                  "Sesudah langkah EDA cukup: untuk tahu skor jujur dan menyimpan model siap prediksi.",
                  "Mode lengkap bisa lama dan membuat laptop panas. Skor OOF sesudah ambang sedikit optimis (ambang "
                  "dicari di OOF yang sama). Baris tanpa label tidak ikut dilatih.")
    if not common.need_data():
        return
    label = common.label()
    if not label:
        st.info("Pilih kolom label di sidebar **Data**.")
        return
    steps = common.project()["steps"]
    st.subheader("Langkah → blok EDA + Pipeline (E1)")
    st.caption("Pipeline = akhiran daftar langkah yang semuanya punya transformer. Langkah pandas (dan semua langkah "
               "sebelumnya) jalan sebagai blok EDA sebelum pipeline, di train dan test, urutan tidak berubah.")
    table = plan_table(steps)
    if len(table):
        show_table(table, hide_index=True)
    data = common.state(0)["train"]
    chosen, models = settings(data, label)
    left, right = st.columns(2)
    if left.button("Perkiraan lama (E10)", disabled=jobs.running("latih_perkiraan") or not models):
        jobs.start("latih_perkiraan", "Perkiraan lama latih", training.estimate_seconds, chosen,
                   common.raw_train().copy(), models)
        st.rerun()
    jobs.status("latih_perkiraan")
    guess = jobs.result("latih_perkiraan")
    if guess:
        total = sum(guess.values())
        left.caption(f"Perkiraan total ±{total:.0f} s (" + ", ".join(f"{k} {v:.0f} s" for k, v in guess.items()) + ")"
                     + (" → melewati anggaran, sebagian model akan dilewati." if total > chosen["anggaran"] else ""))
    if right.button("Latih + papan skor", type="primary", disabled=jobs.running("latih") or not models):
        test = common.raw_test().copy() if common.raw_test() is not None else None
        jobs.start("latih", "Latih + papan skor", train_job, chosen, common.raw_train().copy(), test, list(models),
                   chosen["anggaran"], note=f"{len(models)} model, {chosen['mode']}, metrik {chosen['metrik']}")
        st.rerun()
    jobs.status("latih")
    board = jobs.result("latih")
    if board:
        show_results(board)
    show_log()


def show_results(board: dict) -> None:
    st.subheader("Papan skor (E5)")
    show_table(board["papan"], hide_index=True)
    results = board["hasil"]
    if not results:
        return
    first = next(iter(results.values()))
    metric = first["setelan"]["metrik"]
    if first["baris_tanpa_label"]:
        st.caption(f"{first['baris_tanpa_label']:,} baris train tanpa label tidak ikut dilatih.")
    if "ensemble" in board:
        choice = board["ensemble"]
        st.subheader("Ensemble dari OOF (E9)")
        st.write(f"Pilihan: **{choice['pilihan']}** ({metric} OOF {choice['skor']:.4f}). Tunggal terbaik "
                 f"{choice['tunggal_terbaik']} {choice['skor_tunggal']:.4f} · rata-rata berbobot "
                 f"{choice['skor_bobot']:.4f} · stacking {choice['skor_stacking']:.4f}. Ensemble hanya dipilih "
                 "bila lebih baik; bobot dicari di OOF, bukan test.")
        show_table(pd.DataFrame({"model": list(choice["bobot"]), "bobot": list(choice["bobot"].values())}),
                   {"bobot": "Bobot rata-rata berbobot (≥ 0, jumlah 1) yang dicari di OOF."}, hide_index=True)
    name = st.selectbox("Model untuk analisis dan simpan", list(results), key="train_pick",
                        format_func=lambda m: training.MODELS[m]["title"])
    result = results[name]
    if result["kelas"] is not None:
        st.subheader("Analisis kesalahan (E11) dan skor per kelas (E12)")
        table = diagnose.confusion(result)
        left, right = st.columns(2)
        with left:
            show_table(table.rename_axis(None, axis=1).reset_index(),
                       {str(c): f"Jumlah baris berlabel asli (baris) yang ditebak kelas {c}." for c in table.columns},
                       hide_index=True)
        with right:
            show_table(diagnose.per_class(result), hide_index=True)
        show_table(diagnose.most_confused(table), hide_index=True)
        worst = diagnose.worst_rows(result)
        st.caption("Baris dengan salah terbesar (nomor baris = indeks train asli).")
        show_table(worst, hide_index=True)
        st.button("Buka baris ini di halaman Data", key="open_worst", on_click=open_rows,
                  args=(worst["baris"].tolist(),))
        if st.button("Simpan figur 16:9 (confusion matrix + kolom penting)", key="save_figures"):
            folder = training.OUTPUT / "figur"
            paths = [diagnose.figure_confusion(table, folder / f"confusion_{name}.png"),
                     diagnose.figure_importance(diagnose.importance(result), folder / f"importance_{name}.png")]
            st.success("Tersimpan: " + ", ".join(str(p) for p in paths))
    if result.get("pipe") is not None and st.button("Simpan pipeline.joblib (E2) untuk tab Submission", type="primary"):
        path = training.OUTPUT / "pipeline.joblib"
        meta = pipes.save(result["pipe"], path, submission.info_for(result, common.label()))
        st.session_state["pipeline_path"] = str(path)
        st.success(f"Tersimpan {path} (+ .json, sklearn {meta['sklearn']}).")
    if len(result["test"]):
        st.subheader("Skema test (E4, pandera mode lapor)")
        report = schema.validate(result["test"].reindex(columns=result["fitur"]), schema.learn(result["X"]))
        if report.empty:
            st.caption("Test lolos skema yang dipelajari dari train.")
        else:
            show_table(report, hide_index=True)


def open_rows(rows: list) -> None:
    """Callback (sebelum rerun): halaman Data menampilkan baris ini saja."""
    st.session_state["fokus_baris"] = rows
    st.session_state["page"] = "Data"


def show_log() -> None:
    st.subheader("Log percobaan (E8)")
    log = training.read_log()
    st.caption(f"File: {training.OUTPUT / 'percobaan.csv'} (di luar repo).")
    if log.empty:
        st.caption("Belum ada percobaan.")
        return
    show_table(log.drop(columns=["setelan"]), hide_index=True)
    row = st.selectbox("Buka ulang percobaan", list(log.index)[::-1], key="log_pick",
                       format_func=lambda i: f"{log.loc[i, 'waktu']} · {log.loc[i, 'model']} · {log.loc[i, 'skor_rata']}")
    st.button("Pakai setelan + langkah percobaan ini", key="log_open", on_click=reopen, args=(log.loc[row].to_dict(),),
              help="Daftar langkah dan split diganti dengan milik percobaan itu; latih lagi → skor sama.")


def reopen(row: dict) -> None:
    setelan = training.reopen(row)
    project = common.project()
    project["steps"], project["split"] = setelan["langkah"], setelan.get("split")
    for key, name in (("train_task", "tugas"), ("train_metric", "metrik"), ("train_fold", "fold"),
                      ("train_imbalance", "timpang"), ("train_exclude", "kecuali"), ("train_mode", "mode")):
        if setelan.get(name) is not None:
            st.session_state[key] = setelan[name]
    st.session_state["train_models"] = [setelan["model"]]
