"""Tugas model cepat Laya (F1–F4, T1). Dijalankan python venv di proses terpisah, bukan oleh Laya.exe:

    python tugas.py <tugas.json>

tugas.json: {"jenis", "latih": parquet, "uji": parquet|null, "teks": [kolom], "label": kolom|null, "id": kolom|null,
"keluar": folder, "seed": 42, "fold": 5, "validasi": {cara, fold, seed, kolom, porsi} | null}. Kemajuan dicetak ke stdout (`## pesan`); hasil ke <keluar>/hasil.json.
Butuh scikit-learn + polars (+ cleanlab untuk label mencurigakan). Crash di sini tidak menjatuhkan Laya.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy.sparse import vstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # SGD dibatasi 30 putaran demi kecepatan, disengaja
MULAI = time.perf_counter()
KATA = dict(ngram_range=(1, 2), n_features=2 ** 20, alternate_sign=False, norm=None)


def lapor(pesan):
    print(f"## [{time.perf_counter() - MULAI:5.1f} s] {pesan}", flush=True)


def teks_dari(df, kolom):
    return df.select(pl.concat_str([pl.col(c).cast(pl.String).fill_null("") for c in kolom], separator="\n")) \
        .to_series().to_list()


def _hash(teks):
    return HashingVectorizer(**KATA).transform(teks)


def fitur(teks, tfidf=True):
    """n-gram kata 1–2 di-hash paralel (tanpa kosakata, jadi bisa dipecah per potongan): GDOC 52 rb artikel ± 8 s,
    TfidfVectorizer biasa 51 s. n-gram huruf sengaja tidak dipakai: 3x lebih lambat dan macro-F1 GDOC tidak naik."""
    n = max(1, min(os.cpu_count() or 1, 16))
    k = -(-len(teks) // n)
    X = vstack(Parallel(n_jobs=n)(delayed(_hash)(teks[i:i + k]) for i in range(0, len(teks), k))).tocsr()
    return TfidfTransformer(sublinear_tf=True).fit_transform(X) if tfidf else X


def nama_fitur(teks, seed, n=5000):
    """Peta indeks hash → n-gram dari sampel dokumen (hashing tidak menyimpan kosakata). Yang tak terpetakan = '#indeks'."""
    ambil = np.random.default_rng(seed).choice(len(teks), min(n, len(teks)), replace=False)
    an = HashingVectorizer(**KATA).build_analyzer()
    gram = list(dict.fromkeys(g for i in ambil for g in an(teks[i])))
    X = HashingVectorizer(**{**KATA, "analyzer": lambda d: d}).transform([[x] for x in gram]).tocsr()
    peta = {}
    for x, j in zip(gram, X.indices):
        peta.setdefault(int(j), x)
    return lambda j: peta.get(int(j), f"#{j}")


def kata_teratas(urutan, nama, n=15):
    """n-gram terpetakan pertama; indeks hash tanpa nama (#123) dibuang karena tidak terbaca."""
    out = []
    for j in urutan:
        x = nama(j)
        if not x.startswith("#"):
            out.append(x)
            if len(out) == n:
                break
    return out


def lipatan(t, df, y, g_teks=None, klasifikasi=True):
    """Daftar (indeks latih, indeks validasi) sesuai pengaturan validasi proyek. Otomatis: grup teks kembar bila ada
    kolom teks, selain itu stratified (klasifikasi) atau k-fold biasa (regresi)."""
    v = t.get("validasi") or {}
    cara, k, seed = v.get("cara") or "otomatis", int(v.get("fold") or t["fold"]), int(v.get("seed") or t["seed"])
    if cara == "otomatis":
        cara = "grup: teks kembar" if g_teks is not None else "stratified k-fold" if klasifikasi else "k-fold biasa"
    nol = np.zeros(len(y))
    if cara == "holdout":
        porsi = float(v.get("porsi") or 0.2)
        a, b = train_test_split(np.arange(len(y)), test_size=porsi, random_state=seed,
                                stratify=y if klasifikasi else None)
        return [(a, b)], f"holdout {porsi:.0%} (seed {seed})"
    if cara.startswith("grup"):
        if cara == "grup: teks kembar":
            g, nama = g_teks, "teks kembar"
        else:
            g, nama = df[v["kolom"]].cast(pl.String).fill_null("").to_numpy(), v["kolom"]
        if g is None:
            raise ValueError("validasi grup teks kembar butuh kolom teks; pilih cara lain")
        cv = StratifiedGroupKFold(k, shuffle=True, random_state=seed) if klasifikasi else GroupKFold(k)
        return list(cv.split(nol, y, g)), f"{k} fold berkelompok per {nama} (seed {seed})"
    if cara == "stratified k-fold" and klasifikasi:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(nol, y)), \
            f"stratified {k} fold (seed {seed})"
    return list(KFold(k, shuffle=True, random_state=seed).split(nol)), f"k-fold {k} (seed {seed})"


def grup(teks):
    """Teks persis sama (setelah huruf kecil + spasi dirapikan) = satu grup, supaya duplikat tidak bocor antar-fold."""
    kunci = pl.Series(teks).str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()
    return kunci.rank("dense").to_numpy()


def model_linear(seed):
    """Regresi logistik lewat SGD: detik per fold di 400 rb fitur (lbfgs: menit). Probabilitas dipakai cleanlab."""
    return SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=30, tol=1e-4, random_state=seed)


def oof(X, y, bagi, seed, label_teks="fold"):
    """Prob OOF; baris yang tidak pernah jadi validasi (holdout) tetap 0 dan ditandai lewat `dinilai`."""
    p = np.zeros((X.shape[0], y.max() + 1))
    skor, model, dinilai = [], None, np.zeros(len(y), bool)
    for k, (a, b) in enumerate(bagi, 1):
        model = model_linear(seed).fit(X[a], y[a])
        p[np.ix_(b, model.classes_)] = model.predict_proba(X[b])
        dinilai[b] = True
        skor.append(f1_score(y[b], p[b].argmax(1), average="macro"))
        lapor(f"{label_teks} {k}/{len(bagi)}: macro-F1 {skor[-1]:.4f}")
    return p, skor, model, dinilai


def baseline(t, df, uji):
    """F1 baseline TF-IDF kata+huruf + LR, CV berkelompok per teks kembar; F2 label mencurigakan dari prob OOF."""
    teks, lab = teks_dari(df, t["teks"]), df[t["label"]].cast(pl.String).fill_null("(kosong)").to_numpy()
    kelas, y = np.unique(lab, return_inverse=True)
    lapor(f"{len(teks)} baris, {len(kelas)} kelas; membuat fitur TF-IDF")
    X = fitur(teks)
    lapor(f"fitur: {X.nnz:,} isi tak nol")
    bagi, ket = lipatan(t, df, y, grup(teks))
    lapor(f"validasi: {ket}")
    p, skor, model, dinilai = oof(X, y, bagi, t["seed"])
    tebak = p.argmax(1)
    yv, tv = y[dinilai], tebak[dinilai]
    lapor_kelas = classification_report(yv, tv, labels=range(len(kelas)), target_names=[str(k) for k in kelas],
                                        output_dict=True, zero_division=0)
    w = model.coef_
    if len(kelas) == 2:
        w = np.vstack([-w[0], w[0]])
    nama = nama_fitur(teks, t["seed"])
    kata = {str(k): kata_teratas(np.argsort(w[i])[::-1], nama) for i, k in enumerate(kelas)}
    hasil = {"macro_f1": f1_score(yv, tv, average="macro"), "akurasi": accuracy_score(yv, tv), "validasi": ket,
             "per_fold": skor, "kelas": [str(k) for k in kelas], "pendek": _teks_pendek(teks),
             "matriks": confusion_matrix(yv, tv, labels=range(len(kelas))).tolist(), "laporan": lapor_kelas,
             "kata": kata,
             "kelas_terbanyak": float(np.bincount(y).max() / len(y))}
    pl.DataFrame({"baris": np.arange(len(y)), "label": lab, "tebak": kelas[tebak]}) \
        .with_columns([pl.Series(f"p_{k}", p[:, i]) for i, k in enumerate(kelas)]) \
        .write_parquet(Path(t["keluar"]) / "oof.parquet")
    try:
        from cleanlab.filter import find_label_issues
        lapor("mencari label mencurigakan (cleanlab)")
        nilai = np.flatnonzero(dinilai)
        idx = nilai[find_label_issues(yv, p[dinilai], return_indices_ranked_by="self_confidence")]
        hasil["curiga"] = [{"baris": int(i), "id": (str(df[t["id"]][int(i)]) if t.get("id") else None),
                            "label": str(lab[i]), "tebak": str(kelas[tebak[i]]), "yakin": float(p[i].max()),
                            "teks": teks[i][:200]} for i in idx[:300]]
        hasil["curiga_total"] = int(len(idx))
    except ImportError:
        hasil["curiga_galat"] = "cleanlab tidak terpasang di venv"
    return hasil


def adversarial(t, df, uji):
    """F3: bisakah model membedakan train dari test? AUC ≈ 0,5 = sebaran sama."""
    a, b = teks_dari(df, t["teks"]), teks_dari(uji, t["teks"])
    semua, y = a + b, np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
    lapor(f"{len(a)} baris latih vs {len(b)} baris uji; membuat fitur TF-IDF kata")
    X = fitur(semua)
    p = np.zeros(len(y))
    for k, (i, j) in enumerate(StratifiedKFold(t["fold"], shuffle=True, random_state=t["seed"]).split(X, y), 1):
        m = model_linear(t["seed"]).fit(X[i], y[i])
        p[j] = m.predict_proba(X[j])[:, 1]
        lapor(f"fold {k}/{t['fold']}")
    w = m.coef_[0]
    urut = np.argsort(p[len(a):])[::-1][:30] + len(a)
    nama = nama_fitur(semua, t["seed"])
    return {"auc": roc_auc_score(y, p), "kata_uji": kata_teratas(np.argsort(w)[::-1], nama, 20),
            "kata_latih": kata_teratas(np.argsort(w), nama, 20),
            "paling_beda": [{"baris": int(i - len(a)), "p": float(p[i]), "teks": semua[i][:200]} for i in urut]}


def acak(t, df, uji):
    """F4: CV dengan label diacak harus ≈ tebakan. Jauh di atas = ada kebocoran (urutan, duplikat, dsb.)."""
    teks, lab = teks_dari(df, t["teks"]), df[t["label"]].cast(pl.String).fill_null("(kosong)").to_numpy()
    _, y = np.unique(lab, return_inverse=True)
    y = np.random.default_rng(t["seed"]).permutation(y)
    lapor(f"{len(teks)} baris, label diacak (seed {t['seed']}); membuat fitur TF-IDF kata")
    X = fitur(teks)
    bagi, _ = lipatan({**t, "validasi": {**(t.get("validasi") or {}), "fold": 3}}, df, y, grup(teks))
    p, skor, _, _ = oof(X, y, bagi, t["seed"], "fold acak")
    frek = np.bincount(y) / len(y)
    return {"macro_f1": f1_score(y, p.argmax(1), average="macro"), "per_fold": skor,
            "tebakan_seragam": float(np.mean([2 * f / (1 + f * len(frek)) for f in frek])),
            "tebakan_terbanyak": float(2 * frek.max() / (1 + frek.max()) / len(frek))}


def kata_khas(t, df, uji):
    """T1: kata/bigram khas per label, log-odds dengan prior Dirichlet (Monroe dkk. 2008), skor z."""
    teks, lab = teks_dari(df, t["teks"]), df[t["label"]].cast(pl.String).fill_null("(kosong)").to_numpy()
    kelas, y = np.unique(lab, return_inverse=True)
    X = fitur(teks, tfidf=False)
    nama = nama_fitur(teks, t["seed"])
    total = np.asarray(X.sum(0)).ravel().astype(float)
    a = total / total.sum() * 1000 + 0.01  # prior dari seluruh korpus
    hasil = {}
    for i, k in enumerate(kelas):
        di = np.asarray(X[y == i].sum(0)).ravel()
        dj = total - di
        ni, nj, a0 = di.sum(), dj.sum(), a.sum()
        d = np.log((di + a) / (ni + a0 - di - a)) - np.log((dj + a) / (nj + a0 - dj - a))
        z = d / np.sqrt(1 / (di + a) + 1 / (dj + a))
        z[total < 5] = -np.inf  # kata langka: z besar tapi tidak bermakna
        hasil[str(k)] = [{"kata": nama(j), "z": float(z[j]), "n": int(di[j])} for j in np.argsort(z)[::-1][:300]
                         if not nama(j).startswith("#")][:25]
        lapor(f"{k}: {hasil[str(k)][0]['kata']} …")
    return {"kata": hasil, "kelas": [str(k) for k in kelas]}


def _teks_pendek(teks):
    """Kolom 'teks' yang sebenarnya kategori/angka (mis. rentang pendapatan): median < 4 kata atau <= 50 nilai unik."""
    return float(np.median([len(x.split()) for x in teks[:5000]] or [0])) < 4 or len(set(teks)) <= 50


def _matriks_fitur(df, kolom):
    """Kolom polars -> pandas untuk LightGBM: teks jadi kategori, tanggal jadi angka hari, list dibuang. Teks bebas
    (nilai unik > 50% baris) dilewati: sebagai kategori ia hanya menghafal."""
    ubah, lewat, n = [], [], df.height
    for c in kolom:
        tp = df.schema[c]
        if tp.is_nested() or tp == pl.Object:
            lewat.append(f"{c} (list/struct)")
        elif tp in (pl.String, pl.Categorical) and df[c].n_unique() > max(50, 0.5 * n):
            lewat.append(f"{c} (teks bebas/kode unik; pakai Baseline teks)")
        elif tp.is_temporal():
            ubah.append(pl.col(c).dt.epoch("d").cast(pl.Float64))
        elif tp == pl.Boolean:
            ubah.append(pl.col(c).cast(pl.Int8))
        elif tp in (pl.String, pl.Categorical, pl.Enum):
            ubah.append(pl.col(c).cast(pl.String))
        else:
            ubah.append(pl.col(c))
    X = df.select(ubah).to_pandas()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype).startswith("str"):
            X[c] = X[c].astype("category")
    return X, lewat


def tabular(t, df, uji):
    """Baseline tabular: LightGBM di semua kolom selain label/id/kolom grup, CV sesuai pengaturan validasi.
    Klasifikasi bila label teks atau angka dengan <= 20 nilai; selain itu regresi."""
    import lightgbm as lgb
    lab, v, seed = t["label"], t.get("validasi") or {}, t["seed"]
    buang = {lab, t.get("id"), v.get("kolom") if v.get("cara") == "grup: kolom" else None}
    X, lewat = _matriks_fitur(df, [c for c in df.columns if c not in buang])
    if not X.shape[1]:
        raise ValueError("tidak ada kolom fitur yang bisa dipakai")
    s = df[lab]
    klas = not s.dtype.is_numeric() or s.n_unique() <= 20
    lapor(f"{len(X)} baris, {X.shape[1]} kolom fitur, {'klasifikasi' if klas else 'regresi'}; LightGBM")
    if klas:
        lab_s = s.cast(pl.String).fill_null("(kosong)").to_numpy()
        kelas, y = np.unique(lab_s, return_inverse=True)
    else:
        y = s.cast(pl.Float64).to_numpy()
        if np.isnan(y).any():
            raise ValueError(f"label {lab} punya {int(np.isnan(y).sum())} nilai kosong; isi atau buang dulu")
    bagi, ket = lipatan(t, df, y, None, klas)
    lapor(f"validasi: {ket}")
    par = dict(n_estimators=200, learning_rate=0.1, num_leaves=31, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, random_state=seed, verbose=-1,
               n_jobs=min(8, os.cpu_count() or 1))  # 32 thread malah 2x lebih lambat (data kecil)
    p = np.zeros((len(y), len(kelas))) if klas else np.zeros(len(y))
    dinilai, gain, skor, fold = np.zeros(len(y), bool), np.zeros(X.shape[1]), [], np.full(len(y), -1)
    for k, (a, b) in enumerate(bagi, 1):
        m = (lgb.LGBMClassifier(**par) if klas else lgb.LGBMRegressor(**par)).fit(X.iloc[a], y[a])
        if klas:
            p[np.ix_(b, m.classes_)] = m.predict_proba(X.iloc[b])
            skor.append(f1_score(y[b], p[b].argmax(1), average="macro"))
        else:
            p[b] = m.predict(X.iloc[b])
            skor.append(float(np.sqrt(mean_squared_error(y[b], p[b]))))
        dinilai[b], fold[b] = True, k
        gain += m.booster_.feature_importance("gain")
        lapor(f"fold {k}/{len(bagi)}: {'macro-F1' if klas else 'RMSE'} {skor[-1]:.4f}")
    urut = np.argsort(gain)[::-1]
    hasil = {"klasifikasi": klas, "validasi": ket, "per_fold": skor, "dilewati": lewat, "label": lab,
             "penting": [(str(X.columns[i]), float(gain[i] / (gain.sum() or 1))) for i in urut[:25]],
             "n_fitur": int(X.shape[1])}
    kunci = df.select(pl.struct(list(X.columns)).hash().alias("k")).with_columns(pl.Series("f", fold))
    lintas = kunci.filter(pl.col("f") >= 0).group_by("k").agg(pl.col("f").n_unique().alias("nf"), pl.len()) \
        .filter(pl.col("nf") > 1)
    hasil["kembar_lintas"] = int(lintas["len"].sum()) if lintas.height else 0
    yv = y[dinilai]
    if klas:
        tv = p[dinilai].argmax(1)
        hasil.update({"macro_f1": f1_score(yv, tv, average="macro"), "akurasi": accuracy_score(yv, tv),
                      "kelas": [str(k) for k in kelas], "kelas_terbanyak": float(np.bincount(y).max() / len(y)),
                      "matriks": confusion_matrix(yv, tv, labels=range(len(kelas))).tolist(),
                      "laporan": classification_report(yv, tv, labels=range(len(kelas)),
                                                       target_names=[str(k) for k in kelas], output_dict=True,
                                                       zero_division=0)})
        keluar = pl.DataFrame({"baris": np.arange(len(y)), "fold": fold, "label": lab_s,
                               "tebak": kelas[p.argmax(1)]}).with_columns(
            [pl.Series(f"p_{k}", p[:, i]) for i, k in enumerate(kelas)])
    else:
        pv = p[dinilai]
        hasil.update({"rmse": float(np.sqrt(mean_squared_error(yv, pv))), "mae": float(mean_absolute_error(yv, pv)),
                      "r2": float(r2_score(yv, pv)), "rmse_rata": float(np.std(yv))})
        keluar = pl.DataFrame({"baris": np.arange(len(y)), "fold": fold, "label": y, "tebak": p})
    keluar.write_parquet(Path(t["keluar"]) / "oof.parquet")
    return hasil


def _label(s):
    """(klasifikasi, y, kelas): klasifikasi bila label teks atau angka dengan <= 20 nilai (sama dengan tabular)."""
    if not s.dtype.is_numeric() or s.n_unique() <= 20:
        kelas, y = np.unique(s.cast(pl.String).fill_null("(kosong)").to_numpy(), return_inverse=True)
        return True, y, kelas
    y = s.cast(pl.Float64).to_numpy()
    if np.isnan(y).any():
        raise ValueError(f"label {s.name} punya {int(np.isnan(y).sum())} nilai kosong; isi atau buang dulu")
    return False, y, None


def fold(t, df, uji):
    """Kolom fold sesuai pengaturan validasi -> fold.csv, supaya notebook/ensemble memakai split yang sama persis."""
    klas, y, kelas = _label(df[t["label"]])
    g = grup(teks_dari(df, t["teks"])) if t.get("teks") else None
    bagi, ket = lipatan(t, df, y, g, klas)
    f = np.full(len(y), -1)
    for k, (_, b) in enumerate(bagi):
        f[b] = k
    kunci = t.get("id") if t.get("id") in df.columns else None
    out = pl.DataFrame({kunci or "baris": df[kunci] if kunci else np.arange(len(y)), "fold": f})
    path = Path(t["keluar"]) / "fold.csv"
    out.write_csv(path)
    lapor(f"validasi: {ket}; fold.csv {len(out)} baris")
    per = []
    for k in range(len(bagi)):
        m = f == k
        per.append({"fold": k, "n": int(m.sum()),
                    "sebaran": (np.bincount(y[m], minlength=len(kelas)) / max(m.sum(), 1)).tolist() if klas
                    else float(y[m].mean())})
    return {"validasi": ket, "path": str(path), "kunci": kunci or "baris", "per_fold": per, "klasifikasi": klas,
            "kelas": [str(k) for k in kelas] if klas else None, "label": t["label"],
            "sebaran_semua": (np.bincount(y, minlength=len(kelas)) / len(y)).tolist() if klas else float(y.mean()),
            "tanpa_fold": int((f < 0).sum())}


def bocor(t, df, uji):
    """Model kecil per SATU kolom (CV 3 fold): kolom yang sendirian hampir menebak label (atau satu kelasnya)
    sempurna patut dicurigai bocor (dibuat dari label, atau baru diketahui sesudah label)."""
    import lightgbm as lgb
    klas, y, kelas = _label(df[t["label"]])
    X, lewat = _matriks_fitur(df, [c for c in df.columns if c not in {t["label"], t.get("id")}])
    bagi, ket = lipatan(t, df, y, None, klas)
    bagi = bagi[:3]
    lapor(f"{X.shape[1]} kolom x {len(bagi)} fold; validasi {ket}")
    par = dict(n_estimators=60, learning_rate=0.2, num_leaves=15, min_child_samples=10, random_state=t["seed"],
               verbose=-1, n_jobs=min(8, os.cpu_count() or 1))
    if klas:
        acuan = f1_score(y, np.full(len(y), np.bincount(y).argmax()), average="macro")
    else:
        acuan = 0.0
    hasil = []
    for i, c in enumerate(X.columns, 1):
        p, dinilai = np.zeros(len(y), int if klas else float), np.zeros(len(y), bool)
        for a, b in bagi:
            m = (lgb.LGBMClassifier(**par) if klas else lgb.LGBMRegressor(**par)).fit(X[[c]].iloc[a], y[a])
            p[b], dinilai[b] = m.predict(X[[c]].iloc[b]), True
        yv, pv = y[dinilai], p[dinilai]
        if klas:
            per = f1_score(yv, pv, average=None, labels=range(len(kelas)), zero_division=0)
            hasil.append({"kolom": c, "skor": float(f1_score(yv, pv, average="macro")),
                          "kelas_maks": str(kelas[per.argmax()]), "f1_kelas_maks": float(per.max())})
        else:
            hasil.append({"kolom": c, "skor": float(r2_score(yv, pv))})
        if i % 5 == 0:
            lapor(f"{i}/{X.shape[1]} kolom")
    hasil.sort(key=lambda h: -h["skor"])
    return {"klasifikasi": klas, "validasi": ket, "acuan": float(acuan), "kolom": hasil, "dilewati": lewat,
            "label": t["label"], "metrik": "macro-F1" if klas else "R²"}


def _kode_kolom(s):
    """Kolom -> kode bilangan untuk Cramér's V: angka jadi ember desil, kategori 50 teratas, kosong = kode sendiri."""
    if s.dtype.is_numeric() and s.n_unique() > 10:
        tepi = np.unique(np.nanquantile(s.cast(pl.Float64).to_numpy(), np.linspace(0.1, 0.9, 9)))
        v = s.cast(pl.Float64).to_numpy()
        return np.where(np.isnan(v), -1, np.searchsorted(tepi, v, side="right"))
    v = s.cast(pl.String).fill_null("(kosong)")
    atas = set(v.value_counts(sort=True).head(50)[v.name].to_list())
    return np.unique(np.where(v.is_in(list(atas)).to_numpy(), v.to_numpy(), "(lainnya)"), return_inverse=True)[1]


def _cramer(a, b):
    ia, ib = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    m = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(m, (ia, ib), 1)
    if min(m.shape) < 2:
        return 0.0
    e = np.outer(m.sum(1), m.sum(0)) / m.sum()
    return float(np.sqrt(((m - e) ** 2 / e).sum() / m.sum() / (min(m.shape) - 1)))


def asosiasi(t, df, uji):
    """Peta asosiasi antar kolom (|Spearman| untuk angka-angka, Cramér's V selain itu; 0 = tak terkait,
    1 = sama) + korelasi pola kosong (kolom yang kosong bersamaan)."""
    d = df.sample(min(df.height, 20000), seed=t["seed"]) if df.height > 20000 else df
    kol = [c for c in d.columns if c != t.get("id") and not d.schema[c].is_nested()
           and not (d.schema[c] == pl.String and d[c].n_unique() > max(50, 0.5 * d.height))][:40]
    angka = {c for c in kol if d.schema[c].is_numeric() and d[c].n_unique() > 10}
    lapor(f"{len(kol)} kolom, {d.height} baris")
    kode = {c: _kode_kolom(d[c]) for c in kol}
    rank = d.select([pl.col(c).cast(pl.Float64).rank() for c in angka]) if angka else None
    m = np.eye(len(kol))
    for i, a in enumerate(kol):
        for j in range(i):
            b = kol[j]
            if a in angka and b in angka:
                v = rank.select(pl.corr(a, b)).item()
                v = abs(v) if v is not None and np.isfinite(v) else 0.0
            else:
                v = _cramer(kode[a], kode[b])
            m[i, j] = m[j, i] = v
    pasang = sorted(((m[i, j], kol[i], kol[j], "Spearman" if kol[i] in angka and kol[j] in angka else "Cramér's V")
                     for i in range(len(kol)) for j in range(i)), reverse=True)[:15]
    kos = [c for c in df.columns if df[c].null_count() and c in kol]
    mk = np.corrcoef(df.select([pl.col(c).is_null().cast(pl.Float64) for c in kos]).to_numpy().T) if len(kos) > 1 else None
    return {"kolom": kol, "matriks": m.round(3).tolist(), "pasangan": [(float(v), a, b, cara) for v, a, b, cara in pasang],
            "label": t.get("label"), "kosong": {"kolom": kos, "matriks": np.nan_to_num(mk).round(3).tolist()} if mk is not None
            else {"kolom": kos, "matriks": None}, "baris": d.height}


JENIS = {"tabular": tabular, "baseline": baseline, "adversarial": adversarial, "acak": acak, "kata": kata_khas,
         "fold": fold, "bocor": bocor, "asosiasi": asosiasi}


def main(path):
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    t.setdefault("seed", 42)
    t.setdefault("fold", 5)
    df = pl.read_parquet(t["latih"])
    uji = pl.read_parquet(t["uji"]) if t.get("uji") else None
    hasil = JENIS[t["jenis"]](t, df, uji)
    hasil["detik"] = round(time.perf_counter() - MULAI, 1)
    (Path(t["keluar"]) / "hasil.json").write_text(json.dumps(hasil, ensure_ascii=False, default=float),
                                                  encoding="utf-8")
    lapor(f"selesai dalam {hasil['detik']} s")


if __name__ == "__main__":
    main(sys.argv[1])
