# %% [markdown]
# # 5m. Ensemble + laporan MULTILABEL (gabung OOF `3m` dan `4m`, bahan slide aman NDA)
# **Kapan dipakai:** sesudah `3m_baseline_multilabel` dan/atau `4m_finetune_multilabel` menulis OOF
# ke `oof_multilabel/`. Tanpa GPU, beberapa detik.
#
# **Alur:** muat semua OOF → skor jujur tiap model (ambang dicari di lipatan lain) → rata-rata
# probabilitas semua model → gabungan dipakai hanya bila naik ≥ `MIN_GAIN` → ambang akhir →
# submission + validator → laporan per label, jumlah label per teks, label yang muncul bersama →
# `fakta_multilabel.md` (hanya angka agregat).

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

# %% [markdown]
# ## Konfigurasi

# %%
MODEL_NAMES = None               # None = semua oof_*.npy yang cocok, atau ["ml_tfidf_logreg", ...]
MIN_GAIN = 0.002                 # gabungan dipakai hanya bila skor jujur naik sekurangnya ini
MIN_ONE_LABEL = False            # True: tiap teks minimal 1 label
N_FOLDS = 5
SEED = 42
SUBMISSION_NAME = "submission.csv"

# ---- final-pipeline: nilai soal dibaca dari konfig.py (ubah DI SANA, bukan di sini) ----
exec(open("konfig.py", encoding="utf-8").read())
if LABEL_COLS is None:
    raise ValueError("Isi LABEL_COLS di konfig.py (daftar kolom label 0/1) untuk soal multilabel")
TRAIN_CSV = f"{HASIL}/praproses/train_bersih.csv"
TEST_CSV = f"{HASIL}/praproses/test_bersih.csv"
OUTPUT_FOLDER = f"{HASIL}/ensemble_multilabel"
OOF_FOLDER = f"{HASIL}/oof_multilabel"
SAMPLE_CSV = f"{OUTPUT_FOLDER}/sample_submission_demo.csv"
if SAMPLE_PANITIA is not None:
    SAMPLE_CSV = SAMPLE_PANITIA

# %% [markdown]
# ## Metrik
# Samakan dengan `3m` dan `4m`.

# %%
METRIC_NAME = "f1_macro"         # GANTI saat lomba, ikut baris return di bawah


def score(truth, predicted):
    return f1_score(truth, predicted, average="macro", zero_division=0)
    # return f1_score(truth, predicted, average="micro", zero_division=0)
    # return f1_score(truth, predicted, average="samples", zero_division=0)


# %% [markdown]
# ## Baca data dan OOF

# %%
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
truth = train[LABEL_COLS].to_numpy().astype(int)
demo_sample = pd.DataFrame()
demo_sample[ID_COL] = test[ID_COL]
for column in LABEL_COLS:
    demo_sample[column] = 0
if SAMPLE_CSV.startswith(OUTPUT_FOLDER):
    demo_sample.to_csv(SAMPLE_CSV, index=False)
sample = pd.read_csv(SAMPLE_CSV)

oofs, tests = {}, {}
for path in sorted(glob.glob(f"{OOF_FOLDER}/oof_*.npy")):
    name = os.path.basename(path)[4:-4]
    test_path = f"{OOF_FOLDER}/test_{name}.npy"
    if MODEL_NAMES is not None and name not in MODEL_NAMES:
        continue
    if not os.path.exists(test_path):
        print("dilewati (tanpa file test):", name)
        continue
    oof_part, test_part = np.load(path), np.load(test_path)
    if oof_part.shape != truth.shape or test_part.shape != (len(test), len(LABEL_COLS)):
        print("dilewati (bentuk tidak cocok):", name, oof_part.shape)
        continue
    oofs[name], tests[name] = oof_part.astype("float64"), test_part.astype("float64")
if not oofs:
    raise ValueError("Tidak ada OOF yang cocok: jalankan 3m / 4m dulu, cek LABEL_COLS dan HASIL")
print("model:", list(oofs))

# %% [markdown]
# ## Ambang per label dan skor jujur
# Skor jujur: ambang dicari di 4 lipatan, dinilai di lipatan ke-5 (berbeda dari skor "dalam
# sampel" yang ambangnya dicari dan dinilai di data yang sama).

# %%
GRID = np.arange(0.05, 0.96, 0.05)


def find_thresholds(labels, proba):
    cuts = np.full(proba.shape[1], 0.5)
    for column in range(proba.shape[1]):
        best = -1.0
        for cut in GRID:
            value = f1_score(labels[:, column], proba[:, column] >= cut, zero_division=0)
            if value > best:
                best, cuts[column] = value, cut
    return cuts


def to_labels(proba, cuts):
    labels = (proba >= cuts).astype(int)
    if MIN_ONE_LABEL:
        empty = labels.sum(axis=1) == 0
        labels[empty, proba[empty].argmax(axis=1)] = 1
    return labels


def honest_score(proba):
    predicted = np.zeros(truth.shape, dtype=int)
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 1)
    for fit_index, val_index in splitter.split(proba):
        cuts = find_thresholds(truth[fit_index], proba[fit_index])
        predicted[val_index] = to_labels(proba[val_index], cuts)
    return score(truth, predicted)


candidates = dict(oofs)
candidate_tests = dict(tests)
if len(oofs) > 1:
    candidates["rata-rata semua"] = np.mean(list(oofs.values()), axis=0)
    candidate_tests["rata-rata semua"] = np.mean(list(tests.values()), axis=0)
rows = []
for name, proba in candidates.items():
    in_sample = score(truth, to_labels(proba, find_thresholds(truth, proba)))
    rows.append({"cara": name, "skor jujur": round(honest_score(proba), 4),
                 "dalam sampel": round(in_sample, 4), "@0.5": round(score(truth, proba >= 0.5), 4)})
comparison = pd.DataFrame(rows).sort_values("skor jujur", ascending=False).reset_index(drop=True)
comparison.to_csv(f"{OUTPUT_FOLDER}/banding_model.csv", index=False)
comparison

# %% [markdown]
# ## Putusan

# %%
singles = comparison[comparison["cara"] != "rata-rata semua"]
chosen = singles.iloc[0]["cara"]
chosen_score = singles.iloc[0]["skor jujur"]
blend = comparison[comparison["cara"] == "rata-rata semua"]
if len(blend) and blend.iloc[0]["skor jujur"] >= chosen_score + MIN_GAIN:
    chosen, chosen_score = "rata-rata semua", blend.iloc[0]["skor jujur"]
final_oof, final_test = candidates[chosen], candidate_tests[chosen]
thresholds = find_thresholds(truth, final_oof)
oof_labels = to_labels(final_oof, thresholds)
print(f"DIPAKAI: {chosen} | {METRIC_NAME} jujur {chosen_score:.4f}")

# %% [markdown]
# ## Submission dan validator

# %%
test_labels = pd.DataFrame(to_labels(final_test, thresholds), columns=LABEL_COLS)
test_labels.index = test[ID_COL].to_numpy()
submission = pd.DataFrame()
submission[ID_COL] = sample[ID_COL]
for column in sample.columns:
    if column != ID_COL:
        submission[column] = sample[ID_COL].map(test_labels[column]).to_numpy()
submission.to_csv(f"{OUTPUT_FOLDER}/{SUBMISSION_NAME}", index=False)

written = pd.read_csv(f"{OUTPUT_FOLDER}/{SUBMISSION_NAME}")
problems = []
if written.columns.tolist() != sample.columns.tolist():
    problems.append("kolom / urutan kolom beda")
if written[ID_COL].tolist() != sample[ID_COL].tolist():
    problems.append("id / urutan id beda dari sample")
for column in LABEL_COLS:
    if written[column].isna().any():
        problems.append(f"{column}: ada nilai kosong")
    if not set(written[column].unique()) <= {0, 1}:
        problems.append(f"{column}: nilai selain 0/1")
for problem in problems:
    print("-", problem)
if SAMPLE_CSV.startswith(OUTPUT_FOLDER):
    print("PERINGATAN: SAMPLE_CSV masih file demo, format panitia belum dicek")
print("MASALAH:", len(problems))

# %% [markdown]
# ## Laporan per label (dari OOF model terpilih)

# %%
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
per_label = pd.DataFrame({
    "label": LABEL_COLS,
    "positif train": truth.sum(axis=0),
    "porsi %": (100 * truth.mean(axis=0)).round(1),
    "ambang": thresholds.round(2),
    "precision": precision_score(truth, oof_labels, average=None, zero_division=0).round(3),
    "recall": recall_score(truth, oof_labels, average=None, zero_division=0).round(3),
    "F1": f1_score(truth, oof_labels, average=None, zero_division=0).round(3),
    "positif prediksi test": test_labels.sum(axis=0).to_numpy(),
})
per_label.to_csv(f"{OUTPUT_FOLDER}/laporan_per_label.csv", index=False)

figure, axis = plt.subplots(figsize=(max(8, 0.9 * len(LABEL_COLS) + 3), 4.5))
position = np.arange(len(LABEL_COLS))
axis.bar(position - 0.27, per_label["precision"], 0.27, label="precision", color="#4c78a8")
axis.bar(position, per_label["recall"], 0.27, label="recall", color="#f58518")
axis.bar(position + 0.27, per_label["F1"], 0.27, label="F1", color="#54a24b")
axis.set_xticks(position, LABEL_COLS, rotation=30, ha="right")
axis.set_ylim(0, 1.05)
axis.grid(axis="y", linestyle="--", alpha=0.4)
axis.set_axisbelow(True)
axis.legend(ncol=3, frameon=False)
axis.set_title(f"Skor OOF per label ({chosen})")
figure.savefig(f"{OUTPUT_FOLDER}/01_skor_per_label.png")
plt.show()
per_label

# %% [markdown]
# ## Jumlah label per teks dan label yang muncul bersama

# %%
count_table = pd.DataFrame({
    "train (label asli)": pd.Series(truth.sum(axis=1)).value_counts(normalize=True),
    "train (prediksi OOF)": pd.Series(oof_labels.sum(axis=1)).value_counts(normalize=True),
    "test (prediksi)": pd.Series(test_labels.sum(axis=1).to_numpy()).value_counts(normalize=True),
}).fillna(0).sort_index()
count_table.index.name = "jumlah label"
count_table.round(3).reset_index().to_csv(f"{OUTPUT_FOLDER}/jumlah_label_per_teks.csv", index=False)
axis = count_table.plot.bar(figsize=(8, 4.2), color=["#4c78a8", "#f58518", "#54a24b"], rot=0)
axis.set_ylabel("porsi teks")
axis.grid(axis="y", linestyle="--", alpha=0.4)
axis.set_axisbelow(True)
axis.set_title("Jumlah label per teks")
axis.figure.savefig(f"{OUTPUT_FOLDER}/02_jumlah_label_per_teks.png")
plt.show()

together = truth.T @ truth                       # [i, j] = teks train yang punya label i DAN j
given = together / np.maximum(truth.sum(axis=0)[:, None], 1)   # porsi teks berlabel i yang juga j
figure, axis = plt.subplots(figsize=(0.9 * len(LABEL_COLS) + 3, 0.7 * len(LABEL_COLS) + 2))
axis.imshow(given, cmap="Blues", vmin=0, vmax=1)
axis.set_xticks(position, LABEL_COLS, rotation=30, ha="right")
axis.set_yticks(position, LABEL_COLS)
for row in range(len(LABEL_COLS)):
    for column in range(len(LABEL_COLS)):
        axis.text(column, row, f"{100 * given[row, column]:.0f}%", ha="center", va="center",
                  fontsize=8, color=["black", "white"][int(given[row, column] > 0.6)])
axis.set_title("Dari teks berlabel (baris), berapa % juga berlabel (kolom)")
figure.savefig(f"{OUTPUT_FOLDER}/03_label_bersama.png")
pd.DataFrame(given.round(3), index=LABEL_COLS, columns=LABEL_COLS).reset_index().to_csv(
    f"{OUTPUT_FOLDER}/label_bersama.csv", index=False)
plt.show()

# %% [markdown]
# ## Fakta untuk slide

# %%
best_label = per_label.sort_values("F1").iloc[-1]
worst_label = per_label.sort_values("F1").iloc[0]
lines = [
    "# Fakta klasifikasi multilabel",
    "",
    f"- {len(train)} teks latih, {len(test)} teks uji, {len(LABEL_COLS)} label: {', '.join(LABEL_COLS)}.",
    f"- Rata-rata {truth.sum(axis=1).mean():.2f} label per teks; "
    f"{100 * (truth.sum(axis=1) == 0).mean():.1f}% teks tanpa label, "
    f"{100 * (truth.sum(axis=1) > 1).mean():.1f}% teks dengan lebih dari satu label.",
    f"- Model dibandingkan dengan {METRIC_NAME} (ambang dicari di lipatan lain):",
]
for _, item in comparison.iterrows():
    lines.append(f"  - {item['cara']}: {item['skor jujur']:.4f} (ambang 0,5: {item['@0.5']:.4f})")
lines += [
    f"- Dipakai: {chosen}, {METRIC_NAME} {chosen_score:.4f}.",
    f"- Label termudah: {best_label['label']} (F1 {best_label['F1']:.3f}, {best_label['porsi %']}% teks); "
    f"tersulit: {worst_label['label']} (F1 {worst_label['F1']:.3f}, {worst_label['porsi %']}% teks).",
    f"- Ambang per label antara {thresholds.min():.2f} dan {thresholds.max():.2f} (bukan 0,5 tetap).",
    "",
    "```",
    per_label.to_string(index=False),
    "```",
]
with open(f"{OUTPUT_FOLDER}/fakta_multilabel.md", "w", encoding="utf-8") as report_file:
    report_file.write("\n".join(lines) + "\n")
print("\n".join(lines))
