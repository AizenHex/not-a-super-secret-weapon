# %% [markdown]
# # Visualisasi — Evaluasi Klasifikasi
# **Kapan dipakai:** "seberapa bagus modelnya? salahnya di mana? ambang berapa yang terbaik?
# apakah probabilitasnya bisa dipercaya?"
#
# **Isi:** confusion matrix (angka + %), ROC, PR, sebaran skor, F1 vs threshold, kalibrasi, F1 per
# kelas, learning curve, ROC interaktif. Semua gambar disimpan di folder notebook.
#
# **Data demo:** Mammography (biner, hanya 2% positif) dan Covertype (7 kelas). Model dilatih
# cepat di sel Model, LogisticRegression sebagai pembanding. Fokus notebook ini pada grafiknya.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/tabular_extra"   # GANTI saat lomba
BINARY_CSV = f"{DATA_FOLDER}/imbalanced/mammography_train.csv"
MULTI_CSV = f"{DATA_FOLDER}/multiclass/covertype_train.csv"

ID_COL = "id"
BINARY_TARGET = "label"
MULTI_TARGET = "Cover_Type"
SEED = 42

# %% [markdown]
# ## Gaya grafik (sama di semua notebook visualisasi)
# Font besar supaya terbaca di slide, palet tetap, garis tepi atas/kanan dimatikan.

# %%
PALETTE = ["#2E6FB7", "#F28E2B", "#3BA272", "#D1495B", "#7B5EA7", "#9AA0A6"]
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = PALETTE
plt.rcParams.update({
    "figure.figsize": (8, 5), "figure.dpi": 100, "savefig.dpi": 200,
    "font.size": 14, "axes.titlesize": 18, "axes.titleweight": "bold", "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})
sns.set_palette(PALETTE)

# %% [markdown]
# ## Data dan model biner (dilatih cepat, bukan fokus notebook ini)
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris LightGBM.
# LogisticRegression selalu dilatih sebagai pembanding.

# %%
binary = pd.read_csv(BINARY_CSV)
X = binary.drop(columns=[ID_COL, BINARY_TARGET])
y = binary[BINARY_TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y,
                                                  random_state=SEED)

model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED, verbose=-1)
# model = RandomForestClassifier(n_estimators=300, random_state=SEED)
model.fit(X_train, y_train)
baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
baseline.fit(X_train, y_train)

model_scores = model.predict_proba(X_val)[:, 1]
baseline_scores = baseline.predict_proba(X_val)[:, 1]
print("ROC-AUC model:", round(roc_auc_score(y_val, model_scores), 3))
print("ROC-AUC LogReg:", round(roc_auc_score(y_val, baseline_scores), 3))
curves = [("model", model_scores, BLUE), ("LogReg", baseline_scores, ORANGE)]

# %% [markdown]
# ## Data dan model multikelas

# %%
multi = pd.read_csv(MULTI_CSV)
X_multi = multi.drop(columns=[ID_COL, MULTI_TARGET])
y_multi = multi[MULTI_TARGET]
X_multi_train, X_multi_val, y_multi_train, y_multi_val = train_test_split(
    X_multi, y_multi, test_size=0.3, stratify=y_multi, random_state=SEED)
multi_model = LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=SEED, verbose=-1)
multi_model.fit(X_multi_train, y_multi_train)
multi_predictions = multi_model.predict(X_multi_val)
print("akurasi multikelas:", round((multi_predictions == y_multi_val).mean(), 3))

# %% [markdown]
# ## 1. Confusion matrix (angka + persen)
# **Kapan dipakai:** klasifikasi multikelas atau biner, melihat kelas mana tertukar dengan mana.
# **Cara baca:** baris = kelas asli, kolom = tebakan. Diagonal = benar. Persen dihitung per baris
# (recall kelas itu). Kotak non-diagonal yang gelap = kesalahan khas.

# %%
labels = sorted(y_multi.unique())
counts = confusion_matrix(y_multi_val, multi_predictions, labels=labels)
row_share = counts / counts.sum(axis=1, keepdims=True)
cell_text = np.empty(counts.shape, dtype=object)
for row in range(len(labels)):
    for column in range(len(labels)):
        cell_text[row, column] = f"{counts[row, column]}\n{row_share[row, column]:.0%}"
plt.figure(figsize=(9, 7))
sns.heatmap(row_share, annot=cell_text, fmt="", cmap="Blues", xticklabels=labels,
            yticklabels=labels, cbar_kws={"label": "proporsi per baris"}, annot_kws={"size": 10})
plt.title("Confusion matrix Covertype")
plt.xlabel("tebakan model")
plt.ylabel("kelas asli")
plt.grid(False)
# Alternatif cepat: ConfusionMatrixDisplay.from_predictions(y_multi_val, multi_predictions)
plt.savefig("evalklas_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Kurva ROC
# **Kapan dipakai:** membandingkan model biner tanpa memilih ambang, kelas relatif seimbang.
# **Cara baca:** makin dekat sudut kiri-atas makin baik, diagonal putus-putus = tebakan acak.
# AUC = peluang model memberi skor lebih tinggi ke positif daripada negatif.

# %%
for name, scores, color in curves:
    fpr, tpr, _ = roc_curve(y_val, scores)
    auc = roc_auc_score(y_val, scores)
    plt.plot(fpr, tpr, color=color, linewidth=3, label=f"{name} (AUC {auc:.3f})")
plt.plot([0, 1], [0, 1], color=GREY, linestyle="--")
plt.title("Kurva ROC")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate (recall)")
plt.legend(loc="lower right")
plt.savefig("evalklas_roc.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Kurva Precision-Recall
# **Kapan dipakai:** kelas positif langka (fraud, penyakit, anomali). ROC bisa terlihat bagus
# padahal PR jelek. **Cara baca:** makin ke kanan-atas makin baik, garis abu = tebakan acak
# (sama dengan proporsi positif). AP = luas di bawah kurva.

# %%
for name, scores, color in curves:
    precision, recall, _ = precision_recall_curve(y_val, scores)
    ap = average_precision_score(y_val, scores)
    plt.plot(recall, precision, color=color, linewidth=3, label=f"{name} (AP {ap:.3f})")
plt.axhline(y_val.mean(), color=GREY, linestyle="--", label="tebakan acak")
plt.title("Kurva Precision-Recall")
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
plt.savefig("evalklas_pr.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Sebaran skor per kelas
# **Kapan dipakai:** melihat seberapa terpisah skor positif dan negatif, dan di mana menaruh
# ambang. **Cara baca:** dua gunung yang jauh = model tegas, yang menumpuk = ragu. Skala log
# supaya kelas langka tetap terlihat.

# %%
class_names = y_val.map({0: "negatif", 1: "positif"})
sns.histplot(x=model_scores, hue=class_names, bins=30, palette=[GREY, RED], element="step",
             stat="count", common_norm=False)
plt.yscale("log")
plt.axvline(0.5, color="black", linestyle="--")
plt.title("Sebaran skor model per kelas asli")
plt.xlabel("probabilitas positif")
plt.ylabel("jumlah data (skala log)")
plt.savefig("evalklas_sebaran_skor.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. F1 / precision / recall terhadap threshold
# **Kapan dipakai:** memilih ambang keputusan (default 0.5 jarang terbaik pada data tak seimbang).
# **Cara baca:** puncak kurva F1 = ambang terbaik. Naikkan ambang = precision naik, recall turun.

# %%
precision, recall, thresholds = precision_recall_curve(y_val, model_scores)
precision = precision[:-1]   # elemen terakhir tidak punya threshold
recall = recall[:-1]
f1 = 2 * precision * recall / (precision + recall + 1e-9)
best = f1.argmax()
plt.plot(thresholds, precision, color=ORANGE, label="precision")
plt.plot(thresholds, recall, color=GREEN, label="recall")
plt.plot(thresholds, f1, color=BLUE, linewidth=3, label="F1")
plt.scatter(thresholds[best], f1[best], color=RED, zorder=3, s=80)
plt.annotate(f"terbaik: ambang {thresholds[best]:.2f}, F1 {f1[best]:.2f}",
             (thresholds[best], f1[best]), xytext=(0.45, 0.9), arrowprops={"arrowstyle": "->"})
plt.title("Memilih ambang keputusan")
plt.xlabel("threshold")
plt.ylabel("skor")
plt.legend(loc="center right")
plt.savefig("evalklas_f1_threshold.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Kurva kalibrasi
# **Kapan dipakai:** probabilitas dipakai langsung (ranking risiko, ekspektasi biaya, metrik
# logloss/Brier). **Cara baca:** titik di atas diagonal = model terlalu pesimis, di bawah =
# terlalu optimis. Kalau menyimpang, kalibrasi (CalibratedClassifierCV).

# %%
for name, scores, color in curves:
    actual_share, predicted_share = calibration_curve(y_val, scores, n_bins=8,
                                                      strategy="quantile")
    plt.plot(predicted_share, actual_share, marker="o", color=color, linewidth=2, label=name)
plt.plot([0, 1], [0, 1], color=GREY, linestyle="--", label="sempurna")
plt.title("Kurva kalibrasi")
plt.xlabel("probabilitas prediksi (rata-rata per bin)")
plt.ylabel("proporsi positif sebenarnya")
plt.legend()
plt.savefig("evalklas_kalibrasi.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. F1 per kelas
# **Kapan dipakai:** multikelas, terutama bila kelas tak seimbang. Akurasi total bisa menipu.
# **Cara baca:** bar merah = kelas terlemah. Angka di ujung bar = jumlah data validasi kelas itu.

# %%
report = classification_report(y_multi_val, multi_predictions, output_dict=True)
report_table = pd.DataFrame(report).T
report_table = report_table.loc[labels]          # buang baris accuracy / macro avg
report_table = report_table.sort_values("f1-score")
lowest_f1 = report_table["f1-score"].min()
colors = []
for value in report_table["f1-score"]:
    if value == lowest_f1:
        colors.append(RED)
    else:
        colors.append(BLUE)
plt.barh(report_table.index, report_table["f1-score"], color=colors)
for i in range(len(report_table)):
    value = report_table["f1-score"].iloc[i]
    support = int(report_table["support"].iloc[i])
    plt.text(value + 0.01, i, f"{value:.2f}  (n={support})", va="center")
plt.xlim(0, 1.2)
plt.title("F1 per kelas")
plt.xlabel("F1-score")
plt.savefig("evalklas_f1_per_kelas.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Learning curve
# **Kapan dipakai:** memutuskan perlu data lebih banyak atau model lebih kompleks.
# **Cara baca:** jarak besar antara latih dan validasi = overfit. Keduanya rendah dan berdekatan
# = underfit. Validasi masih naik di ujung kanan = tambahan data masih membantu.

# %%
sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=3, scoring="roc_auc", train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=1)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)
plt.plot(sizes, train_mean, marker="o", color=ORANGE, label="latih")
plt.plot(sizes, val_mean, marker="o", color=BLUE, label="validasi")
plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, color=BLUE, alpha=0.2)
plt.title("Learning curve (ROC-AUC)")
plt.xlabel("jumlah data latih")
plt.ylabel("ROC-AUC")
plt.legend()
plt.savefig("evalklas_learning_curve.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. ROC interaktif (plotly)
# **Kapan dipakai:** diskusi memilih ambang bersama tim/juri. **Cara baca:** arahkan kursor ke
# kurva untuk melihat threshold, FPR, dan TPR di titik itu.

# %%
fpr, tpr, roc_thresholds = roc_curve(y_val, model_scores)
roc_table = pd.DataFrame()
roc_table["fpr"] = fpr
roc_table["tpr"] = tpr
roc_table["threshold"] = roc_thresholds
fig = px.line(roc_table, x="fpr", y="tpr", hover_data=["threshold"],
              title="ROC model (interaktif)")
fig.update_traces(line_color=BLUE)
fig.write_html("evalklas_roc_interaktif.html", include_plotlyjs=True)
fig.show()
