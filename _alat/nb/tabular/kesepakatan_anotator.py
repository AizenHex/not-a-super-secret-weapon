# %% [markdown]
# # Tabular — Kesepakatan Antar Anotator (kappa, majority vote, Dawid-Skene)
# **Kapan dipakai:** tiap item dilabeli beberapa orang, dan labelnya tidak selalu sama.
#
# **Yang dihitung:**
# 1. Seberapa sepakat mereka: Cohen kappa per pasangan, Fleiss kappa untuk semua.
# 2. Label gabungan dengan majority vote (suara terbanyak).
# 3. Label gabungan dengan Dawid-Skene: ikut menaksir seberapa andal tiap anotator, sehingga
#    anotator buruk otomatis kurang didengar.
#
# **Data demo SINTETIS** (`anotasi_SINTETIS.csv`): 1.000 label SmSA asli, lima anotator
# simulasi dengan akurasi berbeda. Saat lomba, ganti sel "Buat data simulasi" dengan
# `pd.read_csv` tabel soal (satu baris per item, satu kolom per anotator, nilai = kode kelas
# 0, 1, 2, ...).

# %%
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/sentiment"   # GANTI saat lomba
SOURCE_CSV = f"{DATA_FOLDER}/smsa_train.csv"
ANNOTATION_CSV = "anotasi_SINTETIS.csv"

N_ITEMS = 1000
# akurasi tiap anotator simulasi (salah = memilih kelas lain secara acak)
ACCURACY = {"A": 0.90, "B": 0.85, "C": 0.75, "D": 0.60, "E": 0.50}
ANNOTATORS = ["A", "B", "C", "D", "E"]      # nama kolom anotator
N_ITERATIONS = 20                           # putaran Dawid-Skene
SEED = 42

# %% [markdown]
# ## Buat data simulasi (HAPUS saat lomba, ganti dengan tabel anotasi dari soal)
# `pd.factorize` mengubah label teks menjadi kode 0, 1, 2 (urut abjad).

# %%
random_generator = np.random.default_rng(SEED)
source_labels = pd.read_csv(SOURCE_CSV)["label"]
source_labels = source_labels.sample(N_ITEMS, random_state=SEED)
true_codes, class_names = pd.factorize(source_labels, sort=True)
n_classes = len(class_names)

simulation = pd.DataFrame()
simulation["asli"] = true_codes
for name in ANNOTATORS:
    is_wrong = random_generator.random(N_ITEMS) > ACCURACY[name]
    wrong_codes = (true_codes + random_generator.integers(1, n_classes, N_ITEMS)) % n_classes
    simulation[name] = np.where(is_wrong, wrong_codes, true_codes)
simulation.to_csv(ANNOTATION_CSV, index=False)

# %% [markdown]
# ## Baca data
# `labels` = tabel angka: baris = item, kolom = anotator.

# %%
annotations = pd.read_csv(ANNOTATION_CSV)
labels = annotations[ANNOTATORS].values
distinct_per_item = annotations[ANNOTATORS].nunique(axis=1)
print(annotations.shape, "| kelas:", list(class_names))
print("item yang semua anotator sepakat:", round((distinct_per_item == 1).mean(), 3))
annotations.head()

# %% [markdown]
# ## Cohen kappa per pasangan anotator
# 1 = sepakat penuh, 0 = sepakat sebatas kebetulan.

# %%
kappa = pd.DataFrame(1.0, index=ANNOTATORS, columns=ANNOTATORS)
for first, second in combinations(ANNOTATORS, 2):
    kappa.loc[first, second] = cohen_kappa_score(annotations[first], annotations[second])
    kappa.loc[second, first] = kappa.loc[first, second]
kappa.round(2)

# %% [markdown]
# ## Fleiss kappa (semua anotator sekaligus)
# Aturan kasar: < 0.2 buruk, 0.2–0.4 lemah, 0.4–0.6 sedang, 0.6–0.8 baik, > 0.8 sangat baik.
# `aggregate_raters`: baris = item, kolom = jumlah suara tiap kelas.

# %%
vote_counts, categories = aggregate_raters(labels)
print("Fleiss kappa:", round(fleiss_kappa(vote_counts, method="fleiss"), 3))

# %% [markdown]
# ## Label gabungan 1: majority vote
# Kalau seri, dipilih kelas dengan kode terkecil.

# %%
modes = annotations[ANNOTATORS].mode(axis=1)
majority_vote = modes[0].astype(int).values
print("akurasi majority vote:", round((majority_vote == annotations["asli"]).mean(), 3))

# %% [markdown]
# ## Label gabungan 2: Dawid-Skene — langkah awal
# - `initial_probabilities`: peluang awal tiap kelas per item = proporsi suara.
# - `estimate_confusion`: tiap anotator punya tabel konfusi sendiri:
#   `confusion[anotator, kelas_benar, label_diberikan]` = peluang memberi label itu.

# %%
def initial_probabilities(labels, n_classes):
    n_items, n_annotators = labels.shape
    probabilities = np.zeros((n_items, n_classes))
    for annotator in range(n_annotators):
        for item in range(n_items):
            probabilities[item, labels[item, annotator]] += 1
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def estimate_confusion(labels, probabilities, n_classes):
    n_annotators = labels.shape[1]
    confusion = np.full((n_annotators, n_classes, n_classes), 0.01)
    for annotator in range(n_annotators):
        for given in range(n_classes):
            rows = labels[:, annotator] == given
            confusion[annotator, :, given] += probabilities[rows].sum(axis=0)
    return confusion / confusion.sum(axis=2, keepdims=True)


# coba: tabel konfusi awal anotator pertama (diagonal besar = sering benar)
start_probabilities = initial_probabilities(labels, n_classes)
start_confusion = estimate_confusion(labels, start_probabilities, n_classes)
print(start_confusion[0].round(2))

# %% [markdown]
# ## Dawid-Skene — ulangi sampai stabil (EM)
# `update_probabilities`: peluang kelas tiap item dihitung ulang dari tabel konfusi semua
# anotator (dalam bentuk log agar angka kecil tidak menjadi 0).

# %%
def update_probabilities(labels, probabilities, confusion):
    log_probabilities = np.log(probabilities.mean(axis=0)) + np.zeros(probabilities.shape)
    for annotator in range(labels.shape[1]):
        given_labels = labels[:, annotator]
        log_probabilities += np.log(confusion[annotator][:, given_labels]).T
    log_probabilities = log_probabilities - log_probabilities.max(axis=1, keepdims=True)
    new_probabilities = np.exp(log_probabilities)
    return new_probabilities / new_probabilities.sum(axis=1, keepdims=True)


def dawid_skene(labels, n_classes):
    probabilities = initial_probabilities(labels, n_classes)
    for iteration in range(N_ITERATIONS):
        confusion = estimate_confusion(labels, probabilities, n_classes)
        probabilities = update_probabilities(labels, probabilities, confusion)
    return probabilities, confusion


ds_probabilities, ds_confusion = dawid_skene(labels, n_classes)
ds_labels = ds_probabilities.argmax(axis=1)
print("akurasi Dawid-Skene:", round((ds_labels == annotations["asli"]).mean(), 3))

# %% [markdown]
# ## Bandingkan: akurasi tiap anotator, majority vote, Dawid-Skene
# Kolom `taksiran_ds` = akurasi anotator yang ditaksir Dawid-Skene **tanpa** melihat label asli.

# %%
class_share = ds_probabilities.mean(axis=0)
rows = []
for number in range(len(ANNOTATORS)):
    name = ANNOTATORS[number]
    real_accuracy = (annotations[name] == annotations["asli"]).mean()
    estimated_accuracy = (class_share * np.diag(ds_confusion[number])).sum()
    rows.append([name, real_accuracy, estimated_accuracy])
rows.append(["majority_vote", (majority_vote == annotations["asli"]).mean(), None])
rows.append(["dawid_skene", (ds_labels == annotations["asli"]).mean(), None])

comparison = pd.DataFrame(rows, columns=["anotator", "akurasi", "taksiran_ds"])
comparison.round(3)

# %% [markdown]
# ## Simpan label gabungan
# `yakin_ds` kecil = item yang paling layak dicek ulang manual.

# %%
combined = pd.DataFrame()
combined["item"] = annotations.index
combined["majority_vote"] = class_names[majority_vote]
combined["dawid_skene"] = class_names[ds_labels]
combined["yakin_ds"] = ds_probabilities.max(axis=1)
combined.to_csv("label_gabungan.csv", index=False)
combined.sort_values("yakin_ds").head()
