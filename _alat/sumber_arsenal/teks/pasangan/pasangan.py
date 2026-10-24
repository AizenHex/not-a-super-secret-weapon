# %% [markdown]
# # Pasangan teks: apakah kalimat A dan B sama makna / B tersimpul dari A / duplikat?
#
# **Butuh:** `train` (kolom kalimat A, kalimat B, label), `test` (A, B, id).
# **Menghasilkan:** tabel fitur pasangan, `submission.csv`.
#
# Contoh soal: entailment (B tersimpul dari A), parafrase, pertanyaan duplikat, cocokkan
# judul–isi, cek klaim vs fakta. Tiga jenis fitur, digabung lalu dimasukkan ke model tabel:
#
# | fitur | menangkap | sel |
# |---|---|---|
# | leksikal: kata sama (Jaccard), selisih panjang, fuzzy, angka beda | salinan / sedikit ubah | 2 |
# | embedding: cosine, \|a−b\|, a·b (Reimers & Gurevych 2019) | makna mirip | 3 |
# | NLI pretrained (mDeBERTa XNLI): proba entail/netral/kontradiksi, dua arah | tersimpul | 4 |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi | **ya** |
# | 2 | `lexical_features` | **ya** |
# | 3 | fitur embedding | disarankan |
# | 4 | fitur NLI (mDeBERTa) | soal entailment/kontradiksi |
# | 5 | bandingkan kombinasi fitur dengan cross-validation | **ya** |
# | 6 | latih SEMUA → `submission.csv` | **ya** |
#
# Paling akurat (kalau waktu ada): fine-tune cross-encoder = `teks/klasifikasi/finetune_bert`
# dengan `tokenizer(teks_A, teks_B, ...)` (dua kalimat masuk bersama). IndoNLU: IndoBERT-base
# F1 78,6 di WReTE.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# WReTE: pasangan kalimat Wikipedia, label `Entail_or_Paraphrase` / `NotEntail`.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/text/pair/wrete_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/text/pair/wrete_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/text/pair/wrete_test_labels.csv")
print(train.shape, test.shape)
print(train["label"].value_counts())

# %% [markdown]
# ## 1. Konfigurasi

# %%
import torch

PRETRAINED_FOLDER = "pretrained"   # UBAH
TEXT_A = "sent_A"                  # UBAH
TEXT_B = "sent_B"                  # UBAH
TARGET = "label"                   # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for column in [TEXT_A, TEXT_B]:
    train[column] = train[column].fillna("").astype(str)
    test[column] = test[column].fillna("").astype(str)
print(train[[TEXT_A, TEXT_B]].iloc[0].tolist())

# %% [markdown]
# ## 2. `lexical_features(table)` → fitur kata/huruf
# Angka beda (`tahun 1945` vs `1950`) sering jadi tanda TIDAK tersimpul.

# %%
import re

import pandas as pd
from rapidfuzz import fuzz


def lexical_row(text_a, text_b):
    words_a, words_b = set(text_a.lower().split()), set(text_b.lower().split())
    numbers_a, numbers_b = set(re.findall(r"\d+", text_a)), set(re.findall(r"\d+", text_b))
    shared = len(words_a & words_b)
    return {"jaccard": shared / max(len(words_a | words_b), 1),
            "b_in_a": shared / max(len(words_b), 1),          # porsi kata B yang ada di A
            "length_ratio": len(text_b) / max(len(text_a), 1),
            "fuzzy": fuzz.token_set_ratio(text_a, text_b) / 100,
            "new_numbers": len(numbers_b - numbers_a)}


def lexical_features(table):
    rows = []
    for text_a, text_b in zip(table[TEXT_A], table[TEXT_B]):
        rows.append(lexical_row(text_a, text_b))
    return pd.DataFrame(rows)


train_lexical, test_lexical = lexical_features(train), lexical_features(test)
lexical_means = train_lexical.groupby(train[TARGET]).mean()
print(lexical_means.round(3))   # rata-rata fitur per label: makin beda makin berguna

# %% [markdown]
# ## 3. Fitur embedding → `train_embedding`, `test_embedding`
# Vektor A dan B (norma 1) → [cosine, |a−b|, a·b]. 1 + 384 + 384 kolom.

# %%
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer(f"{PRETRAINED_FOLDER}/minilm", device=DEVICE)


def embedding_features(table):
    vectors_a = encoder.encode(table[TEXT_A].tolist(), batch_size=64, normalize_embeddings=True)
    vectors_b = encoder.encode(table[TEXT_B].tolist(), batch_size=64, normalize_embeddings=True)
    cosine = (vectors_a * vectors_b).sum(axis=1, keepdims=True)
    return np.hstack([cosine, np.abs(vectors_a - vectors_b), vectors_a * vectors_b])


train_embedding, test_embedding = embedding_features(train), embedding_features(test)
cosine = pd.Series(train_embedding[:, 0])
cosine_means = cosine.groupby(train[TARGET].to_numpy()).mean()
print(train_embedding.shape, "| cosine rata-rata per label:\n", cosine_means.round(3))

# %% [markdown]
# ## 4. Fitur NLI → `train_nli`, `test_nli` (6 kolom)
# Model NLI menebak hubungan premis → hipotesis: entailment / neutral / contradiction.
# Dihitung dua arah (A→B dan B→A); parafrase = entail dua arah.

# %%
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

nli_tokenizer = AutoTokenizer.from_pretrained(f"{PRETRAINED_FOLDER}/nli_mdeberta")
nli = AutoModelForSequenceClassification.from_pretrained(f"{PRETRAINED_FOLDER}/nli_mdeberta")
nli = nli.to(DEVICE)
nli.eval()
print(nli.config.id2label)


def nli_proba(premises, hypotheses, batch_size=32):
    result = []
    for start in range(0, len(premises), batch_size):
        tokens = nli_tokenizer(premises[start:start + batch_size],
                               hypotheses[start:start + batch_size], padding=True,
                               truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = nli(**tokens.to(DEVICE)).logits
        probability = torch.softmax(logits, dim=1)
        result.append(probability.cpu().numpy())
    return np.concatenate(result)


print(nli_proba(["Budi makan nasi goreng."], ["Budi sedang makan."]).round(3))

# %% [markdown]
# Terapkan ke SEMUA train dan test (dua arah).

# %%
import numpy as np


def nli_features(table):
    texts_a, texts_b = table[TEXT_A].tolist(), table[TEXT_B].tolist()
    return np.hstack([nli_proba(texts_a, texts_b), nli_proba(texts_b, texts_a)])


train_nli, test_nli = nli_features(train), nli_features(test)
print(train_nli.shape)

# %% [markdown]
# ## 5. Bandingkan kombinasi fitur (cross-validation 5 fold) → `FEATURE_SET`
# Kalimat A yang sama muncul di banyak pasangan? Ganti ke `GroupKFold` dengan grup = A
# (supaya val tidak "kenal" kalimatnya).

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

feature_sets = {
    "leksikal": (train_lexical.to_numpy(), test_lexical.to_numpy()),
    "leksikal+nli": (np.hstack([train_lexical, train_nli]), np.hstack([test_lexical, test_nli])),
    "semua": (np.hstack([train_lexical, train_nli, train_embedding]),
              np.hstack([test_lexical, test_nli, test_embedding])),
}   # UBAH
y = train[TARGET].to_numpy()
pair_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000,
                                                                class_weight="balanced"))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, (features, _) in feature_sets.items():
    scores = cross_val_score(pair_model, features, y, cv=folds, scoring="f1_macro")
    print(f"{name:14s} F1 macro: {scores.mean():.4f} ± {scores.std():.4f}")

# %% [markdown]
# ## 6. Latih di SEMUA train dengan kombinasi terbaik → `submission.csv`

# %%
import pandas as pd

FEATURE_SET = "leksikal+nli"   # UBAH: nama terbaik dari sel 5
X, X_test = feature_sets[FEATURE_SET]
pair_model.fit(X, y)
submission = pd.DataFrame()
submission["id"] = test["id"]                      # UBAH
submission["label"] = pair_model.predict(X_test)   # UBAH
submission.to_csv("submission.csv", index=False)
print(submission["label"].value_counts())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
from sklearn.metrics import f1_score

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
print("F1 macro test:", round(f1_score(test_labels["label"], submission["label"],
                                       average="macro"), 4))
