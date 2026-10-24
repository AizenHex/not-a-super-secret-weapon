# %% [markdown]
# # Pencarian teks (retrieval), jawab pertanyaan dari dokumen, dan teks kembar
#
# **Butuh:** `documents` (kumpulan teks yang dicari), `queries` (pertanyaan / teks pencarian).
# **Menghasilkan:** top-k dokumen per query, jawaban pendek dari dokumen, pasangan teks kembar.
#
# Contoh soal: cari pasal/FAQ yang relevan, cocokkan keluhan ke kategori jawaban, sistem
# tanya-jawab dari kumpulan dokumen (RAG tanpa LLM), deteksi plagiat / berita duplikat.
#
# | cara | menangkap | catatan |
# |---|---|---|
# | BM25 (sel 2) | kata yang sama persis | kuat untuk nama, angka, istilah |
# | embedding MiniLM (sel 3) | makna mirip, sinonim | kuat untuk parafrase |
# | gabungan RRF (sel 4) | keduanya | Cormack dkk. 2009 (k = 60), dipakai Elasticsearch |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + `tokenize` + `ranking_scores` (recall@k, MRR) | **ya** |
# | 2 | BM25 → `bm25_top` | pilih ≥1 |
# | 3 | embedding → `dense_top` | pilih ≥1 |
# | 4 | gabungan RRF berbobot → `hybrid_top` + cari bobot | disarankan |
# | 5 | `answer_question`: ambil potongan jawaban dari dokumen (mDeBERTa QA) | soal tanya-jawab |
# | 6 | teks kembar / hampir kembar | soal duplikat, bersih data |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# FacQA: pertanyaan + paragraf konteks + jawaban. Dokumen = semua paragraf unik; tugasnya
# mencari paragraf yang benar untuk tiap pertanyaan (`gold`).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

qa = pd.read_csv(f"{DEMO_FOLDER}/text/qa/facqa_train.csv")
documents = pd.DataFrame({"text": qa["konteks"].unique()})
documents["doc_id"] = range(len(documents))
queries = qa[["pertanyaan", "jawaban"]].copy()
queries["gold"] = qa["konteks"].map(dict(zip(documents["text"], documents["doc_id"])))
print(len(documents), "dokumen |", len(queries), "pertanyaan")

# %% [markdown]
# ## 1. Konfigurasi + `tokenize` + `ranking_scores(top_lists, gold)`
# `ranking_scores`: recall@k = porsi query yang dokumen benarnya ada di k teratas; MRR =
# rata-rata 1/posisi dokumen benar (1 = selalu di posisi pertama).

# %%
import re

import numpy as np

DOC_TEXT = "text"           # UBAH
QUERY_TEXT = "pertanyaan"   # UBAH
TOP_K = 10                  # UBAH


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def ranking_scores(top_lists, gold):
    hits = {1: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    for top, answer in zip(top_lists, gold):
        found = np.where(np.asarray(top) == answer)[0]
        if len(found) > 0:
            reciprocal += 1 / (found[0] + 1)
            for top_k in hits:
                hits[top_k] += found[0] < top_k
    result = {}
    for top_k in hits:
        result[f"recall@{top_k}"] = round(float(hits[top_k]) / len(gold), 4)
    result["MRR"] = round(float(reciprocal) / len(gold), 4)
    return result


print(tokenize("Di pulau manakah, PT Mutiara berlokasi?"))

# %% [markdown]
# ## 2. BM25 → `bm25_top` (nomor dokumen, urut paling relevan)
# BM25 = TF-IDF yang lebih baik untuk pencarian (memperhitungkan panjang dokumen).

# %%
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

tokenized_documents = []
for text in documents[DOC_TEXT]:
    tokenized_documents.append(tokenize(text))
bm25 = BM25Okapi(tokenized_documents)   # UBAH: k1=1.5, b=0.75 (bawaan)

bm25_top = []
for query in tqdm(queries[QUERY_TEXT]):
    scores = bm25.get_scores(tokenize(query))
    bm25_top.append(np.argsort(-scores)[:TOP_K])
print("BM25:", ranking_scores(bm25_top, queries["gold"]))

# %% [markdown]
# ## 3. Embedding MiniLM → `dense_top`
# Vektor dokumen dihitung sekali; query × dokumen = cosine (vektor norma 1).
# Dokumen panjang (> 256 token) terpotong: pecah dulu jadi paragraf / 3–5 kalimat.

# %%
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PRETRAINED_FOLDER = "pretrained"   # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SentenceTransformer(f"{PRETRAINED_FOLDER}/minilm", device=DEVICE)
document_vectors = encoder.encode(documents[DOC_TEXT].tolist(), batch_size=64,
                                  normalize_embeddings=True, show_progress_bar=True)
query_vectors = encoder.encode(queries[QUERY_TEXT].tolist(), batch_size=64,
                               normalize_embeddings=True)
similarity = query_vectors @ document_vectors.T          # (query, dokumen)
dense_top = np.argsort(-similarity, axis=1)[:, :TOP_K]
print("embedding:", ranking_scores(dense_top, queries["gold"]))

# %% [markdown]
# ## 4. Gabungan RRF berbobot → `hybrid_top`
# Skor dokumen = Σ bobot / (60 + peringkat) dari tiap metode. Hanya memakai peringkat, jadi
# skor BM25 (tak terbatas) dan cosine (−1..1) tidak perlu disamakan skalanya.
# Bobot sama TIDAK selalu menang: kalau satu metode jauh lebih lemah, ia menarik turun
# (Bruch dkk. 2023) → cari bobot di query yang punya kunci (mis. 200 query berlabel sendiri).
# Di data contoh, pertanyaan menyalin kata dari paragraf → BM25 saja menang (bobot 0).
# Query berupa parafrase / bahasa sehari-hari vs dokumen formal → embedding naik perannya.

# %%
import numpy as np


def reciprocal_rank_fusion(rank_lists, weights, rank_constant=60):
    scores = {}
    for ranking, weight in zip(rank_lists, weights):
        for position, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + weight / (rank_constant + position + 1)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return np.array(ordered[:TOP_K])


def fuse_all(dense_weight):
    result = []
    for bm25_ranking, dense_ranking in zip(bm25_top, dense_top):
        result.append(reciprocal_rank_fusion([bm25_ranking, dense_ranking],
                                             [1 - dense_weight, dense_weight]))
    return result


best_weight, best_mrr = 0.0, -1.0
for dense_weight in [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:   # UBAH
    scores = ranking_scores(fuse_all(dense_weight), queries["gold"])
    print(f"bobot embedding {dense_weight:.1f}:", scores)
    if scores["MRR"] > best_mrr:
        best_weight, best_mrr = dense_weight, scores["MRR"]
print("bobot terpilih:", best_weight)
hybrid_top = fuse_all(best_weight)

# %% [markdown]
# Hasil pencarian → tabel (query, peringkat, dokumen) untuk dicek / dikumpulkan.

# %%
import pandas as pd

rows = []
for query_index, top in enumerate(hybrid_top[:3]):
    for rank, doc in enumerate(top[:3]):
        rows.append({"query": queries[QUERY_TEXT].iloc[query_index], "rank": rank + 1,
                     "doc_id": doc, "document": documents[DOC_TEXT].iloc[doc][:80]})
print(pd.DataFrame(rows).to_string())

# %% [markdown]
# ## 5. `answer_question(question, context)` → potongan jawaban dari dokumen
# Model QA menebak posisi awal dan akhir jawaban di dalam konteks (extractive QA).
# Dipakai setelah retrieval: pertanyaan → dokumen teratas → jawaban.

# %%
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MAX_ANSWER_TOKENS = 30   # UBAH
qa_tokenizer = AutoTokenizer.from_pretrained(f"{PRETRAINED_FOLDER}/qa_mdeberta")
qa_model = AutoModelForQuestionAnswering.from_pretrained(f"{PRETRAINED_FOLDER}/qa_mdeberta")
qa_model = qa_model.to(DEVICE)
qa_model.eval()


def best_span(start_logits, end_logits, sequence_ids):
    best, best_score = (0, 0), -1e9
    for start in range(len(sequence_ids)):
        if sequence_ids[start] != 1:                 # 1 = token konteks
            continue
        for end in range(start, min(start + MAX_ANSWER_TOKENS, len(sequence_ids))):
            if sequence_ids[end] != 1:
                break
            if start_logits[start] + end_logits[end] > best_score:
                best, best_score = (start, end), start_logits[start] + end_logits[end]
    return best


print("siap")

# %% [markdown]
# `answer_question` + coba pada beberapa pertanyaan (dokumen dari RRF peringkat 1).

# %%
import torch


def answer_question(question, context):
    tokens = qa_tokenizer(question, context, truncation="only_second", max_length=384,
                          return_offsets_mapping=True, return_tensors="pt")
    offsets = tokens.pop("offset_mapping")[0].tolist()
    sequence_ids = tokens.sequence_ids(0)
    with torch.no_grad():
        output = qa_model(**tokens.to(DEVICE))
    start_logits = output.start_logits[0].tolist()
    end_logits = output.end_logits[0].tolist()
    start, end = best_span(start_logits, end_logits, sequence_ids)
    return context[offsets[start][0]:offsets[end][1]].strip()


for i in range(5):
    context = documents[DOC_TEXT].iloc[hybrid_top[i][0]]
    print(queries[QUERY_TEXT].iloc[i][:60], "→", answer_question(queries[QUERY_TEXT].iloc[i],
                                                                  context),
          "| kunci:", queries["jawaban"].iloc[i])

# %% [markdown]
# Nilai jawaban: exact match + F1 kata (metrik SQuAD) pada 200 pertanyaan.

# %%
import numpy as np


def answer_f1(guess, truth):
    guess_words, truth_words = tokenize(guess), tokenize(truth)
    shared = 0
    for word in set(guess_words):
        shared += min(guess_words.count(word), truth_words.count(word))
    if shared == 0:
        return 0.0
    precision, recall = shared / len(guess_words), shared / len(truth_words)
    return 2 * precision * recall / (precision + recall)


exact, f1_values = [], []
for i in range(200):   # UBAH: semua → range(len(queries))
    guess = answer_question(queries[QUERY_TEXT].iloc[i], documents[DOC_TEXT].iloc[hybrid_top[i][0]])
    exact.append(tokenize(guess) == tokenize(queries["jawaban"].iloc[i]))
    f1_values.append(answer_f1(guess, queries["jawaban"].iloc[i]))
print("exact match:", round(np.mean(exact), 4), "| F1 kata:", round(np.mean(f1_values), 4))

# %% [markdown]
# ## 6. Teks kembar / hampir kembar → tabel `pairs`
# (a) kembar persis setelah dinormalisasi (huruf kecil, tanpa tanda baca);
# (b) hampir kembar: cosine TF-IDF potongan huruf ≥ `SIMILAR_LEVEL` (tahan typo & sisipan).

# %%
import pandas as pd

texts = documents[DOC_TEXT]   # UBAH: kolom teks yang dicek
normalized = []
for text in texts:
    normalized.append(" ".join(tokenize(text)))
normalized = pd.Series(normalized)
exact_groups = normalized[normalized.duplicated(keep=False)]
print("teks kembar persis:", len(exact_groups), "baris")

# %% [markdown]
# Hampir kembar: tetangga terdekat cosine (cepat, tidak membandingkan semua pasangan).

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

SIMILAR_LEVEL = 0.8   # UBAH: 0.9 = hampir sama persis
char_vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(normalized)
neighbors = NearestNeighbors(n_neighbors=5, metric="cosine").fit(char_vectors)
distances, indices = neighbors.kneighbors(char_vectors)
rows = []
for i in range(len(normalized)):
    for distance, other in zip(distances[i][1:], indices[i][1:]):
        if i < other and 1 - distance >= SIMILAR_LEVEL:
            rows.append({"a": i, "b": other, "similarity": round(1 - distance, 3),
                         "text_a": texts.iloc[i][:60], "text_b": texts.iloc[other][:60]})
pairs = pd.DataFrame(rows, columns=["a", "b", "similarity", "text_a", "text_b"])
pairs = pairs.sort_values("similarity")   # paling tidak mirip di atas: cek batas ambang
print(len(pairs), "pasangan hampir kembar")
print(pairs.head(5).to_string())
