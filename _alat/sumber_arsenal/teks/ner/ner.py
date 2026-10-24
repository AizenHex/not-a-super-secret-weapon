# %% [markdown]
# # NER / pelabelan per kata: cari nama orang, tempat, organisasi, produk di dalam kalimat
#
# **Butuh:** tabel per kata: `sentence_id`, `token`, `tag` (format BIO: `B-PPL` = awal nama
# orang, `I-PPL` = lanjutan, `O` = bukan entitas). **Menghasilkan:** tag per kata di test,
# `submission.csv` (format sama dengan train), tabel entitas yang ditemukan.
#
# Juga untuk: POS tagging, ambil field dari teks (merek, ukuran, harga), aspek di ulasan.
#
# | cara | kecepatan | acuan NERP (F1 entitas) |
# |---|---|---|
# | CRF + fitur kata (sel 2–4) | ±1 menit, CPU | baseline kuat |
# | IndoBERT token classification (sel 6–9) | ±3 menit, GPU | IndoNLU: 75,5 (base) |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | tabel per kata → daftar kalimat + daftar tag | **ya** |
# | 2 | `word_features`: fitur tiap kata untuk CRF | CRF |
# | 3 | `entity_spans` + `entity_report` (F1 per jenis entitas) | **ya** |
# | 4 | latih CRF + nilai di val | CRF |
# | 5 | tebak test → `submission.csv` + tabel entitas | **ya** |
# | 6–9 | IndoBERT: dataset, model, latih, tebak | kalau ada GPU + waktu |
#
# Format beda? Teks utuh + kolom entitas (`"Jokowi|PPL"`) → pecah ke kata dulu dengan
# `str.split()` dan beri tag `O` / `B-` / `I-` per kata.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# NERP: berita, entitas PPL (orang), PLC (tempat), IND (organisasi), EVT (acara), FNB
# (makanan). `keep_default_na=False`: token "NA"/"null" tetap teks, bukan NaN.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

words_train = pd.read_csv(f"{DEMO_FOLDER}/text/ner/nerp_train.csv", keep_default_na=False)
words_test = pd.read_csv(f"{DEMO_FOLDER}/text/ner/nerp_test.csv", keep_default_na=False)
print(words_train.head(8))
print(words_train["tag"].value_counts())

# %% [markdown]
# ## 1. Tabel per kata → `train_words`, `train_tags`, `test_words` (list kalimat)
# Satu kalimat = list kata. Urutan kata dalam kalimat harus dijaga (`sort=False`).

# %%
SENTENCE = "sentence_id"   # UBAH
TOKEN = "token"            # UBAH
TAG = "tag"                # UBAH


def to_sentences(table, column):
    sentences = []
    for _, group in table.groupby(SENTENCE, sort=False):
        sentences.append(group[column].astype(str).tolist())
    return sentences


train_words, train_tags = to_sentences(words_train, TOKEN), to_sentences(words_train, TAG)
test_words = to_sentences(words_test, TOKEN)
test_ids = words_test[SENTENCE].unique()
TAGS = sorted(words_train[TAG].unique())
print(len(train_words), "kalimat train |", len(test_words), "kalimat test |", TAGS)
print(list(zip(train_words[1], train_tags[1]))[:12])

# %% [markdown]
# ## 2. `word_features(words, i)` → dict fitur kata ke-i (untuk CRF)
# Kata itu sendiri, akhiran/awalan, huruf besar, angka, dan kata tetangga (±2).
# Punya daftar nama (kota, merek)? Tambahkan `"in_list": word.lower() in DAFTAR`.

# %%
def word_features(words, i):
    word = words[i]
    features = {"bias": 1.0, "lower": word.lower(), "suffix3": word[-3:].lower(),
                "suffix2": word[-2:].lower(), "prefix3": word[:3].lower(),
                "is_title": word.istitle(), "is_upper": word.isupper(),
                "is_digit": word.isdigit(), "has_hyphen": "-" in word,
                "length": min(len(word), 12)}
    for offset in [-2, -1, 1, 2]:
        position = i + offset
        if 0 <= position < len(words):
            features[f"word{offset}"] = words[position].lower()
            features[f"title{offset}"] = words[position].istitle()
        else:
            features[f"edge{offset}"] = True     # awal/akhir kalimat
    return features


def sentence_features(words):
    result = []
    for i in range(len(words)):
        result.append(word_features(words, i))
    return result


print(word_features(["presiden", "joko", "widodo", "di", "solo"], 1))

# %% [markdown]
# ## 3. `entity_spans(tags)` + `entity_report(true, pred)` → F1 per jenis entitas
# Penilaian per **entitas utuh** (CoNLL): "joko widodo" benar hanya kalau awal, akhir, dan
# jenisnya sama persis.

# %%
def entity_spans(tags):
    spans, start, kind = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if start is not None and (tag == "O" or tag.startswith("B-") or tag[2:] != kind):
            spans.append((start, i, kind))
            start = None
        if tag != "O" and start is None:
            start, kind = i, tag[2:]
    return spans


print(entity_spans(["O", "B-PPL", "I-PPL", "O", "B-PLC", "I-IND"]))

# %% [markdown]
# `entity_report`: hitung benar / tebakan / seharusnya per jenis → presisi, recall, F1.

# %%
import pandas as pd


def entity_report(true_lists, pred_lists):
    counts = {}
    for true_tags, pred_tags in zip(true_lists, pred_lists):
        true_set, pred_set = set(entity_spans(true_tags)), set(entity_spans(pred_tags))
        for span in true_set | pred_set:
            row = counts.setdefault(span[2], {"benar": 0, "tebakan": 0, "seharusnya": 0})
            row["benar"] += span in true_set and span in pred_set
            row["tebakan"] += span in pred_set
            row["seharusnya"] += span in true_set
    report = pd.DataFrame(counts).T
    report.loc["SEMUA (micro)"] = report.sum()
    report["presisi"] = report["benar"] / report["tebakan"].clip(lower=1)
    report["recall"] = report["benar"] / report["seharusnya"].clip(lower=1)
    report["f1"] = 2 * report["presisi"] * report["recall"] / (
        report["presisi"] + report["recall"]).clip(lower=1e-9)
    return report.round(4)


print(entity_report([["B-PPL", "I-PPL", "O"]], [["B-PPL", "O", "O"]]))

# %% [markdown]
# ## 4. Latih CRF di 80% kalimat, nilai di 20% → `crf`
# CRF belajar fitur kata + aturan urutan tag (mis. `I-PPL` hanya setelah `B-PPL`/`I-PPL`).
# `c1`, `c2` = regularisasi (naikkan kalau overfit).

# %%
import sklearn_crfsuite
from sklearn.model_selection import train_test_split

index_train, index_val = train_test_split(range(len(train_words)), test_size=0.2,
                                          random_state=42)
features_train, tags_train, features_val, tags_val = [], [], [], []
for i in index_train:
    features_train.append(sentence_features(train_words[i]))
    tags_train.append(train_tags[i])
for i in index_val:
    features_val.append(sentence_features(train_words[i]))
    tags_val.append(train_tags[i])
crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=150,
                           all_possible_transitions=True)   # UBAH: c1, c2
crf.fit(features_train, tags_train)
val_pred = crf.predict(features_val)
print(entity_report(tags_val, val_pred))

# %% [markdown]
# ## 5. Latih CRF di SEMUA train → tebak test → `submission.csv` + tabel entitas
# Submission = format train (satu baris per kata). Soal minta format lain (mis. daftar
# entitas per kalimat)? Pakai tabel `entities`.

# %%
import pandas as pd

all_features = []
for words in train_words:
    all_features.append(sentence_features(words))
crf.fit(all_features, train_tags)
test_features = []
for words in test_words:
    test_features.append(sentence_features(words))
test_pred = crf.predict(test_features)

submission = pd.DataFrame()
submission[SENTENCE] = words_test[SENTENCE]
submission[TOKEN] = words_test[TOKEN]
flat_tags = []
for tags in test_pred:
    flat_tags.extend(tags)
submission[TAG] = flat_tags
submission.to_csv("submission.csv", index=False)
print(submission.head())

# %% [markdown]
# Tabel entitas yang ditemukan: satu baris per entitas (kalimat, teks entitas, jenis).

# %%
import pandas as pd

rows = []
for sentence_id, words, tags in zip(test_ids, test_words, test_pred):
    for start, end, kind in entity_spans(tags):
        rows.append({"sentence_id": sentence_id, "entity": " ".join(words[start:end]),
                     "type": kind})
entities = pd.DataFrame(rows)
print(entities.head(10))
print(entities["type"].value_counts())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo: test NERP punya tag).

# %%
print(entity_report(to_sentences(words_test, TAG), test_pred))

# %% [markdown]
# ## 6. IndoBERT token classification: konfigurasi + `align_labels`
# BERT memecah kata jadi potongan (`widodo` → `wi`, `##dodo`). Tag hanya diberikan ke
# potongan PERTAMA tiap kata; potongan lain diberi -100 (diabaikan oleh loss).

# %%
import torch
from transformers import AutoTokenizer

PRETRAINED_FOLDER = "pretrained"   # UBAH
MODEL_NAME = "indobert"            # UBAH
MAX_LENGTH = 128                   # UBAH: kalimat panjang → 256
BATCH_SIZE = 16
EPOCHS = 3                         # UBAH
LEARNING_RATE = 3e-5               # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAG_TO_ID = dict(zip(TAGS, range(len(TAGS))))
tokenizer = AutoTokenizer.from_pretrained(f"{PRETRAINED_FOLDER}/{MODEL_NAME}")


def align_labels(word_ids, tags):
    labels, previous = [], None
    for word_id in word_ids:
        if word_id is None or word_id == previous:
            labels.append(-100)
        else:
            labels.append(TAG_TO_ID[tags[word_id]])
        previous = word_id
    return labels


example = tokenizer(train_words[1][:6], is_split_into_words=True)
print(tokenizer.convert_ids_to_tokens(example["input_ids"]))
print(align_labels(example.word_ids(), train_tags[1][:6]))

# %% [markdown]
# ## 7. `TokenDataset` + DataLoader (pakai split sel 4)

# %%
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForTokenClassification


class TokenDataset(Dataset):
    def __init__(self, sentences, tag_lists):
        self.tokens = tokenizer(sentences, is_split_into_words=True, truncation=True,
                                max_length=MAX_LENGTH)
        self.labels = []
        for row, tags in enumerate(tag_lists):
            self.labels.append(align_labels(self.tokens.word_ids(row), tags))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.tokens["input_ids"][i],
                "attention_mask": self.tokens["attention_mask"][i], "labels": self.labels[i]}


collate = DataCollatorForTokenClassification(tokenizer)
words_part, words_val = [], []
for i in index_train:
    words_part.append(train_words[i])
for i in index_val:
    words_val.append(train_words[i])
train_loader = DataLoader(TokenDataset(words_part, tags_train), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)
val_loader = DataLoader(TokenDataset(words_val, tags_val), batch_size=32, collate_fn=collate)
print(len(train_loader), "batch per epoch")

# %% [markdown]
# ## 8. Model + optimizer, lalu `run_epoch`
# Tebakan diambil hanya di posisi yang label-nya bukan -100 (= potongan pertama tiap kata).
# Kata yang terpotong `MAX_LENGTH` diberi `O` (sel 9).

# %%
import torch
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

network = AutoModelForTokenClassification.from_pretrained(
    f"{PRETRAINED_FOLDER}/{MODEL_NAME}", num_labels=len(TAGS),
    id2label=dict(zip(range(len(TAGS)), TAGS)), label2id=TAG_TO_ID)
network = network.to(DEVICE)
optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = EPOCHS * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
print("langkah total:", total_steps)

# %% [markdown]
# `run_epoch(loader, training)` → (loss rata-rata, nomor tag tebakan per kalimat).

# %%
import torch


def run_epoch(loader, training):
    network.train(training)
    total_loss, predictions = 0.0, []
    for batch in loader:
        batch = batch.to(DEVICE)
        with torch.set_grad_enabled(training):
            output = network(**batch)
        if training:
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            scheduler.step()
        total_loss += output.loss.item()
        guesses = output.logits.argmax(dim=-1)
        for guess, labels in zip(guesses, batch["labels"]):
            predictions.append(guess[labels != -100].tolist())
    return total_loss / len(loader), predictions


val_loss, _ = run_epoch(val_loader, training=False)
print("sebelum latih | val loss:", round(val_loss, 3))

# %% [markdown]
# ## 9. Loop training + nilai per entitas
# `to_tags`: nomor → nama tag, kata yang terpotong diisi `O`.

# %%
import torch


def to_tags(id_lists, word_lists):
    result = []
    for ids, words in zip(id_lists, word_lists):
        tags = []
        for tag_id in ids:
            tags.append(TAGS[tag_id])
        result.append(tags + ["O"] * (len(words) - len(tags)))
    return result


best_f1 = -1.0
for epoch in range(EPOCHS):
    train_loss, _ = run_epoch(train_loader, training=True)
    _, val_ids = run_epoch(val_loader, training=False)
    report = entity_report(tags_val, to_tags(val_ids, words_val))
    val_f1 = report.loc["SEMUA (micro)", "f1"]
    print(f"epoch {epoch + 1}: train loss {train_loss:.3f} | F1 entitas val {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(network.state_dict(), "best_ner.pt")
print(report)

# %% [markdown]
# Tebak test dengan bobot terbaik → `bert_test_pred` (lalu pakai sel 5 untuk submission:
# ganti `test_pred` dengan `bert_test_pred`).

# %%
import torch
from torch.utils.data import DataLoader

network.load_state_dict(torch.load("best_ner.pt"))
dummy_tags = []
for words in test_words:
    dummy_tags.append(["O"] * len(words))
test_loader = DataLoader(TokenDataset(test_words, dummy_tags), batch_size=32,
                         collate_fn=collate)
_, test_ids_pred = run_epoch(test_loader, training=False)
bert_test_pred = to_tags(test_ids_pred, test_words)
print(entity_report(to_sentences(words_test, TAG), bert_test_pred))   # hanya demo
