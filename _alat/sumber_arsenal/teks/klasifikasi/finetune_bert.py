# %% [markdown]
# # Klasifikasi teks: fine-tune IndoBERT (paling akurat, butuh GPU)
#
# **Butuh:** `train` (kolom teks + label), `test` (kolom teks + id), GPU.
# **Menghasilkan:** `submission.csv`, `outputs/best_model.pt`, `outputs/test_proba.csv`.
#
# Kapan: setelah baseline `tfidf.ipynb` jalan dan masih ada waktu ≥30 menit. Seluruh bobot
# BERT ikut dilatih, jadi biasanya 3–8 poin F1 di atas TF-IDF. Acuan paper IndoNLU (2020),
# IndoBERT-base: F1 EmoT 75,5 · SmSA 87,7 · CASA 93,2 · WReTE 78,6.
#
# Pengaturan dari paper (disalin ke sel 1):
# - lr 1e-5–4e-5 (IndoNLU), 2e-5 mencegah "lupa" pengetahuan pretrained (Sun dkk. 2019);
# - batch 16, warmup 10% lalu lr turun linear, pilih epoch dengan F1 val terbaik;
# - teks > `MAX_LENGTH` token: ambil awal + akhir (Sun dkk.: 128 awal + 382 akhir terbaik di
#   ulasan panjang; untuk berita, awal saja sering lebih baik);
# - data kecil (< 1000): hasil bisa beda jauh antar seed → coba 2–3 `SEED`, rata-ratakan proba.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + cek panjang token | **ya** |
# | 2 | `TextDataset` + split train/val + DataLoader (dua sel) | **ya** |
# | 3 | model + optimizer + jadwal lr | **ya** |
# | 4 | `run_epoch(loader, training)` | **ya** |
# | 5 | loop training + simpan bobot terbaik + berhenti dini | **ya** |
# | 6 | muat bobot terbaik → laporan val + salah tebak | **ya** |
# | 7 | tebak test → `submission.csv` + proba | **ya** |
#
# "CUDA out of memory": turunkan `BATCH_SIZE` ke 8 atau `MAX_LENGTH` ke 64.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Tweet berlabel emosi (5 kelas); `test_labels` = kunci jawaban (hanya ada di demo).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/text/classification/emot_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/text/classification/emot_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/text/classification/emot_test_labels.csv")
print(train.shape, test.shape)

# %% [markdown]
# ## 1. Konfigurasi + cek panjang token
# Pilih `MAX_LENGTH` ≈ persentil 95 panjang token (teks lebih panjang dipotong).
# Model lain di `pretrained/`: `distilbert_id` (2× lebih cepat, sedikit kalah), `minilm`
# (multibahasa, kecil), `distilbert_en` (teks Inggris).

# %%
import os

import numpy as np
import torch
from transformers import AutoTokenizer

PRETRAINED_FOLDER = "pretrained"   # UBAH
MODEL_NAME = "indobert"            # UBAH
TEXT = "text"                      # UBAH
TARGET = "label"                   # UBAH
MAX_LENGTH = 128                   # UBAH: lihat persentil di bawah
BATCH_SIZE = 16                    # UBAH
EPOCHS = 4                         # UBAH: 3–5 biasanya cukup
LEARNING_RATE = 3e-5               # UBAH: 1e-5 .. 5e-5
PATIENCE = 2                       # berhenti kalau F1 val tidak naik 2 epoch berturut-turut
SEED = 42                          # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = DEVICE == "cuda"        # UBAH: False kalau GPU lama / error bf16
os.makedirs("outputs", exist_ok=True)
torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(f"{PRETRAINED_FOLDER}/{MODEL_NAME}")
train[TEXT] = train[TEXT].fillna("").astype(str)
test[TEXT] = test[TEXT].fillna("").astype(str)
token_counts = []
for text in train[TEXT].head(2000):
    token_counts.append(len(tokenizer(text)["input_ids"]))
print("token per teks, persentil 50/95/99:", np.percentile(token_counts, [50, 95, 99]))

# %% [markdown]
# ## 2. `TextDataset`, split 80/20, DataLoader
# **Menghasilkan:** `CLASSES`, `part_train`, `part_val`, `train_loader`, `val_loader`,
# `test_loader`. `DataCollatorWithPadding`: tiap batch dipadding sepanjang teks terpanjang di
# batch itu saja (lebih cepat daripada selalu `MAX_LENGTH`).

# %%
from torch.utils.data import Dataset

CLASSES = sorted(train[TARGET].unique())
train["target"] = train[TARGET].map(dict(zip(CLASSES, range(len(CLASSES)))))
test["target"] = 0                                # pengisi, tidak dipakai


class TextDataset(Dataset):
    def __init__(self, table):
        self.tokens = tokenizer(table[TEXT].tolist(), truncation=True, max_length=MAX_LENGTH)
        self.targets = table["target"].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        item = {"input_ids": self.tokens["input_ids"][i],
                "attention_mask": self.tokens["attention_mask"][i]}
        item["labels"] = self.targets[i]
        return item


print(TextDataset(train.head(2))[0])

# %% [markdown]
# Split 80/20 + DataLoader.

# %%
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

part_train, part_val = train_test_split(train, test_size=0.2, stratify=train["target"],
                                        random_state=42)
collate = DataCollatorWithPadding(tokenizer)
train_loader = DataLoader(TextDataset(part_train), batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate)
val_loader = DataLoader(TextDataset(part_val), batch_size=BATCH_SIZE * 2, collate_fn=collate)
test_loader = DataLoader(TextDataset(test), batch_size=BATCH_SIZE * 2, collate_fn=collate)
print(len(CLASSES), "kelas |", len(part_train), "train |", len(part_val), "val")

# %% [markdown]
# ## 3. Model + optimizer + jadwal lr → `network`, `optimizer`, `scheduler`
# `AutoModelForSequenceClassification` = BERT + kepala baru `len(CLASSES)` keluaran (peringatan
# "newly initialized" itu normal). Kelas timpang: aktifkan `class_weight`.

# %%
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

network = AutoModelForSequenceClassification.from_pretrained(
    f"{PRETRAINED_FOLDER}/{MODEL_NAME}", num_labels=len(CLASSES),
    id2label=dict(zip(range(len(CLASSES)), CLASSES)))
network = network.to(DEVICE)
loss_function = nn.CrossEntropyLoss()
# counts = np.bincount(part_train["target"], minlength=len(CLASSES))          # kelas timpang
# class_weight = torch.tensor(len(part_train) / (len(CLASSES) * counts), dtype=torch.float)
# loss_function = nn.CrossEntropyLoss(weight=class_weight.to(DEVICE))
optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = EPOCHS * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
print("langkah total:", total_steps)

# %% [markdown]
# ## 4. `run_epoch(loader, training)` → (loss rata-rata, probabilitas per kelas)
# `autocast` bfloat16 (`USE_BF16`, sel 1): ±2× lebih cepat di RTX 30xx ke atas.

# %%
import numpy as np
import torch
from tqdm import tqdm


def run_epoch(loader, training):
    network.train(training)
    total_loss, probabilities = 0.0, []
    for batch in tqdm(loader, leave=False):
        batch = batch.to(DEVICE)
        targets = batch.pop("labels")
        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits = network(**batch).logits.float()
            loss = loss_function(logits, targets)
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        total_loss += loss.item() * len(targets)
        probability = torch.softmax(logits.detach(), dim=1)
        probabilities.append(probability.cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(probabilities)


val_loss, val_probabilities = run_epoch(val_loader, training=False)
print("sebelum latih | val loss:", round(val_loss, 3), "| bentuk:", val_probabilities.shape)

# %% [markdown]
# ## 5. Loop training → `outputs/best_model.pt` (bobot dengan F1 val terbaik)

# %%
import torch
from sklearn.metrics import f1_score

best_f1, waited = -1.0, 0
for epoch in range(EPOCHS):
    train_loss, _ = run_epoch(train_loader, training=True)
    val_loss, val_probabilities = run_epoch(val_loader, training=False)
    val_f1 = f1_score(part_val["target"], val_probabilities.argmax(axis=1), average="macro")
    print(f"epoch {epoch + 1}: train {train_loss:.3f} | val {val_loss:.3f} | F1 {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1, waited = val_f1, 0
        torch.save(network.state_dict(), "outputs/best_model.pt")
    else:
        waited += 1
    if waited >= PATIENCE:
        print("berhenti dini")
        break
print("F1 val terbaik:", round(best_f1, 4))

# %% [markdown]
# ## 6. Muat bobot terbaik → laporan per kelas + salah tebak
# Setelah kernel restart: jalankan sel 0–3, lalu sel ini (tanpa sel 5).

# %%
import torch
from sklearn.metrics import classification_report

network.load_state_dict(torch.load("outputs/best_model.pt"))
_, val_probabilities = run_epoch(val_loader, training=False)
part_val = part_val.copy()
part_val["pred"] = np.array(CLASSES)[val_probabilities.argmax(axis=1)]
print(classification_report(part_val[TARGET], part_val["pred"], digits=4))
wrong = part_val[part_val[TARGET] != part_val["pred"]]
print(wrong[[TEXT, TARGET, "pred"]].head(6).to_string(max_colwidth=70))

# %% [markdown]
# ## 7. Tebak test → `submission.csv` + `outputs/test_proba.csv`
# Proba disimpan untuk dirata-rata dengan seed/model lain (ensembling).

# %%
import numpy as np
import pandas as pd

_, test_probabilities = run_epoch(test_loader, training=False)
proba_table = pd.DataFrame(test_probabilities, columns=CLASSES)
proba_table.insert(0, "id", test["id"].to_numpy())
proba_table.to_csv(f"outputs/test_proba_seed{SEED}.csv", index=False)

submission = pd.DataFrame()
submission["id"] = test["id"]                                               # UBAH
submission["label"] = np.array(CLASSES)[test_probabilities.argmax(axis=1)]  # UBAH
submission.to_csv("submission.csv", index=False)
print(submission["label"].value_counts())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
from sklearn.metrics import f1_score

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
print("F1 macro test:", round(f1_score(test_labels["label"], submission["label"],
                                       average="macro"), 4))
