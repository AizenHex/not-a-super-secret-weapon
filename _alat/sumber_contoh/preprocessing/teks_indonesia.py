# %% [markdown]
# # Preprocessing — Teks bebas Bahasa Indonesia (tweet, ulasan, komentar)
# **Kapan dipakai:** teks seperti `gak bisa dibuka tp udh bayar [USERNAME] #kecewa`,
# `mantappp bgt 👍👍` perlu dirapikan untuk EDA, kata kunci, topik, atau model TF-IDF.
#
# **Langkah**, masing-masing satu baris dan bisa dimatikan dengan `#`:
# token khusus → URL → mention → hashtag → emoji jadi kata → huruf berulang → sisa simbol →
# slang → stopword → stemming.
#
# Kamus slang:
# - `kamus_alay_salsabila.csv`: kamus publik, 4.330 kata.
# - `kamus_slang.csv`: tambahan sendiri, menimpa kamus publik kalau ada slang yang sama.
#   Tambahkan slang baru dari soal ke file ini.
#
# Catatan penting:
# - Stopword Sastrawi berisi `tidak`, `belum`, `tapi`. Untuk sentimen/emosi kata-kata ini
#   **disimpan** lewat `KEEP_WORDS`.
# - Membersihkan teks **belum tentu** menaikkan skor. Sel "Tahap mana yang membantu" mengukur skor
#   tiap tahap dengan model dan fold yang sama.
# - EmoT sudah tidak berisi emoji, jadi langkah emoji tidak berpengaruh di data demo ini.
# - Pembersihan tidak memakai target, jadi aman dilakukan sebelum split.
#
# **Data demo:** tweet EmoT (IndoNLU, 5 emosi).

# %%
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/text/classification"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/emot_train.csv"
DICTIONARY_FOLDER = "../../../data_latihan/preprocessing"
PUBLIC_SLANG_CSV = f"{DICTIONARY_FOLDER}/kamus_alay_salsabila.csv"
OWN_SLANG_CSV = f"{DICTIONARY_FOLDER}/kamus_slang.csv"

TEXT_COL = "text"
TARGET = "label"
EMOJI = {"👍": " emojipositif ", "😍": " emojipositif ", "🔥": " emojipositif ",
         "😊": " emojipositif ", "❤️": " emojipositif ", "😂": " emojitawa ",
         "😡": " emojinegatif ", "👎": " emojinegatif ", "😭": " emojisedih ",
         "🙄": " emojinegatif "}
KEEP_WORDS = {"tidak", "enggak", "belum", "bukan", "kurang", "tapi", "sangat", "banget"}
SEED = 42

# %% [markdown]
# ## Baca data + kamus slang

# %%
table = pd.read_csv(TRAIN_CSV)
dictionary = pd.concat([pd.read_csv(PUBLIC_SLANG_CSV), pd.read_csv(OWN_SLANG_CSV)])
slang = dict(zip(dictionary["slang"], dictionary["baku"]))   # baris belakang menimpa depan
print(table.shape, "|", len(slang), "slang di kamus")
print(table[TARGET].value_counts())
table[[TEXT_COL, TARGET]].head()

# %% [markdown]
# ## Tahap 1: regex

# %%
text = table[TEXT_COL].fillna("")
text = text.str.lower()
text = text.str.replace(r"\[username\]|\[url\]", " ", regex=True)    # token khusus dataset
text = text.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)    # URL
text = text.str.replace(r"@\w+", " ", regex=True)                     # mention
text = text.str.replace(r"#(\w+)", r" \1 ", regex=True)               # hashtag: buang # saja
for symbol, word in EMOJI.items():
    text = text.str.replace(symbol, word, regex=False)                # emoji -> kata
text = text.str.replace(r"(\w)\1{2,}", r"\1", regex=True)             # mantappp -> mantap
text = text.str.replace(r"[^a-z\s]", " ", regex=True)                 # sisa angka dan simbol
text = text.str.replace(r"\s+", " ", regex=True)                      # spasi ganda
text_regex = text.str.strip()
pd.DataFrame({"mentah": table[TEXT_COL], "regex": text_regex}).head()

# %% [markdown]
# ## Tahap 2–4: slang, stopword, stemming (per kata)
# Stemming Sastrawi lambat. Karena itu stemming dijalankan sekali per **kata unik**, lalu hasilnya
# dipakai lewat kamus `roots`.

# %%
stopwords = set(StopWordRemoverFactory().get_stop_words()) - KEEP_WORDS
stemmer = StemmerFactory().create_stemmer()


def replace_slang(sentence):
    words = []
    for word in sentence.split():
        words.append(slang.get(word, word))
    return " ".join(words)


def remove_stopwords(sentence):
    words = []
    for word in sentence.split():
        if word not in stopwords:
            words.append(word)
    return " ".join(words)


print(replace_slang("gak bisa dibuka tp udh bayar"), "|", remove_stopwords("saya tidak suka"))

# %%
text_slang = text_regex.map(replace_slang)
text_stopword = text_slang.map(remove_stopwords)

vocabulary = set(" ".join(text_stopword).split())
roots = {}
for word in vocabulary:
    roots[word] = stemmer.stem(word)


def stem_sentence(sentence):
    words = []
    for word in sentence.split():
        words.append(roots[word])
    return " ".join(words)


text_stem = text_stopword.map(stem_sentence)
pd.DataFrame({"regex": text_regex, "slang": text_slang, "stem": text_stem}).head()

# %% [markdown]
# ## Ukuran kosakata tiap tahap + slang yang belum ada di kamus

# %%
stages = [("mentah", table[TEXT_COL].fillna("")), ("regex", text_regex), ("slang", text_slang),
          ("stopword", text_stopword), ("stem", text_stem)]
for name, texts in stages:
    unique_words = set(" ".join(texts).split())
    print(f"kosakata {name:9s}: {len(unique_words):6d} kata")

word_counts = pd.Series(" ".join(text_slang).split()).value_counts()
short_words = word_counts[word_counts.index.str.len() <= 3]
print("kata pendek tersering sesudah ganti slang (masih ada slang? tambahkan ke kamus_slang.csv):")
print(list(short_words.head(30).index))

# %% [markdown]
# ## Tahap mana yang membantu? (TF-IDF + LogisticRegression, 5-fold, F1 macro)
# Model dan fold sama untuk semua tahap, yang berubah hanya teksnya.

# %%
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, texts in stages:
    scores = cross_val_score(model, texts, table[TARGET], cv=cv, scoring="f1_macro")
    print(f"{name:10s} F1 macro {scores.mean():.4f} ± {scores.std():.4f}")

# %% [markdown]
# Hasil di EmoT: mentah 0.659, regex 0.660, slang 0.658, stopword 0.669, stem 0.669.
# - Regex dan slang tidak mengubah skor TF-IDF.
# - Stopword dan stemming menambah sekitar +0.01, kira-kira sebesar simpangan antar-fold.
# - Kenaikan besar datang dari model, bukan dari pembersihan. Contoh: fine-tune IndoBERT
#   (`text/klasifikasi_finetune`). Untuk model transformer, cukup pakai `text_regex`, karena
#   tokenizer-nya sudah mengenal kata tidak baku.
# - Pembersihan tetap penting untuk EDA, kata kunci, dan topik, yaitu saat kata harus dibaca
#   manusia.

# %% [markdown]
# ## Simpan hasil (+ fitur sederhana dari teks mentah)

# %%
result = pd.DataFrame()
result["id"] = table["id"]
result["teks_regex"] = text_regex
result["teks_slang"] = text_slang
result["teks_stem"] = text_stem
result["jumlah_seru"] = table[TEXT_COL].str.count("!")
result["jumlah_tanya"] = table[TEXT_COL].str.count(r"\?")
result["rasio_kapital"] = table[TEXT_COL].str.count(r"[A-Z]") / table[TEXT_COL].str.len()
result.to_csv("hasil_teks.csv", index=False)
result.head()
