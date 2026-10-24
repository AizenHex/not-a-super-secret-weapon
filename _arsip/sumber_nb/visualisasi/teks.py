# %% [markdown]
# # Visualisasi — Teks (kata, panjang, ciri khas tiap kelas)
# **Kapan dipakai:** "kata apa yang paling sering? kelas mana punya kosakata khas? seberapa
# panjang teksnya? kata apa yang sering muncul bersama?"
#
# **Isi:** sebaran kelas, frekuensi kata dan n-gram, wordcloud, panjang teks per kelas, kata khas
# per kelas (TF-IDF), jaringan ko-okurensi, sunburst interaktif. Semua gambar disimpan di folder
# notebook.
#
# **Data demo:** SmSA (ulasan Indonesia, 3 sentimen).

# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/sentiment"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/smsa_train.csv"

TEXT_COL = "text"
LABEL_COL = "label"
STOPWORDS = StopWordRemoverFactory().get_stop_words() + ["nya", "yg", "gak", "aja", "banget"]
CLASSES = ["negative", "neutral", "positive"]
TOP_WORDS = 30        # jumlah kata di jaringan ko-okurensi
MIN_TOGETHER = 60     # dua kata dihubungkan bila muncul bersama minimal sekian kali
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
# ## Baca data

# %%
table = pd.read_csv(TRAIN_CSV)
class_colors = {"negative": RED, "neutral": GREY, "positive": BLUE}
print(table.shape)
table[[TEXT_COL, LABEL_COL]].head(3)

# %% [markdown]
# ## 1. Sebaran kelas
# **Kapan dipakai:** pertama kali memeriksa data teks berlabel (seimbang atau tidak?).
# **Cara baca:** tinggi bar = jumlah teks. Kelas yang sangat sedikit perlu penanganan khusus
# (class weight, F1-macro).

# %%
class_counts = table[LABEL_COL].value_counts().reindex(CLASSES)
bar_colors = []
for name in CLASSES:
    bar_colors.append(class_colors[name])
class_counts.plot.bar(color=bar_colors, rot=0)
for i, value in enumerate(class_counts):
    plt.text(i, value + 20, str(value), ha="center")
plt.title("Jumlah ulasan per sentimen")
plt.xlabel("")
plt.ylabel("jumlah ulasan")
plt.savefig("teks_sebaran_kelas.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Frekuensi kata teratas
# **Kapan dipakai:** gambaran awal isi korpus, mencari kata yang perlu masuk daftar stopword.
# **Cara baca:** bar panjang = kata sering muncul (stopword sudah dibuang). Kata umum yang tidak
# informatif (mis. "nya") tambahkan ke `STOPWORDS` di Konfigurasi.

# %%
vectorizer = CountVectorizer(stop_words=STOPWORDS, max_features=2000)
word_matrix = vectorizer.fit_transform(table[TEXT_COL])    # baris = ulasan, kolom = kata
word_names = vectorizer.get_feature_names_out()
word_totals = np.asarray(word_matrix.sum(axis=0)).ravel()
frequency = pd.Series(word_totals, index=word_names)
top_words = frequency.nlargest(20)
top_words.sort_values().plot.barh(color=BLUE, figsize=(8, 6))
plt.title("20 kata paling sering")
plt.xlabel("jumlah kemunculan")
plt.savefig("teks_frekuensi_kata.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Frekuensi n-gram (frasa)
# **Kapan dipakai:** kata tunggal ambigu, frasa dua-tiga kata lebih bermakna ("tidak enak").
# **Cara baca:** sama dengan frekuensi kata, tapi tiap bar = pasangan kata berurutan.

# %%
phrase_vectorizer = CountVectorizer(ngram_range=(2, 3), min_df=5, max_features=3000)
phrase_matrix = phrase_vectorizer.fit_transform(table[TEXT_COL])
phrase_totals = np.asarray(phrase_matrix.sum(axis=0)).ravel()
phrases = pd.Series(phrase_totals, index=phrase_vectorizer.get_feature_names_out())
top_phrases = phrases.nlargest(15)
top_phrases.sort_values().plot.barh(color=ORANGE, figsize=(8, 6))
plt.title("15 frasa (2-3 kata) paling sering")
plt.xlabel("jumlah kemunculan")
plt.savefig("teks_frekuensi_ngram.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Wordcloud
# **Kapan dipakai:** ilustrasi cepat untuk slide pembuka, bukan untuk analisis presisi.
# **Cara baca:** ukuran huruf = frekuensi. Untuk perbandingan kelas, buat satu wordcloud per kelas.

# %%
cloud = WordCloud(width=1200, height=600, background_color="white", colormap="Blues",
                  max_words=100, random_state=SEED)
cloud.generate_from_frequencies(frequency.to_dict())
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.title("Kata yang sering muncul")
# Alternatif per kelas: positive = " ".join(table[TEXT_COL][table[LABEL_COL] == "positive"])
# lalu WordCloud().generate(positive)
plt.savefig("teks_wordcloud.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Panjang teks per kelas
# **Kapan dipakai:** menentukan batas panjang token model, dan mengecek apakah panjang teks
# sendiri jadi petunjuk kelas. **Cara baca:** kotak = 50% teks tengah, garis = median,
# titik = teks sangat panjang.

# %%
words_per_review = table[TEXT_COL].str.split()
table["jumlah_kata"] = words_per_review.str.len()
sns.boxplot(data=table, x=LABEL_COL, y="jumlah_kata", order=CLASSES, hue=LABEL_COL,
            palette=class_colors, fliersize=2, legend=False)
plt.title("Jumlah kata per ulasan menurut sentimen")
plt.xlabel("")
plt.ylabel("jumlah kata")
plt.savefig("teks_panjang_per_kelas.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Kata khas per kelas (TF-IDF)
# **Kapan dipakai:** memahami apa yang membedakan kelas, bahan interpretasi dan fitur manual.
# **Cara baca:** tiap panel = 10 kata dengan bobot TF-IDF rata-rata tertinggi pada kelas itu.
# Kata yang muncul di semua panel tidak membedakan kelas.

# %%
tfidf = TfidfVectorizer(stop_words=STOPWORDS, max_features=3000, min_df=3)
weights = tfidf.fit_transform(table[TEXT_COL])
tfidf_words = tfidf.get_feature_names_out()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, name in zip(axes, CLASSES):
    in_class = (table[LABEL_COL] == name).to_numpy()
    class_mean = np.asarray(weights[in_class].mean(axis=0)).ravel()
    top_index = class_mean.argsort()[-10:]
    ax.barh(tfidf_words[top_index], class_mean[top_index], color=class_colors[name])
    ax.set_title(name)
    ax.set_xlabel("rata-rata TF-IDF")
plt.savefig("teks_kata_khas_kelas.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Jaringan ko-okurensi kata
# **Kapan dipakai:** melihat topik/kelompok kata yang sering muncul bersama.
# **Cara baca:** simpul = kata (besar = sering), garis = dua kata sering satu kalimat (tebal =
# sering). Gerombol simpul yang rapat = satu topik.

# %%
common_words = list(frequency.nlargest(TOP_WORDS).index)
column_numbers = []
for word in common_words:
    column_numbers.append(vectorizer.vocabulary_[word])
present = (word_matrix[:, column_numbers] > 0).astype(int)   # 1 bila kata ada di ulasan
together = (present.T @ present).toarray()                    # [kata, kata] = jumlah ulasan

graph = nx.Graph()
for i in range(TOP_WORDS):
    graph.add_node(common_words[i], size=frequency[common_words[i]])
    for other in range(i + 1, TOP_WORDS):
        if together[i, other] >= MIN_TOGETHER:
            graph.add_edge(common_words[i], common_words[other], weight=together[i, other])
graph.remove_nodes_from(list(nx.isolates(graph)))

positions = nx.spring_layout(graph, seed=SEED, k=1.2)
node_sizes = []
for word in graph.nodes:
    node_sizes.append(graph.nodes[word]["size"] * 0.6)
edge_widths = []
for edge in graph.edges:
    edge_widths.append(graph.edges[edge]["weight"] / MIN_TOGETHER / 2)
plt.figure(figsize=(9, 7))
nx.draw_networkx(graph, positions, node_size=node_sizes, node_color=BLUE, width=edge_widths,
                 edge_color=GREY, font_size=11, alpha=0.9)
plt.title("Kata yang sering muncul bersama")
plt.axis("off")
plt.savefig("teks_jaringan_kata.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Sunburst interaktif: kelas lalu kata (plotly)
# **Kapan dipakai:** menelusuri kata dominan tiap kelas di laporan interaktif.
# **Cara baca:** lingkaran dalam = kelas, luar = kata. Klik sebuah kelas untuk memperbesar.

# %%
rows = []
for name in CLASSES:
    in_class = (table[LABEL_COL] == name).to_numpy()
    class_totals = np.asarray(word_matrix[in_class].sum(axis=0)).ravel()
    for index in class_totals.argsort()[-8:]:
        rows.append([name, word_names[index], class_totals[index]])
sunburst_table = pd.DataFrame(rows, columns=["kelas", "kata", "jumlah"])
fig = px.sunburst(sunburst_table, path=["kelas", "kata"], values="jumlah", color="kelas",
                  color_discrete_map=class_colors, title="Kata dominan per sentimen")
fig.write_html("teks_sunburst.html", include_plotlyjs=True)
fig.show()
