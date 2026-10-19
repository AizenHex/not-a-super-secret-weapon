# %% [markdown]
# # Tabular — Statistik Inferensial + Uji A/B (bahan "ketepatan metode" dan presentasi)
# **Kapan dipakai:** menjawab pertanyaan "apakah X benar-benar berpengaruh?" dengan p-value dan
# selang kepercayaan (confidence interval, CI).
#
# **Aturan baca:** p < 0.05 → perbedaan/hubungan **signifikan** (kecil kemungkinan hanya
# kebetulan). Ingat: signifikan ≠ besar. Selalu laporkan juga ukuran efeknya.
#
# **Data demo:** Titanic (`survived`) dan Hillstrom email marketing (uji A/B).

# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular/classification"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/titanic_train.csv"
AB_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular_extra/uplift"
AB_CSV = f"{AB_FOLDER}/hillstrom.csv"

TARGET = "survived"
N_BOOTSTRAP = 2000
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
print(data.shape)
data.head()

# %% [markdown]
# ## 1. Kategori vs kategori: uji chi-square + Cramér's V
# Cramér's V = ukuran efek 0–1 (0 = tidak berhubungan, 1 = berhubungan sempurna).

# %%
table = pd.crosstab(data["sex"], data[TARGET])
chi2, p_chi2, degrees_of_freedom, expected = stats.chi2_contingency(table)
n_rows = table.to_numpy().sum()
cramers_v = np.sqrt(chi2 / (n_rows * (min(table.shape) - 1)))
print(table)
print("chi2 =", round(chi2, 2), "| p =", f"{p_chi2:.2e}", "| Cramér's V =", round(cramers_v, 3))

# %% [markdown]
# ## 2. Angka antar 2 kelompok: Welch t-test + Mann-Whitney
# Mann-Whitney tidak butuh data berdistribusi normal. Cohen's d = selisih rata-rata dalam
# satuan simpangan baku (0.2 kecil, 0.5 sedang, 0.8 besar).

# %%
survived_fare = data.loc[data[TARGET] == 1, "fare"]
died_fare = data.loc[data[TARGET] == 0, "fare"]
t_stat, p_ttest = stats.ttest_ind(survived_fare, died_fare, equal_var=False)
u_stat, p_mannwhitney = stats.mannwhitneyu(survived_fare, died_fare)
pooled_std = np.sqrt((survived_fare.var() + died_fare.var()) / 2)
cohens_d = (survived_fare.mean() - died_fare.mean()) / pooled_std
print("rata-rata fare selamat:", round(survived_fare.mean(), 1),
      "| tidak:", round(died_fare.mean(), 1))
print("Welch t p =", f"{p_ttest:.2e}", "| Mann-Whitney p =", f"{p_mannwhitney:.2e}",
      "| Cohen's d =", round(cohens_d, 2))

# %% [markdown]
# ## 3. Angka antar >2 kelompok: ANOVA + Kruskal-Wallis
# Contoh: apakah umur berbeda antar kelas tiket (pclass 1, 2, 3)?

# %%
groups = []
for pclass_value, group in data.groupby("pclass"):
    groups.append(group["age"].dropna())

f_stat, p_anova = stats.f_oneway(*groups)
h_stat, p_kruskal = stats.kruskal(*groups)
mean_age = data.groupby("pclass")["age"].mean()
print(mean_age.round(1))
print("ANOVA p =", f"{p_anova:.2e}", "| Kruskal p =", f"{p_kruskal:.2e}")

# %% [markdown]
# ## 4. Korelasi Pearson (linear) dan Spearman (urutan) + p-value

# %%
pairs = data[["age", "fare"]].dropna()
pearson_r, p_pearson = stats.pearsonr(pairs["age"], pairs["fare"])
spearman_rho, p_spearman = stats.spearmanr(pairs["age"], pairs["fare"])
print("Pearson r =", round(pearson_r, 3), "(p =", round(p_pearson, 3), ")")
print("Spearman rho =", round(spearman_rho, 3), "(p =", round(p_spearman, 3), ")")

# %% [markdown]
# ## 5. Regresi logistik: odds ratio tiap faktor, dengan faktor lain dikontrol
# OR > 1 = menaikkan peluang selamat, OR < 1 = menurunkan. `C(...)` = kolom kategori.

# %%
logit = smf.logit("survived ~ C(sex) + age + C(pclass) + fare", data=data).fit(disp=0)
interval = logit.conf_int()

odds_table = pd.DataFrame()
odds_table["odds_ratio"] = np.exp(logit.params)
odds_table["ci_bawah"] = np.exp(interval[0])
odds_table["ci_atas"] = np.exp(interval[1])
odds_table["p_value"] = logit.pvalues
odds_table.to_csv("odds_ratio.csv")
odds_table.round(4)

# %% [markdown]
# ## 6. Bootstrap: selang kepercayaan 95% untuk selisih peluang selamat (wanita − pria)
# Ambil sampel ulang (dengan pengembalian) ribuan kali, hitung selisihnya tiap kali.
# CI 95% = persentil 2.5% sampai 97.5% dari semua selisih itu.

# %%
random_generator = np.random.default_rng(SEED)
women = data.loc[data["sex"] == "female", TARGET].to_numpy()
men = data.loc[data["sex"] == "male", TARGET].to_numpy()

differences = []
for i in range(N_BOOTSTRAP):
    women_sample = random_generator.choice(women, len(women))
    men_sample = random_generator.choice(men, len(men))
    differences.append(women_sample.mean() - men_sample.mean())

ci_95 = np.percentile(differences, [2.5, 97.5]).round(3)
print("selisih =", round(women.mean() - men.mean(), 3), "| CI 95% =", ci_95)

# %% [markdown]
# ## 7. Banyak uji sekaligus → koreksi p-value (Holm)
# Makin banyak uji, makin besar peluang "signifikan" karena kebetulan. Koreksi menaikkan p.

# %%
test_names = ["sex", "fare", "age_pclass", "age_fare"]
p_values = [p_chi2, p_ttest, p_anova, p_pearson]
is_significant, p_corrected, sidak_alpha, bonferroni_alpha = multipletests(p_values,
                                                                           method="holm")
correction = pd.DataFrame(index=test_names)
correction["p"] = p_values
correction["p_holm"] = p_corrected
correction["signifikan"] = is_significant
correction

# %% [markdown]
# ## 8. Uji A/B: apakah email menaikkan kunjungan? (z-test dua proporsi)
# Lift = kenaikan relatif tingkat kunjungan grup email dibanding kontrol.

# %%
ab = pd.read_csv(AB_CSV)
ab["grup"] = np.where(ab["segment"] == "No E-Mail", "kontrol", "email")
ab_summary = ab.groupby("grup")["visit"].agg(["sum", "count", "mean"])
z_stat, p_ab = proportions_ztest(ab_summary["sum"].to_numpy(), ab_summary["count"].to_numpy())
lift = ab_summary.loc["email", "mean"] / ab_summary.loc["kontrol", "mean"] - 1
print(ab_summary)
print("z =", round(z_stat, 2), "| p =", f"{p_ab:.2e}", "| lift =", f"{lift:.1%}")

# %% [markdown]
# ## 9. Ukuran sampel: berapa orang per grup agar kenaikan 10.6% → 12% terdeteksi?
# alpha 0.05 = batas p. power 0.8 = 80% peluang menemukan efek kalau efeknya memang ada.

# %%
effect_size = proportion_effectsize(0.12, 0.106)
needed = NormalIndPower().solve_power(effect_size=effect_size, alpha=0.05, power=0.8)
print("butuh ±", int(np.ceil(needed)), "orang per grup")
