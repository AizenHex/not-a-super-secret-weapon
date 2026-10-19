# KATALOG — semua metode per notebook + skor uji

> **Catatan 19 Okt:** notebook sudah ditulis ulang gaya pemula. Teks baris alternatif di bawah
> masih memakai gaya lama (`ROOT / ...`, nama Indonesia). Metode dan skornya tetap berlaku.

Dibuat otomatis 2026-10-16. Tiap alternatif yang di-comment di sel Model dijalankan
penuh (Run All) satu per satu pada data demo. Skor = metrik validasi yang sama dengan
notebook aslinya, jadi bisa dibandingkan langsung di dalam satu notebook.

- **asli** = model yang aktif sekarang. ⭐ = lebih baik dari asli.
- Skor demo hanya petunjuk: di data lomba urutannya bisa berbeda. Coba 2–3 teratas.
- GAGAL = alternatif error di data demo; pesan error ditulis di kolom skor.

## `audio/klasifikasi_cnn`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.3772**.


| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `DATA = ROOT / "data_latihan" / "audio" / "speaker" / "speech_commands_speakers"` | 0.1228 | LOLOS (data lain, skor tidak sebanding) |

## `audio/klasifikasi_embedding`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.8749**.

Aktif sekarang: `model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), max_iter=1000, random_stat...` | 0.8628 | LOLOS |
| `model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))` | 0.7776 | LOLOS |
| `DATA = ROOT / "data_latihan" / "audio" / "speaker" / "speech_commands_speakers"` | 0.7586 | LOLOS (data lain, skor tidak sebanding) |
| `model = make_pipeline(StandardScaler(), LGBMClassifier(verbose=-1, random_state=SEED))` | 0.7245 | LOLOS |
| `model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, random_state=SEED))` | 0.6052 | LOLOS |
| `model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))` | 0.4985 | LOLOS |

## `audio/klasifikasi_mfcc`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.449**.

Aktif sekarang: `model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, random_state=SEED))`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `DATA = ROOT / "data_latihan" / "audio" / "speaker" / "speech_commands_speakers"` | 0.8871 | LOLOS (data lain, skor tidak sebanding) |
| `model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))` | 0.5109 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))` | 0.5107 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), max_iter=1000, random_stat...` | 0.5065 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), LGBMClassifier(random_state=SEED, verbose=-1))` | 0.5054 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), ExtraTreesClassifier(n_estimators=500, random_state=SEED))` | 0.4837 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))` | 0.3419 | LOLOS |

## `audio/retrieval`

Metrik: `recall@1` (makin besar makin baik). Asli: **0.525**.


| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `DATA = ROOT / "data_latihan" / "audio" / "speaker" / "speech_commands_speakers"` | 0.8167 | LOLOS (data lain, skor tidak sebanding) |

## `graph/klasifikasi_node`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.7211**.

Aktif sekarang: `model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=SEED, verbose=-1)` | 0.8117 ⭐ | LOLOS |
| `model = ExtraTreesClassifier(n_estimators=500, random_state=SEED)` | 0.7627 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, random_state=SEED))` | 0.7555 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_stat...` | 0.6997 | LOLOS |
| `model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))` | 0.6526 | LOLOS |

## `graph/komunitas`

Metrik: `modularitas cora` (makin besar makin baik). Asli: **0.8039**.

Aktif sekarang: `komunitas = nx.community.louvain_communities(graf, seed=SEED)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `komunitas = nx.community.greedy_modularity_communities(graf)` | 0.7958 | LOLOS |
| `komunitas = nx.community.label_propagation_communities(graf)` | 0.6368 | LOLOS |
| `komunitas = nx.community.asyn_fluidc(graf, k=4, seed=SEED)` | 0.6308 | LOLOS |
| `komunitas = nx.community.asyn_lpa_communities(graf, seed=SEED)` | 0.5725 | LOLOS |
| `komunitas = nx.community.kernighan_lin_bisection(graf, seed=SEED)` | 0.4007 | LOLOS |

## `graph/prediksi_link`

Metrik: `AUC model` (makin besar makin baik). Asli: **0.7681**.

Aktif sekarang: `model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED)` | 0.7734 ⭐ | LOLOS |
| `model = HistGradientBoostingClassifier(random_state=SEED)` | 0.7719 ⭐ | LOLOS |
| `model = LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=SEED)` | 0.7695 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=SEED))` | 0.7684 ⭐ | LOLOS |
| `model = ExtraTreesClassifier(n_estimators=300, random_state=SEED)` | 0.7586 | LOLOS |

## `multimodal/klasifikasi`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.4812**.

Aktif sekarang: `model = LogisticRegression(C=1.0, max_iter=2000)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = LinearSVC(C=0.5)` | 0.5567 ⭐ | LOLOS |
| `model = ExtraTreesClassifier(n_estimators=500, random_state=SEED)` | 0.5425 ⭐ | LOLOS |
| `model = RandomForestClassifier(n_estimators=500, random_state=SEED)` | 0.5179 ⭐ | LOLOS |
| `model = SGDClassifier(loss="modified_huber", random_state=SEED)` | 0.5056 ⭐ | LOLOS |
| `model = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED, verbose=-1)` | 0.5046 ⭐ | LOLOS |

## `multimodal/regresi`

Metrik: `RMSE` (makin kecil makin baik). Asli: **0.7619**.

Aktif sekarang: `model = Ridge(alpha=1.0)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = XGBRegressor(n_estimators=300, learning_rate=0.03, random_state=SEED)` | 0.7418 ⭐ | LOLOS |
| `model = LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=SEED, verbose=-1)` | 0.7476 ⭐ | LOLOS |
| `model = LinearSVR(C=1.0, max_iter=10000, random_state=SEED)` | 0.7492 ⭐ | LOLOS |
| `model = RandomForestRegressor(n_estimators=500, random_state=SEED)` | 0.7770 | LOLOS |
| `model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)` | 0.8464 | LOLOS |

## `recsys/prediksi_rating`

Metrik: `RMSE model ` (makin kecil makin baik). Asli: **1.026**.

Aktif sekarang: `model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.05, random_state=SEED)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = SVDpp(n_factors=20, n_epochs=20, random_state=SEED)` | 1.0261 | LOLOS |
| `model = KNNBaseline(k=40, sim_options={"name": "pearson_baseline", "user_based": False})` | 1.0295 | LOLOS |
| `model = BaselineOnly()` | 1.0300 | LOLOS |
| `model = KNNWithMeans(k=40, sim_options={"name": "pearson_baseline", "user_based": False})` | 1.1004 | LOLOS |
| `model = SlopeOne()` | 1.1023 | LOLOS |
| `model = NMF(n_factors=30, n_epochs=50, random_state=SEED)` | 1.1059 | LOLOS |
| `model = CoClustering(n_cltr_u=3, n_cltr_i=3, random_state=SEED)` | 1.1108 | LOLOS |
| `model = KNNBasic(k=40, sim_options={"name": "pearson_baseline", "user_based": False})` | 1.1196 | LOLOS |
| `model = NormalPredictor()` | 1.5356 | LOLOS |

## `recsys/top_n`

Metrik: `precision@10` (makin besar makin baik). Asli: **0.2297**.

Aktif sekarang: `skor = np.tile(popularitas, (N_USER, 1)).astype(float)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `bpr = BayesianPersonalizedRanking(factors=64, iterations=100, use_gpu=False, random_state=SEED) bpr.fit(lat...` | 0.2217 | LOLOS |
| `skor = latih_matrix @ cosine_similarity(latih_matrix.T)` | 0.0962 | LOLOS |
| `skor = cosine_similarity(latih_matrix) @ latih_matrix # user-based` | 0.0941 | LOLOS |
| `als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, use_gpu=False, random_state=S...` | 0.0824 | LOLOS |

## `tabular/deep_klasifikasi`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.8187**.

Aktif sekarang: `return nn.Sequential(nn.Linear(N_FITUR, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn...`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `return nn.Sequential(nn.Linear(N_FITUR, 32), nn.ReLU(), nn.Linear(32, N_KELAS))` | 0.8269 ⭐ | LOLOS |
| `nn.Linear(N_FITUR, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.BatchNorm...` | 0.8201 ⭐ | LOLOS |

## `tabular/deep_regresi`

Metrik: `RMSE` (makin kecil makin baik). Asli: **1181.898**.

Aktif sekarang: `return nn.Sequential(nn.Linear(N_FITUR, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1), )`  
Aktif sekarang: `return nn.Sequential(nn.Linear(N_FITUR, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),...`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `nn.Linear(N_FITUR, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.BatchNorm...` | 1433.1861 | LOLOS |
| `return nn.Sequential(nn.Linear(N_FITUR, 64), nn.ReLU(), nn.Linear(64, 1))` | 3096.6274 | LOLOS |

## `tabular/ensembling_klasifikasi`

Metrik: `ENSEMBLE F1 macro` (makin besar makin baik). Asli: **0.8187**.

Aktif sekarang: `ensemble = StackingClassifier(base_models, final_estimator=LogisticRegression(), cv=5)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `ensemble = VotingClassifier(base_models, voting="hard")` | 0.8187 | LOLOS |
| `ensemble = VotingClassifier(base_models, voting="soft", weights=[1, 1, 2, 2])` | 0.8187 | LOLOS |
| `ensemble = VotingClassifier(base_models, voting="soft")` | 0.8119 | LOLOS |
| `ensemble = StackingClassifier(base_models, final_estimator=LogisticRegression(), cv=5, passthrough=True)` | 0.8066 | LOLOS |
| `ensemble = StackingClassifier(base_models, final_estimator=LGBMClassifier(verbose=-1), cv=5)` | 0.8051 | LOLOS |

## `tabular/ensembling_regresi`

Metrik: `ENSEMBLE RMSE` (makin kecil makin baik). Asli: **749.2753**.

Aktif sekarang: `ensemble = StackingRegressor(base_models, final_estimator=RidgeCV(), cv=5)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `ensemble = StackingRegressor(base_models, final_estimator=LGBMRegressor(verbose=-1), cv=5)` | 718.1530 ⭐ | LOLOS |
| `ensemble = VotingRegressor(base_models)` | 726.2079 ⭐ | LOLOS |
| `ensemble = StackingRegressor(base_models, final_estimator=RidgeCV(), cv=5, passthrough=True)` | 727.7629 ⭐ | LOLOS |
| `ensemble = VotingRegressor(base_models, weights=[1, 1, 2, 2])` | 728.5370 ⭐ | LOLOS |

## `tabular/explain_klasifikasi`

Metrik: `F1 macro validasi` (makin besar makin baik). Asli: **0.8201**.

> RF/ExtraTrees: aktifkan juga `shap_values = shap_values[:, :, 1]`.

Aktif sekarang: `model = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED, verbose=-1)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = XGBClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED)` | 0.8187 | LOLOS |
| `model = CatBoostClassifier(iterations=300, random_seed=SEED, verbose=0, allow_writing_files=False)` | 0.8187 | LOLOS |
| `model = RandomForestClassifier(n_estimators=300, random_state=SEED) shap_values = shap_values[:, :, 1] # Ra...` | 0.7999 | LOLOS |
| `model = ExtraTreesClassifier(n_estimators=300, random_state=SEED) shap_values = shap_values[:, :, 1] # Rand...` | 0.7879 | LOLOS |

## `tabular/explain_regresi`

Metrik: `RMSE validasi` (makin kecil makin baik). Asli: **822.7913**.

Aktif sekarang: `model = LGBMRegressor(n_estimators=800, learning_rate=0.03, random_state=SEED, verbose=-1)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = CatBoostRegressor(iterations=800, random_seed=SEED, verbose=0, allow_writing_files=False)` | 767.9437 ⭐ | LOLOS |
| `model = ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)` | 770.3516 ⭐ | LOLOS |
| `model = XGBRegressor(n_estimators=800, learning_rate=0.03, random_state=SEED)` | 785.2498 ⭐ | LOLOS |
| `model = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)` | 833.7101 | LOLOS |

## `tabular/feature_engineering_klasifikasi`

Metrik: `F1 sesudah FE` (makin besar makin baik). Asli: **0.7989**.

Aktif sekarang: `model = LogisticRegression(C=1.0, max_iter=2000)`  
Aktif sekarang: `encoder = OneHotEncoder(handle_unknown="ignore")`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `encoder = TargetEncoder(random_state=SEED)` | 0.8057 ⭐ | LOLOS |
| `model = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED, verbose=-1)` | 0.7987 | LOLOS |
| `model = RandomForestClassifier(n_estimators=500, random_state=SEED)` | 0.7986 | LOLOS |
| `encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` | 0.7972 | LOLOS |

## `tabular/feature_engineering_regresi`

Metrik: `RMSE sesudah FE` (makin kecil makin baik). Asli: **1046.9**.

Aktif sekarang: `model = Ridge(alpha=1.0)`  
Aktif sekarang: `encoder = OneHotEncoder(handle_unknown="ignore")`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = LGBMRegressor(n_estimators=800, learning_rate=0.03, random_state=SEED, verbose=-1)` | 634.9000 ⭐ | LOLOS |
| `model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)` | 664.1000 ⭐ | LOLOS |
| `encoder = TargetEncoder(random_state=SEED)` | 1071.4000 | LOLOS |
| `encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` | 1099.4000 | LOLOS |

## `tabular/imbalanced`

Metrik: `F1 positif` (makin besar makin baik). Asli: **0.6234**.

Aktif sekarang: `pipe = make_pipeline(StandardScaler(), SMOTE(random_state=SEED), LGBMClassifier(n_estimators=300, verbose=-...`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `pipe = make_pipeline(XGBClassifier(scale_pos_weight=40, n_estimators=300))` | 0.7368 ⭐ | LOLOS |
| `pipe = make_pipeline(StandardScaler(), ADASYN(random_state=SEED),` | 0.7105 ⭐ | LOLOS |
| `pipe = make_pipeline(LGBMClassifier(class_weight="balanced", verbose=-1))` | 0.7000 ⭐ | LOLOS |
| `pipe = make_pipeline(TomekLinks(), LGBMClassifier(n_estimators=300, verbose=-1))` | 0.6769 ⭐ | LOLOS |
| `pipe = make_pipeline(StandardScaler(), SMOTETomek(random_state=SEED),` | 0.6753 ⭐ | LOLOS |
| `pipe = make_pipeline(StandardScaler(), SMOTEENN(random_state=SEED),` | 0.6596 ⭐ | LOLOS |
| `pipe = make_pipeline(CatBoostClassifier(auto_class_weights="Balanced", verbose=0, allow_writing_files=False...` | 0.6593 ⭐ | LOLOS |
| `pipe = make_pipeline(StandardScaler(), SVMSMOTE(random_state=SEED),` | 0.6582 ⭐ | LOLOS |
| `pipe = make_pipeline(StandardScaler(), BorderlineSMOTE(random_state=SEED), RandomForestClassifier(n_estimat...` | 0.6410 ⭐ | LOLOS |
| `pipe = make_pipeline(BalancedRandomForestClassifier(n_estimators=300, random_state=SEED))` | 0.5037 | LOLOS |
| `pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))` | 0.4719 | LOLOS |
| `LogisticRegression(max_iter=2000))` | 0.4599 | LOLOS |
| `pipe = make_pipeline(BalancedBaggingClassifier(n_estimators=50, random_state=SEED))` | 0.4533 | LOLOS |
| `pipe = make_pipeline(RandomUnderSampler(random_state=SEED), RandomForestClassifier(n_estimators=300, random...` | 0.3700 | LOLOS |
| `pipe = make_pipeline(EasyEnsembleClassifier(n_estimators=10, random_state=SEED))` | 0.2802 | LOLOS |
| `pipe = make_pipeline(RUSBoostClassifier(n_estimators=200, random_state=SEED))` | 0.2500 | LOLOS |

## `tabular/klasifikasi`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.8012**.

Aktif sekarang: `model = LogisticRegression(C=1.0, max_iter=2000)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = SVC(C=1.0, random_state=SEED)` | 0.8407 ⭐ | LOLOS |
| `model = XGBClassifier(n_estimators=500, learning_rate=0.03, random_state=SEED)` | 0.8201 ⭐ | LOLOS |
| `model = KNeighborsClassifier(n_neighbors=15)` | 0.8201 ⭐ | LOLOS |
| `model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=SEED)` | 0.8201 ⭐ | LOLOS |
| `model = HistGradientBoostingClassifier(random_state=SEED)` | 0.8187 ⭐ | LOLOS |
| `model = CatBoostClassifier(iterations=500, random_seed=SEED, verbose=0)` | 0.8119 ⭐ | LOLOS |
| `model = RidgeClassifier(alpha=1.0)` | 0.8080 ⭐ | LOLOS |
| `model = BaggingClassifier(n_estimators=100, random_state=SEED)` | 0.8012 | LOLOS |
| `model = RandomForestClassifier(n_estimators=500, random_state=SEED)` | 0.7999 | LOLOS |
| `model = LGBMClassifier(n_estimators=500, learning_rate=0.03, random_state=SEED, verbose=-1)` | 0.7999 | LOLOS |
| `model = GradientBoostingClassifier(random_state=SEED)` | 0.7984 | LOLOS |
| `model = AdaBoostClassifier(n_estimators=300, random_state=SEED)` | 0.7970 | LOLOS |
| `model = DecisionTreeClassifier(max_depth=6, random_state=SEED)` | 0.7917 | LOLOS |
| `model = ExtraTreesClassifier(n_estimators=500, random_state=SEED)` | 0.7879 | LOLOS |

## `tabular/multikelas`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.6547**.

> XGBoost/CatBoost dihapus dari daftar: XGB butuh label angka, CatBoost memberi prediksi 2D.

Aktif sekarang: `model = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=SEED, verbose=-1)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)` | 0.7155 ⭐ | LOLOS |
| `model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=SEED)` | 0.7007 ⭐ | LOLOS |
| `model = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)` | 0.6886 ⭐ | LOLOS |
| `model = HistGradientBoostingClassifier(random_state=SEED)` | 0.6188 | LOLOS |
| `model = LogisticRegression(C=1.0, max_iter=3000)` | 0.5818 | LOLOS |
| `model = DecisionTreeClassifier(random_state=SEED)` | 0.5737 | LOLOS |
| `model = SVC(C=1.0, random_state=SEED)` | 0.5450 | LOLOS |
| `model = KNeighborsClassifier(n_neighbors=5)` | 0.5403 | LOLOS |
| `model = LinearDiscriminantAnalysis()` | 0.5179 | LOLOS |
| `model = AdaBoostClassifier(n_estimators=300, random_state=SEED)` | 0.3278 | LOLOS |
| `model = GaussianNB()` | 0.1619 | LOLOS |

## `tabular/multilabel`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.3827**.

Aktif sekarang: `model = MultiOutputClassifier(LogisticRegression(C=1.0, max_iter=2000))`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=SEED)` | 0.4290 ⭐ | LOLOS |
| `model = ClassifierChain(LGBMClassifier(n_estimators=300, verbose=-1), random_state=SEED)` | 0.4246 ⭐ | LOLOS |
| `model = MultiOutputClassifier(XGBClassifier(n_estimators=300, random_state=SEED))` | 0.4218 ⭐ | LOLOS |
| `model = MultiOutputClassifier(LGBMClassifier(n_estimators=300, verbose=-1))` | 0.4155 ⭐ | LOLOS |
| `model = ClassifierChain(LogisticRegression(C=1.0, max_iter=2000), random_state=SEED)` | 0.4142 ⭐ | LOLOS |
| `model = KNeighborsClassifier(n_neighbors=15)` | 0.3898 ⭐ | LOLOS |
| `model = MultiOutputClassifier(SVC(C=1.0, random_state=SEED))` | 0.3695 | LOLOS |
| `model = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)` | 0.3247 | LOLOS |
| `model = ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)` | 0.3095 | LOLOS |

## `tabular/regresi`

Metrik: `RMSE` (makin kecil makin baik). Asli: **828.0993**.

Aktif sekarang: `model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=SEED, verbose=-1)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = CatBoostRegressor(iterations=1000, random_seed=SEED, verbose=0)` | 755.9525 ⭐ | LOLOS |
| `model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)` | 769.5993 ⭐ | LOLOS |
| `model = XGBRegressor(n_estimators=1000, learning_rate=0.03, random_state=SEED)` | 783.8888 ⭐ | LOLOS |
| `model = HistGradientBoostingRegressor(random_state=SEED)` | 814.4393 ⭐ | LOLOS |
| `model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)` | 836.5067 | LOLOS |
| `model = GradientBoostingRegressor(random_state=SEED)` | 912.6893 | LOLOS |
| `model = DecisionTreeRegressor(max_depth=10, random_state=SEED)` | 1045.7070 | LOLOS |
| `model = Lasso(alpha=1.0)` | 1147.1759 | LOLOS |
| `model = Ridge(alpha=1.0)` | 1148.4149 | LOLOS |
| `model = BayesianRidge()` | 1148.6531 | LOLOS |
| `model = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=SEED)` | 1241.0517 | LOLOS |
| `model = KNeighborsRegressor(n_neighbors=10)` | 1300.8136 | LOLOS |
| `model = HuberRegressor(max_iter=1000)` | 1335.6306 | LOLOS |
| `model = ElasticNet(alpha=0.1, l1_ratio=0.5)` | 1460.8057 | LOLOS |
| `model = AdaBoostRegressor(n_estimators=300, random_state=SEED)` | 1597.4933 | LOLOS |
| `model = LinearSVR(C=1.0, max_iter=10000, random_state=SEED)` | 2603.7357 | LOLOS |

## `tabular/regresi_lanjutan`

Metrik: `RMSE` (makin kecil makin baik). Asli: **784.8936**.

Aktif sekarang: `model = TransformedTargetRegressor(lgbm, func=np.log1p, inverse_func=np.expm1)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = TransformedTargetRegressor(CatBoostRegressor(iterations=800, verbose=0, allow_writing_files=False, ...` | 764.7334 ⭐ | LOLOS |
| `model = LGBMRegressor(objective="poisson", n_estimators=800, learning_rate=0.03, verbose=-1)` | 780.2559 ⭐ | LOLOS |
| `model = LGBMRegressor(objective="tweedie", n_estimators=800, learning_rate=0.03, verbose=-1)` | 789.3693 | LOLOS |
| `model = LGBMRegressor(objective="gamma", n_estimators=800, learning_rate=0.03, verbose=-1)` | 792.4415 | LOLOS |
| `model = LGBMRegressor(objective="huber", alpha=1000, n_estimators=800, verbose=-1)` | 866.1608 | LOLOS |
| `model = TransformedTargetRegressor(Ridge(alpha=1.0), func=np.log1p, inverse_func=np.expm1)` | 1072.5494 | LOLOS |
| `model = HuberRegressor(max_iter=1000)` | 1335.6306 | LOLOS |
| `model = TweedieRegressor(power=2, link="log", max_iter=1000)` | 2009.7147 | LOLOS |

## `tabular/survival`

Metrik: `C-index` (makin besar makin baik). Asli: **0.6201**.

> Model AFT: aktifkan juga dua baris `-model.predict_median(...)`.

Aktif sekarang: `model = CoxPHFitter(penalizer=0.1)`  
Aktif sekarang: `prediksi = model.predict_partial_hazard(test[FEATURES])`  
Aktif sekarang: `risiko_val = model.predict_partial_hazard(data_val[FEATURES])`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)` | 0.6352 ⭐ | LOLOS |
| `model = LogNormalAFTFitter(penalizer=0.1) risiko_val = -model.predict_median(data_val[FEATURES]) prediksi =...` | 0.6208 ⭐ | LOLOS |
| `prediksi = -model.predict_median(test[FEATURES])` | 0.6201 | LOLOS |
| `model = LogLogisticAFTFitter(penalizer=0.1) risiko_val = -model.predict_median(data_val[FEATURES]) prediksi...` | 0.6188 | LOLOS |
| `model = WeibullAFTFitter(penalizer=0.1) risiko_val = -model.predict_median(data_val[FEATURES]) prediksi = -...` | 0.6188 | LOLOS |
| `model = CoxPHFitter(penalizer=0.01)` | 0.6113 | LOLOS |
| `risiko_val = -model.predict_median(data_val[FEATURES])` | 0.4713 | LOLOS |

## `tabular/tuning_klasifikasi`

Metrik: `CV F1 terbaik` (makin besar makin baik). Asli: **0.8256**.

Aktif sekarang: `return LGBMClassifier(n_estimators=trial.suggest_int("n_estimators", 100, 800), learning_rate=trial.suggest...`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `return XGBClassifier(n_estimators=trial.suggest_int("n_estimators", 100, 800), max_depth=trial.suggest_int(...` | 0.8268 ⭐ | LOLOS |
| `return CatBoostClassifier(iterations=trial.suggest_int("iterations", 200, 1000), depth=trial.suggest_int("d...` | 0.8166 | LOLOS |
| `return LogisticRegression(C=trial.suggest_float("C", 1e-3, 100, log=True), max_iter=3000)` | 0.8104 | LOLOS |
| `return RandomForestClassifier(n_estimators=trial.suggest_int("n_estimators", 100, 800), max_depth=trial.sug...` | 0.8056 | LOLOS |

## `tabular/tuning_regresi`

Metrik: `CV RMSE terbaik` (makin kecil makin baik). Asli: **684.211**.

Aktif sekarang: `return LGBMRegressor(n_estimators=trial.suggest_int("n_estimators", 200, 1500), learning_rate=trial.suggest...`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `return CatBoostRegressor(iterations=trial.suggest_int("iterations", 200, 1500), depth=trial.suggest_int("de...` | 670.5892 ⭐ | LOLOS |
| `return XGBRegressor(n_estimators=trial.suggest_int("n_estimators", 200, 1500), max_depth=trial.suggest_int(...` | 723.8228 | LOLOS |
| `return RandomForestRegressor(n_estimators=trial.suggest_int("n_estimators", 100, 500), max_depth=trial.sugg...` | 788.1443 | LOLOS |
| `return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))` | 1145.8901 | LOLOS |

## `tabular/validasi`

Metrik: `rata-rata skor per fold` (bukan lomba skor, hanya pembanding). Asli: **0.806**.

> GroupKFold butuh `groups=train["kolom_grup"]` (kolom grup tidak ada di data demo).

Aktif sekarang: `cv = StratifiedKFold(5, shuffle=True, random_state=SEED)`  
Aktif sekarang: `skor = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `cv = TimeSeriesSplit(5)` | 0.8063 | LOLOS |
| `cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)` | 0.8058 | LOLOS |
| `cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)` | 0.8041 | LOLOS |
| `cv = KFold(5, shuffle=True, random_state=SEED)` | 0.8004 | LOLOS |
| `cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)` | 0.7786 | LOLOS |
| `cv = GroupKFold(5)` | ValueError: The 'groups' parameter should not be None. | GAGAL |
| `skor = cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=train["kolom_grup"], scoring="f1_macro")` | KeyError: 'kolom_grup' | GAGAL |

## `text/clustering`

Metrik: `ARI` (makin besar makin baik). Asli: **0.2691**.

Aktif sekarang: `model = KMeans(n_clusters=K, n_init=10, random_state=SEED)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = make_pipeline(TruncatedSVD(100, random_state=SEED), Normalizer(), KMeans(n_clusters=K, n_init=10, r...` | 0.4161 ⭐ | LOLOS |
| `model = make_pipeline(TruncatedSVD(100, random_state=SEED), Normalizer(), AgglomerativeClustering(n_cluster...` | 0.3520 ⭐ | LOLOS |
| `model = BisectingKMeans(n_clusters=K, random_state=SEED)` | 0.2385 | LOLOS |
| `model = MiniBatchKMeans(n_clusters=K, n_init=10, random_state=SEED)` | 0.2286 | LOLOS |
| `model = SpectralClustering(n_clusters=K, affinity="nearest_neighbors", random_state=SEED)` | 0.0031 | LOLOS |

## `text/klasifikasi_embedding`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.8461**.

Aktif sekarang: `EMBED_DIR = ROOT / "pretrained" / "minilm"`  
Aktif sekarang: `model = LogisticRegression(C=1.0, max_iter=2000)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `EMBED_DIR = ROOT / "pretrained" / "indobert" # mean pooling, khusus Indonesia` | 0.8660 ⭐ | LOLOS |
| `model = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=SEED)` | 0.8584 ⭐ | LOLOS |
| `model = LinearSVC(C=0.5)` | 0.8558 ⭐ | LOLOS |
| `model = RidgeClassifier(alpha=1.0)` | 0.8535 ⭐ | LOLOS |
| `model = SVC(C=10, random_state=SEED)` | 0.8509 ⭐ | LOLOS |
| `model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)` | 0.8400 | LOLOS |
| `EMBED_DIR = ROOT / "pretrained" / "distilbert_id" # lebih ringan` | 0.8355 | LOLOS |
| `model = KNeighborsClassifier(n_neighbors=15, metric="cosine")` | 0.8040 | LOLOS |

## `text/klasifikasi_finetune`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.9242**.

Aktif sekarang: `MODEL_DIR = ROOT / "pretrained" / "indobert"`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `MODEL_DIR = ROOT / "pretrained" / "distilbert_id"` | 0.9033 | LOLOS |
| `MODEL_DIR = ROOT / "pretrained" / "minilm" # multibahasa, kecil` | 0.8835 | LOLOS |

## `text/klasifikasi_tfidf`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.8636**.

Aktif sekarang: `model = LinearSVC(C=0.5)`  
Aktif sekarang: `vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = LogisticRegression(C=5, max_iter=2000)` | 0.8637 ⭐ | LOLOS |
| `model = RidgeClassifier(alpha=1.0)` | 0.8634 | LOLOS |
| `model = CalibratedClassifierCV(LinearSVC(C=0.5))` | 0.8633 | LOLOS |
| `vec = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), TfidfVectorizer(analyzer...` | 0.8590 | LOLOS |
| `model = SGDClassifier(loss="modified_huber", random_state=SEED)` | 0.8531 | LOLOS |
| `vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)` | 0.8403 | LOLOS |
| `vec = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True)` | 0.8335 | LOLOS |
| `model = ComplementNB(alpha=0.3)` | 0.8271 | LOLOS |
| `model = MultinomialNB(alpha=0.1)` | 0.8237 | LOLOS |
| `model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)` | 0.8044 | LOLOS |

## `text/multilabel`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.7264**.

Aktif sekarang: `model = OneVsRestClassifier(LogisticRegression(C=5, max_iter=2000))`  
Aktif sekarang: `vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)` | 0.7911 ⭐ | LOLOS |
| `model = OneVsRestClassifier(LinearSVC(C=0.5))` | 0.7882 ⭐ | LOLOS |
| `model = OneVsRestClassifier(RidgeClassifier(alpha=1.0))` | 0.7872 ⭐ | LOLOS |
| `model = ClassifierChain(LogisticRegression(C=5, max_iter=2000), random_state=SEED)` | 0.7666 ⭐ | LOLOS |
| `model = OneVsRestClassifier(SGDClassifier(loss="modified_huber", random_state=SEED))` | 0.7649 ⭐ | LOLOS |
| `model = OneVsRestClassifier(ComplementNB(alpha=0.3))` | 0.7512 ⭐ | LOLOS |
| `model = MultiOutputClassifier(LogisticRegression(C=5, max_iter=2000))` | 0.7264 | LOLOS |

## `text/ner`

Metrik: `F1 weighted` (makin besar makin baik). Asli: **0.6299**.

Aktif sekarang: `crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=...`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `crf = sklearn_crfsuite.CRF(algorithm="ap", max_iterations=50)` | 0.6460 ⭐ | LOLOS |
| `crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=0.01, max_iterations=200, all_possible_transition...` | 0.6372 ⭐ | LOLOS |
| `crf = sklearn_crfsuite.CRF(algorithm="l2sgd", c2=0.1, max_iterations=200)` | 0.6345 ⭐ | LOLOS |
| `crf = sklearn_crfsuite.CRF(algorithm="pa", max_iterations=50)` | 0.6318 ⭐ | LOLOS |

## `text/retrieval`

Metrik: `MRR` (makin besar makin baik). Asli: **0.8673**.

Aktif sekarang: `skor = np.vstack([bm.get_scores(q) for q in pertanyaan])`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `encoder = SentenceTransformer(str(ROOT / "pretrained" / "minilm")) emb_query = encoder.encode(list(query[TE...` | 0.9138 ⭐ | LOLOS |
| `vec = TfidfVectorizer(stop_words="english", sublinear_tf=True) vec.fit(corpus[TEXT]) skor = cosine_similari...` | 0.8783 ⭐ | LOLOS |
| `vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True) vec.fit(corpus[TEXT]) skor...` | 0.8694 ⭐ | LOLOS |

## `text/topik`

Metrik: `ARI` (makin besar makin baik). Asli: **0.3943**.

Aktif sekarang: `vec = CountVectorizer(stop_words="english", min_df=3, max_df=0.5)`  
Aktif sekarang: `model = LatentDirichletAllocation(n_components=K, learning_method="batch", max_iter=30, random_state=SEED )`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `vec = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5) model = NMF(n_components=K, init="nndsvda...` | 0.4384 ⭐ | LOLOS |
| `model = NMF(n_components=K, beta_loss="kullback-leibler", solver="mu", max_iter=500, random_state=SEED)` | 0.4139 ⭐ | LOLOS |
| `model = LatentDirichletAllocation(n_components=K, learning_method="online", random_state=SEED)` | 0.4031 ⭐ | LOLOS |
| `model = TruncatedSVD(n_components=K, random_state=SEED) # LSA` | 0.0085 | LOLOS |
| `vec = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5)` | -1.8043 | LOLOS |

## `timeseries/anomali`

Metrik: `PR-AUC` (makin besar makin baik). Asli: **0.495**.

Aktif sekarang: `skor = -IsolationForest(n_estimators=200, random_state=SEED).fit(X_latih).score_samples(X_uji)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `skor = KNN(n_neighbors=10).fit(X_latih).decision_function(X_uji) # PyOD` | 0.7620 ⭐ | LOLOS |
| `skor = -LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_latih).score_samples(X_uji)` | 0.7490 ⭐ | LOLOS |
| `skor = -OneClassSVM(nu=0.05, gamma="scale").fit(X_latih).score_samples(X_uji)` | 0.5600 ⭐ | LOLOS |
| `skor = HBOS().fit(X_latih).decision_function(X_uji) # PyOD` | 0.4340 | LOLOS |
| `skor = ECOD().fit(X_latih).decision_function(X_uji) # PyOD` | 0.3140 | LOLOS |
| `skor = z_skor.abs().loc[X_uji.index].to_numpy()` | 0.2580 | LOLOS |
| `skor = -EllipticEnvelope(random_state=SEED).fit(X_latih).score_samples(X_uji)` | 0.2470 | LOLOS |

## `timeseries/klasifikasi`

Metrik: `akurasi` (makin besar makin baik). Asli: **0.9333**.

Aktif sekarang: `model = RandomForestClassifier(n_estimators=500, random_state=SEED)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = RocketClassifier(rocket_transform="minirocket", random_state=SEED)` | 0.9933 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))` | 0.9600 ⭐ | LOLOS |
| `model = TimeSeriesForestClassifier(n_estimators=200, random_state=SEED)` | 0.9533 ⭐ | LOLOS |
| `model = ExtraTreesClassifier(n_estimators=500, random_state=SEED)` | 0.9333 | LOLOS |
| `model = KNeighborsClassifier(n_neighbors=1)` | 0.9133 | LOLOS |
| `model = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")` | 0.9067 | LOLOS |
| `model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))` | 0.8533 | LOLOS |
| `model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)` | 0.8533 | LOLOS |

## `timeseries/peramalan_exog`

Metrik: `Model SARIMAX MAE` (makin kecil makin baik). Asli: **94.44**.

Aktif sekarang: `model = SARIMAX(y_latih, exog=x_latih, order=(1, 1, 1))`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = SARIMAX(y_latih, exog=x_latih, order=(1, 0, 1), seasonal_order=(0, 1, 1, PERIODE))` | 67.1000 ⭐ | LOLOS |
| `model = SARIMAX(y_latih, exog=x_latih, order=(1, 1, 1), trend="t")` | 121.9400 | LOLOS |
| `model = SARIMAX(y_latih, exog=x_latih, order=(2, 1, 0))` | 129.1400 | LOLOS |
| `model = SARIMAX(y_latih, exog=x_latih, order=(0, 1, 1))` | 250.0300 | LOLOS |

## `timeseries/peramalan_ml`

Metrik: `MAE ` (makin kecil makin baik). Asli: **1.77**.

Aktif sekarang: `model = RandomForestRegressor(n_estimators=300, random_state=SEED)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))` | 1.1500 ⭐ | LOLOS |
| `model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED)` | 1.6600 ⭐ | LOLOS |
| `model = ExtraTreesRegressor(n_estimators=300, random_state=SEED)` | 1.8100 | LOLOS |
| `model = CatBoostRegressor(iterations=300, verbose=0, allow_writing_files=False, random_seed=SEED)` | 2.5400 | LOLOS |
| `model = make_pipeline(StandardScaler(), SVR(C=10))` | 2.6200 | LOLOS |
| `model = KNeighborsRegressor(n_neighbors=5)` | 2.7300 | LOLOS |
| `model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)` | 3.1300 | LOLOS |
| `model = HistGradientBoostingRegressor(random_state=SEED)` | 3.1500 | LOLOS |

## `timeseries/peramalan_statistik`

Metrik: `Model       MAE` (makin kecil makin baik). Asli: **0.83**.

Aktif sekarang: `model = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=PERIODE)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = ThetaModel(data, period=PERIODE)` | 0.6700 ⭐ | LOLOS |
| `model = ExponentialSmoothing(data, trend="add", damped_trend=True, seasonal="add", seasonal_periods=PERIODE)` | 0.8300 | LOLOS |
| `model = SARIMAX(data, order=(2, 1, 2), seasonal_order=(0, 1, 1, PERIODE))` | 0.9300 | LOLOS |
| `model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, PERIODE))` | 1.0000 | LOLOS |
| `model = ExponentialSmoothing(data, trend="add", seasonal="mul", seasonal_periods=PERIODE)` | 1.0400 | LOLOS |
| `model = STLForecast(data, ARIMA, model_kwargs={"order": (1, 1, 0)}, period=PERIODE)` | 1.6800 | LOLOS |
| `model = Holt(data, damped_trend=True, initialization_method="estimated")` | 2.5300 | LOLOS |
| `model = SimpleExpSmoothing(data, initialization_method="estimated")` | 2.8400 | LOLOS |

## `unsupervised/anomali`

Metrik: `ROC-AUC` (makin besar makin baik). Asli: **0.957**.

Aktif sekarang: `model = IsolationForest(contamination=CONT, random_state=SEED)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = COPOD(contamination=CONT) # PyOD pred = model.predict(X_skala) # PyOD: 1 = anomali skor = model.dec...` | 0.9670 ⭐ | LOLOS |
| `model = IForest(contamination=CONT, random_state=SEED) # PyOD pred = model.predict(X_skala) # PyOD: 1 = ano...` | 0.9570 | LOLOS |
| `model = HBOS(contamination=CONT) # PyOD pred = model.predict(X_skala) # PyOD: 1 = anomali skor = model.deci...` | 0.9540 | LOLOS |
| `model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=CONT)` | 0.9520 | LOLOS |
| `model = EllipticEnvelope(contamination=CONT, random_state=SEED)` | 0.9430 | LOLOS |
| `model = KNN(contamination=CONT) # PyOD pred = model.predict(X_skala) # PyOD: 1 = anomali skor = model.decis...` | 0.9300 | LOLOS |
| `model = ECOD(contamination=CONT) # PyOD pred = model.predict(X_skala) # PyOD: 1 = anomali skor = model.deci...` | 0.9130 | LOLOS |
| `model = OneClassSVM(nu=CONT, gamma="scale")` | 0.8970 | LOLOS |

## `unsupervised/aturan_asosiasi`

Metrik: `jumlah aturan` (bukan lomba skor, hanya pembanding). Asli: **234.0**.

Aktif sekarang: `itemset = apriori(onehot, min_support=MIN_SUP, use_colnames=True)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `itemset = fpgrowth(onehot, min_support=MIN_SUP, use_colnames=True)` | 234.0000 | LOLOS |
| `itemset = hmine(onehot, min_support=MIN_SUP, use_colnames=True)` | 234.0000 | LOLOS |

## `unsupervised/clustering`

Metrik: `ARI vs label asli` (makin besar makin baik). Asli: **0.897**.

> Data circles/moons: ganti juga `LABEL = "true"` dan `K = 2`.

Aktif sekarang: `model = KMeans(n_clusters=K, n_init=10, random_state=SEED)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = GaussianMixture(n_components=K, random_state=SEED)` | 0.8970 | LOLOS |
| `model = MiniBatchKMeans(n_clusters=K, n_init=10, random_state=SEED)` | 0.8970 | LOLOS |
| `model = SpectralClustering(n_clusters=K, affinity="nearest_neighbors", random_state=SEED)` | 0.8800 | LOLOS |
| `model = AgglomerativeClustering(n_clusters=K, linkage="ward")` | 0.7900 | LOLOS |
| `model = Birch(n_clusters=K)` | 0.7900 | LOLOS |
| `model = BisectingKMeans(n_clusters=K, random_state=SEED)` | 0.6910 | LOLOS |
| `DATA = ROOT / "data_latihan" / "unsupervised" / "moons_SINTETIS.csv"` | 0.4800 | LOLOS (data lain, skor tidak sebanding) |
| `model = OPTICS(min_samples=8, xi=0.01)` | 0.4140 | LOLOS |
| `model = DBSCAN(eps=2.0, min_samples=8) # wine: 3 cluster, banyak noise (eps perlu dicari)` | 0.2880 | LOLOS |
| `model = AffinityPropagation(random_state=SEED)` | 0.2730 | LOLOS |
| `model = HDBSCAN(min_cluster_size=10)` | 0.2600 | LOLOS |
| `DATA = ROOT / "data_latihan" / "unsupervised" / "circles_SINTETIS.csv"` | -0.0010 | LOLOS (data lain, skor tidak sebanding) |
| `model = MeanShift()` | -0.0060 | LOLOS |

## `unsupervised/imputasi`

Metrik: `RMSE imputasi (skala standar)` (makin kecil makin baik). Asli: **1.025**.

Aktif sekarang: `imputer = SimpleImputer(strategy="mean")`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `estimator = ExtraTreesRegressor(n_estimators=100, random_state=SEED) imputer = IterativeImputer(estimator=e...` | 0.6260 ⭐ | LOLOS |
| `imputer = KNNImputer(n_neighbors=10, weights="distance")` | 0.6630 ⭐ | LOLOS |
| `imputer = KNNImputer(n_neighbors=5)` | 0.6720 ⭐ | LOLOS |
| `imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=SEED)` | 0.7110 ⭐ | LOLOS |
| `estimator = LGBMRegressor(n_estimators=100, verbose=-1, random_state=SEED) imputer = IterativeImputer(estim...` | 0.7130 ⭐ | LOLOS |
| `imputer = SimpleImputer(strategy="median")` | 1.0530 | LOLOS |
| `imputer = SimpleImputer(strategy="most_frequent")` | 1.2970 | LOLOS |

## `unsupervised/preprocessing`

Metrik: `ARI sesudah` (makin besar makin baik). Asli: **0.897**.

Aktif sekarang: `scaler = StandardScaler()`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `scaler = QuantileTransformer(output_distribution="uniform", n_quantiles=100, random_state=SEED)` | 0.9330 ⭐ | LOLOS |
| `scaler = RobustScaler()` | 0.8950 | LOLOS |
| `scaler = PowerTransformer(method="yeo-johnson")` | 0.8820 | LOLOS |
| `scaler = PowerTransformer(method="box-cox")` | 0.8820 | LOLOS |
| `scaler = MinMaxScaler()` | 0.8540 | LOLOS |
| `scaler = QuantileTransformer(output_distribution="normal", n_quantiles=100, random_state=SEED)` | 0.8300 | LOLOS |
| `scaler = MaxAbsScaler()` | 0.7550 | LOLOS |
| `scaler = Normalizer()` | 0.3470 | LOLOS |

## `unsupervised/reduksi_dimensi`

Metrik: `silhouette embedding` (makin besar makin baik). Asli: **0.055**.

> LDA: aktifkan juga `emb = model.fit_transform(X_skala, df[LABEL])`.

Aktif sekarang: `model = PCA(n_components=2, random_state=SEED)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = umap.UMAP(n_components=2, random_state=SEED)` | 0.5890 ⭐ | LOLOS |
| `model = TSNE(n_components=2, perplexity=30, random_state=SEED)` | 0.4850 ⭐ | LOLOS |
| `model = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=SEED)` | 0.2750 ⭐ | LOLOS |
| `model = Isomap(n_components=2, n_neighbors=10)` | 0.2250 ⭐ | LOLOS |
| `model = LinearDiscriminantAnalysis(n_components=2) emb = model.fit_transform(X_skala, df[LABEL]) # LDA butu...` | 0.1880 ⭐ | LOLOS |
| `model = KernelPCA(n_components=2, kernel="rbf", random_state=SEED)` | 0.0840 ⭐ | LOLOS |
| `model = FastICA(n_components=2, random_state=SEED)` | 0.0560 ⭐ | LOLOS |
| `model = TruncatedSVD(n_components=2, random_state=SEED)` | 0.0550 | LOLOS |
| `model = MDS(n_components=2, random_state=SEED)` | 0.0490 | LOLOS |
| `model = FactorAnalysis(n_components=2, random_state=SEED)` | -0.0120 | LOLOS |

## `unsupervised/seleksi_fitur`

Metrik: `F1 CV K fitur` (makin besar makin baik). Asli: **0.945**.

Aktif sekarang: `pilih = SelectKBest(f_classif, k=K)`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `pilih = RFE(LogisticRegression(max_iter=5000), n_features_to_select=K)` | 0.9560 ⭐ | LOLOS |
| `lr_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1) pilih = SelectFromModel(lr_l1, max_feat...` | 0.9520 ⭐ | LOLOS |
| `pilih = SequentialFeatureSelector(LogisticRegression(max_iter=5000), n_features_to_select=K)` | 0.9520 ⭐ | LOLOS |
| `hutan = RandomForestClassifier(n_estimators=300, random_state=SEED) pilih = SelectFromModel(hutan, max_feat...` | 0.9480 ⭐ | LOLOS |
| `pilih = SelectKBest(mutual_info_classif, k=K)` | 0.9400 | LOLOS |

## `vision/klasifikasi_clip`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.8586**.

Aktif sekarang: `TEMPLATE = "a photo of a {}"`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `TEMPLATE = "a low-resolution photo of a {}"` | 0.8662 ⭐ | LOLOS |
| `TEMPLATE = "a photo of the small {}"` | 0.8566 | LOLOS |
| `TEMPLATE = "a blurry picture of a {}"` | 0.8426 | LOLOS |

## `vision/klasifikasi_cnn`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.674**.

Backbone aktif sekarang: `resnet18`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| backbone `convnext_tiny` | 0.773 ⭐ | LOLOS |
| backbone `efficientnet_b0` | 0.737 ⭐ | LOLOS |
| backbone `resnet50` | 0.733 ⭐ | LOLOS |
| backbone `mobilenet_v3_small` | 0.625 | LOLOS |

## `vision/klasifikasi_embedding`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.887**.

Aktif sekarang: `model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))`  
Backbone aktif sekarang: `resnet50`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| backbone `vit_b_16` | 0.908 ⭐ | LOLOS |
| backbone `convnext_tiny` | 0.895 ⭐ | LOLOS |
| backbone `efficientnet_b0` | 0.87 | LOLOS |
| backbone `mobilenet_v3_small` | 0.835 | LOLOS |
| backbone `resnet18` | 0.823 | LOLOS |
| `model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))` | 0.8870 ⭐ | LOLOS |
| `model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)` | 0.8668 | LOLOS |
| `model = make_pipeline(StandardScaler(), LinearSVC(C=0.01))` | 0.8535 | LOLOS |
| `model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))` | 0.8532 | LOLOS |
| `model = KNeighborsClassifier(n_neighbors=10, metric="cosine")` | 0.8431 | LOLOS |

## `vision/klasifikasi_klasik`

Metrik: `F1 macro` (makin besar makin baik). Asli: **0.4531**.

Aktif sekarang: `model = make_pipeline(StandardScaler(), LinearSVC(C=0.1))`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| `model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))` | 0.5589 ⭐ | LOLOS |
| `model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)` | 0.5483 ⭐ | LOLOS |
| `model = ExtraTreesClassifier(n_estimators=300, random_state=SEED)` | 0.4975 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))` | 0.4706 ⭐ | LOLOS |
| `model = RandomForestClassifier(n_estimators=300, random_state=SEED)` | 0.4641 ⭐ | LOLOS |
| `model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))` | 0.4278 | LOLOS |

## `vision/retrieval`

Metrik: `Recall@1` (makin besar makin baik). Asli: **0.796**.

Backbone aktif sekarang: `resnet50`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| backbone `vit_b_16` | 0.901 ⭐ | LOLOS |
| backbone `convnext_tiny` | 0.869 ⭐ | LOLOS |
| backbone `efficientnet_b0` | 0.828 ⭐ | LOLOS |
| backbone `resnet18` | 0.762 | LOLOS |

## `vision/segmentasi`

Metrik: `mean IoU (validasi)` (makin besar makin baik). Asli: **0.658**.

Backbone aktif sekarang: `lraspp_mobilenet_v3_large`  

| Alternatif (hapus `#`) | Skor | Status |
|---|---|---|
| backbone `deeplabv3_mobilenet_v3_large` | 0.668 ⭐ | LOLOS |

---

# Tambahan 18 Okt — alternatif di notebook baru

Diuji dengan `_alat/uji_varian.py`: tiap baris `# nama = ...` diaktifkan satu per satu, lalu
seluruh notebook dijalankan (hasil mentah: `_alat/log/varian.txt`). 72 alternatif: **71 OK**,
1 LAMA (lebih dari 15 menit karena CPU sedang penuh, bukan error).

| Notebook | Asli | Alternatif terbaik (hapus `#`) | Catatan |
|---|---|---|---|
| `graph/gcn` | F1 weighted 0.80 | `A_pakai = A_rata` 0.80 | sama |
| `graph/jalur_terpendek` | 15367 km | A*, Bellman-Ford, shortest_path: 15367 km | TSP greedy / annealing: 18916 km |
| `graph/klasifikasi_graf` | F1 weighted 0.88 | ⭐ `LGBMClassifier` 0.90 | WL saja 0.88, statistik saja 0.87 |
| `graph/klasifikasi_node` | F1 weighted 0.73 | ⭐ ExtraTrees / pipeline 0.78 | `LGBMClassifier`: LAMA (uji ulang saat CPU longgar) |
| `graph/prediksi_link` | AP 0.827 | ⭐ `XGBClassifier` 0.837 | LGBM 0.831, HistGB 0.831, ExtraTrees 0.818 |
| `timeseries/deep_lstm` | loss latih 0.142 | ⭐ `nn.GRU` 0.124 | `nn.RNN` 0.186 (loss latih, bukan skor uji) |
| `timeseries/peramalan_global` | MAPE 3.391 | HistGB 3.394 | CatBoost 3.484, LGBM 3.507, XGB 3.610 |
| `timeseries/peramalan_ml` | MAPE 0.48 | ⭐ `Ridge` MAPE 0.31 | ⭐ XGB 0.45, ExtraTrees 0.49, CatBoost 0.69 |
| `timeseries/rul` | RMSE 12.14 | HistGB 12.55 | ExtraTrees 12.70, RF 12.73, CatBoost 13.61, Ridge 16.94 |
| `timeseries/titik_perubahan` | selisih 1 tahun | Binseg, Window, Dynp, Pelt-rbf: 1 tahun | BottomUp: 4 tahun |
| `timeseries/klasifikasi` | akurasi 0.93 | ⭐ MiniRocket 0.99 | sama dengan tabel di atas |
| `timeseries/anomali`, `clustering`, `intermittent`, `var_multivariat` | — | semua alternatif OK | skor ada di tabel perbandingan di dalam notebook |

**Perubahan model aktif:** `tabular/semi_supervised` kini memakai LGBM `class_weight="balanced"`
(F1 macro test 0.431 → 0.461).

**Catatan LightGBM / XGBoost:** jika CPU sedang dipakai proses lain, pengaturan thread bawaan
bisa ratusan kali lebih lambat. Tulis `n_jobs=4` (atau jumlah inti fisik) bila notebook terasa macet.
