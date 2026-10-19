# %% [markdown]
# # Video — Deteksi Anomali per Frame
# **Kapan dipakai:** hanya ada rekaman kondisi **normal** (CCTV jalan, ban berjalan) dan
# diminta menandai frame/waktu yang janggal (sepeda di jalur pejalan kaki, benda jatuh).
#
# **Idenya:** tiap frame diubah jadi embedding oleh CNN pretrained yang tidak dilatih ulang.
# Skor anomali = seberapa jauh embedding frame dari rata-rata embedding frame normal. Skor
# dihaluskan beberapa frame, karena kejadian asli berlangsung lebih dari satu frame.
#
# **Langkah:**
# 1. Embedding semua frame video normal dan video uji.
# 2. Skor anomali tiap frame, dihaluskan, ambang dari skor frame normal.
# 3. Lihat frame dengan skor tertinggi, simpan skor per frame.
#
# **Data demo:** UCSD Ped2 (pejalan kaki, anomali = sepeda, skateboard, mobil). Video normal =
# 16 klip, video uji = 12 klip berurutan. Label per frame hanya untuk cek.

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/video/anomali_ped2"   # GANTI saat lomba
NORMAL_VIDEO = f"{DATA_FOLDER}/latih_normal.mp4"
TEST_VIDEO = f"{DATA_FOLDER}/uji.mp4"
TEST_KEY_CSV = f"{DATA_FOLDER}/uji_label.csv"      # hanya ada di data latihan
PRETRAINED = "Z:/GEMASTIK-DATABASE/pretrained"

FRAME_HEIGHT = 240        # ukuran frame saat masuk CNN
FRAME_WIDTH = 360
GRID = 3                  # peta fitur dirata-rata per kotak 3x3 (posisi objek tetap tersimpan)
BATCH_SIZE = 64
SMOOTH_FRAMES = 5         # rata-rata bergerak sepanjang 5 frame
THRESHOLD_PERCENTILE = 95   # ambang = persentil 95 skor frame normal
TOP_N = 6                 # jumlah frame skor tertinggi yang ditampilkan
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406])
MEAN = MEAN.view(1, 3, 1, 1).to(DEVICE)    # normalisasi warna ImageNet
STD = torch.tensor([0.229, 0.224, 0.225])
STD = STD.view(1, 3, 1, 1).to(DEVICE)

# %% [markdown]
# ## Lihat satu frame

# %%
video = cv2.VideoCapture(TEST_VIDEO)
video.set(cv2.CAP_PROP_POS_FRAMES, 100)
ok, frame = video.read()
plt.imshow(frame[:, :, ::-1])               # BGR (OpenCV) → RGB
plt.axis("off")
plt.show()

# %% [markdown]
# ## Langkah 1 — embedding tiap frame
# ResNet18 tanpa 2 lapisan terakhir = peta fitur. Peta dirata-rata per kotak `GRID` x `GRID`.
# Video dibaca bertahap, `BATCH_SIZE` frame sekali, supaya memori cukup.

# %%
network = models.resnet18(weights=None)
network.load_state_dict(torch.load(f"{PRETRAINED}/resnet18/model.pth", map_location="cpu"))
layers = list(network.children())
backbone = nn.Sequential(*layers[:-2])       # buang avgpool + fc
backbone = backbone.to(DEVICE)
backbone.eval()


def embed_frames(frames):
    batch = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)
    batch = batch.float().to(DEVICE) / 255
    batch = functional.interpolate(batch, size=(FRAME_HEIGHT, FRAME_WIDTH), mode="bilinear")
    with torch.no_grad():
        feature_map = backbone((batch - MEAN) / STD)
    pooled = functional.adaptive_avg_pool2d(feature_map, GRID)
    pooled = pooled.flatten(1)
    return pooled.cpu().numpy()


# coba pada frame tadi
print(embed_frames([frame[:, :, ::-1].copy()]).shape)

# %%
def embed_video(path):
    video = cv2.VideoCapture(path)
    batches = []
    frames = []
    ok, frame = video.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) == BATCH_SIZE:
            batches.append(embed_frames(frames))
            frames = []
        ok, frame = video.read()
    if len(frames) > 0:
        batches.append(embed_frames(frames))
    return np.vstack(batches)


normal_embeddings = embed_video(NORMAL_VIDEO)
test_embeddings = embed_video(TEST_VIDEO)
print("normal:", normal_embeddings.shape, "| uji:", test_embeddings.shape)

# %% [markdown]
# ## Langkah 2 — skor anomali
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok lain.
# Blok aktif: jarak ke rata-rata normal, tiap feature dibagi simpangan bakunya.

# %%
normal_mean = normal_embeddings.mean(axis=0)
normal_std = normal_embeddings.std(axis=0) + 1e-6       # + 1e-6 supaya tidak dibagi nol
normal_scores = np.linalg.norm((normal_embeddings - normal_mean) / normal_std, axis=1)
test_scores = np.linalg.norm((test_embeddings - normal_mean) / normal_std, axis=1)

# model = IForest(n_estimators=200, random_state=42)      # pyod
# model = KNN(n_neighbors=5)                              # jarak ke frame normal terdekat
# model.fit(normal_embeddings)
# normal_scores = model.decision_function(normal_embeddings)
# test_scores = model.decision_function(test_embeddings)

score_series = pd.Series(test_scores)
smooth = score_series.rolling(SMOOTH_FRAMES, center=True, min_periods=1).mean()
smooth_scores = smooth.to_numpy()
threshold = np.percentile(normal_scores, THRESHOLD_PERCENTILE)
flags = smooth_scores > threshold
print("ambang:", round(threshold, 2), "| frame ditandai:", flags.sum(), "dari", len(flags))

plt.figure(figsize=(12, 4))
plt.plot(test_scores, alpha=0.4, label="skor mentah")
plt.plot(smooth_scores, label="skor dihaluskan")
plt.axhline(threshold, color="red", linestyle="--", label="ambang")
plt.xlabel("frame uji")
plt.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## Langkah 3a — frame dengan skor tertinggi
# Frame yang berjarak < 30 dari frame yang sudah dipilih dilewati (biasanya kejadian yang sama).

# %%
chosen = []
for number in np.argsort(-smooth_scores):
    is_far = True
    for previous in chosen:
        if abs(number - previous) < 30:
            is_far = False
    if is_far:
        chosen.append(number)
    if len(chosen) == TOP_N:
        break

video = cv2.VideoCapture(TEST_VIDEO)
fig, axes = plt.subplots(2, 3, figsize=(10, 5))
axes = axes.flatten()
for i in range(len(chosen)):
    video.set(cv2.CAP_PROP_POS_FRAMES, int(chosen[i]))
    ok, frame = video.read()
    axes[i].imshow(frame[:, :, ::-1])
    axes[i].set_title(f"frame {chosen[i]} skor {smooth_scores[chosen[i]]:.1f}", fontsize=8)
    axes[i].axis("off")
plt.show()

# %% [markdown]
# ## Langkah 3b — simpan skor per frame

# %%
submission = pd.DataFrame()
submission["frame"] = np.arange(len(smooth_scores))
submission["anomaly_score"] = smooth_scores
submission["is_anomaly"] = flags.astype(int)
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# AUC = seberapa sering frame anomali diberi skor lebih tinggi dari frame normal (1 = sempurna).

# %%
truth = pd.read_csv(TEST_KEY_CSV)["anomali"]
is_true_anomaly = (truth == 1).to_numpy()
print("AUC mentah    :", round(roc_auc_score(truth, test_scores), 4))
print("AUC dihaluskan:", round(roc_auc_score(truth, smooth_scores), 4))
print("recall anomali:", round(flags[is_true_anomaly].mean(), 3))
print("alarm palsu di frame normal:", round(flags[~is_true_anomaly].mean(), 3))
