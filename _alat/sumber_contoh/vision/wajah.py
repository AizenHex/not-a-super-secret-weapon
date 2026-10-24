# %% [markdown]
# # Vision — Wajah: Deteksi (YuNet) + Pengenalan / Verifikasi (SFace)
# **Kapan dipakai:** soal tentang wajah: hitung/temukan wajah di foto, "apakah dua foto ini
# orang yang sama?" (verifikasi), atau "ini siapa?" (identifikasi dari galeri).
#
# **Langkah:** `cv2.FaceDetectorYN` (YuNet, 0.2 MB) mencari kotak wajah + 5 titik (mata, hidung,
# mulut) → `cv2.FaceRecognizerSF` (SFace) meluruskan wajah dan mengubahnya jadi vektor 128 angka
# → cosine antar vektor → "orang sama" bila cosine di atas ambang (dipilih di data latih).
# Ringan: jalan di CPU tanpa PyTorch. Model ONNX ada di `pretrained/opencv_wajah/`.
#
# **Data demo:** LFW (Labeled Faces in the Wild), 400 pasangan latih + 400 pasangan test.

# %%
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/vision/wajah_lfw"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/pasangan_train.csv"
TEST_CSV = f"{DATA_FOLDER}/pasangan_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/pasangan_test_labels.csv"   # hanya ada di data latihan
IMAGE_FOLDER = f"{DATA_FOLDER}/gambar"
MODEL_FOLDER = "../../../pretrained/opencv_wajah"
DETECTOR_ONNX = f"{MODEL_FOLDER}/face_detection_yunet_2023mar.onnx"
RECOGNIZER_ONNX = f"{MODEL_FOLDER}/face_recognition_sface_2021dec.onnx"

IMAGE_A_COL = "gambar_a"   # nama file di IMAGE_FOLDER
IMAGE_B_COL = "gambar_b"
TARGET = "sama"
MIN_SCORE = 0.7        # keyakinan minimal YuNet. Turunkan (0.5) jika ada wajah terlewat
SFACE_THRESHOLD = 0.363   # ambang saran SFace (pembanding)

# %% [markdown]
# ## Baca data + muat model

# %%
detector = cv2.FaceDetectorYN.create(DETECTOR_ONNX, "", (320, 320), MIN_SCORE, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_ONNX, "")

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())

# %% [markdown]
# ## Deteksi wajah: kotak + 5 titik
# Tiap baris hasil `detect`: kiri, atas, lebar, tinggi, 10 angka titik (mata, hidung, mulut),
# skor. Demo: kolase 6 wajah dengan ukuran berbeda-beda di satu kanvas.

# %%
canvas = np.full((520, 900, 3), 200, dtype=np.uint8)
photos = sorted(train[IMAGE_A_COL])[:6]
places = [(20, 20, 200), (260, 40, 150), (460, 20, 250),
          (30, 280, 120), (240, 270, 220), (560, 300, 170)]   # (kiri, atas, ukuran)
for photo, (left, top, size) in zip(photos, places):
    face = cv2.imread(f"{IMAGE_FOLDER}/{photo}")
    canvas[top:top + size, left:left + size] = cv2.resize(face, (size, size))

detector.setInputSize((canvas.shape[1], canvas.shape[0]))
status, detections = detector.detect(canvas)
print("wajah ditempel: 6 | terdeteksi:", len(detections))
for detection in detections:
    left, top, width, height = detection[:4].astype(int)
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (0, 255, 0), 2)
    for point_number in range(5):
        point = detection[4 + 2 * point_number:6 + 2 * point_number].astype(int)
        cv2.circle(canvas, (point[0], point[1]), 3, (0, 0, 255), -1)
cv2.imwrite("deteksi_kolase.png", canvas)

# %% [markdown]
# ## Embedding wajah tiap foto
# Satu foto = satu vektor 128 angka, panjang 1. Foto tanpa wajah terdeteksi diberi vektor nol
# (cosine 0) dan dihitung supaya terlihat.

# %%
def face_vector(photo):
    image = cv2.imread(f"{IMAGE_FOLDER}/{photo}")
    detector.setInputSize((image.shape[1], image.shape[0]))
    status, faces = detector.detect(image)
    if faces is None:
        return None
    main_face = faces[faces[:, -1].argmax()]     # skor tertinggi = wajah utama foto
    aligned = recognizer.alignCrop(image, main_face)
    vector = recognizer.feature(aligned)[0]
    return vector / np.linalg.norm(vector)


all_photos = set(train[IMAGE_A_COL]) | set(train[IMAGE_B_COL])
all_photos = sorted(all_photos | set(test[IMAGE_A_COL]) | set(test[IMAGE_B_COL]))
vectors = {}
no_face = 0
for photo in all_photos:
    vectors[photo] = face_vector(photo)
    if vectors[photo] is None:
        vectors[photo] = np.zeros(128)
        no_face = no_face + 1
print("foto:", len(all_photos), "| tanpa wajah terdeteksi:", no_face)

# %% [markdown]
# ## Verifikasi pasangan: cosine + ambang
# Ambang dicari di pasangan latih (akurasi tertinggi).

# %%
def pair_cosine(table):
    cosines = []
    for photo_a, photo_b in zip(table[IMAGE_A_COL], table[IMAGE_B_COL]):
        cosines.append(np.dot(vectors[photo_a], vectors[photo_b]))
    return np.array(cosines)


train_cosine = pair_cosine(train)
print("AUC pasangan latih:", round(roc_auc_score(train[TARGET], train_cosine), 3))

best_accuracy = 0
threshold = SFACE_THRESHOLD
for candidate in np.arange(0.0, 0.8, 0.01):
    accuracy = accuracy_score(train[TARGET], train_cosine > candidate)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        threshold = candidate
print("ambang terbaik:", round(threshold, 2), "| akurasi latih:", round(best_accuracy, 3))

# %% [markdown]
# ## Prediksi test + submission

# %%
test_cosine = pair_cosine(test)
predictions = (test_cosine > threshold).astype(int)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["sama"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)[TARGET]
print("AUC test     :", round(roc_auc_score(test_key, test_cosine), 3))
print("akurasi test :", round(accuracy_score(test_key, predictions), 3))
print("akurasi dengan ambang SFace:",
      round(accuracy_score(test_key, test_cosine > SFACE_THRESHOLD), 3))

# %% [markdown]
# ## Identifikasi 1:N — "ini siapa?" (demo, memakai kunci)
# Galeri = foto pertama tiap pasangan "sama". Foto kedua dicocokkan ke seluruh galeri (cosine
# tertinggi). Nama orang = bagian awal nama file (sebelum "__").

# %%
same_pairs = test[test_key == 1]
gallery_photos = list(same_pairs[IMAGE_A_COL])
gallery = []
for photo in gallery_photos:
    gallery.append(vectors[photo])
gallery = np.array(gallery)

correct = 0
for photo in same_pairs[IMAGE_B_COL]:
    similarity = gallery @ vectors[photo]     # cosine ke semua foto di galeri
    guess_name = gallery_photos[similarity.argmax()].split("__")[0]
    true_name = photo.split("__")[0]
    if guess_name == true_name:
        correct = correct + 1
print("akurasi top-1 dari", len(gallery), "foto di galeri:", round(correct / len(gallery), 3))
