# %% [markdown]
# # Baca anotasi kotak → tabel standar
# Semua format diubah ke dua tabel yang sama, supaya potongan lain bisa langsung dipakai:
# - `images`: satu baris per gambar → `path, width, height`
# - `boxes` : satu baris per kotak  → `path, label, x1, y1, x2, y2` (piksel, pojok kiri-atas dan
#   kanan-bawah)
#
# **Butuh:** file anotasi soal + folder gambarnya.
**Menghasilkan:** tabel `images`, `boxes` (di memori) → dipakai `anotasi_tulis.ipynb`.

## Cara menempel ke notebook-mu
1. Lihat format anotasi soal (buka satu file anotasinya). Salin **SATU** sel baca yang cocok:
   COCO json / YOLO txt / Pascal VOC xml / CVAT xml. Jangan salin semuanya.
2. Ubah baris bertanda `# UBAH` (path).
3. Salin sel "Cek isi" dan "Lihat" di bawahnya (dianjurkan), lalu lanjut ke `anotasi_tulis.ipynb`.

# %% [markdown]
# ## COCO json (Roboflow, CVAT "COCO 1.0", Label Studio)
# Ubah: `COCO_JSON`, `IMAGE_FOLDER`. Kotak COCO = `[x, y, lebar, tinggi]`.

# %%
import json

import pandas as pd

COCO_JSON = "data/train/_annotations.coco.json"   # UBAH
IMAGE_FOLDER = "data/train"                         # UBAH: folder tempat file_name berada

with open(COCO_JSON, encoding="utf-8") as file:
    coco = json.load(file)
category_name = {}
for category in coco["categories"]:
    category_name[category["id"]] = category["name"]
image_path = {}
image_rows = []
for image in coco["images"]:
    image_path[image["id"]] = f"{IMAGE_FOLDER}/{image['file_name']}"
    image_rows.append([image_path[image["id"]], image["width"], image["height"]])
box_rows = []
for annotation in coco["annotations"]:
    left, top, width, height = annotation["bbox"]
    box_rows.append([image_path[annotation["image_id"]], category_name[annotation["category_id"]],
                     left, top, left + width, top + height])
images = pd.DataFrame(image_rows, columns=["path", "width", "height"])
boxes = pd.DataFrame(box_rows, columns=["path", "label", "x1", "y1", "x2", "y2"])
print("gambar:", len(images), "| kotak:", len(boxes))
print(boxes["label"].value_counts())


# %% [markdown]
# ## YOLO txt (satu .txt per gambar: `kelas x_tengah y_tengah lebar tinggi`, nilai 0–1)
# Ubah: folder gambar, folder label, `CLASS_NAMES` (urutan = nomor kelas, lihat data.yaml).

# %%
import glob
import os

import pandas as pd
from PIL import Image

IMAGE_FOLDER = "data/images/train"    # UBAH
LABEL_FOLDER = "data/labels/train"    # UBAH
CLASS_NAMES = ["entete", "vin"]       # UBAH: nomor kelas 0, 1, ...

image_rows = []
box_rows = []
for path in sorted(glob.glob(f"{IMAGE_FOLDER}/*")):
    width, height = Image.open(path).size
    image_rows.append([path, width, height])
    label_file = f"{LABEL_FOLDER}/{os.path.splitext(os.path.basename(path))[0]}.txt"
    if not os.path.exists(label_file):
        continue                      # gambar tanpa kotak
    for line in open(label_file).read().split("\n"):
        if line.strip() == "":
            continue
        number, x_center, y_center, box_width, box_height = line.split()[:5]
        x_center, box_width = float(x_center) * width, float(box_width) * width
        y_center, box_height = float(y_center) * height, float(box_height) * height
        box_rows.append([path, CLASS_NAMES[int(number)], x_center - box_width / 2,
                         y_center - box_height / 2, x_center + box_width / 2,
                         y_center + box_height / 2])
images = pd.DataFrame(image_rows, columns=["path", "width", "height"])
boxes = pd.DataFrame(box_rows, columns=["path", "label", "x1", "y1", "x2", "y2"])
print("gambar:", len(images), "| kotak:", len(boxes))


# %% [markdown]
# ## Pascal VOC xml (satu .xml per gambar, `<object><name>..<bndbox>xmin ymin xmax ymax`)
# Ubah: folder xml, folder gambar.

# %%
import glob
import xml.etree.ElementTree as ElementTree

import pandas as pd

XML_FOLDER = "data/annotations"   # UBAH
IMAGE_FOLDER = "data/images"      # UBAH

image_rows = []
box_rows = []
for xml_file in sorted(glob.glob(f"{XML_FOLDER}/*.xml")):
    root = ElementTree.parse(xml_file).getroot()
    path = f"{IMAGE_FOLDER}/{root.find('filename').text}"
    size = root.find("size")
    image_rows.append([path, int(size.find("width").text), int(size.find("height").text)])
    for item in root.findall("object"):
        box = item.find("bndbox")
        corners = []
        for name in ["xmin", "ymin", "xmax", "ymax"]:
            corners.append(float(box.find(name).text))
        box_rows.append([path, item.find("name").text] + corners)
images = pd.DataFrame(image_rows, columns=["path", "width", "height"])
boxes = pd.DataFrame(box_rows, columns=["path", "label", "x1", "y1", "x2", "y2"])
print("gambar:", len(images), "| kotak:", len(boxes))


# %% [markdown]
# ## CVAT "for images 1.1" (satu annotations.xml: `<image name=..><box label=.. xtl ytl xbr ybr>`)

# %%
import xml.etree.ElementTree as ElementTree

import pandas as pd

CVAT_XML = "data/annotations.xml"   # UBAH
IMAGE_FOLDER = "data/images"        # UBAH

root = ElementTree.parse(CVAT_XML).getroot()
image_rows = []
box_rows = []
for image in root.findall("image"):
    path = f"{IMAGE_FOLDER}/{image.get('name')}"
    image_rows.append([path, int(image.get("width")), int(image.get("height"))])
    for box in image.findall("box"):
        box_rows.append([path, box.get("label"), float(box.get("xtl")), float(box.get("ytl")),
                         float(box.get("xbr")), float(box.get("ybr"))])
images = pd.DataFrame(image_rows, columns=["path", "width", "height"])
boxes = pd.DataFrame(box_rows, columns=["path", "label", "x1", "y1", "x2", "y2"])
print("gambar:", len(images), "| kotak:", len(boxes))


# %% [markdown]
# ## Cek isi: kelas, kotak per gambar, gambar tanpa kotak, kelas yang tidak dipakai
# Roboflow sering menambah kelas induk (mis. "Images") yang tidak punya kotak: abaikan saja.

# %%
print("kotak per kelas:")
print(boxes["label"].value_counts())
per_image = boxes.groupby(["path", "label"]).size()
print("kotak per gambar per kelas (berapa gambar):")
print(per_image.value_counts())
no_box = images[~images["path"].isin(boxes["path"])]
print("gambar tanpa kotak:", len(no_box))
image_area = dict(zip(images["path"], images["width"] * images["height"]))
box_area = (boxes["x2"] - boxes["x1"]) * (boxes["y2"] - boxes["y1"])
boxes["relative_area"] = box_area / boxes["path"].map(image_area)   # 0.01 = 1% luas gambar
area_summary = boxes.groupby("label")["relative_area"].describe()
print(area_summary.round(3))


# %% [markdown]
# ## Lihat beberapa gambar beserta kotaknya
# Warna per kelas. Cek: kotak pas dengan objek? ada gambar terputar?

# %%
import matplotlib.pyplot as plt
from PIL import Image

N_SHOW = 8   # UBAH
colors = ["red", "lime", "blue", "yellow", "cyan", "magenta"]
class_list = sorted(boxes["label"].unique())
fig, axes = plt.subplots(2, N_SHOW // 2, figsize=(4 * N_SHOW // 2, 8))
preview_paths = images["path"].sample(N_SHOW, random_state=0)   # UBAH: random_state
for ax, path in zip(axes.ravel(), preview_paths):
    ax.imshow(Image.open(path).convert("RGB"))
    for box in boxes[boxes["path"] == path].itertuples():
        color = colors[class_list.index(box.label) % len(colors)]
        ax.add_patch(plt.Rectangle((box.x1, box.y1), box.x2 - box.x1, box.y2 - box.y1,
                                   fill=False, color=color, linewidth=2))
        ax.text(box.x1, box.y1, box.label, color=color, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()

