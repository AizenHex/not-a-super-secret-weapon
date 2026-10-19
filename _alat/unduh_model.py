"""Unduh SEMUA bobot pretrained yang dipakai notebook (sekitar 9 GB, butuh internet sekali).
Pakai: python unduh_model.py [folder_tujuan]
Default folder tujuan: folder `pretrained` di SEBELAH folder _.NEW-GEN
(notebook membacanya sebagai "../../pretrained"). Model yang sudah ada dilewati."""
import os
import sys
import urllib.request

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.abspath(os.path.join(HERE, "..", "..", "pretrained"))
if len(sys.argv) > 1:
    TARGET = os.path.abspath(sys.argv[1])

SKIP = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "*.tflite", "rust_model*",
        "coreml/*", "tf_model*", "flax_model*"]

# folder: repo Hugging Face
HF = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "minilm_en": "sentence-transformers/all-MiniLM-L6-v2",
    "distilbert_id": "cahya/distilbert-base-indonesian",
    "distilbert_en": "distilbert/distilbert-base-uncased",
    "indobert": "indobenchmark/indobert-base-p1",
    "nli_mdeberta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "qa_mdeberta": "timpal0l/mdeberta-v3-base-squad2",
    "qwen_05b": "Qwen/Qwen2.5-0.5B-Instruct",
    "whisper_tiny": "openai/whisper-tiny",
    "wav2vec2_base": "facebook/wav2vec2-base",
    "clip_b32": "openai/clip-vit-base-patch32",
    "blip_caption": "Salesforce/blip-image-captioning-base",
    "vilt_vqa": "dandelin/vilt-b32-finetuned-vqa",
    "trocr_cetak": "microsoft/trocr-small-printed",
    "trocr_tangan": "microsoft/trocr-small-handwritten",
    "plat_yolo11": "carlravel/indonesian-plate-detector-yolo11",
    "chronos_2": "amazon/chronos-2",
    "chronos_bolt_small": "amazon/chronos-bolt-small",
}

# folder: (fungsi torchvision, nama bobot)
TORCHVISION = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    "vit_b_16": ("vit_b_16", "ViT_B_16_Weights"),
    "fasterrcnn_mobilenet": ("detection.fasterrcnn_mobilenet_v3_large_320_fpn",
                             "detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights"),
    "lraspp_mobilenet": ("segmentation.lraspp_mobilenet_v3_large",
                         "segmentation.LRASPP_MobileNet_V3_Large_Weights"),
    "deeplabv3_mobilenet": ("segmentation.deeplabv3_mobilenet_v3_large",
                            "segmentation.DeepLabV3_MobileNet_V3_Large_Weights"),
}

# folder: [(nama file, url)]
FILES = {
    "yolov8n": [("yolov8n.pt",
                 "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")],
    "yolov8n_seg": [("yolov8n-seg.pt",
                     "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt")],
    "yolov8n_pose": [("yolov8n-pose.pt",
                      "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt")],
    "opencv_wajah": [
        ("face_detection_yunet_2023mar.onnx", "https://github.com/opencv/opencv_zoo/raw/main/"
         "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
        ("face_recognition_sface_2021dec.onnx", "https://github.com/opencv/opencv_zoo/raw/main/"
         "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
    ],
}


def get_attr(module, dotted):
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


def download_torchvision(folder, builder, weights_name):
    import torch
    import torchvision.models as tv_models

    weights = get_attr(tv_models, weights_name).DEFAULT
    model = get_attr(tv_models, builder)(weights=weights)
    os.makedirs(f"{TARGET}/{folder}", exist_ok=True)
    torch.save(model.state_dict(), f"{TARGET}/{folder}/model.pth")


def download_r3d():
    import torch
    from torchvision.models.video import R3D_18_Weights, r3d_18

    weights = R3D_18_Weights.DEFAULT
    os.makedirs(f"{TARGET}/r3d_18", exist_ok=True)
    torch.save(r3d_18(weights=weights).state_dict(), f"{TARGET}/r3d_18/model.pth")
    with open(f"{TARGET}/r3d_18/classes.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(weights.meta["categories"]))


def download_easyocr():
    import easyocr

    folder = f"{TARGET}/easyocr"
    os.makedirs(folder, exist_ok=True)
    easyocr.Reader(["en", "id"], gpu=False, model_storage_directory=folder,
                   user_network_directory=folder, download_enabled=True, verbose=False)


def is_done(folder):
    path = f"{TARGET}/{folder}"
    return os.path.isdir(path) and len(os.listdir(path)) > 0


def main():
    print("Folder tujuan:", TARGET)
    os.makedirs(TARGET, exist_ok=True)
    jobs = list(HF) + list(TORCHVISION) + list(FILES) + ["r3d_18", "easyocr", "laya"]
    for number, folder in enumerate(jobs, start=1):
        if is_done(folder):
            print(f"[{number}/{len(jobs)}] {folder}: sudah ada, dilewati")
            continue
        print(f"[{number}/{len(jobs)}] {folder}: mengunduh ...", flush=True)
        if folder in HF:
            snapshot_download(HF[folder], local_dir=f"{TARGET}/{folder}", ignore_patterns=SKIP)
        elif folder in TORCHVISION:
            download_torchvision(folder, TORCHVISION[folder][0], TORCHVISION[folder][1])
        elif folder in FILES:
            os.makedirs(f"{TARGET}/{folder}", exist_ok=True)
            for name, url in FILES[folder]:
                urllib.request.urlretrieve(url, f"{TARGET}/{folder}/{name}")
        elif folder == "r3d_18":
            download_r3d()
        elif folder == "easyocr":
            download_easyocr()
        elif folder == "laya":
            snapshot_download("convaiinnovations/laya", local_dir=f"{TARGET}/laya",
                              allow_patterns=["multilingual/*", "*.json", "*.py", "README.md"])
    print("SELESAI. Semua model ada di", TARGET)


main()
