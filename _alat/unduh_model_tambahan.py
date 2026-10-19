"""Unduh bobot pretrained tambahan (2026-10-17) ke Z:/GEMASTIK-DATABASE/pretrained/.
Pakai: python unduh_model_tambahan.py [folder ...]   (default: semua). Butuh internet sekali."""
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

PRE = Path("Z:/GEMASTIK-DATABASE/pretrained")
HF = {
    "nli_mdeberta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "qa_mdeberta": "timpal0l/mdeberta-v3-base-squad2",
    "qwen_05b": "Qwen/Qwen2.5-0.5B-Instruct",
    "minilm_en": "sentence-transformers/all-MiniLM-L6-v2",
    "distilbert_en": "distilbert/distilbert-base-uncased",
    "trocr_cetak": "microsoft/trocr-small-printed",
    "trocr_tangan": "microsoft/trocr-small-handwritten",
    "blip_caption": "Salesforce/blip-image-captioning-base",
}
YOLO = {
    "yolov8n_seg": "yolov8n-seg.pt",
    "yolov8n_pose": "yolov8n-pose.pt",
}
BUANG = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "*.tflite",
         "rust_model*", "coreml/*", "tf_model*", "flax_model*", "*.md"]


def unduh_hf(folder, repo):
    buang = list(BUANG)
    if any(f.endswith(".safetensors") for f in list_repo_files(repo)):
        buang.append("*.bin")
    snapshot_download(repo, local_dir=PRE / folder, ignore_patterns=buang)


def unduh_yolo(folder, nama):
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/" + nama
    (PRE / folder).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, PRE / folder / nama)


def unduh_r3d():
    import torch
    from torchvision.models.video import R3D_18_Weights, r3d_18

    bobot = R3D_18_Weights.DEFAULT
    folder = PRE / "r3d_18"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(r3d_18(weights=bobot).state_dict(), folder / "model.pth")
    (folder / "classes.txt").write_text("\n".join(bobot.meta["categories"]), encoding="utf-8")


if __name__ == "__main__":
    pilihan = sys.argv[1:] or list(HF) + list(YOLO) + ["r3d_18"]
    for folder in pilihan:
        print("unduh", folder, flush=True)
        if folder in HF:
            unduh_hf(folder, HF[folder])
        elif folder in YOLO:
            unduh_yolo(folder, YOLO[folder])
        else:
            unduh_r3d()
    print("selesai")
