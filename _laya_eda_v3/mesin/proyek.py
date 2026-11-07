"""Proyek `.laya.json`: folder data + opsi baca per tabel (+ langkah, mulai M2).
Folder disimpan absolut dan relatif terhadap file proyek, supaya proyek tetap jalan setelah dipindah."""
import hashlib
import json
import os
from pathlib import Path

VERSI = 1


def baru(folder):
    return {"versi": VERSI, "folder": str(Path(folder).resolve()), "opsi_baca": {}, "tabel_aktif": None}


def path_untuk(folder, folder_kerja):
    """File proyek bawaan untuk sebuah folder data: proyek/<nama>-<hash>.laya.json."""
    folder = Path(folder).resolve()
    kode = hashlib.md5(str(folder).lower().encode()).hexdigest()[:6]
    return Path(folder_kerja) / f"{folder.name or 'data'}-{kode}.laya.json"


def simpan(proyek, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(proyek)
    try:
        data["folder_relatif"] = os.path.relpath(proyek["folder"], path.parent)
    except ValueError:  # beda drive
        data["folder_relatif"] = None
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)  # tulis-lalu-ganti: proyek tidak pernah setengah tertulis


def muat(path):
    """Folder dicari absolut dulu, lalu relatif. Tidak ketemu -> folder tetap nilai lama; cek dengan folder_ada()."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rel = data.get("folder_relatif")
    if not Path(data["folder"]).is_dir() and rel and (path.parent / rel).is_dir():
        data["folder"] = str((path.parent / rel).resolve())
    data.setdefault("opsi_baca", {})
    data.setdefault("tabel_aktif", None)
    return data


def folder_ada(proyek):
    return Path(proyek["folder"]).is_dir()
