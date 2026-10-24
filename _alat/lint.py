"""Cek ATURAN_KODE.md pada file .py bersel `# %%`. Pakai: lint.py nb/tabular/*.py"""
import ast

from sel import baca_teks_sel
import re
import sys

LARANGAN = {
    r"^\s*try:": "try/except",
    r"^\s*assert ": "assert",
    r"_authoring|from src|sys\.path": "impor helper lama",
    r"\.merge\(|\.join\(\s*\w+\s*,?\s*on=|(?<!re)set_index\(": "merge/join/set_index",
    # class hanya untuk pola standar PyTorch: model (nn.Module) dan dataset (Dataset)
    r"^\s*class (?!\w+\((nn\.Module|Dataset|torch\.utils\.data\.Dataset)\))": "class selain nn.Module/Dataset",
    r"lambda.*lambda": "lambda bertumpuk",
    r"\bif METHOD\b|\bif MODEL\b": "cabang metode",
    r"[^\"'#]*;\s*\S": "dua statement satu baris",
    r"save_model|save_json": "save_model/save_json",
    r"^ROOT = |\bPath\(|\.mkdir\(": "ROOT/Path/mkdir (pakai string + f-string)",
}
NAMA_1_HURUF_BOLEH = {"X", "y", "i", "_"}
# satu-satunya if satu baris yang boleh: pilih GPU/CPU
TERNARY_BOLEH = 'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"'


def jumlah_rantai(node):
    # .a().b().c() = 3 pemanggilan method berantai
    jumlah = 0
    while isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            jumlah = jumlah + 1
        if isinstance(node, ast.Call):
            node = node.func
        else:
            node = node.value
    return jumlah


def cek_gaya(sel, kode_sel):
    # pola yang membuat kode sulit dibaca pemula (dicek dengan pengurai Python)
    masalah = []
    pohon = ast.parse(kode_sel)
    baris = kode_sel.splitlines()
    sudah_rantai = set()
    for node in ast.walk(pohon):
        teks = ""
        if hasattr(node, "lineno"):
            teks = baris[node.lineno - 1].strip()[:70]
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            masalah.append(f"comprehension (pakai loop biasa): {teks}")
        elif isinstance(node, ast.Lambda):
            masalah.append(f"lambda (pakai fungsi biasa): {teks}")
        elif isinstance(node, ast.IfExp) and teks != TERNARY_BOLEH:
            masalah.append(f"if satu baris (pakai if biasa): {teks}")
        elif isinstance(node, ast.FunctionDef) and node.end_lineno - node.lineno + 1 > 20:
            masalah.append(f"fungsi > 20 baris: {teks}")
        elif isinstance(node, ast.Call) and node.lineno not in sudah_rantai:
            if jumlah_rantai(node) >= 3:
                sudah_rantai.add(node.lineno)
                masalah.append(f"rantai method >= 3 (pecah per baris): {teks}")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if len(node.id) == 1 and node.id not in NAMA_1_HURUF_BOLEH:
                masalah.append(f"nama 1 huruf '{node.id}': {teks}")
    isi = pohon.body
    if isi and all(isinstance(bagian, ast.FunctionDef) for bagian in isi):
        masalah.append(f"sel hanya berisi fungsi (coba fungsinya di sel yang sama): {sel[:40]}")
    return masalah


total_masalah = 0
for path in sys.argv[1:]:
    text = baca_teks_sel(path)
    cells = re.split(r"^# %%.*$", text, flags=re.M)
    headers = re.findall(r"^# %%.*$", text, flags=re.M)
    masalah = []
    kode_total = 0
    for header, cell in zip(headers, cells[1:]):
        if "[markdown]" in header:
            for line in cell.splitlines():
                if len(line) > 100:
                    masalah.append(f"markdown {len(line)} kar: {line[:60]}")
            continue
        lines = [line for line in cell.strip("\n").splitlines()]
        kode = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        kode_total += len(kode)
        if len(lines) > 30:
            masalah.append(f"sel {len(lines)} baris: {lines[0][:50]}")
        for line in lines:
            if len(line) > 100:
                masalah.append(f"baris {len(line)} kar: {line[:60]}")
            for pola, nama in LARANGAN.items():
                # ATURAN §1: merge/join/set_index dilarang hanya di sel yang menyusun submission
                if nama == "merge/join/set_index" and "submission" not in cell:
                    continue
                if re.search(pola, line) and not line.strip().startswith("#"):
                    masalah.append(f"{nama}: {line.strip()[:70]}")
        masalah = masalah + cek_gaya(cell.strip()[:40], cell)
    if "submission" in text and 'submission = pd.DataFrame()' not in text:
        masalah.append("submission tidak memakai pola pd.DataFrame()")
    total_masalah += len(masalah)
    print(f"{path}: {kode_total} baris kode", "OK" if not masalah else f"{len(masalah)} masalah")
    for item in masalah:
        print("   -", item)
print("TOTAL MASALAH:", total_masalah)
