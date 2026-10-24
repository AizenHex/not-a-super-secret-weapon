"""Baca .ipynb (atau .py bersel `# %%`) sebagai teks bersel `# %%` — dipakai lint.py dan uji_arsenal.py."""
import json


def baca_teks_sel(path):
    if not path.endswith(".ipynb"):
        return open(path, encoding="utf-8").read()
    notebook = json.load(open(path, encoding="utf-8"))
    bagian = []
    for cell in notebook["cells"]:
        isi = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            baris = []
            for line in isi.splitlines():
                baris.append(("# " + line).rstrip())
            bagian.append("# %% [markdown]\n" + "\n".join(baris))
        elif cell["cell_type"] == "code":
            bagian.append("# %%\n" + isi)
    return "\n\n".join(bagian) + "\n"
