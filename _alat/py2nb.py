"""Ubah file .py bersel `# %%` / `# %% [markdown]` menjadi .ipynb.

Pakai: python py2nb.py sumber.py tujuan.ipynb
Sel markdown: tiap baris diawali "# " (diambil isinya).
"""
import sys
from pathlib import Path

import nbformat as nbf

source, target = Path(sys.argv[1]), Path(sys.argv[2])
cells = []
kind, lines = None, []


def flush():
    text = "\n".join(lines).strip("\n")
    if kind == "markdown":
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines())
        cells.append(nbf.v4.new_markdown_cell(text))
    elif kind == "code" and text.strip():
        cells.append(nbf.v4.new_code_cell(text))


for line in source.read_text(encoding="utf-8").splitlines():
    if line.startswith("# %%"):
        flush()
        kind = "markdown" if "[markdown]" in line else "code"
        lines = []
    else:
        lines.append(line)
flush()

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, target)
print("ditulis", target, len(cells), "sel")
