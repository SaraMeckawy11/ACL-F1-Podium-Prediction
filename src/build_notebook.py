"""Build notebooks/f1_podium_prediction.ipynb from the percent-format source src/f1_podium_prediction.py.

The .py file is the source of truth (easy to diff and review); run this after editing it:
    python src/build_notebook.py
"""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "f1_podium_prediction.py"
DST = ROOT / "notebooks" / "f1_podium_prediction.ipynb"


def cells_from_percent(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body:
            return
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.splitlines()]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


nb = nbformat.v4.new_notebook()
nb.cells = cells_from_percent(SRC.read_text(encoding="utf-8"))
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
DST.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, DST)
print(f"wrote {DST} ({len(nb.cells)} cells)")
