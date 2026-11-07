"""Model tabel virtual: Qt hanya meminta sel yang terlihat; baris diambil dari polars per blok 200."""
import json
from collections import OrderedDict

import polars as pl
from tampilan import pahami
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

BLOK = 200
MAKS_BLOK = 64  # ± 12.800 baris di memori; cukup untuk gulir bolak-balik
PANJANG_SEL = 300


def teks_sel(v):
    if v is None:
        return "null"
    s = v if isinstance(v, str) else str(v)
    pendek = s[:PANJANG_SEL].replace("\r\n", "⏎").replace("\n", "⏎").replace("\t", "→")
    return pendek + "…" if len(s) > PANJANG_SEL else pendek


def teks_penuh(v):
    if v is None:
        return "(null)"
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, indent=2, default=str)
    return str(v)


class ModelTabel(QAbstractTableModel):
    def __init__(self, df: pl.DataFrame, parent=None):
        super().__init__(parent)
        self.df = df
        self._blok = OrderedDict()
        self._angka = [df[c].dtype.is_numeric() for c in df.columns]
        self._abu = QColor(pahami.W()["redup"])

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.df.height

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.df.width

    def baris(self, r):
        b = r // BLOK
        rows = self._blok.get(b)
        if rows is None:
            rows = self.df.slice(b * BLOK, BLOK).rows()
            self._blok[b] = rows
            if len(self._blok) > MAKS_BLOK:
                self._blok.popitem(last=False)
        else:
            self._blok.move_to_end(b)
        return rows[r - b * BLOK]

    def nilai(self, r, c):
        return self.baris(r)[c]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return teks_sel(self.nilai(index.row(), index.column()))
        if role == Qt.ForegroundRole and self.nilai(index.row(), index.column()) is None:
            return self._abu
        if role == Qt.TextAlignmentRole and self._angka[index.column()]:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return self.df.columns[section]
            if role == Qt.ToolTipRole:
                return f"{self.df.columns[section]}\n{self.df.dtypes[section]}"
        elif role == Qt.DisplayRole:
            return str(section)  # indeks 0-based = indeks baris di polars/pandas
        return None
