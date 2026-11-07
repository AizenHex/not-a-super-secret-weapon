"""Penyusun tata letak kecil yang dipakai jendela dan tab Model."""
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


def kolom(*widget):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widget:
        lay.addWidget(x)
    return w


def baris(*widget):
    """Deretan mendatar; None = ruang kosong yang melar."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 2, 0, 2)
    for x in widget:
        lay.addStretch(1) if x is None else lay.addWidget(x)
    return w


def bagian(teks):
    return QLabel(teks, objectName="bagian")
