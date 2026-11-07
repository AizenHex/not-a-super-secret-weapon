"""Form langkah resep (D1): isian dibangun dari langkah.JENIS, kode polars tampil langsung, pratinjau di latar."""
import html

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout)

from tampilan import pahami
from mesin import langkah
from mesin.resep import contoh_ubah, pesan_galat, ringkas, teks_ringkas
from tampilan.model import teks_sel


class FormLangkah(QDialog):
    """`pratinjau(kode, ok, gagal)` menjalankan kode di latar lalu memanggil ok((sebelum, sesudah))."""

    def __init__(self, induk, jenis, kolom, tabel, lama=None, pratinjau=None):
        super().__init__(induk)
        self.jenis, self.spek = jenis, langkah.JENIS[jenis]
        self.pratinjau_fn = pratinjau
        self.setWindowTitle(self.spek["judul"])
        self.resize(720, 640)
        p = dict(langkah.BAWAAN.get(jenis, {}), **(lama or {}).get("param", {}))
        self.w = {}
        form = QFormLayout()
        for nama, tipe, label in self.spek["isian"]:
            w = self._widget(tipe, p.get(nama), kolom, tabel)
            self.w[nama] = (tipe, w)
            form.addRow(label, w)
            self._sambung(w, self._perbarui)
        if "preset" in self.w:
            self.w["preset"][1].currentTextChanged.connect(self._preset)
        self.latih = QCheckBox("hanya untuk data latih (dilewati untuk tabel pasangan, mis. test)")
        self.latih.setChecked(lama["hanya_latih"] if lama else self.spek.get("hanya_latih", False))
        self.manual = QCheckBox("ubah kode langsung (isian di atas tidak dipakai lagi)")
        self.kode = QPlainTextEdit()
        self.kode.setFont(QFont("Consolas", 10))
        self.kode.setReadOnly(True)
        self.manual.toggled.connect(lambda on: (self.kode.setReadOnly(not on), self._perbarui()))
        self.hasil = QLabel(wordWrap=True, textInteractionFlags=Qt.TextSelectableByMouse)
        tombol_pra = QPushButton("Pratinjau (F5)", shortcut="F5")
        tombol_pra.clicked.connect(self.pratinjau)
        tombol = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        tombol.accepted.connect(self._ok)
        tombol.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.latih)
        lay.addWidget(QLabel("Kode polars (sama persis dengan yang diekspor):"))
        lay.addWidget(self.kode, 1)
        lay.addWidget(self.manual)
        lay.addWidget(tombol_pra)
        lay.addWidget(self.hasil)
        lay.addWidget(tombol)
        if lama and lama.get("manual"):
            self.manual.setChecked(True)
            self.kode.setPlainText(lama["kode"])
        self._perbarui()

    # ---------- isian ----------
    @staticmethod
    def _widget(tipe, nilai, kolom, tabel):
        if tipe in ("kolom_banyak", "tabel_banyak"):
            w = QListWidget(maximumHeight=140)
            pilih = set(nilai or [])
            for c in (kolom if tipe == "kolom_banyak" else tabel):
                it = QListWidgetItem(c)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if c in pilih else Qt.Unchecked)
                w.addItem(it)
            return w
        if tipe in ("kolom", "tabel") or isinstance(tipe, tuple):
            w = QComboBox(editable=tipe != "tabel" and not isinstance(tipe, tuple))
            w.addItems(tipe[1] if isinstance(tipe, tuple) else kolom if tipe == "kolom" else tabel)
            if nilai is not None:
                w.setCurrentText(str(nilai))
            return w
        if tipe == "teks_panjang":
            w = QPlainTextEdit(maximumHeight=130)
            w.setFont(QFont("Consolas", 10))
            w.setPlainText(str(nilai or ""))
            return w
        if tipe == "angka":
            return QSpinBox(minimum=0, maximum=99, value=int(nilai or 0))
        if tipe == "cek":
            return QCheckBox(checked=bool(nilai))
        return QLineEdit(str(nilai or ""))

    @staticmethod
    def _sambung(w, fn):
        for sinyal in ("itemChanged", "currentTextChanged", "textChanged", "valueChanged", "toggled"):
            if hasattr(w, sinyal):
                getattr(w, sinyal).connect(lambda *_: fn())
                return

    def param(self):
        p = {}
        for nama, (tipe, w) in self.w.items():
            if isinstance(w, QListWidget):
                p[nama] = [w.item(i).text() for i in range(w.count()) if w.item(i).checkState() == Qt.Checked]
            elif isinstance(w, QComboBox):
                p[nama] = w.currentText()
            elif isinstance(w, QPlainTextEdit):
                p[nama] = w.toPlainText()
            elif isinstance(w, QSpinBox):
                p[nama] = w.value()
            elif isinstance(w, QCheckBox):
                p[nama] = w.isChecked()
            else:
                p[nama] = w.text()
        return p

    def _preset(self, nama):
        if nama in langkah.PRESET_REGEX:
            pola, ganti = langkah.PRESET_REGEX[nama]
            self.w["pola"][1].setText(pola)
            self.w["ganti"][1].setText(ganti)

    def _perbarui(self):
        if self.manual.isChecked():
            return
        if self.param().get("aksi") == "buang baris":
            self.latih.setChecked(True)  # membuang baris test merusak submission
        try:
            self.kode.setPlainText(self.spek["gen"](self.param()))
        except Exception as e:  # isian belum lengkap: tampilkan alasannya di tempat kode
            self.kode.setPlainText(f"# {e}")

    def langkah(self):
        return {"jenis": self.jenis, "param": self.param(), "kode": self.kode.toPlainText(), "nyala": True,
                "hanya_latih": self.latih.isChecked(), "manual": self.manual.isChecked()}

    def _ok(self):
        if self.kode.toPlainText().startswith("# "):
            self.hasil.setText(f"<span style='color:{pahami.W()['penting']}'>{html.escape(self.kode.toPlainText()[2:])}</span>")
            return
        self.accept()

    # ---------- pratinjau ----------
    def pratinjau(self):
        if not self.pratinjau_fn:
            return
        self.hasil.setText("menghitung…")
        self.pratinjau_fn(self.kode.toPlainText(), self._tampil_pratinjau, self._gagal_pratinjau)

    def _tampil_pratinjau(self, ab):
        a, b = ab
        teks = html.escape(teks_ringkas(ringkas(a, b))).replace("\n", "<br>")
        for c, i, x, y in contoh_ubah(a, b):
            teks += f"<br><tt>{html.escape(c)}[{i}]: {html.escape(teks_sel(x)[:80])} → {html.escape(teks_sel(y)[:80])}</tt>"
        self.hasil.setText(teks)

    def _gagal_pratinjau(self, e):
        self.hasil.setText(f"<span style='color:{pahami.W()['penting']}'>{html.escape(pesan_galat(e)).replace(chr(10), '<br>')}</span>")
