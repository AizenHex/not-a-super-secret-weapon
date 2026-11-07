"""Jendela utama Laya v3: daftar tabel, grid virtual, panel isi sel, opsi baca, resep langkah (M2), proyek
tersimpan otomatis."""
import copy
import html
import json
import logging
import time
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QKeySequence, QPainter
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter,
                               QStackedWidget, QStyle, QStyledItemDelegate, QTableView, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from mesin import ekspor, langkah, muat, resep, temuan, venv
from mesin import proyek as prj
from tampilan import pahami
from tampilan.form import FormLangkah
from tampilan.model import ModelTabel, teks_penuh, teks_sel
from tampilan.tab_model import TabModel
from tampilan.tata import bagian, baris, kolom

def kuas(kunci):
    """Warna tema saat ini (pahami.W) sebagai QBrush untuk item daftar/pohon."""
    return QBrush(QColor(pahami.W()[kunci]))


def sel_kosong():
    return pahami.redup("Klik sel di tab Data untuk melihat isi penuhnya.")


ALUR = [("Data", "buka folder, pilih tabel train"), ("Pahami", "label, profil, temuan"),
         ("Rapikan", "langkah resep + pasangan test"), ("Validasi & model", "skor baseline"),
         ("Ekspor", "notebook / tabel bersih")]
TAMPIL = ["semua baris", "sampel 1.000 acak (seed 42)", "20 per label (seed 42)"]
TANPA_LABEL = "(tanpa label)"

log = logging.getLogger("laya")
PEMISAH = {"otomatis": None, "koma ,": ",", "titik koma ;": ";", "tab": "\t", "pipa |": "|"}
ENCODING = ["otomatis", "utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"]


def ribuan(n):
    return f"{n:,}".replace(",", ".")


def ukuran(b):
    for s in ("B", "KB", "MB"):
        if b < 1024:
            return f"{b:.0f} {s}" if s == "B" else f"{b:.1f} {s}"
        b /= 1024
    return f"{b:.1f} GB"


def nama_pemisah(sep):
    return next((k for k, v in PEMISAH.items() if v == sep), repr(sep))


GAYA = """
QLabel#bagian { color: %(redup)s; font-weight: 600; padding: 6px 2px 2px 2px; }
QLabel#meta { color: %(redup)s; }
QLabel#alur { padding: 6px 8px; border: 1px solid %(garis)s; }
QListWidget::item { padding: 3px 2px; }
QTreeWidget::item { padding: 2px 0; }
QTabBar::tab { padding: 6px 18px; }
QTabBar::tab:selected { font-weight: 600; }
QTextBrowser { padding: 4px; }
"""


class Sel(QTableWidgetItem):
    """Sel tabel profil yang diurutkan menurut nilai aslinya, bukan teksnya."""

    def __init__(self, teks, kunci=None):
        super().__init__(teks)
        self.kunci = teks if kunci is None else kunci

    def __lt__(self, lain):
        return self.kunci < lain.kunci


class Batang(QStyledItemDelegate):
    """Grafik batang mini di tabel profil: sebaran panjang/nilai, atau porsi tiap kategori."""

    def paint(self, p, opsi, i):
        super().paint(p, opsi, i)
        n = i.data(Qt.UserRole) or []
        if not n:
            return
        r = opsi.rect.adjusted(4, 4, -4, -4)
        lebar, maks = r.width() / len(n), max(n) or 1
        p.save()
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setPen(Qt.NoPen)
        p.setBrush(opsi.palette.highlightedText() if opsi.state & QStyle.State_Selected else QColor(pahami.batang_warna()))
        for k, c in enumerate(n):
            h = max(1, round(r.height() * c / maks)) if c else 0
            p.drawRect(round(r.left() + k * lebar), r.bottom() - h + 1, max(1, round(lebar) - 1), h)
        p.restore()


class Pekerja(QObject):
    """Satu thread latar untuk semua hitungan data, supaya layar tidak pernah membeku."""

    selesai = Signal(int, object)
    gagal = Signal(int, object)

    @Slot(int, object)
    def jalankan(self, token, fn):
        try:
            hasil = fn()
        except Exception as e:
            log.exception("tugas %s gagal", token)
            self.gagal.emit(token, e)
            return
        self.selesai.emit(token, hasil)


class Jendela(QMainWindow):
    kirim = Signal(int, object)

    def __init__(self, folder_kerja):
        super().__init__()
        self.folder_kerja = Path(folder_kerja)
        self.proyek, self.path_proyek = None, None
        self.entri, self.cache, self.tunggu = {}, {}, {}
        self.token, self.aktif = 0, None
        self.mesin, self.hasil_aktif, self.indeks = resep.Mesin({}, {}), None, 0
        self.sudah = {}  # tab -> (df, label) yang sudah dihitung profil/temuannya
        self.daftar_temuan = []
        self.setWindowTitle("Laya")
        self.resize(1400, 850)
        self._bangun()
        self._thread = QThread(self)
        self._pekerja = Pekerja()
        self._pekerja.moveToThread(self._thread)
        self.kirim.connect(self._pekerja.jalankan)
        self._pekerja.selesai.connect(self._selesai)
        self._pekerja.gagal.connect(self._gagal)
        self._thread.start()

    # ---------- tata letak ----------
    def _aksi(self, menu, teks, fn, kunci=None):
        a = QAction(teks, self)
        if kunci:
            a.setShortcut(QKeySequence(kunci))
        a.triggered.connect(fn)
        menu.addAction(a)

    def _bangun(self):
        m = self.menuBar().addMenu("&File")
        self._aksi(m, "Buka folder data…", self.pilih_folder, "Ctrl+O")
        self._aksi(m, "Buka proyek…", self.pilih_proyek)
        self._aksi(m, "Simpan proyek sebagai…", self.simpan_sebagai)
        self._aksi(m, "Ganti folder data proyek…", self.ganti_folder)
        m.addSeparator()
        self._aksi(m, "Keluar", self.close, "Ctrl+Q")
        m = self.menuBar().addMenu("&Tabel")
        self._aksi(m, "Opsi baca…", self.atur_opsi, "Ctrl+B")
        self._aksi(m, "Muat ulang folder", self.muat_ulang, "F5")
        m.addSeparator()
        self._aksi(m, "Tabel turunan baru…", lambda: self.ubah_turunan(None))
        self._aksi(m, "Ubah kode tabel turunan…", lambda: self.ubah_turunan(self.aktif))
        m = self.menuBar().addMenu("&Resep")
        self._aksi(m, "Tabel pasangan (ikut resep ini)…", self.atur_pasangan)
        self._aksi(m, "Ekspor ke .py / .ipynb…", self.ekspor, "Ctrl+E")
        self._aksi(m, "Simpan tabel bersih (parquet + csv)…", self.simpan_bersih, "Ctrl+S")

        self.model = TabModel(self)
        m = self.menuBar().addMenu("&Model")
        self._aksi(m, "Validasi file submission…", self.model.validasi_submisi, "Ctrl+M")
        self._aksi(m, "Batalkan tugas model", self.model.batal)
        m = self.menuBar().addMenu("&Lihat")
        for i, nama in enumerate(["Data", "Profil", "Temuan", "Model"]):
            self._aksi(m, nama, lambda i=i: self.tab.setCurrentIndex(i), f"Ctrl+{i + 1}")
        self.setStyleSheet(GAYA % pahami.W())

        # kiri: tabel di atas, resep tabel aktif di bawah
        self.saring = QLineEdit(placeholderText="saring nama tabel…", clearButtonEnabled=True)
        self.saring.textChanged.connect(self._saring)
        self.daftar = QListWidget()
        self.daftar.currentItemChanged.connect(lambda cur, _: cur and self.buka_tabel(cur.data(Qt.UserRole)))

        self.judul_resep = QLabel("RESEP", objectName="bagian", wordWrap=True)
        self.resep = QListWidget()
        self.resep.currentRowChanged.connect(self._tahap_dipilih)
        self.resep.itemChanged.connect(self._nyala_diubah)
        self.resep.itemDoubleClicked.connect(lambda _: self.ubah_langkah())
        menu = QMenu(self)
        for j, spek in langkah.JENIS.items():
            menu.addAction(spek["judul"], lambda j=j: self.tambah_langkah(j))
        tambah = QPushButton("+ Langkah")
        tambah.setMenu(menu)
        tombol = [tambah]
        for teks, tip, fn in [("Ubah", "Ubah langkah terpilih (klik ganda)", self.ubah_langkah),
                              ("Hapus", "Hapus langkah terpilih", self.hapus_langkah),
                              ("↑", "Naikkan langkah", lambda: self.geser_langkah(-1)),
                              ("↓", "Turunkan langkah", lambda: self.geser_langkah(1))]:
            b = QPushButton(teks, toolTip=tip)
            b.clicked.connect(fn)
            tombol.append(b)
        kiri = QSplitter(Qt.Vertical)
        kiri.addWidget(kolom(bagian("TABEL"), self.saring, self.daftar))
        kiri.addWidget(kolom(self.judul_resep, self.resep, baris(*tombol)))
        kiri.setSizes([260, 560])

        # tengah: judul tabel + tab Data / Profil / Temuan
        self.alur = QLabel(objectName="alur", wordWrap=True, textInteractionFlags=Qt.LinksAccessibleByMouse)
        self.alur.linkActivated.connect(self._ke_alur)
        self.info = QLabel("", wordWrap=True, textInteractionFlags=Qt.TextSelectableByMouse)
        self.grid = QTableView()
        self.grid.setWordWrap(False)
        vh = self.grid.verticalHeader()
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setDefaultSectionSize(self.fontMetrics().height() + 8)
        self.grid.horizontalHeader().setDefaultSectionSize(160)
        self.mentah = QPlainTextEdit(readOnly=True)
        self.mentah.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.mentah.setFont(QFont("Consolas", 10))
        self.tumpuk = QStackedWidget()
        self.tumpuk.addWidget(self.grid)
        self.tumpuk.addWidget(self.mentah)

        self.profil = QTableWidget(0, 6)
        self.profil.setHorizontalHeaderLabels(["Kolom", "Tipe", "Kosong", "Unik", "Sebaran", "Ringkasan"])
        self.profil.setItemDelegateForColumn(4, Batang(self.profil))
        self.profil.setEditTriggers(QTableWidget.NoEditTriggers)
        self.profil.setSelectionBehavior(QTableWidget.SelectRows)
        self.profil.setSelectionMode(QTableWidget.SingleSelection)
        self.profil.verticalHeader().hide()
        self.profil.setWordWrap(False)
        self.profil.verticalHeader().setDefaultSectionSize(self.fontMetrics().height() + 10)
        self.profil.horizontalHeader().setStretchLastSection(True)
        self.profil.setSortingEnabled(True)
        self.profil.currentCellChanged.connect(lambda r, *_: self._kolom_dipilih(r))
        self.data_profil = []

        self.tab = QTabWidget(documentMode=True)
        self.tab.addTab(kolom(self._bangun_cari(), self.tumpuk), "Data")
        self.tab.addTab(self.profil, "Profil")
        self.tab.addTab(self._bangun_temuan(), "Temuan")
        self.tab.addTab(self.model, "Model")
        for i, tip in enumerate(["Isi tabel, cari, sampel (Ctrl+1)", "Ringkasan tiap kolom (Ctrl+2)",
                                 "Masalah yang ditemukan otomatis (Ctrl+3)",
                                 "Baseline, label mencurigakan, kata khas, cek kebocoran (Ctrl+4)"]):
            self.tab.setTabToolTip(i, tip)

        # kanan: rincian dari yang dipilih di tengah (sel / kolom / temuan)
        self.judul_panel = QLabel(sel_kosong(), wordWrap=True, textInteractionFlags=Qt.TextSelectableByMouse)
        self.panel = QPlainTextEdit(readOnly=True)
        self.detail_kolom = QTextBrowser()
        self.temuan_rinci = QTextBrowser()
        self.lihat = QPushButton("Lihat baris", toolTip="Tampilkan baris terkait di tab Data")
        self.lihat.clicked.connect(self.lihat_baris)
        self.jadikan = QPushButton("Jadikan langkah…", toolTip="Buka form langkah resep yang sudah terisi")
        self.jadikan.clicked.connect(self.temuan_jadi_langkah)
        self.kanan = QStackedWidget()
        self.kanan.addWidget(kolom(bagian("ISI SEL"), self.judul_panel, self.panel))
        self.kanan.addWidget(kolom(bagian("RINCIAN KOLOM"), self.detail_kolom))
        self.kanan.addWidget(kolom(bagian("RINCIAN TEMUAN"), self.temuan_rinci, baris(self.lihat, self.jadikan, None)))
        self.tab.currentChanged.connect(self._tab_diganti)

        split = QSplitter()
        split.addWidget(kiri)
        split.addWidget(kolom(self.alur, self.info, self.tab))
        split.addWidget(self.kanan)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 800, 440])
        self.setCentralWidget(split)

        self.status = QLabel("Buka folder data: File → Buka folder data… (Ctrl+O)")
        self.sibuk = QProgressBar(maximumWidth=120)
        self.sibuk.setRange(0, 0)
        self.batal = QPushButton("Batal")
        self.batal.clicked.connect(self.batalkan)
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.sibuk)
        self.statusBar().addPermanentWidget(self.batal)
        self._perbarui_sibuk()
        self._kosongkan_rincian()
        self._perbarui_alur()

    # ---------- alur kerja ----------
    def _langkah_alur(self):
        """(nomor tahap 1..5 yang sedang dikerjakan, kalimat langkah berikutnya)."""
        if not self.proyek:
            return 1, "Buka folder data: <b>File → Buka folder data…</b> (Ctrl+O)."
        if not self.aktif:
            return 1, "Pilih tabel <b>train</b> di daftar TABEL kiri."
        lab = self._label()
        if not self.sudah.get(2):
            return 2, "Buka tab <b>Temuan</b> (Ctrl+3): pastikan kolom label benar, baca temuan Penting dulu."
        if not lab:
            return 2, "Pilih kolom <b>Label</b> (target) di tab Temuan; tanpa label model dan cek kebocoran tidak jalan."
        pem, r = resep.resep_dari(self.proyek, self.aktif)
        if not r.get("langkah"):
            return 3, ("Pilih satu temuan lalu <b>Jadikan langkah…</b>, atau <b>+ Langkah</b> di RESEP kiri "
                       "(outlier, isi kosong, ubah tipe, teks, …).")
        if pem == self.aktif and not r.get("pasangan") and any("test" in t.lower() for t in self._nama_tabel()):
            return 3, ("Pasangkan tabel test: <b>Resep → Tabel pasangan</b>, supaya langkah yang sama dipakai di test "
                       "dengan statistik dari train.")
        if not any(x["status"] == "ok" and x["tabel"] == self.aktif and x["jenis"] in ("tabular", "baseline")
                   for x in self.model.riwayat):
            return 4, "Buka tab <b>Model</b> (Ctrl+4): pilih validasi, jalankan <b>Baseline tabular</b> atau teks."
        return 5, ("Ekspor resep ke notebook: <b>Resep → Ekspor</b> (Ctrl+E), atau simpan tabel bersih (Ctrl+S). "
                   "Ubah resep lalu jalankan baseline lagi untuk membandingkan skor.")

    def _perbarui_alur(self):
        k, kalimat = self._langkah_alur()
        w = pahami.W()
        bagian_ = []
        for i, (nama, ket) in enumerate(ALUR, 1):
            gaya = "font-weight:600" if i == k else f"color:{w['redup']}"
            bagian_.append(f"<a href='{i}' style='text-decoration:none; {gaya}'>{i}. {nama}</a>")
        self.alur.setText(" &nbsp;&nbsp;›&nbsp;&nbsp; ".join(bagian_) + f"<br>Berikutnya: {kalimat}")
        self.alur.setToolTip("\n".join(f"{i}. {n}: {k_}" for i, (n, k_) in enumerate(ALUR, 1)))

    def _ke_alur(self, href):
        i = int(href)
        if i == 1 and not self.proyek:
            return self.pilih_folder()
        if i == 5:
            return self.ekspor()
        self.tab.setCurrentIndex({1: 0, 2: 2, 3: 0, 4: 3}[i])
        if i == 3:
            self.resep.setFocus()

    def _bangun_cari(self):
        self.cari_mode = QComboBox(toolTip="teks: tanpa beda huruf besar/kecil · regex · SQL: ekspresi WHERE · "
                                           "kode polars: ekspresi pl.col(...)")
        self.cari_mode.addItems(["teks", "regex", "SQL", "kode polars"])
        self.cari_kolom = QComboBox(minimumContentsLength=14)
        self.cari_isi = QLineEdit(placeholderText="cari lalu Enter (kosong = semua baris)", clearButtonEnabled=True)
        self.cari_isi.returnPressed.connect(self.cari)
        self.tampil = QComboBox()
        self.tampil.addItems(TAMPIL)
        self.tampil.activated.connect(lambda _: self.cari())
        jadi = QPushButton("Jadikan langkah saring", toolTip="Tambahkan pencarian ini ke resep (hanya data latih)")
        jadi.clicked.connect(self.cari_jadi_langkah)
        self.hasil_cari = QLabel(wordWrap=True, objectName="meta")
        bar = baris(QLabel("Cari"), self.cari_mode, self.cari_kolom, self.cari_isi, QLabel("Tampilkan"), self.tampil,
                    jadi)
        bar.layout().setStretch(3, 1)
        return kolom(bar, self.hasil_cari)

    def _bangun_temuan(self):
        self.label_pilih = QComboBox(minimumContentsLength=16, toolTip="Kolom target; ditebak = kolom train yang "
                                                                       "tidak ada di test")
        self.label_pilih.activated.connect(self._label_diubah)
        relasi = QPushButton("Relasi antartabel", toolTip="Kolom kunci bersama antara tabel ini dan tabel lain")
        relasi.clicked.connect(self.cari_relasi)
        self.temuan = QTreeWidget(columnCount=2, uniformRowHeights=True)
        self.temuan.setHeaderLabels(["Temuan", "Jenis"])
        self.temuan.header().setStretchLastSection(False)
        self.temuan.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.temuan.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.temuan.currentItemChanged.connect(lambda cur, _: self._temuan_dipilih(cur))
        return kolom(baris(QLabel("Label"), self.label_pilih, None, relasi), self.temuan)

    # ---------- tugas latar ----------
    def jalankan(self, label, fn, ok, gagal=None):
        self.token += 1
        self.tunggu[self.token] = (label, ok, gagal)
        self._perbarui_sibuk()
        self.kirim.emit(self.token, fn)

    def _selesai(self, token, hasil):
        tugas = self.tunggu.pop(token, None)
        self._perbarui_sibuk()
        if tugas:
            tugas[1](hasil)

    def _gagal(self, token, e):
        tugas = self.tunggu.pop(token, None)
        self._perbarui_sibuk()
        if tugas and tugas[2]:
            tugas[2](e)
        elif tugas:
            self.galat(f"{tugas[0]} gagal", e)

    def batalkan(self):
        # shortcut: hitungan yang sedang jalan tetap selesai di latar lalu hasilnya dibuang; batal sungguhan di G1
        n = len(self.tunggu)
        self.tunggu.clear()
        self._perbarui_sibuk()
        self.status.setText(f"{n} tugas dibatalkan")

    def _perbarui_sibuk(self):
        ada = bool(self.tunggu)
        self.sibuk.setVisible(ada)
        self.batal.setVisible(ada)
        sibuk = self.status.text().endswith("…")
        if ada:
            if not sibuk:
                self.diam = self.status.text()  # dipulihkan saat semua tugas selesai
            self.status.setText(next(iter(self.tunggu.values()))[0] + "…")
        elif sibuk:
            self.status.setText(getattr(self, "diam", ""))

    def galat(self, judul, e):
        log.error("%s: %r", judul, e)
        QMessageBox.warning(self, "Laya", f"{judul}.\n\n{type(e).__name__}: {e}\n\nRincian di laya_log.txt "
                                          f"({self.folder_kerja}).")

    # ---------- proyek ----------
    def pilih_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pilih folder data", self.proyek["folder"] if self.proyek else "")
        if d:
            self.buka_folder(d)

    def buka_folder(self, folder):
        path = prj.path_untuk(folder, self.folder_kerja)
        self._pakai_proyek(prj.muat(path) if path.exists() else prj.baru(folder), path)

    def pilih_proyek(self):
        f, _ = QFileDialog.getOpenFileName(self, "Buka proyek", str(self.folder_kerja), "Proyek Laya (*.laya.json)")
        if f:
            self._pakai_proyek(prj.muat(f), f)

    def simpan_sebagai(self):
        if not self.proyek:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Simpan proyek sebagai", str(self.path_proyek),
                                           "Proyek Laya (*.laya.json)")
        if f:
            self.path_proyek = Path(f)
            self.simpan()

    def ganti_folder(self):
        if not self.proyek:
            return
        d = QFileDialog.getExistingDirectory(self, "Folder data untuk proyek ini")
        if d:
            self.proyek["folder"] = str(Path(d).resolve())
            self._pakai_proyek(self.proyek, self.path_proyek)

    def buka_terakhir(self):
        try:
            path = Path((self.folder_kerja / "terakhir.txt").read_text(encoding="utf-8").strip())
            p = prj.muat(path)
        except (OSError, ValueError, KeyError) as e:
            log.info("tidak ada proyek terakhir: %r", e)
            return
        self._pakai_proyek(p, path)

    def muat_ulang(self):
        if self.proyek:
            self._pakai_proyek(self.proyek, self.path_proyek)

    def _pakai_proyek(self, p, path):
        self.proyek, self.path_proyek = p, Path(path)
        self.cache.clear()
        self.entri.clear()
        self.daftar.clear()
        self.resep.clear()
        self.aktif, self.hasil_aktif = None, None
        self.mesin = resep.Mesin({}, {})  # cache tahap lama dibuang: file bisa sudah berubah
        self.setWindowTitle(f"Laya — {p['folder']}")
        if not prj.folder_ada(p):
            QMessageBox.warning(self, "Folder data tidak ada", f"Folder {p['folder']} tidak ditemukan.\n"
                                "Pilih lewat File → Ganti folder data proyek.")
            return
        self.simpan()
        folder = p["folder"]
        self.jalankan("Memindai folder", lambda: muat.pindai(folder), self._isi_daftar)
        self._perbarui_alur()

    def simpan(self):
        if not self.proyek:
            return
        try:
            prj.simpan(self.proyek, self.path_proyek)
            (self.folder_kerja / "terakhir.txt").write_text(str(self.path_proyek), encoding="utf-8")
        except OSError as e:
            self.galat("Proyek tidak tersimpan", e)

    def _isi_daftar(self, entri):
        self.entri = {e["nama"]: e for e in entri}
        akt = self.proyek.get("tabel_aktif")
        self.daftar.clear()
        for e in entri:
            it = QListWidgetItem(f"{e['nama']}   ({ukuran(e['ukuran'])})")
            it.setData(Qt.UserRole, e["nama"])
            it.setToolTip(e["path"])
            self.daftar.addItem(it)
        for t in self.proyek.get("turunan", {}):
            it = QListWidgetItem(f"{t}   (turunan)")
            it.setData(Qt.UserRole, t)
            it.setToolTip(self.proyek["turunan"][t])
            self.daftar.addItem(it)
        self.status.setText(f"{len(entri)} tabel di {self.proyek['folder']}")
        if not entri:
            self.info.setText("Tidak ada file data (csv, tsv, txt, json, jsonl, parquet, xlsx) di folder ini.")
        nama = [self.daftar.item(i).data(Qt.UserRole) for i in range(self.daftar.count())]
        if akt not in nama:  # proyek baru: langsung buka tabel train (atau tabel pertama) agar layar tidak kosong
            akt = next((n for n in nama if "train" in n.lower()), nama[0] if nama else None)
        if akt in nama:
            self.daftar.setCurrentRow(nama.index(akt))
        self._perbarui_alur()

    def _saring(self, teks):
        t = teks.lower()
        for i in range(self.daftar.count()):
            it = self.daftar.item(i)
            it.setHidden(t not in it.data(Qt.UserRole).lower())

    # ---------- tabel + resep ----------
    def buka_tabel(self, nama, indeks=None):
        """Hitung tabel lewat resepnya di latar (tahap yang sama diambil dari cache); tampilkan tahap `indeks`
        (0 = sumber, k = setelah langkah k, None = hasil akhir)."""
        if self.aktif != nama:
            self.aktif = nama
            self.proyek["tabel_aktif"] = nama
            self.simpan()
        m, snap, entri = self.mesin, copy.deepcopy(self.proyek), dict(self.entri)

        def kerja():
            m.proyek, m.entri = snap, entri
            h = m.hasil(nama)
            i = len(h.tahap) - 1 if indeks is None else min(indeks, len(h.tahap) - 1)
            beda = None
            if i > 0 and h.tahap[i] is not h.tahap[i - 1]:
                beda = (resep.ringkas(h.tahap[i - 1], h.tahap[i]), resep.contoh_ubah(h.tahap[i - 1], h.tahap[i]))
            return h, i, beda

        self.jalankan(f"Menghitung {nama}", kerja, lambda r: self._terhitung(nama, indeks, *r),
                      lambda err: self._gagal_baca(nama, err))

    def _terhitung(self, nama, diminta, h, i, beda):
        if self.aktif != nama:
            return
        self.hasil_aktif, self.indeks = h, i
        self._gambar_resep(i if diminta is None else diminta)
        self._tampilkan(nama, h.tahap[i], h.lap if i == 0 else None, beda, h.galat, i)
        if self.cari_isi.text().strip() or self.tampil.currentIndex():
            self.cari()
        self._pahami()
        self._perbarui_alur()

    def _gambar_resep(self, baris):
        nama, h = self.aktif, self.hasil_aktif
        pem, r = resep.resep_dari(self.proyek, nama)
        if pem != nama:
            judul = f"RESEP · ikut {html.escape(pem)}<br><span style='font-weight:normal'>mengubah langkah di sini " \
                    "= mengubah resep itu</span>"
        else:
            pas = r.get("pasangan", [])
            judul = f"RESEP · {html.escape(nama)}<br><span style='font-weight:normal'>" + (
                f"pasangan: {html.escape(', '.join(pas))}" if pas else "tanpa pasangan (menu Resep)") + "</span>"
        self.judul_resep.setText(judul)
        self.resep.blockSignals(True)
        self.resep.clear()
        sumber = "kode turunan (klik ganda untuk ubah)" if nama in self.proyek.get("turunan", {}) else "baca file"
        self.resep.addItem(f"0 · sumber: {sumber}")
        for k, l in enumerate(r["langkah"], 1):
            it = QListWidgetItem(f"{k} · {langkah.ringkas(l)}")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if l.get("nyala", True) else Qt.Unchecked)
            it.setToolTip(l["kode"])
            if h.galat and k == h.galat[0]:
                it.setForeground(kuas("penting"))
                it.setToolTip(h.galat[1] + "\n\n" + l["kode"])
            elif (h.galat and k > h.galat[0]) or not l.get("nyala", True) or (pem != nama and l.get("hanya_latih")):
                it.setForeground(kuas("redup"))
            self.resep.addItem(it)
        self.resep.setCurrentRow(min(baris, self.resep.count() - 1))
        self.resep.blockSignals(False)

    def _tahap_dipilih(self, baris):
        if baris >= 0 and self.aktif and self.hasil_aktif:
            self.buka_tabel(self.aktif, baris)

    def _resep_aktif(self):
        pem = resep.pemilik(self.proyek, self.aktif)
        return self.proyek.setdefault("resep", {}).setdefault(pem, {"langkah": [], "pasangan": []})

    def _ubah_resep(self, baris):
        self.simpan()
        self._perbarui_alur()
        self.buka_tabel(self.aktif, baris)

    def _nyala_diubah(self, it):
        baris = self.resep.row(it)
        if baris >= 1:
            self._resep_aktif()["langkah"][baris - 1]["nyala"] = it.checkState() == Qt.Checked
            self._ubah_resep(baris)

    def _form(self, jenis, s, lama=None):
        """Form langkah ke-s (0..): kolom pilihan = kolom tabel sebelum langkah itu."""
        h = self.hasil_aktif
        df = h.tahap[min(s, len(h.tahap) - 1)]
        nama, m = self.aktif, self.mesin
        tabel = [t for t in self._nama_tabel() if t != nama]

        def pratinjau(kode, ok, gagal):
            snap, entri = copy.deepcopy(self.proyek), dict(self.entri)

            def kerja():
                m.proyek, m.entri = snap, entri
                return m.pratinjau(nama, s, kode)

            self.jalankan("Pratinjau", kerja, ok, gagal)

        dlg = FormLangkah(self, jenis, df.columns, tabel, lama, pratinjau)
        return dlg.langkah() if dlg.exec() == QDialog.Accepted else None

    def tambah_langkah(self, jenis, lama=None):
        if not self.hasil_aktif:
            return
        s = max(self.resep.currentRow(), 0)  # sisipkan setelah tahap yang sedang dilihat
        l = self._form(jenis, s, lama)
        if l:
            self._resep_aktif()["langkah"].insert(s, l)
            self._ubah_resep(s + 1)

    def ubah_langkah(self):
        baris = self.resep.currentRow()
        if baris == 0 and self.aktif in self.proyek.get("turunan", {}):
            return self.ubah_turunan(self.aktif)
        if baris < 1 or not self.hasil_aktif:
            return
        ls = self._resep_aktif()["langkah"]
        lama = ls[baris - 1]
        l = self._form(lama["jenis"], baris - 1, lama)
        if l:
            l["nyala"] = lama.get("nyala", True)
            ls[baris - 1] = l
            self._ubah_resep(baris)

    def hapus_langkah(self):
        baris = self.resep.currentRow()
        if baris < 1:
            return
        ls = self._resep_aktif()["langkah"]
        if QMessageBox.question(self, "Hapus langkah", f"Hapus langkah {baris}: {langkah.ringkas(ls[baris - 1])}?") \
                != QMessageBox.Yes:
            return
        ls.pop(baris - 1)
        self._ubah_resep(baris - 1)

    def geser_langkah(self, d):
        baris = self.resep.currentRow()
        ls = self._resep_aktif()["langkah"]
        j = baris - 1 + d
        if baris < 1 or not 0 <= j < len(ls):
            return
        ls[baris - 1], ls[j] = ls[j], ls[baris - 1]
        self._ubah_resep(j + 1)

    def _nama_tabel(self):
        return list(self.entri) + [t for t in self.proyek.get("turunan", {}) if t not in self.entri]

    def atur_pasangan(self):
        if not self.aktif:
            return
        pem = resep.pemilik(self.proyek, self.aktif)
        r = self.proyek.setdefault("resep", {}).setdefault(pem, {"langkah": [], "pasangan": []})
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Pasangan resep {pem}")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"Tabel yang ikut resep <b>{html.escape(pem)}</b> (mis. test): langkah sama, statistik "
                             "(median, modus, peta) dari data latih, langkah 'latih saja' dilewati.", wordWrap=True))
        daftar = QListWidget()
        for t in self._nama_tabel():
            if t == pem:
                continue
            it = QListWidgetItem(t)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if t in r["pasangan"] else Qt.Unchecked)
            daftar.addItem(it)
        lay.addWidget(daftar)
        tombol = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        tombol.accepted.connect(dlg.accept)
        tombol.rejected.connect(dlg.reject)
        lay.addWidget(tombol)
        if dlg.exec() != QDialog.Accepted:
            return
        r["pasangan"] = [daftar.item(i).text() for i in range(daftar.count())
                         if daftar.item(i).checkState() == Qt.Checked]
        for lain, rr in self.proyek["resep"].items():  # satu tabel hanya boleh ikut satu resep
            if lain != pem:
                rr["pasangan"] = [p for p in rr.get("pasangan", []) if p not in r["pasangan"]]
        self._ubah_resep(None)

    def ubah_turunan(self, nama):
        if nama is not None and nama not in self.proyek.get("turunan", {}):
            return QMessageBox.information(self, "Laya", "Tabel ini bukan tabel turunan. Buat lewat Tabel → Tabel "
                                                         "turunan baru…")
        if not self.proyek:
            return
        tabel = self._nama_tabel()
        dlg = QDialog(self)
        dlg.setWindowTitle("Tabel turunan")
        dlg.resize(700, 420)
        lay = QFormLayout(dlg)
        nm = QLineEdit(nama or "gabungan")
        kode = QPlainTextEdit()
        kode.setFont(QFont("Consolas", 10))
        contoh = ", ".join(f'T["{t}"]' for t in tabel[:2]) or 'T["a.csv"], T["b.csv"]'
        kode.setPlainText(self.proyek["turunan"][nama] if nama else
                          f'df = pl.concat([{contoh}], how="diagonal_relaxed")')
        lay.addRow("Nama", nm)
        lay.addRow(QLabel("Kode polars yang mengisi <b>df</b>. <tt>T[\"nama\"]</tt> = tabel itu setelah resepnya. "
                          "Tabel: " + html.escape(", ".join(tabel)), wordWrap=True))
        lay.addRow(kode)
        tombol = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        tombol.accepted.connect(dlg.accept)
        tombol.rejected.connect(dlg.reject)
        lay.addRow(tombol)
        while dlg.exec() == QDialog.Accepted:
            baru = nm.text().strip()
            try:
                compile(kode.toPlainText(), "<turunan>", "exec")
                if not baru or (baru in tabel and baru != nama):
                    raise ValueError(f"nama '{baru}' kosong atau sudah dipakai")
            except (SyntaxError, ValueError) as e:
                QMessageBox.warning(self, "Tabel turunan", str(e))
                continue
            tur = self.proyek.setdefault("turunan", {})
            tur.pop(nama, None)
            tur[baru] = kode.toPlainText()
            self.proyek["tabel_aktif"] = baru
            self.aktif = None
            self.simpan()
            self._isi_daftar(list(self.entri.values()))
            return

    def ekspor(self):
        if not self.aktif:
            return
        pem = resep.pemilik(self.proyek, self.aktif)
        awal = str(self.folder_kerja / f"bersihkan_{ekspor.bantu.nama_aman(pem)}.ipynb")
        f, _ = QFileDialog.getSaveFileName(self, "Ekspor resep", awal, "Notebook (*.ipynb);;Skrip Python (*.py)")
        if not f:
            return
        keluar = Path(f).parent / "bersih"
        fn = ekspor.ekspor_py if f.endswith(".py") else ekspor.ekspor_ipynb
        m, snap, entri, target = self.mesin, copy.deepcopy(self.proyek), dict(self.entri), self.aktif

        def kerja():
            m.proyek, m.entri = snap, entri
            fn(m, target, f, keluar)
            return f

        self.jalankan("Mengekspor", kerja, lambda p: self.status.setText(
            f"Diekspor: {p} (hasil ditulis ke {keluar}; ubah FOLDER/KELUAR di sel atas bila perlu)"))

    def simpan_bersih(self):
        if not self.aktif:
            return
        d = QFileDialog.getExistingDirectory(self, "Simpan tabel bersih ke folder", str(self.folder_kerja))
        if not d:
            return
        m, snap, entri, nama = self.mesin, copy.deepcopy(self.proyek), dict(self.entri), self.aktif

        def kerja():
            m.proyek, m.entri = snap, entri
            out = []
            for t in resep.kelompok(snap, nama):
                h = m.hasil(t)
                if h.galat:
                    raise resep.TabelGagal(f"{t}: langkah {h.galat[0]} gagal: {h.galat[1]}")
                out += ekspor.simpan_tabel(h.akhir, d, t)
            return out

        self.jalankan("Menyimpan tabel bersih", kerja,
                      lambda out: self.status.setText("Tersimpan: " + ", ".join(p.name for p in out) + f" di {d}"))

    # ---------- pahami: cari, sampel, profil, temuan (M3) ----------
    def _df_tahap(self):
        h = self.hasil_aktif
        return h.tahap[min(self.indeks, len(h.tahap) - 1)] if h and h.nama == self.aktif else None

    def _label(self):
        """Label tersimpan untuk resep tabel ini: None = belum ditebak, '' = tanpa label."""
        return self.proyek.get("label", {}).get(resep.pemilik(self.proyek, self.aktif)) if self.aktif else None

    def cari(self):
        df = self._df_tahap()
        if df is None:
            return
        mode, kol, isi, tampil, lab = (self.cari_mode.currentText(), self.cari_kolom.currentText(),
                                       self.cari_isi.text().strip(), self.tampil.currentIndex(), self._label())
        lab = temuan.tebak_label(df) if lab is None else (lab or None)  # belum dipilih: tebak dari nama kolom
        if tampil == 2 and lab not in df.columns:
            self.hasil_cari.setText("Pilih kolom label dulu di tab Temuan.")
            return

        def kerja():
            out, per = (temuan.cari(df, temuan.ekspresi_cari(mode, kol, isi, df), lab) if isi else (df, []))
            n = out.height
            if tampil == 1:
                out = temuan.sampel(out)
            elif tampil == 2:
                out = temuan.per_label(out, lab)
            return out, n, per

        def ok(r):
            out, n, per = r
            self._isi_grid(out)
            teks = f"{ribuan(n)} dari {ribuan(df.height)} baris cocok" if isi else ""
            if per:
                teks += " · per label: " + ", ".join(f"{html.escape(str(v))} {ribuan(c)} ({c / n:.0%})" for v, c in per)
            if tampil:
                teks += f" · ditampilkan {ribuan(out.height)} ({TAMPIL[tampil]})"
            self.hasil_cari.setText(teks)

        self.jalankan("Mencari", kerja, ok, lambda e: self.hasil_cari.setText(
            f"<span style='color:{pahami.W()['penting']}'>{html.escape(resep.pesan_galat(e))}</span>"))

    def cari_jadi_langkah(self):
        df, isi = self._df_tahap(), self.cari_isi.text().strip()
        if df is None or not isi:
            return
        try:
            kode = temuan.ekspresi_cari(self.cari_mode.currentText(), self.cari_kolom.currentText(), isi, df)
        except Exception as e:
            return self.galat("Pencarian tidak bisa dijadikan langkah", e)
        l = langkah.buat("kode", {"isi": f"df = df.filter({kode})  # simpan baris yang cocok; ~(...) untuk membuang"})
        l["hanya_latih"] = True  # menyaring baris test akan mengacaukan submission
        self.tambah_langkah("kode", l)

    def _pahami(self):
        """Hitung profil/temuan tahap yang sedang dilihat bila tabnya terbuka dan belum dihitung."""
        t, df = self.tab.currentIndex(), self._df_tahap()
        if t == 3:
            return self.model.segarkan()
        if t == 0 or df is None:
            return
        lab = self._label()
        lama = self.sudah.get(t)
        if lama and lama[0] is df and lama[1] == lab:
            return
        self.sudah[t] = (df, lab)
        nama, i = self.aktif, self.indeks
        if t == 1:
            self.profil.setRowCount(0)
            self.detail_kolom.setHtml(f"<p>{pahami.redup('Menghitung profil…')}</p>")
            self.jalankan("Menghitung profil", lambda: temuan.profil(df),
                          lambda pr: self.aktif == nama and self._isi_profil(pr))
            return
        self.temuan.clear()
        self._kosongkan_rincian("Mencari temuan…")
        m, snap, entri = self.mesin, copy.deepcopy(self.proyek), dict(self.entri)
        pem, r = resep.resep_dari(snap, nama)
        pasangan = r.get("pasangan", []) if pem == nama else []

        def kerja():
            m.proyek, m.entri = snap, entri
            uji = {}
            for p in pasangan:
                hp = m.hasil(p)
                uji[p] = hp.tahap[min(i, len(hp.tahap) - 1)]
            contoh = None
            if lab is None and not uji:  # tanpa tabel test: kolom sample_submission selain id = label
                e = next((e for n, e in entri.items() if "sample" in n.lower() or "submission" in n.lower()), None)
                try:
                    contoh = muat.baca(e["path"], e["sheet"])[0].columns if e else None
                except Exception as x:
                    log.info("sample_submission tidak terbaca: %r", x)
            label = lab if lab is not None else (temuan.tebak_label(df, next(iter(uji.values()), None), contoh) or "")
            ts = temuan.temuan(df, label or None)
            for p, d in uji.items():
                ts += temuan.banding(df, d, label or None, p)
            return label, temuan.urut(ts)

        def ok(r):
            label, ts = r
            if self.aktif != nama:
                return
            if lab is None:
                self.proyek.setdefault("label", {})[resep.pemilik(self.proyek, nama)] = label
                self.sudah[2] = (df, label)
                self.simpan()
            self._isi_temuan(ts, df.columns, label)

        self.jalankan("Mencari temuan", kerja, ok)

    def _isi_temuan(self, ts, kolom_df=None, label=None):
        """Pohon temuan: Penting / Perhatian / Info, tiap anak menyimpan indeksnya di daftar_temuan."""
        if kolom_df is not None:
            self.label_pilih.clear()
            self.label_pilih.addItems([TANPA_LABEL] + kolom_df)
            self.label_pilih.setCurrentText(label or TANPA_LABEL)
        self.daftar_temuan = ts
        self.temuan.clear()
        tebal = QFont()
        tebal.setBold(True)
        for tingkat in (2, 1, 0):
            anak = [(i, t) for i, t in enumerate(ts) if t["tingkat"] == tingkat]
            if not anak:
                continue
            nama, warna = pahami.tingkat(tingkat)
            kepala = QTreeWidgetItem(self.temuan, [f"{nama} ({len(anak)})"])
            kepala.setFont(0, tebal)
            kepala.setForeground(0, QBrush(QColor(warna)))
            kepala.setFirstColumnSpanned(True)
            for i, t in anak:
                it = QTreeWidgetItem(kepala, [t["judul"], t["grup"]])
                it.setData(0, Qt.UserRole, i)
                it.setToolTip(0, t["judul"])
                it.setForeground(1, kuas("redup"))
            kepala.setExpanded(True)
        n2 = sum(t["tingkat"] == 2 for t in ts)
        self.tab.setTabText(2, f"Temuan ({len(ts)})")
        self._kosongkan_rincian(f"{len(ts)} temuan, {n2} penting. Pilih satu di tengah untuk rinciannya, lalu "
                                "<b>Lihat baris</b> atau <b>Jadikan langkah</b>.")
        self._perbarui_alur()

    def _label_diubah(self, _):
        if self.aktif:
            teks = self.label_pilih.currentText()
            self.proyek.setdefault("label", {})[resep.pemilik(self.proyek, self.aktif)] = \
                "" if teks == TANPA_LABEL else teks
            self.simpan()
            self._pahami()

    def _temuan_terpilih(self):
        it = self.temuan.currentItem()
        i = it.data(0, Qt.UserRole) if it else None
        return self.daftar_temuan[i] if i is not None and i < len(self.daftar_temuan) else None

    def _temuan_dipilih(self, _):
        t = self._temuan_terpilih()
        if not t:
            return self._kosongkan_rincian()
        self.temuan_rinci.setHtml(pahami.html_temuan(t))
        self.lihat.setEnabled(bool(t["saring"]))
        self.jadikan.setEnabled(bool(t["langkah"]))

    def _kosongkan_rincian(self, pesan=None):
        self.temuan_rinci.setHtml(f"<p>{pahami.redup(pesan or 'Pilih satu temuan di tengah.')}</p>")
        self.lihat.setEnabled(False)
        self.jadikan.setEnabled(False)

    def lihat_baris(self):
        t = self._temuan_terpilih()
        if t and t["saring"]:
            self.cari_mode.setCurrentText("kode polars")
            self.cari_isi.setText(t["saring"])
            self.tab.setCurrentIndex(0)
            self.cari()

    def temuan_jadi_langkah(self):
        t = self._temuan_terpilih()
        if not t or not t["langkah"]:
            return
        if t.get("tabel") and t["tabel"] != self.aktif:
            return QMessageBox.information(self, "Laya", f"Langkah ini dipasang di resep {t['tabel']}. Buka tabel "
                                                         "itu lalu tekan lagi tombol Relasi antartabel.")
        self.tambah_langkah(t["langkah"][0], langkah.buat(*t["langkah"]))

    def _tab_diganti(self, i):
        self.kanan.setVisible(i != 3)  # tab Model menampilkan hasilnya sendiri selebar mungkin
        if i < 3:
            self.kanan.setCurrentIndex(i)
        self._pahami()

    def _isi_profil(self, pr):
        self.data_profil = pr
        self.profil.setSortingEnabled(False)
        self.profil.setRowCount(len(pr))
        for r, d in enumerate(pr):
            k = pahami.kosong(d)
            sel = [Sel(d["kolom"], r), Sel(d["tipe"]),
                   Sel(f"{ribuan(k)} ({k / (d['baris'] or 1):.0%})" if k else "–", k),
                   Sel(ribuan(d["unik"]) if d["unik"] is not None else "–", d["unik"] or 0),
                   Sel(""), Sel(pahami.ringkas(d))]
            sel[0].setData(Qt.UserRole, r)
            sel[4].setData(Qt.UserRole, pahami.sebaran(d))
            if k:
                sel[2].setForeground(kuas("perhatian"))
            for c in (2, 3):
                sel[c].setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            for c, x in enumerate(sel):
                self.profil.setItem(r, c, x)
            sel[5].setToolTip(sel[5].text())
        self.profil.setSortingEnabled(True)
        self.profil.resizeColumnsToContents()
        self.profil.setColumnWidth(4, 120)
        self.profil.horizontalHeader().setStretchLastSection(True)
        self.profil.setCurrentCell(0, 0)  # rincian kanan langsung terisi; panah atas/bawah pindah kolom

    def _kolom_dipilih(self, r):
        it = self.profil.item(r, 0) if r >= 0 else None
        if it is not None and it.data(Qt.UserRole) is not None:
            self.detail_kolom.setHtml(pahami.html_kolom(self.data_profil[it.data(Qt.UserRole)]))

    def cari_relasi(self):
        if not self.proyek:
            return
        m, snap, entri, nama, semua = self.mesin, copy.deepcopy(self.proyek), dict(self.entri), self.aktif, \
            self._nama_tabel()

        def kerja():
            m.proyek, m.entri = snap, entri
            tabel = {}
            for t in semua:
                try:
                    tabel[t] = m.hasil(t).akhir
                except Exception as e:
                    log.info("relasi: %s dilewati: %r", t, e)
            return [t for t in temuan.relasi(tabel) if nama is None or nama in t["antara"]]

        self.jalankan("Mencari relasi antartabel", kerja, lambda ts: self._isi_temuan(
            ts or [temuan.T(0, "relasi", f"Tidak ada kolom kunci bersama antara {nama} dan tabel lain.")]))

    def _gagal_baca(self, nama, e):
        if self.aktif != nama:
            return
        self.info.setText(f"<b>{html.escape(nama)}: gagal dibaca.</b> {html.escape(str(e))}<br>"
                          "Atur lewat Tabel → Opsi baca (Ctrl+B): pemisah, encoding, baris header, lewati baris.")
        self.mentah.setPlainText("100 baris awal apa adanya:\n\n" + getattr(e, "mentah", ""))
        self.tumpuk.setCurrentWidget(self.mentah)
        self.judul_panel.setText("")
        self.panel.setPlainText("")
        self.status.setText(f"{nama}: gagal dibaca")

    def _isi_grid(self, df):
        lama = self.grid.model()
        self.grid.setModel(ModelTabel(df, self.grid))
        if lama:
            lama.deleteLater()
        self.grid.selectionModel().currentChanged.connect(self._sel_dipilih)
        self.tumpuk.setCurrentWidget(self.grid)

    def _tampilkan(self, nama, df, lap=None, beda=None, galat=None, i=0):
        self._isi_grid(df)
        kol = [temuan.SEMUA] + df.columns
        if [self.cari_kolom.itemText(k) for k in range(self.cari_kolom.count())] != kol:
            lama = self.cari_kolom.currentText()
            self.cari_kolom.clear()
            self.cari_kolom.addItems(kol)
            self.cari_kolom.setCurrentText(lama if lama in kol else temuan.SEMUA)
        self.hasil_cari.setText("")
        ringkas = (f"<span style='font-size:13pt; font-weight:600'>{html.escape(nama)}</span>&nbsp;&nbsp;"
                   f"<span style='color:{pahami.W()['redup']}'>{ribuan(df.height)} baris × {df.width} kolom")
        if not lap:
            ringkas += f" · setelah langkah {i}"
        if lap:
            ringkas += f" · {lap.get('format')}"
            if "encoding" in lap:
                ringkas += f" · {lap['encoding']} · pemisah {nama_pemisah(lap['pemisah'])}"
            ringkas += f" · {lap['detik']} s"
        ringkas += "</span>"
        for c in (lap or {}).get("catatan", []):
            ringkas += f"<br><span style='color:{pahami.W()['perhatian']}'>Catatan baca: {html.escape(c)}</span>"
        if beda:
            ringkas += "<br>" + html.escape(resep.teks_ringkas(beda[0])).replace("\n", " · ")
            for c, r, x, y in beda[1]:
                ringkas += (f"<br><tt>{html.escape(c)}[{r}]: {html.escape(teks_sel(x)[:70])} → "
                            f"{html.escape(teks_sel(y)[:70])}</tt>")
        if galat:
            ringkas += (f"<br><span style='color:{pahami.W()['penting']}'><b>Langkah {galat[0]} gagal:</b> "
                        f"{html.escape(galat[1]).replace(chr(10), '<br>')}</span>")
        self.info.setText(ringkas)
        self.judul_panel.setText(sel_kosong())
        self.panel.setPlainText("")
        self.tab.setTabText(2, "Temuan")
        self.status.setText(f"{nama}: {ribuan(df.height)} baris × {df.width} kolom")

    def _sel_dipilih(self, cur, _):
        if not cur.isValid():
            return
        m = self.grid.model()
        r, c = cur.row(), cur.column()
        v = m.nilai(r, c)
        judul = f"<b>{html.escape(m.df.columns[c])}</b> · baris {r} · {m.df.dtypes[c]}"
        if isinstance(v, str):
            judul += f" · {ribuan(len(v))} huruf"
        self.judul_panel.setText(judul)
        self.panel.setPlainText(teks_penuh(v))

    def atur_opsi(self):
        if self.aktif not in self.entri:
            return
        opsi = dict(self.proyek["opsi_baca"].get(self.aktif, {}))
        lap = self.hasil_aktif.lap if self.hasil_aktif and self.hasil_aktif.nama == self.aktif else {}
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Opsi baca: {self.aktif}")
        form = QFormLayout(dlg)
        pemisah = QComboBox(editable=True)
        pemisah.addItems(PEMISAH)
        pemisah.setCurrentText(nama_pemisah(opsi.get("pemisah")) if opsi.get("pemisah") else "otomatis")
        enc = QComboBox(editable=True)
        enc.addItems(ENCODING)
        enc.setCurrentText(opsi.get("encoding") or "otomatis")
        header = QCheckBox("baris pertama = nama kolom", checked=opsi.get("header", True))
        lewati = QSpinBox(maximum=10_000_000, value=int(opsi.get("lewati", 0)))
        teks = QCheckBox("semua kolom sebagai teks (tanpa tebak angka)", checked=bool(opsi.get("semua_teks")))
        terdeteksi = f"pemisah {nama_pemisah(lap['pemisah'])}, encoding {lap['encoding']}" if "encoding" in lap else "-"
        form.addRow("Terdeteksi", QLabel(terdeteksi))
        form.addRow("Pemisah", pemisah)
        form.addRow("Encoding", enc)
        form.addRow("", header)
        form.addRow("Lewati baris awal", lewati)
        form.addRow("", teks)
        tombol = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        tombol.accepted.connect(dlg.accept)
        tombol.rejected.connect(dlg.reject)
        tombol.button(QDialogButtonBox.RestoreDefaults).clicked.connect(lambda: dlg.done(2))
        form.addRow(tombol)
        hasil = dlg.exec()
        if hasil == QDialog.Rejected:
            return
        baru = {}
        if hasil != 2:
            sep = PEMISAH.get(pemisah.currentText(), pemisah.currentText())
            if sep:
                baru["pemisah"] = sep
            if enc.currentText() != "otomatis":
                baru["encoding"] = enc.currentText()
            if not header.isChecked():
                baru["header"] = False
            if lewati.value():
                baru["lewati"] = lewati.value()
            if teks.isChecked():
                baru["semua_teks"] = True
        self.proyek["opsi_baca"][self.aktif] = baru
        self.simpan()
        self.buka_tabel(self.aktif, 0)

    def closeEvent(self, ev):
        self.simpan()
        p = self.model.proses
        self.model.batal()  # tugas model tidak dibiarkan memakan CPU setelah Laya ditutup
        if p:
            p.waitForFinished(3000)
        self._thread.quit()
        self._thread.wait(3000)
        ev.accept()

    # ---------- uji exe ----------
    def uji_otomatis(self, folder):
        """`Laya.exe --uji <folder>`: buka + tampilkan semua tabel, catat waktu ke laya_log.txt, lalu keluar."""
        t = time.perf_counter()
        entri = muat.pindai(folder)
        log.info("UJI pindai %s: %d tabel, %.2f s", folder, len(entri), time.perf_counter() - t)
        self.entri = {e["nama"]: e for e in entri}
        gagal = 0
        for e in entri:
            t = time.perf_counter()
            try:
                self.cache[e["nama"]] = muat.baca(e["path"], e["sheet"])
            except muat.BacaGagal as g:
                gagal += 1
                log.info("UJI %s: GAGAL %s", e["nama"], g)
                continue
            self.aktif = e["nama"]
            df, lap = self.cache[e["nama"]]
            self._tampilkan(e["nama"], df, lap)
            self.grid.scrollToBottom()
            QApplication.processEvents()
            log.info("UJI %s: %d x %d, baca+tampil %.2f s", e["nama"], df.height, df.width, time.perf_counter() - t)
        # resep di exe beku: map_elements (tanpa numpy) + salinan sumber untuk ekspor
        m = resep.Mesin({"folder": folder, "turunan": {"x": 'df = pl.DataFrame({"t": ["[\'a\']", "kerennnn"]})'},
                         "resep": {"x": {"pasangan": [], "langkah": [
                             langkah.buat("list_teks", {"kolom": ["t"]}),
                             langkah.buat("kode", {"isi": 'df = df.with_columns(u=ganti_regex_py(pl.lit("aaab"), '
                                                          'r"(a)\\1+", r"\\1"))'})]}}}, {})
        h = m.hasil("x")
        log.info("UJI resep: galat=%s, hasil=%s, panjang skrip ekspor=%d", h.galat, h.akhir.rows(),
                 len(ekspor.skrip(m, "x", "keluar")))
        if self.aktif in self.cache:
            t = time.perf_counter()
            df = self.cache[self.aktif][0]
            ts = temuan.temuan(df, temuan.tebak_label(df))
            log.info("UJI temuan %s: %d butir, %d cek gagal, profil %d kolom, %.2f s", self.aktif, len(ts),
                     sum(x["grup"] == "cek gagal" for x in ts), len(temuan.profil(df)), time.perf_counter() - t)
        py = self.model._cari_python()
        log.info("UJI tugas model: python=%s, skrip ada=%s", py, venv.skrip_tugas().is_file())
        if py and self.aktif in self.cache:  # G1 sungguhan: tugas kata khas pada 2.000 baris tabel terakhir
            df = self.cache[self.aktif][0]
            teks, lab = temuan.kolom_teks(df)[:1], temuan.tebak_label(df)
            if teks and lab:
                t = time.perf_counter()
                d = self.folder_kerja / "tugas" / "uji_exe"
                d.mkdir(parents=True, exist_ok=True)
                df.head(2000).select(teks + [lab]).write_parquet(d / "latih.parquet")
                (d / "tugas.json").write_text(json.dumps({"jenis": "kata", "latih": str(d / "latih.parquet"),
                                                          "teks": teks, "label": lab, "keluar": str(d)}), encoding="utf-8")
                p = QProcess()
                p.setProcessChannelMode(QProcess.MergedChannels)
                p.start(py, [str(venv.skrip_tugas()), str(d / "tugas.json")])
                p.waitForFinished(300_000)
                log.info("UJI tugas kata (%s, label %s): exit %s, hasil.json ada=%s, %.1f s\n%s", teks, lab, p.exitCode(),
                         (d / "hasil.json").exists(), time.perf_counter() - t,
                         bytes(p.readAll()).decode("utf-8", "replace")[-600:])
        log.info("UJI selesai: %d tabel, %d gagal", len(entri), gagal)
        QApplication.quit()
