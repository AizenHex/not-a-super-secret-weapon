"""Tab Model (M4): tugas model cepat (mesin/tugas.py) dijalankan python venv di proses terpisah (G1), satu per satu,
dengan log langsung dan tombol Batal yang benar-benar menghentikan proses. Hasil tampil lebar di kanan tab ini."""
import copy
import html
import json
import sys
import time
from pathlib import Path

import polars as pl
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtGui import QFont
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QComboBox, QFileDialog, QLabel, QListWidget, QMenu, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QSplitter, QTextBrowser, QWidget, QVBoxLayout)

from mesin import langkah, resep, submisi, temuan, venv
from tampilan import pahami
from tampilan.tata import bagian, baris, kolom

VALIDASI = ["otomatis", "stratified k-fold", "k-fold biasa", "grup: teks kembar", "grup: kolom", "holdout"]
TIP_VALIDASI = ("otomatis: grup teks kembar bila ada kolom teks, selain itu stratified (klasifikasi) / k-fold (regresi)."
                "\ngrup: baris dengan teks/nilai kolom sama tidak pernah terpisah ke latih dan validasi (anti bocor)."
                "\nholdout: 20% baris sekali pisah; cepat tapi skornya lebih goyang.")
TIP = {"tabular": "LightGBM di semua kolom (angka, kategori, tanggal) selain label/id: skor CV, kolom paling "
                  "berpengaruh, matriks kebingungan. Untuk data tabel; tidak butuh kolom teks.",
       "baseline": "macro-F1 CV 5 fold, matriks kebingungan, kata per label, dan baris yang labelnya mencurigakan "
                   "(cleanlab). GDOC 52 rb artikel ± 40 s.",
       "kata": "Kata/bigram yang khas untuk tiap label (log-odds).",
       "acak": "CV dengan label diacak; skor jauh di atas tebakan = ada kebocoran.",
       "adversarial": "Bisakah model membedakan train dari test? Butuh tabel pasangan (menu Resep)."}


class TabModel(QWidget):
    def __init__(self, j):
        super().__init__()
        self.j = j
        self.riwayat, self.proses, self.python = [], None, None
        self.teks_tombol = QPushButton("Kolom teks: –", toolTip="Kolom yang digabung jadi teks masukan model teks")
        self.teks_tombol.setMenu(QMenu(self))
        self.ket = QLabel(objectName="meta")
        py = QPushButton("Python…", toolTip="Pilih python.exe venv (yang punya scikit-learn) secara manual")
        py.clicked.connect(self.pilih_python)
        tombol = []
        for jenis in TIP:
            b = QPushButton(pahami.JUDUL[jenis], toolTip=TIP[jenis])
            b.clicked.connect(lambda _=False, jenis=jenis: self.mulai(jenis))
            tombol.append(b)
        batal = QPushButton("Batal", toolTip="Hentikan tugas yang berjalan dan kosongkan antrean")
        batal.clicked.connect(self.batal)

        self.v_cara = QComboBox(toolTip=TIP_VALIDASI)
        self.v_cara.addItems(VALIDASI)
        self.v_fold = QSpinBox(minimum=2, maximum=20, value=5, toolTip="Jumlah fold (diabaikan untuk holdout)")
        self.v_kolom = QComboBox(minimumContentsLength=14, toolTip="Kolom pengelompokan untuk 'grup: kolom'")
        for w in (self.v_cara, self.v_kolom):
            w.activated.connect(lambda _: self._simpan_validasi())
        self.v_fold.valueChanged.connect(lambda _: self._simpan_validasi())

        self.daftar = QListWidget()
        self.daftar.currentRowChanged.connect(self._dipilih)
        self.log = QPlainTextEdit(readOnly=True, maximumBlockCount=3000)
        self.log.setFont(QFont("Consolas", 9))
        self.rinci = QTextBrowser()
        self.lihat = QPushButton("Lihat baris", toolTip="Tampilkan 50 baris paling mencurigakan di tab Data")
        self.lihat.clicked.connect(self.lihat_baris)
        self.jadikan = QPushButton("Jadikan langkah…", toolTip="Langkah buang baris mencurigakan (latih saja); "
                                                                "periksa dulu lewat Lihat baris")
        self.jadikan.clicked.connect(self.jadi_langkah)

        kiri = QSplitter(Qt.Vertical)
        kiri.addWidget(kolom(bagian("RIWAYAT"), self.daftar))
        kiri.addWidget(kolom(bagian("LOG"), self.log))
        kiri.setSizes([300, 200])
        isi = QSplitter()
        isi.addWidget(kiri)
        isi.addWidget(kolom(bagian("HASIL"), self.rinci, baris(self.lihat, self.jadikan, None)))
        isi.setStretchFactor(1, 1)
        isi.setSizes([330, 900])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(baris(self.ket, None, self.teks_tombol, py))
        lay.addWidget(baris(QLabel("Validasi"), self.v_cara, QLabel("fold"), self.v_fold, self.v_kolom, None))
        lay.addWidget(baris(*tombol, None, batal))
        lay.addWidget(isi, 1)
        self._dipilih(-1)

    # ---------- pilihan teks + label ----------
    def _pemilik(self):
        return resep.pemilik(self.j.proyek, self.j.aktif)

    def _label(self, df):
        lab = self.j._label()
        lab = temuan.tebak_label(df) if lab is None else lab
        return lab if lab in df.columns else None

    def _teks(self, df):
        simpan = self.j.proyek.get("teks", {}).get(self._pemilik())
        kol = [c for c in simpan if c in df.columns] if simpan is not None else temuan.kolom_teks(df)
        return kol

    def segarkan(self):
        """Dipanggil saat tab Model dibuka atau tabel/tahap berganti."""
        df = self.j._df_tahap()
        if df is None:
            self.ket.setText("Buka tabel train dulu.")
            return
        teks, lab = self._teks(df), self._label(df)
        menu = self.teks_tombol.menu()
        menu.clear()
        for c in df.columns:
            if df[c].dtype == pl.String:
                a = menu.addAction(c)
                a.setCheckable(True)
                a.setChecked(c in teks)
                a.toggled.connect(lambda nyala, c=c: self._ganti_teks(c, nyala))
        self.teks_tombol.setText("Kolom teks: " + (", ".join(teks) or "tidak ada"))
        self.ket.setText(f"tabel <b>{html.escape(self.j.aktif)}</b> tahap {self.j.indeks} · label "
                         f"<b>{html.escape(lab or 'belum dipilih')}</b> (ubah di tab Temuan)")
        v = self.j.proyek.get("validasi") or {}
        for w in (self.v_cara, self.v_kolom, self.v_fold):
            w.blockSignals(True)
        self.v_cara.setCurrentText(v.get("cara") or "otomatis")
        self.v_fold.setValue(int(v.get("fold") or 5))
        self.v_kolom.clear()
        self.v_kolom.addItems([c for c in df.columns if c != lab])
        self.v_kolom.setCurrentText(v.get("kolom") or "")
        for w in (self.v_cara, self.v_kolom, self.v_fold):
            w.blockSignals(False)
        self._atur_validasi()

    def _atur_validasi(self):
        cara = self.v_cara.currentText()
        self.v_kolom.setVisible(cara == "grup: kolom")
        self.v_fold.setEnabled(cara != "holdout")

    def _simpan_validasi(self):
        if not self.j.proyek:
            return
        self.j.proyek["validasi"] = {"cara": self.v_cara.currentText(), "fold": self.v_fold.value(),
                                     "kolom": self.v_kolom.currentText() or None}
        self.j.simpan()
        self._atur_validasi()

    def _ganti_teks(self, c, nyala):
        df = self.j._df_tahap()
        teks = [x for x in self._teks(df) if x != c] + ([c] if nyala else [])
        self.j.proyek.setdefault("teks", {})[self._pemilik()] = [x for x in df.columns if x in teks]
        self.j.simpan()
        self.segarkan()

    def pilih_python(self):
        f, _ = QFileDialog.getOpenFileName(self, "Pilih python.exe venv", "", "python.exe (python.exe)")
        if f:
            (self.j.folder_kerja / "python.txt").write_text(f, encoding="utf-8")
            self.python = f
            self.log.appendPlainText(f"python: {f}")

    def _cari_python(self):
        if not self.python:
            pilihan = self.j.folder_kerja / "python.txt"
            pilihan = pilihan.read_text(encoding="utf-8").strip() if pilihan.exists() else None
            self.python = venv.cari_python(Path(sys.executable).parent, Path(__file__).parent, self.j.folder_kerja,
                                           (self.j.proyek or {}).get("folder"), pilihan=pilihan)
        return self.python

    # ---------- antrean + proses ----------
    def mulai(self, jenis):
        j, df = self.j, self.j._df_tahap()
        if df is None:
            return QMessageBox.information(self, "Laya", "Buka tabel train dulu.")
        teks, lab = self._teks(df), self._label(df)
        if not teks and jenis != "tabular":
            return QMessageBox.information(self, "Laya", "Tugas ini butuh kolom teks (tombol Kolom teks). Untuk data "
                                                         "tabel pakai Baseline tabular.")
        if jenis != "adversarial" and not lab:
            return QMessageBox.information(self, "Laya", "Tabel ini belum punya label. Pilih kolom label di tab Temuan.")
        pem, r = resep.resep_dari(j.proyek, j.aktif)
        pasangan = r.get("pasangan", []) if pem == j.aktif else []
        if jenis == "adversarial" and not pasangan:
            return QMessageBox.information(self, "Laya", "Adversarial butuh tabel test sebagai pasangan resep "
                                                         "(Resep → Tabel pasangan).")
        if not self._cari_python():
            return QMessageBox.warning(self, "Laya", "Python venv tidak ditemukan (dicari .venv/venv di folder Laya, "
                                                     "folder data, dan semua folder di atasnya). Jalankan setup.bat "
                                                     "atau pilih python.exe lewat tombol Python….")
        folder = j.folder_kerja / "tugas" / f"{time.strftime('%Y%m%d_%H%M%S')}_{jenis}"
        idk = next((c for c in temuan.kolom_id(df) if c not in teks and c != lab), None)
        m, snap, entri, nama, i = j.mesin, copy.deepcopy(j.proyek), dict(j.entri), j.aktif, j.indeks
        v = j.proyek.get("validasi") or None
        grup = v.get("kolom") if v and v.get("cara") == "grup: kolom" else None
        if v and v.get("cara") == "grup: kolom" and grup not in df.columns:
            return QMessageBox.information(self, "Laya", "Pilih kolom grup untuk validasi 'grup: kolom'.")
        t = {"jenis": jenis, "latih": str(folder / "latih.parquet"), "uji": None, "teks": teks, "label": lab,
             "id": idk, "keluar": str(folder), "seed": 42, "fold": 5, "validasi": v}
        pakai = df.columns if jenis == "tabular" else list(dict.fromkeys(teks + [c for c in (lab, idk, grup) if c]))

        def kerja():
            folder.mkdir(parents=True, exist_ok=True)
            df.select(pakai).write_parquet(t["latih"])
            if jenis == "adversarial":
                m.proyek, m.entri = snap, entri
                hp = m.hasil(pasangan[0])
                d = hp.tahap[min(i, len(hp.tahap) - 1)]
                hilang = [c for c in teks if c not in d.columns]
                if hilang:
                    raise ValueError(f"kolom teks {hilang} tidak ada di {pasangan[0]}")
                t["uji"] = str(folder / "uji.parquet")
                d.select(teks).write_parquet(t["uji"])
            (folder / "tugas.json").write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")

        ket = f"{nama} tahap {i} · " + (f"{len(pakai)} kolom" if jenis == "tabular" else f"teks {', '.join(teks)}") \
            + (f" · label {lab}" if lab else "") + \
              (f" · test {pasangan[0]}" if jenis == "adversarial" else "")
        j.jalankan(f"Menyiapkan {pahami.JUDUL[jenis]}", kerja, lambda _: self._antre(
            {"jenis": jenis, "folder": folder, "ket": ket, "tabel": nama, "id": idk, "status": "antre", "log": [],
             "jam": time.strftime("%H.%M"), "hasil": None}))

    def _antre(self, r):
        self.riwayat.append(r)
        self._gambar(len(self.riwayat) - 1)
        self._berikut()

    def _berikut(self):
        r = next((x for x in self.riwayat if x["status"] == "antre"), None)
        if self.proses or not r:
            return
        r["status"], r["t0"] = "jalan", time.perf_counter()
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(lambda: self._baca(r, p))
        p.finished.connect(lambda kode, _: self._selesai(r, p, kode))
        p.errorOccurred.connect(lambda e: e == QProcess.FailedToStart and self._selesai(r, p, -1))
        self.proses = p
        self.log.appendPlainText(f"\n=== {pahami.JUDUL[r['jenis']]} · {r['ket']}\n{self.python}")
        p.start(self.python, [str(venv.skrip_tugas()), str(r["folder"] / "tugas.json")])
        self._gambar()

    def _baca(self, r, p):
        teks = bytes(p.readAllStandardOutput()).decode("utf-8", "replace").rstrip()
        if teks:
            r["log"] += teks.splitlines()
            self.log.appendPlainText(teks)

    def _selesai(self, r, p, kode):
        if self.proses is not p:
            return
        self._baca(r, p)
        self.proses = None
        f = r["folder"] / "hasil.json"
        if r["status"] == "jalan":
            if kode == 0 and f.exists():
                r["hasil"], r["status"] = json.loads(f.read_text(encoding="utf-8")), "ok"
            else:
                r["status"] = "gagal"
                if kode == -1:
                    r["log"].append(f"python tidak bisa dijalankan: {self.python} ({p.errorString()})")
        try:
            (r["folder"] / "log.txt").write_text("\n".join(r["log"]), encoding="utf-8")
        except OSError:
            pass
        p.deleteLater()
        self._gambar(self.riwayat.index(r))
        self._berikut()
        self.j._perbarui_alur()

    def batal(self):
        for r in self.riwayat:
            if r["status"] in ("antre", "jalan"):
                r["status"] = "batal"
        if self.proses:
            self.proses.kill()  # finished menyusul lalu _selesai merapikan
        self._gambar()

    # ---------- riwayat + panel kanan ----------
    def _gambar(self, pilih=None):
        baris_ = self.daftar.currentRow() if pilih is None else pilih
        self.daftar.blockSignals(True)
        self.daftar.clear()
        w = pahami.W()
        for r in self.riwayat:
            n = pahami.ringkas_hasil(r) if r["status"] == "ok" else r["status"]
            detik = f" · {r['hasil']['detik']} s" if r["status"] == "ok" and "detik" in r["hasil"] else ""
            self.daftar.addItem(f"{r['jam']}  {pahami.JUDUL[r['jenis']]}\n{n} · {r['tabel']}{detik}")
            if r["status"] in ("gagal", "batal"):
                self.daftar.item(self.daftar.count() - 1).setForeground(QBrush(QColor(w["penting"])))
        self.daftar.setCurrentRow(baris_)
        self.daftar.blockSignals(False)
        self._dipilih(baris_)

    def _terpilih(self):
        i = self.daftar.currentRow()
        return self.riwayat[i] if 0 <= i < len(self.riwayat) else None

    def _dipilih(self, _):
        r = self._terpilih()
        self.rinci.setHtml(pahami.html_hasil(r) if r else
                           "<h3>Belum ada hasil</h3><p>Atur <b>Validasi</b>, lalu tekan <b>Baseline tabular</b> (data "
                           "tabel) atau <b>Baseline teks</b> (kolom kalimat). Tugas berjalan di python venv terpisah; "
                           "Laya tetap bisa dipakai. Hasil yang dipilih di RIWAYAT tampil di sini.</p>")
        ada = bool(r and r["status"] == "ok" and r["hasil"].get("curiga"))
        self.lihat.setEnabled(ada)
        self.jadikan.setEnabled(ada)

    def _saring(self, r, n=50):
        c = r["hasil"]["curiga"][:n]
        if r["id"]:
            return f"pl.col({langkah.lit(r['id'])}).cast(pl.String).is_in({langkah.lits([x['id'] for x in c])})"
        return f"pl.int_range(pl.len()).is_in({[x['baris'] for x in c]})"  # nomor baris tahap saat tugas dijalankan

    def _tabel_sama(self, r):
        if self.j.aktif == r["tabel"]:
            return True
        QMessageBox.information(self, "Laya", f"Hasil ini dari tabel {r['tabel']}. Buka tabel itu dulu.")
        return False

    def lihat_baris(self):
        r = self._terpilih()
        if r and self._tabel_sama(r):
            self.j.cari_mode.setCurrentText("kode polars")
            self.j.cari_isi.setText(self._saring(r))
            self.j.tab.setCurrentIndex(0)
            self.j.cari()

    def jadi_langkah(self):
        r = self._terpilih()
        if r and self._tabel_sama(r):
            l = langkah.buat("kode", {"isi": f"# periksa dulu lewat Lihat baris; sisakan hanya yang benar-benar salah "
                                              f"label\ndf = df.filter(~{self._saring(r)})"})
            l["hanya_latih"] = True  # baris test tidak boleh dibuang
            self.j.tambah_langkah("kode", l)

    # ---------- validasi submission (H1) ----------
    def validasi_submisi(self):
        j = self.j
        f, _ = QFileDialog.getOpenFileName(self, "File submission yang akan dikumpulkan", str(j.folder_kerja),
                                           "CSV (*.csv)")
        if not f:
            return
        contoh = next((e["path"] for n, e in j.entri.items() if n.lower().endswith(".csv")
                       and ("sample" in n.lower() or "submission" in n.lower()) and Path(e["path"]) != Path(f)), None)
        if not contoh:
            contoh, _ = QFileDialog.getOpenFileName(self, "sample_submission dari panitia",
                                                    (j.proyek or {}).get("folder", ""), "CSV (*.csv)")
            if not contoh:
                return
        df = j._df_tahap()
        lab = self._label(df) if df is not None else None
        kelas = df[lab].drop_nulls().unique().to_list() if lab else None

        def kerja():
            return submisi.periksa(submisi.baca(f), submisi.baca(contoh), kelas)

        ket = f"{Path(f).name} vs {Path(contoh).name}" + (f" · kelas dari {j.aktif}.{lab}" if lab else "")
        j.jalankan("Memeriksa submission", kerja, lambda cek: self._antre_selesai(
            {"jenis": "submisi", "folder": None, "ket": ket, "tabel": Path(f).name, "id": None, "status": "ok",
             "log": [], "jam": time.strftime("%H.%M"), "hasil": {"cek": cek}}))

    def _antre_selesai(self, r):
        self.riwayat.append(r)
        self.j.tab.setCurrentIndex(3)
        self._gambar(len(self.riwayat) - 1)
