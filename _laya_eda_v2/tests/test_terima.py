"""Uji terima di data riil (VISI.md bagian 6). Semua uji DILEWATI bila `../_laya_eda/data/train.csv` tidak ada.

Angka patokan (875, 101, 4.867, ...) dari tabel audit RENCANA.md bagian 1. Butir yang belum punya pemeriksaan
otomatis di aplikasi ditandai 'belum diotomasi' (dilewati), bukan dianggap lulus.
"""
from __future__ import annotations

import pandas as pd

from tests import bantu

from core import audit, text_tools  # noqa: E402

LABEL = bantu.LABEL


def real() -> pd.DataFrame:
    bantu.perlu_data_riil()
    return pd.read_csv(bantu.TRAIN_RIIL)


def test_ukuran_data_riil():
    data = real()
    assert data.shape == (30380, 28) and LABEL in data


def test_penjaga_2147483647_dan_varian():
    data = real()
    found = set(audit.sentinels(data)["kolom"])
    assert len(found) >= 6, found


def test_riwayat_beasiswa_kipk_dan_strip():
    data = real()
    assert text_tools.key_of("kipk") == text_tools.key_of("KIPK")
    history = text_tools.merge_proposals(data["Riwayat Beasiswa atau Bantuan Pendidikan"]).set_index("kunci")
    assert history.loc["", "jadi"].lower().startswith("tidak"), "'-' = jawaban tidak, bukan kosong"


def test_ejaan_kota_dan_jenjang():
    real()
    key = text_tools.key_of
    assert key("Kota Semarang ") == key("Kota Semarang") and key("Kab. Pati") == key("Kabupaten Pati")
    assert key("S-3") == key("S3 ") == key("S- 3")


def test_program_studi_tidak_dipecah():
    data = real()
    assert not text_tools.is_free_text(data["Program Studi"])


def test_kolom_kembar():
    data = real()
    twins = audit.twin_columns(data.drop(columns=["id"]))
    pairs = {frozenset(row[:2]) for row in twins.itertuples(index=False)}
    assert frozenset({"Lokasi Tempat Tinggal", "Kota"}) in pairs
    assert frozenset({"Kepemilikan Aset", "Status Kepemilikan Tempat Tinggal"}) in pairs


def test_logika_antar_kolom():
    data = real()
    assert len(audit.check_rule(data, "`Luas Bangunan Rumah (m2)` > `Luas Tanah Rumah (m2)`")) == 875
    assert len(audit.check_rule(data, "`Cicilan Hutang per Bulan` > `Total Hutang`")) == 101


def test_satuan_campur():
    data = real()
    assert (data["Jumlah (Perkiraan Harga Pasar) Kendaraan yang Dimiliki"] < 100).sum() == 4867
    raise bantu.Lewati("belum diotomasi: aplikasi belum punya penanda 'satuan campur' (VISI bagian 3), "
                       "hanya angka patokan yang dicek")


def test_nol_mencurigakan():
    real()
    raise bantu.Lewati("belum diotomasi: usulan nol mencurigakan (Daya Listrik 0 di 281 baris) vs nol wajar")


def test_label_timpang_kelas_1():
    data = real()
    counts = data[LABEL].value_counts()
    assert counts.idxmin() == 1 and counts.min() == 125


if __name__ == "__main__":
    bantu.main(globals())
