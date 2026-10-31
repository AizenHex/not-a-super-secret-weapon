"""Uji modul analisis di data riil terhadap kasus penerimaan RENCANA.md bagian D.

Jalankan dari folder _laya_eda_v2:  python -m tests.test_tools
"""
from __future__ import annotations

import pandas as pd

from core import audit, outlier_tools, text_tools
from tests.test_steps import LABEL, TRAIN as DATA

SENTINEL_COLUMNS = ["Pengeluaran Bulanan Keluarga", "Jumlah (Perkiraan Harga Pasar) Kendaraan yang Dimiliki",
                    "Luas Tanah Rumah (m2)", "NJOP per Meter", "Total Hutang", "Cicilan Hutang per Bulan"]


def main() -> None:
    data = pd.read_csv(DATA)

    profile = audit.column_profile(data).set_index("kolom")["tipe tebakan"]
    print(profile.to_string())
    assert profile["Bahan Atap Rumah"] == "isian bebas"
    assert profile["Pendidikan Orang Tua/Wali"] == "kategori"
    assert profile["Luas Tanah Rumah (m2)"] == "angka"

    found = set(audit.sentinels(data)["kolom"])
    assert found == set(SENTINEL_COLUMNS) | {"Rerata Pengeluaran Listrik & Air per Bulan"}, found
    assert "Riwayat Beasiswa atau Bantuan Pendidikan" in set(audit.fake_missing(data)["kolom"])
    single = audit.single_value_columns(data)
    assert "Status Kepesertaan Program Bantuan Pemerintah" in set(single["kolom"])
    twins = audit.twin_columns(data.drop(columns=["id"]))
    print(twins.to_string())
    assert {"Lokasi Tempat Tinggal", "Kota"} <= set(twins.iloc[0, :2])
    print(audit.row_findings(data, LABEL))

    expectations = {
        "Bahan Atap Rumah": {"gentng": "genting", "asbas": "asbes"},
        "Bahan Tembok Rumah": {"bta": "bata"},
        "Bahan Lantai Rumah": {"kermaik": "keramik"},
    }
    materials = list(expectations)
    # 'bta' tidak ada di data riil: sisipkan beberapa sel buatan untuk menguji kasusnya
    data.loc[data.index[:3], "Bahan Tembok Rumah"] = ["bta", "Bta merah", "semen dan bta"]
    for column, pairs in expectations.items():
        related = [data[other] for other in materials if other != column]
        table = text_tools.word_map(data[column], related=related).set_index("kata asli")["usulan"]
        for word, target in pairs.items():
            assert table.get(word) == target, (column, word, table.get(word))
        assert "batu" not in table.index, "batu kata sah (Bahan Tembok), bukan salah ketik bata"
        for word in text_tools.STOPWORDS:
            assert word not in table.index and word not in set(table), (column, word)
        print(f"{column}: {len(table)} kata dipetakan, contoh {dict(list(table.items())[:6])}")
    print(text_tools.similar_vocabulary(data["Bahan Atap Rumah"]).head(8).to_string())

    groups = text_tools.fingerprint_groups(data["Kota"])
    print(groups.head(10).to_string())
    assert "Kota Semarang " in set(groups["asli"]) or "Kota Semarang" in set(groups["asli"])
    assert text_tools.fingerprint("Kab. Pati") == text_tools.fingerprint("Kabupaten Pati")
    assert text_tools.fingerprint("Kab. Semarang") != text_tools.fingerprint("Kota Semarang")
    assert text_tools.fingerprint("S- 3") == text_tools.fingerprint("S-3")

    words = dict(text_tools.word_map(data["Bahan Atap Rumah"])[["kata asli", "usulan"]].to_numpy())
    print(text_tools.material_preview(data["Bahan Atap Rumah"], words, top=8).to_string())

    for column in ["Luas Tanah Rumah (m2)", "NJOP per Meter", "Jumlah Tanggungan dalam Keluarga"]:
        clean = data[column].where(data[column] < 2e9)
        print(f"\n{column}\n", outlier_tools.compare_methods(clean).to_string())
    near = text_tools.neighbor_groups(data["Kota"])
    assert (near["peringatan"] != "").any(), "nearest neighbor harus memberi peringatan (Malang ≠ Magelang)"
    flags = outlier_tools.isolation_flags(data, ["Luas Tanah Rumah (m2)", "NJOP per Meter"])
    print("IsolationForest menandai:", int(flags.sum()))
    print("SEMUA UJI ALAT LULUS")


if __name__ == "__main__":
    main()
