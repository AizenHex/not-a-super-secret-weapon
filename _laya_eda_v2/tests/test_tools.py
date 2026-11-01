"""Uji modul analisis di data riil terhadap kasus penerimaan RENCANA.md bagian D.

Jalankan dari folder _laya_eda_v2:  python -m tests.test_tools
"""
from __future__ import annotations

import pandas as pd

from core import audit, outlier_tools, steps, text_tools
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
        "Bahan Atap Rumah": {"gentng": "genteng", "asbas": "asbes"},   # istilah BPS: genteng
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

    # aturan kunci: beda huruf besar/spasi/tanda baca/singkatan = satu kunci, tanpa daftar per nilai
    key_of = text_tools.key_of
    assert key_of("S- 3") == key_of("S-3") == key_of("s3") == key_of(" S 3 ")
    assert key_of("Kab. Pati") == key_of("Kabupaten Pati") == key_of("KAB PATI")
    assert key_of("Kab. Semarang") != key_of("Kota Semarang")
    assert key_of("1-2") != key_of("12"), "angka terpisah tidak boleh menyatu (1-2 orang ≠ 12 orang)"
    assert (text_tools.keys(data["Kota"].dropna().head(200)) == data["Kota"].dropna().head(200).map(key_of)).all(), \
        "keys (pandas, = kode ekspor) harus sama dengan key_of"
    same = text_tools.same_key_table(data["Kota"])
    print(same.head(8).to_string())
    assert not same.empty

    education = "Pendidikan Orang Tua/Wali"
    proposals = text_tools.merge_proposals(data[education]).set_index("kunci")
    print(proposals.to_string())
    assert proposals.loc["sma", "gabung ke"] == proposals.loc["ma", "gabung ke"], "SMA = MA (jenjang sama)"
    assert "s2" not in proposals.index or proposals.loc["s2", "gabung ke"] not in ("s1", "s3"), "S1 ≠ S2 ≠ S3"
    assert proposals["setuju"].all(), "usulan kamus jenjang dicentang awal"
    groups, examples = text_tools.plan_groups(proposals.reset_index())
    assert groups and "" not in groups, "kunci kosong tidak pernah jadi baku"
    history = text_tools.merge_proposals(data["Riwayat Beasiswa atau Bantuan Pendidikan"]).set_index("kunci")
    assert history.loc["", "jadi"].lower().startswith("tidak"), history.loc[""]

    typos = text_tools.merge_proposals(data["Kota"], typos=True)
    risky = typos[typos["alasan"].str.contains("salah ketik")]
    print(risky.head(8).to_string())
    assert not risky["setuju"].any(), "salah ketik tidak pernah dicentang otomatis"
    assert risky["alasan"].str.contains("periksa|Malang").any(), "salah ketik harus memberi peringatan"

    order = text_tools.ordinal_order(data["Pendapatan Keluarga"])
    print(order)
    assert order and order["urutan"][0].lower().startswith("kurang"), order
    assert text_tools.is_free_text(data["Bahan Lantai Rumah"]) and not text_tools.is_free_text(data["Kota"])

    words = dict(text_tools.word_map(data["Bahan Atap Rumah"])[["kata asli", "usulan"]].to_numpy())
    print(text_tools.material_preview(data["Bahan Atap Rumah"], words, top=8).to_string())
    floor = text_tools.resolve_chains(dict(text_tools.word_map(data["Bahan Lantai Rumah"])[["kata asli", "usulan"]]
                                           .to_numpy()))
    assert floor.get("kramik") == "keramik", floor.get("kramik")
    assert not set(floor) & set(floor.values()), "peta kata tidak boleh berantai (a→b, b→c)"

    # frasa: kata pengisi + kamus KBBI + urutan kata; pratinjau = hasil kode langkah persis
    wall = "Bahan Tembok Rumah"
    table = text_tools.word_map(data[wall])
    words = text_tools.resolve_chains(dict(zip(table.loc[table["setuju"], "kata asli"], table.loc[table["setuju"], "usulan"])))
    fillers = text_tools.fillers_for(wall)
    assert "tembok" in fillers and "dari" in fillers
    phrases = text_tools.phrase_map(data[wall], words, fillers).set_index("frasa")
    print(phrases.head(12).to_string())
    assert phrases.loc["bata merah", "usulan"] == "batu bata" and phrases.loc["bata merah", "setuju"]
    contained = phrases[phrases["alasan"].str.startswith("memuat")]
    assert len(contained) and not contained["setuju"].any(), "'memuat bahan lain' tidak pernah dicentang otomatis"
    chosen = text_tools.resolve_chains(phrases.loc[phrases["setuju"], "usulan"].to_dict())
    params = {"column": wall, "words": words, "fillers": fillers, "phrases": chosen}
    preview = text_tools.material_preview(data[wall], words, 10, fillers, chosen).set_index("bahan")["baris"]
    result = steps.run([{"op": "free_text", "params": params, "on": True}], None, data[[wall]].copy(), None)
    produced = result["train"][wall].str.split(", ").explode().value_counts()
    assert (produced.reindex(preview.index) == preview).all(), "pratinjau harus sama dengan kode langkah"
    assert "ubin ubin" not in set(text_tools.phrases_of(pd.Series(["tegel ubin"]), {"tegel": "ubin"}))

    for column in ["Luas Tanah Rumah (m2)", "NJOP per Meter", "Jumlah Tanggungan dalam Keluarga"]:
        clean = data[column].where(data[column] < 2e9)
        print(f"\n{column}\n", outlier_tools.compare_methods(clean).to_string())
    flags = outlier_tools.isolation_flags(data, ["Luas Tanah Rumah (m2)", "NJOP per Meter"])
    print("IsolationForest menandai:", int(flags.sum()))
    print("SEMUA UJI ALAT LULUS")


if __name__ == "__main__":
    main()
