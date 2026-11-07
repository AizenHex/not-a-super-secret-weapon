"""Uji mesin Laya v3 tanpa layar.

    python uji.py                      # korpus file jahat (VISI.md 6.3) + proyek
    python uji.py <folder> [<folder>]  # + baca semua tabel di folder data nyata, cetak ukuran dan waktu

Tiap file korpus harus terbaca dengan bentuk yang benar, atau gagal dengan pesan + baris mentah. Tidak boleh crash.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesin import bantu, ekspor, langkah, muat, proyek, resep, temuan  # noqa: E402


def buat_korpus(d):
    d = Path(d)

    def tulis(nama, isi, enc="utf-8"):
        p = d / nama
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(isi if isinstance(isi, bytes) else isi.encode(enc))

    tulis("koma.csv", "id,nama,nilai\n1,Ani,3.5\n2,Budi,4\n")
    tulis("titikkoma.csv", "id;harga;kota\n1;1.234,5;Bandung\n2;20.000;Jakarta\n")
    tulis("tab.tsv", "a\tb\n1\tx\n2\ty\n")
    tulis("pipa.txt", "a|b|c\n1|2|3\n4|5|6\n")
    tulis("bom.csv", "\ufeffid,teks\n1,halo\n")
    tulis("utf16.csv", "id,teks\n1,héllo\n", "utf-16")
    tulis("cp1252.csv", "id,teks\n1,café © 2021\n", "cp1252")
    tulis("multiline.csv", 'id,teks,label\n1,"baris satu\nbaris dua, pakai koma",pos\n2,"kutip ""dalam"" sel",neg\n')
    tulis("tak_rata.csv", "a,b,c\n1,2,3\n4,5\n6,7,8,9\n")
    tulis("kosong.csv", "")
    tulis("header_saja.csv", "a,b,c\n")
    tulis("kolom_ganda.csv", "a,a,,b\n1,2,3,4\n")
    tulis("tipe_campur.csv", "id,kode,x\n" + "".join(f"{i},{i:05d},{i}\n" for i in range(15000)) + "15000,00001,abc\n")
    tulis("sel_besar.csv", "id,teks\n1," + "x" * 100_000 + "\n2,pendek\n")
    tulis("emoji.csv", "id,teks\n1,mantap 😀🔥\n2,#viral @budi\n")
    tulis("penanda_kosong.csv", "a,b\n1,-\n2,NA\n3,null\n4,\n")
    tulis("dokumen.txt", "kalimat pertama tanpa pemisah\nkalimat kedua\n")
    tulis("rusak.parquet", b"bukan parquet sama sekali")
    tulis("sub/dalam/inner.csv", "a\n1\n")
    tulis("bersarang.json", json.dumps([{"id": 1, "tokoh": {"nama": "A", "jabatan": ["x", "y"]}}, {"id": 2, "tokoh": None}]))
    tulis("objek.json", json.dumps({"meta": {"n": 2}, "data": [{"a": 1}, {"a": 2}]}))
    tulis("baris.jsonl", '{"a": 1, "b": "x"}\n{"a": "dua", "b": {"c": 1}}\nbukan json\n')
    tulis("besar_id.csv", "id\n12345678901234567890123\n1\n")
    tulis("rusak_akhir.csv", ("id,teks\n" + "1,aman\n" * 200_000).encode() + b"2,caf\xe9\n")  # byte rusak setelah 1 MB
    pl.DataFrame({"x": [1, 2], "y": ["a", None]}).write_parquet(d / "baik.parquet")
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Satu"
    ws.append(["id", "nilai", "id", None])
    ws.append([1, 2.5, "a", "x"])
    ws.append([2, "teks", None, None])
    wb.create_sheet("Dua").append(["x"])
    wb["Dua"].append([1])
    wb.save(d / "buku.xlsx")


# nama -> (baris, kolom) atau "gagal"
HARAP = {
    "koma.csv": (2, 3), "titikkoma.csv": (2, 3), "tab.tsv": (2, 2), "pipa.txt": (2, 3), "bom.csv": (1, 2),
    "utf16.csv": (1, 2), "cp1252.csv": (1, 2), "multiline.csv": (2, 3), "tak_rata.csv": (3, 3), "kosong.csv": (0, 0),
    "header_saja.csv": (0, 3), "kolom_ganda.csv": (1, 4), "tipe_campur.csv": (15001, 3), "sel_besar.csv": (2, 2),
    "emoji.csv": (2, 2), "penanda_kosong.csv": (4, 2), "dokumen.txt": (2, 1), "rusak.parquet": "gagal",
    "sub/dalam/inner.csv": (1, 1), "bersarang.json": (2, 2), "objek.json": (2, 1), "baris.jsonl": (2, 2),
    "besar_id.csv": (2, 1), "baik.parquet": (2, 2), "buku.xlsx:Satu": (2, 4), "buku.xlsx:Dua": (1, 1),
    "rusak_akhir.csv": (200_001, 2),
}


def uji_korpus():
    with tempfile.TemporaryDirectory() as d:
        buat_korpus(d)
        entri = {e["nama"]: e for e in muat.pindai(d)}
        assert set(entri) == set(HARAP), f"pindai: lebih {set(entri) - set(HARAP)}, kurang {set(HARAP) - set(entri)}"
        hasil = {}
        for nama, harap in HARAP.items():
            e = entri[nama]
            try:
                df, lap = muat.baca(e["path"], e["sheet"])
            except muat.BacaGagal as g:
                assert harap == "gagal", f"{nama}: gagal tak terduga: {g}"
                assert g.mentah, f"{nama}: gagal tanpa baris mentah"
                continue
            assert harap != "gagal", f"{nama}: harusnya gagal"
            assert df.shape == harap, f"{nama}: bentuk {df.shape}, harap {harap}; catatan {lap['catatan']}"
            assert len(set(df.columns)) == df.width
            hasil[nama] = (df, lap)

        def kol(nama, c):
            return hasil[nama][0][c]

        assert kol("koma.csv", "id").dtype == pl.Int64 and kol("koma.csv", "nilai").dtype == pl.Float64
        assert kol("titikkoma.csv", "harga").to_list() == ["1.234,5", "20.000"]  # format Indonesia tidak ditebak
        assert hasil["bom.csv"][0].columns == ["id", "teks"]
        assert kol("utf16.csv", "teks")[0] == "héllo"
        assert kol("cp1252.csv", "teks")[0] == "café © 2021"
        assert kol("multiline.csv", "teks").to_list() == ["baris satu\nbaris dua, pakai koma", 'kutip "dalam" sel']
        assert kol("tipe_campur.csv", "id").dtype == pl.Int64
        assert kol("tipe_campur.csv", "kode").dtype == pl.String and kol("tipe_campur.csv", "kode")[0] == "00000"
        assert kol("tipe_campur.csv", "x").dtype == pl.String
        assert any("nol-depan" in c for c in hasil["tipe_campur.csv"][1]["catatan"])
        assert len(kol("sel_besar.csv", "teks")[0]) == 100_000
        assert kol("emoji.csv", "teks")[0] == "mantap 😀🔥"
        assert kol("besar_id.csv", "id").dtype == pl.String  # > Int64: jangan jadi float yang membulatkan
        assert hasil["kolom_ganda.csv"][0].columns == ["a", "a_duplicated_0", "kolom_3", "b"]  # nama ganda: aturan polars
        assert hasil["buku.xlsx:Satu"][0].columns == ["id", "nilai", "id_2", "kolom_4"]
        assert kol("buku.xlsx:Satu", "nilai").to_list() == ["2.5", "teks"]
        assert any("bukan JSON" in c for c in hasil["baris.jsonl"][1]["catatan"])
        assert any("bukan utf-8" in c for c in hasil["rusak_akhir.csv"][1]["catatan"])
        print(f"korpus: {len(HARAP)} file lulus")


def uji_proyek():
    with tempfile.TemporaryDirectory() as d:
        data, kerja = Path(d) / "data", Path(d) / "kerja"
        data.mkdir()
        p = proyek.baru(data)
        p["opsi_baca"]["a.csv"] = {"pemisah": ";"}
        path = proyek.path_untuk(data, kerja)
        proyek.simpan(p, path)
        assert proyek.muat(path)["opsi_baca"] == {"a.csv": {"pemisah": ";"}}
        # dipindah ke PC lain: path absolut tidak ada lagi, relatif masih cocok
        isi = json.loads(path.read_text(encoding="utf-8"))
        isi["folder"] = "Q:/tidak/ada"
        path.write_text(json.dumps(isi), encoding="utf-8")
        q = proyek.muat(path)
        assert proyek.folder_ada(q) and Path(q["folder"]) == data.resolve()
        print("proyek: lulus")


MATRIKS_DF = """df = pl.DataFrame({
    "bulat": [1, 2, None, 2], "pecahan": [1.5, None, 2.5, 1.5], "teks": ["a b", "-", None, "aaa  B"],
    "nol": pl.Series([None] * 4, dtype=pl.String), "daftar": [["x"], [], None, ["y", "z"]],
    "struk": [{"p": 1}, {"p": 2}, None, {"p": 1}],
    "tgl": pl.Series(["2024-01-02", None, "2024-03-04", "2024-01-02"]).str.to_date(),
    "benar": [True, False, None, True], "daftar_teks": ["['x', 'y']", "[]", None, "bukan list"]})"""


def _param_matriks(c):
    """Semua jenis langkah x varian isian untuk satu kolom c."""
    yield "regex_ganti", {"kolom": [c], "pola": "a", "ganti": "b"}
    yield "regex_ganti", {"kolom": [c], "pola": r"(\w)\1{2,}", "ganti": r"\1"}
    yield "regex_ganti", {"kolom": [c], "pola": r"(\w)\1{2,}x", "ganti": r"\1"}  # jalur mesin Python
    yield "regex_ambil", {"kolom": c, "pola": r"(\w)", "grup": 1, "baru": "x"}
    yield "rapikan_teks", {"kolom": [c], "nfkc": True, "kecil": True, "spasi": True}
    for t in langkah.TIPE:
        yield "ubah_tipe", {"kolom": [c], "tipe": t}
    yield "jadi_kosong", {"kolom": [c], "nilai": langkah.NIL_BAWAAN}
    for cara in ["median", "rata-rata", "modus", "nilai tetap", "isi dari atas", "isi dari bawah"]:
        yield "isi_kosong", {"kolom": [c], "cara": cara, "nilai": "0"}
    yield "ganti_nilai", {"kolom": [c], "peta": "a => b\n1 => 9  # angka"}
    yield "ganti_nama", {"peta": f"{c} => baru"}
    yield "buang_kolom", {"kolom": [c]}
    for cara in ["cocok regex", "tidak cocok regex", "kosong", "tidak kosong"]:
        yield "saring_baris", {"cara": cara, "kolom": c, "isi": "a"}
    yield "saring_baris", {"cara": "ekspresi SQL", "isi": f'"{c}" IS NOT NULL'}
    for s in ["pertama", "terakhir", "buang semua salinan"]:
        yield "buang_duplikat", {"kolom": [c], "simpan": s}
    yield "list_teks", {"kolom": [c]}
    yield "pecah", {"kolom": c, "cara": "list → baris"}
    yield "pecah", {"kolom": c, "cara": "struct → kolom"}
    yield "join", {"tabel": "lain", "kiri": [c], "kanan": "k", "cara": "left", "validasi": "m:1"}
    yield "tumpuk", {"tabel": ["lain"], "asal": True}
    yield "agregasi", {"kolom": [c], "isi": "pecahan: mean, max\nteks: n_unique"}
    yield "sql", {"isi": f'SELECT "{c}", COUNT(*) AS n FROM df GROUP BY "{c}"'}
    yield "kode", {"isi": f'df = df.with_columns(pl.col("{c}").alias("salin"))'}


def uji_langkah():
    """Tiap jenis langkah x tiap tipe kolom: jalan, atau gagal dengan pesan (tidak boleh crash)."""
    m = resep.Mesin({"folder": ".", "resep": {}, "turunan": {
        "m": MATRIKS_DF, "lain": 'df = pl.DataFrame({"k": ["a b", "x"], "isi": [1, 2]})'}}, {})
    jalan = gagal = 0
    pesan = {}
    for c in m.hasil("m").akhir.columns:
        for jenis, p in _param_matriks(c):
            m.proyek["resep"]["m"] = {"langkah": [langkah.buat(jenis, p)], "pasangan": []}
            h = m.hasil("m")
            if h.galat:
                gagal += 1
                assert h.galat[1], (jenis, c)
                pesan.setdefault(jenis, []).append(f"{c}: {h.galat[1].splitlines()[0][:110]}")
            else:
                jalan += 1
                assert isinstance(h.akhir, pl.DataFrame)
    rapi = {"nfkc": True, "kecil": True, "spasi": True}
    cek = [("jadi_kosong", {"kolom": ["teks"], "nilai": "-"}, lambda d: d["teks"].to_list() == ["a b", None, None, "aaa  B"]),
           ("rapikan_teks", {"kolom": ["teks"]} | rapi, lambda d: d["teks"].to_list() == ["a b", "-", None, "aaa b"]),
           ("regex_ganti", {"kolom": ["teks"], "pola": r"(\w)\1{2,}", "ganti": r"\1"}, lambda d: d["teks"][3] == "a  B"),
           ("regex_ganti", {"kolom": ["teks"], "pola": r"(\w) (\w)", "ganti": r"\2-\1$"}, lambda d: d["teks"][0] == "b-a$"),
           ("isi_kosong", {"kolom": ["pecahan"], "cara": "median"}, lambda d: d["pecahan"].to_list() == [1.5, 1.5, 2.5, 1.5]),
           ("isi_kosong", {"kolom": ["bulat"], "cara": "nilai tetap", "nilai": "0"}, lambda d: d["bulat"].to_list() == [1, 2, 0, 2]),
           ("buang_duplikat", {"kolom": ["bulat"], "simpan": "buang semua salinan"}, lambda d: d["bulat"].to_list() == [1, None]),
           ("list_teks", {"kolom": ["daftar_teks"]}, lambda d: d["daftar_teks"].to_list() == [["x", "y"], [], None, None]),
           ("ubah_tipe", {"kolom": ["daftar_teks"], "tipe": "angka"}, lambda d: d["daftar_teks"].null_count() == 4)]
    for jenis, p, f in cek:
        m.proyek["resep"]["m"] = {"langkah": [langkah.buat(jenis, p)], "pasangan": []}
        h = m.hasil("m")
        assert not h.galat and f(h.akhir), (jenis, p, h.galat, h.akhir)
    print(f"langkah: {jalan} jalan, {gagal} gagal dengan pesan, {len(cek)} cek nilai lulus")
    return pesan


def buat_data_resep(d):
    d = Path(d)
    (d / "train.csv").write_text(
        "id,teks,kota,angka,label\n"
        "1,Halo   DUNIA https://x.co,BDG,10,pos\n2,-,JKT,,neg\n3,Ｆｕｌｌ width,BDG,30,pos\n"
        "4,Halo   DUNIA https://x.co,BDG,10,pos\n5,keren bangettt,SBY,NA,neg\n", encoding="utf-8")
    (d / "test.csv").write_text("id,teks,kota,angka\n10,HALO lagi,JKT,\n11,-,MDN,5\n", encoding="utf-8")
    (d / "kota.csv").write_text("kota,pulau\nBDG,Jawa\nJKT,Jawa\nSBY,Jawa\nMDN,Sumatra\n", encoding="utf-8")


def proyek_resep(d):
    L = langkah.buat
    p = proyek.baru(d)
    p["resep"] = {"train.csv": {"pasangan": ["test.csv"], "langkah": [
        L("jadi_kosong", {"kolom": [], "nilai": "-\nNA"}),
        L("rapikan_teks", {"kolom": ["teks"], "nfkc": True, "kecil": True, "spasi": True}),
        L("regex_ganti", {"kolom": ["teks"], "pola": r"https?://\S+", "ganti": " "}),
        L("regex_ganti", {"kolom": ["teks"], "pola": r"(\w)\1{2,}", "ganti": r"\1"}),
        L("ubah_tipe", {"kolom": ["angka"], "tipe": "angka"}),
        L("isi_kosong", {"kolom": ["angka"], "cara": "median"}),
        L("buang_duplikat", {"kolom": ["teks"], "simpan": "pertama"}),
        L("join", {"tabel": "kota.csv", "kiri": ["kota"], "cara": "left", "validasi": "m:1"}),
        L("ganti_nilai", {"kolom": ["label"], "peta": "pos => 1\nneg => 0"}),
        L("kode", {"isi": 'df = df.with_columns(panjang=pl.col("teks").str.len_chars())'}),
    ]}}
    p["resep"]["train.csv"]["langkah"][8]["hanya_latih"] = True  # test tidak punya label
    p["turunan"] = {"gabung": 'df = pl.concat([T["train.csv"], T["test.csv"]], how="diagonal_relaxed")'}
    return p


def uji_ekspor():
    """Gerbang 6.5: skrip/notebook ekspor menghasilkan tabel yang sama persis dengan app."""
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        data, keluar = Path(d) / "data", Path(d) / "keluar"
        data.mkdir()
        buat_data_resep(data)
        p = proyek_resep(data)
        m = resep.Mesin(p, {e["nama"]: e for e in muat.pindai(data)})
        tr, te = m.hasil("train.csv"), m.hasil("test.csv")
        assert not tr.galat and not te.galat, (tr.galat, te.galat)
        assert tr.akhir.height == 4 and te.akhir.height == 2, (tr.akhir, te.akhir)
        assert te.akhir["angka"].to_list() == [10.0, 5.0], te.akhir  # median latih (10,30,10), bukan dari test
        assert "label" not in te.akhir.columns and te.akhir["pulau"].to_list() == ["Jawa", "Sumatra"]
        assert tr.akhir["teks"][3] == "keren banget", tr.akhir
        g = m.hasil("gabung")
        assert not g.galat and g.akhir.height == 6
        nama_app = {"train": "train.csv", "test": "test.csv"}
        for target, berkas in [("train.csv", ["train", "test"]), ("gabung", ["gabung"])]:
            for jenis in ("py", "ipynb"):
                out = keluar / f"{bantu.nama_aman(target)}_{jenis}"
                f = Path(d) / f"ekspor_{bantu.nama_aman(target)}.{jenis}"
                (ekspor.ekspor_py if jenis == "py" else ekspor.ekspor_ipynb)(m, target, f, out)
                if jenis == "ipynb":
                    nb = json.loads(f.read_text(encoding="utf-8"))
                    f = f.with_suffix(".dari_nb.py")
                    f.write_text("\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"),
                                 encoding="utf-8")
                r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, cwd=d)
                assert r.returncode == 0, r.stderr[-3000:]
                for b in berkas:
                    app, luar = m.hasil(nama_app.get(b, b)).akhir, pl.read_parquet(out / f"{b}.parquet")
                    assert app.equals(luar, null_equal=True), (target, jenis, b, app, luar)
        ekspor.simpan_tabel(tr.akhir, keluar / "e2", "train.csv")
        assert pl.read_parquet(keluar / "e2" / "train.parquet").equals(tr.akhir)
        # ubah langkah ke-3: tahap 0-2 dipakai ulang dari cache
        lama = list(tr.kunci)
        p["resep"]["train.csv"]["langkah"][2] = langkah.buat("regex_ganti", {"kolom": ["teks"], "pola": "x", "ganti": "y"})
        baru = m.hasil("train.csv").kunci
        assert baru[:3] == lama[:3] and baru[3] != lama[3]
        p["turunan"]["putar"] = 'df = T["putar"]'
        try:
            m.hasil("putar")
            raise AssertionError("acuan berputar lolos")
        except resep.TabelGagal as e:
            assert "berputar" in str(e)
        print("ekspor: .py dan .ipynb = hasil app (train, test, turunan); cache tahap + acuan berputar lulus")


def judul(ts):
    return "\n".join(t["judul"] for t in ts)


def uji_temuan():
    """M3: temuan muncul dari data kecil buatan; korpus tidak membuat cek gagal; cari/sampel/relasi jalan."""
    n = 200
    tr = pl.DataFrame({
        "id": [f"x{i:04d}" for i in range(n)],
        "teks": [f"Jakarta (ANTARA) - berita nomor {i % 150} tentang hal {i % 150 % 7} COPYRIGHT ¬© ANTARA 2021" for i in range(n)],
        "kota": ["Jakarta", "JAKARTA", "Bandung", "bandung "] * (n // 4),
        "nilai": [str(i) for i in range(n - 5)] + ["-", "NA", "abc", "?", "1"],
        "tanggal": [f"2021-01-{1 + i % 28:02d}" for i in range(n)],
        "tokoh": ["['Ani', 'Budi']", "[]"] * (n // 2),
        "petunjuk": ["pos", "neg"] * (n // 2),
        "label": ["pos"] * 150 + ["neg"] * 50,
    })
    te = tr.drop("label").head(40).vstack(tr.drop("label").tail(10).with_columns(kota=pl.lit("Medan")))
    lab = temuan.tebak_label(tr, te)
    assert lab == "label", lab
    ts = temuan.temuan(tr, lab)
    j = judul(ts)
    assert not [t for t in ts if t["grup"] == "cek gagal"], j
    for harus in ["teks: 50 salinan berlebih", "Mojibake", "List Python", "kota: 2 nilai ditulis beda-beda",
                  "Penanda kosong", "tanggal tersimpan sebagai teks", "Label label: pos 75%, neg 25%",
                  "nilai: tipe campur", "COPYRIGHT ¬© ANTARA 2021", "(ANTARA) -"]:
        assert harus in j, f"tidak ada temuan '{harus}':\n{j}"
    for t in ts:  # saring dan langkah usulan harus jalan
        if t["saring"]:
            tr.filter(temuan.evaluasi(t["saring"]))
        if t["langkah"]:
            m = resep.Mesin({}, {})
            exec(compile(langkah.buat(*t["langkah"])["kode"], "<u>", "exec"), m.glob, {"df": tr})
    jb = judul(temuan.banding(tr, te, lab))
    assert "50 dari 50 baris test" in jb and "kota: 1 nilai di test" in jb, jb
    h, per = temuan.cari(tr, temuan.ekspresi_cari("teks", temuan.SEMUA, "HAL 3", tr), lab)
    assert h.height == tr.filter(pl.col("teks").str.contains("hal 3 ")).height and per
    h, _ = temuan.cari(tr, temuan.ekspresi_cari("regex", "teks", r"(\d)\1", tr), lab)  # backreference → Python
    assert h.height and temuan.cari(tr, temuan.ekspresi_cari("SQL", None, "label = 'neg'", tr))[0].height == 50
    assert temuan.per_label(tr, lab, 7).height == 14 and temuan.sampel(tr, 30).height == 30
    p = temuan.profil(tr)
    assert p[0]["unik"] == n and p[1]["panjang"]["maks"] > 0 and len(p[1]["hist"]) == 10
    induk = pl.DataFrame({"kid": [1, 2, 3], "x": ["a", "b", "c"]})
    anak = pl.DataFrame({"kid": [1, 1, 2, 9], "y": [1, 2, 3, 4]})
    r = temuan.relasi({"anak": anak, "induk": induk})
    assert len(r) == 1 and r[0]["tabel"] == "anak" and r[0]["langkah"][1]["validasi"] == "m:1", r
    assert "25% nilai anak.kid tidak punya pasangan" in r[0]["rincian"], r[0]["rincian"]
    with tempfile.TemporaryDirectory() as d:
        buat_korpus(d)
        for e in muat.pindai(d):
            try:
                df = muat.baca(e["path"], e["sheet"])[0]
            except muat.BacaGagal:
                continue
            temuan.profil(df)
            gagal = [t for t in temuan.temuan(df, temuan.tebak_label(df)) if t["grup"] == "cek gagal"]
            assert not gagal, f"{e['nama']}: {gagal[0]['rincian']}"
    print("temuan: data buatan, banding, cari, sampel, relasi, korpus lulus")


def uji_emas():
    """Gerbang 6.4 pada data latihan nyata (dilewati bila folder tidak ada)."""
    g = Path(r"Z:\GEMASTIK\GDOC\data")
    if (g / "train.csv").exists():
        tr, te = muat.baca(str(g / "train.csv"))[0], muat.baca(str(g / "test.csv"))[0]
        lab = temuan.tebak_label(tr, te)
        j = judul(temuan.temuan(tr, lab)) + "\n" + judul(temuan.banding(tr, te, lab))
        for harus in ["5.451 salinan berlebih", "konten_berita: 11.806 salinan berlebih", "Mojibake",
                      "COPYRIGHT ¬© ANTARA 2021", "(ANTARA) -", "List Python tersimpan sebagai teks di 6 kolom: nama_tokoh",
                      "Positif 52%, Netral 32%, Negatif 16%", "sentimen_kutipan sama dengan label di 58%",
                      "konten_berita: 7.214 dari 22.665", "2021-01-01 s.d. 2021-06-30"]:
            assert harus in j, f"GDOC: tidak ada '{harus}'"
        print("emas GDOC lulus")
    b = Path(__file__).resolve().parents[2] / "data_latihan" / "multitabel" / "berka"
    if b.exists():
        T = {e["nama"]: muat.baca(e["path"], e["sheet"])[0] for e in muat.pindai(str(b))}
        r = [t for t in temuan.relasi(T) if t["judul"].startswith("account.csv ↔ loan.csv lewat account_id")]
        assert r and r[0]["tabel"] == "loan.csv" and r[0]["langkah"][1]["tabel"] == "account.csv", r
        print("emas Berka: loan → account lulus")


def ukur(folder):
    t = time.perf_counter()
    entri = muat.pindai(folder)
    print(f"\n{folder}: {len(entri)} tabel, pindai {time.perf_counter() - t:.2f} s")
    for e in entri:
        try:
            df, lap = muat.baca(e["path"], e["sheet"])
            print(f"  {e['nama']}: {df.height} x {df.width}, {lap['detik']} s {lap['catatan'] or ''}")
        except muat.BacaGagal as g:
            print(f"  {e['nama']}: GAGAL {g}")


def uji_submisi():
    """H1: lima submission rusak harus ditolak, yang benar lulus."""
    from mesin import submisi
    contoh = pl.DataFrame({"id": ["a", "b", "c", "d"], "label": ["0", "0", "0", "0"]})
    benar = pl.DataFrame({"id": ["a", "b", "c", "d"], "label": ["1", "0", "2", "1"]})
    rusak = {"3.0 bukan 3": benar.with_columns(pl.lit("1.0").alias("label")),
             "baris kurang": benar.head(3),
             "indeks ikut": benar.with_columns(pl.lit("0").alias("")).select("", "id", "label"),
             "NaN": benar.with_columns(pl.when(pl.col("id") == "b").then(None).otherwise("label").alias("label")),
             "id asing": benar.with_columns(pl.col("id").replace("d", "z"))}
    assert not [p for t, p in submisi.periksa(benar, contoh, [0, 1, 2]) if t == "galat"]
    for nama, df in rusak.items():
        g = [p for t, p in submisi.periksa(df, contoh, [0, 1, 2]) if t == "galat"]
        assert g, f"submission rusak ({nama}) lolos"
        print(f"  ditolak ({nama}): {g[0]}")
    teks = pl.DataFrame({"id": ["a", "b"], "label": ["pos", "neg"]})
    assert [t for t, _ in submisi.periksa(teks.with_columns(pl.lit("POS").alias("label")), teks, ["pos", "neg"])
            ].count("galat") == 1
    print("submisi: 5 rusak ditolak, yang benar lulus")


def jalankan_tugas(d, jenis, latih, uji=None, **kw):
    """Jalankan mesin/tugas.py di proses terpisah, persis cara Laya (G1)."""
    import subprocess
    keluar = Path(d) / jenis
    keluar.mkdir(exist_ok=True)
    t = {"jenis": jenis, "latih": str(latih), "uji": str(uji) if uji else None, "teks": ["teks"], "label": "label",
         "id": "id", "keluar": str(keluar), **kw}
    (keluar / "tugas.json").write_text(json.dumps(t), encoding="utf-8")
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "mesin" / "tugas.py"), str(keluar / "tugas.json")],
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads((keluar / "hasil.json").read_text(encoding="utf-8"))


def uji_tugas():
    """F1–F4 + T1 pada data buatan: sinyal kuat harus terbaca, label acak ≈ tebakan, test beda → AUC tinggi."""
    import random
    r = random.Random(42)
    isi = {"pos": "senang bagus mantap", "neg": "kecewa buruk rusak"}
    baris = [(i, lab, f"{r.choice(isi[lab].split())} {r.choice(['hari', 'ini', 'produk', 'toko'])} {i % 37}")
             for i, lab in ((i, r.choice(["pos", "neg"])) for i in range(400))]
    with tempfile.TemporaryDirectory() as d:
        df = pl.DataFrame(baris, schema=["id", "label", "teks"], orient="row")
        df = df.with_columns(pl.when(pl.col("id") < 6).then(pl.col("label").replace({"pos": "neg", "neg": "pos"}))
                             .otherwise("label").alias("label"))  # 6 label sengaja dibalik
        df.write_parquet(Path(d) / "latih.parquet")
        sama = df.drop("label").head(100)
        sama.write_parquet(Path(d) / "sama.parquet")
        beda = pl.DataFrame({"id": range(100), "teks": [f"cuaca dingin sekali {i}" for i in range(100)]})
        beda.write_parquet(Path(d) / "beda.parquet")
        b = jalankan_tugas(d, "baseline", Path(d) / "latih.parquet")
        dibalik = {c["baris"] for c in b["curiga"][:20]} & set(range(6))
        print(f"  baseline macro-F1 {b['macro_f1']:.3f}, label curiga {b['curiga_total']}, 6 dibalik tertangkap "
              f"{len(dibalik)}, kata pos {b['kata']['pos'][:3]}")
        assert b["macro_f1"] > 0.9 and len(dibalik) >= 4 and len(set(b["kata"]["pos"][:3]) & set(isi["pos"].split())) >= 2
        a = jalankan_tugas(d, "acak", Path(d) / "latih.parquet")
        print(f"  label acak macro-F1 {a['macro_f1']:.3f} (seragam {a['tebakan_seragam']:.3f})")
        assert a["macro_f1"] < 0.65
        k = jalankan_tugas(d, "kata", Path(d) / "latih.parquet")
        assert k["kata"]["neg"][0]["kata"] in isi["neg"].split(), k["kata"]["neg"][:3]
        v1 = jalankan_tugas(d, "adversarial", Path(d) / "latih.parquet", Path(d) / "sama.parquet")
        v2 = jalankan_tugas(d, "adversarial", Path(d) / "latih.parquet", Path(d) / "beda.parquet")
        print(f"  adversarial AUC test mirip {v1['auc']:.3f}, test beda {v2['auc']:.3f} ({v2['kata_uji'][:3]})")
        assert v1["auc"] < 0.7 and v2["auc"] > 0.95
        # tabular: label ditentukan kolom angka x (> 0) + kategori kota; holdout dan grup kolom
        tab = pl.DataFrame({"id": range(400), "x": [r.uniform(-1, 1) for _ in range(400)],
                            "kota": [r.choice("abcd") for _ in range(400)]})
        tab = tab.with_columns(label=pl.when((pl.col("x") > 0) | (pl.col("kota") == "a")).then(pl.lit("ya"))
                               .otherwise(pl.lit("tidak")))
        tab.write_parquet(Path(d) / "tab.parquet")
        tb = jalankan_tugas(d, "tabular", Path(d) / "tab.parquet", seed=42, fold=5)
        th = jalankan_tugas(d, "tabular", Path(d) / "tab.parquet", seed=42, fold=5,
                            validasi={"cara": "holdout", "porsi": 0.25})
        print(f"  tabular macro-F1 {tb['macro_f1']:.3f} ({tb['validasi']}), holdout {th['macro_f1']:.3f}, "
              f"penting {[p[0] for p in tb['penting']]}")
        assert tb["macro_f1"] > 0.9 and th["macro_f1"] > 0.85 and {p[0] for p in tb["penting"][:2]} == {"x", "kota"}
        from mesin import venv
        from tampilan import pahami
        for jenis, h in (("baseline", b), ("acak", a), ("kata", k), ("adversarial", v2), ("tabular", tb)):
            r = {"jenis": jenis, "ket": "uji", "status": "ok", "hasil": h}
            assert pahami.JUDUL[jenis] in pahami.html_hasil(r) and pahami.ringkas_hasil(r)
        assert Path(venv.cari_python(Path(d))).is_file() and venv.skrip_tugas().is_file()
    print("tugas: baseline, label curiga, label acak, kata khas, adversarial lulus")


if __name__ == "__main__":
    uji_submisi()
    uji_tugas()
    uji_korpus()
    uji_proyek()
    for j, ps in uji_langkah().items():
        print(f"  {j}:", *ps[:12], sep="\n    ")
    uji_ekspor()
    uji_temuan()
    uji_emas()
    for f in sys.argv[1:]:
        ukur(f)
