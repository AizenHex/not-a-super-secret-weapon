"""Validator submission (H1): bandingkan file submission dengan sample_submission panitia sebelum dikumpulkan.
Keduanya dibaca sebagai teks apa adanya, supaya `3.0` vs `3` dan spasi tersembunyi kelihatan."""
import polars as pl

BULAT = r"^-?\d+$"
DESIMAL = r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$"


def baca(path):
    return pl.read_csv(path, infer_schema=False, encoding="utf8-lossy")


def periksa(sub, contoh, kelas=None):
    """Kembalikan daftar (tingkat, pesan); tingkat 'galat' = jangan dikumpulkan, 'peringatan' = cek lagi, 'ok'."""
    out = []
    galat = lambda p: out.append(("galat", p))  # noqa: E731
    if sub.columns != contoh.columns:
        kurang = [c for c in contoh.columns if c not in sub.columns]
        lebih = [c for c in sub.columns if c not in contoh.columns]
        p = f"Kolom harus persis {contoh.columns}, file ini {sub.columns}."
        if kurang:
            p += f" Kurang: {kurang}."
        if lebih:
            p += f" Lebih: {lebih}" + (" (indeks pandas ikut tersimpan? pakai index=False)."
                                      if any(c in ("", "Unnamed: 0", "index") for c in lebih) else ".")
        if not kurang and not lebih:
            p += " Urutan kolom berbeda."
        galat(p)
    if sub.height != contoh.height:
        galat(f"Jumlah baris {sub.height:,} padahal sample {contoh.height:,}.")
    idk = contoh.columns[0]
    if idk in sub.columns:
        a, b = sub[idk], contoh[idk]
        if a.n_unique() != a.len():
            galat(f"Kolom {idk} punya {a.len() - a.n_unique():,} nilai kembar.")
        hilang, asing = b.filter(~b.is_in(a.implode())), a.filter(~a.is_in(b.implode()))
        if hilang.len() or asing.len():
            galat(f"{idk}: {hilang.len():,} id sample tidak ada di submission, {asing.len():,} id submission tidak ada di "
                  f"sample (contoh: {(hilang.head(3).to_list() + asing.head(3).to_list())}).")
        elif a.len() == b.len() and not (a == b).all():
            out.append(("peringatan", f"Urutan {idk} berbeda dari sample. Aman bila dinilai lewat id; lebih aman "
                                      "disamakan."))
    for c in [c for c in contoh.columns[1:] if c in sub.columns]:
        s, ref = sub[c], contoh[c].drop_nulls()
        n_kosong = s.is_null().sum() + (s.str.strip_chars() == "").sum()
        if n_kosong:
            galat(f"{c}: {n_kosong:,} sel kosong/NaN.")
        isi = s.drop_nulls()
        if isi.len() and (isi != isi.str.strip_chars()).any():
            galat(f"{c}: ada nilai dengan spasi di awal/akhir (mis. {isi.filter(isi != isi.str.strip_chars())[0]!r}).")
        if ref.len() and ref.str.contains(BULAT).all():
            salah = isi.filter(~isi.str.contains(BULAT))
            if salah.len():
                galat(f"{c}: sample berisi bilangan bulat, {salah.len():,} nilai bukan (mis. {salah[0]!r}; `3.0` harus "
                      "`3`: ubah tipe ke int sebelum to_csv).")
        elif ref.len() and ref.str.contains(DESIMAL).all():
            salah = isi.filter(~isi.str.contains(DESIMAL))
            if salah.len():
                galat(f"{c}: sample berisi angka, {salah.len():,} nilai bukan angka (mis. {salah[0]!r}).")
        elif kelas:
            asing = isi.filter(~isi.is_in(pl.Series([str(k) for k in kelas]).implode())).unique()
            if asing.len():
                galat(f"{c}: {asing.len()} nilai bukan kelas label train: {asing.head(5).to_list()} (kelas: "
                      f"{sorted(map(str, kelas))[:10]}).")
        if isi.len():
            sebar = isi.value_counts(sort=True).head(5)
            out.append(("ok", f"{c}: " + ", ".join(f"{v} {n / isi.len():.0%}" for v, n in sebar.iter_rows())))
    if not any(t == "galat" for t, _ in out):
        out.insert(0, ("ok", f"Lulus: {sub.height:,} baris, kolom {sub.columns} sama dengan sample."))
    return out
