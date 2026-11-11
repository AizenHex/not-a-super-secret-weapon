"""Katalog figur paper untuk referensi visual (lokal; gambar tidak di-commit, repo publik).

python katalog.py unduh   -> arxiv/<id>/ : figur + caption dari arxiv.org/html atau ar5iv (butuh internet)
python katalog.py         -> katalog.md + galeri.html dari arxiv/ dan paperbench/ (offline)

paperbench/ = `hf download josancamon/paperbench --repo-type=dataset --local-dir paperbench`
"""
import html, json, os, re, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
# id arXiv -> kelompok. Kelompok dipakai sebagai filter di galeri.
PAPER = {
    # EDA / dokumentasi data teks
    "1803.02324": "eda", "2104.08758": "eda", "2101.00027": "eda", "2306.01116": "eda", "2406.17557": "eda",
    "2402.00159": "eda", "2310.20707": "eda", "1908.07898": "eda", "2009.10795": "eda", "1812.05159": "eda",
    "2103.14749": "eda", "1911.00068": "eda", "2004.07780": "eda",
    # training / dinamika / hiperparameter
    "2002.06305": "training", "2006.04884": "training", "2006.05987": "training", "1905.05583": "training",
    "1712.09913": "training", "2203.15556": "training", "2001.08361": "training", "1803.09820": "training",
    "1912.02292": "training", "1611.03530": "training", "1810.04805": "training",
    # evaluasi / kalibrasi / analisis galat
    "1706.04599": "evaluasi", "2003.07892": "evaluasi", "1906.02530": "evaluasi", "2207.05221": "evaluasi",
    "2004.02709": "evaluasi", "2005.04118": "evaluasi", "2104.14337": "evaluasi", "2211.09110": "evaluasi",
    "2206.04615": "evaluasi", "2504.01848": "evaluasi",
    # interpretasi / embedding
    "1602.04938": "interpretasi", "1705.07874": "interpretasi", "1703.01365": "interpretasi",
    "1906.04341": "interpretasi", "1906.05714": "interpretasi", "2002.12327": "interpretasi", "1802.03426": "interpretasi",
    # topic modeling
    "2203.05794": "topik", "2004.03974": "topik", "2008.09470": "topik", "2107.02173": "topik", "2004.14914": "topik",
    "1703.01488": "topik", "1907.04907": "topik",
    # Indonesia
    "2009.05387": "indonesia", "2205.15960": "indonesia", "2011.00677": "indonesia", "2212.09648": "indonesia",
    "2309.10661": "indonesia",
}
MAKS_GAMBAR = 20
# <title> HTML kadang rusak
JUDUL = {"2504.01848": "PaperBench: Evaluating AI's Ability to Replicate AI Research",
         "1611.03530": "Understanding Deep Learning Requires Rethinking Generalization"}


def ambil(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (katalog-figur)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def teks(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def unduh():
    for pid in PAPER:
        out = ROOT / "arxiv" / pid
        if (out / "meta.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        figs, judul = [], pid
        for base in (f"https://arxiv.org/html/{pid}", f"https://ar5iv.labs.arxiv.org/html/{pid}"):
            try:
                page = ambil(base).decode("utf-8", "replace")
            except Exception as e:
                print(pid, base, "gagal:", e)
                continue
            m = re.search(r"<title>(.*?)</title>", page, re.S)
            judul = teks(m.group(1)).replace(f"[{pid}] ", "") if m else pid
            # versi html arxiv memakai <base href="/html/<id>vN/">
            b = re.search(r'<base href="([^"]+)"', page)
            # tanpa <base>, src arxiv.org berbentuk "<id>vN/x.png" (relatif ke /html/)
            dasar = "https://arxiv.org" + b.group(1) if b else base.rsplit("/", 1)[0] + "/"
            figs = []
            for blok in re.findall(r"<figure\b.*?</figure>", page, re.S):
                srcs = [s for s in re.findall(r'<img[^>]+src="([^"]+)"', blok)
                        if re.search(r"\.(png|jpe?g|gif)$", s, re.I) and "logo" not in s]
                cap = re.search(r"<figcaption.*?</figcaption>", blok, re.S)
                if srcs and cap:
                    figs.append((srcs, teks(cap.group(0))))
            if figs:
                break
        meta = {"id": pid, "judul": judul, "kelompok": PAPER[pid], "figur": []}
        n = 0
        for srcs, cap in figs:
            for s in srcs:
                if n >= MAKS_GAMBAR:
                    break
                url = s if s.startswith("http") else ("https://ar5iv.labs.arxiv.org" + s if s.startswith("/")
                                                      else dasar + s)
                nama = f"{n:02d}_{Path(s).name}"
                try:
                    (out / nama).write_bytes(ambil(url))
                except Exception as e:
                    print(pid, url, "gagal:", e)
                    continue
                meta["figur"].append({"file": nama, "caption": cap[:600]})
                n += 1
        if not meta["figur"]:
            meta["figur"] = dari_pdf(pid, out)
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(pid, len(meta["figur"]), "figur |", judul[:70])


def dari_pdf(pid, out):
    """Figur vektor tidak ada di HTML: render halaman PDF yang memuat caption 'Figure N'."""
    import fitz  # pymupdf
    try:
        doc = fitz.open(stream=ambil(f"https://arxiv.org/pdf/{pid}"), filetype="pdf")
    except Exception as e:
        print(pid, "pdf gagal:", e)
        return []
    figur = []
    for i, hal in enumerate(doc):
        caps = re.findall(r"^(Fig(?:ure|\.)\s*\d+[.:].{0,300})", hal.get_text(), re.M)
        if not caps or len(figur) >= MAKS_GAMBAR // 2:
            continue
        nama = f"hal{i + 1:02d}.png"
        hal.get_pixmap(dpi=110).save(out / nama)
        figur.append({"file": nama, "caption": "[halaman PDF] " + " | ".join(c.strip() for c in caps)[:600]})
    return figur


def baca_paperbench():
    for d in sorted((ROOT / "paperbench").glob("*/paper.md")):
        md = d.read_text(encoding="utf-8", errors="replace").splitlines()
        cfg = (d.parent / "config.yaml").read_text(encoding="utf-8", errors="replace")
        m = re.search(r'^title:\s*"?(.*?)"?\s*$', cfg, re.M)
        judul = m.group(1) if m else d.parent.name
        figur = []
        for i, l in enumerate(md):
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", l.strip())
            if not m:
                continue
            cap = next((x.strip() for x in md[i + 1:i + 4] if re.match(r"\s*(Figure|Fig\.|Table)", x)), "")
            figur.append({"file": m.group(1), "caption": cap[:600]})
        yield {"id": "pb/" + d.parent.name, "judul": judul, "kelompok": "paperbench",
               "folder": f"paperbench/{d.parent.name}", "figur": figur}


def baca_arxiv():
    for f in sorted((ROOT / "arxiv").glob("*/meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        m["folder"] = f"arxiv/{m['id']}"
        m["judul"] = JUDUL.get(m["id"], m["judul"])
        yield m


def bangun():
    papers = list(baca_arxiv()) + list(baca_paperbench())
    assert papers, "belum ada data: jalankan `python katalog.py unduh` dan/atau hf download paperbench"
    md = ["# Katalog figur (dibuat otomatis oleh katalog.py, jangan diedit)", "",
          f"{sum(len(p['figur']) for p in papers)} figur dari {len(papers)} paper. Path relatif ke folder ini.", ""]
    kartu = []
    import hashlib
    for p in papers:  # buang gambar kembar (mis. ikon yang diulang di tiap figur)
        seen, unik = set(), []
        for f in p["figur"]:
            path = ROOT / p["folder"] / f["file"]
            h = hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None
            if h and h not in seen:
                seen.add(h)
                unik.append(f)
        p["figur"] = unik
    for p in papers:
        url = f"https://arxiv.org/abs/{p['id']}" if not p["id"].startswith("pb/") else ""
        md += [f"## [{p['kelompok']}] {p['judul']}", f"`{p['folder']}` {url}", ""]
        for f in p["figur"]:
            md.append(f"- `{p['folder']}/{f['file']}`: {f['caption'] or '(tanpa caption)'}")
            kartu.append(
                f'<figure data-k="{p["kelompok"]}" data-t="{html.escape((p["judul"] + " " + f["caption"]).lower())}">'
                f'<img loading="lazy" src="{p["folder"]}/{f["file"]}"><figcaption><b>{html.escape(p["judul"][:90])}</b>'
                f'<br>{html.escape(f["caption"][:300])}</figcaption></figure>')
        md.append("")
    (ROOT / "katalog.md").write_text("\n".join(md), encoding="utf-8")
    kel = sorted({p["kelompok"] for p in papers})
    tombol = "".join(f'<button onclick="k(\'{x}\')">{x}</button>' for x in ["semua"] + kel)
    (ROOT / "galeri.html").write_text(f"""<!doctype html><html lang="id"><meta charset="utf-8"><title>Galeri Figur</title>
<style>body{{font:14px/1.4 system-ui,sans-serif;margin:0;padding:16px;background:#f6f6f4;color:#222}}
#bar{{position:sticky;top:0;background:#f6f6f4;padding:8px 0;z-index:1}}input{{font-size:15px;padding:6px;width:280px}}
button{{margin:2px;padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}}
#g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}}
img{{width:100%;max-height:300px;object-fit:contain;cursor:zoom-in}}img.besar{{position:fixed;inset:2vh 2vw;width:96vw;
height:96vh;max-height:none;background:#fffe;z-index:9;cursor:zoom-out;border:1px solid #888}}
figcaption{{font-size:12px;color:#444}}</style>
<div id="bar"><input id="q" placeholder="cari: calibration, heatmap, topic, loss..." oninput="f()"> {tombol} <span id="n"></span></div>
<div id="g">{''.join(kartu)}</div>
<script>let K='semua';const F=[...document.querySelectorAll('figure')];
function k(x){{K=x;f()}}function f(){{const q=document.getElementById('q').value.toLowerCase();let n=0;
for(const e of F){{const ok=(K==='semua'||e.dataset.k===K)&&e.dataset.t.includes(q);e.style.display=ok?'':'none';n+=ok}}
document.getElementById('n').textContent=n+' figur'}}
document.addEventListener('click',e=>{{if(e.target.tagName==='IMG')e.target.classList.toggle('besar')}});f()</script></html>""",
                                      encoding="utf-8")
    print("katalog.md + galeri.html:", sum(len(p["figur"]) for p in papers), "figur,", len(papers), "paper")


if __name__ == "__main__":
    unduh() if sys.argv[1:] == ["unduh"] else bangun()
