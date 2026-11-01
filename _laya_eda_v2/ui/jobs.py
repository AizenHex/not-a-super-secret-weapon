"""Tugas lama (AutoGluon, cleanlab, skrub, Laya, IsolationForest) dijalankan di THREAD terpisah.

Kenapa perlu: Streamlit menjalankan ulang SELURUH skrip setiap ada interaksi (pindah halaman, centang,
geser slider). Rerun baru menghentikan run yang sedang berjalan, jadi perhitungan yang dijalankan langsung
di skrip ikut mati dan hasilnya hilang. Thread di sini tidak terikat pada run skrip mana pun:

- tetap jalan walau halaman dipindah atau widget lain diklik;
- hasilnya disimpan di registri tingkat server (st.cache_resource), jadi juga bertahan bila tab browser
  di-refresh; hanya hilang bila server dimatikan;
- satu nama tugas = satu tugas: tombol yang sama tidak bisa memulai tugas kedua selama yang pertama jalan.

Fungsi yang dijalankan di thread TIDAK boleh memanggil st.* (tidak ada konteks halaman di sana):
ia hanya menghitung dan mengembalikan hasil. Halaman membaca hasilnya lewat result().
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field

import streamlit as st


@dataclass
class Job:
    name: str
    title: str
    note: str = ""                      # untuk apa hasil ini dihitung (mis. "5 langkah aktif")
    started: float = field(default_factory=time.time)
    finished: float | None = None
    result: object = None
    error: str | None = None

    @property
    def running(self) -> bool:
        return self.finished is None

    def elapsed(self) -> str:
        seconds = int((self.finished or time.time()) - self.started)
        return f"{seconds // 60} mnt {seconds % 60} dtk" if seconds >= 60 else f"{seconds} dtk"


@st.cache_resource
def _registry() -> dict[str, Job]:
    """Satu registri untuk seluruh server (bukan per run skrip): nama tugas → Job."""
    return {}


def start(name: str, title: str, func, *args, note: str = "", **kwargs) -> bool:
    """Mulai func(*args, **kwargs) di thread. False bila tugas bernama sama masih berjalan.

    Argumen harus salinan yang tidak akan diubah halaman selama tugas berjalan (mis. copy.deepcopy(steps)).
    """
    registry = _registry()
    old = registry.get(name)
    if old is not None and old.running:
        return False
    job = Job(name, title, note)

    def work():
        try:
            job.result = func(*args, **kwargs)
        except Exception:  # galat ditampilkan di halaman, bukan ditelan
            job.error = traceback.format_exc()
        finally:
            job.finished = time.time()

    registry[name] = job
    threading.Thread(target=work, name=f"laya-{name}", daemon=True).start()
    return True


def get(name: str) -> Job | None:
    return _registry().get(name)


def running(name: str) -> bool:
    job = get(name)
    return job is not None and job.running


def result(name: str):
    """Hasil tugas yang sudah selesai tanpa galat, selain itu None."""
    job = get(name)
    return job.result if job is not None and not job.running and job.error is None else None


def forget(name: str) -> None:
    job = get(name)
    if job is not None and not job.running:
        del _registry()[name]


@st.fragment(run_every=2)
def _watch(name: str) -> None:
    """Hanya bagian kecil ini yang diperbarui tiap 2 detik selama tugas berjalan; saat selesai → rerun penuh
    sekali agar hasilnya tampil. Sesudah itu tidak dipanggil lagi (tidak ada polling sia-sia)."""
    job = get(name)
    if job is None or not job.running:
        st.rerun()
    st.info(f"⏳ **{job.title}** berjalan di latar belakang: {job.elapsed()}. Boleh pindah halaman atau "
            "mengubah apa pun; proses tetap jalan dan hasilnya muncul di sini saat selesai.")


def status(name: str) -> Job | None:
    """Tampilkan keadaan tugas di halaman pemiliknya. Kembalikan Job (atau None bila belum pernah dijalankan)."""
    job = get(name)
    if job is None:
        return None
    if job.running:
        _watch(name)
    elif job.error:
        st.error(f"{job.title} gagal sesudah {job.elapsed()}.")
        with st.expander("Rincian galat"):
            st.code(job.error)
    else:
        st.caption(f"✅ {job.title} selesai dalam {job.elapsed()}" + (f" · {job.note}" if job.note else ""))
    return job


@st.fragment(run_every=3)
def _sidebar_watch() -> None:
    if not any(job.running for job in _registry().values()):
        st.rerun()                      # semua selesai: segarkan halaman sekali, lalu berhenti memperbarui
    for job in _registry().values():
        if job.running:
            st.caption(f"⏳ {job.title} · {job.elapsed()}")
        elif time.time() - job.finished < 600:
            st.caption(f"{'⛔' if job.error else '✅'} {job.title} · selesai {job.elapsed()}")


def sidebar() -> None:
    """Daftar tugas di sidebar (tampil di semua halaman). Memperbarui diri hanya bila ada yang berjalan."""
    jobs = list(_registry().values())
    if not jobs:
        return
    with st.sidebar:
        st.subheader("Tugas latar belakang")
        if any(job.running for job in jobs):
            _sidebar_watch()
        else:
            for job in jobs:
                if time.time() - job.finished < 600:
                    st.caption(f"{'⛔' if job.error else '✅'} {job.title} · selesai {job.elapsed()}")
