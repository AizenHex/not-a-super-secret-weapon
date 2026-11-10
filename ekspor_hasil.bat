@echo off
rem Kumpulkan bahan presentasi yang BOLEH keluar dari PC lomba ke ..\outputs\bawa_pulang\
rem (PNG, catatan .md, tabel ringkas, file kumpul). Tabel per baris / teks asli / OOF ditolak.
rem Pakai: ekspor_hasil.bat                       semua ..\outputs\text\*
rem        ekspor_hasil.bat --notebook a.ipynb b.ipynb --model ..\outputs\text\x\model
rem Baca ..\outputs\bawa_pulang\MANIFEST.md sebelum menyalin keluar.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0cari_python.bat"
if not defined PY (
    echo Venv belum ada. Jalankan setup.bat dulu.
    pause
    exit /b 1
)
"%PY%" _alat\ekspor_hasil.py %*
if errorlevel 1 (
    echo GAGAL. Baca pesan di atas.
)
pause
