@echo off
rem Unduh SEMUA model pretrained (sekitar 9 GB, butuh internet). Jalankan setup.bat dulu.
rem Hasil: folder "pretrained" di SEBELAH folder ini, contoh:
rem   C:\GEMASTIK\<repo ini>\     (folder ini)
rem   C:\GEMASTIK\pretrained\     (hasil unduhan)
rem Kalau terputus, jalankan lagi: model yang sudah ada dilewati.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0cari_python.bat"
if not defined PY (
    echo Venv belum ada. Jalankan setup.bat dulu.
    pause
    exit /b 1
)
"%PY%" _alat\unduh_model.py %*
if errorlevel 1 (
    echo GAGAL. Cek internet lalu jalankan lagi, model yang sudah selesai tidak diunduh ulang.
)
pause
