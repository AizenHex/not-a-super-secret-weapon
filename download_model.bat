@echo off
rem Unduh SEMUA model pretrained (sekitar 9 GB, butuh internet). Jalankan setup.bat dulu.
rem Hasil: folder "pretrained" di SEBELAH folder ini, contoh:
rem   C:\GEMASTIK\_.NEW-GEN\      (folder ini)
rem   C:\GEMASTIK\pretrained\     (hasil unduhan)
rem Kalau terputus, jalankan lagi: model yang sudah ada dilewati.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=..\.venv\Scripts\python.exe
if not exist %PY% (
    echo .venv belum ada. Jalankan setup.bat dulu.
    pause
    exit /b 1
)
%PY% _alat\unduh_model.py %*
if errorlevel 1 (
    echo GAGAL. Cek internet lalu jalankan lagi, model yang sudah selesai tidak diunduh ulang.
)
pause
