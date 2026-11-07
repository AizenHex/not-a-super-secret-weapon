@echo off
rem Laya v3 tanpa exe (cadangan bila antivirus memblokir Laya.exe): jalan langsung dari venv.
cd /d "%~dp0"
call ..\cari_python.bat
if not defined PY (
  echo Python venv tidak ditemukan. Jalankan ..\setup.bat dulu.
  pause
  exit /b 1
)
"%PY%" laya.py %*
if errorlevel 1 pause
