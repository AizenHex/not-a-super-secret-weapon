@echo off
rem Buka terminal dengan venv proyek aktif. Ketik `deactivate` untuk keluar dari venv.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0cari_python.bat"
if not defined PY (
    echo Venv belum ada. Jalankan setup.bat dulu.
    pause
    exit /b 1
)
rem activate.bat ada di folder yang sama dengan python.exe
for %%P in ("%PY%") do set "SCRIPTS=%%~dpP"
cmd /k ""%SCRIPTS%activate.bat""
