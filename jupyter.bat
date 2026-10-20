@echo off
rem Buka JupyterLab memakai venv proyek (jalankan setup.bat dulu kalau belum).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0cari_python.bat"
if not defined PY (
    echo Venv belum ada. Jalankan setup.bat dulu.
    pause
    exit /b 1
)
"%PY%" -m jupyterlab
pause
