@echo off
rem Buka JupyterLab di folder final-pipeline memakai venv repo (dibuat setup.bat).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0..\cari_python.bat"
if not defined PY (
    echo Venv belum ada. Jalankan setup.bat di root repo dulu.
    pause
    exit /b 1
)
"%PY%" -m jupyterlab
pause
