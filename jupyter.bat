@echo off
rem Buka JupyterLab memakai .venv di folder ini (jalankan setup.bat dulu kalau belum).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -m jupyterlab
pause
