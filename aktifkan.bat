@echo off
rem Buka terminal dengan .venv aktif. Ketik `deactivate` untuk keluar dari venv.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
cmd /k ".venv\Scripts\activate.bat"
