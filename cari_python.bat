@echo off
rem Dipanggil oleh .bat lain: isi PY dengan python dari venv. Urutan cari:
rem   1) venv yang sedang aktif (terminal PyCharm otomatis mengaktifkan venv proyek)
rem   2) .venv atau venv di folder ini (PyCharm membuat venv di root proyek)
rem   3) .venv atau venv satu folder di atas
rem Tidak ketemu = PY kosong.
set "PY="
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PY=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\venv\Scripts\python.exe" set "PY=%~dp0..\venv\Scripts\python.exe"
