@echo off
rem Laya EDA v2. Cari python dari venv (lihat ..\cari_python.bat), lalu buka aplikasi di browser.
cd /d "%~dp0"
call ..\cari_python.bat
if not defined PY set "PY=%~dp0..\..\.venv\Scripts\python.exe"
"%PY%" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
