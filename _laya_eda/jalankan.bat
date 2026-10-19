@echo off
rem Dobel klik file ini. Browser terbuka di http://localhost:8501 . Tutup jendela ini untuk berhenti.
cd /d "%~dp0"
set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\..\.venv\Scripts\python.exe"
"%PY%" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
