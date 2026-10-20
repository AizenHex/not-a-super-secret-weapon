@echo off
rem Dobel klik file ini. Browser terbuka di http://localhost:8501 . Tutup jendela ini untuk berhenti.
cd /d "%~dp0"
call "%~dp0..\cari_python.bat"
if not defined PY (
    echo Venv belum ada. Jalankan setup.bat di folder atas dulu.
    pause
    exit /b 1
)
"%PY%" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
