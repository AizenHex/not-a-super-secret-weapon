@echo off
rem Jalankan sekali (dobel klik): buat .venv di folder ini lalu pasang semua library.
rem Kalau folder ..\env\wheelhouse ada, pemasangan TANPA internet. Kalau tidak, lewat internet.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

if not exist .venv\Scripts\python.exe (
    echo [1/3] Membuat .venv ...
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo GAGAL membuat .venv. Pasang Python 3.13 dulu ^(centang "Add python.exe to PATH"^).
    pause
    exit /b 1
)
.venv\Scripts\python.exe --version

echo [2/3] Memasang library ...
if exist "..\env\wheelhouse" (
    echo Wheelhouse offline ditemukan: pasang tanpa internet.
    .venv\Scripts\python.exe -m pip install --no-index --find-links "..\env\wheelhouse" -r requirements.txt
) else (
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)
if errorlevel 1 (
    echo GAGAL memasang library. Baca pesan error di atas.
    echo Kalau ada kata "Long Path": pindahkan folder ke path pendek, misalnya C:\GEMASTIK\
    pause
    exit /b 1
)

echo [3/3] Cek import ...
.venv\Scripts\python.exe -c "import sklearn, lightgbm, catboost, xgboost, transformers, torch; print('OK. GPU:', torch.cuda.is_available())"
echo.
echo Selesai. Buka jupyter.bat untuk mulai, atau aktifkan.bat untuk terminal.
pause
