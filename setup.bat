@echo off
rem Jalankan sekali: pasang semua library ke venv proyek.
rem Pakai PyCharm? Jalankan dari tab Terminal PyCharm (venv proyek sudah aktif di sana),
rem atau dobel klik: venv .venv / venv di folder ini dipakai kalau sudah ada.
rem Kalau folder ..\env\wheelhouse ada, pemasangan TANPA internet. Kalau tidak, lewat internet.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

call "%~dp0cari_python.bat"
if not defined PY (
    echo [1/3] Venv belum ada, membuat .venv ...
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
    call "%~dp0cari_python.bat"
)
if not defined PY (
    echo GAGAL membuat .venv. Pasang Python 3.13 dulu ^(centang "Add python.exe to PATH"^).
    pause
    exit /b 1
)
echo Venv yang dipakai: %PY%
"%PY%" --version

echo [2/3] Memasang library ...
if exist "..\env\wheelhouse" (
    echo Wheelhouse offline ditemukan: pasang tanpa internet.
    "%PY%" -m pip install --no-index --find-links "..\env\wheelhouse" -r requirements.txt
) else (
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
)

rem sukses atau gagal ditentukan oleh cek import, bukan pesan pip
echo [3/3] Cek import ...
"%PY%" -c "import sklearn, lightgbm, catboost, xgboost, transformers, torch, streamlit, laya; print('OK. GPU:', torch.cuda.is_available())"
if errorlevel 1 (
    echo GAGAL: ada library yang belum terpasang. Baca pesan error di atas.
    echo Kalau ada kata "Long Path": pindahkan folder ke path pendek, misalnya C:\GEMASTIK\
    pause
    exit /b 1
)
echo.
echo SUKSES. Buka jupyter.bat untuk mulai, atau aktifkan.bat untuk terminal.
pause
