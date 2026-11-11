@echo off
rem Figur gaya paper untuk PPT dari CSV hasil pipeline -> <HASIL>\figur_ppt\*.png dan *.svg (SVG untuk Figma).
rem Folder lain (mis. di laptop): 9_figur_ppt.bat ..\..\outputs\bawa_pulang
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0..\cari_python.bat"
if not defined PY (
    echo Venv belum ada. Jalankan setup.bat di root repo dulu.
    pause
    exit /b 1
)
"%PY%" "%~dp09b_slide.py" %*
if errorlevel 1 (
    echo GAGAL. Baca pesan di atas.
)
pause
