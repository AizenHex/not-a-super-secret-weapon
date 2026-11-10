@echo off
rem Kumpulkan luaran yang BOLEH keluar dari PC lomba ke ..\..\outputs\bawa_pulang\ (PNG, .md, tabel
rem ringkas, submission). Data per baris / teks asli / OOF ditolak otomatis.
rem Bila HASIL di konfig.py diganti, ganti juga HASIL di bawah ini.
rem Baca ..\..\outputs\bawa_pulang\MANIFEST.md sebelum menyalin keluar.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set "HASIL=..\..\outputs\final"
call "%~dp0..\cari_python.bat"
if not defined PY (
    echo Venv belum ada. Jalankan setup.bat di root repo dulu.
    pause
    exit /b 1
)
"%PY%" "%~dp0..\_alat\ekspor_hasil.py" "%HASIL%\eda" "%HASIL%\praproses" "%HASIL%\baseline" "%HASIL%\finetune" "%HASIL%\ensemble" "%HASIL%\topik" "%HASIL%\bertopic" "%HASIL%\laporan"
if errorlevel 1 (
    echo GAGAL. Baca pesan di atas.
)
pause
