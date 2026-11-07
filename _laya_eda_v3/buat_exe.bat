@echo off
rem Membuat dist\Laya\Laya.exe (folder onedir, tidak di-commit). Butuh venv dengan PySide6 + polars + openpyxl +
rem pyinstaller (..\requirements.txt). Salin seluruh folder dist\Laya ke flashdisk, bukan hanya Laya.exe.
cd /d "%~dp0"
call ..\cari_python.bat
if not defined PY (
  echo Python venv tidak ditemukan. Jalankan ..\setup.bat dulu.
  pause
  exit /b 1
)
rem Library berat tidak dipakai Laya.exe (latih berjalan di venv lewat proses terpisah): jangan ikut dikemas.
"%PY%" -m PyInstaller --noconfirm --clean --windowed --name Laya --distpath dist --workpath build --specpath build ^
  --exclude-module torch --exclude-module torchvision --exclude-module transformers --exclude-module pandas ^
  --exclude-module pyarrow --exclude-module matplotlib --exclude-module scipy --exclude-module sklearn ^
  --exclude-module IPython --exclude-module tkinter --exclude-module numba --exclude-module PIL ^
  --exclude-module rasterio --exclude-module pyogrio --exclude-module pyproj --exclude-module geopandas ^
  --exclude-module shapely --exclude-module botocore --exclude-module boto3 --exclude-module s3fs ^
  --exclude-module cryptography --exclude-module lxml --exclude-module numpy --exclude-module PySide6.QtQuick ^
  --exclude-module PySide6.QtQml --exclude-module PySide6.QtPdf --exclude-module PySide6.QtNetwork ^
  --add-data "%~dp0mesin\muat.py;sumber" --add-data "%~dp0mesin\bantu.py;sumber" --add-data "%~dp0mesin\tugas.py;sumber" ^
  laya.py
if errorlevel 1 (
  echo Gagal membuat exe.
  pause
  exit /b 1
)
echo Selesai: %~dp0dist\Laya\Laya.exe
echo Uji cepat: dist\Laya\Laya.exe --uji ^<folder data^>  lalu baca dist\Laya\proyek\laya_log.txt
pause
