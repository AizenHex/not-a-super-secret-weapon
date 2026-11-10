@echo off
rem Cek cepat PC lomba (jalankan setelah setup.bat): GPU + Run All baseline_oof di data demo.
rem Butuh folder ..\data_latihan (data demo, di luar repo). Hasil: ..\outputs\text\baseline_oof\
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call "%~dp0cari_python.bat"
if not defined PY (
    echo Venv belum ada. Jalankan setup.bat dulu.
    pause
    exit /b 1
)

echo [1/2] Cek GPU ...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>nul || echo nvidia-smi tidak ada: driver NVIDIA belum terpasang?
"%PY%" -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU dipakai:', ok); print(torch.cuda.get_device_name(0) if ok else 'GPU TIDAK TERPAKAI: torch cu128 butuh driver NVIDIA 570 ke atas')"

echo [2/2] Run All contoh\text\baseline_oof.ipynb (CPU, beberapa menit) ...
if not exist "..\data_latihan\text\sentiment\smsa_train.csv" (
    echo GAGAL: ..\data_latihan\text\sentiment\smsa_train.csv tidak ada. Salin folder data_latihan dulu.
    pause
    exit /b 1
)
set "HASIL=%~dp0..\outputs\text\baseline_oof"
if not exist "%HASIL%" mkdir "%HASIL%"
pushd contoh\text
"%PY%" -m jupyter nbconvert --to notebook --execute baseline_oof.ipynb --output-dir "%HASIL%" --output baseline_oof_hasil --ExecutePreprocessor.timeout=1800
set "RUN_ERROR=%errorlevel%"
popd
if not "%RUN_ERROR%"=="0" (
    echo GAGAL: ada sel yang error. Buka baseline_oof.ipynb di jupyter.bat dan jalankan per sel.
    pause
    exit /b 1
)
"%PY%" -c "import json, sys; nb = json.load(open(sys.argv[1], encoding='utf-8')); text = ''.join(''.join(o.get('text', '')) for c in nb['cells'] for o in c.get('outputs', [])); [print(line) for line in text.splitlines() if 'MASALAH' in line]; sys.exit(0 if 'MASALAH: 0' in text else 1)" "%HASIL%\baseline_oof_hasil.ipynb"
if errorlevel 1 (
    echo GAGAL: validator submission menemukan masalah. Lihat %HASIL%\baseline_oof_hasil.ipynb
    pause
    exit /b 1
)
echo.
echo SUKSES: baseline jalan dan MASALAH: 0. Notebook hasil: %HASIL%\baseline_oof_hasil.ipynb
pause
