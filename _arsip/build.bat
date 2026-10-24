@echo off
rem Ubah _alat\nb\<folder>\<nama>.py jadi notebook, lalu jalankan semua selnya (Run All).
rem Pakai:
rem   build.bat                 semua folder
rem   build.bat vision text     folder tertentu saja
rem   build.bat vision\ocr_ekstraksi_field   satu notebook
rem Hasil LOLOS/GAGAL: _alat\log\hasil_build.txt. Pesan error tiap notebook: _alat\log\log_*.txt
chcp 65001 >nul
setlocal
set PYTHONIOENCODING=utf-8
set NEW=%~dp0
set ALAT=%~dp0_alat
call "%~dp0cari_python.bat"
set HASIL=%ALAT%\log\hasil_build.txt
echo build %date% %time% > "%HASIL%"

if "%~1"=="" goto semua_folder
for %%A in (%*) do call :satu_argumen %%A
goto selesai

:semua_folder
for /d %%F in ("%ALAT%\nb\*") do call :satu_folder %%~nxF
goto selesai

:satu_argumen
if exist "%ALAT%\nb\%~1.py" (
    for /f "tokens=1,2 delims=\/" %%a in ("%~1") do call :satu_notebook %%a %%b
) else (
    call :satu_folder %~1
)
exit /b

:satu_folder
for %%N in ("%ALAT%\nb\%1\*.py") do call :satu_notebook %1 %%~nN
exit /b

:satu_notebook
set FOLDER=%1
echo menjalankan %FOLDER%\%2 ...
"%PY%" "%ALAT%\py2nb.py" "%ALAT%\nb\%FOLDER%\%2.py" "%NEW%%FOLDER%\%2.ipynb"
"%PY%" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "%NEW%%FOLDER%\%2.ipynb" 2> "%ALAT%\log\log_%FOLDER%_%2.txt"
if errorlevel 1 (
    echo    GAGAL %FOLDER%\%2
    echo GAGAL %FOLDER%\%2>> "%HASIL%"
) else (
    echo    LOLOS %FOLDER%\%2
    echo LOLOS %FOLDER%\%2>> "%HASIL%"
)
exit /b

:selesai
echo.
echo ===== RINGKASAN =====
for /f %%c in ('findstr /b "LOLOS" "%HASIL%" ^| find /c /v ""') do echo LOLOS: %%c
for /f %%c in ('findstr /b "GAGAL" "%HASIL%" ^| find /c /v ""') do echo GAGAL: %%c
findstr /b "GAGAL" "%HASIL%"
echo Pesan error: buka _alat\log\log_^<folder^>_^<nama^>.txt
pause
