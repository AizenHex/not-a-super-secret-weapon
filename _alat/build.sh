#!/bin/sh
# Pakai: sh build.sh tabular/klasifikasi [text/klasifikasi_tfidf ...]
# Ubah _alat/nb/<nama>.py -> _.NEW-GEN/<nama>.ipynb lalu jalankan (Run All).
SP="Z:/GEMASTIK-DATABASE/_.NEW-GEN/_alat"
NEW="Z:/GEMASTIK-DATABASE/_.NEW-GEN"
PY="Z:/GEMASTIK-DATABASE/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
for name in "$@"; do
  "$PY" "$SP/py2nb.py" "$SP/nb/$name.py" "$NEW/$name.ipynb" || exit 1
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 "$NEW/$name.ipynb" 2> "$SP/log/log_$(echo $name | tr / _).txt" \
    && echo "LOLOS $name" || { echo "GAGAL $name"; tail -30 "$SP/log/log_$(echo $name | tr / _).txt"; }
done
