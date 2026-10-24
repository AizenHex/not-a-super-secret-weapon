#!/bin/sh
# Pakai: sh build.sh tabular/klasifikasi [text/klasifikasi_tfidf ...]
# Ubah _alat/nb/<nama>.py -> _.NEW-GEN/<nama>.ipynb lalu jalankan (Run All).
SP="$(cd "$(dirname "$0")" && pwd)"
NEW="$(dirname "$SP")"
PY="$NEW/.venv/Scripts/python.exe"
[ -f "$PY" ] || PY="$(dirname "$NEW")/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
for name in "$@"; do
  "$PY" "$SP/py2nb.py" "$SP/nb/$name.py" "$NEW/$name.ipynb" || exit 1
  touch "$SP/log/mulai"
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 "$NEW/$name.ipynb" 2> "$SP/log/log_$(echo $name | tr / _).txt" \
    && echo "LOLOS $name" || { echo "GAGAL $name"; tail -30 "$SP/log/log_$(echo $name | tr / _).txt"; }
  # hasil notebook harus ke luar repo (../../outputs/...), bukan ke folder notebook
  find "$NEW/$(dirname "$name")" -mindepth 1 -maxdepth 1 -newer "$SP/log/mulai" ! -name "*.ipynb" \
    -exec echo "SAMPAH $name:" {} \;
done
