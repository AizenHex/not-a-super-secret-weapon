#!/usr/bin/env bash
# Hook Claude Code untuk _memori/ (dipasang di .claude/settings.json).
#   memori.sh mulai   SessionStart: tempel status.md + aturan.md ke konteks agen.
#   memori.sh potret  UserPromptSubmit: catat keadaan file sebelum agen bekerja.
#   memori.sh cek     Stop: kalau giliran ini mengubah file di luar _memori/ tapi
#                     _memori/ tidak ikut berubah, minta agen memperbarui memori dulu.
# Cukup git + bash, tanpa jq/python.
cd "$(dirname "$0")/../.." && cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
SNAP="$(git rev-parse --git-dir)/claude-memori-snap"

# "<hash> <path>" untuk tiap file yang beda dari HEAD (termasuk file baru), HEAD di baris pertama.
keadaan() {
    git rev-parse HEAD 2>/dev/null
    { git diff -z --name-only HEAD 2>/dev/null; git ls-files -z -o --exclude-standard; } |
        sort -zu | while IFS= read -r -d '' f; do
            if [ -f "$f" ]; then echo "$(git hash-object -- "$f") $f"; else echo "hapus $f"; fi
        done
}

case "$1" in
mulai)
    echo "Memori repo (_memori/status.md dan aturan.md), dimuat otomatis oleh hook:"
    cat _memori/status.md _memori/aturan.md 2>/dev/null
    ;;
potret)
    cat >/dev/null
    keadaan >"$SNAP"
    ;;
cek)
    masukan="$(cat)"
    # jangan memblokir dua kali berturut-turut
    echo "$masukan" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true' && exit 0
    [ -f "$SNAP" ] || exit 0
    head_lama="$(head -n1 "$SNAP")"
    sekarang="$(keadaan)"
    # file yang berubah selama giliran ini: commit baru + file kotor yang isinya berbeda dari potret
    berubah="$( {
        [ "$head_lama" != "$(echo "$sekarang" | head -n1)" ] && git diff --name-only "$head_lama" HEAD 2>/dev/null
        diff <(tail -n +2 "$SNAP") <(echo "$sekarang" | tail -n +2) | sed -n 's/^[<>] [^ ]* //p'
    } | sort -u)"
    [ -z "$berubah" ] && exit 0
    echo "$berubah" | grep -q '^_memori/' && exit 0
    daftar="$(echo "$berubah" | head -n 8 | tr '\n' ' ' | tr -d '"\\')"
    printf '{"decision":"block","reason":"Hook memori: giliran ini mengubah %s tetapi _memori/ belum diperbarui. Sesuai CLAUDE.md, tambah entri _memori/log.md (dan status/aturan/keputusan bila perlu) sebelum menjawab. Kalau perubahan itu bukan dari giliran ini atau memang tidak perlu dicatat, katakan singkat lalu selesai."}\n' "$daftar"
    ;;
esac
exit 0
