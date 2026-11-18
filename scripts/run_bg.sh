#!/usr/bin/env bash
# Start `readaloud run <book> <args>` detached from the terminal so it survives SSH disconnects
# (for a run started over SSH on the GPU host). Appends to books/<book>/run.log.
# usage: scripts/run_bg.sh <book_id> [readaloud run args...]   e.g. scripts/run_bg.sh pride --stages characters,attribute
cd "$(dirname "$0")/.."
book="$1"; shift
{ echo; echo "=== $(date '+%F %T') readaloud run $book $*"; } >> "books/$book/run.log"
PYTHONUNBUFFERED=1 setsid -f .venv/bin/readaloud run "$book" "$@" >> "books/$book/run.log" 2>&1 < /dev/null
echo "started; log: $(pwd)/books/$book/run.log"
