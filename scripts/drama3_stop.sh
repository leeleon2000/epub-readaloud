#!/usr/bin/env bash
# Stop Drama 3 and its stage workers (they outlive the API server and hold the memory). Run on the GPU host.
# -u "$USER" leaves other users' vLLM workers (same process names) alone.
cd "${DRAMA3_DIR:-$HOME/drama3}"
P=$(cat serve.pid 2>/dev/null || true)
# serve.pid may be stale and its number reused: only touch it if it is our Drama 3 server
if [ -n "$P" ] && ! ps -o args= -p "$P" 2>/dev/null | grep -q "[f]ishaudio/drama-3"; then P=""; fi
[ -n "$P" ] && kill "$P" 2>/dev/null
sleep 5
pkill -u "$USER" -f "[S]tageEngineCoreProc"
sleep 5
[ -n "$P" ] && ps -p "$P" >/dev/null 2>&1 && kill -9 "$P"
ps -u "$USER" -o pid,args | grep -E "[S]tageEngineCoreProc|[v]llm serve fishaudio" || echo "Drama 3 stopped"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
