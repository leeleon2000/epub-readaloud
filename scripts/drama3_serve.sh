#!/usr/bin/env bash
# Start Fish Audio Drama 3 (vLLM-Omni) on port 8091. Run on the GPU host:
#   scripts/drama3_serve.sh [full|fast|fast48|fast64]
# Setup this script expects (docs/MODELS.md): a directory $DRAMA3_DIR (default ~/drama3) holding a venv with
# vllm + vllm-omni and `fish_omni_local.yaml`, a copy of vllm_omni/deploy/fish_omni.yaml with max_num_seqs 16,
# gpu_memory_utilization 0.55 (stage 0) / 0.08 (codec), fewer cudagraph sizes and `cuda_mps: false`.
# full (default): 16 streams, about 69 GB on a 128 GB unified-memory machine. An LLM server of ~50 GB does not
#   fit next to it: run the text stages first, stop the LLM, then start this.
# Ready in ~8 min (weights ~4 min, then CUDA graphs): until curl -sf localhost:8091/v1/models; do sleep 10; done
set -euo pipefail
cd "${DRAMA3_DIR:-$HOME/drama3}"
if [ "${1:-full}" = full ]; then
  CFG=fish_omni_local.yaml
elif [[ "${1:-full}" =~ ^fast([0-9]*)$ ]]; then
  # fast = 32 streams in both stages, fast48 / fast64 = 48 / 64, more memory (stage 0 0.65, codec 0.12; ~94 GB):
  # only when nothing else uses the GPU. Pair it with tts.drama3.concurrency: <streams>. The KV cache
  # (35 GiB at 0.65) holds 64 of our short requests; the codec stage has no KV cache.
  N=${BASH_REMATCH[1]:-32}
  CFG=fish_omni_local_fast$N.yaml
  sed -e "s/max_num_seqs: 16/max_num_seqs: $N/" \
      -e "0,/gpu_memory_utilization: 0.55/s//gpu_memory_utilization: 0.65/" \
      -e "0,/gpu_memory_utilization: 0.08/s//gpu_memory_utilization: 0.12/" fish_omni_local.yaml > $CFG
else
  echo "usage: $0 [full|fast|fast48|fast64]" >&2; exit 2
fi
if curl -sf localhost:8091/v1/models >/dev/null; then echo "already running"; exit 0; fi
export PATH=$PWD/.venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# Weights are in the HF cache. Without OFFLINE the stage-0 worker can hang on a connection to the hub
# (2026-11-15); the default 600 s orchestrator limit is too short when memory is tight.
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
# Unified memory: the page cache (weights just read, a stopped model's files) is not handed back to CUDA in time.
# With 37 GiB cached the codec stage died at CUDA context creation ("CUDA error: out of memory", 2026-11-18).
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "could not drop the page cache (sudo)"
nohup vllm serve fishaudio/drama-3 --omni --port 8091 --deploy-config $CFG \
  --init-timeout 1800 --stage-init-timeout 1800 > serve.log 2>&1 < /dev/null &
echo $! > serve.pid
echo "started pid $(cat serve.pid) with $CFG (log $PWD/serve.log)"
