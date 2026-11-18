#!/usr/bin/env bash
# Smoke test for Fish Audio Drama 3 served by vLLM-Omni.
# Run on the GPU host after `vllm serve fishaudio/drama-3 --omni --port 8091` is up.
# Writes wavs to ~/drama3/out/.
set -euo pipefail
URL=${URL:-http://localhost:8091/v1/audio/speech}
OUT=${OUT:-$HOME/drama3/out}
mkdir -p "$OUT"

REF_AUDIO="https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REF_TEXT="We asked over twenty different people, and they all said it was his."

say() { # name, json-body
  local name=$1 body=$2 t0 t1
  t0=$(date +%s.%N)
  curl -sf -X POST "$URL" -H "Content-Type: application/json" -d "$body" -o "$OUT/$name.wav"
  t1=$(date +%s.%N)
  printf '%-14s %6.2fs  %s\n' "$name" "$(echo "$t1-$t0" | bc)" "$(file -b "$OUT/$name.wav")"
}

# 1. Plain narration, default voice
say 01_narration '{"model":"fishaudio/drama-3","response_format":"wav",
 "input":"It was a bright cold day in April, and the clocks were striking thirteen. The old house at the end of the lane had been empty for years."}'

# 2. Point tags (non-speech events)
say 02_tags '{"model":"fishaudio/drama-3","response_format":"wav",
 "input":"[sighs] Fine. If we miss the last train, it is on you. [laughs] Well, maybe that is fair."}'

# 3. Voice cloning from a reference clip
say 03_clone "{\"model\":\"fishaudio/drama-3\",\"response_format\":\"wav\",\"voice\":\"default\",
 \"ref_audio\":\"$REF_AUDIO\",\"ref_text\":\"$REF_TEXT\",
 \"input\":\"The letter arrived on a Tuesday, sealed with red wax and smelling faintly of smoke.\"}"

# 4. Multi-speaker dialogue (novel-style scene)
say 04_dialogue '{"model":"fishaudio/drama-3","response_format":"wav",
 "input":"<|speaker:0|>Where have you been all night? <|speaker:1|>Out. Walking. Thinking. <|speaker:0|>[sighs] You could have called. <|speaker:1|>I know. I am sorry."}'

ls -la "$OUT"
