# Serving the models

`readaloud` never loads a model itself. Every model runs as its own HTTP service and the pipeline talks to it
through an adapter, so the pipeline runs on a laptop while the models sit wherever there is a GPU. Each adapter
also has a `mock`, which is how the tests and the quick start in the README run with no model at all.

| What | Used for | Adapter (`config.yaml`) | Default endpoint | Needed for |
|---|---|---|---|---|
| A language model | deciding which quotes a man or a woman says; classifying odd sections | `llm.adapter: openai_compat` | `http://localhost:8000/v1` | three voices |
| Kokoro | speech, on a laptop | `tts.adapter: kokoro` | `http://localhost:9001` | a quick, light audiobook |
| Fish Audio Drama 3 | speech and voice design, on a GPU | `tts.adapter: drama3` | `http://localhost:8091` | the best-sounding result |

The Models page of `readaloud ui` asks each of these addresses once and says which ones the current settings
need.

If a service runs on another machine, forward its port (`ssh -N -L 8000:localhost:8000 user@gpu-host`) and leave
the endpoints as they are. For a whole book it is more reliable to run `readaloud` on the GPU machine itself: a
long-lived tunnel can stall without closing, and a whole book is several hours of requests.

## The language model

Any server that speaks the OpenAI chat API **and** can force its answer to a JSON schema works. The pipeline sends
`response_format: {"type": "json_schema", ...}` with an `enum` of the allowed speakers, so a model cannot answer
with a name that is not on the list.

Everything in this repository was measured with **Qwen3.6-35B-A3B** in a 4-bit quantisation
(`nvidia/Qwen3.6-35B-A3B-NVFP4`, about 19 GB) served by vLLM, with thinking turned off:

```bash
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model-name qwen3.6 --port 8000
# add the flags the model card recommends for your hardware (memory share, context length)
```

```yaml
# config.yaml (these are the defaults)
llm:
  adapter: openai_compat
  endpoint: http://localhost:8000/v1
  model: qwen3.6
  thinking: false        # sent as chat_template_kwargs.enable_thinking; other servers may need their own switch
  temperature: 0.2
```

Notes from running it:

- The pipeline cuts a book into chunks of about 1,500 words and asks about one chunk at a time, so the context a
  model needs does not grow with the book. A prompt is roughly 2,700 tokens; 8K of context is enough.
- Requests are sent several at a time (`voices.concurrency: 8`). A server that batches requests finishes a novel's
  text stages in 10 to 25 minutes.
- Answers are **not** reproducible from one server start to the next, even at `temperature: 0` with a fixed
  `seed` (see "Run-to-run variance" in [DESIGN.md](DESIGN.md)). Answers are cached per quote in the book's
  database, so a finished book does not change when you run it again.
- Other models have not been measured. The same model in another runtime or quantisation (llama.cpp, MLX) should
  be close, but check it on a book you can score before trusting it; `docs/USAGE.md` shows how to score.

## Kokoro (laptop)

```bash
pip install -e ".[kokoro]"
python services/kokoro/server.py --port 9001
curl -s localhost:9001/info        # {"model_version": "kokoro-0.9.4-82M", "sample_rate": 24000}
```

A voice is a Kokoro voice name (`af_heart`) or a mix (`b|bf_emma:0.70+bf_lily:0.30`; `a` American, `b` British
pronunciation). The cast stage makes the male and female voices as mixes. The server handles one request at a
time; on an M3 Pro laptop speech comes out at about 0.13 times real time (an hour of audio in about 8 minutes).

## Fish Audio Drama 3 (GPU)

Drama 3 is served by vLLM-Omni. The pipeline uses it twice: the cast stage asks it to read a sample passage with
no reference voice and keeps a candidate whose pitch suits the part (see "Speech synthesis" in
[DESIGN.md](DESIGN.md)); synth then clones those frozen samples for every line.

Setup that `scripts/drama3_serve.sh` expects:

1. A directory (default `~/drama3`, or set `DRAMA3_DIR`) with a virtualenv `.venv` holding `vllm` and `vllm-omni`.
2. The weights in the Hugging Face cache (`fishaudio/drama-3`; read its licence first, below).
3. `fish_omni_local.yaml` in that directory: a copy of `vllm_omni/deploy/fish_omni.yaml` with `max_num_seqs: 16`,
   `gpu_memory_utilization: 0.55` for the first stage and `0.08` for the codec stage, fewer cudagraph capture
   sizes, and `cuda_mps: false`.

```bash
scripts/drama3_serve.sh            # 16 streams, about 69 GB
scripts/drama3_serve.sh fast64     # 64 streams, about 94 GB; needs the GPU to itself
until curl -sf localhost:8091/v1/models; do sleep 10; done      # ready in about 8 minutes
scripts/drama3_smoke.sh            # four sample clips
scripts/drama3_stop.sh             # also stops the stage workers, which outlive the server and hold the memory
```

Match the pipeline to the server in the book's `config.yaml`:

```yaml
tts:
  adapter: drama3
  drama3: {concurrency: 64}        # the number of streams the server was started with
```

What was measured on one GPU machine with 128 GB of unified memory:

| Streams | Audio produced per second of wall time | Measured with |
|---|---|---|
| 1 | 1.35 s | one long passage, default voice |
| 4 | 3.95 s | one long passage, default voice |
| 8 | 6.07 s | one long passage, default voice |
| 16 | 9.94 s | one long passage, default voice |
| 16 | 6.65 s | real book lines, cloned voice |
| 32 | 8.99 s in one test, 10.3 s in another | real book lines, cloned voice |
| 64 | 13.0 s (1.26 times the 32-stream run in the same test) | real book lines, cloned voice |

The pipeline sends many short lines in a cloned voice, where the reference prompt is a large part of each request,
so the last three rows are the ones that predict a real run. At 64 streams a twelve-hour audiobook takes about
an hour of synthesis.

Things that cost time to find out:

- **The language model and Drama 3 did not fit in memory together** on that machine (Drama 3's weights alone are
  37.5 GiB). Run the text stages, stop the language model, then start Drama 3:
  `readaloud run <id> --stages ingest,segment,classify,voices`, swap the servers, `readaloud run <id> --tts drama3`.
- Start-up needs `--stage-init-timeout 1800` and `--init-timeout 1800` (loading the weights takes about four
  minutes cold) and `HF_HUB_OFFLINE=1` (without it a worker hung on a connection to the hub). The script sets
  them.
- Killing the server process is not enough: its stage workers keep the memory. Use `scripts/drama3_stop.sh`.
- A request carries one reference voice, so one request is one speaker.
- The request seed does not reproduce the audio when requests are batched. The pipeline therefore keeps every
  generated take on disk and never regenerates one it already has.
- Every request needs a length limit. The adapter sets it from the length of the text; output that fills the
  limit is treated as a runaway and retried with the next seed. Text nobody would read aloud (ISBNs, call
  numbers) runs away on every seed, which is one reason copyright pages are classified as boilerplate.

## Licences of the models

The code in this repository is MIT. The models are not part of it and have their own terms; you fetch them
yourself.

| Model | Licence, as we read it | What it asks of you |
|---|---|---|
| Fish Audio Drama 3 | Fish Audio Community License v1.0 | Free for research and non-commercial use, and for commercial use under a revenue threshold; no imitating an identifiable real person's voice without consent; do not remove its watermark; show "Built with Fish Audio" where you distribute something made with it |
| Kokoro | Apache-2.0 | |
| Qwen3.6-35B-A3B (the NVFP4 build used here) | Apache-2.0 | |

Read the licence text yourself before publishing audio; this table is a reading aid, not advice. The voices this
pipeline designs with Drama 3 are generated from no reference, not cloned from a person.
