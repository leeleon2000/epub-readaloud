# epub-readaloud

Turn a DRM-free English EPUB into an audiobook you can read along with: an EPUB 3 with Media Overlays, where a
compatible reader plays the audio and highlights each sentence as it is spoken. Dialogue is read by a male
voice, a female voice and a narrator.

It runs offline. The pipeline itself needs only Python and ffmpeg; the speech and language models are separate
services that you point it at, and every one of them has a mock so you can try the whole thing first.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
readaloud init tests/fixtures/alice_epub2.epub --id alice
readaloud run alice --tts mock --llm mock        # no models: placeholder audio, real EPUB
# -> books/alice/out/alice-s-adventures-in-wonderland.epub
```

Then put a real voice behind it:

| You have | You get | Start here |
|---|---|---|
| A laptop | One narrator, with [Kokoro](https://github.com/hexgrad/kokoro) | [USAGE.md, step 2](docs/USAGE.md#2-a-real-voice-on-a-laptop-kokoro-one-narrator) |
| A laptop and any OpenAI-compatible language model server | Three voices | [USAGE.md, step 3](docs/USAGE.md#3-three-voices-add-a-language-model) |
| A GPU machine | Three designed voices with Fish Audio Drama 3 | [USAGE.md, step 4](docs/USAGE.md#4-the-best-voices-drama-3) |

## Hear it

Two complete books made with this pipeline and Drama 3 voices, as videos with a caption for every sentence:

| Book | Volume 1 | Volume 2 | Volume 3 |
|---|---|---|---|
| Pride and Prejudice, 15.5 hours | [chapters 1–23](https://youtu.be/61VGvM-R3Ss) | [chapters 24–42](https://youtu.be/VdSdBjGZT_k) | [chapters 43–61](https://youtu.be/kuk_pyEFZN0) |
| Oliver Twist, 20 hours | [chapters 1–19](https://youtu.be/WpKMd9KxKQ4) | [chapters 20–37](https://youtu.be/UzeRJoKAZjU) | [chapters 38–53](https://youtu.be/mNZ9Aj9eibA) |

These two are **corrected editions**. After the pipeline had assigned the voices, the speaker labels of the
Project Dialogism Novel Corpus were written into each book's corrections file
([USAGE.md, step 8](docs/USAGE.md#8-a-corrected-edition)), so they show how the result sounds, not how often the
pipeline is right on its own; that is the table further down. The texts are Project Gutenberg ebooks #1342 and
#730, the voices are synthetic, and no human narrator was recorded or imitated.

## What it does

- **Keeps the book intact.** The only change to the pages is a span around each sentence, and a check on every
  run stops if anything a reader would see has changed. EPUB 2 books are upgraded to EPUB 3.
- **Reads the story, not the wrapper.** Contents pages, captions, licence text and page numbers are recognised
  and left out.
- **Decides who reads each line of dialogue.** Speech tags (`said Mrs. Bennet`) are found by rule; a language
  model is asked only about the lines without one; anything uncertain stays with the narrator.
- **Can be stopped and resumed.** Every stage is idempotent, and no audio is ever generated twice.
- **Lets you overrule it.** Section categories, name genders, single attributions and the cast are plain files
  that always win over the models.

## How well the voices are assigned

Measured on five novels against the speaker labels of the
[Project Dialogism Novel Corpus](https://github.com/Priya22/project-dialogism-novel-corpus):

| | Third-person novels (3) | First-person novels (2) |
|---|---|---|
| Quotations read in the wrong gender | 0.04% to 0.92% | none |
| Quotations given a male or female voice | 90% to 97% | 41% to 61% |

Across all five, 22 of 9,922 quotations (0.22%) were read in the wrong gender. The design choice behind these
numbers is that a line read by the narrator is unremarkable and a man's line in a woman's voice is not, so the
pipeline voices a line only when it has evidence. [DESIGN.md](docs/DESIGN.md) has the method, the error
analysis, and the experiments that did not help.

## Limits

- English only, and only books that mark dialogue with quotation marks.
- Three voices, not one per character.
- In a first-person novel about half the dialogue stays with the narrator.
- There is no automatic check yet that the speech matches the text. Drama 3 occasionally misreads or cuts a
  line; the pipeline catches runaways, not subtle errors.
- Highlighting is per sentence, and has been tested in [Readest](https://readest.com) only.
- The numbers above are for one language model (Qwen3.6-35B-A3B, 4-bit) and five nineteenth- and early
  twentieth-century novels.

## Documentation

- [docs/USAGE.md](docs/USAGE.md): step by step, from the mock run to a whole novel; the files you can edit;
  what the warnings mean; how to score a book.
- [docs/MODELS.md](docs/MODELS.md): serving the language model, Kokoro and Drama 3; throughput; model licences.
- [docs/DESIGN.md](docs/DESIGN.md): how each part works, what was measured, what was tried and dropped, known
  open issues.

## Use it on books you may use

The tool does not remove DRM and is meant for books you are entitled to process. If you publish what it makes,
the book's copyright in each country where it will be available, and the licence of the speech model, are yours
to check.

## Licence and credits

The code is MIT ([LICENSE](LICENSE)). The models are separate downloads under their own licences; see
[docs/MODELS.md](docs/MODELS.md).

- Speech with Fish Audio Drama 3 is **Built with Fish Audio**.
- Evaluation uses the Project Dialogism Novel Corpus (Vishnubhotla, Hammond and Hirst, 2022), CC-BY 4.0. It is
  downloaded by `scripts/fetch_pdnc.sh` and is not part of this repository.
- The test books in `tests/fixtures/` are Project Gutenberg editions; see
  [tests/fixtures/README.md](tests/fixtures/README.md).
- Sentence splitting is [pySBD](https://github.com/nipunsadvilkar/pySBD); EPUB validation is
  [EPUBCheck](https://github.com/w3c/epubcheck).
