# How it works, what was measured, and what did not help

This is the long version of the README: the rules the tool keeps, how each part works, the numbers behind the
choices, the things that were tried and dropped, and what is still open. Step-by-step use is in
[USAGE.md](USAGE.md); serving the models is in [MODELS.md](MODELS.md).

## 1. What it does and the rules it keeps

Input: one DRM-free English EPUB. Output: an EPUB 3 with Media Overlays, which a compatible reader plays as an
audiobook while highlighting the sentence being read. Dialogue is split between a male voice, a female voice
and the narrator.

Five rules shaped everything else:

1. **The displayed text is never altered.** The only change to the book's pages is a `<span>` around each
   sentence. A check runs every time and stops the run if anything a reader would see has changed.
2. **Every stage can be run again and picks up where it stopped.** All state is in one SQLite file per book.
   Generated audio is stored under a key made from the text, voice, model version, parameters and seed, so
   nothing is generated twice.
3. **Models sit behind adapters.** Speech, voice design and the language model each run as a separate HTTP
   service. Every adapter has a mock, so the whole pipeline runs and is tested on a laptop with no model.
4. **What a person edits is a file, and it wins.** Section categories, name genders, attributions and the cast
   are YAML or CSV files. The database holds state and caches only.
5. **A language model call never depends on the length of the book.** The text is cut into chunks of about
   1,500 words that never cross a chapter file. The model fills in one form per chunk; the order of work is
   fixed Python, not something a model decides.

Audio stays lossless (WAV) until packaging.

## 2. Pipeline

| Stage | What it does |
|---|---|
| `ingest` | Unpacks the EPUB. EPUB 2 is upgraded to EPUB 3, and the display check confirms the upgrade changed nothing visible |
| `segment` | Splits the text into sentences, wraps each in a span with a stable id, runs the display check again |
| `classify` | Decides for every section whether it is story, front or back matter, or something that is not read |
| `cast` | Makes the three voices and a sample of each |
| `voices` | Splits sentences into narration and quotations, and gives every quotation a voice |
| `synth` | Generates one take per piece of text |
| `assemble` | Joins takes into one WAV per chapter, levels it, and records when each sentence starts and ends |
| `package` | Encodes the audio, writes the SMIL timing files and the EPUB, runs epubcheck |

With `multi_voice: false` the `cast` and `voices` stages are skipped and the narrator reads everything.

A book's directory:

```
books/<id>/
  source.epub                 the input, untouched
  project.db                  every intermediate result
  config.yaml                 this book's settings (optional)
  sections.yaml  genders.yaml  attribution_overrides.csv  cast*.yaml     files a person may edit
  voices/                     voice samples and the reference clips that are cloned
  takes/<key>.wav             generated audio, one file per take
  work/                       unpacked source, upgraded copy, segmented pages, per-chapter audio, build tree
  out/<title>.epub            the result
  reports/                    epubcheck output, lines that needed attention, evaluation reports
```

## 3. Segmentation and the display check

Sentence splitting has to survive real book markup: a sentence can start in one `<i>` and end in another, and a
footnote anchor can sit in the middle of it.

- Text between block elements is treated as one run. Its whitespace is collapsed, pySBD finds the sentence
  boundaries, and the boundaries are mapped back to offsets in the original nodes.
- Each sentence is rebuilt by slicing the inline tree. An inline element that a boundary cuts through is cloned,
  and only the first clone keeps its `id`, so links into the book still land.
- Empty nodes (comments, `<br/>`, empty anchors, `svg`, `script`) go with the sentence they sit in.
- Sentence ids are `ab<file>s<number>` and are recomputed from the unsegmented copy on every run, so they are
  the same every time.

The display check compares, for every character of text, the chain of enclosing tags and their attributes
before and after, ignoring the added spans and a short list of attributes that do not affect rendering. Any
difference raises an error and the run stops. It runs twice: after the EPUB 2 to 3 upgrade and after
segmentation.

A sentence is further split into **utterances**: narration and quotation pieces. `"Come in," said Tom, "and sit
down."` is one sentence and three utterances. Audio is generated per utterance, so each can have its own voice;
highlighting stays per sentence.

## 4. Packaging for read-aloud players

| Choice | Reason |
|---|---|
| AAC-LC, 48 kbps, mono, in m4a | Stayed in sync on every device tried |
| MP3 CBR 64 kbps as a fallback | Also in sync |
| Not Opus | The file plays, but the highlight drifts on device even though the timings in the book are exact. Seeking in Ogg is the likely cause |
| One audio file per chapter | Fewer files, and seeking within a chapter is what players do well |
| SMIL clips are contiguous, and the last clip ends before the file does | Players skip or stall on gaps and on a clip that runs to the exact end |
| EPUB 2 input is upgraded to EPUB 3 | Media Overlays exist only in EPUB 3 |

Many real books fail epubcheck before anything is done to them. The package stage therefore fails only on
errors the source did not already have, matching on error code, file and message and ignoring line numbers.

Chapters are levelled to an RMS of -20 dB with peaks under -3 dB. Pauses are 0.35 s after a sentence, 0.8 s
after a paragraph and 1.0 s after a heading.

## 5. What gets read

The first whole-book run read 409 sentences of licence text, preface and list of illustrations as if they were
the novel. The `classify` stage exists because of that.

- The unit is a section: the sentences between two headings in one file. Whole files are too coarse.
- Rules decide most sections: Project Gutenberg's start and end markers, headings such as Contents, Preface,
  Introduction and List of Illustrations, page numbers, and a section with two or more kinds of copyright
  wording.
- The language model decides the rest. It sees only the heading, the first 100 words and the position in the
  book.
- An illustration's caption (a `figcaption`, or an element whose class says caption) is marked when the text is
  segmented and is never read, whatever section it sits in. An illustrated Pride and Prejudice has 54 of them,
  22 being the same copyright line under a picture, and some would otherwise have been voiced as dialogue.
- Single lines that no rule catches (a printer's imprint after the last sentence) go in the `skip:` list of
  `sections.yaml`.

| Category | Searched for dialogue | Read aloud |
|---|---|---|
| `story` | yes | yes |
| `front_matter`, `notes`, `back_matter` | no | yes, by the narrator |
| `toc`, `captions`, `boilerplate`, page numbers | no | no, and left out of the timing files |

`sections.yaml` overrides all of it.

## 6. Who reads which line

### 6.1 The first approach: one voice per character

The original goal was a voice for each character. That needs a roster, so the pipeline collected every way
each person is referred to, decided who is who (name rules, then a relation graph for references such as "his
wife"), and then asked the model for the speaker of each quotation from a closed list of the people in the
scene. A rule for two people speaking in turn corrected untagged lines.

Scored on speaker identity against PDNC (section 7):

| Book | Quotations whose speaker was right |
|---|---|
| Pride and Prejudice | 97.7% |
| Oliver Twist | 88.7% |
| The Sun Also Rises | 83.1% |

It worked on the book it was developed on and fell off on the next two. In Oliver Twist many speakers are
tagged by description rather than name ("said the Jew", "replied the girl"), and the roster did not accept
descriptions as names: the right person was missing from the model's list for 200 of 289 errors. In The Sun
Also Rises the narrator is a character, and a third of the errors were his own lines given to someone else.
A wrong roster entry also costs every line that person speaks. This code is still in the repository
(`attribute.version: v2`) but is not being developed.

### 6.2 Three voices

A listener does not need to know which of five sisters is speaking to follow a novel. What breaks the illusion
is a man's line in a woman's voice. So the target became: every quotation is read by a male voice, a female
voice or the narrator, and **a line the pipeline is unsure of stays with the narrator**. Falling back to the
narrator is cheap, because that is how a single-voice audiobook reads the whole book anyway. A wrong gender is
the error that counts.

Three definitions settle the edge cases: a first-person narrator's own speech is read by the narrator; a letter
read out takes its writer's gender; thoughts are treated as speech.

How a quotation gets its voice:

1. **Quotations are found by rule**, from the quotation marks, with the book's quote style detected first.
2. **Speech tags are found by rule**: `"...," said Mrs. Bennet`, `she cried`, `replied the girl`. A **tagged**
   quotation has one in its own paragraph; the rest are **untagged**.
3. **A name-to-gender table** is built for the book: titles (Mr., Mrs., Lady), nouns (mother, gentleman),
   the pronouns the narration uses for a name, and for what is left one model call with
   example sentences. `genders.yaml` holds the table, and its `manual:` entries win.
4. **Quoted words that are not speech** (a quoted phrase inside a sentence, a title, a sign) go to the
   narrator by rule.
5. **The model is asked only about untagged quotations**, one chunk at a time. It is given the passage with the
   quotations numbered, the tagged ones as known facts, and the names seen in this chunk and the one before.
   It is asked twice, in separate calls: *who* says each one (a name from the list, "another man", "another
   woman", "not speech" or "unknown"), and *which voice* (male, female, narrator, unknown).
6. **Evidence becomes a class and a confidence.** The confidence is not the model's own number:

| Confidence | Evidence |
|---|---|
| 0.95 | speech tag with a name, title, noun or pronoun of known gender; "I said" |
| 0.9 | tag whose gender was learned from the book; a not-speech rule |
| 0.85 | same paragraph as a tagged quotation; a scene rule that agrees with the model |
| 0.8 | the two model questions agree |
| 0.75 | a scene rule fills a quotation the model could not decide |
| 0.65 | one model answer only, gender confirmed by the table |
| 0.6 | a model answer inside a crowded exchange |
| 0.55 | one model answer only, person not in the table |
| 0.5 | the model contradicts the paragraph's tag |
| 0.3 | the two model questions disagree, or the model contradicts a scene rule |
| 0.2 | the model's person and the model's gender contradict each other |

7. **Scene rules** work on runs of consecutive paragraphs that contain speech: quotations in the same
   paragraph share a speaker; in an exchange where every known speaker has the same gender, the untagged lines
   have it too; where two voices alternate, the paragraph position decides. In an exchange with four or more
   people of both genders, model answers are held under the threshold.
8. **A quotation is voiced male or female only at confidence 0.7 or more.** In a first-person book (twenty or
   more "I said" tags) the threshold is 0.85, because there the two questions agreeing proved too weak: the
   model kept giving the narrator's own lines to other characters.

Results on five novels, scored against PDNC. "Wrong gender" counts quotations given a male or female voice of
the wrong gender, as a share of all scored quotations. "Voiced" is the share of quotations spoken by a man or a
woman that got a male or female voice at all.

| Book | Role in development | Wrong gender | Voiced | Narrator's own lines voiced |
|---|---|---|---|---|
| Pride and Prejudice | tuned on | 6 of 1,270 (0.47%) | 91% | |
| Oliver Twist | tuned on | 1 of 2,566 (0.04%) | 97% | |
| The Sun Also Rises | tuned on | 0 | 61% | 1.15% |
| A Room with a View | held out | 15 of 1,634 (0.92%) | 90% | |
| The Mysterious Affair at Styles | held out | 0 of 1,853 | 41% | 0% |
| All five | | 22 of 9,922 (0.22%) | | |

The three books the rules were tuned on come to 0.11%, the two held-out books to 0.43%. The honest expectation
for a new third-person novel is a few wrong-gender lines per thousand quotations, with about nine in ten
quotations voiced; for a first-person novel, almost no wrong lines and roughly half the dialogue left to the
narrator.

The target was 0.5%. A Room with a View missed it. Of its 15 errors, two are mislabelled in PDNC and two more
can be read either way, which says the pipeline's error rate is now of the same size as the noise in the
labels.

### 6.3 What is left

The 22 wrong-gender quotations, each read in context:

| Cause | Quotations |
|---|---|
| The model was wrong and the passage is hard: the speaker follows from narration | 5 |
| The model was wrong where existing information could have stopped it (three paragraphs in a row given to one person; the person addressed taken for the speaker) | 4 |
| PDNC label wrong, or two readings possible | 4 |
| A chunk boundary fell inside the conversation | 3 |
| The speech tag is at the end of the previous paragraph (`...she said,--` and then the quotation on a new line) | 2 |
| The speech-tag rule attached a tag to the wrong quotation | 2 |
| The gender table was wrong for a description | 1 |
| The model could not decide and a scene rule filled in the wrong gender | 1 |

Twelve of the 22 were voiced because the two model questions agreed, seven by a scene rule, three by a speech
tag.

Why 1,883 quotations with a male or female speaker went to the narrator instead:

| Reason | Quotations | Was the withheld answer right? |
|---|---|---|
| First-person book: the two questions agreed, but the threshold there is 0.85 | 1,073 | 99% |
| Crowded exchange, model answer not used | 371 | 95% |
| The two questions disagreed, or the model contradicted a scene rule | 188 | 61% |
| Only one of the two questions gave an answer | 108 | 99% |
| Taken for the narrator's own speech | 88 | |
| Everything else (quotation not found, not-speech rule, model said unknown) | 55 | |

The first row is the price of the first-person threshold. It keeps the narrator's own lines out of other
characters' voices (1.15% at 0.85 in The Sun Also Rises, 6.2% at 0.7), and on the second first-person book it
turned out to be more caution than needed.

The model's raw answers, before any rule, on the 4,138 untagged quotations with a male or female speaker:

| | Answered the other gender |
|---|---|
| "Who" question | 159 (3.8%) |
| "Which voice" question | 154 (3.7%) |
| Both questions agree | 86 (2.3%) |

The two questions catch each other's errors less than half the time, because the same model reads the same
passage twice. Errors rise with the distance from the last speech tag (third-person books, "who" question):

| Paragraphs since the last tagged quotation | Quotations | Wrong gender |
|---|---|---|
| 1 | 708 | 1.3% |
| 2 | 345 | 1.7% |
| 3 | 236 | 3.8% |
| 4 to 5 | 249 | 3.2% |
| 6 or more | 230 | 6.5% |
| No tag yet in this chunk | 228 | 9.6% |

The number of quotations in one call made no difference (2.9% to 4.5%, with no trend, from 1-15 quotations up to 46 or more).

### 6.4 Things tried that did not help

After the five-book measurement, one more round tried to improve the result without touching the rules, only by
asking the model differently. Each idea was measured; none is on by default.

**Asking who is being spoken to.** The listener cannot be the speaker, and in a two-person exchange the listener
of one line is the speaker of the previous one. Compared with PDNC's own listener labels the model's answer was
right about half the time. As a check on the speaker it caught 20% to 50% of errors and blocked 6% to 15% of
right answers, no better than the second question already in use.

**Asking for the evidence.** The model was asked to quote the words that show who speaks and to say what kind of
evidence it is, before naming the speaker. In about seven answers out of ten the "evidence" was the quotation
itself, and the kind was "turn-taking". As a filter it was useless. It does change the answers, though, and not
only for the better (see below).

**Scene notes.** A first call per chunk writes notes (the scenes, who is present under which names, who talks to
whom about what); the speaker question is then asked with the notes for the chunk before, this chunk and the
next. On ten chunks picked because they contained known errors, wrong answers fell from 22-24 to 6-10 of 177.
On two whole books it did not hold:

| "Who" question on 1,861 untagged quotations | Wrong gender | Fixed against plain | Broken against plain |
|---|---|---|---|
| Plain | 75 | | |
| With notes and evidence, run 1 | 56 | 46 | 26 |
| With notes and evidence, run 2 | 72 | 44 | 40 |

The notes help far from a speech tag (six or more paragraphs: 16 wrong became 5 and 6) and hurt next to one
(one paragraph away: 8 wrong became 14 and 18). Reading the 46 answers that broke: 17 took a neighbouring
quotation's speech tag, or narration about the other person, as this quotation's evidence (`"...," said Mrs.
Bennet, "..."` followed by Mr. Bennet's reply, attributed to her); 3 quoted the right evidence and concluded the
opposite; 11 followed from an earlier error in the same alternation; 7 had no right name to choose because the
speaker was missing from the list or the notes; 7 were in scenes with three or more people; 1 misread a
description used as a speech tag. More than half trace back to the request for evidence, which sends the model
to the nearest speech tag whether or not it belongs to the quotation.

Used as a second opinion instead (voice a quotation only when the plain answer and the notes answer agree) the
raw errors fell from 47 to 29 with slightly more quotations voiced, and put through the rest of the decision
step on those two books the final errors went from 21 to 15 and 16. Wired into the pipeline and measured on all
five books, with both variants deciding from one shared set of "who" answers:

| Second answer | Pride | Oliver Twist | Sun | Room | Styles | All five |
|---|---|---|---|---|---|---|
| "Which voice" (default) | 1 | 10 | 0 | 19 | 0 | 30 |
| "Who" again, with notes | 2 | 15 | 0 | 16 | 0 | 33 |

It was slightly better on the two books it had been developed on and worse on Oliver Twist, at 1.8 times the
model time. Two reasons, both the same weakness: asked "who" twice, the model picks the same wrong name twice
(Monks's lines went to "the relict", a description the table marks as female, both times, where the direct
voice question had said male); and when the chosen name has no known gender ("the surgeon" in that run) the
second answer carries no gender at all, where the direct question gives one. The code is in place behind
`voices.second: notes`.

**Cutting chunks at scene boundaries.** By default a chunk ends at the first paragraph boundary after 1,500
words, and across five books 104 of 262 such cuts fall between two paragraphs of dialogue, leaving the second
half of the conversation without its beginning. `chunk_mode: scene` waits for the first boundary that is not
between two paragraphs with quotations. On two books:

| | Words | Scene |
|---|---|---|
| Wrong, among the 117 quotations that had been in a split conversation | 16 | 3 |
| Wrong, all 1,859 untagged quotations ("who" question) | 69 | 55 |
| Final wrong-gender quotations | 20 | 20 |

The model's raw answers improve as predicted and the final count does not move: Pride and Prejudice lost four
errors and A Room with a View gained four. On Oliver Twist the final count was 10 against 1 in the
earlier five-book measurement (a different session, so not a controlled comparison), and eight of those are
one scene: with the old cut, two lines of it happened to fall at the start of the next chunk, were
answered "the man", and gave the scene rule a male anchor; with the scene cut they stay with the rest, are
answered "the relict" like the others, and the whole exchange turns female. The earlier result was right by
accident, but the count is what it is, so the default stays `words`.

Both findings point the same way: once the model's answer is roughly right, the remaining errors are decided by
the rules that spread one answer across a scene, and a better second opinion does not reach them.

### 6.5 Run-to-run variance

At `temperature: 0` with a fixed seed, the same prompt gives the same answer while the server stays up, and can
give a different one after it is restarted. One chunk, sent unchanged, came back with no wrong answers three
times in one session and four wrong answers three times in the next. Across sessions the "who" question on A
Room with a View had 31, 37 and 34 wrong answers. The gender table moves too, where it relies on the model: one
description was "male" in one session and "unknown" in the next.

Two consequences. A book's answers are cached, so a finished book does not change. And two settings can only be
compared within one server session, ideally from one shared set of answers; a difference of a few quotations
between sessions means nothing.

### 6.6 Known open issues

- A second quotation in a paragraph inherits the first one's speech tag even when it has its own.
- The "person addressed is not the speaker" check runs only in crowded exchanges.
- Three untagged paragraphs in a row given to one person are accepted, although that pattern is often wrong.
- A scene rule can still voice a quotation on which the two model answers disagree.
- A speech tag at the end of the previous paragraph is not seen.
- The first-person threshold costs about half the dialogue; at 0.7 the second first-person book would have had
  2 wrong lines, 1.35% of the narrator's lines voiced, and 89% voiced.
- First person is decided per book, so a story collection with one first-person story is treated as first
  person throughout.
- A book that marks dialogue without quotation marks gets no character voices at all.
- Names offered to the model include descriptions that belong to nobody in the scene.

## 7. Evaluation

The [Project Dialogism Novel Corpus](https://github.com/Priya22/project-dialogism-novel-corpus) (Vishnubhotla et
al. 2022, CC-BY 4.0) gives the speaker of every quotation in a set of novels. It is used to score results and
for nothing else: the pipeline does not read it, and no rule or prompt contains anything from it.
(`scripts/pdnc_overrides.py` is a separate tool for making a hand-corrected edition; see USAGE.md.)

- `readaloud eval <id> --pdnc data/pdnc/<Book> --tag <name>` aligns PDNC's quotations with the book's by text,
  tolerating small differences between editions (`to-night` and `tonight`), and writes a report with the
  metrics above, the confusions and every miss. Short quotations are matched only close to where they are
  expected, so one stray match cannot drag the rest out of line.
- `readaloud eval-compare a:tag b:tag` puts runs side by side and lists every quotation that changed.
- `data/pdnc_overrides/<Book>.yaml` holds small corrections to PDNC's character tables, each with its reason:
  a mother listed as an alias of her daughter, a family surname listed as one person's alias, a narrator's
  placeholder name.
- PDNC labels some speakers by an alias rather than the main name. Until that was handled, 605 quotations of one
  novel were silently left out of its score.
- "Extra quotations voiced" (ours, not in PDNC) looks like an error count and is not one: most are real dialogue
  that did not line up, or letters.

## 8. Speech synthesis

Kokoro runs on a laptop and is what the pipeline was developed against. Drama 3 gives the voices worth
listening to for ten hours. How to serve both, and throughput figures, are in [MODELS.md](MODELS.md).

- **Voices are designed, then frozen.** For each of the three voices Drama 3 reads a fixed passage with no
  reference, four candidates at a time, and the first candidate that fits is kept: a median pitch at most 115 Hz
  for the male voice, above 170 Hz for the female voice, 150 to 172 Hz for the narrator, plausible speed, and
  at least 15% away in pitch from every voice already chosen. The narrator is neutral by default because a
  female narrator sounded too much like the female character voice. The kept clip becomes the reference that
  every later request clones.
- **One request is one speaker**, because a request carries one reference. A sentence with narration and a
  quotation is two requests, joined without a pause.
- **Takes are never regenerated.** The key of a take is a hash of the text, voice, adapter, model version,
  parameters and seed. With batching the seed does not reproduce the audio, so the files on disk are the only
  record; a take whose file exists is adopted, not redone.
- **Runaways are retried.** The adapter gives each request room to read its text at five characters a second.
  Output that fills that room is treated as a runaway and retried with the next seed, twice. In one novel
  eleven lines needed it, all of them one- or two-word fragments left by a quotation split. In Oliver Twist
  three of 9,815 pieces failed on all three seeds, again short speech-tag fragments. A lone request with the same
  seed gives the same audio, so `regen --next-seed` moves such a line on to seeds not yet tried; the new take
  keeps its key, and the chapter is rebuilt because the chapter's fingerprint also holds each take's row.
- **Scale by sending more at once.** One stream is only a little faster than real time; sixty-four streams
  produce about thirteen seconds of audio per second. At sixty-four streams Pride and Prejudice (15.6 hours of
  audio) took 64 minutes of synthesis and Oliver Twist (20.2 hours) took 86.

Not built yet: a check that transcribes the result and flags lines read wrongly or cut short; forced alignment
for word-level highlighting; turning numbers and abbreviations into words before synthesis; delivery or emotion
tags.

## 9. Observability and error handling

- Every run, model call, metric and warning is recorded in the book's database; raw requests and answers are
  kept under `work/llm/`.
- Model failures are typed (answer cut off, not valid, server unreachable). A chunk that fails is rolled back,
  marked, and retried once at the end of the run, so one bad chunk does not cost the rest.
- An answer cut off at the length limit makes the pipeline split the request in half and ask again.
- Warnings have codes (USAGE.md lists them). A warning that needs a person stops the run with exit code 3
  instead of guessing.

## 10. The browser UI

`readaloud ui` is the command line in a page, for one person on one machine. The rules it keeps:

- **Nothing new is computed.** A view reads a book's database through a read-only connection, so it never waits
  on a running job. A change is made by the command line itself: a job is one or more `readaloud` commands in a
  process of its own, and its log shows them. The files a person may edit stay the source of truth; the page is an
  editor for them.
- **No dependency and no build step.** The server is Python's `http.server`, as the Kokoro service is; the page is
  one HTML file, one stylesheet and plain JavaScript. The documents are rendered by a small Markdown converter
  that covers what they use.
- **A job outlives the page.** Its state is two files in `books/<id>/ui/`: what was asked and how it ended, and
  everything it printed. Closing the browser, or the UI server, loses nothing; stages are idempotent, so "continue"
  is the same job started again. One job per book at a time, and changes that need the database wait for it.
- **Chapters limit the audio only.** A request for some chapters is two commands: the text stages for the whole
  book, then synth, assemble and package for those chapters. With `--docs` on the voices stage, a chapter's voices
  were decided from that chapter alone and a third of its quotations fell back to the narrator.
- **This machine only.** It listens on 127.0.0.1, answers only requests that name it in their Host header, and
  accepts a change only with a header the page's own script sends, which a form on another site cannot add. Book
  ids, file names and stage names from a request are checked against what exists before they reach a command.

The browser tests and the pictures in the documents are one thing: `scripts/ui_demo.py` holds the scenarios,
`tests/test_ui_browser.py` runs them and checks the files they leave, and the same script records them as GIFs.
Writing the scenarios as tests found what looking at the page had not: a field wiped by a redraw, a state file
written by two threads at once, and a scenario that, run on one chapter only, would have shown a voice change
on the wrong line.
