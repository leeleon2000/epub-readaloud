/* readaloud ui: the panels for looking at what the pipeline decided, and changing it.
   Each one edits the same file a person would edit by hand; the pipeline reads it on its next run. */
"use strict";

function busy(book) { return book.job && book.job.status === "running"; }
function busyNote(book) {
  return busy(book) ? h("div", { class: "help", "data-test": "busy-note" },
    "A job is running for this book. Changes here can be made when it has finished.") : null;
}

// ---- warnings ---------------------------------------------------------------------------------------------------------

async function viewWarnings(body, book, rest) {
  const all = rest[0] === "all";
  const data = await api("GET", `/api/books/${book.id}/warnings${all ? "?all=1" : ""}`);
  async function act(w, action, note) {
    const r = await attempt(() => api("POST", `/api/books/${book.id}/warnings/${w.id}`, { action, note }));
    if (r) { toast(action === "ack" ? "Acknowledged" : "Resolved"); route(true); }
  }
  const items = data.warnings.map((w) => {
    const note = h("input", { type: "text", placeholder: "a note (optional)", size: 28, "data-test": "warning-note" });
    return h("div", { class: "panel", "data-test": "warning", "data-code": w.code, "data-status": w.status },
      h("div", { class: "row" },
        h("span", { class: "chip " + (w.status === "open" ? (w.severity === "pause" ? "bad" : "warn") : "ok") }, w.code),
        h("span", { class: "grow muted small" }, [w.where, w.count > 1 ? `seen ${w.count} times` : "", w.created].filter(Boolean).join(" · ")),
        h("span", { class: "chip" }, w.status)),
      h("p", { style: "margin:.5rem 0" }, w.message || h("span", { class: "muted" }, "(no message)")),
      w.opening ? h("p", { class: "muted small", style: "margin:.2rem 0 .5rem" }, `Begins: “${w.opening}”`) : null,
      w.what ? h("div", { class: "help" }, h("b", {}, w.what), " ", w.todo, " ", helpLink("USAGE", "6-when-it-stops", "When it stops")) : null,
      w.status === "open" ? h("div", { class: "row", style: "margin-top:.6rem" },
        w.section ? h("a", { class: "btn", href: `#/book/${book.id}/sections/${w.section}`, "data-test": "warning-section" }, "Open the section") : null,
        h("span", { class: "grow" }), note,
        h("button", { onclick: () => act(w, "ack", note.value), "data-test": "warning-ack",
          title: "I have seen it; carry on without changing anything" }, "Acknowledge"),
        h("button", { onclick: () => act(w, "resolve", note.value), "data-test": "warning-resolve" }, "Resolved"))
        : (w.note || w.handled_by) ? h("p", { class: "muted small", style: "margin:.5rem 0 0" },
          `${w.handled_by || "someone"}: ${w.note || "(no note)"}`) : null);
  });
  put(body,
    busyNote(book),
    h("div", { class: "row", style: "margin-bottom:.8rem" },
      h("span", { class: "grow muted" }, data.open ? `${data.open} open warning${data.open > 1 ? "s" : ""}.`
        : "No open warnings. A run pauses when something needs a person; it shows up here."),
      h("a", { href: `#/book/${book.id}/warnings${all ? "" : "/all"}`, "data-test": "warnings-toggle" },
        all ? "Only open ones" : "Show handled ones too")),
    items);
}

// ---- sections ---------------------------------------------------------------------------------------------------------

async function viewSections(body, book, rest) {
  const id = book.id;
  const data = await api("GET", `/api/books/${id}/sections`);
  if (!data.sections.length) {
    put(body, h("div", { class: "help", "data-test": "sections-empty" },
      "No sections yet: run the book at least as far as the classify stage."));
    return;
  }
  const locked = busy(book);
  let open = rest[0] || null;
  const lines = h("div", { "data-test": "lines" });

  async function setCategory(sec, cat) {
    const r = await attempt(() => api("POST", `/api/books/${id}/sections`, { set: { [sec.id]: cat } }));
    if (r) { toast(`${sec.id}: ${cat}`); route(true); } else route(true);
  }
  async function showLines(sid) {
    open = sid;
    const d = await attempt(() => api("GET", `/api/books/${id}/sections/${sid}`));
    if (!d) return;
    async function toggle(s, box) {
      box.disabled = true;
      const r = await attempt(() => api("POST", `/api/books/${id}/skip`, box.checked ? { remove: [s.id] } : { add: [s.id] }));
      if (r) toast(box.checked ? "This line is read again" : "This line will not be read");
      showLines(sid);
    }
    const wholeSection = !d.sentences.some((s) => s.read) && !d.sentences.some((s) => s.skipped);
    put(lines, h("div", { class: "panel", "data-test": "lines-panel", "data-section": sid },
      h("div", { class: "row" }, h("h2", { class: "grow" }, `Lines of ${sid}`),
        h("button", { class: "link", onclick: () => { open = null; put(lines); } }, "Close")),
      h("p", { class: "help" }, wholeSection
        ? `Nothing in this section is read, because its category is “${d.category}”. Change the category above to read it.`
        : "Untick a line to keep it from being read: a printer’s imprint, a stray credit. The text in the book is not changed."),
      h("table", {}, h("tbody", {}, d.sentences.map((s) => {
        const box = h("input", { type: "checkbox", checked: s.read, disabled: locked || (!s.read && !s.skipped),
          "data-test": "line-read", "data-sentence": s.id, onchange: () => toggle(s, box) });
        return h("tr", { class: s.read ? "" : "sel" },
          h("td", {}, box), h("td", { class: "muted small" }, s.id),
          h("td", {}, s.kind === "heading" ? h("b", {}, s.text) : s.text,
            s.skipped ? h("span", { class: "chip warn", style: "margin-left:.5rem" }, "not read") : null));
      })))));
    lines.scrollIntoView({ block: "nearest" });
  }

  const rows = data.sections.map((sec) => {
    const pick = h("select", { "data-test": "section-category", "data-section": sec.id, disabled: locked,
      onchange: () => setCategory(sec, pick.value) },
      sec.category === "uncertain" ? h("option", { value: "uncertain", selected: true }, "uncertain: choose") : null,
      data.categories.map((c) => h("option", { value: c.id, selected: c.id === sec.category, title: c.help },
        c.id + (c.read ? "" : " (not read)"))));
    return h("tr", { "data-test": "section", "data-section": sec.id, class: sec.id === open ? "sel" : "" },
      h("td", { class: "num" }, sec.chapter),
      h("td", {}, h("div", {}, sec.title || sec.first || h("span", { class: "muted" }, "(no heading)")),
        sec.reason && sec.source !== "rule" ? h("div", { class: "muted small" }, sec.reason) : null),
      h("td", {}, pick),
      h("td", {}, h("span", { class: "chip " + (sec.source === "manual" ? "run" : sec.category === "uncertain" ? "warn" : "") },
        sec.source === "manual" ? "set by you" : sec.source === "model" ? "the model" : "a rule")),
      h("td", { class: "num" }, `${sec.spoken} / ${sec.sentences}`),
      h("td", {}, h("button", { class: "link", "data-test": "section-lines", onclick: () => showLines(sec.id) }, "Lines")));
  });
  put(body,
    busyNote(book),
    h("div", { class: "help", style: "margin-bottom:.8rem" },
      "Each section has a category. Story is searched for dialogue and read; front matter, notes and back matter are read by " +
      "the narrator; contents, captions and licence text are not read. What you set here is kept in ",
      h("code", {}, "sections.yaml"), " and always wins. ",
      helpLink("USAGE", "looking-at-what-it-decided-and-changing-it", "Changing what it decided")),
    h("div", { class: "panel" },
      h("table", { "data-test": "sections" },
        h("thead", {}, h("tr", {}, h("th", { class: "num" }, "Ch."), h("th", {}, "Section"), h("th", {}, "Category"),
          h("th", {}, "Decided by"), h("th", { class: "num" }, "Read / lines"), h("th", {}, ""))),
        h("tbody", {}, rows))),
    lines);
  if (open) await showLines(open);
}

// ---- voices -----------------------------------------------------------------------------------------------------------

const voiceFilter = {};   // per book: what the Voices tab is showing

async function viewVoices(body, book) {
  const id = book.id;
  const f = voiceFilter[id] || (voiceFilter[id] = { chapter: "", voice: "", q: "", review: false, changed: false, offset: 0 });
  const qs = new URLSearchParams();
  if (f.chapter) qs.set("chapter", f.chapter);
  if (f.voice) qs.set("voice", f.voice);
  if (f.q) qs.set("q", f.q);
  if (f.review) qs.set("review", "1");
  if (f.changed) qs.set("changed", "1");
  qs.set("offset", f.offset);
  const [data, names] = await Promise.all([api("GET", `/api/books/${id}/quotes?${qs}`), api("GET", `/api/books/${id}/genders`)]);
  if (!data.total) {
    put(body, h("div", { class: "help", "data-test": "voices-empty" },
      "No quotations yet: run the book at least as far as the voices stage."));
    return;
  }
  const again = (patch) => { Object.assign(f, { offset: 0 }, patch); route(true); };

  async function setVoice(q, v) {
    // choosing the voice the pipeline already gave takes a pending request back
    const r = await attempt(() => api("POST", `/api/books/${id}/quotes`, { set: { [q.id]: v === q.voice && !q.manual ? null : v } }));
    if (r) route(true);
  }
  async function apply() {
    const r = await attempt(() => api("POST", `/api/books/${id}/quotes/apply`, {}));
    if (!r) return;
    toast("Applying ...");
    const t = setInterval(async () => {
      const j = await api("GET", `/api/books/${id}/job`).catch(() => null);
      if (j && j.job && j.job.status !== "running") {
        clearInterval(t);
        toast(j.job.status === "ok" ? "Applied. Run the book again to remake the audio of those lines."
          : "The voices stage did not finish; see the Run tab.", j.job.status !== "ok");
        route(true);
      }
    }, 500);
    timers.push(t);
  }

  const sel = (test, value, options, on) => h("select", { "data-test": test, onchange: (e) => on(e.target.value) },
    options.map(([v, t]) => h("option", { value: v, selected: String(v) === String(value) }, t)));
  const search = h("input", { type: "text", placeholder: "words in the quotation", value: f.typed !== undefined ? f.typed : f.q,
    size: 24, "data-test": "voice-search", oninput: (e) => { f.typed = e.target.value; },
    onkeydown: (e) => { if (e.key === "Enter") again({ q: search.value }); } });
  const tick = (test, key, label) => h("label", { class: "row", style: "gap:.3rem;margin:0;color:var(--ink);font-size:.92rem" },
    h("input", { type: "checkbox", checked: f[key], "data-test": test, onchange: (e) => again({ [key]: e.target.checked }) }), label);

  const rows = data.quotes.map((q) => h("tr", { "data-test": "quote", "data-quote": q.id, "data-voice": q.voice,
    "data-pending": q.pending ? "1" : null },
    h("td", { class: "num muted small" }, q.chapter),
    h("td", {}, h("div", {}, `“${q.text}”`),
      h("div", { class: "muted small" }, [q.tag ? `tag: ${q.tag}` : "", q.who ? `the model: ${q.who}` : "", q.why,
        q.manual ? "set by you" : q.unsure ? "not sure enough, so the narrator reads it" : ""].filter(Boolean).join(" · "))),
    h("td", {}, h("div", { class: "seg" }, ["male", "female", "narrator"].map((v) =>
      h("button", { class: (q.pending ? q.asked === v : q.voice === v) ? "on " + v : "", "data-test": "voice-" + v,
        onclick: () => setVoice(q, v) }, v))),
      q.pending ? h("div", { class: "small", style: "color:var(--warn)" }, `now ${q.voice}; not applied yet`) : null)));

  const more = data.offset + data.limit < data.matching;
  const manual = names.manual;
  // what is being typed is kept in the tab's state, so a redraw (a filter, a finished job) does not wipe it
  const nameBox = h("input", { type: "text", placeholder: "a name, e.g. Mr. Collins", size: 22, "data-test": "gender-name",
    value: f.name || "", oninput: (e) => { f.name = e.target.value; } });
  const genderBox = h("select", { "data-test": "gender-value", onchange: (e) => { f.gender = e.target.value; } },
    ["male", "female", "unknown", "narrator"].map((g) => h("option", { value: g, selected: g === (f.gender || "male") }, g)));
  async function setGender(name, g) {
    if (!name.trim()) { toast("Type a name first", true); return; }
    const r = await attempt(() => api("POST", `/api/books/${id}/genders`, { set: { [name]: g } }));
    if (r) { f.name = ""; toast(g ? `${name}: ${g}` : `${name}: back to what the pipeline finds`); route(true); }
  }

  put(body,
    busyNote(book),
    h("div", { class: "help", style: "margin-bottom:.8rem" },
      "Every quotation is read by the male voice, the female voice or the narrator. A line the pipeline is not sure about stays " +
      "with the narrator. Change any of them here; your choices are kept in ", h("code", {}, "attribution_overrides.csv"),
      " and always win. ", helpLink("DESIGN", "6-who-reads-which-line", "How it decides")),
    h("div", { class: "panel" },
      h("div", { class: "row" },
        h("span", { class: "chip", style: "color:var(--male);border-color:var(--male)" }, `${data.counts.male.toLocaleString()} male`),
        h("span", { class: "chip", style: "color:var(--female);border-color:var(--female)" }, `${data.counts.female.toLocaleString()} female`),
        h("span", { class: "chip" }, `${data.counts.narrator.toLocaleString()} narrator`),
        h("span", { class: "grow" }),
        data.pending ? h("button", { class: "primary", "data-test": "apply-voices", disabled: busy(book), onclick: apply },
          `Apply ${data.pending} change${data.pending > 1 ? "s" : ""}`) : null),
      h("div", { class: "row", style: "margin-top:.8rem" },
        sel("voice-chapter", f.chapter, [["", "All chapters"]].concat(data.chapters.map((c) => [c, `Chapter ${c}`])), (v) => again({ chapter: v })),
        sel("voice-voice", f.voice, [["", "Any voice"], ["male", "male"], ["female", "female"], ["narrator", "narrator"]], (v) => again({ voice: v })),
        search, h("button", { onclick: () => again({ q: search.value }), "data-test": "voice-search-go" }, "Search"),
        tick("voice-review", "review", "Only the unsure ones"), tick("voice-changed", "changed", "Only my changes")),
      h("p", { class: "muted small", "data-test": "voice-count" },
        `${data.matching.toLocaleString()} of ${data.total.toLocaleString()} quotations` +
        (data.matching > data.limit ? `, showing ${data.offset + 1} to ${Math.min(data.matching, data.offset + data.limit)}` : "")),
      h("table", { "data-test": "quotes" }, h("tbody", {}, rows)),
      h("div", { class: "row", style: "margin-top:.7rem" },
        f.offset ? h("button", { onclick: () => { f.offset = Math.max(0, f.offset - data.limit); route(); }, "data-test": "voice-prev" }, "Earlier") : null,
        more ? h("button", { onclick: () => { f.offset += data.limit; route(); }, "data-test": "voice-next" }, "Later") : null)),
    h("div", { class: "panel", "data-test": "genders" },
      h("h2", {}, "Names"),
      h("p", { class: "help" }, "When a speech tag names someone (", h("i", {}, "said Mr. Collins"),
        "), this table gives the voice. Fix a name once here instead of line by line; it is kept in ",
        h("code", {}, "genders.yaml"), " and applies with the next “Apply”."),
      h("div", { class: "row" }, nameBox, genderBox,
        h("button", { "data-test": "gender-add", onclick: () => setGender(nameBox.value, genderBox.value) }, "Set")),
      manual.length ? h("table", { style: "margin-top:.6rem", "data-test": "gender-manual" }, h("tbody", {}, manual.map((m) =>
        h("tr", {}, h("td", {}, m.name), h("td", {}, m.gender), h("td", {}, h("span", { class: "chip run" }, "set by you")),
          h("td", {}, h("button", { class: "link", "data-test": "gender-remove", onclick: () => setGender(m.name, null) }, "Remove")))))) : null,
      names.auto.length ? h("details", { style: "margin-top:.7rem" },
        h("summary", {}, `What the pipeline found for ${names.auto.length} names`),
        h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Name"), h("th", {}, "Gender"), h("th", {}, "From"), h("th", { class: "num" }, "Used"))),
          h("tbody", {}, names.auto.map((a) => h("tr", {}, h("td", {}, a.name), h("td", {}, a.gender),
            h("td", { class: "muted small" }, a.source), h("td", { class: "num" }, a.used)))))) : null));
}

// ---- settings ---------------------------------------------------------------------------------------------------------

async function viewSettings(body, book) {
  const id = book.id;
  const data = await api("GET", `/api/books/${id}/settings`);
  const locked = busy(book);
  const inputs = {};
  const rows = data.fields.map((fl) => {
    let el;
    if (fl.type === "bool") {
      el = h("input", { type: "checkbox", checked: !!fl.value, disabled: locked });
    } else if (fl.type === "select") {
      el = h("select", { disabled: locked }, fl.options.map((o) => h("option", { value: o, selected: o === fl.value }, o)));
    } else {
      el = h("input", { type: fl.type === "number" ? "number" : "text", step: "any", value: fl.value === null ? "" : fl.value,
        size: 30, disabled: locked });
    }
    el.setAttribute("data-test", "setting");
    el.setAttribute("data-key", fl.key);
    inputs[fl.key] = [el, fl];
    return h("tr", {},
      h("td", { style: "width:15rem" }, h("b", {}, fl.label), h("div", { class: "muted small" }, h("code", {}, fl.key))),
      h("td", { style: "width:20rem" }, el,
        fl.own ? h("div", { class: "small", style: "color:var(--accent)" }, `this book's own value; the default is ${fl.default}`) : null),
      h("td", { class: "muted small" }, fl.help));
  });
  const read = ([el, fl]) => fl.type === "bool" ? el.checked : fl.type === "number" ? (el.value === "" ? null : Number(el.value)) : el.value;
  async function save() {
    const values = {};
    for (const [key, pair] of Object.entries(inputs)) {
      const v = read(pair);
      if (v !== pair[1].value) values[key] = v;
    }
    if (!Object.keys(values).length) { toast("Nothing changed"); return; }
    const r = await attempt(() => api("PUT", `/api/books/${id}/settings`, { values }));
    if (r) { toast("Saved"); route(true); }
  }
  const text = h("textarea", { rows: 8, style: "width:100%;font:13px ui-monospace,Menlo,monospace", "data-test": "settings-text", disabled: locked });
  text.value = data.text;
  async function saveText() {
    const r = await attempt(() => api("PUT", `/api/books/${id}/settings`, { text: text.value }));
    if (r) { toast("Saved"); route(true); }
  }
  put(body,
    busyNote(book),
    h("div", { class: "help", style: "margin-bottom:.8rem" },
      "These are this book's settings. Only what differs from the defaults is written, to ", h("code", {}, "config.yaml"),
      " in the book's folder; the defaults for every book are in the repository's ", h("code", {}, "config.yaml"), ". ",
      helpLink("MODELS", "", "Serving the models")),
    h("div", { class: "panel" },
      h("table", { "data-test": "settings" }, h("tbody", {}, rows)),
      h("div", { class: "row", style: "margin-top:.8rem" }, h("span", { class: "grow" }),
        h("button", { class: "primary", "data-test": "settings-save", disabled: locked, onclick: save }, "Save"))),
    h("div", { class: "panel" },
      h("details", { "data-test": "settings-advanced" },
        h("summary", {}, "The file itself, for settings that are not in the form"),
        h("p", { class: "muted small" }, data.file), text,
        h("div", { class: "row", style: "margin-top:.6rem" }, h("span", { class: "grow" }),
          h("button", { "data-test": "settings-text-save", disabled: locked, onclick: saveText }, "Save the file")))));
}


// ---- the three voices ---------------------------------------------------------------------------------------------------

async function viewCast(body, book) {
  const id = book.id;
  const data = await api("GET", `/api/books/${id}/cast`);
  if (!data.voices.length) {
    put(body, h("div", { class: "help", "data-test": "cast-empty" },
      "No voices yet: they are made by the cast stage, which needs the speech service. Run the book at least as far as “cast”."));
    return;
  }
  const locked = busy(book);
  async function act(v, action) {
    toast(action === "reseed" ? `Designing another ${v.label.toLowerCase()} ...` : "Saving ...");
    const r = await attempt(() => api("POST", `/api/books/${id}/cast/${encodeURIComponent(v.id)}`, { action }));
    if (r) { toast(r.message || (action === "freeze" ? `${v.label} is kept as it is` : `${v.label} can change again`)); route(true); }
  }
  const stamp = Date.now();     // a new sample must not be played from the browser's memory of the old one
  const cards = data.voices.map((v) => h("div", { class: "panel", "data-test": "voice-card", "data-voice": v.id },
    h("div", { class: "row" },
      h("h2", { class: "grow", style: "margin:0" }, v.label),
      v.frozen ? h("span", { class: "chip run", "data-test": "voice-frozen" }, "kept") : null),
    h("div", { class: "muted small", style: "margin:.2rem 0 .6rem" }, v.design || "", v.design ? " · " : "", h("code", {}, v.voice)),
    v.sample ? h("audio", { controls: true, preload: "none", style: "width:100%", "data-test": "voice-sample",
      src: `/api/books/${id}/cast/${encodeURIComponent(v.id)}/sample?t=${stamp}` })
      : h("div", { class: "muted small" }, "No sample yet."),
    h("div", { class: "row", style: "margin-top:.7rem" },
      h("span", { class: "grow" }),
      v.can_reseed ? h("button", { "data-test": "voice-reseed", disabled: locked || v.frozen, onclick: () => act(v, "reseed"),
        title: v.frozen ? "This voice is kept; let it change first" : "Design a different voice for this part" }, "Try another")
        : h("span", { class: "muted small" }, "Set in Settings (Kokoro narrator voice)"),
      h("button", { "data-test": "voice-freeze", disabled: locked, onclick: () => act(v, v.frozen ? "unfreeze" : "freeze") },
        v.frozen ? "Let it change" : "Keep this one"))));
  put(body,
    busyNote(book),
    h("div", { class: "help", style: "margin-bottom:.8rem" },
      "The whole book is read in these voices, so listen before spending hours of synthesis on them. Each sample reads the same " +
      "passage. “Try another” designs a different voice (it needs the speech service); “Keep this one” stops later runs from " +
      "changing it. They are kept in ", h("code", {}, data.file), ". ",
      helpLink("USAGE", "4-the-best-voices-drama-3", "Choosing voices")),
    h("div", { class: "grid" }, cards),
    h("p", { class: "muted small" }, `Sample text: “${data.sample_text}”`));
}

// ---- listen, sentence by sentence -----------------------------------------------------------------------------------------

async function viewListen(body, book, rest) {
  const id = book.id;
  const ready = book.chapters.filter((c) => c.assembled);
  if (!ready.length) {
    put(body, h("div", { class: "help", "data-test": "listen-empty" },
      "No chapter has audio yet: run the book (or the audio for one chapter) on the Run tab."));
    return;
  }
  const n = Number(rest[0]) && ready.some((c) => c.id === Number(rest[0])) ? Number(rest[0]) : ready[0].id;
  const [data, fl] = await Promise.all([api("GET", `/api/books/${id}/listen/${n}`), api("GET", `/api/books/${id}/flagged`)]);
  const locked = busy(book);
  const audio = h("audio", { controls: true, preload: "metadata", style: "width:100%", "data-test": "chapter-audio",
    src: `/api/books/${id}/audio/${n}?t=${Date.now()}` });
  const rows = {};
  let stopAt = null, current = null;

  function mark(sid) {
    if (current === sid) return;
    if (current && rows[current]) rows[current].classList.remove("now");
    current = sid;
    if (sid && rows[sid]) {
      rows[sid].classList.add("now");
      const r = rows[sid].getBoundingClientRect();
      if (r.top < 120 || r.bottom > window.innerHeight - 40) rows[sid].scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }
  audio.addEventListener("timeupdate", () => {
    const ms = audio.currentTime * 1000;
    if (stopAt !== null && ms >= stopAt) { audio.pause(); stopAt = null; }
    const s = data.sentences.find((x) => x.start !== null && ms >= x.start && ms < x.end);
    mark(s ? s.id : null);
  });
  function play(s, whole) {
    if (s.start === null) return;
    audio.currentTime = s.start / 1000;
    stopAt = whole ? null : s.end - 30;
    mark(s.id);
    audio.play().catch(() => {});
  }
  async function redo(s, nextSeed) {
    const pieces = s.pieces.map((p) => p.id);
    const r = await attempt(() => api("POST", `/api/books/${id}/redo`, { pieces, next_seed: nextSeed }));
    if (!r) return;
    toast("Making it again ...");
    const t = setInterval(async () => {
      const j = await api("GET", `/api/books/${id}/job`).catch(() => null);
      if (j && j.job && j.job.status !== "running") {
        clearInterval(t);
        toast(j.job.status === "ok" ? "Made again. Run the book once more to put it in the audiobook."
          : "It did not finish; see the Run tab.", j.job.status !== "ok");
        route(true);
      }
    }, 500);
    timers.push(t);
  }

  const list = h("div", { "data-test": "sentences" }, data.sentences.map((s) => {
    const bad = s.pieces.some((p) => p.flagged !== null && p.flagged !== undefined);
    const row = h("div", { class: "sent" + (bad ? " bad" : ""), "data-test": "sentence", "data-sentence": s.id },
      h("button", { class: "play", title: "Play this sentence", "data-test": "sentence-play", disabled: s.start === null,
        onclick: () => play(s, false) }, "▶"),
      h("div", { class: "grow" },
        h("div", { class: s.kind === "heading" ? "head" : "" }, s.pieces.length > 1 || (s.pieces[0] && s.pieces[0].quote)
          ? s.pieces.map((p) => h("span", { class: "v " + p.voice, title: `${p.voice}${p.quote ? "" : " (narration)"}` }, p.text, " "))
          : s.text),
        bad ? h("div", { class: "small", style: "color:var(--bad)" },
          "The speech model could not read this line; its audio was cut short. ", s.pieces.filter((p) => p.flagged).map((p) => p.flagged).join(" ")) : null),
      h("div", { class: "row", style: "gap:.3rem" },
        h("span", { class: "muted small" }, s.start === null ? "" : clock(s.start)),
        h("button", { class: "link small", "data-test": "sentence-redo", disabled: locked, onclick: () => redo(s, false),
          title: "Drop this sentence's audio and make it again" }, "Again"),
        h("button", { class: "link small", "data-test": "sentence-reseed", disabled: locked, onclick: () => redo(s, true),
          title: "Make it again from a different seed, for a line that came out wrong the same way twice" }, "Differently")));
    rows[s.id] = row;
    return row;
  }));

  put(body,
    busyNote(book),
    h("div", { class: "panel player" },
      h("div", { class: "row" },
        h("select", { "data-test": "listen-chapter", onchange: (e) => { location.hash = `#/book/${id}/listen/${e.target.value}`; } },
          ready.map((c) => h("option", { value: c.id, selected: c.id === n }, `${c.id}. ${c.heading || c.file}`))),
        h("span", { class: "grow muted small" }, `${data.sentences.length} sentences · ${clock(data.ms)}`),
        h("span", { class: "v male" }, "male"), h("span", { class: "v female" }, "female"), h("span", { class: "muted small" }, "narrator")),
      audio),
    fl.flagged.length ? h("div", { class: "panel", "data-test": "flagged" },
      h("h2", {}, `${fl.flagged.length} line${fl.flagged.length > 1 ? "s" : ""} the speech model could not read`),
      h("p", { class: "help" }, "Their audio was cut short so that the book could finish. Listen, then make them again. ",
        helpLink("USAGE", "4-the-best-voices-drama-3", "Runaway lines")),
      h("table", {}, h("tbody", {}, fl.flagged.map((f) => h("tr", {},
        h("td", { class: "num" }, f.chapter), h("td", {}, `“${f.text}”`), h("td", { class: "muted small" }, f.why),
        h("td", {}, h("a", { href: `#/book/${id}/listen/${f.chapter}` }, "Open"))))))) : null,
    h("div", { class: "help", style: "margin-bottom:.8rem" },
      "Press ▶ to hear one sentence; the player above plays the chapter and the sentence being read is marked, as a read-along " +
      "reader does. “Again” makes a sentence's audio once more; “Differently” does it from another seed. ",
      helpLink("USAGE", "4-the-best-voices-drama-3", "When a line goes wrong")),
    h("div", { class: "panel" }, list));
}

function clock(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return (m >= 60 ? Math.floor(m / 60) + ":" + String(m % 60).padStart(2, "0") : m) + ":" + String(s % 60).padStart(2, "0");
}

// the tabs follow the work: run, see what stopped it, what was decided, the voices, the result, the settings
TABS.push(
  ["warnings", "Warnings", viewWarnings, (b) => b.warnings ? { text: b.warnings, warn: true } : null],
  ["sections", "Sections", viewSections],
  ["voices", "Voices", viewVoices],
  ["cast", "Cast", viewCast],
  ["listen", "Listen", viewListen],
  ["settings", "Settings", viewSettings]);

// ---- do the model services answer? (a page of its own) ------------------------------------------------------------------

const SERVICE = { up: ["ok", "answering"], down: ["bad", "not answering"], wrong: ["warn", "answering, with another model"], off: ["", "not used"] };

PAGES.services = async (parts) => {
  const book = parts[0] || "";
  const [data, lib] = await Promise.all([api("GET", "/api/services" + (book ? `?book=${encodeURIComponent(book)}` : "")), api("GET", "/api/books")]);
  const pick = h("select", { "data-test": "services-book", onchange: (e) => { location.hash = "#/services" + (e.target.value ? "/" + e.target.value : ""); } },
    h("option", { value: "" }, "the defaults for every book"),
    lib.books.map((b) => h("option", { value: b.id, selected: b.id === book }, `${b.title} (${b.id})`)));
  const cards = data.services.map((s) => {
    const [cls, text] = SERVICE[s.state] || ["", s.state];
    return h("div", { class: "panel", "data-test": "service", "data-service": s.id, "data-state": s.state },
      h("div", { class: "row" }, h("h2", { class: "grow", style: "margin:0" }, s.name),
        h("span", { class: "chip " + cls, "data-test": "service-state" }, text)),
      h("div", { class: "muted small", style: "margin:.2rem 0 .6rem" }, s.role),
      h("div", {}, h("code", {}, s.endpoint || "(no address)")),
      h("p", { style: "margin:.5rem 0" }, s.detail, s.ms !== undefined ? h("span", { class: "muted small" }, ` · ${s.ms} ms`) : null),
      h("div", { class: "row" },
        s.used ? h("span", { class: "chip run" }, "these settings use it") : h("span", { class: "chip" }, "not needed by these settings"),
        h("span", { class: "grow" }), helpLink("MODELS", s.doc, "How to serve it")));
  });
  const needed = data.services.filter((s) => s.used && s.state !== "up");
  put($app,
    h("div", { class: "row" }, h("h1", { class: "grow" }, "Models"),
      h("button", { "data-test": "services-again", onclick: () => route(true) }, "Check again")),
    h("p", { class: "muted" }, "The pipeline loads no model itself: each one is a service it talks to. This page asks each address once."),
    h("div", { class: "row", style: "margin-bottom:1rem" }, h("label", { style: "margin:0" }, "Settings of"), pick,
      h("span", { class: "muted small" }, `speech: ${data.tts} · language model: ${data.llm}` + (data.multi_voice ? "" : " · one narrator"))),
    h("div", { class: "help", style: "margin-bottom:1rem", "data-test": "services-verdict" },
      needed.length ? `Not ready: ${needed.map((s) => s.name).join(" and ")} ${needed.length > 1 ? "do" : "does"} not answer. ` +
        "A run with these settings would stop there. To try everything with no model, choose “mock” on the Run tab."
        : "Every service these settings need is answering."),
    h("div", { class: "grid" }, cards));
};

// ---- scoring against known speakers --------------------------------------------------------------------------------------

async function viewEvaluate(body, book, rest) {
  const id = book.id;
  const data = await api("GET", `/api/books/${id}/eval`);
  const locked = busy(book);
  const out = h("div", { "data-test": "eval-output" });
  const chosen = new Set();
  const src = h("select", { "data-test": "eval-source" }, data.sources.map((s) => h("option", { value: s }, s)));
  const tag = h("input", { type: "text", placeholder: "a name, e.g. first", size: 14, "data-test": "eval-tag" });
  const chapters = h("input", { type: "text", placeholder: "all, or 1-13", size: 10, "data-test": "eval-chapters" });
  const note = h("input", { type: "text", placeholder: "a note (optional)", size: 26, "data-test": "eval-note" });

  async function run() {
    toast("Scoring ...");
    const r = await attempt(() => api("POST", `/api/books/${id}/eval`,
      { source: src.value, tag: tag.value.trim(), chapters: chapters.value.trim(), note: note.value.trim() }));
    if (r) { toast("Scored"); location.hash = `#/book/${id}/evaluate/${tag.value.trim()}`; route(true); }
  }
  async function show(t) {
    const r = await attempt(() => api("GET", `/api/books/${id}/eval/${encodeURIComponent(t)}`));
    if (r) put(out, h("div", { class: "panel prose", "data-test": "eval-report", "data-tag": t, html: r.html }));
  }
  async function compare() {
    const r = await attempt(() => api("POST", `/api/books/${id}/eval-compare`, { tags: [...chosen] }));
    if (r) put(out, h("div", { class: "panel prose", "data-test": "eval-compare", html: r.html }));
  }
  async function corrected() {
    const r = await attempt(() => api("POST", `/api/books/${id}/corrected`, { source: src.value }));
    if (r) { toast("Writing the known speakers into the corrections file ..."); location.hash = `#/book/${id}/run`; }
  }

  const cmp = h("button", { "data-test": "eval-compare-go", disabled: true, onclick: compare }, "Compare the ticked ones");
  const pct = (v) => v === null || v === undefined ? "–" : v + "%";
  const rows = data.results.map((r) => h("tr", { "data-test": "eval-result", "data-tag": r.tag },
    h("td", {}, h("input", { type: "checkbox", "data-test": "eval-pick", onchange: (e) => {
      if (e.target.checked) chosen.add(r.tag); else chosen.delete(r.tag);
      cmp.disabled = chosen.size < 2;
    } })),
    h("td", {}, h("b", {}, r.tag), r.note ? h("div", { class: "muted small" }, r.note) : null),
    h("td", { class: "muted small" }, r.source), h("td", { class: "num" }, r.quotes), h("td", { class: "num" }, pct(r.aligned_pct)),
    h("td", { class: "num", "data-test": "eval-wrong" }, pct(r.wrong_gender_pct)), h("td", { class: "num" }, pct(r.voiced_pct)),
    h("td", { class: "muted small" }, ago(r.modified)),
    h("td", {}, h("button", { class: "link", "data-test": "eval-open", onclick: () => show(r.tag) }, "Report"))));

  put(body,
    busyNote(book),
    h("div", { class: "help", style: "margin-bottom:.8rem" },
      "For a novel whose speakers are known, this measures how the voices were assigned: how many quotations are read in the wrong " +
      "gender, and how many get a male or female voice at all. The known speakers come from the Project Dialogism Novel Corpus " +
      "(PDNC, CC-BY 4.0); the edition must match its text closely. ", helpLink("USAGE", "7-measuring-it-on-a-book-with-known-speakers", "Measuring a book")),
    data.sources.length ? h("div", { class: "panel" },
      h("h2", {}, "Score the book as it is now"),
      h("div", { class: "row" },
        h("div", {}, h("label", {}, "Known speakers of"), src), h("div", {}, h("label", {}, "Name of this result"), tag),
        h("div", {}, h("label", {}, "Chapters"), chapters), h("div", {}, h("label", {}, "Note"), note),
        h("span", { class: "grow" }), h("button", { class: "primary", "data-test": "eval-run", disabled: locked, onclick: run }, "Score")))
      : h("div", { class: "panel", "data-test": "eval-none" }, h("h2", {}, "No known speakers here yet"),
        h("p", {}, "Fetch a novel's labels into ", h("code", {}, data.where), " (they are not part of this repository):"),
        h("pre", {}, h("code", {}, "scripts/fetch_pdnc.sh PrideAndPrejudice")),
        h("p", { class: "muted small" }, "PDNC covers a few dozen novels in English; its repository lists them.")),
    data.results.length ? h("div", { class: "panel" },
      h("div", { class: "row" }, h("h2", { class: "grow" }, "Results"), cmp),
      h("table", { "data-test": "eval-results" },
        h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "Result"), h("th", {}, "Against"), h("th", { class: "num" }, "Quotations"),
          h("th", { class: "num" }, "Lined up"), h("th", { class: "num" }, "Wrong gender"), h("th", { class: "num" }, "Given a voice"),
          h("th", {}, ""), h("th", {}, ""))),
        h("tbody", {}, rows)),
      h("p", { class: "muted small", style: "margin:.6rem 0 0" },
        "Compare two results only within one session of the language model server: its answers differ from one start to the next.")) : null,
    data.sources.length ? h("div", { class: "panel", "data-test": "corrected" },
      h("h2", {}, "A corrected edition"),
      h("p", { class: "help" }, "Writes the known speaker's voice of every quotation that lines up into ",
        h("code", {}, "attribution_overrides.csv"), ", then applies it. This is a different thing from the pipeline's own result: an " +
        "audiobook made this way should say so and credit PDNC. Rows you wrote yourself are kept and still win. ",
        helpLink("USAGE", "8-a-corrected-edition", "A corrected edition")),
      h("div", { class: "row" },
        data.corrected ? h("span", { class: "chip run", "data-test": "corrected-yes" }, "this book has PDNC's corrections") : null,
        h("span", { class: "grow" }),
        h("button", { "data-test": "corrected-go", disabled: locked, onclick: corrected }, `Use the known speakers of ${data.sources[0]}${data.sources.length > 1 ? " (as chosen above)" : ""}`))) : null,
    out);
  if (rest[0] && data.results.some((r) => r.tag === rest[0])) await show(rest[0]);
}

// ---- video files ---------------------------------------------------------------------------------------------------------

const videoForm = {};   // per book: what the Video tab's form holds

async function viewVideo(body, book) {
  const id = book.id;
  const f = videoForm[id] || (videoForm[id] = { how: "auto", parts: "", cut: [], max_hours: "11.5", part_name: "Part", edition: "", repo_url: "", no_video: false });
  const req = () => ({ max_hours: f.max_hours, part_name: f.part_name, edition: f.edition, repo_url: f.repo_url, no_video: f.no_video,
    parts: f.how === "parts" ? f.parts : "", cut: f.how === "cut" ? f.cut : [] });
  const data = await api("POST", `/api/books/${id}/video/plan`, req());
  const locked = busy(book);
  const files = h("div", { "data-test": "video-files" }, data.files.length ? h("div", { class: "panel" },
    h("h2", {}, "Files"),
    h("table", {}, h("tbody", {}, data.files.map((x) => h("tr", {},
      h("td", {}, h("a", { href: `/api/books/${id}/video/${encodeURIComponent(x.name)}`, download: x.name, "data-test": "video-file" }, x.name)),
      h("td", { class: "num muted small" }, bytes(x.bytes)), h("td", { class: "muted small" }, ago(x.modified)))))),
    h("p", { class: "muted small", style: "margin:.6rem 0 0" },
      "For each part: the video, a caption file with one caption per sentence, and a description with the chapter list to paste.")) : null);
  if (!data.ready) {
    put(body, h("div", { class: "help", "data-test": "video-not-ready" }, data.reason), files);
    return;
  }
  const again = (patch) => { Object.assign(f, patch); route(true); };
  const text = (key, test, size, ph) => h("input", { type: "text", value: f[key], size, placeholder: ph || "", "data-test": test,
    oninput: (e) => { f[key] = e.target.value; }, onchange: () => route(true) });
  const how = (v, label) => h("label", { class: "row", style: "gap:.35rem;margin:0;color:var(--ink);font-size:.95rem" },
    h("input", { type: "radio", name: "how", checked: f.how === v, "data-test": "video-how-" + v, onchange: () => again({ how: v }) }), label);
  const marks = data.marks.filter((m) => m.ms > 0);
  async function go() {
    const r = await attempt(() => api("POST", `/api/books/${id}/video`, req()));
    if (r) { toast("Making the files ..."); location.hash = `#/book/${id}/run`; }
  }
  put(body,
    busyNote(book),
    h("div", { class: "help", style: "margin-bottom:.8rem" },
      "A finished book as video files: the cover as a still picture over the audio, a description with chapter timestamps, and a caption " +
      "for every sentence. Nothing is uploaded. Before publishing, check the book's copyright where the video will be shown and the " +
      "speech model's licence, and say that the voices are synthetic. ", helpLink("USAGE", "9-a-video-with-chapters-and-captions", "Video export")),
    h("div", { class: "panel" },
      h("h2", {}, "Parts"),
      h("p", { class: "muted small" }, `The book is ${clock(data.total)} long, with ${data.marks.length} chapter headings. A part is cut at a heading.`),
      h("div", { class: "row", style: "gap:1.4rem" }, how("auto", "As few parts as the length allows"), how("parts", "A number of parts"),
        how("cut", "Start a part at chosen headings (the book's own volumes)")),
      f.how === "parts" ? h("div", { style: "margin-top:.6rem" }, h("label", {}, "How many"), text("parts", "video-parts", 4, "3")) : null,
      f.how === "cut" ? h("div", { style: "margin-top:.6rem", "data-test": "video-marks" },
        h("label", {}, "A new part starts at"),
        h("div", { style: "max-height:11rem;overflow:auto;border:1px solid var(--line);border-radius:7px;padding:.3rem .6rem" },
          marks.map((m) => h("label", { class: "row", style: "gap:.4rem;margin:.1rem 0;color:var(--ink);font-size:.92rem" },
            h("input", { type: "checkbox", checked: f.cut.includes(m.title), "data-test": "video-mark", "data-title": m.title,
              onchange: (e) => again({ cut: e.target.checked ? f.cut.concat([m.title]) : f.cut.filter((t) => t !== m.title) }) }),
            h("span", { class: "muted small", style: "width:4.5rem" }, clock(m.ms)), m.title)))) : null,
      h("div", { class: "row", style: "margin-top:.8rem" },
        h("div", {}, h("label", {}, "Longest part, hours"), text("max_hours", "video-max", 5)),
        h("div", {}, h("label", {}, "A part is called"), text("part_name", "video-name", 8))),
      data.error ? h("p", { style: "color:var(--bad)", "data-test": "video-error" }, data.error)
        : h("table", { style: "margin-top:.8rem", "data-test": "video-parts-table" },
          h("thead", {}, h("tr", {}, h("th", {}, f.part_name || "Part"), h("th", {}, "Starts with"), h("th", { class: "num" }, "Chapters"), h("th", { class: "num" }, "Length"))),
          h("tbody", {}, data.parts.map((p, i) => h("tr", { "data-test": "video-part" },
            h("td", {}, i + 1), h("td", {}, p.first), h("td", { class: "num" }, p.chapters), h("td", { class: "num" }, clock(p.ms))))))),
    h("div", { class: "panel" },
      h("h2", {}, "Description"),
      h("div", { class: "row" },
        h("div", {}, h("label", {}, "Where the text comes from"), text("edition", "video-edition", 34, "Project Gutenberg ebook no. 1342")),
        h("div", {}, h("label", {}, "Link to the tool (optional)"), text("repo_url", "video-repo", 34, "https://...")),
        h("label", { class: "row", style: "gap:.35rem;margin:1rem 0 0;color:var(--ink);font-size:.95rem" },
          h("input", { type: "checkbox", checked: f.no_video, "data-test": "video-novideo", onchange: (e) => { f.no_video = e.target.checked; } }),
          "Captions and descriptions only, no video"),
        h("span", { class: "grow" }),
        h("button", { class: "primary", "data-test": "video-go", disabled: locked || !!data.error, onclick: go }, "Make the files")),
      h("p", { class: "muted small", style: "margin:.6rem 0 0" }, "Encoding runs at about thirty times real time; the job's log is on the Run tab.")),
    files);
}

TABS.splice(TABS.findIndex((t) => t[0] === "settings"), 0, ["evaluate", "Evaluate", viewEvaluate], ["video", "Video", viewVideo]);
