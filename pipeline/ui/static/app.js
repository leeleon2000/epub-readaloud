/* readaloud ui: one page of plain JavaScript. Every view asks the server for data and draws it; every change is a
   request the server turns into the same code the command line runs. */
"use strict";

// ---- helpers ----------------------------------------------------------------------------------------------------

async function api(method, path, body) {
  const opt = { method, headers: { "X-Requested-With": "readaloud" } };
  if (body instanceof Blob) {
    opt.body = body;
    opt.headers["Content-Type"] = "application/octet-stream";
  } else if (body !== undefined) {
    opt.body = JSON.stringify(body);
    opt.headers["Content-Type"] = "application/json";
  }
  const r = await fetch(path, opt);
  let data = {};
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error(data.error || `${r.status} ${r.statusText}`);
  return data;
}

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

/* Replace an element's children. The DOM's own replaceChildren() would print a missing piece as the word "null". */
function put(el, ...kids) {
  el.replaceChildren(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
  if (el.ready) el.ready();      // a panel that was drawn off the page goes on it now (viewBook)
  return el;
}

const $app = document.getElementById("app");
let timers = [];
function every(ms, fn) { const t = setInterval(fn, ms); timers.push(t); return t; }
function clearTimers() { timers.forEach(clearInterval); timers = []; }

function toast(msg, bad) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "show" + (bad ? " bad" : "");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { t.className = ""; }, bad ? 6000 : 2500);
}
async function attempt(fn) {
  try { return await fn(); } catch (e) { toast(e.message, true); return undefined; }
}

function bytes(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
  return Math.max(1, Math.round(n / 1e3)) + " kB";
}
function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  if (s < 5400) return Math.round(s / 60) + " min ago";
  if (s < 129600) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " days ago";
}
function took(a, b) {
  const s = Math.max(0, Math.round((b || Date.now() / 1000) - a));
  return s < 60 ? s + " s" : s < 3600 ? Math.floor(s / 60) + " min " + (s % 60) + " s" : Math.floor(s / 3600) + " h " + Math.floor((s % 3600) / 60) + " min";
}
const JOB = {
  running: ["run", "running"], ok: ["ok", "finished"], paused: ["warn", "paused by a warning"],
  failed: ["bad", "failed"], stopped: ["warn", "stopped"], ended: ["", "ended"],
};
function jobChip(job) {
  if (!job) return null;
  const [cls, text] = JOB[job.status] || ["", job.status];
  return h("span", { class: "chip " + cls, "data-test": "job-status", "data-status": job.status }, text);
}
function helpLink(doc, anchor, text) {
  return h("a", { href: `#/help/${doc}${anchor ? "/" + anchor : ""}` }, text || "Read more");
}

// ---- library ----------------------------------------------------------------------------------------------------

async function viewLibrary() {
  const data = await api("GET", "/api/books");
  const list = h("div", { class: "grid", "data-test": "books" });
  for (const b of data.books) {
    const pct = b.planned ? Math.round(100 * b.done / b.planned) : 0;
    list.append(h("a", { class: "panel card", href: `#/book/${b.id}`, "data-test": "book-card", "data-book": b.id },
      h("div", { class: "row" }, h("h2", { class: "grow" }, b.title), jobChip(b.job)),
      h("div", { class: "muted small" }, b.id),
      h("div", { class: "bar" }, h("i", { style: `width:${pct}%` })),
      h("div", { class: "row small muted" },
        h("span", { class: "grow" }, b.error ? b.error : `${b.done} of ${b.planned} stages done`),
        b.warnings ? h("span", { class: "chip warn" }, `${b.warnings} warning${b.warnings > 1 ? "s" : ""}`) : null,
        b.outputs.length ? h("span", { class: "chip ok" }, "audiobook ready") : null)));
  }

  const idBox = h("input", { type: "text", placeholder: "from the file name", "data-test": "new-id", size: 22 });
  const file = h("input", { type: "file", accept: ".epub,.azw3,.mobi", hidden: true, "data-test": "file",
    onchange: () => file.files[0] && upload(file.files[0]) });
  const drop = h("div", { class: "drop", "data-test": "drop" },
    h("div", {}, "Drop an EPUB here, or ",
      h("button", { class: "link", onclick: () => file.click() }, "choose a file"), "."),
    h("div", { class: "small" }, "DRM-free and in English. AZW3 and MOBI are converted with Calibre."));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
  });
  async function upload(f) {
    toast(`Adding ${f.name} ...`);
    const q = new URLSearchParams({ name: f.name });
    if (idBox.value.trim()) q.set("id", idBox.value.trim());
    const r = await attempt(() => api("POST", "/api/books?" + q, f));
    if (r) location.hash = `#/book/${r.id}`;
  }
  async function sample() {
    const r = await attempt(() => api("POST", "/api/books/sample", {}));
    if (r) location.hash = `#/book/${r.id}`;
  }

  put($app, 
    h("h1", {}, "Library"),
    h("p", { class: "muted" }, "Each book becomes an EPUB 3 audiobook with read-along highlighting. Books are kept in ",
      h("code", {}, data.root), "."),
    h("div", { class: "panel" },
      h("h2", {}, "Add a book"), drop, file,
      h("div", { class: "row", style: "margin-top:.7rem" },
        h("div", {}, h("label", {}, "Id (optional)"), idBox),
        h("span", { class: "grow" }),
        data.sample && !data.books.some((b) => b.id === "alice")
          ? h("button", { onclick: sample, "data-test": "add-sample" }, "Add the sample book (Alice)") : null)),
    data.books.length ? list : h("div", { class: "help", "data-test": "empty" },
      "No books yet. Add one above; the sample book runs with no models at all, to see every step first. ",
      helpLink("USAGE", "1-the-smallest-run-no-models", "The smallest run")));
  if (data.books.some((b) => b.job && b.job.status === "running")) every(3000, () => route(true));
}

// ---- one book -----------------------------------------------------------------------------------------------------

const TABS = [["run", "Run", viewRun]];   // later panels add themselves here

async function viewBook(id, tab, rest) {
  const book = await api("GET", `/api/books/${id}`);
  const tabs = h("div", { class: "tabs", "data-test": "tabs" });
  for (const [key, label, , badge] of TABS) {
    const n = badge ? badge(book) : null;
    tabs.append(h("a", { href: `#/book/${id}/${key}`, class: key === tab ? "on" : "", "data-test": "tab-" + key }, label,
      n ? h("span", { class: "n" + (n.warn ? " warn" : "") }, n.text) : null));
  }
  const body = h("div", { "data-test": "tab-body" });
  const head = h("div", { class: "row" },
    h("div", { class: "grow" }, h("h1", { "data-test": "book-title" }, book.title),
      h("div", { class: "muted small" }, `${book.id}` + (book.source_version ? ` · EPUB ${book.source_version} source` : ""))),
    jobChip(book.job));
  // The panel is drawn off the page and put on it when it has its data, so a redraw never shows an empty page.
  let shown = false;
  body.ready = () => { if (!shown) { shown = true; put($app, head, tabs, body); } };
  const view = (TABS.find((t) => t[0] === tab) || TABS[0])[2];
  await view(body, book, rest);
  body.ready();
}

function stageStrip(book) {
  return h("ol", { class: "stages", "data-test": "stages" }, book.stages.map((s) =>
    h("li", { class: "stage " + s.state, title: s.help, "data-stage": s.stage, "data-state": s.state },
      h("b", {}, s.stage), h("span", { class: "d" }, s.detail || s.help))));
}

async function viewRun(body, book) {
  const id = book.id;
  const running = book.job && book.job.status === "running";
  const cfg = book.config;
  const sel = (test, options, value) => h("select", { "data-test": test },
    options.map(([v, t]) => h("option", { value: v, selected: v === value }, t)));
  // the form shows what the last job was asked for, so a redraw (or a reload) does not put the defaults back
  const was = (flag) => {
    const a = (book.job && book.job.args) || [];
    const i = a.indexOf(flag);
    return i >= 0 ? a[i + 1] : "";
  };
  const tts = sel("run-tts", [["", `As configured (${cfg.tts})`], ["mock", "mock: a placeholder tone, no model"],
    ["kokoro", "kokoro: on this machine"], ["drama3", "drama3: Fish Audio Drama 3"]], was("--tts"));
  const llm = sel("run-llm", [["", `As configured (${cfg.llm})`], ["mock", "mock: no model"],
    ["openai_compat", "openai_compat: an OpenAI-compatible server"]], was("--llm"));
  const names = book.stages.map((s) => s.stage);
  const until = sel("run-until", names.map((n) => [n, n === "package" ? "the finished audiobook" : `stop after ${n}`]),
    names.includes(was("--until")) ? was("--until") : "package");
  const chapters = h("input", { type: "text", placeholder: "all", size: 14, "data-test": "run-docs", value: was("--docs"),
    title: "Make the audio for these chapters only: numbers from the table below, separated by commas. Empty = the whole book." });

  const log = h("pre", { class: "log", "data-test": "log", hidden: !book.job });
  const state = h("div", { class: "row small muted", "data-test": "job-line" });
  const start = h("button", { class: "primary", "data-test": "start", disabled: running, onclick: go },
    book.done && book.done < book.planned ? "Continue" : book.done ? "Run again" : "Start");
  const stop = h("button", { class: "danger", "data-test": "stop", hidden: !running, onclick: halt }, "Stop");

  async function go() {
    const req = { tts: tts.value || undefined, llm: llm.value || undefined, until: until.value,
      docs: chapters.value.trim() || undefined };
    const r = await attempt(() => api("POST", `/api/books/${id}/run`, req));
    if (r) { offset = 0; log.textContent = ""; route(true); }
  }
  async function halt() {
    stop.disabled = true;
    await attempt(() => api("POST", `/api/books/${id}/stop`, {}));
    route(true);
  }

  let offset = 0, last = book.job ? book.job.status : null;
  async function poll() {
    let r;
    try { r = await api("GET", `/api/books/${id}/job?offset=${offset}`); } catch (e) { return; }
    if (r.log) {
      const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
      log.append(r.log);
      if (stick) log.scrollTop = log.scrollHeight;
    }
    offset = r.offset;
    const j = r.job;
    if (j) {
      log.hidden = false;
      put(state, h("span", {}, `${j.label} · `, j.status === "running" ? `running for ${took(j.started)}`
        : `${(JOB[j.status] || ["", j.status])[1]} ${ago(j.ended || j.started)} after ${took(j.started, j.ended)}`),
        j.status === "paused" ? h("a", { href: `#/book/${id}/warnings` }, "See the warnings") : null);
      if (last === "running" && j.status !== "running") { last = j.status; route(true); return; }
      last = j.status;
    }
  }

  const outs = book.outputs.length ? h("div", { class: "panel", "data-test": "outputs" },
    h("h2", {}, "Audiobook"),
    book.outputs.map((o) => h("div", { class: "row" },
      h("a", { class: "btn primary", href: `/api/books/${id}/out/${encodeURIComponent(o.name)}`, download: o.name,
        "data-test": "download" }, "Download"),
      h("span", { class: "grow" }, h("code", {}, o.name)),
      h("span", { class: "muted small" }, `${bytes(o.bytes)} · ${ago(o.modified)}`))),
    h("p", { class: "help", style: "margin:.7rem 0 0" },
      "Open it in a reader that supports EPUB 3 Media Overlays. ", helpLink("USAGE", "5-reading-it", "Reading it"))) : null;

  const rows = book.chapters.map((c) => h("tr", {},
    h("td", { class: "num" }, c.id), h("td", {}, c.heading || h("span", { class: "muted" }, c.file)),
    h("td", { class: "num" }, c.sentences.toLocaleString()),
    h("td", { class: "num" }, c.pieces ? `${c.synthesised.toLocaleString()} / ${c.pieces.toLocaleString()}` : "–"),
    h("td", {}, c.assembled ? h("span", { class: "chip ok" }, "audio ready") : null)));

  put(body, 
    h("div", { class: "panel" },
      h("div", { class: "row" }, h("h2", { class: "grow" }, "Stages"),
        helpLink("USAGE", "3-three-voices-add-a-language-model", "What each stage does")),
      stageStrip(book)),
    h("div", { class: "panel", "data-test": "run-panel" },
      h("h2", {}, "Run"),
      h("div", { class: "row" },
        h("div", {}, h("label", {}, "Speech"), tts),
        h("div", {}, h("label", {}, "Language model"), llm),
        h("div", {}, h("label", {}, "Go as far as"), until),
        h("div", {}, h("label", {}, "Audio for chapters"), chapters),
        h("span", { class: "grow" }), stop, start),
      h("p", { class: "help", style: "margin:.8rem 0 0" },
        "Work that is already done is skipped, so after stopping, or an error, the same button carries on. " +
        "A job keeps running when this page is closed. ", helpLink("MODELS", "", "Serving the models")),
      h("p", { class: "help", style: "margin:.5rem 0 0", "data-test": "chapters-hint" },
        "Chapters limit the audio only. The text stages always cover the whole book, because which voice reads a line is " +
        "decided from all of its text: names and their genders are learnt across the book."),
      state, log),
    outs,
    book.chapters.length ? h("div", { class: "panel" },
      h("h2", {}, "Chapters"),
      h("table", { "data-test": "chapters" },
        h("thead", {}, h("tr", {}, h("th", { class: "num" }, "No."), h("th", {}, "Starts with"),
          h("th", { class: "num" }, "Sentences"), h("th", { class: "num" }, "Audio pieces"), h("th", {}, ""))),
        h("tbody", {}, rows))) : null);
  await poll();
  if (book.job) log.scrollTop = log.scrollHeight;
  every(1000, poll);
}

// ---- help -----------------------------------------------------------------------------------------------------------

async function viewHelp(docId, anchor) {
  const [list, doc] = await Promise.all([api("GET", "/api/docs"), api("GET", `/api/docs/${docId || "USAGE"}`)]);
  const nav = h("nav", { "data-test": "doc-nav" });
  for (const d of list.docs) {
    nav.append(h("a", { class: "t" + (d.id === doc.id ? " on" : ""), href: `#/help/${d.id}` }, d.title));
    if (d.id === doc.id) {
      for (const hd of doc.headings.filter((x) => x.level > 1)) {
        nav.append(h("a", { class: "l" + hd.level, href: `#/help/${doc.id}/${hd.id}` }, hd.text));
      }
    }
  }
  put($app, h("div", { class: "doc" }, nav, h("article", { class: "prose panel", html: doc.html, "data-test": "doc" })));
  const target = anchor && document.getElementById(anchor);
  if (target) target.scrollIntoView(); else window.scrollTo(0, 0);
}

// ---- routing --------------------------------------------------------------------------------------------------------

const PAGES = {};      // other top-level pages add themselves here: name -> async (parts) => {}
let routing = 0;

async function route(keepScroll) {
  const mine = ++routing;
  clearTimers();
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  const page = parts[0] || "library";
  document.querySelectorAll("header nav a").forEach((a) =>
    a.classList.toggle("on", a.dataset.nav === (page === "book" ? "library" : page)));
  const y = window.scrollY;
  try {
    if (page === "library") await viewLibrary();
    else if (page === "book" && parts[1]) await viewBook(parts[1], parts[2] || "run", parts.slice(3));
    else if (page === "help") await viewHelp(parts[1], parts[2]);
    else if (PAGES[page]) await PAGES[page](parts.slice(1));
    else location.hash = "#/";
  } catch (e) {
    if (mine !== routing) return;
    put($app, h("div", { class: "panel", "data-test": "error" }, h("h2", {}, "That did not work"),
      h("p", {}, e.message), h("a", { href: "#/" }, "Back to the library")));
  }
  if (keepScroll === true) window.scrollTo(0, y);
}

window.addEventListener("hashchange", () => route());
document.addEventListener("DOMContentLoaded", () => route());   // after the panel scripts have added their tabs
