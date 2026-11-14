"use strict";
// Small DOM helper: text is always set via textContent, never innerHTML.
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
};
const $ = (s, r = document) => r.querySelector(s);
const pct = (x) => `${Math.round(x * 100)}%`;
const nice = (s) => String(s).replace(/_/g, " ");
const HEAD_NAMES = {
  heart_abnormal: "Normal / abnormal", heart_murmur: "Murmur", heart_valve: "Valve disease pattern",
  lung_sound: "Adventitious sounds", lung_diagnosis: "Likely condition",
};

const passcode = () => { try { return sessionStorage.getItem("doctorPasscode") || ""; } catch { return ""; } };
const MAX_UPLOAD = 3.5 * 1024 * 1024; // host rejects bodies over ~4.5 MB

// Large recordings are re-encoded in the browser as 4 kHz mono 16-bit WAV (what the models use anyway).
async function shrink(blob) {
  if (blob.size <= MAX_UPLOAD) return blob;
  const AC = window.AudioContext || window.webkitAudioContext;
  const decoded = await new AC().decodeAudioData(await blob.arrayBuffer());
  const secs = Math.min(decoded.duration, 120);
  const off = new OfflineAudioContext(1, Math.ceil(secs * 4000), 4000);
  const src = off.createBufferSource();
  src.buffer = decoded; src.connect(off.destination); src.start();
  const pcm = (await off.startRendering()).getChannelData(0);
  const view = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const str = (o, t) => [...t].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); view.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVEfmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 4000, true); view.setUint32(28, 8000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, "data"); view.setUint32(40, pcm.length * 2, true);
  pcm.forEach((v, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 32767, true));
  return new Blob([view], { type: "audio/wav" });
}

async function api(path, opts = {}) {
  if (path.startsWith("/api/cases") && (opts.method || "GET") === "GET") opts.headers = { ...opts.headers, "x-doctor-passcode": passcode() };
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : "Invalid request") : `Error ${r.status}`);
  return body;
}

// ---------- tabs ----------
document.querySelectorAll("nav.tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("nav.tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== `tab-${b.dataset.tab}`; });
  if (b.dataset.tab === "inbox") loadCases();
}));

// ---------- analyse ----------
const chosen = { heart: null, lung: null }; // {blob, name, truth}
let lastAnalysis = null;

function setChosen(kind, value) {
  chosen[kind] = value;
  const slot = $(`.slot[data-kind="${kind}"]`);
  $(".chosen", slot).textContent = value ? `Ready: ${value.name}` : "No recording yet";
  $(".clear", slot).hidden = !value;
  $("#analyse").disabled = !chosen.heart && !chosen.lung;
}

async function loadSamples() {
  try {
    const samples = await api("/api/samples");
    for (const s of samples) {
      const sel = $(`.slot[data-kind="${s.kind}"] .sample`);
      sel.append(el("option", { value: s.id, text: `${nice(s.true_label)} · ${s.dataset} · ${nice(s.task)}` }));
    }
  } catch { /* samples are optional */ }
}

document.querySelectorAll(".slot").forEach((slot) => {
  const kind = slot.dataset.kind;
  $(".sample", slot).addEventListener("change", async (e) => {
    const id = e.target.value;
    if (!id) return setChosen(kind, null);
    const opt = e.target.selectedOptions[0].textContent;
    const blob = await (await fetch(`/api/samples/${encodeURIComponent(id)}/audio`)).blob();
    setChosen(kind, { blob, name: `sample (${opt})`, truth: opt.split(" · ")[0] });
  });
  $(".upload", slot).addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) { $(".sample", slot).value = ""; setChosen(kind, { blob: f, name: f.name }); }
  });
  $(".clear", slot).addEventListener("click", () => { $(".sample", slot).value = ""; setChosen(kind, null); });
  let rec = null;
  $(".rec", slot).addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    if (rec) { rec.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      const parts = [];
      rec = new MediaRecorder(stream);
      rec.ondataavailable = (ev) => parts.push(ev.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(parts, { type: rec.mimeType || "audio/webm" });
        rec = null; btn.textContent = "● Record"; btn.classList.remove("recording");
        $(".sample", slot).value = "";
        setChosen(kind, { blob, name: `recording (${Math.round(blob.size / 1024)} KB)` });
      };
      rec.start(); btn.textContent = "■ Stop"; btn.classList.add("recording");
    } catch (err) { $(".chosen", slot).textContent = "Microphone not available: " + err.message; }
  });
});

$("#analyse").addEventListener("click", async () => {
  const status = $("#analyse-status");
  status.textContent = "Analysing… (first request after idle can take ~30 s)";
  $("#analyse").disabled = true;
  try {
    const fd = new FormData();
    for (const k of ["heart", "lung"]) {
      if (!chosen[k]) continue;
      const blob = await shrink(chosen[k].blob).catch(() => chosen[k].blob);
      if (blob.size > MAX_UPLOAD) throw new Error("Recording is too large; keep it under 2 minutes.");
      fd.append(k, blob, chosen[k].name.replace(/[^\w.-]/g, "_") + (blob.type.includes("webm") ? ".webm" : blob.type.includes("wav") ? ".wav" : ""));
    }
    lastAnalysis = await api("/api/analyze/combined", { method: "POST", body: fd });
    status.textContent = "";
    renderResult(lastAnalysis);
  } catch (err) {
    status.textContent = err.message;
  } finally { $("#analyse").disabled = !chosen.heart && !chosen.lung; }
});

function renderHeads(kind, r) {
  const card = el("div", { class: "card" }, el("h2", { text: kind === "heart" ? "Heart" : "Lungs" }));
  if (chosen[kind]?.truth) card.append(el("p", { class: "truth muted", text: `Dataset label (ground truth): ${nice(chosen[kind].truth)}` }));
  for (const [name, h] of Object.entries(r.heads)) {
    card.append(el("div", { class: "head" },
      el("div", { class: "head-top" }, el("span", { text: HEAD_NAMES[name] || nice(name) }), el("span", { text: `${nice(h.label)} · ${pct(h.confidence)}` })),
      el("div", { class: `bar${h.low_confidence ? " low" : ""}`, role: "img", "aria-label": `confidence ${pct(h.confidence)}` },
        el("span", { style: `width:${pct(h.confidence)}` })),
      el("div", { class: "muted", text: `Trained on: ${h.trained_on}${h.low_confidence ? " · low confidence" : ""}` })));
  }
  const spec = el("div", { class: "spec" }, el("img", { src: `data:image/png;base64,${r.spectrogram_png_b64}`, alt: `${kind} spectrogram; red shading marks windows the model scored as abnormal` }));
  for (const w of r.window_scores) {
    if (w.score < 0.5) continue;
    spec.append(el("span", { class: "hl", style: `left:${(w.start / r.duration_s) * 100}%;width:${((w.end - w.start) / r.duration_s) * 100}%;opacity:${(w.score - 0.5) * 0.6}` }));
  }
  card.append(spec, el("audio", { controls: "", src: URL.createObjectURL(chosen[kind].blob) }));
  return card;
}

function renderResult(res) {
  const f = res.fusion;
  const out = $("#result");
  out.replaceChildren(
    el("div", { class: "card" },
      el("div", { class: "row between" }, el("h2", { text: "Result" }), el("span", { class: `pill ${f.status}`, text: f.status })),
      f.flags.map((x) => el("div", { class: "flag", text: x.message })),
      el("p", { class: "muted", text: res.disclaimer })),
    res.heart ? renderHeads("heart", res.heart) : null,
    res.lung ? renderHeads("lung", res.lung) : null,
    el("button", { class: "btn ghost block", type: "button", text: "Discuss in chat", onclick: () => $("#nav-chat").click() }),
    el("button", { class: "btn primary block", type: "button", text: "Send to doctor", onclick: sendAnalysisCase }));
  if (window.__lenis) window.__lenis.scrollTo(out, { offset: -80 }); else out.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function sendAnalysisCase(e) {
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    const f = lastAnalysis.fusion;
    await api("/api/cases", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      patient_name: $("#patient-name").value || "Anonymous",
      urgency: f.status === "abnormal" ? "soon" : "routine",
      summary: { symptoms: "", duration: "", history: "", ai_findings: f.summary, urgency: f.status === "abnormal" ? "soon" : "routine" },
      analysis: { fusion: f } }) });
    btn.textContent = "Sent ✓";
  } catch (err) { btn.textContent = err.message; btn.disabled = false; }
}

// ---------- chat ----------
const history = [];
const log = $("#chat-log");
const say = (cls, text) => { log.append(el("div", { class: `msg ${cls}`, text })); log.lastChild.scrollIntoView({ block: "end" }); };

$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  history.push({ role: "user", content: text });
  say("me", text);
  try {
    const role = document.querySelector('input[name="role"]:checked').value;
    const r = await api("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, history, analysis: lastAnalysis && { fusion: lastAnalysis.fusion, summary: lastAnalysis.fusion.summary } }) });
    history.push({ role: "assistant", content: r.reply });
    say(r.red_flag ? "bot alert" : "bot", r.reply);
  } catch (err) { say("bot alert", err.message); }
});

$("#send-summary").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  if (!history.length) { say("bot", "Tell me about your symptoms first."); return; }
  btn.disabled = true; btn.textContent = "Sending…";
  try {
    await api("/api/cases/from-chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      patient_name: $("#patient-name").value || "Anonymous", history,
      analysis: lastAnalysis && { fusion: lastAnalysis.fusion, summary: lastAnalysis.fusion.summary } }) });
    btn.textContent = "Sent to doctor ✓";
  } catch (err) { btn.textContent = err.message; btn.disabled = false; }
});

// ---------- inbox ----------
async function loadCases() {
  const list = $("#cases");
  try {
    const r = await fetch("/api/cases", { headers: { "x-doctor-passcode": passcode() } });
    $("#passcode-form").hidden = r.status !== 401;
    if (r.status === 401) { list.replaceChildren(el("li", { class: "muted", text: "Enter the doctor passcode to view cases." })); return; }
    if (!r.ok) throw new Error(`Error ${r.status}`);
    const cases = await r.json();
    list.replaceChildren(...(cases.length ? cases.map((c) => el("li", {}, el("button", { type: "button", onclick: () => showCase(c) },
      el("span", { class: `badge ${c.urgency}`, text: c.urgency }),
      el("strong", { text: c.patient_name }), el("br"),
      el("span", { class: "muted", text: `${new Date(c.created * 1000).toLocaleString()} · ${c.summary.ai_findings || c.summary.symptoms || "no findings"}`.slice(0, 140) }))))
      : [el("li", { class: "muted", text: "No cases yet." })]));
  } catch (err) { list.replaceChildren(el("li", { class: "muted", text: err.message })); }
}

function showCase(c) {
  const s = c.summary;
  $("#case-detail").replaceChildren(el("div", { class: "card" },
    el("div", { class: "row between" }, el("h2", { text: `Case #${c.id} · ${c.patient_name}` }), el("span", { class: `badge ${c.urgency}`, text: c.urgency })),
    el("dl", {}, ["symptoms", "duration", "history", "ai_findings"].flatMap((k) => [el("dt", { text: nice(k) }), el("dd", { text: s[k] || "—" })])),
    c.analysis?.fusion ? el("p", {}, el("span", { class: `pill ${c.analysis.fusion.status}`, text: c.analysis.fusion.status })) : null,
    c.transcript.length ? el("details", {}, el("summary", { text: "Transcript" }),
      c.transcript.map((m) => el("p", { class: "muted", text: `${m.role}: ${m.content}` }))) : null));
  if (window.__lenis) window.__lenis.scrollTo("#case-detail", { offset: -80 }); else $("#case-detail").scrollIntoView({ behavior: "smooth" });
}
$("#refresh").addEventListener("click", loadCases);
$("#passcode-form").addEventListener("submit", (e) => {
  e.preventDefault();
  try { sessionStorage.setItem("doctorPasscode", $("#passcode").value); } catch { /* private mode */ }
  loadCases();
});

loadSamples();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
