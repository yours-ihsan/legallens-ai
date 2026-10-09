/* LegalLens AI — vanilla JS frontend (no build step, no external libs).
   Renders ONLY data returned by the backend: never fabricates scores,
   citations, evidence or answers. */
"use strict";

const $ = (id) => document.getElementById(id);
/* Escape for both text nodes and attribute contexts (quotes included). */
const escapeHtml = (s) =>
  String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const state = {
  docs: 0,
  busy: false,
  timers: [],
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch (_) { /* keep status */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ================= header status ================= */
async function refreshStatus() {
  try {
    const h = await api("/api/health");
    const el = $("sysStatus");
    el.classList.remove("down");
    el.classList.add("ok");
    el.title = h.model;
    $("sysLine1").textContent = "SYSTEM READY";
    $("sysLine2").textContent = `${h.documents} doc${h.documents === 1 ? "" : "s"} · ${h.chunks} chunks`;
    state.docs = h.documents;
    updateChecklist();
    setPipelineIdle();
  } catch (_) {
    const el = $("sysStatus");
    el.classList.remove("ok");
    el.classList.add("down");
    $("sysLine1").textContent = "SYSTEM OFFLINE";
    $("sysLine2").textContent = "backend unreachable";
  }
}

/* ================= grounding pipeline (header) ================= */
const PIPE_ORDER = ["document", "retrieve", "verify", "answer"];

function resetPipe() {
  document.querySelectorAll(".pipe-step").forEach((step) => {
    step.classList.remove("done", "active", "refused");
    const dot = step.querySelector(".pipe-dot");
    dot.textContent = dot.dataset.num;
  });
  document.querySelectorAll(".pipe-line").forEach((l) => l.classList.remove("done"));
}

function setPipeStep(name, cls) {
  const step = document.querySelector(`.pipe-step[data-step="${name}"]`);
  if (!step) return;
  step.classList.remove("done", "active", "refused");
  if (cls) {
    step.classList.add(cls);
    const dot = step.querySelector(".pipe-dot");
    if (cls === "done") dot.textContent = "\u2713";
    else if (cls === "refused") dot.textContent = "!";
    else dot.textContent = dot.dataset.num;
  }
}

/* idle: Document step reflects real indexing state; the rest are standby */
function setPipelineIdle() {
  resetPipe();
  setPipeStep("document", state.docs > 0 ? "done" : null);
}

/* running: steps before activeIdx are done, activeIdx pulses */
function setPipelineRun(activeIdx) {
  resetPipe();
  setPipeStep("document", "done");
  PIPE_ORDER.slice(1, activeIdx).forEach((n) => setPipeStep(n, "done"));
  if (activeIdx >= 1 && PIPE_ORDER[activeIdx]) setPipeStep(PIPE_ORDER[activeIdx], "active");
}

function setPipelineDone() {
  resetPipe();
  PIPE_ORDER.forEach((n) => setPipeStep(n, "done"));
}

function setPipelineRefused() {
  resetPipe();
  PIPE_ORDER.slice(0, 3).forEach((n) => setPipeStep(n, "done"));
  setPipeStep("answer", "refused");
}

/* ================= documents ================= */
async function loadDocs() {
  try {
    const docs = await api("/api/documents");
    renderDocs(docs);
    state.docs = docs.length;
    updateChecklist();
    setPipelineIdle();
  } catch (_) { /* backend not up yet */ }
}

function updateChecklist() {
  const li = $("chkIndexed");
  const indexed = state.docs > 0;
  li.classList.toggle("pending", !indexed);
  li.querySelector(".check").textContent = indexed ? "\u2713" : "\u2013";
}

function renderDocs(docs) {
  const list = $("docList");
  list.innerHTML = "";
  $("docCount").textContent = docs.length ? `${docs.length} indexed` : "";
  $("docEmpty").hidden = docs.length > 0;

  for (const d of docs) {
    const li = document.createElement("li");
    li.className = "doc-card";
    li.innerHTML = `
      <span class="doc-badge">PDF</span>
      <div class="doc-meta">
        <div class="doc-name" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</div>
        <div class="doc-sub">${d.pages} page${d.pages === 1 ? "" : "s"} · ${d.chunks} chunk${d.chunks === 1 ? "" : "s"}</div>
        <div class="doc-ready">✓ Ready for retrieval</div>
      </div>
      <button class="doc-del" title="Remove document" aria-label="Remove ${escapeHtml(d.filename)}">✕</button>`;

    li.querySelector(".doc-del").addEventListener("click", async (e) => {
      const id = e.currentTarget.dataset.id || d.doc_id;
      if (!confirm(`Remove "${d.filename}" and its indexed vectors?`)) return;
      try {
        await api(`/api/documents/${id}`, { method: "DELETE" });
        await loadDocs();
        await refreshStatus();
      } catch (err) {
        setUploadStatus(`Remove failed: ${err.message}`, "error");
      }
    });
    list.appendChild(li);
  }
}

/* ================= upload ================= */
function setUploadStatus(msg, kind) {
  const el = $("uploadStatus");
  el.hidden = !msg;
  el.textContent = msg || "";
  el.className = `upload-status${kind ? " " + kind : ""}`;
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setUploadStatus("Only PDF files are supported.", "error");
    return;
  }
  setUploadStatus(`Indexing ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    const info = await api("/api/documents", { method: "POST", body: form });
    setUploadStatus(`✓ ${info.filename}: ${info.pages} pages → ${info.chunks} chunks indexed.`, "ok");
    await loadDocs();
    await refreshStatus();
  } catch (err) {
    setUploadStatus(`Upload failed: ${err.message}`, "error");
  }
}

function wireUpload() {
  const dz = $("dropzone");
  const input = $("fileInput");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
  input.addEventListener("change", () => { uploadFile(input.files[0]); input.value = ""; });
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => uploadFile(e.dataTransfer.files[0]));
}

/* ================= messages ================= */
function addMessage(node) {
  const welcome = $("welcome");
  if (welcome) welcome.remove();
  const box = $("messages");
  box.appendChild(node);
  box.scrollTop = box.scrollHeight;
}

function clearTimers() {
  state.timers.forEach((t) => clearTimeout(t));
  state.timers = [];
}

/* ---- answer card (grounded success) ---- */
/* Which of the three-way decision branches produced a SUCCESSFUL answer. */
const DECISION_META = {
  grounded_answer: { cls: "answer", label: "ANSWER", title: "Decision: ANSWER — the evidence supports the question" },
  premise_correction: { cls: "correction", label: "CORRECTION", title: "Decision: CORRECT — a contradicted assumption was corrected with a citation" },
};

function renderAnswerCard(a) {
  const band = (a.confidence.band || "Low").toLowerCase();
  const pct = Math.round((a.confidence.score ?? 0) * 100);
  const dec = DECISION_META[a.decision] || DECISION_META.grounded_answer;

  const chips = a.citations.map((c) => `
    <button type="button" class="cite-chip" data-ev="${escapeHtml(c.evidence_id)}"
            title="${escapeHtml(c.quote || "")}">
      <span class="chip-id">${escapeHtml(c.evidence_id)}</span>
      <span class="chip-sep" aria-hidden="true">•</span>
      <span class="chip-page">Page ${c.page}</span>
      <span class="chip-file">${escapeHtml(c.filename)}</span>
    </button>`).join("");

  const premises = Array.isArray(a.premises) ? a.premises : [];
  const premiseSection = premises.length ? `
        <div class="ac-section">
          <div class="sec-label">Premise Check · computed from the evidence</div>
          <ul class="premise-list">
            ${premises.map((p) => `
            <li class="premise-item ${escapeHtml(String(p.verdict || "").toLowerCase())}">
              <span class="pv-tag">${escapeHtml(p.verdict || "")}</span>
              <span class="pa-text">“${escapeHtml(p.assumption || "")}”</span>
              ${p.evidence_id ? `<span class="pa-ev">${escapeHtml(p.evidence_id)}</span>` : ""}
            </li>`).join("")}
          </ul>
        </div>` : "";

  const card = document.createElement("div");
  card.className = "msg msg-answer";
  card.innerHTML = `
    <article class="answer-card">
      <header class="ac-head">
        <div class="ac-title">
          <span class="ac-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor"
                 stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2.4 19.2 5v5.5c0 4.35-3 7.45-7.2 8.85C7.8 17.95 4.8 14.85 4.8 10.5V5L12 2.4Z"/>
              <path d="m8.9 11.9 2.2 2.2 4-4.2"/>
            </svg>
          </span>
          Grounded Answer
        </div>
        <div class="ac-head-right">
          <span class="dec-badge ${dec.cls}" title="${escapeHtml(dec.title)}">${escapeHtml(dec.label)}</span>
          <span class="conf-badge ${band}">
            ${escapeHtml((a.confidence.band || "").toUpperCase())}
            <span class="cb-pct">${pct}%</span>
          </span>
        </div>
      </header>
      <div class="ac-body">
        <div class="ac-section">
          <div class="sec-label">Answer</div>
          <p class="ac-answer">${escapeHtml(a.answer)}</p>
        </div>
        ${premiseSection}
        <div class="ac-section">
          <div class="sec-label">Confidence</div>
          <div class="conf-meter">
            <div class="conf-track">
              <div class="conf-fill ${band}" style="width:${Math.max(0, Math.min(100, pct))}%"></div>
            </div>
            <div class="conf-stats">
              <span class="conf-chip">Combined <b>${pct}%</b></span>
              <span class="conf-chip">Retrieval <b>${Number(a.confidence.retrieval).toFixed(2)}</b></span>
              <span class="conf-chip">Model <b>${Number(a.confidence.model).toFixed(2)}</b></span>
            </div>
          </div>
        </div>
        <div class="ac-section">
          <div class="sec-label">Evidence Used · ${a.citations.length} source${a.citations.length === 1 ? "" : "s"}</div>
          <div class="cites">${chips}</div>
        </div>
        <div class="ac-section">
          <div class="sec-label">Why this answer is trusted</div>
          <div class="trust-flow">
            <div class="trust-step">
              <span class="ts-num">1</span>
              <div class="ts-body">
                <div class="ts-title">Retrieved evidence <span class="ts-check" aria-hidden="true">✓</span></div>
                <div class="ts-sub">${a.evidence.length} excerpt${a.evidence.length === 1 ? "" : "s"} · best score ${Number(a.confidence.retrieval).toFixed(2)}</div>
              </div>
            </div>
            <span class="tsep" aria-hidden="true">→</span>
            <div class="trust-step">
              <span class="ts-num">2</span>
              <div class="ts-body">
                <div class="ts-title">Evidence validation <span class="ts-check" aria-hidden="true">✓</span></div>
                <div class="ts-sub">${a.citations.length} citation${a.citations.length === 1 ? "" : "s"} verified against source pages</div>
              </div>
            </div>
            <span class="tsep" aria-hidden="true">→</span>
            <div class="trust-step">
              <span class="ts-num">3</span>
              <div class="ts-body">
                <div class="ts-title">Grounded response <span class="ts-check" aria-hidden="true">✓</span></div>
                <div class="ts-sub">${escapeHtml(a.confidence.band || "")} confidence · page-linked citations</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </article>`;

  card.querySelectorAll(".cite-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      card.querySelectorAll(".cite-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      highlightEvidence(chip.dataset.ev);
    });
  });

  addMessage(card);
}

/* ---- refusal card (insufficient evidence) ---- */
/* Accurate banner text: which stage refused? Labels come from backend refusal_stage. */
const REFUSAL_META = {
  model: { label: "Refused by model", sub: "The model declined to answer without sufficient support" },
  verifier: { label: "Removed by verifier", sub: "The draft answer failed source verification and was rejected" },
  retrieval: { label: "No supporting evidence", sub: "No relevant evidence was found in the indexed documents" },
  premise: { label: "Assumptions not found", sub: "Every assumption in the question was absent from the retrieved evidence" },
  system: { label: "Model error", sub: "The language model could not be reached" },
};

/* Which case of the three-way decision produced this refusal. */
const REFUSAL_DECISION = {
  refusal_low_evidence: "REFUSE — retrieval below the low floor",
  refusal_no_premise_support: "REFUSE — all assumptions NOT_FOUND",
  refusal_system: "REFUSE — model error",
  refusal_model: "REFUSE — model declined",
  refusal_verifier: "REFUSE — failed verification",
};

function refusalMeta(a) {
  return REFUSAL_META[a.refusal_stage] || { label: "Refused", sub: "The system refused rather than guess" };
}

function renderRefusalCard(a) {
  const card = document.createElement("div");
  card.className = "msg msg-refusal";
  const meta = refusalMeta(a);

  const reasonRow = a.reason
    ? `<div class="rc-meta-row"><span class="meta-tag">Reason</span><span class="meta-val">${escapeHtml(a.reason)}</span></div>`
    : "";
  const retrievalRow = a.confidence && typeof a.confidence.retrieval === "number"
    ? `<div class="rc-meta-row"><span class="meta-tag">Retrieval</span><span class="meta-val">best score ${Number(a.confidence.retrieval).toFixed(2)}</span></div>`
    : "";
  const decisionRow = a.decision && REFUSAL_DECISION[a.decision]
    ? `<div class="rc-meta-row"><span class="meta-tag">Decision</span><span class="meta-val">${escapeHtml(REFUSAL_DECISION[a.decision])}</span></div>`
    : "";
  const premises = Array.isArray(a.premises) ? a.premises : [];
  const premiseSection = premises.length ? `
        <div class="ac-section">
          <div class="sec-label">Premise Check · computed from the evidence</div>
          <ul class="premise-list">
            ${premises.map((p) => `
            <li class="premise-item ${escapeHtml(String(p.verdict || "").toLowerCase())}">
              <span class="pv-tag">${escapeHtml(p.verdict || "")}</span>
              <span class="pa-text">“${escapeHtml(p.assumption || "")}”</span>
              ${p.evidence_id ? `<span class="pa-ev">${escapeHtml(p.evidence_id)}</span>` : ""}
            </li>`).join("")}
          </ul>
        </div>` : "";

  card.innerHTML = `
    <article class="refusal-card">
      <header class="rc-head">
        <span class="rc-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
               stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2.4 19.2 5v5.5c0 4.35-3 7.45-7.2 8.85C7.8 17.95 4.8 14.85 4.8 10.5V5L12 2.4Z"/>
            <path d="m9 12 2.2 2.2L15.4 10"/>
          </svg>
        </span>
        <div class="rc-titles">
          <div class="rc-title">Insufficient Evidence</div>
          <div class="rc-sub">${escapeHtml(meta.sub)}</div>
        </div>
        <div class="rc-badges">
          <span class="conf-badge insufficient">INSUFFICIENT</span>
          <span class="rc-refused">${escapeHtml(meta.label)}</span>
        </div>
      </header>
      <div class="rc-body">
        <p class="rc-statement">${escapeHtml(a.answer)}</p>
        <p class="rc-explain">The provided documents do not contain enough evidence to answer this question.</p>
        <div class="rc-meta">
          ${decisionRow}
          ${retrievalRow}
          ${reasonRow}
        </div>
        ${premiseSection}
        <div class="rc-guard">
          <span class="rc-guard-check" aria-hidden="true">✓</span>
          Guardrail working as designed — nothing was guessed, and no source was misquoted.
        </div>
      </div>
    </article>`;

  addMessage(card);
}

/* ---- loading ---- */
function renderLoading() {
  const card = document.createElement("div");
  card.className = "msg";
  card.innerHTML = `
    <div class="loading-card">
      <div class="loader-top"><span class="spinner"></span><span id="loadTitle">Retrieving evidence…</span></div>
      <div class="load-steps">
        <span class="lstep active" data-ls="retrieve">Retrieve</span><span class="lsep">▸</span>
        <span class="lstep" data-ls="verify">Verify</span><span class="lsep">▸</span>
        <span class="lstep" data-ls="answer">Answer</span>
      </div>
      <div class="load-sub">Grounding your answer in retrieved evidence. This can take up to a minute.</div>
    </div>`;
  addMessage(card);
  return card;
}

function markLoadStep(card, name) {
  const order = ["retrieve", "verify", "answer"];
  const idx = order.indexOf(name);
  order.forEach((n, i) => {
    const el = card.querySelector(`.lstep[data-ls="${n}"]`);
    if (!el) return;
    el.classList.toggle("done", i < idx);
    el.classList.toggle("active", i === idx);
  });
  const map = { retrieve: 1, verify: 2, answer: 3 };
  setPipelineRun(map[name] ?? 1);
}

/* ---- error ---- */
function renderErrorCard(err) {
  const card = document.createElement("div");
  card.className = "msg";
  card.innerHTML = `
    <div class="error-card">
      <div class="err-title"><span>✕</span> Request failed</div>
      <p class="err-msg">${escapeHtml(err.message)}</p>
    </div>`;
  addMessage(card);
}

/* ================= evidence panel ================= */
function renderEvidencePanel(a) {
  const wrap = $("evidenceList");
  wrap.innerHTML = "";

  /* grounding status — real backend fields only */
  const badge = $("gsBadge");
  if (a.insufficient) {
    badge.className = "gs-badge refused";
    badge.textContent = refusalMeta(a).label.toUpperCase();
    $("gsDetail").textContent = a.reason || "Retrieved evidence did not meet the grounding threshold.";
  } else {
    badge.className = "gs-badge grounded";
    badge.textContent = "GROUNDED · SOURCES VERIFIED";
    $("gsDetail").textContent =
      `Answer built from ${a.evidence.length} retrieved excerpt${a.evidence.length === 1 ? "" : "s"} · ` +
      `${a.citations.length} cited.`;
  }
  $("gsScores").hidden = false;
  $("gsConf").textContent = a.insufficient ? "—" : `${Math.round((a.confidence.score ?? 0) * 100)}%`;
  $("gsRetr").textContent = Number(a.confidence.retrieval ?? 0).toFixed(2);
  $("gsModel").textContent = Number(a.confidence.model ?? 0).toFixed(2);

  /* evidence cards */
  if (!a.evidence.length) {
    wrap.innerHTML = `<div class="ev-empty"><div class="ev-empty-icon">◧</div>
      <p>No evidence was retrieved for this question.</p></div>`;
    return;
  }

  for (const ev of a.evidence) {
    const used = a.citations.some((c) => c.evidence_id === ev.id);
    const div = document.createElement("div");
    div.className = `ev-card${used ? " used" : ""}`;
    div.id = `ev-${ev.id}`;
    div.innerHTML = `
      <div class="ev-top">
        <span class="ev-id">${escapeHtml(ev.id)}</span>
        <span class="ev-file" title="${escapeHtml(ev.filename)}">${escapeHtml(ev.filename)}</span>
        <span class="ev-page">Page ${ev.page}</span>
        <span class="ev-score">score ${Number(ev.score).toFixed(2)}</span>
      </div>
      <div class="ev-text">${escapeHtml(ev.text)}</div>
      <div class="ev-foot">
        <span class="${used ? "ev-used-tag" : "ev-notused-tag"}">
          ${used ? "✓ Cited in answer" : "Retrieved · not cited"}
        </span>
      </div>`;
    wrap.appendChild(div);
  }
}

function highlightEvidence(evId) {
  const el = document.getElementById(`ev-${evId}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 2000);
}

/* ================= ask ================= */
async function ask(question) {
  const q = question.trim();
  if (!q || state.busy) return;
  state.busy = true;
  clearTimers();

  const user = document.createElement("div");
  user.className = "msg msg-user";
  user.innerHTML = `<div class="bubble">${escapeHtml(q)}</div>`;
  addMessage(user);

  const loading = renderLoading();
  markLoadStep(loading, "retrieve");
  $("askBtn").disabled = true;

  state.timers.push(setTimeout(() => {
    const t = $("loadTitle");
    if (t) t.textContent = "Verifying sources…";
    markLoadStep(loading, "verify");
  }, 3500));
  state.timers.push(setTimeout(() => {
    const t = $("loadTitle");
    if (t) t.textContent = "Generating grounded answer…";
    markLoadStep(loading, "answer");
  }, 9000));

  try {
    const answer = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    loading.remove();
    if (answer.insufficient) {
      renderRefusalCard(answer);
      setPipelineRefused();
    } else {
      renderAnswerCard(answer);
      setPipelineDone();
    }
    renderEvidencePanel(answer);
  } catch (err) {
    loading.remove();
    renderErrorCard(err);
    setPipelineIdle();
  } finally {
    clearTimers();
    state.busy = false;
    $("askBtn").disabled = false;
    $("question").focus();
  }
}

function wireChat() {
  $("askForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("question");
    const value = input.value;
    input.value = "";
    ask(value);
  });
}

/* ================= init ================= */
wireUpload();
wireChat();
setPipelineIdle();
loadDocs();
refreshStatus();
setInterval(refreshStatus, 15000);

/* ============================================================
   EVALUATION VIEW — Groundedness vs baseline
   Renders ONLY numbers returned by /api/evaluation*. When no run
   exists yet it shows "Not evaluated yet" — never a fake score.
   ============================================================ */
const evalState = { loaded: false, running: false, timer: null };

const VIEW_TABS = {
  chat: { view: "chatView", tab: "tabChat" },
  draft: { view: "draftView", tab: "tabDraft" },
  research: { view: "researchView", tab: "tabResearch" },
  eval: { view: "evalView", tab: "tabEval" },
};

function switchView(name) {
  for (const [key, ids] of Object.entries(VIEW_TABS)) {
    const on = key === name;
    $(ids.view).classList.toggle("active", on);
    $(ids.tab).classList.toggle("active", on);
    $(ids.tab).setAttribute("aria-selected", String(on));
  }
  if (name === "eval" && !evalState.loaded && !evalState.running) loadEvaluation();
  if (name === "draft") refreshDraftSource();
  if (name === "research") refreshResearchSource();
}

function setEvalStatus(kind, html) {
  const el = $("evalStatus");
  el.className = `eval-status ${kind}`;
  el.innerHTML = html;
}

const pctText = (v) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(1)}%`);
const numText = (v) => (v === null || v === undefined ? "—" : String(v));
const pctWidth = (v) =>
  v === null || v === undefined ? 0 : Math.max(0, Math.min(100, Number(v)));

async function loadEvaluation() {
  evalState.loaded = true;
  try {
    const data = await api("/api/evaluation");
    if (data.evaluated && data.results) {
      setEvalStatus("idle", "");
      renderEvaluation(data);
    } else {
      $("evalResults").hidden = true;
      setEvalStatus("idle",
        `<div class="eval-idle-note"><b>Not evaluated yet.</b> No evaluation run has ` +
        `been executed. Press <b>Run Groundedness Evaluation</b> to calculate real ` +
        `results from the current index — this dashboard never shows pre-filled numbers.</div>`);
    }
  } catch (err) {
    setEvalStatus("error", `Could not load evaluation status: ${escapeHtml(err.message)}`);
  }
}

function evalFlag(text, kind) {
  return `<span class="eq-flag ${kind}">${escapeHtml(text)}</span>`;
}

function claimCell(row) {
  if (row.kind === "abstention") return `<span class="ct abstain">– Abstention</span>`;
  if (row.grounded) return `<span class="ct ok">✓ Grounded</span>`;
  return `<span class="ct bad">✗ Unsupported</span>`;
}

function renderEvaluation(data) {
  const r = data.results;
  const b = r.baseline;
  const l = r.legallens;
  const c = r.comparison;
  const box = $("evalResults");

  const imp = c.groundedness_improvement;
  const impText = imp === null || imp === undefined
    ? "Not comparable (a system produced no factual claims)"
    : `${imp >= 0 ? "+" : ""}${Number(imp).toFixed(1)} percentage points`;
  const impCls = imp === null || imp === undefined ? "neutral" : imp > 0 ? "good" : imp < 0 ? "bad" : "neutral";

  const errNote = (b.errors || 0) + (l.errors || 0) > 0
    ? `<div class="eval-note warn">⚠ ${numText(b.errors + l.errors)} model call(s) failed during the run and were recorded as errors (0 claims).</div>`
    : "";

  const details = (r.questions || []).map((q) => {
    const rows = (r.details || []).filter(
      (d) => d.question_id === q.id && d.system === "baseline"
    ).concat((r.details || []).filter((d) => d.question_id === q.id && d.system === "legallens"));

    const rowsHtml = rows.map((d) => `
      <tr>
        <td class="ct-sys ct-sys-${escapeHtml(d.system)}">${d.system === "baseline" ? "Baseline" : "LegalLens"}</td>
        <td class="ct-claim">“${escapeHtml(d.claim)}”</td>
        <td>${claimCell(d)}</td>
        <td class="ct-ev">${d.evidence_ids && d.evidence_ids.length
          ? escapeHtml(d.evidence_ids.join(", ")) + (d.filename ? ` · ${escapeHtml(d.filename)}` : "")
          : (d.kind === "abstention" ? `<span class="ct dim">not applicable</span>` : `<span class="ct bad">No supporting evidence</span>`)}</td>
        <td class="ct-page">${d.page !== null && d.page !== undefined ? `Page ${escapeHtml(String(d.page))}` : "—"}</td>
      </tr>`).join("");

    const answers = `
      <div class="eq-answers">
        <div class="eq-ans"><span class="eq-ans-sys">Baseline answer</span>“${escapeHtml(baselineAnswer(q))}”</div>
        <div class="eq-ans"><span class="eq-ans-sys">LegalLens answer</span>“${escapeHtml(legallensAnswer(q))}”</div>
      </div>`;

    const flags = [
      q.answerable
        ? (q.legallens.retrieval && q.legallens.retrieval.success
            ? evalFlag("Evidence retrieved ✓", "ok")
            : evalFlag("Evidence missing ✗", "bad"))
        : evalFlag("Unanswerable by design", "warn"),
      q.baseline.refused ? evalFlag("Baseline: refused", "warn") : evalFlag("Baseline: answered", "plain"),
      q.legallens.refused
        ? evalFlag("LegalLens: refused", "warn")
        : evalFlag("LegalLens: answered", "ok"),
    ].join(" ");

    return `
      <details class="eval-q">
        <summary>
          <span class="eq-id">${escapeHtml(q.id)}</span>
          <span class="eq-q">${escapeHtml(q.question)}</span>
          <span class="eq-flags">${flags}</span>
        </summary>
        ${answers}
        <table class="claim-table">
          <thead><tr><th>System</th><th>Claim</th><th>Grounded?</th><th>Evidence</th><th>Page</th></tr></thead>
          <tbody>${rowsHtml || `<tr><td colspan="5" class="ct dim">No factual claims were extracted for this question (refusal / abstention only).</td></tr>`}</tbody>
        </table>
      </details>`;
  }).join("");

  box.innerHTML = `
    <div class="eval-ran">Run at ${escapeHtml(data.ran_at || "—")} · dataset ${numText(r.dataset_size)} questions
      (${numText(r.baseline.answerable_questions + r.baseline.unanswerable_questions)} total, 2 systems each)</div>
    ${errNote}

    <div class="eval-section-title">Grounding Evaluation</div>
    <table class="eval-table">
      <thead>
        <tr><th>Metric</th><th>Baseline</th><th>LegalLens</th></tr>
      </thead>
      <tbody>
        <tr><td>Groundedness</td><td>${pctText(b.groundedness)}</td><td>${pctText(l.groundedness)}</td></tr>
        <tr><td>Retrieval Success</td><td>${pctText(b.retrieval_success)}</td><td>${pctText(l.retrieval_success)}</td></tr>
        <tr><td>Unsupported Claims</td><td>${numText(b.unsupported_claims)}</td><td>${numText(l.unsupported_claims)}</td></tr>
        <tr><td>Correct Refusals</td>
            <td>${numText(b.correct_refusals)} / ${numText(b.unanswerable_questions)}</td>
            <td>${numText(l.correct_refusals)} / ${numText(l.unanswerable_questions)}</td></tr>
      </tbody>
    </table>

    <div class="eval-imp ${impCls}">
      <span class="eval-imp-label">Groundedness Improvement:</span>
      <span class="eval-imp-value">${escapeHtml(impText)}</span>
    </div>

    <div class="eval-section-title">Groundedness</div>
    <div class="eval-bars">
      <div class="eval-bar-row">
        <span class="eval-bar-label">Baseline</span>
        <div class="eval-bar-track"><div class="eval-bar-fill baseline" style="width:${pctWidth(b.groundedness)}%"></div></div>
        <span class="eval-bar-num">${pctText(b.groundedness)}</span>
      </div>
      <div class="eval-bar-row">
        <span class="eval-bar-label">LegalLens</span>
        <div class="eval-bar-track"><div class="eval-bar-fill legallens" style="width:${pctWidth(l.groundedness)}%"></div></div>
        <span class="eval-bar-num">${pctText(l.groundedness)}</span>
      </div>
    </div>

    <div class="eval-grid2">
      <div class="eval-card">
        <div class="eval-card-title">Fabricated / unsupported claims</div>
        <div class="eval-vs">
          <div><span class="evs-sys">Baseline</span><span class="evs-num bad">${numText(b.unsupported_claims)}</span></div>
          <div><span class="evs-sys">LegalLens</span><span class="evs-num ${l.unsupported_claims > 0 ? "warn" : "good"}">${numText(l.unsupported_claims)}</span></div>
        </div>
      </div>
      <div class="eval-card">
        <div class="eval-card-title">Correct refusals</div>
        <div class="eval-vs">
          <div><span class="evs-sys">Baseline</span><span class="evs-num">${numText(b.correct_refusals)} / ${numText(b.unanswerable_questions)}</span></div>
          <div><span class="evs-sys">LegalLens</span><span class="evs-num ${l.correct_refusals === l.unanswerable_questions ? "good" : "warn"}">${numText(l.correct_refusals)} / ${numText(l.unanswerable_questions)}</span></div>
        </div>
      </div>
    </div>

    <div class="eval-section-title">Claims Evaluated</div>
    <div class="eval-grid3">
      <div class="eval-stat"><span class="es-num">${numText(b.total_claims + l.total_claims)}</span><span class="es-lab">Claims Evaluated</span></div>
      <div class="eval-stat"><span class="es-num">${numText(b.supported_claims + l.supported_claims)}</span><span class="es-lab">Supported Claims</span></div>
      <div class="eval-stat"><span class="es-num">${numText(b.unsupported_claims + l.unsupported_claims)}</span><span class="es-lab">Unsupported Claims</span></div>
    </div>

    <div class="eval-section-title">Claim-level details</div>
    <div class="eval-note">Each percentage is traceable here: expand a question to see every
      extracted claim, whether it was grounded, and the exact evidence and page.</div>
    <div class="eval-details">${details}</div>`;

  box.hidden = false;
}

function baselineAnswer(q) { return (q.baseline && q.baseline.answer) || "(no answer)"; }
function legallensAnswer(q) { return (q.legallens && q.legallens.answer) || "(no answer)"; }

async function runEvaluation() {
  if (evalState.running) return;
  evalState.running = true;
  const btn = $("evalRunBtn");
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Running…";

  const started = Date.now();
  const tick = () => {
    const s = Math.round((Date.now() - started) / 1000);
    setEvalStatus("running",
      `<div class="eval-loading"><span class="spinner"></span>
         <div class="eval-loading-text"><b>Running groundedness evaluation…</b>
         <span>${s}s elapsed · baseline retrieve→generate and LegalLens retrieve→verify→answer for every dataset question. This can take several minutes.</span></div>
       </div>`);
  };
  tick();
  evalState.timer = setInterval(tick, 1000);

  try {
    const data = await api("/api/evaluation/run", { method: "POST" });
    evalState.loaded = true;
    setEvalStatus("done",
      `<div class="eval-done">✓ Evaluation completed at ${escapeHtml(data.ran_at || "—")} — all numbers below were calculated from this run.</div>`);
    renderEvaluation(data);
  } catch (err) {
    setEvalStatus("error", `Evaluation failed: ${escapeHtml(err.message)}`);
  } finally {
    clearInterval(evalState.timer);
    evalState.timer = null;
    evalState.running = false;
    btn.disabled = false;
    btn.textContent = label;
  }
}

$("tabChat").addEventListener("click", () => switchView("chat"));
$("tabEval").addEventListener("click", () => switchView("eval"));
$("evalRunBtn").addEventListener("click", runEvaluation);

/* ============================================================
   LEGAL DRAFTING VIEW — EVIDENCE FIRST → DRAFT SECOND
   Renders ONLY data returned by POST /api/draft: draft text,
   sources (evidence ID + filename + page from metadata) and the
   server-computed missing-information list. Nothing is invented.
   ============================================================ */
const DRAFT_BADGES = {
  grounded: { cls: "ok", text: "🟢 GROUNDED DRAFT" },
  missing: { cls: "warn", text: "🟠 DRAFT WITH MISSING INFORMATION" },
  insufficient: { cls: "bad", text: "🔴 INSUFFICIENT EVIDENCE" },
};

async function refreshDraftSource() {
  const el = $("draftSource");
  try {
    const docs = await api("/api/documents");
    if (!docs.length) {
      el.className = "draft-source empty";
      el.textContent = "No documents in the workspace — upload a PDF first.";
      return;
    }
    el.className = "draft-source";
    el.innerHTML =
      `<b>${docs.length} document${docs.length === 1 ? "" : "s"} indexed:</b> ` +
      docs.map((d) => escapeHtml(d.filename)).join(" · ");
  } catch (_) {
    el.className = "draft-source empty";
    el.textContent = "Could not load the workspace.";
  }
}

function renderDraftSources(sources) {
  const box = $("draftSources");
  if (!sources || !sources.length) {
    box.innerHTML = `<div class="ev-empty"><p class="muted">No supporting evidence attached.</p></div>`;
    return;
  }
  box.innerHTML = sources
    .map(
      (s) => `
      <div class="src-card">
        <div class="src-head">
          <span class="src-id">${escapeHtml(s.evidence_id)}</span>
          <span class="src-page">Page ${escapeHtml(String(s.page))}</span>
        </div>
        <div class="src-file">${escapeHtml(s.filename)}</div>
        <div class="src-snippet">“${escapeHtml(s.snippet)}”</div>
      </div>`
    )
    .join("");
}

function renderDraftMissing(missing, available) {
  const list = $("draftMissing");
  if (!missing || !missing.length) {
    list.innerHTML = `<li class="dm-none">✓ All required information for this draft type was found in the source.</li>`;
  } else {
    list.innerHTML = missing
      .map((m) => `<li class="dm-item">⚠ ${escapeHtml(m)} — not found in the document.</li>`)
      .join("");
  }
  const wrap = $("draftAvailWrap");
  const avail = $("draftAvailable");
  if (available && available.length) {
    wrap.hidden = false;
    avail.innerHTML = available
      .map(
        (a) => `<li class="da-item">✓ ${escapeHtml(a.label)}
          <span class="da-ref">${escapeHtml(a.evidence_id || "")} · Page ${escapeHtml(
          String(a.page ?? "—")
        )}</span></li>`
      )
      .join("");
  } else {
    wrap.hidden = true;
    avail.innerHTML = "";
  }
}

function renderClaimTrace(elId, data) {
  const check = data.claim_check || {};
  const corrections = data.citation_corrections || [];
  const removed = data.removed_claims || [];
  const el = $(elId);
  if (!check.claims_total && !corrections.length && !removed.length) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  const bad = check.unverified_claims || [];
  let html = check.claims_total
    ? `<b>Claim traceability:</b> ${check.claims_total} factual claim(s) checked · ` +
      `${check.claims_supported} supported · ${check.claims_unverified} not traceable` +
      (bad.length
        ? `<ul class="dc-unverified">${bad
            .map((c) => `<li>✗ “${escapeHtml(c.claim)}” <span>${escapeHtml(c.reason)}</span></li>`)
            .join("")}</ul>`
        : "")
    : "";
  if (corrections.length) {
    html += `<div class="dc-corrected"><b>✓ ${corrections.length} citation(s) verified and corrected:</b>` +
      `<ul class="dc-fixes">${corrections
        .map((c) => `<li>${escapeHtml(c)}</li>`)
        .join("")}</ul></div>`;
  }
  if (removed.length) {
    html += `<div class="dc-removed"><b>✗ ${removed.length} claim(s) removed — no ` +
      `supporting evidence in the source:</b>` +
      `<ul class="dc-unverified">${removed
        .map((c) => `<li>“${escapeHtml(c)}” — replaced with [NOT PROVIDED]</li>`)
        .join("")}</ul></div>`;
  }
  el.innerHTML = html;
}

function renderDraftResult(data) {
  const badge = DRAFT_BADGES[data.status] || DRAFT_BADGES.insufficient;
  const b = $("draftBadge");
  b.className = `draft-badge ${badge.cls}`;
  b.textContent = badge.text;

  const label = ($("draftType").selectedOptions[0] || {}).textContent || data.draft_type;
  $("draftMeta").textContent =
    label + (data.reason ? ` · ${data.reason}` : "") +
    (data.refusal_stage ? ` · stage: ${data.refusal_stage}` : "");

  $("draftBody").textContent = data.draft || "(no draft produced)";
  renderDraftSources(data.sources);
  renderDraftMissing(data.missing_information, data.available_information);
  renderClaimTrace("draftClaims", data);
  $("draftResult").hidden = false;
}

const draftState = { busy: false };

async function generateDraft(ev) {
  ev.preventDefault();
  if (draftState.busy) return;
  draftState.busy = true;
  const btn = $("draftBtn");
  btn.disabled = true;

  setEvalStatusOn("draftStatus", "running",
    `<div class="eval-loading"><span class="spinner"></span>
       <div class="eval-loading-text"><b>Generating grounded draft…</b>
       <span>Retrieving evidence, checking required information, then drafting from verified facts only.</span></div>
     </div>`);
  $("draftResult").hidden = true;

  try {
    const data = await api("/api/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        draft_type: $("draftType").value,
        instructions: $("draftInstr").value || "",
      }),
    });
    setEvalStatusOn("draftStatus", "idle", "");
    renderDraftResult(data);
  } catch (err) {
    setEvalStatusOn("draftStatus", "error",
      `<div class="eval-idle-note">Drafting failed: ${escapeHtml(err.message)}</div>`);
  } finally {
    draftState.busy = false;
    btn.disabled = false;
  }
}

function setEvalStatusOn(id, kind, html) {
  const el = $(id);
  el.className = `eval-status ${kind}`;
  el.innerHTML = html;
}

$("tabDraft").addEventListener("click", () => switchView("draft"));
$("draftForm").addEventListener("submit", generateDraft);
refreshDraftSource();

/* ============================================================
   LEGAL RESEARCH VIEW
   Renders ONLY what /api/research returns: status badge, legal
   issue, answer, verified law / judgments, law-to-case
   connections, sources with ✓ / ⚠ verification flags, missing ·
   limitations and claim traceability. No score is invented here.
   ============================================================ */
const RESEARCH_BADGES = {
  GROUNDED_RESEARCH: { cls: "ok", text: "🟢 GROUNDED RESEARCH" },
  RESEARCH_WITH_LIMITATIONS: { cls: "warn", text: "🟠 RESEARCH WITH LIMITATIONS" },
  NO_VERIFIED_PRECEDENT_FOUND: { cls: "warn", text: "🔴 NO VERIFIED PRECEDENT FOUND" },
  UNVERIFIED_RESEARCH: { cls: "bad", text: "🔴 UNVERIFIED / INSUFFICIENT RESEARCH" },
};

const rTier = (t) =>
  t === 1 ? "Tier 1 · official repository"
  : t === 2 ? "Tier 2 · established legal database"
  : t === 3 ? "Tier 3 · general web"
  : "authority tier not recorded";

function rFoot(authority, section, page, url) {
  const bits = [];
  if (authority) bits.push(escapeHtml(authority));
  if (section) bits.push(`Section ${escapeHtml(String(section))}`);
  if (page !== null && page !== undefined && page !== "") bits.push(`p. ${escapeHtml(String(page))}`);
  let html = bits.join(" · ");
  if (url && /^https:\/\//.test(url)) {
    html += (html ? " · " : "") +
      `<a class="src-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">open official source ↗</a>`;
  }
  return html ? `<div class="src-meta">${html}</div>` : "";
}

function rCard({ ok, right, title, snippet, foot }) {
  return `
    <div class="src-card ${ok ? "rs-ok" : "rs-warn-box"}">
      <div class="src-head">
        <span class="src-id ${ok ? "" : "rs-warn"}">${ok ? "✓ verified" : "⚠ unverified"}</span>
        <span class="src-page">${escapeHtml(right || "")}</span>
      </div>
      <div class="src-file">${escapeHtml(title || "")}</div>
      ${snippet ? `<div class="src-snippet">${escapeHtml(snippet)}</div>` : ""}
      ${foot || ""}
    </div>`;
}

async function refreshResearchSource() {
  const el = $("researchSource");
  try {
    const docs = await api("/api/documents");
    if (!docs.length) {
      el.className = "draft-source empty";
      el.textContent = "No documents in the workspace — upload a PDF first.";
      return;
    }
    el.className = "draft-source";
    el.innerHTML =
      `<b>${docs.length} case document${docs.length === 1 ? "" : "s"} indexed:</b> ` +
      docs.map((d) => escapeHtml(d.filename)).join(" · ");
  } catch (_) {
    el.className = "draft-source empty";
    el.textContent = "Could not load the workspace.";
  }
}

function renderResearchLaw(law) {
  const box = $("researchLaw");
  if (!law || !law.length) {
    box.innerHTML = `<div class="rs-empty">No verified provision could be retrieved — see Missing · Limitations.</div>`;
    return;
  }
  box.innerHTML = law
    .map((l) => rCard({
      ok: l.verified !== false,
      right: rTier(l.tier),
      title: l.provision || "(provision)",
      snippet: l.covers || "",
      foot: rFoot(l.authority, l.section || (l.evidence_ids || []).join(" "), l.page, l.url),
    }))
    .join("");
}

function renderResearchConnections(conns) {
  const box = $("researchConnections");
  if (!conns || !conns.length) {
    box.innerHTML = `<div class="rs-empty">No verified law-to-case connection could be established from the retrieved material.</div>`;
    return;
  }
  box.innerHTML = conns
    .map((c) => {
      const ids = (c.evidence_ids || []).concat(c.law_evidence_ids || []).join(" ");
      return `
      <div class="src-card ${c.verified ? "rs-ok" : "rs-warn-box"}">
        <div class="src-head">
          <span class="src-id ${c.verified ? "" : "rs-warn"}">${c.verified ? "✓ verified" : "⚠ unverified"}</span>
          <span class="src-page">${escapeHtml(ids)}</span>
        </div>
        <div class="src-file">${escapeHtml(c.provision || "")}</div>
        <div class="src-snippet">
          <b>What it covers:</b> ${escapeHtml(c.what_it_covers || "—")}<br>
          <b>Case connection:</b> ${escapeHtml(c.case_fact || "—")}<br>
          <b>Sufficiency:</b> ${escapeHtml(c.sufficiency || "—")}
        </div>
        ${c.missing ? `<div class="src-meta"><b>Missing:</b> ${escapeHtml(c.missing)}</div>` : ""}
      </div>`;
    })
    .join("");
}

function renderResearchJudgments(items) {
  const box = $("researchJudgments");
  if (!items || !items.length) {
    box.innerHTML = `<div class="rs-empty">No verified judgment was found in the retrieved sources — an unverified case is never presented as precedent.</div>`;
    return;
  }
  box.innerHTML = items
    .map((j) => {
      const meta = [j.court, j.date].filter(Boolean).join(" · ");
      const link = j.url && /^https:\/\//.test(j.url)
        ? `<div class="src-meta"><a class="src-link" href="${escapeHtml(j.url)}" target="_blank" rel="noopener noreferrer">open source ↗</a></div>`
        : "";
      return `
      <div class="src-card ${j.verified ? "rs-ok" : "rs-warn-box"}">
        <div class="src-head">
          <span class="src-id ${j.verified ? "" : "rs-warn"}">${j.verified ? "✓ verified" : "⚠ unverified"}</span>
          <span class="src-page">${escapeHtml(j.citation || "")}</span>
        </div>
        <div class="src-file">${escapeHtml(j.case_name || "(unnamed judgment)")}</div>
        <div class="src-snippet">${escapeHtml(meta || "—")}<br>${escapeHtml(j.principle || "")}
          ${j.relationship ? `<br><b>Relationship:</b> ${escapeHtml(j.relationship)}` : ""}</div>
        ${link}
      </div>`;
    })
    .join("");
}

function renderResearchSources(sources) {
  const box = $("researchSources");
  if (!sources || !sources.length) {
    box.innerHTML = `<div class="rs-empty">No sources were cited in this response.</div>`;
    return;
  }
  box.innerHTML = sources
    .map((s) => {
      const meta = [s.authority, s.section ? `Section ${s.section}` : "", rTier(s.tier)]
        .filter(Boolean).map(escapeHtml).join(" · ");
      const link = s.url && /^https:\/\//.test(s.url)
        ? `<a class="src-link" href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">open ↗</a>`
        : "";
      return `
      <div class="src-card ${s.verified ? "rs-ok" : "rs-warn-box"}">
        <div class="src-head">
          <span class="src-id ${s.verified ? "" : "rs-warn"}">${s.verified ? "✓ verified" : "⚠ not supported"}</span>
          <span class="src-page">${escapeHtml(s.source_type || "")} · ${escapeHtml(String(s.page ?? "—"))}</span>
        </div>
        <div class="src-file">${escapeHtml(s.title || s.filename || "")}</div>
        <div class="src-snippet">${meta || "—"}${s.relevance ? `<br>${escapeHtml(s.relevance)}` : ""}</div>
        ${link ? `<div class="src-meta">${link}</div>` : ""}
      </div>`;
    })
    .join("");
}

function renderResearchLimits(limits) {
  const list = $("researchLimits");
  if (!limits || !limits.length) {
    list.innerHTML = `<li class="dm-none">✓ No limitation was reported for this run.</li>`;
  } else {
    list.innerHTML = limits.map((m) => `<li class="dm-item">⚠ ${escapeHtml(m)}</li>`).join("");
  }
}

function renderResearch(data) {
  const badge = RESEARCH_BADGES[data.status] || RESEARCH_BADGES.UNVERIFIED_RESEARCH;
  const b = $("researchBadge");
  b.className = `draft-badge ${badge.cls}`;
  b.textContent = badge.text;

  const typeLabel = ($("researchType").selectedOptions[0] || {}).textContent || data.research_type;
  $("researchMeta").textContent =
    typeLabel +
    (data.reason ? ` · ${data.reason}` : "") +
    (data.refusal_stage ? ` · stage: ${data.refusal_stage}` : "");

  $("researchIssue").textContent = data.legal_issue || "—";
  $("researchAnswer").textContent = data.answer || "(no answer produced)";
  renderResearchLaw(data.relevant_law);
  renderResearchConnections(data.case_connections);
  renderResearchJudgments(data.judgments);
  renderResearchSources(data.sources);
  renderResearchLimits(data.limitations);
  renderClaimTrace("researchClaims", data);
  $("researchResult").hidden = false;
}

const researchState = { busy: false };

async function runResearch(ev) {
  ev.preventDefault();
  if (researchState.busy) return;
  const question = $("researchQuestion").value.trim();
  if (!question) return;
  researchState.busy = true;
  const btn = $("researchBtn");
  btn.disabled = true;

  setEvalStatusOn("researchStatus", "running",
    `<div class="eval-loading"><span class="spinner"></span>
       <div class="eval-loading-text"><b>Researching…</b>
       <span>Identifying the legal issue, retrieving case + official legal sources, then verifying every claim before answering.</span></div>
     </div>`);
  $("researchResult").hidden = true;

  try {
    const data = await api("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        case_context: $("researchContext").value.trim() || null,
        research_type: $("researchType").value,
      }),
    });
    setEvalStatusOn("researchStatus", "idle", "");
    renderResearch(data);
  } catch (err) {
    setEvalStatusOn("researchStatus", "error",
      `<div class="eval-idle-note">Research failed: ${escapeHtml(err.message)}</div>`);
  } finally {
    researchState.busy = false;
    btn.disabled = false;
  }
}

$("tabResearch").addEventListener("click", () => switchView("research"));
$("researchForm").addEventListener("submit", runResearch);
refreshResearchSource();
