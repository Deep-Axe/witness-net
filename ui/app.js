const KEYS = ["schema", "t_replay_us", "type", "source_id", "dest_id", "message_id",
  "message_type", "evidence_ref", "bytes", "check_codes", "state", "detail"];
const TYPES = ["observation_published", "packet_sent", "packet_dropped", "packet_delivered",
  "check", "observer_disconnected", "observer_reconnected",
  "coverage_published", "assessment", "map_snapshot",
  "query", "answer", "verdict", "model_call"];
const QUERY_TIERS = new Set(["T0", "T1", "T2", "T3"]);
const ANSWER_RESULTS = new Set(["detected", "not_detected", "observed_free", "not_covered",
  "unsupported", "unavailable"]);
const VERDICT_STATES = new Set(["corroborated", "contradicted", "unverified", "contested"]);
const MODEL_PRODUCTS = new Set(["muse-spark-1.3", "sam-3.1", "muse-voice-transcribe-1.0"]);
const MODEL_STAGES = new Set(["plan", "judge", "verify", "segment", "transcribe"]);
const CHECK_CODES = new Set([
  "OK", "DECODE_ERROR", "UNKNOWN_SOURCE", "KEY_MISMATCH", "BAD_SIGNATURE",
  "STALE", "EXPIRED", "FUTURE_TIME", "DUPLICATE", "OUTSIDE_REPLAY_WINDOW",
  "OUT_OF_BOUNDS", "OUTSIDE_PERMITTED_REGION",
]);
const ASSESSMENT_CODES = new Set([
  "CORROBORATED", "CONTRADICTED_BY_FREE_SPACE",
  "POSITION_CORROBORATED_CLASS_UNVERIFIED", "UNVERIFIED_SINGLE_SOURCE", "CONTESTED",
]);
const ASSESSMENT_STATES = new Set(["corroborated", "contradicted", "unverified", "contested"]);
const FALLBACK_NODES = ["pole1", "pole2", "pole3", "car1", "lidar1", "ghost1", "ghost2", "ego"];
const ASSESSMENT_TTL_US = 1_000_000;
const DEFAULT_GRID = { origin: [-10, -30], cell_m: 1, nx: 60, ny: 60 };
const POSES = {
  lidar1: [0, 0], pole1: [0, -4], pole2: [-2, -8], pole3: [2, -8],
  ego: [-2, -2], ghost1: [5.5, -8.5], ghost2: [24, 15.2],
};

const state = {
  events: [],
  errors: 0,
  t: 0,
  tmax: 0,
  filter: "all",
  playing: false,
  timer: null,
  selected: -1,
  mapPick: null,
};

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function isInt(value) {
  return typeof value === "number" && Number.isInteger(value) && Number.isFinite(value);
}

function isHex64(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function isCell(value) {
  return Array.isArray(value) && value.length === 2 && isInt(value[0]) && isInt(value[1]);
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function isRefList(value) {
  return Array.isArray(value) && value.every(isHex64);
}

function isNonemptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function validate(event, index) {
  if (!event || typeof event !== "object") return `line ${index}: not an object`;
  if (event.schema !== "coopx.events/1") return `line ${index}: schema ${event.schema}`;
  for (const key of KEYS) {
    if (!(key in event)) return `line ${index}: missing ${key}`;
  }
  for (const key of Object.keys(event)) {
    if (!KEYS.includes(key)) return `line ${index}: extra ${key}`;
  }
  if (!TYPES.includes(event.type)) return `line ${index}: type ${event.type}`;
  if (!isInt(event.t_replay_us) || event.t_replay_us < 0) return `line ${index}: t_replay_us`;
  if (!isNonemptyString(event.source_id)) return `line ${index}: source_id`;
  for (const name of ["dest_id", "message_id", "message_type"]) {
    if (event[name] != null && !isNonemptyString(event[name])) return `line ${index}: ${name}`;
  }
  if (event.evidence_ref != null && !isHex64(event.evidence_ref)) return `line ${index}: evidence_ref`;
  if (event.bytes != null && (!isInt(event.bytes) || event.bytes < 0)) return `line ${index}: bytes`;
  if (!Array.isArray(event.check_codes) || event.check_codes.some((c) => typeof c !== "string")) {
    return `line ${index}: check_codes`;
  }
  if (event.detail == null || typeof event.detail !== "object" || Array.isArray(event.detail)) {
    return `line ${index}: detail`;
  }
  if (event.type === "check") {
    if (!event.check_codes.length || event.check_codes.some((c) => !CHECK_CODES.has(c))) {
      return `line ${index}: unknown check code`;
    }
    const ok = event.check_codes.length === 1 && event.check_codes[0] === "OK";
    if (ok && event.state !== "accepted") return `line ${index}: accepted state`;
    if (!ok && event.state !== "rejected") return `line ${index}: rejected state`;
  } else if (event.type === "assessment") {
    if (!ASSESSMENT_STATES.has(event.state)) return `line ${index}: assessment state`;
    if (!event.check_codes.length || event.check_codes.some((c) => !ASSESSMENT_CODES.has(c))) {
      return `line ${index}: assessment codes`;
    }
    const cells = event.detail.cells;
    if (!Array.isArray(cells) || cells.some((c) => !isCell(c))) return `line ${index}: assessment cells`;
  } else if (event.type === "map_snapshot") {
    if (event.state != null || event.check_codes.length) return `line ${index}: map_snapshot state`;
    for (const key of ["occupied", "free", "contested"]) {
      if (!Array.isArray(event.detail[key]) || event.detail[key].some((c) => !isCell(c))) {
        return `line ${index}: map ${key}`;
      }
    }
    if (!isInt(event.detail.unknown_count) || event.detail.unknown_count < 0) {
      return `line ${index}: unknown_count`;
    }
  } else if (event.type === "coverage_published") {
    if (event.state != null || event.check_codes.length) return `line ${index}: coverage state`;
    if (event.message_type !== "coverage") return `line ${index}: coverage message_type`;
    const free = event.detail.free;
    const occupied = event.detail.occupied;
    if (!Array.isArray(free) || free.some((c) => !isCell(c))) return `line ${index}: coverage free`;
    if (!Array.isArray(occupied) || occupied.some((c) => !isCell(c))) return `line ${index}: coverage occupied`;
  } else if (event.type === "query") {
    if (event.state != null || event.check_codes.length) return `line ${index}: query state`;
    if (event.message_type !== "query" || !event.dest_id || !event.message_id) return `line ${index}: query id`;
    if (!QUERY_TIERS.has(event.detail.tier)) return `line ${index}: query tier`;
    const region = event.detail.region;
    if (!region || !Array.isArray(region.center) || region.center.length !== 2
        || !region.center.every(isFiniteNumber) || !isFiniteNumber(region.radius_m)
        || region.radius_m <= 0) return `line ${index}: query region`;
    if (!isInt(event.detail.time_us) || event.detail.time_us < 0) return `line ${index}: query time`;
    if (event.detail.concept != null && typeof event.detail.concept !== "string") return `line ${index}: query concept`;
    if (typeof event.detail.reason !== "string") return `line ${index}: query reason`;
    if (event.detail.question != null && typeof event.detail.question !== "string") return `line ${index}: query question`;
  } else if (event.type === "answer") {
    if (!ANSWER_RESULTS.has(event.state) || event.check_codes.length) return `line ${index}: answer state`;
    if (event.message_type !== "answer" || !event.dest_id || !event.message_id) return `line ${index}: answer id`;
    if (!isRefList(event.detail.evidence_refs)) return `line ${index}: answer refs`;
    if (event.detail.coverage_ref != null && !isHex64(event.detail.coverage_ref)) return `line ${index}: answer coverage_ref`;
    if (event.detail.withheld_refs != null && !isRefList(event.detail.withheld_refs)) return `line ${index}: answer withheld`;
    if (event.state === "detected" && !event.detail.evidence_refs.length) return `line ${index}: detected without refs`;
  } else if (event.type === "verdict") {
    if (!VERDICT_STATES.has(event.state) || event.check_codes.length) return `line ${index}: verdict state`;
    if (event.message_type !== "verdict" || !event.message_id) return `line ${index}: verdict id`;
    if (!isNonemptyString(event.detail.subject) || !isRefList(event.detail.evidence_refs)
        || typeof event.detail.reason !== "string"
        || !Array.isArray(event.detail.model_call_ids)
        || !event.detail.model_call_ids.every(isNonemptyString)) {
      return `line ${index}: verdict detail`;
    }
    if (event.state !== "unverified" && !event.detail.evidence_refs.length) return `line ${index}: verdict without refs`;
  } else if (event.type === "model_call") {
    if (event.state != null || event.check_codes.length) return `line ${index}: model_call state`;
    if (!event.message_id) return `line ${index}: model_call id`;
    const d = event.detail;
    if (!MODEL_PRODUCTS.has(d.product) || !MODEL_STAGES.has(d.stage)) return `line ${index}: model product`;
    if (typeof d.cache_hit !== "boolean" || event.t_replay_us !== d.availability_time_us) {
      return `line ${index}: model_call time honesty`;
    }
    if (!isFiniteNumber(d.latency_ms) || d.latency_ms < 0 || !isFiniteNumber(d.cost_usd) || d.cost_usd < 0
        || !isInt(d.query_time_us) || d.query_time_us > d.availability_time_us) {
      return `line ${index}: model_call accounting`;
    }
    if (d.question != null && typeof d.question !== "string") return `line ${index}: model_call question`;
  } else if (event.state != null || event.check_codes.length) {
    return `line ${index}: unexpected state/codes`;
  }
  return null;
}

function parseJsonl(text) {
  const events = [];
  let errors = 0;
  const messages = [];
  text.split(/\r?\n/).forEach((line, i) => {
    if (!line.trim()) return;
    let event;
    try { event = JSON.parse(line); }
    catch (err) {
      errors += 1;
      messages.push(`line ${i + 1}: ${err.message}`);
      return;
    }
    const problem = validate(event, i + 1);
    if (problem) {
      errors += 1;
      messages.push(problem);
      return;
    }
    events.push(event);
  });
  return { events, errors, messages };
}

function until(t) {
  return state.events.filter((e) => e.t_replay_us <= t);
}

function nodeIds() {
  const ids = new Set();
  for (const event of state.events) {
    if (event.source_id) ids.add(event.source_id);
    if (event.dest_id) ids.add(event.dest_id);
  }
  return ids.size ? [...ids].sort() : FALLBACK_NODES.slice();
}

function disconnectedAt(t) {
  const off = new Set();
  for (const event of until(t)) {
    if (event.type === "observer_disconnected") off.add(event.source_id);
    if (event.type === "observer_reconnected") off.delete(event.source_id);
  }
  return off;
}

function counters(t) {
  const out = { sent: 0, received: 0, loss: 0, disconnected: 0, budget: 0, accepted: 0, rejected: 0, schemaErrors: state.errors };
  for (const event of until(t)) {
    if (event.type === "packet_sent") out.sent += event.bytes || 0;
    if (event.type === "packet_delivered") out.received += event.bytes || 0;
    if (event.type === "packet_dropped") {
      const reason = (event.detail && event.detail.reason) || "loss";
      if (reason in out) out[reason] += 1;
    }
    if (event.type === "check" && event.state === "accepted") out.accepted += 1;
    if (event.type === "check" && event.state === "rejected") out.rejected += 1;
  }
  return out;
}

function txOf(event) {
  return event.detail && event.detail.tx_id;
}

function inFlight(t) {
  const done = new Set();
  const sent = [];
  for (const event of until(t)) {
    if (event.type === "packet_sent") sent.push(event);
    if (event.type === "packet_delivered" || event.type === "packet_dropped") {
      done.add(`${txOf(event)}|${event.dest_id}`);
    }
  }
  const nodes = nodeIds();
  const flights = [];
  for (const event of sent) {
    const tx = txOf(event);
    for (const dest of nodes) {
      if (dest === event.source_id) continue;
      if (done.has(`${tx}|${dest}`)) continue;
      const later = state.events.find((e) =>
        e.t_replay_us > t && txOf(e) === tx && e.dest_id === dest &&
        (e.type === "packet_delivered" || e.type === "packet_dropped"));
      if (later) {
        flights.push({
          source: event.source_id, dest, tx_id: tx,
          evidence_ref: event.evidence_ref, bytes: event.bytes,
        });
      }
    }
  }
  return flights;
}

function latestOf(type, t) {
  let found = null;
  for (const event of until(t)) {
    if (event.type === type) found = event;
  }
  return found;
}

function latestAssessments(t) {
  const byRef = new Map();
  for (const event of until(t)) {
    // Latest judgement per source and footprint, within the map TTL (1 s);
    // older judgements expire like the claims they were about.
    if (event.type === "assessment" && event.t_replay_us > t - ASSESSMENT_TTL_US) {
      byRef.set(`${event.source_id}|${JSON.stringify(event.detail.cells || [])}`, event);
    }
  }
  return [...byRef.values()];
}

function gridSpec() {
  const cov = latestOf("coverage_published", state.t);
  if (cov && cov.detail && cov.detail.grid) return cov.detail.grid;
  return DEFAULT_GRID;
}

function rowClass(event, idx) {
  const bits = [];
  if (idx === state.selected) bits.push("sel");
  if (event.type === "check" && event.state === "rejected") bits.push("rejected");
  if (event.type === "assessment" && event.state === "contradicted") bits.push("contradicted");
  if (event.type === "assessment" && event.state === "contested") bits.push("contested");
  return bits.join(" ");
}

function renderNodes(off) {
  const host = $("nodes");
  clear(host);
  for (const id of nodeIds()) {
    const down = off.has(id);
    const node = el("div", `node ${id}${down ? " off" : ""}`);
    node.appendChild(el("strong", "", id));
    node.appendChild(el("span", "", down ? "disconnected" : "connected"));
    host.appendChild(node);
  }
}

function renderPackets(t) {
  const host = $("packets");
  clear(host);
  const nowEvents = state.events.filter((e) => e.t_replay_us === t);
  const flights = inFlight(t);
  if (!flights.length && !nowEvents.some((e) => e.type === "packet_delivered" || e.type === "packet_dropped")) {
    host.appendChild(el("div", "pkt", "No packets at this instant."));
    return;
  }
  for (const flight of flights) {
    host.appendChild(el("div", "pkt flight",
      `${flight.source} → ${flight.dest} in flight · ${flight.bytes} B · ${flight.tx_id || ""}`));
  }
  for (const event of nowEvents) {
    if (event.type === "packet_delivered") {
      host.appendChild(el("div", "pkt done",
        `${event.source_id} → ${event.dest_id} delivered · ${event.bytes} B`));
    }
    if (event.type === "packet_dropped") {
      const reason = event.detail && event.detail.reason ? event.detail.reason : "?";
      host.appendChild(el("div", "pkt drop",
        `${event.source_id} → ${event.dest_id} dropped (${reason})`));
    }
  }
}

function renderTable() {
  const host = $("rows");
  clear(host);
  const rows = state.events.filter((event) => {
    if (event.t_replay_us > state.t) return false;
    return state.filter === "all" || event.type === state.filter;
  });
  const maxRows = 400;
  const hidden = Math.max(0, rows.length - maxRows);
  const visible = hidden ? rows.slice(-maxRows) : rows;
  if (hidden) {
    const notice = el("tr", "muted");
    const cell = el("td", "", `… ${hidden} earlier events (table follows t)`);
    cell.colSpan = 8;
    notice.appendChild(cell);
    host.appendChild(notice);
  }
  for (const event of visible) {
    const idx = state.events.indexOf(event);
    const tr = el("tr", rowClass(event, idx));
    tr.dataset.i = String(idx);
    for (const value of [
      event.t_replay_us, event.type, event.source_id, event.dest_id ?? "",
      event.message_id ?? "", event.bytes ?? "",
      Array.isArray(event.check_codes) ? event.check_codes.join(",") : "",
      event.state ?? "",
    ]) {
      tr.appendChild(el("td", "", value === "" ? "" : String(value)));
    }
    host.appendChild(tr);
  }
  const wrap = $("table-wrap");
  const focus = host.querySelector("tr.sel") || host.lastElementChild;
  if (focus && wrap) focus.scrollIntoView({ block: "nearest" });
}

function cellKey(cell) {
  return `${cell[0]},${cell[1]}`;
}

function eventByRef(ref, t) {
  // Publication row for a claim's evidence ref (coverage or observation).
  for (const event of until(t)) {
    if ((event.type === "coverage_published" || event.type === "observation_published")
        && event.evidence_ref === ref) return event;
  }
  return null;
}

function claimsForCell(ix, iy) {
  const key = cellKey([ix, iy]);
  const snap = latestOf("map_snapshot", state.t);
  const contested = (snap && Array.isArray(snap.detail.contested_claims))
    ? snap.detail.contested_claims.find((c) => isCell(c.cell) && cellKey(c.cell) === key)
    : null;
  if (contested && Array.isArray(contested.claims)) {
    // The ego map's own per-source claims behind this contested cell.
    const claims = [];
    for (const pair of contested.claims) {
      if (!Array.isArray(pair) || !isHex64(pair[0])) continue;
      const event = eventByRef(pair[0], state.t);
      if (event) claims.push({ kind: String(pair[1]), event });
    }
    return claims;
  }
  // Otherwise: the latest coverage from each source that covers this cell.
  const latestCoverage = new Map();
  for (const event of until(state.t)) {
    if (event.type === "coverage_published") latestCoverage.set(event.source_id, event);
  }
  const claims = [];
  for (const event of latestCoverage.values()) {
    if ((event.detail.free || []).some((c) => cellKey(c) === key)) claims.push({ kind: "free", event });
    if ((event.detail.occupied || []).some((c) => cellKey(c) === key)) claims.push({ kind: "occupied", event });
  }
  return claims;
}

function showEvidence(title, claims) {
  $("map-info").textContent = title;
  const host = $("evidence-list");
  clear(host);
  if (!claims.length) {
    host.appendChild(el("li", "", "No linked evidence at this cell."));
    return;
  }
  for (const claim of claims) {
    const item = el("li");
    const ref = claim.event.evidence_ref || claim.event.message_id || claim.event.type;
    const button = el("button", "link", ref);
    button.type = "button";
    button.addEventListener("click", () => {
      const idx = state.events.indexOf(claim.event);
      if (idx >= 0) select(idx);
    });
    item.appendChild(button);
    item.appendChild(el("span", "evidence-kind", ` ${claim.kind} · ${claim.event.source_id}`));
    host.appendChild(item);
  }
}

function drawHatch(ctx, x, y, w, h, color) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  for (let i = -h; i < w + h; i += 4) {
    ctx.beginPath();
    ctx.moveTo(x + i, y);
    ctx.lineTo(x + i + h, y + h);
    ctx.stroke();
  }
  ctx.restore();
}

function renderMap() {
  const canvas = $("map");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const grid = gridSpec();
  const nx = grid.nx || 60;
  const ny = grid.ny || 60;
  const snap = latestOf("map_snapshot", state.t);
  const occupied = new Set((snap && snap.detail.occupied || []).map(cellKey));
  const free = new Set((snap && snap.detail.free || []).map(cellKey));
  const contested = new Set((snap && snap.detail.contested || []).map(cellKey));
  const cw = canvas.width / nx;
  const ch = canvas.height / ny;
  ctx.fillStyle = "#121820";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const key = `${ix},${iy}`;
      const x = ix * cw;
      const y = canvas.height - (iy + 1) * ch;
      if (contested.has(key)) {
        ctx.fillStyle = "#7c4a0a";
        ctx.fillRect(x, y, cw, ch);
        drawHatch(ctx, x, y, cw, ch, "#f0b429");
      } else if (occupied.has(key)) {
        ctx.fillStyle = "#c23b3b";
        ctx.fillRect(x, y, cw, ch);
      } else if (free.has(key)) {
        ctx.fillStyle = "#2f6f4a";
        ctx.fillRect(x, y, cw, ch);
      }
    }
  }
  const origin = grid.origin || DEFAULT_GRID.origin;
  const cellM = grid.cell_m || 1;
  function toPx(xy) {
    const ix = (xy[0] - origin[0]) / cellM;
    const iy = (xy[1] - origin[1]) / cellM;
    return [ix * cw, canvas.height - iy * ch];
  }
  for (const [id, pose] of Object.entries(POSES)) {
    if (!nodeIds().includes(id) && id !== "lidar1" && id !== "pole1") continue;
    const [px, py] = toPx(pose);
    ctx.fillStyle = id.startsWith("ghost") ? "#fb923c" : (id === "lidar1" ? "#5eead4" : "#6cb6ff");
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#e7eef6";
    ctx.font = "10px ui-sans-serif";
    ctx.fillText(id, px + 5, py - 4);
  }
  const colors = {
    corroborated: "#3dd68c", contradicted: "#ff6b6b",
    unverified: "#f0b429", contested: "#fb923c",
  };
  for (const assessment of latestAssessments(state.t)) {
    const cells = assessment.detail.cells || [];
    if (!cells.length) continue;
    const mx = cells.reduce((s, c) => s + c[0], 0) / cells.length;
    const my = cells.reduce((s, c) => s + c[1], 0) / cells.length;
    const [px, py] = [mx * cw + cw / 2, canvas.height - (my * ch + ch / 2)];
    ctx.strokeStyle = colors[assessment.state] || "#e7eef6";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (state.mapPick) {
    const [ix, iy] = state.mapPick;
    ctx.strokeStyle = "#e7eef6";
    ctx.lineWidth = 2;
    ctx.strokeRect(ix * cw + 0.5, canvas.height - (iy + 1) * ch + 0.5, cw - 1, ch - 1);
  }
}

function onMapClick(ev) {
  const canvas = $("map");
  const grid = gridSpec();
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
  const y = (ev.clientY - rect.top) * (canvas.height / rect.height);
  const ix = Math.floor(x / (canvas.width / grid.nx));
  const iy = Math.floor((canvas.height - y) / (canvas.height / grid.ny));
  if (ix < 0 || iy < 0 || ix >= grid.nx || iy >= grid.ny) return;
  state.mapPick = [ix, iy];
  const snap = latestOf("map_snapshot", state.t);
  const contested = new Set((snap && snap.detail.contested || []).map(cellKey));
  const label = contested.has(cellKey([ix, iy]))
    ? `Contested cell [${ix}, ${iy}]`
    : `Cell [${ix}, ${iy}]`;
  const claims = claimsForCell(ix, iy);
  const objects = latestAssessments(state.t).filter((event) =>
    (event.detail.cells || []).some((c) => c[0] === ix && c[1] === iy));
  for (const event of objects) {
    claims.push({ kind: event.state, event });
  }
  showEvidence(label, claims);
  renderMap();
}

function jumpToRef(ref) {
  const idx = state.events.findIndex((event) =>
    event.evidence_ref === ref || event.message_id === ref);
  if (idx >= 0) select(idx);
}

function renderReasoner() {
  const host = $("reasoner");
  if (!host) return;
  clear(host);
  const evs = until(state.t);
  const asked = evs.find((e) => (e.type === "query" || e.type === "model_call")
    && typeof e.detail.question === "string");
  const question = evs.find((e) => e.type === "query");
  host.appendChild(el("h3", "", "Question"));
  host.appendChild(el("p", "", asked ? asked.detail.question : question
    ? (question.detail.reason || question.detail.concept || "region question")
    : "No query yet."));
  host.appendChild(el("h3", "", "Plan"));
  const queries = evs.filter((e) => e.type === "query");
  if (!queries.length) host.appendChild(el("p", "muted", "No planned queries."));
  for (const event of queries) {
    const line = `${event.detail.tier} → ${event.dest_id} · ${event.detail.concept || "—"} · ${event.detail.reason || ""}`;
    host.appendChild(el("p", "", line));
  }
  host.appendChild(el("h3", "", "Answers"));
  const answers = evs.filter((e) => e.type === "answer");
  if (!answers.length) host.appendChild(el("p", "muted", "No answers yet."));
  for (const event of answers) {
    const row = el("p");
    row.appendChild(el("span", "", `${event.source_id}: ${event.state} `));
    for (const ref of event.detail.evidence_refs || []) {
      const button = el("button", "link", ref.slice(0, 8));
      button.type = "button";
      button.addEventListener("click", () => jumpToRef(ref));
      row.appendChild(button);
    }
    host.appendChild(row);
  }
  host.appendChild(el("h3", "", "Verdict"));
  const verdicts = evs.filter((e) => e.type === "verdict");
  if (!verdicts.length) host.appendChild(el("p", "muted", "No verdict yet."));
  for (const event of verdicts) {
    host.appendChild(el("p", "", `${event.state} · ${event.detail.reason || ""}`));
    for (const ref of event.detail.evidence_refs || []) {
      const button = el("button", "link", ref.slice(0, 12));
      button.type = "button";
      button.addEventListener("click", () => jumpToRef(ref));
      host.appendChild(button);
    }
  }
  host.appendChild(el("h3", "", "Model calls"));
  const calls = evs.filter((e) => e.type === "model_call");
  if (!calls.length) host.appendChild(el("p", "muted", "No model calls yet."));
  for (const event of calls) {
    const d = event.detail;
    const line = `${d.product} ${d.stage} · ${Math.round(d.latency_ms)} ms · cache ${d.cache_hit} · $${d.cost_usd.toFixed(4)}`;
    host.appendChild(el("p", "", line));
  }
}

function render() {
  const t = state.t;
  $("clock").textContent = `t = ${t} µs`;
  $("scrub").value = String(t);
  const counts = counters(t);
  for (const [key, value] of Object.entries(counts)) {
    const node = document.querySelector(`[data-k="${key}"]`);
    if (node) node.textContent = String(value);
  }
  renderNodes(disconnectedAt(t));
  renderPackets(t);
  renderTable();
  renderMap();
  renderReasoner();
}

function select(index) {
  state.selected = index;
  const event = state.events[index];
  $("detail").textContent = event ? JSON.stringify(event, null, 2) : "Click a row.";
  if (event) {
    state.t = event.t_replay_us;
    $("scrub").value = String(state.t);
    if (event.evidence_ref || (event.detail && event.detail.cells)) {
      const cells = event.detail && event.detail.cells;
      if (cells && cells[0]) state.mapPick = cells[0];
      showEvidence(
        event.type === "assessment" ? `Assessment ${event.state}` : `Event ${event.type}`,
        [{ kind: event.state || event.type, event }]);
    }
  }
  render();
}

function setLog(text, label) {
  const parsed = parseJsonl(text);
  state.events = parsed.events;
  state.errors = parsed.errors;
  state.t = parsed.events.length ? parsed.events[0].t_replay_us : 0;
  state.tmax = parsed.events.reduce((max, event) => Math.max(max, event.t_replay_us), 0);
  state.selected = -1;
  state.mapPick = null;
  $("scrub").min = String(state.t);
  $("scrub").max = String(state.tmax);
  const status = `${label}: ${parsed.events.length} events, ${parsed.errors} schema errors`;
  $("load-status").textContent = parsed.messages[0] ? `${status}. ${parsed.messages[0]}` : status;
  $("load-status").className = parsed.errors ? "status error" : "status";
  const filters = $("filters");
  clear(filters);
  for (const type of ["all", ...TYPES]) {
    const button = el("button", type === state.filter ? "on" : "", type);
    button.type = "button";
    button.dataset.type = type;
    filters.appendChild(button);
  }
  showEvidence("Click a cell or assessed object.", []);
  const img = $("s3-overlay");
  if (img) {
    img.hidden = true;
    img.onload = () => { img.hidden = false; };
    img.onerror = () => { img.hidden = true; };
    img.src = "/data/s3/overlay.png";
  }
  render();
}

function play() {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "Pause" : "Play";
  if (state.playing) {
    state.timer = setInterval(() => {
      const step = Math.max(1, Math.floor((state.tmax || 1) / 200));
      state.t = Math.min(state.tmax, state.t + step);
      if (state.t >= state.tmax) play();
      render();
    }, 50);
  } else if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

$("play").addEventListener("click", play);
$("scrub").addEventListener("input", (ev) => {
  state.t = Number(ev.target.value);
  render();
});
$("file").addEventListener("change", (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  file.text().then((text) => setLog(text, file.name));
});
$("rows").addEventListener("click", (ev) => {
  const tr = ev.target.closest("tr");
  if (tr) select(Number(tr.dataset.i));
});
$("filters").addEventListener("click", (ev) => {
  const button = ev.target.closest("button");
  if (!button) return;
  state.filter = button.dataset.type;
  for (const node of $("filters").querySelectorAll("button")) {
    node.classList.toggle("on", node === button);
  }
  render();
});
$("map").addEventListener("click", onMapClick);

const params = new URLSearchParams(location.search);
const log = params.get("log");
if (log) {
  fetch(log).then((response) => {
    if (!response.ok) throw new Error(response.status + " " + response.statusText);
    return response.text();
  }).then((text) => setLog(text, log)).catch((err) => {
    $("load-status").textContent = `Could not fetch ${log}: ${err.message}`;
    $("load-status").className = "status error";
  });
}
