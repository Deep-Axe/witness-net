const KEYS = ["schema", "t_replay_us", "type", "source_id", "dest_id", "message_id",
  "message_type", "evidence_ref", "bytes", "check_codes", "state", "detail"];
const TYPES = ["observation_published", "packet_sent", "packet_dropped", "packet_delivered",
  "check", "observer_disconnected", "observer_reconnected"];
const NODES = ["pole1", "car1", "ego"];

const state = {
  events: [],
  errors: 0,
  t: 0,
  tmax: 0,
  filter: "all",
  playing: false,
  timer: null,
  selected: -1,
};

const $ = (id) => document.getElementById(id);

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

function inFlight(t) {
  const delivered = new Set();
  const dropped = new Set();
  const sent = [];
  for (const event of until(t)) {
    const key = `${event.evidence_ref}|${event.dest_id || ""}`;
    if (event.type === "packet_sent") sent.push(event);
    if (event.type === "packet_delivered") delivered.add(key);
    if (event.type === "packet_dropped") dropped.add(key);
  }
  const flights = [];
  for (const event of sent) {
    for (const dest of NODES) {
      if (dest === event.source_id) continue;
      const key = `${event.evidence_ref}|${dest}`;
      if (delivered.has(key) || dropped.has(key)) continue;
      const later = state.events.find((e) =>
        e.t_replay_us > t && e.evidence_ref === event.evidence_ref &&
        ((e.type === "packet_delivered" && e.dest_id === dest) ||
         (e.type === "packet_dropped" && e.dest_id === dest)));
      if (later) flights.push({ source: event.source_id, dest, evidence_ref: event.evidence_ref, bytes: event.bytes });
    }
  }
  return flights;
}

function render() {
  const t = state.t;
  const off = disconnectedAt(t);
  $("clock").textContent = `t = ${t} µs`;
  $("scrub").value = String(t);
  const counts = counters(t);
  for (const [key, value] of Object.entries(counts)) {
    const el = document.querySelector(`[data-k="${key}"]`);
    if (el) el.textContent = String(value);
  }

  $("nodes").innerHTML = NODES.map((id) => {
    const down = off.has(id);
    return `<div class="node ${id}${down ? " off" : ""}"><strong>${id}</strong><span>${down ? "disconnected" : "connected"}</span></div>`;
  }).join("");

  const nowEvents = state.events.filter((e) => e.t_replay_us === t);
  const flights = inFlight(t);
  const bits = [];
  for (const f of flights) bits.push(`<div class="pkt flight">${f.source} → ${f.dest} in flight · ${f.bytes} B</div>`);
  for (const event of nowEvents) {
    if (event.type === "packet_delivered") {
      bits.push(`<div class="pkt done">${event.source_id} → ${event.dest_id} delivered · ${event.bytes} B</div>`);
    }
    if (event.type === "packet_dropped") {
      bits.push(`<div class="pkt drop">${event.source_id} → ${event.dest_id} dropped (${event.detail.reason})</div>`);
    }
  }
  $("packets").innerHTML = bits.length ? bits.join("") : "<div class='pkt'>No packets at this instant.</div>";

  const rows = state.events.filter((e) => state.filter === "all" || e.type === state.filter);
  $("rows").innerHTML = rows.map((event, i) => {
    const idx = state.events.indexOf(event);
    const cls = idx === state.selected ? "sel" : "";
    return `<tr class="${cls}" data-i="${idx}">
      <td>${event.t_replay_us}</td><td>${event.type}</td>
      <td>${event.source_id}</td><td>${event.dest_id ?? ""}</td>
      <td>${event.message_id ?? ""}</td><td>${event.bytes ?? ""}</td>
      <td>${(event.check_codes || []).join(",")}</td><td>${event.state ?? ""}</td>
    </tr>`;
  }).join("");
}

function select(index) {
  state.selected = index;
  const event = state.events[index];
  $("detail").textContent = event ? JSON.stringify(event, null, 2) : "Click a row.";
  if (event) {
    state.t = event.t_replay_us;
    $("scrub").value = String(state.t);
  }
  render();
}

function setLog(text, label) {
  const parsed = parseJsonl(text);
  state.events = parsed.events;
  state.errors = parsed.errors;
  state.t = parsed.events.length ? parsed.events[0].t_replay_us : 0;
  state.tmax = parsed.events.reduce((m, e) => Math.max(m, e.t_replay_us), 0);
  $("scrub").min = String(state.t);
  $("scrub").max = String(state.tmax);
  const status = `${label}: ${parsed.events.length} events, ${parsed.errors} schema errors`;
  $("load-status").textContent = parsed.messages[0] ? `${status}. ${parsed.messages[0]}` : status;
  $("load-status").className = parsed.errors ? "status error" : "status";
  const filters = $("filters");
  filters.innerHTML = ["all", ...TYPES].map((type) => `<button type="button" data-type="${type}">${type}</button>`).join("");
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
  const btn = ev.target.closest("button");
  if (!btn) return;
  state.filter = btn.dataset.type;
  for (const el of $("filters").querySelectorAll("button")) el.classList.toggle("on", el === btn);
  render();
});

const params = new URLSearchParams(location.search);
const log = params.get("log");
if (log) {
  fetch(log).then((r) => {
    if (!r.ok) throw new Error(r.status + " " + r.statusText);
    return r.text();
  }).then((text) => setLog(text, log)).catch((err) => {
    $("load-status").textContent = `Could not fetch ${log}: ${err.message}`;
    $("load-status").className = "status error";
  });
}
