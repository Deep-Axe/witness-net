# Perception exchange protocol — draft v0.2

**Status:** design draft, updated 2026-11-03. Implemented: CX1 object records
(`src/coopx/radio_codec.py`), signed CX2 observation envelopes and authenticated
lineage (`src/coopx/envelope.py`), Ed25519 identities and trust-anchor metadata
(`src/coopx/identity.py`), claim-scoped witness counting, the deterministic
admission checker with a bounded replay window (`src/coopx/checker.py`), a
binary packet transport with end-to-end S1 and S2 runs (`src/coopx/pipeline.py`),
signed coverage messages as three row-major bitmaps (observed, free, occupied;
magic `CX2C`) with a per-source coverage map (`src/coopx/coverage.py`), and
deterministic assessment of admitted observations (`src/coopx/assess.py`).
Other message-body codecs (coverage, query, answer, verdict, capability, label
table), coverage/free-space checks and advisories remain specified, not built.

> Give any robot access to the senses around it. Machines discover observers,
> ask about the world, and cross-check the answers, even when the robot has no
> camera or language model of its own.

The interaction model is **Discover / Observe / Verify**: tool-style discovery
and querying of perception. (Analogy only: this is not Anthropic's Model
Context Protocol and does not use the MCP spec.)

## Meta models in the network

Meta models do the two jobs that make the network more than late fusion:
turning cameras into queryable observers, and coordinating the network.
Access was verified with small API probes on 2026-11-02 (internal record). A
passing probe proves access, not integration or accuracy on our dataset.

| Model | Role in the network | Access evidence (2026-11-02) | Demo status |
|---|---|---|---|
| **Muse Spark 1.3** | The **network Reasoner**: compiles questions into typed queries, selects observers, runs active verification, issues evidence-linked verdicts. Hackathon: fixed plan → execute → judge calls (below); open tool loop later | Strict JSON, function call + result round trip, image grounding passed | To build; **the hero** |
| **SAM 3.1** | Makes any camera an **open-vocabulary Observer endpoint**; executes T3 (one noun phrase per request; boxes + masks; tracking within a clip) | Image and video passed | To build |
| **Muse Voice Transcribe** | Spoken operator question ("what's behind that truck?") → Reasoner | File transcription passed; realtime untested | Demo add-on |
| **Muse Wonder 1.0** | Escalation Reasoner for hard contested cases, if it beats Spark on them | Text only; tools/vision untested | Evaluate before use |
| **Spark grounding** | Alternative Observer backend when SAM is unsuitable | Synthetic grounding passed | Fallback |
| **Muse Glimmer** (open weights, 30B, Apache 2.0) | Same Reasoner interface at the edge: pole computer, site server, capable robot | Not run; does not fit the 6 GB laptop | Design direction |
| **Muse Spark Open** (open weights, 816B MoE) | Self-hosted network Reasoner for an operator | Not run | Design direction |
| Muse Image / Muse Code | Presentation assets (labelled synthetic, never GT) / development tooling | Catalog / CLI only | Not in the network runtime |

A robot without any model still participates: it consumes typed observations
and can call the Muse Reasoner service over the network.

### Constraints the Reasoner design must respect

- **`tool_choice` is `auto` only** (named choice rejected with HTTP 400), so an
  open-ended tool loop is unpredictable. **Hackathon pattern: plan → execute →
  judge.** Call 1 (strict JSON): Spark returns a bounded plan: which queries
  (T1–T3), to which observers, why. Our code validates and executes it.
  Call 2 (strict JSON): Spark receives the typed answers and returns verdicts
  with evidence references. At most one extra verification round. The
  function-calling loop (already shown to work) is a later upgrade, not the
  demo path. Meta never executes our tools; our code does.
- **Latency:** small probes took ~1.7–3.2 s. Muse cannot sit on the per-packet
  path; see two-rate fusion below.
- **Schema guarantees format, not truth.** Strict JSON output still goes through
  semantic validation.
- **SAM segments; the sensor backend measures.** SAM returns masks and boxes
  as segmentation text. The observer's backend aligns each mask with the
  depth image (RGB-D/stereo) or LiDAR points, lifts it to world coordinates,
  tracks it, and emits typed observations. SAM never needs to produce depth.
- **Live path is per-frame, latest-frame-wins.** Each live query sends the
  newest frame as a SAM image request (~2 Hz); frames that are superseded
  before they are sent are dropped. Every answer carries its `capture_time`,
  so consumers see its age, predict forward, and expire it. SAM video
  requests (50-frame billing minimum, answer only after the clip) are for
  recording, offline analysis and track refinement, **never** for live hazard
  answers: by the time a batch returns, the object may have moved on.
- **Replay honesty:** cached SAM results are released at capture time plus
  measured latency, never earlier.
- **Prompt caching:** stable instructions and tool definitions first, scene data last.
- Standard models only; Contributor IDs are not used (different data-use terms).

## Design rules

1. **The wire carries typed observation claims and supporting evidence**, not
   prose and not raw sensor data. Claims have explicit spatial, temporal and
   provenance semantics that permit checks; they are not facts until checked.
2. **Language appears only in queries, never in observations.** A T3 query may
   carry a short noun phrase for an open-vocabulary observer. Every answer
   returns as typed records. (Contrast: LangCoop/SafeCoop exchange prose.)
3. **Silence is not free space.** Negative evidence exists only where an
   observer declares what it actually covered and what it could detect.
4. **We count independent observations, not repeated messages.** Every claim
   carries lineage.
5. **No learned cross-agent feature adapter.** Heterogeneous nodes still need
   sensor adapters: calibration, coordinate conversion, measurement semantics.

## Roles

| Role | Does | Example |
|---|---|---|
| Observer | Publishes observations and coverage; answers queries | Pole camera + SAM; LiDAR node; UWB tag reader |
| Consumer | Subscribes, queries, uses results | Ego car; AMR at a blind aisle |
| Reasoner | Query compilation, observer selection, verification, verdicts | Muse Spark service; Muse Glimmer at the edge |
| Relay | Forwards; adds no observations; never becomes the original observer | Roadside unit, Wi-Fi AP |

## Common envelope (every message)

| Field | Meaning |
|---|---|
| `protocol_version` | Envelope version |
| `message_type` | capability, observation, coverage, query, query_ack, answer, verdict, label_table |
| `message_id` | Unique per source |
| `source_id` | Sender identity (pseudonym allowed) |
| `session_id` | Groups a query exchange or a subscription |
| `spatial_frame_id` | Coordinate frame of all geometry (e.g. a site/map frame). **Not** an image or sensor frame number |
| `capture_time` | When the underlying measurement was taken |
| `generation_time` | When this message was created |
| `valid_until` | Expiry; consumers drop or downgrade after this |
| `auth` | Signature scheme, key/cert reference, signature (Ed25519 in the demo) |

Image/sensor frame sequence numbers, where needed, are separate fields
(`sensor_seq`).

## Message types

### Capability (Discover)

- `roles`, `modalities` (camera, lidar, radar, ultrasonic, uwb/rfid, self-pose only)
- `nominal_coverage`: footprint polygon or sector, range, height band, static
  or moving with the node. This is **reach**, not evidence.
- `detectable`: what it can detect and at what minimum size/height, per modality
  (e.g. 2D scanner: obstacles crossing its scan plane at 0.2 m height)
- `vocabulary`: `fixed` class list | `geometric` (class UNKNOWN + extent) | `open` (accepts T3)
- `kinematics`: typical position std, velocity yes/no, latency, max rate
- `query_tiers`: which of T0–T3 it answers
- `cost_hint`: relative cost/latency of answering (open-vocab inference is paid)

### Observation (Observe)

Extends CX1 (class, local object id, sequence, capture time, position + anchor,
velocity, dimensions, yaw, confidence, position std) with:

- `kind`: `perceived` ("I see something there", CPM-style) or `self`
  ("I am here", CAM-style)
- `geometry`: point+std | oriented box | region (polygon/cells). Weak sensors
  report regions instead of fake precise points.
- `label_id`: optional open-vocabulary label from the label table. Fixed-class
  consumers read `class` and ignore it.
- **Lineage:** `origin_observer`, `origin_observation_id`, `origin_capture_time`,
  `derivation` (`direct` | `relayed` | `tracked` | `fused` | `verdict`),
  `parents` (evidence references).

### Coverage (makes negative evidence meaningful)

Coverage is a claim about what was **actually inspected** during a capture
window, separate from nominal reach.

- `window`: start/end capture time
- `observed_region`: geometry actually inspected, with holes for estimated
  occlusion (e.g. LiDAR: up to first return per ray; nothing behind it)
- `height_band` and `min_detectable_size` the evidence applies to
- `free_region`: sub-regions with **positive evidence of being unoccupied**
  for those target characteristics, with uncertainty
- `modality`

Rules:

- **Free space comes from range measurements, never from absent detections.**
  SAM finding no "pedestrian" does not prove there is none.
- An **RGB-only** camera emits coverage (frustum minus occlusion) and occupied
  detections; everything else in its view stays **unknown**.
- An **RGB-D / stereo** camera is also a range sensor: its depth image gives
  free space up to the first depth return per pixel, within its depth range,
  stated height band and minimum size, independent of what SAM found.
- **LiDAR / radar** give free space up to the first return per ray.
- Consumers keep **per-source evidence per cell** and derive a display state:
  **occupied / free / unknown / contested**. Contested means occupied and free
  claims at compatible times; it is treated conservatively (as occupied for
  advisories) and links to both claims. Merging never discards disagreement.

### Label table

Maps `label_id` to short noun phrases. Bounded size, versioned, and treated as
data when shown to any model.

### Query and answer (Observe / Verify on demand)

| Tier | Query | Who can answer |
|---|---|---|
| T0 | Subscribe: region + classes + max age | Any observer |
| T1 | Objects and occupancy in region R over time window W | Observers covering R |
| T2 | Verify: object near p (class c, size s) at time t? | Any observer, including "not covered" |
| T3 | Find `<noun phrase>` in R | `vocabulary: open` observers |

Natural-language questions are compiled by a Reasoner. SAM 3.1 takes one
concrete noun phrase per request, so "what's behind that truck?" becomes
several T3/T1 queries (e.g. "person", "bicycle", "car") over a geometric region
behind the occluder, plus geometric filtering of the results.

**Lifecycle**

```
query → observer selection → query_ack (accepted | declined + reason)
      → observation(s) + coverage → deterministic checks → answer
      → (optional) Reasoner verdict → expiry at valid_until
```

**Answer record**

| Field | Meaning |
|---|---|
| `query_id` | The query being answered |
| `requested_region`, `requested_time` | What was asked |
| `observed_time` | When the observer actually measured |
| `coverage_ref` | Coverage message the answer relies on |
| `evidence_refs` | Observation IDs used |
| `result` | One of the results below |

| Result | Meaning |
|---|---|
| `detected` | A matching observation exists |
| `not_detected` | The detector ran and found no match. Not proof of absence |
| `observed_free` | A range sensor gives positive evidence the space is unoccupied for the stated size/height |
| `not_covered` | The observer could not inspect the region at that time |
| `unsupported` | It cannot answer this kind of request (e.g. T3 to a LiDAR) |
| `unavailable` | Overloaded, offline, or declined by policy |

### Verdict (from a Reasoner)

Per remote object or region: `corroborated` | `contradicted` | `unverified` |
`contested`. A verdict is a derived claim: it carries `parents` pointing to the
exact observations and coverage used, and a short reason. A verdict built on
one camera's observation is **not** independent of that camera.

A **free-space contradiction** means a claimed object overlaps sufficiently
observed free space at a compatible time and geometry. It yields a supported
contradiction; it does not by itself prove which sender lied.

## Lineage authentication and independence

Lineage fields are claims and can be fabricated. An original observation is
authenticated only by its origin observer's signature. A relay carries the
original signed observation, or a hash resolvable from an authenticated
evidence store of signed originals. If the original cannot be verified, the
relay receives no origin credit.

Four levels are kept distinct: different observation IDs, different sensors,
different physical observers, and independent evidence.

- Successive observations of one object by one sensor are **one witness**.
- Sensors on the same physical observer share a common cause.
- Derived claims inherit their parents: a LiDAR used to lift a camera
  detection cannot also confirm that detection's geometry.
- Independent witnesses are distinct physical observers with authenticated
  evidence, neither derived from the other, without an obvious common cause
  (same upstream sensor, calibration or time source).

Relays, tracks, fusions and verdicts inherit their parents' roots. Five relays
of one observation = one witness.

## Freshness and replay

Honest packets arrive late and out of order. Consumers use: capture-time
TTL, `valid_until`, a clock tolerance for future timestamps, duplicate
`message_id` detection, and a **bounded per-source replay window** on
sequence numbers (out-of-order inside the window is accepted). Accepted older
observations never replace newer track state.

## Two-rate fusion

- **Every packet, deterministic, real-time:** envelope and signature check,
  freshness, replay window and expiry, frame and bounds, coverage consistency, kinematic
  plausibility, free-space contradiction, lineage-based independence, safety
  asymmetry.
- **On conflict, on a user question, or at ≤ 2 Hz: Muse Reasoner.** Query
  compilation, observer selection (coverage, freshness, independence, latency,
  cost), active verification, verdicts with evidence references. Its output is
  validated by deterministic code: referenced IDs must exist, geometry must be
  finite and in frame, permissions must allow the requested queries.

Muse is supervisory, never in the motion control loop.

## Heterogeneous participants

| Node | Emits | Answers | Gets from network |
|---|---|---|---|
| Pole RGB camera + SAM 3.1 | Classified objects, open labels, camera coverage (no free space) | T0–T3 | — |
| Pole RGB-D camera + SAM 3.1 | Classified objects lifted with depth, open labels, coverage **with depth free space** | T0–T3 | — |
| Pole/vehicle LiDAR, no LLM | Geometric objects, coverage **with free space** | T0–T2 | Objects beyond its view |
| AMR with 2D scanner | Objects in scan plane; free space for that plane and height band | T0–T2 | Objects beyond its scan |
| Radar | Objects with good velocity, coarse position (region) | T0–T2 | — |
| UWB/RFID tags | `self` reports | T0–T1 | — |
| Sensorless robot | `self` pose | — | Everything; can call the Muse Reasoner |
| Ego car + Muse | Own detections | T0–T3 | Reasoner for itself and others |

## Bandwidth accounting

Count the **complete exchange**, not just object records: capability,
coverage, query, query_ack, answer, label table, verdict, and authentication
overhead (Ed25519 signature 64 B + key reference). The 84-byte CX1 object
record is one part of that. Report bytes per useful hidden-object detection.

## Encoding and transport

- Compact binary records (CX1 today; CX2 adds envelope, `kind`, `geometry`,
  `label_id`, lineage). JSON mirror for logs.
- Transport-agnostic: radio broadcast, DDS/ROS 2, UDP, MQTT. A ROS 2 bridge
  is the robot adoption path (expansion, not hackathon).
- Identity, trust classes, Sybil handling, opt-in and privacy:
  [security.md](security.md).

## Relation to ETSI CPM

ETSI CPM already includes perception regions, confidence and shadowing
concepts *(verify exact text before citing)*. We do not claim those ideas. The
contribution is making them usable on demand: discovery, tiered queries,
active verification with lineage, heterogeneous non-LLM participants, and a
model-backed Reasoner service.

## Hackathon build order

1. CX2 envelope + lineage, coverage messages (camera vs range semantics).
2. Deterministic checker (two-rate fast path) and safety asymmetry.
3. Discover/Observe/Verify as tools called by Muse Spark; T3 via SAM 3.1.
4. Answer results and verdicts with evidence references.
5. Ed25519 identities with trust classes (see security.md).
