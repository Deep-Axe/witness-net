# Security, trust and participation model — draft v0.2

**Status:** design draft, 2026-11-02. Companion to [protocol.md](protocol.md).
Items marked *(verify)* come from memory of V2X security standards and must be
checked before they reach a pitch or submission page.

## Ground rules

1. **A signature proves who sent a message, not that it is true.** Crypto stops
   outsiders and makes insiders accountable. A certified but compromised, faulty
   or fooled sensor still sends signed false claims. There are always two
   layers: **authentication** (who) and **plausibility** (is it consistent with
   other evidence).
2. **Never majority-vote on identities.** Weight evidence by lineage
   independence, coverage and line of sight, not by message count.
3. **Safety asymmetry.** Remote claims may add caution; they may never delete
   or overwrite a local detection. Disagreement between local and remote
   evidence stays **explicit** (contested), because local sensors can also be
   wrong. See enforcement below.
4. **Contested → unknown → conservative**, within bounds (below). Do not resolve
   what the evidence cannot resolve; report it.
5. **Degraded operation depends on local capability.** A robot with its own
   sensors falls back to standalone operation. A robot that relies on the
   network for perception (sensorless, or blind in a region) must stop or enter
   a restricted operating mode when that coverage expires.

## Bounded advisories (what safety asymmetry actually guarantees)

"Worst case is a slowdown" is not true on its own: braking can create hazards,
and repeated false hazards can immobilize a robot. So remote hazards produce
**advisories**, and the robot's **local motion controller** enforces the bounds:

- An advisory has a region, severity, expiry (`valid_until`) and evidence references.
- Maximum effect per advisory class is set locally (e.g. reduce speed to a
  limit, widen margins), never an unbounded command. Emergency braking requires
  local or independently corroborated evidence.
- Advisories expire unless renewed by fresh evidence.
- Per-source and per-lineage-cluster advisory rate limits; repeated
  uncorroborated advisories from one cluster are reported, and their effect is
  capped.
- The guarantee we claim: **a remote claim cannot remove a local detection, and
  its effect is bounded and expiring.** We do not claim remote lies are harmless.

Metric: count of local positive detections flipped to absent by remote input
(must be 0), and distribution of advisory effects and durations under attack.

## Identity and authorization

Reuse existing V2X security rather than inventing it *(verify specifics)*:

- IEEE 1609.2 / ETSI TS 103 097 signed messages with generation time and location.
- PKI: US SCMS, EU CCMS. Vehicles hold short-lived **pseudonym certificates**,
  several valid at once; only the misbehaviour authority can link them.
- Certificates carry **permissions** (PSID/SSP): allowed message types and, for
  infrastructure, typically a region.

**Demonstrator:** Ed25519 key pairs per node, a static table of trust anchors,
signed envelopes. Not a shared HMAC key: anyone holding a shared key can forge
messages for others, which destroys accountability and transferable evidence.

| Trust class | Issued by | Example | Maximum effect |
|---|---|---|---|
| A: certified infrastructure | Road/site authority, region-bound | Intersection pole, ceiling cameras | Bounded advisory; coverage/free space usable in checks |
| B: certified peers | Public PKI, pseudonymous | Other cars, delivery robots | Bounded advisory; independent evidence only if lineage-independent and anchored |
| C: same fleet / site | Operator CA, linkable | Our AMRs | Bounded advisory; shared history |
| D: unauthenticated | — | Anything else | Logged/displayed; may mark a region unknown; never evidence |

**Trust class is not independence.** Infrastructure and same-fleet devices can
share calibration, software, upstream sensors or a time source. Independence is
decided by lineage and common-cause analysis (protocol.md), not by class.

Capability messages are claims; certificate permissions are authorization. An
observer claiming coverage outside its permitted region is misbehaving.

## Sybil and swarm attacks

### Many identities, few bodies

Pseudonym PKI alone allows one insider several valid identities.

**Physical anchoring:** a peer's `self` claim is anchored when a perceived body
from our own sensors or a class A/C observer is associated with it. Record:

- **Association confidence** (how well the claimed pose/size/motion matches the
  observed body), not a yes/no.
- That anchoring is **not identity binding.** Seeing a car where a message says
  it is does not prove that car sent the message. Several identities can claim
  the same visible body; only one association per body is allowed, and
  duplicates are flagged.

Unanchored identities contribute at most `unverified`. Evidence is counted by
**lineage roots**, so ten pseudonyms relaying or echoing one observation are
one source.

New identities get a small advisory budget; suspiciously correlated claims are
treated as one cluster.

### Many real bodies, one adversary

A deployed swarm of real, certified robots passes anchoring. What still holds:
no voting; coverage and line of sight decide relevance; lineage and
common-cause analysis decide independence; own sensors plus active
verification of a specific class A/C observer; safety asymmetry and bounded
advisories cap the impact; signed claims allow attribution afterwards. This is
a stated limitation, not a solved problem.

## Infrastructure vs many vehicles disagreeing

Example: the pole reports "pedestrian at P"; five vehicles report nothing at P.

1. **Coverage:** which vehicles actually inspected P at that time, with what
   detectable size and height? `not_detected` from a camera is not
   `observed_free`. Vehicles without coverage are not evidence of absence.
2. **Anchoring and lineage independence** of the vehicles.
3. **Own sensors, then active verification** (T2 to a specific observer of a
   different modality where possible).
4. A supported free-space contradiction from a range sensor creates a
   **contested** state with evidence on both sides. It does not by itself prove
   which sender lied.
5. The hazard remains as a bounded advisory until resolved or expired; absence
   claims cannot delete it.
6. **Infrastructure health is an operator question.** Passing anchored traffic
   provides continuous evidence; discrepancies go to the operator as signed
   misbehaviour reports. Vehicles talk back mainly for this, not to override
   the pole in real time.

## Participation: opt-in, sharing and receiving

**Receive policy:** accepted trust anchors; maximum effect per trust class;
rate limits and advisory budgets.

**Share policy:** modes **listen-only** (default), **query**, **contribute**,
**report**; what is shared and at what precision; to whom (public, site, fleet,
emergency services); where (geofenced).

A robot shares when policy and location allow and it has something useful:
coverage of a queried/subscribed region, a hazard near a declared route, or a
misbehaviour report. Infrastructure broadcasts; robots mostly answer and report.

## Model-specific risks

- **Prompt injection via labels or strings.** Labels come from a bounded,
  versioned label table and are passed to Muse as data. JSON-schema output and
  "IDs must come from supplied candidates" constrain the output **structure**;
  they do not guarantee Muse's judgement is unaffected by malicious content.
  Therefore Muse verdicts never override deterministic checks or safety
  asymmetry, and every verdict's evidence references are validated.
- **Reasoner as a source.** A remote Reasoner's verdict is a derived claim with
  lineage and a trust class, not ground truth.

## Privacy

Object-level sharing carries no faces, plates, images or embeddings; keep it
that way. Pedestrians get no persistent cross-node IDs. Non-fleet robots use
rotating pseudonyms. Queries are sensitive: authorize, rate-limit, and answer
at reduced precision when appropriate.

## Other threats

Deferred; tracked in an internal backlog (not published).

## Hackathon scope

- Ed25519 identities, trust classes A–D, trust anchor table.
- Safety asymmetry and bounded, expiring advisories, with the flip-count metric.
- Lineage-based independence; Sybil of N unanchored identities → independent
  weight 0; relayed duplicates count once.
- Pole-vs-majority scene: peers without coverage cannot veto; contested state;
  misbehaviour report.
- Replay with old capture time fails freshness.
- Label table + schema-bound Muse output with validated evidence references.
- Real-body swarm stated as a limitation with bounded, expiring impact.
