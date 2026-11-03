# Perception Network for Physical AI

> Give any robot access to the senses around it. Machines discover observers,
> ask about the world, and cross-check the answers, even when the robot has no
> camera or language model of its own.

Robots only see what their own sensors see. This project lets a robot use the
sensors around it, starting with roadside infrastructure, to see **beyond its
own line of sight**. Observers run perception where the view is good and share
typed object observations, never camera feeds. The first application is
driving: a roadside unit sees a pedestrian hidden behind a parked truck, and
the car learns about it from a few hundred bytes.

The interaction model is **Discover / Observe / Verify**:

- **Discover** which observers can see a region, with what sensors.
- **Observe**: ask an observer what is there, including open-vocabulary
  requests ("find a scooter behind the bus").
- **Verify**: when reports conflict, ask a second observer, ideally of a
  different modality, and keep the evidence chain.

(Tool-style discovery for perception. This is an analogy, not Anthropic's
Model Context Protocol.)

## Built on Meta models

| Model | Role |
|---|---|
| **Muse Spark 1.3** | The network **Reasoner**: turns a question into typed queries, picks which observers to ask, runs active verification, and issues verdicts linked to the evidence used |
| **SAM 3.1** | Turns any camera into an **open-vocabulary observer**. The observer's backend lifts SAM masks to 3D with depth or LiDAR |
| **Muse Voice Transcribe** | Spoken questions to the Reasoner |
| Muse Glimmer / Muse Spark Open | The same Reasoner at the edge or self-hosted (design direction) |

Robots without a language model still participate: they exchange typed
observations and can call the Reasoner as a network service. Muse is
supervisory: deterministic checks run on every message; Muse runs on
conflicts, questions, or at up to 2 Hz.

## Design principles

- **Typed observation claims, not prose.** Language appears only in
  open-vocabulary queries, never in observations.
- **Silence is not free space.** Observers declare what they actually
  covered; only range measurements (LiDAR, depth) can show space is empty.
- **Count independent observations, not repeated messages.** Every claim
  carries lineage.
- **A signature proves who sent a claim, not that it is true.**
- **Remote claims may add caution, never delete a robot's own detections.**

Details: [protocol draft](docs/protocol.md) ·
[security, trust and participation](docs/security.md).

## Status

| Part | State |
|---|---|
| Protocol and security design | Draft v0.2 |
| Compact object record (CX1, 84 bytes), delayed transport, stale detection, depth lifting | Implemented on synthetic data |
| CX2 signed observations, Ed25519 identities, authenticated lineage and witness accounting | Implemented; round-trip, tampering, relay and dependency tests pass |
| Meta API access (Spark JSON + function calls, SAM image/video, Voice) | Spark 1.3 and SAM 3.1 integrated in S3 (cached client with cost log and budget); Voice probed only |
| Deterministic checker (signature, freshness, future time, duplicates, bounded replay window, bounds) | Implemented; honest reordered traffic accepted, replay/forgery rejected |
| S1 slice: signed observations → packet transport (delay, jitter, loss, disconnect) → checker → JSONL event log → browser viewer | Implemented; tested end to end; viewer renders log text safely |
| S2 slice: signed LiDAR coverage (`CX2C` bitmaps), per-source coverage map with contested cells, deterministic assessment (contradicted by free space, position-only support, contested, unverified) | Implemented on a synthetic 2-D world, no LLM; a ghost in LiDAR free space is contradicted, a ghost where no range sensor looks stays unverified, two cameras against LiDAR free space are contested |
| S3 slice: one V2X-Real frame; SAM 3.1 at an infrastructure camera; detections lifted with that observer's LiDAR; signed, checked; Muse Spark plans queries, observers answer, Spark judges with evidence refs | Implemented; one live run on one frame (scenario `2023-04-04-14-28-53_45_0`, frame 000096). Lifted pedestrian 0.16 m from GT. Model verdicts cannot exceed the evidence (one observer is never "corroborated") |
| V2X-Real reader and projection calibration | Implemented; three frames, six cameras, and RSU → vehicle projections inspected; capture synchronization unproven |
| Tests | 203 passing (no network; recorded/fake model responses) |
| Dataset replay ([V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/) validation split) | Data extracted; replay integration pending |
| Scene mining, attacks, evaluation, multi-frame replay, full demo UI | Not started |

A successful synthetic run does not imply the dataset or model integrations work.

## Local setup

Python 3.12. Dependencies include NumPy below 2, cryptography, Pillow,
OpenCV headless and PyYAML. OpenCV is constrained below 4.12 for NumPy 1.x.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --editable .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m coopx synthetic --output data/example-events.jsonl
.venv/bin/python -m coopx replay-packets data/example-events.jsonl
```

Slice S1 (signed observations through a lossy, jittery link to a checking
consumer) and its viewer:

```sh
.venv/bin/python -m coopx s1 --output data/s1-events.jsonl --overwrite --loss 0.1 --jitter-us 150000 --disconnect-at-us 3000000
.venv/bin/python scripts/serve_ui.py      # localhost only; serves ui/ and data/*.jsonl, never .env
# open http://127.0.0.1:8765/ui/index.html?log=/data/s1-events.jsonl
```

Slice S2 (LiDAR free space against camera ghosts, no LLM). Click the
contested cell on the map to see the LiDAR free claim and the ghost's
occupied claim:

```sh
.venv/bin/python -m coopx s2 --output data/s2-events.jsonl --overwrite --key-seed 1
# open http://127.0.0.1:8765/ui/index.html?log=/data/s2-events.jsonl
```

No API key or credits are needed for these checks. The synthetic command
refuses to overwrite an existing file. Expected: a synthetic masked depth plane
becomes a world-space observation, encodes to 84 ASCII bytes, and crosses a
delayed transport (168 bytes sent and received for two transmissions). The
repeated packet keeps its capture time, so the later delivery is stale.

Slice S3 (one dataset frame; needs the V2X-Real validation split under
`data/v2x-real/val` and `MODEL_API_KEY`; makes one SAM call and two Spark calls,
about $0.01 at published rates, then replays from `data/meta-cache/` for free):

```sh
.venv/bin/python -m coopx s3 --output data/s3-events.jsonl --overwrite --budget-usd 0.5
.venv/bin/python -m coopx s3 --output data/s3-events.jsonl --overwrite --offline   # cache only
# open http://127.0.0.1:8765/ui/index.html?log=/data/s3-events.jsonl
```

Paid calls are logged to `data/cost-log.jsonl`. Cached replays report the
original measured latency, so a replay is never shown as instant.

Meta API access check (reads `MODEL_API_KEY` from the environment or `.env`):

```sh
python scripts/check_meta_api.py            # catalog only, no inference cost
python scripts/check_meta_api.py --model muse-spark-1.3   # one small billed call
```

## Deployment target

One laptop (RTX 3060, 6 GB). SAM 3.1 and Muse Spark run as hosted Meta
inference; dataset replay, transport, checks, attacks and metrics run locally.
No simulator or cloud GPU.
