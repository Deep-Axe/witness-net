# Witness

**A perception network for physical AI.** Robots discover nearby cameras and
sensors, ask them to observe what's out of view, and verify the answers before
acting on them.

A car can't see the pedestrian behind a parked truck. The roadside camera on
the corner can. Witness lets the car ask it, receive a signed, typed claim
about what was seen (a few hundred bytes, never a video feed), and decide how
far to trust it.

## How it works

1. **Discover**: find observers whose coverage includes the region you care about.
2. **Observe**: ask them, including open-vocabulary requests ("find a stroller
   near the crosswalk"). Any camera becomes such an observer through **SAM 3.1**.
3. **Verify**: every claim is checked deterministically. **Muse Spark** plans
   follow-up queries and judges conflicting answers, citing only evidence that
   passed the checks.

Robots without a camera or language model still take part: they exchange
typed claims and can ask the network's Reasoner.

## Rules it never breaks

- **Silence is not free space.** A camera that saw nothing proves nothing. Only
  range sensors (LiDAR, depth) can show a space is empty.
- **Count independent witnesses, not messages.** Relays and aliases of one
  observation are one witness; claims carry authenticated lineage.
- **A signature proves who sent a claim, not that it's true.**
- **Remote claims can add caution, never erase what a robot sees itself.**
- **The model never outranks the evidence.** One observer can't make a claim
  "corroborated", whatever the model says.
- **Hazards never wait on a model call.** Deterministic checks run on every
  packet; Muse is supervisory and asynchronous.

## What works today

- **Signed claims and a checker**: Ed25519 envelopes and lineage, plus freshness,
  replay, forgery and region checks, over a lossy, jittery simulated link.
- **Free space versus ghosts**: a LiDAR's free space contradicts a fake
  pedestrian. Where no range sensor looks, the claim stays unverified. Two
  cameras against LiDAR free space are contested.
- **Real data, real models**: on a [V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/)
  frame, SAM finds pedestrians in a roadside camera. They're lifted to 3-D with
  that unit's LiDAR, 0.16 m from ground truth. Spark plans queries, observers
  answer, and Spark returns a verdict linked to the signed evidence.
- **An event-log viewer**: a timeline, the coverage map with contested cells,
  and the Reasoner's plan, answers, verdict and model calls.

Next: a second independent roadside observer for corroboration, free space
from real LiDAR, attacks, and metrics across clips.

## Quick start

Python 3.12. The package is `coopx`.

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --editable .
.venv/bin/python -m unittest discover -s tests          # no network, no key

.venv/bin/python -m coopx s2 --output data/s2.jsonl --overwrite --key-seed 1
.venv/bin/python scripts/serve_ui.py                     # localhost only
# open http://127.0.0.1:8765/ui/index.html?log=/data/s2.jsonl
```

The dataset run needs the V2X-Real validation split in `data/v2x-real/val` and
a `MODEL_API_KEY`. One run makes one SAM call and two Spark calls (about $0.01).
It's then cached, and `--offline` replays it for free:

```sh
.venv/bin/python -m coopx s3 --output data/s3.jsonl --overwrite --budget-usd 0.5
```

Paid calls go to `data/cost-log.jsonl`. Cached replays keep their measured
latency, so nothing is shown as instant.

## Layout

| Path | What |
|---|---|
| `src/coopx/envelope.py`, `checker.py` | Signed claims and the deterministic checker |
| `src/coopx/coverage.py`, `assess.py` | Coverage maps and the free-space/witness rules |
| `src/coopx/query.py`, `meta/` | Queries and answers, the Meta client, SAM, the Spark Reasoner |
| `src/coopx/observers/` | Camera observer and 3-D lift, synthetic LiDAR |
| `src/coopx/pipeline.py` | End-to-end runs (`s1`, `s2`, `s3`) |
| `ui/` | Event-log viewer |
| `docs/` | [Protocol](docs/protocol.md) and [security](docs/security.md) design |
