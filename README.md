# Witness

Witness is a perception network for robots. It lets a robot query nearby
cameras and sensors for observations beyond its own field of view.

A robot can ask about a location, request detections of a new object category,
or seek another view when reports disagree. Observer capabilities describe
sensor coverage and supported queries, helping the system choose whom to ask.
The goal is a common perception interface across vehicles, mobile robots and
fixed infrastructure, including machines that cannot run the models themselves.

A roadside camera, for example, can report a pedestrian that an approaching
car cannot see. Observers exchange signed object reports with positions,
timestamps, uncertainty and provenance. The receiving robot checks the reports
and compares evidence from different sources.

SAM 3.1 provides text-prompted object segmentation at camera observers. Muse
Spark turns questions into observer queries and assesses the returned evidence.
Packet admission and geometric checks run separately from model reasoning.

The prototype includes a Python library, replay commands and a browser viewer.
Development uses simulated sensor traffic and recorded V2X-Real scenes.

## Installation

Witness requires Python 3.12. From the repository root, create an environment
and install the package with `uv`:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --editable .
```

The Python package and command-line entry point are named `coopx`.

## Basic usage

Run the synthetic camera and LiDAR scenario:

```sh
.venv/bin/python -m coopx s2 --output data/s2.jsonl --overwrite --key-seed 1
```

This writes an event log containing signed camera reports, simulated packet
delivery, LiDAR coverage and evidence assessments. It runs locally without a
dataset or API key. The `--overwrite` flag replaces an existing output log.

Start the viewer:

```sh
.venv/bin/python scripts/serve_ui.py
```

Open [the replay](http://127.0.0.1:8765/ui/index.html?log=/data/s2.jsonl) to
inspect the timeline, coverage map and evidence behind each assessment.

Use `--help` to see the available commands and options:

```sh
.venv/bin/python -m coopx --help
```

## Dataset replay

The camera replay uses the
[V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/) validation split,
placed under `data/v2x-real/val/`. Set `MODEL_API_KEY` in the environment or a
local `.env` file to use Meta Model API.

```sh
.venv/bin/python -m coopx s3 --output data/s3.jsonl --overwrite --budget-usd 0.50
```

This run segments a roadside camera image, lifts detections into 3-D using the
observer's LiDAR, and runs a question through Spark. Open
[the dataset replay](http://127.0.0.1:8765/ui/index.html?log=/data/s3.jsonl) with
the same viewer server.

API calls are cached in `data/meta-cache/`; paid usage is recorded in
`data/cost-log.jsonl`. To replay a cached run without contacting the API:

```sh
.venv/bin/python -m coopx s3 --output data/s3.jsonl --overwrite --offline
```

Cached responses retain their original measured latency in the event log.

## Documentation

- [Protocol](docs/protocol.md): observation formats, queries, coverage and
  evidence assessment.
- [Security](docs/security.md): identity, provenance and the threat model.

## Development

Source code is in [`src/coopx/`](src/coopx/), the viewer in [`ui/`](ui/), and
tests in [`tests/`](tests/). Run the test suite with:

```sh
.venv/bin/python -m unittest discover -s tests
```

Tests make no API calls. Dataset tests run when the local validation split is
available and are skipped otherwise.
