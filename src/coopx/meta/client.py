"""Meta Responses client: disk cache, published-rate cost, contract-5 model_call.

The API key is read from the environment or a local `.env` and is never written
to cache files, cost logs, events or exception text.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from coopx.transport import make_event

API_URL = "https://api.meta.ai/v1/responses"
SCHEMA_VERSION = "coopx.meta.client/1"
ALLOWED_MODELS = frozenset({"muse-spark-1.3", "sam-3.1"})
SPARK_IN_USD = 1.25 / 1_000_000
SPARK_CACHED_USD = 0.15 / 1_000_000
SPARK_OUT_USD = 4.25 / 1_000_000
SAM_IMAGE_USD = 2.50 / 1_000
DEFAULT_BUDGET_USD = 3.0
DEFAULT_TIMEOUT_S = 45.0
MAX_RETRIES = 2
# Hard cap on Spark output (incl. reasoning) so a call's worst-case cost is bounded.
SPARK_MAX_OUTPUT_TOKENS = 4000


class CacheMiss(Exception):
    """offline=True and the request is not in the disk cache."""


class BudgetExceeded(Exception):
    """The next uncached call would exceed the remaining budget."""


class MetaAPIError(Exception):
    """HTTP or network failure after redacting any key material."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ModelResult:
    text: str
    json: dict | None
    call_id: str
    product: str
    cache_hit: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    images: int
    cost_usd: float
    query_time_us: int
    availability_time_us: int
    precomputed: bool
    stage: str = "segment"


def load_api_key(*, environ: dict | None = None, env_path: Path | None = None) -> str:
    """Return MODEL_API_KEY. Never print the value."""
    env = os.environ if environ is None else environ
    value = (env.get("MODEL_API_KEY") or "").strip()
    if value:
        return value
    path = Path(".env") if env_path is None else Path(env_path)
    if not path.is_file():
        for parent in Path.cwd().resolve().parents:
            candidate = parent / ".env"
            if candidate.is_file():
                path = candidate
                break
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, raw = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() == "MODEL_API_KEY":
                value = raw.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    return value
    raise MetaAPIError("MODEL_API_KEY is not configured")


def _redact(text: str, key: str | None) -> str:
    if not key:
        return text
    return text.replace(key, "[REDACTED]")


def _canonicalize_input(value):
    if isinstance(value, str):
        return {"text": value, "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
    if isinstance(value, dict):
        if "image_url" in value and isinstance(value["image_url"], str):
            url = value["image_url"]
            payload = url.split(",", 1)[-1] if url.startswith("data:") else url
            digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
            copied = {k: v for k, v in value.items() if k != "image_url"}
            copied["image_sha256"] = digest
            return {k: _canonicalize_input(v) for k, v in copied.items()}
        return {k: _canonicalize_input(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonicalize_input(item) for item in value]
    return value


def request_fingerprint(*, model: str, instructions: str, input, text_format,
                        reasoning_effort: str) -> str:
    canonical = {
        "schema": SCHEMA_VERSION,
        "model": model,
        "instructions": instructions,
        "input": _canonicalize_input(input),
        "text_format": text_format,
        "reasoning_effort": reasoning_effort,
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def estimate_cost(product: str, *, input_tokens: int, output_tokens: int,
                  cached_tokens: int, images: int) -> float:
    if product == "sam-3.1":
        return round(max(images, 1) * SAM_IMAGE_USD, 8)
    billed_in = max(0, input_tokens - cached_tokens)
    return round(billed_in * SPARK_IN_USD + cached_tokens * SPARK_CACHED_USD
                 + output_tokens * SPARK_OUT_USD, 8)


def _extract_text(payload: dict) -> str:
    if isinstance(payload.get("text"), str) and payload["text"]:
        return payload["text"]
    chunks = []
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") in ("output_text", "text") and part.get("text"):
                chunks.append(part["text"])
    return "".join(chunks)


def _usage(payload: dict) -> tuple[int, int, int, int]:
    usage = payload.get("usage") or {}
    details_in = usage.get("input_tokens_details") or {}
    return (
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
        int(details_in.get("cached_tokens") or 0),
        int(usage.get("images_processed") or 0),
    )


def _collect_stream(lines) -> dict:
    """Server-sent events -> {"text", "usage", "output"} (same shape as a JSON reply)."""
    deltas, final = [], {}
    for raw in lines:
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        kind = event.get("type")
        if kind == "response.output_text.delta" and isinstance(event.get("delta"), str):
            deltas.append(event["delta"])
        elif kind in ("response.completed", "response.failed", "response.incomplete"):
            final = event.get("response") or {}
            if kind != "response.completed":
                raise MetaAPIError(f"Meta API stream ended with {kind}")
    return {"text": "".join(deltas), "usage": final.get("usage") or {},
            "output": final.get("output") or [], "status": final.get("status")}


def default_post(url: str, headers: dict, body: bytes, timeout: float):
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if headers.get("Accept") == "text/event-stream":
            return response.status, _collect_stream(response)
        return response.status, json.loads(response.read().decode("utf-8"))


class MetaClient:
    def __init__(self, *, cache_dir: str | Path = "data/meta-cache",
                 cost_log: str | Path = "data/cost-log.jsonl",
                 budget_usd: float = DEFAULT_BUDGET_USD, offline: bool = False,
                 timeout_s: float = DEFAULT_TIMEOUT_S, post=None, clock=None,
                 environ: dict | None = None, env_path: Path | None = None):
        env = os.environ if environ is None else environ
        if env.get("COOPX_BUDGET_USD"):
            budget_usd = float(env["COOPX_BUDGET_USD"])
        if not (isinstance(budget_usd, (int, float)) and budget_usd >= 0):
            raise ValueError("budget_usd must be a nonnegative number")
        self.cache_dir = Path(cache_dir)
        self.cost_log = Path(cost_log)
        self.budget_usd = float(budget_usd)
        self.offline = bool(offline)
        self.timeout_s = float(timeout_s)
        self._post = default_post if post is None else post
        self._clock = time.monotonic if clock is None else clock
        self._environ = env
        self._env_path = env_path

    def logged_cost_usd(self) -> float:
        """Spent plus outstanding reservations (possibly-billed calls count)."""
        if not self.cost_log.is_file():
            return 0.0
        with self._locked_ledger() as handle:
            return self._ledger_total(handle)

    # -- Ledger: one append-only JSONL file shared by every client/process. ---
    # Rows: legacy/settled spend {"cost_usd"}; {"kind":"reserve","rid","amount"};
    # {"kind":"settle","rid","cost_usd",...}; {"kind":"release","rid"}. A
    # reservation without settle/release (timeout, 5xx, crash) stays counted.

    def _locked_ledger(self):
        import contextlib
        import fcntl

        @contextlib.contextmanager
        def locked():
            self.cost_log.parent.mkdir(parents=True, exist_ok=True)
            with self.cost_log.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield handle
                finally:
                    handle.flush()
                    os.fsync(handle.fileno())
                    fcntl.flock(handle, fcntl.LOCK_UN)
        return locked()

    @staticmethod
    def _ledger_total(handle) -> float:
        handle.seek(0)
        spent, outstanding = 0.0, {}
        for line in handle.read().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "reserve":
                outstanding[row["rid"]] = float(row["amount"])
            elif kind in ("settle", "release"):
                outstanding.pop(row.get("rid"), None)
                spent += float(row.get("cost_usd") or 0.0)
            else:
                spent += float(row.get("cost_usd") or 0.0)
        return spent + sum(outstanding.values())

    @staticmethod
    def _write_row(handle, row: dict) -> None:
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    def _reserve(self, product: str, amount: float) -> str:
        """Atomically check budget (spent + outstanding + amount) and record a reservation."""
        import uuid
        with self._locked_ledger() as handle:
            total = self._ledger_total(handle)
            if self.budget_usd <= 0 or total + amount > self.budget_usd + 1e-12:
                raise BudgetExceeded(
                    f"ledger {total:.6f} USD plus a {amount:.6f} USD reservation for the next "
                    f"{product} call would exceed budget {self.budget_usd:.4f} USD")
            rid = uuid.uuid4().hex
            self._write_row(handle, {"kind": "reserve", "rid": rid, "product": product,
                                     "amount": round(amount, 8)})
            return rid

    def _close_reservation(self, rid: str, row: dict) -> None:
        with self._locked_ledger() as handle:
            self._write_row(handle, {"rid": rid, **row})

    def _cache_path(self, digest: str) -> Path:
        return self.cache_dir / f"{digest}.json"

    def _load_cache(self, digest: str) -> dict | None:
        path = self._cache_path(digest)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _store_cache(self, digest: str, record: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(digest)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def reserve_usd(product: str, *, instructions: str, input, text_format=None) -> float:
        """Worst-case cost of ONE attempt, reserved before dispatch.

        SAM bills per image. Spark: at most one input token per serialized byte
        (byte-level tokenizers never emit more tokens than bytes), counting the
        instructions, input and output schema, plus the full output cap.
        """
        if product == "sam-3.1":
            return SAM_IMAGE_USD
        size = (len(instructions.encode("utf-8"))
                + len(json.dumps(input, allow_nan=False, sort_keys=True).encode("utf-8"))
                + len(json.dumps(text_format, allow_nan=False, sort_keys=True).encode("utf-8")))
        return size * SPARK_IN_USD + SPARK_MAX_OUTPUT_TOKENS * SPARK_OUT_USD

    def _live_call(self, *, model: str, instructions: str, input, text_format,
                   reasoning_effort: str) -> dict:
        key = load_api_key(environ=self._environ, env_path=self._env_path)
        if model == "sam-3.1":
            # Documented SAM request: streamed, concept + image only, no
            # instructions/reasoning (dev.meta.ai/docs/sam/segmenting).
            payload = {"model": model, "input": input, "stream": True,
                       "metadata": {"include_confidence": "true"}}
            accept = "text/event-stream"
        else:
            payload = {
                "model": model,
                "instructions": instructions,
                "input": input,
                "reasoning": {"effort": reasoning_effort},
                "store": False,
            }
            if text_format is not None:
                payload["text"] = {"format": text_format}
            payload["max_output_tokens"] = SPARK_MAX_OUTPUT_TOKENS
            accept = "application/json"
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {
            "Authorization": "Bearer " + key,
            "Accept": accept,
            "Content-Type": "application/json",
        }
        last_error = None
        attempts = 1 + MAX_RETRIES
        for attempt in range(attempts):
            try:
                status, response = self._post(API_URL, headers, body, self.timeout_s)
                if status >= 500:
                    last_error = MetaAPIError(
                        _redact(f"Meta API HTTP {status}", key), status=status)
                    if attempt + 1 < attempts:
                        continue
                    raise last_error
                if status >= 400:
                    raise MetaAPIError(_redact(f"Meta API HTTP {status}", key), status=status)
                return response
            except MetaAPIError:
                raise
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                message = _redact(f"Meta API HTTP {status}", key)
                if 500 <= status < 600 and attempt + 1 < attempts:
                    last_error = MetaAPIError(message, status=status)
                    continue
                raise MetaAPIError(message, status=status) from None
            except (TimeoutError, urllib.error.URLError) as exc:
                status = getattr(exc, "code", None)
                if status is not None and 400 <= int(status) < 500:
                    raise MetaAPIError(_redact(f"Meta API HTTP {status}", key),
                                       status=int(status)) from None
                message = _redact(f"Meta API network error: {exc}", key)
                if attempt + 1 < attempts:
                    last_error = MetaAPIError(message)
                    continue
                raise MetaAPIError(message) from None
        raise last_error or MetaAPIError("Meta API request failed")

    def respond(self, *, model: str, instructions: str, input: list | str,
                query_time_us: int, text_format: dict | None = None,
                reasoning_effort: str = "minimal", stage: str,
                precomputed: bool = False) -> ModelResult:
        if model not in ALLOWED_MODELS:
            raise ValueError(f"unsupported model {model!r}; Standard ids only")
        if type(query_time_us) is not int or query_time_us < 0:
            raise ValueError("query_time_us must be a nonnegative int")
        digest = request_fingerprint(
            model=model, instructions=instructions, input=input,
            text_format=text_format, reasoning_effort=reasoning_effort)
        call_id = digest[:16]
        cached = self._load_cache(digest)
        if cached is not None:
            latency_ms = float(cached["latency_ms"])
            availability = query_time_us + int(round(latency_ms * 1000))
            parsed = cached.get("json")
            return ModelResult(
                text=cached["text"], json=parsed, call_id=call_id, product=model,
                cache_hit=True, latency_ms=latency_ms,
                input_tokens=int(cached["input_tokens"]),
                output_tokens=int(cached["output_tokens"]),
                # A replay spends nothing; latency is replayed for time honesty only.
                images=int(cached["images"]), cost_usd=0.0,
                query_time_us=query_time_us, availability_time_us=availability,
                precomputed=precomputed, stage=stage)
        if self.offline:
            raise CacheMiss(f"no cached response for {model} ({stage})")
        # Every attempt may be billed (a timed-out request is not proof of no charge).
        reserve = self.reserve_usd(model, instructions=instructions, input=input,
                                   text_format=text_format) * (1 + MAX_RETRIES)
        rid = self._reserve(model, reserve)
        started = self._clock()
        try:
            payload = self._live_call(
                model=model, instructions=instructions, input=input,
                text_format=text_format, reasoning_effort=reasoning_effort)
        except MetaAPIError as exc:
            if exc.status is not None and 400 <= exc.status < 500:
                # Rejected before inference: not billed.
                self._close_reservation(rid, {"kind": "release", "cost_usd": 0.0})
            # Timeouts, 5xx and network errors keep the reservation (maybe billed).
            raise
        latency_ms = max(0.0, (self._clock() - started) * 1000.0)
        text = _extract_text(payload)
        parsed = None
        if text_format is not None and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        input_tokens, output_tokens, cached_tokens, images = _usage(payload)
        if model == "sam-3.1" and images < 1:
            images = 1
        cost = estimate_cost(model, input_tokens=input_tokens, output_tokens=output_tokens,
                             cached_tokens=cached_tokens, images=images)
        record = {
            "text": text, "json": parsed, "latency_ms": latency_ms,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "images": images, "cost_usd": cost, "product": model,
        }
        self._store_cache(digest, record)
        self._close_reservation(rid, {
            "kind": "settle", "call_id": call_id, "product": model, "stage": stage,
            "cost_usd": cost, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "images": images,
            "latency_ms": latency_ms,
        })
        availability = query_time_us + int(round(latency_ms * 1000))
        return ModelResult(
            text=text, json=parsed, call_id=call_id, product=model,
            cache_hit=False, latency_ms=latency_ms, input_tokens=input_tokens,
            output_tokens=output_tokens, images=images, cost_usd=cost,
            query_time_us=query_time_us, availability_time_us=availability,
            precomputed=precomputed, stage=stage)


def model_call_event(result: ModelResult, source_id: str) -> dict:
    """Contract-5 model_call, logged at the result's availability time."""
    return make_event(
        t_replay_us=result.availability_time_us, type="model_call",
        source_id=source_id, message_id=result.call_id,
        detail={
            "product": result.product, "stage": result.stage,
            "cache_hit": result.cache_hit, "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "images": result.images, "cost_usd": result.cost_usd,
            "query_time_us": result.query_time_us,
            "availability_time_us": result.availability_time_us,
            "precomputed": result.precomputed,
        })
