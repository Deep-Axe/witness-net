"""Check Meta API access without printing credentials. Catalog lookup is default."""

import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.meta.ai/v1"


def load_key():
    if os.environ.get("MODEL_API_KEY"):
        return os.environ["MODEL_API_KEY"].strip()
    path = Path(".env")
    if path.is_file():
        for line in path.read_text().splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() == "MODEL_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    return value
    raise SystemExit("MODEL_API_KEY is not configured")


def request(key, endpoint, payload=None):
    headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
    body = None if payload is None else json.dumps(payload).encode()
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE_URL + endpoint, data=body, headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return {"http_status": response.status, "elapsed_s": round(time.monotonic() - started, 3),
                    "response": json.loads(response.read())}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace").replace(key, "[REDACTED]")
        try:
            error = json.loads(raw)
        except ValueError:
            error = {"message": raw[:500]}
        return {"http_status": exc.code, "elapsed_s": round(time.monotonic() - started, 3),
                "error": error}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"network_error": str(exc).replace(key, "[REDACTED]")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Make one small, potentially billed text response request")
    args = parser.parse_args()
    key = load_key()
    if args.model:
        result = request(key, "/responses", {
            "model": args.model, "input": "Reply with exactly OK.",
            "max_output_tokens": 64, "reasoning": {"effort": "minimal"}, "store": False,
        })
        if "response" in result:
            response = result.pop("response")
            result.update(model=response.get("model"), status=response.get("status"),
                          usage=response.get("usage"),
                          output=[item for item in response.get("output", [])
                                  if item.get("type") != "reasoning"])
    else:
        result = request(key, "/models")
        if "response" in result:
            response = result.pop("response")
            result["models"] = response.get("data", response)
    print(json.dumps(result, indent=2).replace(key, "[REDACTED]"))
    if "error" in result or "network_error" in result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
