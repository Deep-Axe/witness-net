"""Serve the viewer and event logs on localhost only.

Unlike `python -m http.server` from the repo root, this never exposes `.env`,
`internal/`, source or dataset files, and binds to 127.0.0.1 by default.
Allowed: files under `ui/`, and `*.jsonl` / `*.png` / `*.jpg` under `data/`
(excluding the raw dataset folder `data/v2x-real/`).
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
DATA_SUFFIXES = {".jsonl", ".png", ".jpg"}


def allowed(url_path: str) -> bool:
    path = unquote(urlsplit(url_path).path)
    if path in ("", "/"):
        return True
    target = (ROOT / path.lstrip("/")).resolve()
    try:
        relative = target.relative_to(ROOT)
    except ValueError:
        return False
    parts = relative.parts
    if not parts or any(part.startswith(".") for part in parts):
        return False
    if parts[0] == "ui":
        return True
    return (parts[0] == "data" and len(parts) > 1 and parts[1] != "v2x-real"
            and target.suffix in DATA_SUFFIXES)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if urlsplit(self.path).path in ("", "/"):
            self.send_response(302)
            self.send_header("Location", "/ui/index.html")
            self.end_headers()
            return
        if not allowed(self.path):
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if not allowed(self.path):
            self.send_error(404)
            return
        super().do_HEAD()

    def list_directory(self, path):
        self.send_error(404)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), partial(Handler, directory=str(ROOT)))
    print(f"Serving viewer at http://{args.bind}:{args.port}/ui/index.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
