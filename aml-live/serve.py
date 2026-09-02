"""Serve the live view. Standard library only — no Flask, no extra installs.

    python aml-live/serve.py              # http://127.0.0.1:8000
    python aml-live/serve.py --port 9000
    python aml-live/serve.py --no-browser

The page needs a real HTTP origin because it fetches `stream.json`; opening
index.html straight off the filesystem trips the browser's file:// rules. If you
would rather not run this, the page also accepts `stream.json` dropped onto it.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".json": "application/json",
        ".js": "text/javascript",
        ".svg": "image/svg+xml",
    }

    def end_headers(self):
        # The replay file is rebuilt often; never let the browser cache it.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "200" not in (args[1] if len(args) > 1 else ""):
            super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not (WEB / "index.html").exists():
        raise SystemExit(f"missing {WEB / 'index.html'}")
    if not (WEB / "stream.json").exists():
        print("! web/stream.json not found — run: python aml-live/build_stream.py\n")

    socketserver.TCPServer.allow_reuse_address = True
    handler = functools.partial(Handler, directory=str(WEB))
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(f"live view on {url}   (ctrl-c to stop)")
        if not args.no_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
