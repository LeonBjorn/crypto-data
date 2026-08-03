"""A small local dashboard for the paper account.

Milestone 3.3. It serves one page and one endpoint from the standard library,
reads the snapshot the `paper` command writes, and does nothing else. There is
no framework, no build step and no package to install, which is the point: a
dashboard that needs its own toolchain is a dashboard that stops working six
months later for reasons unrelated to trading.

READ-ONLY, AND LOCAL ONLY
-------------------------
The server binds to 127.0.0.1 by default and has no route that changes anything.
It cannot open a position, cannot close one, and cannot edit the config. That is
deliberate and it is the property to defend hardest when this is eventually
pointed at a live account: a page that can place an order is a page that can
place an order by accident, and it would be reachable by anything that can reach
the port.

Binding beyond localhost is possible with --host and prints a warning, because
there is a real use for it -- watching from a phone on the same network -- and
no honest way to pretend that is as safe as not doing it.

WHAT IT SHOWS AND WHAT IT REFUSES TO IMPLY
------------------------------------------
Equity is marked at the last close, which is a price nobody traded at and nobody
is promised, so unrealised numbers are labelled as such rather than folded into
one triumphant figure. The refused count sits next to the trade count, because
the gap between what the rule found and what the wallet could take is the whole
subject of paper trading. And the staleness of the data is shown at the top: a
dashboard whose numbers are eleven hours old while looking exactly like a live
one is worse than no dashboard.

CONNECTING IT TO LIVE DATA
--------------------------
Nothing here polls an exchange. The page re-reads the snapshot on a timer, and
the snapshot is rewritten whenever `paper` runs -- so the way to make this live
is to run `collect && paper` on a schedule, which needs no change to this file.
That is the seam: this server never learns where candles come from.
"""

import argparse
import json
import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from paper.page import PAGE

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "build_parser", "main", "serve"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_SNAPSHOT = "state/snapshot.json"


class Handler(BaseHTTPRequestHandler):
    """Two routes, both GET, neither of which changes anything."""

    snapshot_path = Path(DEFAULT_SNAPSHOT)

    def _send(self, code, body, content_type):
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        # The snapshot changes under the page's feet by design; a cached one
        # would show yesterday's account and look exactly like today's.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        route = self.path.split("?")[0].rstrip("/") or "/"

        if route == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")

        if route == "/api/snapshot":
            try:
                return self._send(
                    200,
                    self.snapshot_path.read_text(encoding="utf-8"),
                    "application/json; charset=utf-8",
                )
            except FileNotFoundError:
                return self._send(
                    404,
                    json.dumps({"error": f"no snapshot at {self.snapshot_path}"}),
                    "application/json; charset=utf-8",
                )

        return self._send(404, json.dumps({"error": "not found"}), "application/json")

    def log_message(self, fmt, *args):
        """Quiet by default. A request log per fifteen-second poll is noise that
        buries anything worth reading."""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="paper-serve",
        allow_abbrev=False,
        description=(
            "Serve a read-only dashboard for the paper account. Reads the "
            "snapshot that `paper` writes; it never fetches, never trades and "
            "has no route that changes anything."
        ),
    )
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, metavar="PATH",
                        help="snapshot to serve (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, metavar="N",
                        help="port to listen on (default: %(default)s)")
    parser.add_argument("--host", default=DEFAULT_HOST, metavar="ADDR",
                        help="address to bind (default: %(default)s, localhost only)")
    return parser


def serve(host, port, snapshot, *, forever=True):
    """Start the server. Returns it, so a test can drive one request and stop."""
    handler = partial(Handler)
    handler.snapshot_path = Path(snapshot)
    # Bound as a class attribute because BaseHTTPRequestHandler is instantiated
    # per request, so there is nowhere else to put per-server configuration.
    Handler.snapshot_path = Path(snapshot)

    httpd = ThreadingHTTPServer((host, port), Handler)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding {host} makes this page reachable from other "
            f"machines. It is read-only and places no orders, but it does show "
            f"your positions.",
            file=sys.stderr,
        )
    print(f"paper dashboard on http://{host}:{port}  (reading {snapshot})")
    print("Ctrl-C to stop.")
    if forever:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
        finally:
            httpd.server_close()
    return httpd


def main():
    args = build_parser().parse_args()
    serve(args.host, args.port, args.snapshot)


if __name__ == "__main__":
    main()
