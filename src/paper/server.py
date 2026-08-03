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

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "build_parser", "main", "serve"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_SNAPSHOT = "state/snapshot.json"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>paper account</title>
<style>
 :root{--bg:#0f1115;--card:#171a21;--line:#252a34;--dim:#8b94a7;--fg:#e6e9ef;
       --up:#3fb950;--down:#f85149;--accent:#58a6ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 .wrap{max-width:1100px;margin:0 auto;padding:24px}
 h1{font-size:16px;font-weight:600;margin:0 0 2px}
 .sub{color:var(--dim);font-size:12px;margin-bottom:20px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
 .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
 .v{font-size:20px;margin-top:6px;font-variant-numeric:tabular-nums}
 .up{color:var(--up)} .down{color:var(--down)} .muted{color:var(--dim)}
 table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
 th{text-align:right;color:var(--dim);font-weight:500;font-size:11px;
    text-transform:uppercase;letter-spacing:.06em;padding:6px 8px;border-bottom:1px solid var(--line)}
 th:first-child,td:first-child{text-align:left}
 td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
 tr:last-child td{border-bottom:none}
 h2{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;
    margin:24px 0 8px;font-weight:500}
 .banner{background:#1b2028;border:1px solid var(--line);border-left:3px solid var(--accent);
         border-radius:6px;padding:10px 14px;margin-bottom:20px;font-size:12px;color:var(--dim)}
 .stale{border-left-color:#d29922}
 svg{width:100%;height:180px;display:block}
 .err{color:var(--down)}
</style></head><body><div class="wrap">
<h1>paper account <span class="muted" id="rule"></span></h1>
<div class="sub" id="sub">loading…</div>
<div class="banner" id="banner"></div>
<div class="grid" id="cards"></div>
<h2>equity</h2><div class="card"><svg id="curve" viewBox="0 0 1000 180" preserveAspectRatio="none"></svg></div>
<h2>open positions <span class="muted" id="opencount"></span></h2>
<div class="card"><table id="open"></table></div>
<h2>recent trades</h2>
<div class="card"><table id="trades"></table></div>
</div><script>
const fmt=(n,d=2)=>n==null?"–":Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=n=>n==null?"–":(n>=0?"+":"")+Number(n).toFixed(2)+"%";
const cls=n=>n==null?"":(n>0?"up":n<0?"down":"");
const ago=ms=>{const s=(Date.now()-ms)/1000;if(s<3600)return Math.round(s/60)+"m";
  if(s<86400)return Math.round(s/3600)+"h";return Math.round(s/86400)+"d";};

function curve(points){
  const svg=document.getElementById("curve");
  if(!points||points.length<2){svg.innerHTML='<text x="12" y="24" fill="#8b94a7" font-size="12">no closed trades yet</text>';return;}
  const ys=points.map(p=>p.equity),lo=Math.min(...ys),hi=Math.max(...ys),span=(hi-lo)||1;
  const d=points.map((p,i)=>{const x=i/(points.length-1)*1000;
    const y=170-((p.equity-lo)/span)*160;return (i?"L":"M")+x.toFixed(1)+" "+y.toFixed(1);}).join(" ");
  const last=ys[ys.length-1],first=ys[0],up=last>=first;
  svg.innerHTML=`<path d="${d}" fill="none" stroke="${up?'#3fb950':'#f85149'}" stroke-width="2"/>`;
}

function render(s){
  document.getElementById("rule").textContent="· "+s.config.rule+" · "+s.config.hold+"-bar hold";
  document.getElementById("sub").textContent=
    s.config.symbols.join("  ")+"   —   "+s.config.exchange+" "+s.config.timeframe;

  const b=document.getElementById("banner");
  if(s.cursor){const stale=Date.now()-s.cursor>3*3600*1000;
    b.className="banner"+(stale?" stale":"");
    b.textContent=(stale?"⚠ ":"")+"last candle acted on: "+s.cursor_utc+"  ("+ago(s.cursor)+" ago)"
      +"  ·  paper only — no orders are placed, no credentials are held";
  } else {b.textContent="nothing processed yet — run `paper` after `collect`";}

  const st=s.stats;
  document.getElementById("cards").innerHTML=[
    ["equity",fmt(s.equity),cls(s.return_pct)],
    ["return",pct(s.return_pct),cls(s.return_pct)],
    ["cash",fmt(s.cash),""],
    ["open value",fmt(s.open_value),""],
    ["closed trades",st.closed,""],
    ["hit rate",st.hit_rate==null?"–":st.hit_rate.toFixed(1)+"%",""],
    ["mean / trade",st.mean_net_pct==null?"–":pct(st.mean_net_pct),cls(st.mean_net_pct)],
    ["refused",st.refused,"muted"],
  ].map(([k,v,c])=>`<div class="card"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join("");

  curve(s.equity_curve);

  document.getElementById("opencount").textContent=s.open_positions.length?"("+s.open_positions.length+")":"";
  document.getElementById("open").innerHTML=s.open_positions.length
    ? "<tr><th>symbol</th><th>entered</th><th>entry</th><th>mark</th><th>unrealised</th></tr>"+
      s.open_positions.map(p=>`<tr><td>${p.symbol}</td><td class="muted">${p.entry_utc}</td>
        <td>${fmt(p.entry_price,4)}</td><td>${fmt(p.mark,4)}</td>
        <td class="${cls(p.unrealised_pct)}">${pct(p.unrealised_pct)}</td></tr>`).join("")
    : '<tr><td class="muted">flat — nothing open</td></tr>';

  const t=s.recent_trades.slice().reverse();
  document.getElementById("trades").innerHTML=t.length
    ? "<tr><th>symbol</th><th>exit</th><th>bars</th><th>entry</th><th>exit px</th><th>net</th></tr>"+
      t.map(r=>`<tr><td>${r.symbol}</td><td class="muted">${r.exit_reason}</td><td>${r.bars_held}</td>
        <td>${fmt(r.entry_price,4)}</td><td>${fmt(r.exit_price,4)}</td>
        <td class="${cls(r.net_return)}">${pct(r.net_return*100)}</td></tr>`).join("")
    : '<tr><td class="muted">no closed trades yet</td></tr>';
}

async function tick(){
  try{const r=await fetch("/api/snapshot",{cache:"no-store"});
    if(!r.ok)throw new Error("snapshot "+r.status);
    render(await r.json());
  }catch(e){document.getElementById("banner").innerHTML=
    '<span class="err">cannot read snapshot: '+e.message+'</span> — run `paper` to write one';}
}
tick();setInterval(tick,15000);
</script></body></html>"""


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
