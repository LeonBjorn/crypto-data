"""The dashboard page: one HTML string, no build step, no dependencies.

Kept in its own module because it is markup rather than logic, and because a
four-hundred-line string sitting in the middle of a server makes the server hard
to read for no benefit. `server.py` imports PAGE and serves it.

WHAT THE LAYOUT IS FOR
----------------------
A dense terminal, meant to be left open on a screen and glanced at. Everything
fits without scrolling on a normal monitor, the type is small and monospaced so
columns line up, and nothing animates except the numbers changing. It is not a
presentation; it is an instrument panel.

THE ONE OPINION IN THE DESIGN
-----------------------------
Drawdown is shown next to return, at the same size, always. A paper account that
reads "+20.2%" is telling the truth and hiding the more useful half of it: this
one peaked at nearly twice its starting capital and gave most of it back, so the
return says the strategy made money and the drawdown says what holding it felt
like. A dashboard that shows only the first is the same failure this project has
been avoiding since the first backtest -- a number with nothing next to it to
keep it honest.

The refused count sits in the same row for the same reason. The wallet turned
down four signals for every one it took, and a page that only counts fills makes
a constrained account look like a quiet market.
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>paper account</title>
<style>
 :root{
   --bg:#0b0d11; --panel:#12151c; --line:#1e232d; --line2:#2a3140;
   --fg:#dfe4ee; --dim:#7d879c; --faint:#4c5568;
   --up:#26d07c; --down:#ff5f56; --warn:#e3b341; --accent:#4f9dff;
 }
 *{box-sizing:border-box;margin:0;padding:0}
 body{background:var(--bg);color:var(--fg);
   font:12px/1.45 ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
   -webkit-font-smoothing:antialiased;padding:14px}
 .wrap{max-width:1400px;margin:0 auto;display:flex;flex-direction:column;gap:10px}

 /* header */
 .top{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap}
 .title{font-size:13px;letter-spacing:.02em}
 .title b{font-weight:600}
 .title span{color:var(--dim)}
 .live{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:11px}
 .dot{width:7px;height:7px;border-radius:50%;background:var(--up);
   box-shadow:0 0 0 3px rgba(38,208,124,.15)}
 .dot.stale{background:var(--warn);box-shadow:0 0 0 3px rgba(227,179,65,.15)}
 .dot.dead{background:var(--down);box-shadow:0 0 0 3px rgba(255,95,86,.15)}

 /* strip of numbers */
 .strip{display:grid;grid-template-columns:repeat(7,1fr);gap:1px;
   background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden}
 .cell{background:var(--panel);padding:9px 12px}
 .k{color:var(--faint);font-size:9.5px;letter-spacing:.11em;text-transform:uppercase}
 .v{font-size:19px;margin-top:3px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
 .v.sm{font-size:15px}

 /* panels */
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:6px}
 .hd{display:flex;justify-content:space-between;align-items:center;
   padding:7px 12px;border-bottom:1px solid var(--line);color:var(--faint);
   font-size:9.5px;letter-spacing:.11em;text-transform:uppercase}
 .hd em{font-style:normal;color:var(--dim);letter-spacing:0;text-transform:none;font-size:11px}
 .body{padding:10px 12px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:10px}
 .cols3{display:grid;grid-template-columns:1.15fr 1fr .85fr;gap:10px}

 /* tables */
 table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
 th{text-align:right;color:var(--faint);font-weight:400;font-size:9.5px;
   letter-spacing:.09em;text-transform:uppercase;padding:0 8px 5px;
   border-bottom:1px solid var(--line)}
 th:first-child,td:first-child{text-align:left}
 td{text-align:right;padding:4px 8px;border-bottom:1px solid rgba(30,35,45,.5)}
 tr:last-child td{border-bottom:none}
 tbody tr:hover td{background:rgba(79,157,255,.05)}
 .sym{color:var(--fg)}
 .up{color:var(--up)} .down{color:var(--down)} .dim{color:var(--dim)}
 .warnc{color:var(--warn)}

 /* progress bar */
 .bar{display:inline-block;width:52px;height:6px;background:var(--line2);
   border-radius:3px;overflow:hidden;vertical-align:middle;margin-left:6px}
 .bar i{display:block;height:100%;background:var(--accent);opacity:.8}

 /* chart */
 svg{display:block;width:100%;height:150px}
 .feed{max-height:150px;overflow-y:auto;font-size:11px}
 .feed div{display:flex;gap:8px;padding:2px 0;color:var(--dim);
   border-bottom:1px solid rgba(30,35,45,.4)}
 .feed div:last-child{border-bottom:none}
 .feed time{color:var(--faint);flex:0 0 84px}
 .feed b{font-weight:400;color:var(--fg);flex:0 0 72px}
 .empty{color:var(--faint);padding:10px 0;text-align:center}
 .err{color:var(--down)}
 ::-webkit-scrollbar{width:8px;height:8px}
 ::-webkit-scrollbar-thumb{background:var(--line2);border-radius:4px}
 ::-webkit-scrollbar-track{background:transparent}
</style></head><body><div class="wrap">

<div class="top">
  <div class="title"><b>paper</b> <span id="meta">·</span></div>
  <div class="live"><span class="dot" id="dot"></span><span id="status">connecting…</span></div>
</div>

<div class="strip" id="strip"></div>

<div class="cols" style="grid-template-columns:1.6fr 1fr">
  <div class="panel">
    <div class="hd">realised equity <em id="curvenote"></em></div>
    <div class="body" style="padding:6px 8px 2px"><svg id="curve" viewBox="0 0 1000 150" preserveAspectRatio="none"></svg></div>
  </div>
  <div class="panel">
    <div class="hd">risk</div>
    <div class="body"><table id="risk"></table></div>
  </div>
</div>

<div class="panel">
  <div class="hd">open positions <em id="opennote"></em></div>
  <div class="body"><table id="open"></table></div>
</div>

<div class="cols3">
  <div class="panel">
    <div class="hd">by symbol <em>whole ledger</em></div>
    <div class="body"><table id="bysym"></table></div>
  </div>
  <div class="panel">
    <div class="hd">recent trades</div>
    <div class="body"><table id="trades"></table></div>
  </div>
  <div class="panel">
    <div class="hd">refused <em id="refnote"></em></div>
    <div class="body"><div class="feed" id="refused"></div></div>
  </div>
</div>

</div><script>
const N=(v,d=2)=>v==null?"–":Number(v).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
const P=v=>v==null?"–":(v>=0?"+":"")+Number(v).toFixed(2)+"%";
const C=v=>v==null?"":v>0?"up":v<0?"down":"dim";
const clock=ms=>new Date(ms).toISOString().slice(11,16);
const day=ms=>new Date(ms).toISOString().slice(5,10);

let snap=null, nextAt=null;

function since(ms){const s=(Date.now()-ms)/1000;
  if(s<90)return Math.max(0,Math.round(s))+"s ago";
  if(s<5400)return Math.round(s/60)+"m ago";
  if(s<172800)return Math.round(s/3600)+"h ago";
  return Math.round(s/86400)+"d ago";}

function countdown(){
  if(!nextAt)return"";
  const left=nextAt-Date.now();
  if(left<=0)return" · next candle due";
  const m=Math.floor(left/60000),s=Math.floor(left%60000/1000);
  return" · next in "+(m?m+"m":"")+(m<10?String(s).padStart(2,"0")+"s":"");
}

function tickClock(){
  if(!snap)return;
  const el=document.getElementById("status");
  const age=Date.now()-snap.cursor;
  const dot=document.getElementById("dot");
  dot.className="dot"+(age>6*3600e3?" dead":age>3*3600e3?" stale":"");
  el.textContent=snap.cursor_utc+" · "+since(snap.cursor)+countdown()
    +(age>3*3600e3?"  ⚠ collector not running?":"");
}

function chart(points,capital,peak){
  const svg=document.getElementById("curve");
  if(!points||points.length<2){svg.innerHTML='<text x="10" y="22" fill="#4c5568" font-size="11">no closed trades yet</text>';return;}
  const ys=points.map(p=>p.equity);
  const lo=Math.min(...ys,capital),hi=Math.max(...ys,capital),span=(hi-lo)||1;
  const X=i=>i/(points.length-1)*1000, Y=v=>142-((v-lo)/span)*132;
  const line=points.map((p,i)=>(i?"L":"M")+X(i).toFixed(1)+" "+Y(p.equity).toFixed(1)).join(" ");
  const area=line+` L1000 142 L0 142 Z`;
  const last=ys[ys.length-1], up=last>=capital;
  const col=up?"#26d07c":"#ff5f56";
  const py=Y(peak), by=Y(capital);
  svg.innerHTML=`
    <defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="${col}" stop-opacity=".22"/>
      <stop offset="100%" stop-color="${col}" stop-opacity="0"/></linearGradient></defs>
    <line x1="0" x2="1000" y1="${by}" y2="${by}" stroke="#2a3140" stroke-dasharray="3 4"/>
    <line x1="0" x2="1000" y1="${py}" y2="${py}" stroke="#e3b341" stroke-opacity=".35" stroke-dasharray="2 5"/>
    <path d="${area}" fill="url(#g)"/>
    <path d="${line}" fill="none" stroke="${col}" stroke-width="1.6"/>`;
  document.getElementById("curvenote").textContent=
    day(points[0].t)+" → "+day(points[points.length-1].t)+"  ·  "+points.length+" trades";
}

function render(s){
  snap=s; nextAt=s.next_candle;
  document.getElementById("meta").textContent=
    "· "+s.config.rule+" · "+s.config.hold+"h hold · "+s.config.exchange+" "+s.config.timeframe
    +" · "+s.config.symbols.length+" symbols"+(s.config.trail?" · trail "+(s.config.trail*100).toFixed(0)+"%":"");

  const st=s.stats, rk=s.risk;
  document.getElementById("strip").innerHTML=[
    ["equity",N(s.equity),C(s.return_pct),""],
    ["return",P(s.return_pct),C(s.return_pct),""],
    ["max drawdown",P(rk.max_drawdown_pct),"down",""],
    ["cash",N(s.cash),"",""],
    ["open",s.open_positions.length+" · "+N(s.open_value,0),"",""],
    ["closed",st.closed,"",""],
    ["refused",N(st.refused,0),"dim",""],
  ].map(([k,v,c])=>`<div class="cell"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join("");

  chart(s.equity_curve,s.starting_capital,rk.peak);

  document.getElementById("risk").innerHTML=`
    <tr><td class="dim">peak realised</td><td>${N(rk.peak)}</td></tr>
    <tr><td class="dim">max drawdown</td><td class="down">${P(rk.max_drawdown_pct)}</td></tr>
    <tr><td class="dim">from peak now</td><td class="${C(rk.current_drawdown_pct)}">${P(rk.current_drawdown_pct)}</td></tr>
    <tr><td class="dim">hit rate</td><td>${st.hit_rate==null?"–":st.hit_rate.toFixed(1)+"%"}</td></tr>
    <tr><td class="dim">mean / trade</td><td class="${C(st.mean_net_pct)}">${P(st.mean_net_pct)}</td></tr>
    <tr><td class="dim">best / worst</td><td><span class="up">${P(st.best_pct)}</span> <span class="dim">/</span> <span class="down">${P(st.worst_pct)}</span></td></tr>`;

  const op=s.open_positions;
  document.getElementById("opennote").textContent=op.length?op.length+" of "+s.config.symbols.length:"flat";
  document.getElementById("open").innerHTML=op.length
    ? `<tr><th>symbol</th><th>entered</th><th>entry</th><th>mark</th><th>unrealised</th><th>held</th><th>exits</th><th>progress</th></tr>`
      + op.map(p=>`<tr>
          <td class="sym">${p.symbol}</td>
          <td class="dim">${p.entry_utc.slice(5,16)}</td>
          <td>${N(p.entry_price,4)}</td>
          <td>${N(p.mark,4)}</td>
          <td class="${C(p.unrealised_pct)}">${P(p.unrealised_pct)}</td>
          <td class="dim">${p.bars_held}/${p.bars_total}h</td>
          <td class="dim">${p.exit_utc.slice(5,16)}</td>
          <td>${p.progress_pct.toFixed(0)}%<span class="bar"><i style="width:${p.progress_pct}%"></i></span></td>
        </tr>`).join("")
    : `<tr><td class="empty" colspan="8">flat — nothing open</td></tr>`;

  document.getElementById("bysym").innerHTML=
    `<tr><th>symbol</th><th>trades</th><th>hit</th><th>mean</th><th>pnl</th></tr>`
    + (s.by_symbol.length? s.by_symbol.map(b=>`<tr>
        <td class="sym">${b.symbol}</td><td class="dim">${b.trades}</td>
        <td class="dim">${b.hit_rate.toFixed(0)}%</td>
        <td class="${C(b.mean_pct)}">${P(b.mean_pct)}</td>
        <td class="${C(b.pnl)}">${N(b.pnl,0)}</td></tr>`).join("")
      : `<tr><td class="empty" colspan="5">nothing closed yet</td></tr>`);

  const t=s.recent_trades.slice(-14).reverse();
  document.getElementById("trades").innerHTML=
    `<tr><th>symbol</th><th>exit</th><th>why</th><th>held</th><th>net</th></tr>`
    + (t.length? t.map(r=>`<tr>
        <td class="sym">${r.symbol}</td>
        <td class="dim">${day(r.exit_time)}</td>
        <td class="dim">${r.exit_reason}</td>
        <td class="dim">${r.bars_held}h</td>
        <td class="${C(r.net_return)}">${P(r.net_return*100)}</td></tr>`).join("")
      : `<tr><td class="empty" colspan="5">nothing closed yet</td></tr>`);

  const rf=(s.refusals&&s.refusals.recent)||[];
  document.getElementById("refnote").textContent=N(s.refusals?s.refusals.total:0,0)+" total";
  document.getElementById("refused").innerHTML=rf.length
    ? rf.slice().reverse().map(r=>`<div><time>${r.at?day(r.at)+" "+clock(r.at):"–"}</time>
        <b>${r.symbol}</b><span>${r.reason}</span></div>`).join("")
    : `<div class="empty">nothing refused</div>`;

  tickClock();
}

async function poll(){
  try{
    const r=await fetch("/api/snapshot",{cache:"no-store"});
    if(!r.ok)throw new Error("snapshot "+r.status);
    render(await r.json());
  }catch(e){
    document.getElementById("dot").className="dot dead";
    document.getElementById("status").innerHTML='<span class="err">'+e.message+'</span> — run `paper`';
  }
}
poll(); setInterval(poll,15000); setInterval(tickClock,1000);
</script></body></html>"""
