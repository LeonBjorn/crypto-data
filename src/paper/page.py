"""The dashboard page: one HTML string, no build step, no dependencies.

Kept in its own module because it is markup rather than logic, and because a
several-hundred-line string sitting in the middle of a server makes the server
hard to read for no benefit. `server.py` imports PAGE and serves it.

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

WHAT IS INTERACTIVE AND WHAT DELIBERATELY IS NOT
------------------------------------------------
You can sort, filter, hover, choose a time range and pause the refresh. All of
that explores *what happened*, and the underlying numbers do not move while you
do it.

There is nothing here that changes the strategy -- no slider for the holding
period, no box for the volume multiple, no re-run button. That absence is
deliberate. This project's whole discipline is that a percentile you can re-roll
until you like it is worthless, and the research CLI is awkward to automate on
purpose. A dashboard that let you tune parameters and watch the equity curve
improve would be a machine for curve-fitting through a user interface, and it
would feel like insight the entire time.

THE CHART
---------
Two things it does that the obvious version does not.

The x-axis is *time*, not trade number. Spacing points evenly by index is easier
and quietly lies: a quiet month and a frantic one end up the same width, so the
shape of the curve stops meaning anything about when.

And it ends with a dashed segment to the account's current value including open
positions. The realised line necessarily stops at the last closed trade, which
is why a chart of it alone finishes below the equity printed beside it. Rather
than leave that gap to be noticed and puzzled over, it is drawn.
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
 .wrap{max-width:1500px;margin:0 auto;display:flex;flex-direction:column;gap:10px}

 .top{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap}
 .title{font-size:13px} .title b{font-weight:600} .title span{color:var(--dim)}
 .live{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:11px}
 .dot{width:7px;height:7px;border-radius:50%;background:var(--up);
   box-shadow:0 0 0 3px rgba(38,208,124,.15);flex:0 0 auto}
 .dot.stale{background:var(--warn);box-shadow:0 0 0 3px rgba(227,179,65,.15)}
 .dot.dead{background:var(--down);box-shadow:0 0 0 3px rgba(255,95,86,.15)}

 button{font:inherit;color:var(--dim);background:var(--panel);
   border:1px solid var(--line2);border-radius:4px;padding:2px 8px;cursor:pointer}
 button:hover{color:var(--fg);border-color:var(--faint)}
 button.on{color:var(--bg);background:var(--accent);border-color:var(--accent)}
 .btns{display:flex;gap:4px;flex-wrap:wrap}
 .tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:2px}
 .tab{background:none;border:none;border-bottom:2px solid transparent;border-radius:0;
   padding:6px 14px;color:var(--faint);font-size:11px;letter-spacing:.06em;text-transform:uppercase}
 .tab:hover{color:var(--fg);border-color:var(--line2)}
 .tab.on{color:var(--accent);border-color:var(--accent);background:none}
 .view{display:none} .view.on{display:flex;flex-direction:column;gap:10px}
 .big{font-size:26px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
 .split{display:flex;height:8px;border-radius:4px;overflow:hidden;background:var(--line2);margin-top:8px}
 .split i{display:block;height:100%}
 tfoot td{border-top:1px solid var(--line2);border-bottom:none;padding-top:6px;color:var(--fg)}

 .strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:1px;
   background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden}
 .cell{background:var(--panel);padding:9px 12px}
 .k{color:var(--faint);font-size:9.5px;letter-spacing:.11em;text-transform:uppercase}
 .v{font-size:19px;margin-top:3px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}

 .panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;min-width:0}
 .hd{display:flex;justify-content:space-between;align-items:center;gap:10px;
   padding:7px 12px;border-bottom:1px solid var(--line);color:var(--faint);
   font-size:9.5px;letter-spacing:.11em;text-transform:uppercase}
 .hd em{font-style:normal;color:var(--dim);letter-spacing:0;text-transform:none;font-size:11px}
 .body{padding:10px 12px}
 .cols{display:grid;grid-template-columns:1.6fr 1fr;gap:10px}
 .cols3{display:grid;grid-template-columns:1.1fr 1.35fr .85fr;gap:10px}
 @media(max-width:1100px){.cols,.cols3{grid-template-columns:1fr}}

 table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
 th{text-align:right;color:var(--faint);font-weight:400;font-size:9.5px;
   letter-spacing:.09em;text-transform:uppercase;padding:0 8px 5px;
   border-bottom:1px solid var(--line);white-space:nowrap}
 th.s{cursor:pointer;user-select:none} th.s:hover{color:var(--dim)}
 th.sorted{color:var(--accent)}
 th:first-child,td:first-child{text-align:left}
 td{text-align:right;padding:4px 8px;border-bottom:1px solid rgba(30,35,45,.5)}
 tr:last-child td{border-bottom:none}
 tbody tr:hover td{background:rgba(79,157,255,.05)}
 tr.click{cursor:pointer} tr.sel td{background:rgba(79,157,255,.12)}
 .up{color:var(--up)} .down{color:var(--down)} .dim{color:var(--dim)}

 .bar{display:inline-block;width:52px;height:6px;background:var(--line2);
   border-radius:3px;overflow:hidden;vertical-align:middle;margin-left:6px}
 .bar i{display:block;height:100%;background:var(--accent);opacity:.8}

 .chartwrap{position:relative}
 svg{display:block;width:100%;height:170px}
 #tip{position:absolute;pointer-events:none;background:#0b0d11ee;border:1px solid var(--line2);
   border-radius:4px;padding:5px 8px;font-size:11px;white-space:nowrap;display:none;z-index:5}
 .axis{display:flex;justify-content:space-between;color:var(--faint);font-size:10px;padding:2px 2px 0}

 .feed{max-height:190px;overflow-y:auto;font-size:11px}
 .feed div{display:flex;gap:8px;padding:2px 0;color:var(--dim);
   border-bottom:1px solid rgba(30,35,45,.4)}
 .feed div:last-child{border-bottom:none}
 .feed time{color:var(--faint);flex:0 0 78px}
 .feed b{font-weight:400;color:var(--fg);flex:0 0 68px}
 .scroll{max-height:250px;overflow-y:auto}
 .empty{color:var(--faint);padding:10px 0;text-align:center}
 .err{color:var(--down)}
 ::-webkit-scrollbar{width:8px;height:8px}
 ::-webkit-scrollbar-thumb{background:var(--line2);border-radius:4px}
 ::-webkit-scrollbar-track{background:transparent}
</style></head><body><div class="wrap">

<div class="top">
  <div class="title"><b>paper</b> <span id="meta">·</span></div>
  <div class="live">
    <span class="btns" id="ranges"></span>
    <button id="pause" title="pause auto-refresh">pause</button>
    <button id="now" title="refresh now">↻</button>
    <span class="dot" id="dot"></span><span id="status">connecting…</span>
  </div>
</div>

<div class="tabs" id="tabs"></div>

<div id="halted" style="display:none;background:#2a1416;border:1px solid #ff5f56;
  border-radius:6px;padding:10px 14px;color:#ff8b85;font-size:12px"></div>

<div class="strip" id="strip"></div>

<div class="view" id="v-overview">
<div class="cols">
  <div class="panel">
    <div class="hd">equity <em id="curvenote"></em></div>
    <div class="body" style="padding:6px 8px 4px">
      <div class="chartwrap"><svg id="curve" viewBox="0 0 1000 170" preserveAspectRatio="none"></svg><div id="tip"></div></div>
      <div class="axis"><span id="ax0"></span><span id="ax1"></span><span id="ax2"></span></div>
    </div>
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
</div>

<div class="view" id="v-holdings">
  <div class="cols" style="grid-template-columns:1fr 1fr 1fr">
    <div class="panel"><div class="hd">total equity</div><div class="body">
      <div class="big" id="h-equity"></div><div class="dim" id="h-equity-sub"></div>
      <div class="split"><i id="h-bar-inv" style="background:var(--accent)"></i><i id="h-bar-cash" style="background:var(--line2)"></i></div>
      <div class="dim" style="margin-top:4px;font-size:10.5px" id="h-split"></div>
    </div></div>
    <div class="panel"><div class="hd">profit and loss</div><div class="body">
      <div class="big" id="h-pnl"></div><div class="dim" id="h-pnl-sub"></div></div></div>
    <div class="panel"><div class="hd">cash available</div><div class="body">
      <div class="big" id="h-cash"></div><div class="dim" id="h-cash-sub"></div></div></div>
  </div>
  <div class="panel"><div class="hd">holdings <em id="h-note"></em></div>
    <div class="body"><table id="h-table"></table></div></div>
  <div class="panel"><div class="hd">realised by symbol <em>closed trades only</em></div>
    <div class="body"><table id="h-realised"></table></div></div>
</div>

<div class="view" id="v-trades">
  <div class="panel"><div class="hd">every closed trade <em id="tradenote"></em></div>
    <div class="body"><div class="scroll" style="max-height:70vh"><table id="trades"></table></div></div></div>
</div>

<div class="view" id="v-risk">
<div class="cols3">
  <div class="panel">
    <div class="hd">by symbol <em id="symnote">click to filter</em></div>
    <div class="body"><table id="bysym"></table></div>
  </div>
  <div class="panel">
    <div class="hd">refused <em id="refnote"></em></div>
    <div class="body"><div class="feed" id="refused"></div></div>
  </div>
</div>
</div>

<div style="color:#4c5568;font-size:10.5px;padding:2px 2px 10px" id="foot"></div>

</div><script>
const N=(v,d=2)=>v==null||isNaN(v)?"–":Number(v).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
const P=v=>v==null||isNaN(v)?"–":(v>=0?"+":"")+Number(v).toFixed(2)+"%";
// Money gets a currency mark and a sign. The quote is USDT rather than dollars;
// "$" is what everyone reads it as, and the footer says which it really is.
const M=(v,d=2)=>v==null||isNaN(v)?"–":"$"+Number(v).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
const MS=(v,d=2)=>v==null||isNaN(v)?"–":(v>=0?"+":"−")+"$"+Math.abs(Number(v)).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
const C=v=>v==null||isNaN(v)?"":v>0?"up":v<0?"down":"dim";
// A glyph as well as a colour: colour alone is lost to colourblind viewing and
// to a glance at a bright screen, and this is the number people glance at.
const ARROW=v=>v==null||isNaN(v)?"":v>0?"▲":v<0?"▼":"·";
const clock=ms=>new Date(ms).toISOString().slice(11,16);
const day=ms=>new Date(ms).toISOString().slice(5,10);
const date=ms=>new Date(ms).toISOString().slice(0,10);

const DAY=86400000, RANGES=[["1M",30],["3M",90],["1Y",365],["ALL",0]];
const VIEWS=[["overview","overview"],["holdings","holdings"],["trades","trades"],["risk","risk"]];
const store={
  get(){try{return JSON.parse(localStorage.getItem("paperui"))||{}}catch(e){return{}}},
  set(v){try{localStorage.setItem("paperui",JSON.stringify(v))}catch(e){}}
};
let ui=Object.assign({range:0,sort:"exit_time",dir:-1,symbol:null,paused:false,view:"overview"},store.get());
const save=()=>store.set(ui);

let snap=null, timer=null;

function since(ms){const s=(Date.now()-ms)/1000;
  if(s<90)return Math.max(0,Math.round(s))+"s ago";
  if(s<5400)return Math.round(s/60)+"m ago";
  if(s<172800)return Math.round(s/3600)+"h ago";
  return Math.round(s/86400)+"d ago";}

function countdown(){
  if(!snap||!snap.next_candle)return"";
  const left=snap.next_candle-Date.now();
  if(left<=0)return" · next candle due";
  const m=Math.floor(left/60000),s=Math.floor(left%60000/1000);
  return" · next in "+(m?m+"m":"")+(m<10?String(s).padStart(2,"0")+"s":"");
}

function tickClock(){
  if(!snap)return;
  const age=Date.now()-snap.cursor;
  document.getElementById("dot").className="dot"+(age>6*3600e3?" dead":age>3*3600e3?" stale":"");
  document.getElementById("status").textContent=
    snap.cursor_utc+" · "+since(snap.cursor)+countdown()+(age>3*3600e3?"  ⚠ collector not running?":"");
}

/* ---- chart: time on the x-axis, and the open-position stub drawn ---- */
let chartPts=[];
function chart(s){
  const svg=document.getElementById("curve"), cap=s.starting_capital;
  let pts=s.equity_curve.slice();
  if(ui.range) {const cut=Date.now()-ui.range*DAY; pts=pts.filter(p=>p.t>=cut);}
  if(pts.length<2){svg.innerHTML='<text x="10" y="24" fill="#4c5568" font-size="11">not enough closed trades in this range</text>';
    chartPts=[];document.getElementById("ax0").textContent="";document.getElementById("ax1").textContent="";
    document.getElementById("ax2").textContent="";return;}

  const now=s.equity_now||null;
  const t0=pts[0].t, t1=Math.max(pts[pts.length-1].t, now?now.t:0);
  let bench=(s.benchmark&&s.benchmark.curve)?s.benchmark.curve.filter(p=>p.t>=t0&&p.t<=t1):[];
  const vals=pts.map(p=>p.equity).concat([cap, now?now.equity:cap]).concat(bench.map(p=>p.equity));
  const lo=Math.min(...vals), hi=Math.max(...vals), span=(hi-lo)||1;
  // Time, not index. Even spacing by trade number makes a quiet month and a
  // frantic one the same width, and the shape stops meaning anything about when.
  const X=t=>(t1===t0?0:(t-t0)/(t1-t0))*1000, Y=v=>158-((v-lo)/span)*146;

  chartPts=pts.map(p=>({x:X(p.t),y:Y(p.equity),...p}));
  const line=chartPts.map((p,i)=>(i?"L":"M")+p.x.toFixed(1)+" "+p.y.toFixed(1)).join(" ");
  const col=pts[pts.length-1].equity>=cap?"#26d07c":"#ff5f56";
  const last=chartPts[chartPts.length-1];
  const stub=now&&now.t>pts[pts.length-1].t
    ? `<line x1="${last.x}" y1="${last.y}" x2="${X(now.t)}" y2="${Y(now.equity)}"
         stroke="${col}" stroke-width="1.4" stroke-dasharray="3 3" opacity=".85"/>
       <circle cx="${X(now.t)}" cy="${Y(now.equity)}" r="2.6" fill="${col}"/>` : "";

  // Drawn first and dimmer, so the strategy reads as the subject and the
  // benchmark as the thing it is being measured against.
  const bline=bench.length>1
    ? `<path d="${bench.map((p,i)=>(i?"L":"M")+X(p.t).toFixed(1)+" "+Y(p.equity).toFixed(1)).join(" ")}"
         fill="none" stroke="#7d879c" stroke-width="1.2" opacity=".55"/>` : "";

  svg.innerHTML=`
    <defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="${col}" stop-opacity=".20"/>
      <stop offset="100%" stop-color="${col}" stop-opacity="0"/></linearGradient></defs>
    <line x1="0" x2="1000" y1="${Y(cap)}" y2="${Y(cap)}" stroke="#2a3140" stroke-dasharray="3 4"/>
    <line x1="0" x2="1000" y1="${Y(s.risk.peak)}" y2="${Y(s.risk.peak)}" stroke="#e3b341" stroke-opacity=".3" stroke-dasharray="2 5"/>
    ${bline}
    <path d="${line}" fill="none" stroke="${col}" stroke-width="1.6"/>
    <path d="${line} L1000 158 L0 158 Z" fill="url(#g)" opacity=".7"/>
    ${stub}
    <line id="cross" x1="0" x2="0" y1="0" y2="170" stroke="#4f9dff" stroke-opacity=".5" style="display:none"/>`;

  document.getElementById("ax0").textContent=date(t0);
  document.getElementById("ax1").textContent=date((t0+t1)/2);
  document.getElementById("ax2").textContent=date(t1);
  const bm=s.benchmark;
  document.getElementById("curvenote").innerHTML=
    pts.length+" closed"+(now?"  ·  realised "+N(now.realised)+"  ·  open "+(now.unrealised>=0?"+":"")+N(now.unrealised):"")
    +(bm?`  ·  <span style="color:#7d879c">— hold ${bm.symbol} ${P(bm.return_pct)}</span>`:"");
}

function hover(ev){
  const svg=document.getElementById("curve"), tip=document.getElementById("tip");
  if(!chartPts.length){tip.style.display="none";return;}
  const r=svg.getBoundingClientRect();
  const x=(ev.clientX-r.left)/r.width*1000;
  let best=chartPts[0],bd=1e9;
  for(const p of chartPts){const d=Math.abs(p.x-x); if(d<bd){bd=d;best=p;}}
  const cross=document.getElementById("cross");
  if(cross){cross.setAttribute("x1",best.x);cross.setAttribute("x2",best.x);cross.style.display="";}
  const dd=snap.risk.peak?((best.equity/snap.risk.peak-1)*100):0;
  tip.innerHTML=`<b>${date(best.t)}</b> &nbsp; ${N(best.equity)} &nbsp; <span class="dim">${P(dd)} from peak</span>`;
  tip.style.display="block";
  const px=best.x/1000*r.width;
  tip.style.left=Math.min(Math.max(px-tip.offsetWidth/2,0),r.width-tip.offsetWidth)+"px";
  tip.style.top=(best.y/170*r.height-30)+"px";
}

/* ---- tables ---- */
function sortBy(key){ ui.dir = ui.sort===key ? -ui.dir : -1; ui.sort=key; save(); render(snap); }
const TH=(label,key)=>`<th class="s ${ui.sort===key?"sorted":""}" onclick="sortBy('${key}')">${label}${ui.sort===key?(ui.dir>0?" ▲":" ▼"):""}</th>`;

function pickSymbol(sym){ ui.symbol = ui.symbol===sym ? null : sym; save(); render(snap); }

function render(s){
  if(!s)return; snap=s;
  document.getElementById("meta").textContent=
    "· "+s.config.rule+" · "+s.config.hold+"h hold · "+s.config.exchange+" "+s.config.timeframe
    +" · "+s.config.symbols.length+" symbols"+(s.config.trail?" · trail "+(s.config.trail*100).toFixed(0)+"%":"")
    +(ui.symbol?"  ▸ filtered to "+ui.symbol:"");

  const st=s.stats, rk=s.risk;

  // Halted is not a statistic, it is a state change: no new positions are being
  // opened at all. Anything less than a banner would let it be scrolled past.
  const halt=document.getElementById("halted");
  if(rk.drawdown_tripped){
    halt.style.display="block";
    halt.innerHTML="<b>HALTED</b> &nbsp; the "+rk.drawdown_limit_pct.toFixed(0)
      +"% drawdown limit has tripped — no new positions are being opened. "
      +"Open positions still run to their exits. Resuming is a manual decision.";
  } else { halt.style.display="none"; }
  document.getElementById("strip").innerHTML=[
    ["equity",M(s.equity),C(s.return_pct)],
    ["return",ARROW(s.return_pct)+" "+P(s.return_pct),C(s.return_pct)],
    ["max drawdown",P(rk.max_drawdown_pct),"down"],
    ["from peak",P(rk.current_drawdown_pct),C(rk.current_drawdown_pct)],
    ["cash",M(s.cash),""],
    ["open",s.open_positions.length+" · "+M(s.open_value,0),""],
    ["p&l",MS(s.pnl_total,0),C(s.pnl_total)],
    ["closed",st.closed,""],
    ["refused",N(st.refused,0),"dim"],
    ...(rk.drawdown_limit_pct!=null?[["dd limit",
        rk.drawdown_tripped?"TRIPPED"
          :P(rk.guard_drawdown_pct)+" / −"+rk.drawdown_limit_pct.toFixed(0)+"%",
        rk.drawdown_tripped?"down":"dim"]]:[]),
    ...(s.benchmark?[["vs hold "+s.benchmark.symbol,
        ARROW(s.return_pct-s.benchmark.return_pct)+" "+P(s.return_pct-s.benchmark.return_pct),
        C(s.return_pct-s.benchmark.return_pct)]]:[]),
  ].map(([k,v,c])=>`<div class="cell"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join("");

  chart(s);

  document.getElementById("risk").innerHTML=`
    <tr><td class="dim">peak realised</td><td>${M(rk.peak)}</td></tr>
    <tr><td class="dim">max drawdown</td><td class="down">${P(rk.max_drawdown_pct)}</td></tr>
    <tr><td class="dim">from peak now</td><td class="${C(rk.current_drawdown_pct)}">${P(rk.current_drawdown_pct)}</td></tr>
    <tr><td class="dim">unrealised</td><td class="${C(s.equity_now?s.equity_now.unrealised:0)}">${s.equity_now?MS(s.equity_now.unrealised):"–"}</td></tr>
    ${s.benchmark?`<tr><td class="dim">hold ${s.benchmark.symbol}</td><td class="${C(s.benchmark.return_pct)}">${P(s.benchmark.return_pct)}</td></tr>
    <tr><td class="dim">its max drawdown</td><td class="down">${P(s.benchmark.max_drawdown_pct)}</td></tr>`:""}
    ${rk.drawdown_limit_pct!=null?`<tr><td class="dim">drawdown limit</td>
      <td class="${rk.drawdown_tripped?"down":"dim"}">${rk.drawdown_tripped?"TRIPPED":P(rk.guard_drawdown_pct)+" of −"+rk.drawdown_limit_pct.toFixed(0)+"%"}</td></tr>`:""}
    ${rk.guard_peak!=null?`<tr><td class="dim">limit measured from</td><td>${M(rk.guard_peak)}</td></tr>`:""}
    ${rk.expected_shortfall_95!=null?`<tr><td class="dim">ES (95%)</td>
      <td class="down">−${rk.expected_shortfall_95.toFixed(2)}%</td></tr>`:""}
    ${rk.sizing?`<tr><td class="dim">sizing</td><td class="dim">${rk.sizing}</td></tr>`:""}
    ${rk.forecast_vol_pct!=null?`<tr><td class="dim">forecast vol</td>
      <td>${rk.forecast_vol_pct.toFixed(0)}%${rk.target_vol_pct!=null?" → "+rk.target_vol_pct.toFixed(0)+"%":""}</td></tr>`:""}
    ${rk.scale!=null?`<tr><td class="dim">book scale</td><td>${rk.scale.toFixed(2)}×</td></tr>`:""}
    <tr><td class="dim">hit rate</td><td>${st.hit_rate==null?"–":st.hit_rate.toFixed(1)+"%"}</td></tr>
    <tr><td class="dim">mean / trade</td><td class="${C(st.mean_net_pct)}">${P(st.mean_net_pct)}</td></tr>
    <tr><td class="dim">best / worst</td><td><span class="up">${P(st.best_pct)}</span> <span class="dim">/</span> <span class="down">${P(st.worst_pct)}</span></td></tr>`;

  const op=s.open_positions.filter(p=>!ui.symbol||p.symbol===ui.symbol);
  document.getElementById("opennote").textContent=op.length?op.length+" open":"flat";
  document.getElementById("open").innerHTML=op.length
    ? `<tr><th>symbol</th><th>entered</th><th>entry</th><th>mark</th><th>unrealised</th><th>held</th><th>exits</th><th>progress</th></tr>`
      + op.map(p=>`<tr>
          <td>${p.symbol}</td><td class="dim">${p.entry_utc.slice(5,16)}</td>
          <td>${N(p.entry_price,4)}</td><td>${N(p.mark,4)}</td>
          <td class="${C(p.unrealised_pct)}">${ARROW(p.unrealised_pct)} ${P(p.unrealised_pct)}</td>
          <td class="dim">${p.bars_held}/${p.bars_total}h</td>
          <td class="dim">${p.exit_utc.slice(5,16)}</td>
          <td>${p.progress_pct.toFixed(0)}%<span class="bar"><i style="width:${p.progress_pct}%"></i></span></td>
        </tr>`).join("")
    : `<tr><td class="empty" colspan="8">${ui.symbol?"nothing open in "+ui.symbol:"flat — nothing open"}</td></tr>`;

  document.getElementById("bysym").innerHTML=
    `<tr><th>symbol</th><th>trades</th><th>hit</th><th>mean</th><th>pnl</th></tr>`
    + (s.by_symbol.length? s.by_symbol.map(b=>`<tr class="click ${ui.symbol===b.symbol?"sel":""}" onclick="pickSymbol('${b.symbol}')">
        <td>${b.symbol}</td><td class="dim">${b.trades}</td><td class="dim">${b.hit_rate.toFixed(0)}%</td>
        <td class="${C(b.mean_pct)}">${P(b.mean_pct)}</td>
        <td class="${C(b.pnl)}">${MS(b.pnl,0)}</td></tr>`).join("")
      : `<tr><td class="empty" colspan="5">nothing closed yet</td></tr>`);

  let rows=(s.trades||[]).filter(r=>!ui.symbol||r.symbol===ui.symbol);
  if(ui.range){const cut=Date.now()-ui.range*DAY; rows=rows.filter(r=>r.exit_time>=cut);}
  rows.sort((a,b)=>{const x=a[ui.sort],y=b[ui.sort];
    return (typeof x==="string"?x.localeCompare(y):x-y)*ui.dir;});
  document.getElementById("tradenote").textContent=rows.length+" shown";
  document.getElementById("trades").innerHTML=
    `<tr>${TH("symbol","symbol")}${TH("exit","exit_time")}${TH("why","exit_reason")}${TH("held","bars_held")}${TH("in","cash_in")}${TH("out","cash_out")}${TH("p&l","net_return")}${TH("net %","net_return")}</tr>`
    + (rows.length? rows.slice(0,400).map(r=>`<tr>
        <td>${r.symbol}</td><td class="dim">${day(r.exit_time)}</td>
        <td class="dim">${r.exit_reason}</td><td class="dim">${r.bars_held}h</td>
        <td class="dim">${M(r.cash_in,0)}</td><td class="dim">${M(r.cash_out,0)}</td>
        <td class="${C(r.cash_out-r.cash_in)}">${MS(r.cash_out-r.cash_in)}</td>
        <td class="${C(r.net_return)}">${P(r.net_return*100)}</td></tr>`).join("")
      : `<tr><td class="empty" colspan="8">no trades in this range</td></tr>`);

  const rf=((s.refusals&&s.refusals.recent)||[]).filter(r=>!ui.symbol||r.symbol===ui.symbol);
  document.getElementById("refnote").textContent=N(s.refusals?s.refusals.total:0,0)+" total";
  document.getElementById("refused").innerHTML=rf.length
    ? rf.slice().reverse().slice(0,120).map(r=>`<div><time>${r.at?day(r.at)+" "+clock(r.at):"–"}</time>
        <b>${r.symbol}</b><span>${r.reason}</span></div>`).join("")
    : `<div class="empty">nothing refused</div>`;

  renderHoldings(s);
  // Said once, plainly. Everything is priced in the quote currency of the pairs
  // being traded, which is USDT and not dollars -- they are close and they are
  // not the same thing, and a page that writes "$" without saying so is quietly
  // asserting a peg it does not check.
  document.getElementById("foot").textContent=
    "amounts are in "+(s.quote||"USDT")+", shown with $ for readability · paper account, no orders are placed, no credentials are held";
  tickClock();
}

function renderHoldings(s){
  const eq=s.equity, cap=s.starting_capital;
  document.getElementById("h-equity").textContent=M(eq);
  document.getElementById("h-equity-sub").innerHTML=
    `started with ${M(cap,0)} &nbsp;·&nbsp; <span class="${C(s.return_pct)}">${P(s.return_pct)}</span>`;
  const inv=s.invested_pct||0, csh=s.cash_pct||0;
  document.getElementById("h-bar-inv").style.width=inv+"%";
  document.getElementById("h-bar-cash").style.width=csh+"%";
  document.getElementById("h-split").textContent=
    `${inv.toFixed(0)}% invested in ${s.open_positions.length} position(s) · ${csh.toFixed(0)}% cash`;

  document.getElementById("h-pnl").innerHTML=
    `<span class="${C(s.pnl_total)}">${MS(s.pnl_total)}</span>`;
  document.getElementById("h-pnl-sub").innerHTML=
    `realised <span class="${C(s.pnl_realised)}">${MS(s.pnl_realised,0)}</span> &nbsp;·&nbsp; `
    +`open <span class="${C(s.pnl_unrealised)}">${MS(s.pnl_unrealised,0)}</span>`;

  document.getElementById("h-cash").textContent=M(s.cash);
  document.getElementById("h-cash-sub").textContent=
    `${csh.toFixed(0)}% of the account is not at risk`;

  const op=s.open_positions;
  document.getElementById("h-note").textContent=op.length?op.length+" open":"flat";
  const totCost=op.reduce((a,p)=>a+(p.cost||0),0), totVal=op.reduce((a,p)=>a+(p.value||0),0);
  document.getElementById("h-table").innerHTML = op.length
    ? `<tr><th>symbol</th><th>quantity</th><th>entry</th><th>price now</th><th>cost</th>
         <th>value now</th><th>profit</th><th>%</th><th>weight</th><th>time left</th></tr>`
      + op.map(p=>`<tr>
          <td>${p.symbol}</td>
          <td class="dim">${Number(p.qty).toLocaleString("en-US",{maximumFractionDigits:4})}</td>
          <td class="dim">${N(p.entry_price,4)}</td><td>${N(p.mark,4)}</td>
          <td class="dim">${M(p.cost)}</td><td>${M(p.value)}</td>
          <td class="${C(p.pnl)}">${MS(p.pnl)}</td>
          <td class="${C(p.unrealised_pct)}">${P(p.unrealised_pct)}</td>
          <td class="dim">${p.weight_pct==null?"–":p.weight_pct.toFixed(0)+"%"}</td>
          <td class="dim">${Math.max(0,p.bars_total-p.bars_held)}h</td></tr>`).join("")
      + `<tfoot><tr><td>total</td><td></td><td></td><td></td>
           <td>${M(totCost)}</td><td>${M(totVal)}</td>
           <td class="${C(totVal-totCost)}">${MS(totVal-totCost)}</td>
           <td colspan="3"></td></tr>
         <tr><td class="dim">cash</td><td colspan="4"></td>
           <td class="dim">${M(s.cash)}</td><td colspan="3"></td>
           <td class="dim">${(s.cash_pct||0).toFixed(0)}%</td></tr></tfoot>`
    : `<tr><td class="empty" colspan="10">flat — the whole account is in cash</td></tr>`;

  document.getElementById("h-realised").innerHTML=
    `<tr><th>symbol</th><th>trades</th><th>won</th><th>lost</th><th>hit</th><th>mean</th><th>realised p&l</th></tr>`
    + (s.by_symbol.length? s.by_symbol.map(b=>`<tr class="click ${ui.symbol===b.symbol?"sel":""}" onclick="pickSymbol('${b.symbol}')">
        <td>${b.symbol}</td><td class="dim">${b.trades}</td>
        <td class="dim">${b.wins==null?"–":b.wins}</td><td class="dim">${b.losses==null?"–":b.losses}</td>
        <td class="dim">${b.hit_rate.toFixed(0)}%</td>
        <td class="${C(b.mean_pct)}">${P(b.mean_pct)}</td>
        <td class="${C(b.pnl)}">${MS(b.pnl)}</td></tr>`).join("")
        + `<tfoot><tr><td>total</td><td class="dim">${s.by_symbol.reduce((a,b)=>a+b.trades,0)}</td>
             <td colspan="4"></td>
             <td class="${C(s.pnl_realised)}">${MS(s.pnl_realised)}</td></tr></tfoot>`
      : `<tr><td class="empty" colspan="7">nothing closed yet</td></tr>`);
}

function showView(name){
  ui.view=name; save();
  VIEWS.forEach(([id])=>{
    document.getElementById("v-"+id).classList.toggle("on", id===name);
  });
  [...document.getElementById("tabs").children].forEach(b=>
    b.classList.toggle("on", b.dataset.v===name));
}

/* ---- controls ---- */
document.getElementById("tabs").innerHTML=
  VIEWS.map(([id,label])=>`<button class="tab" data-v="${id}">${label}</button>`).join("");
document.getElementById("tabs").onclick=e=>{
  if(e.target.dataset.v) showView(e.target.dataset.v);
};
document.getElementById("ranges").innerHTML=
  RANGES.map(([l,d])=>`<button data-d="${d}" class="${ui.range===d?"on":""}">${l}</button>`).join("");
document.getElementById("ranges").onclick=e=>{
  if(e.target.tagName!=="BUTTON")return;
  ui.range=+e.target.dataset.d; save();
  [...e.currentTarget.children].forEach(b=>b.classList.toggle("on",+b.dataset.d===ui.range));
  render(snap);
};
const pauseBtn=document.getElementById("pause");
function setPaused(v){
  ui.paused=v; save();
  pauseBtn.textContent=v?"resume":"pause"; pauseBtn.classList.toggle("on",v);
  if(timer)clearInterval(timer);
  timer=v?null:setInterval(poll,15000);
}
pauseBtn.onclick=()=>setPaused(!ui.paused);
document.getElementById("now").onclick=poll;
const svgEl=document.getElementById("curve");
svgEl.addEventListener("mousemove",hover);
svgEl.addEventListener("mouseleave",()=>{document.getElementById("tip").style.display="none";
  const c=document.getElementById("cross"); if(c)c.style.display="none";});
document.addEventListener("keydown",e=>{
  if(e.key==="r")poll(); if(e.key===" "){e.preventDefault();setPaused(!ui.paused);}
  if(e.key==="Escape"&&ui.symbol){ui.symbol=null;save();render(snap);}
});

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
showView(ui.view||"overview");
poll(); setPaused(ui.paused); setInterval(tickClock,1000);
</script></body></html>"""
