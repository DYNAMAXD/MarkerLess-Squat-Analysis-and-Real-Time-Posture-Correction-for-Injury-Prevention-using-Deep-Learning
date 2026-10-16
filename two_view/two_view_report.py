"""
two_view_report.py -- one self-contained HTML page showing both views on a shared time axis.

    Panel 1: knee flexion (squat depth) from FRONT and SIDE -- these should overlap if sync is right
    Panel 2: front-view metrics (knee valgus L/R)
    Panel 3: side-view metrics (trunk lean, hip flexion, ankle proxy, head angle)
    Table  : per-rep fused features

Side-view time is shifted by the clap offset so both views share the front video's clock.
Plotly.js is loaded from a CDN, so open the file with internet access.
"""
import json
from pathlib import Path

import numpy as np

PANEL2 = ["knee_valgus_L", "knee_valgus_R"]
PANEL3 = ["sagittal_flexion_trunk", "hip_flexion_L", "knee_flexion_L", "ankle_dorsiflexion_proxy_L", "head_forward_angle"]
TABLE_COLS = [
    "rep_id", "front_bottom_s", "side_bottom_s",
    "front_knee_valgus_L_max", "front_knee_valgus_R_max", "front_squat_depth_knee_deg_bottom",
    "side_squat_depth_knee_deg_bottom", "side_sagittal_flexion_trunk_max",
    "side_hip_flexion_L_bottom", "side_ankle_dorsiflexion_proxy_L_max", "side_rep_dur_s",
]

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.0/plotly.min.js"></script>
<style>
 body{font-family:system-ui,sans-serif;margin:16px;color:#1a1a1a;background:#fff}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#555;font-size:13px;margin-bottom:12px}
 table{border-collapse:collapse;font-size:12px;margin-top:12px} th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}
 th{background:#f0f0f0} .wrap{overflow-x:auto}
</style></head><body>
<h1>__TITLE__</h1>
<div class="sub">__SUB__</div>
<div id="plot" style="height:820px"></div>
<h3>Per-rep fused features</h3>
<div class="wrap" id="tbl"></div>
<script>
const D = __DATA__;
const traces = [];
const add = (name, x, y, ax, color, dash) => traces.push({name, x, y, xaxis:'x'+ax, yaxis:'y'+ax, mode:'lines',
  line:{color, width:1.6, dash: dash||'solid'}, connectgaps:false});
add('Front: knee flexion', D.tf, D.front.squat_depth_knee_deg, '', '#1f77b4');
add('Side: knee flexion',  D.ts, D.side.squat_depth_knee_deg,  '', '#d62728', 'dash');
const pal = ['#2ca02c','#9467bd','#ff7f0e','#8c564b','#17becf'];
__P2__.forEach((m,i)=>add('Front: '+m, D.tf, D.front[m], 2, pal[i%5]));
__P3__.forEach((m,i)=>add('Side: '+m,  D.ts, D.side[m],  3, pal[i%5]));
const shapes = D.reps.map(r=>({type:'rect', xref:'x', yref:'paper', x0:r[0], x1:r[1], y0:0, y1:1,
  fillcolor:'rgba(100,100,100,0.08)', line:{width:0}}));
const ann = D.reps.map((r,i)=>({x:(r[0]+r[1])/2, y:1.0, xref:'x', yref:'paper', text:'R'+(i+1), showarrow:false, yanchor:'bottom'}));
Plotly.newPlot('plot', traces, {
  shapes, annotations: ann, margin:{t:30,l:55,r:10,b:40}, legend:{orientation:'h', y:-0.08},
  xaxis:{domain:[0,1], anchor:'y',  matches:'x3', showticklabels:false},
  xaxis2:{domain:[0,1], anchor:'y2', matches:'x3', showticklabels:false},
  xaxis3:{domain:[0,1], anchor:'y3', title:'time (s) on the front-video clock'},
  yaxis:{domain:[0.70,1.0], title:'deg'}, yaxis2:{domain:[0.37,0.64], title:'deg'}, yaxis3:{domain:[0,0.31], title:'deg'},
}, {responsive:true});
// table
const cols = D.table_cols; let h = '<table><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr>';
D.table.forEach(r=>{h+='<tr>'+r.map(v=>'<td>'+(v===null?'':(typeof v==='number'?(Number.isInteger(v)?v:v.toFixed(2)):v))+'</td>').join('')+'</tr>'});
document.getElementById('tbl').innerHTML = h+'</table>';
</script></body></html>"""


def _series(df, cols):
    return {c: [None if not np.isfinite(v) else round(float(v), 2) for v in df[c]] for c in cols}


def build_report(fused, fps_front, fps_side, offset_s, out_html, title="Two-view squat analysis"):
    front, side = fused["front"], fused["side"]
    off = offset_s or 0.0
    cols = sorted(set(PANEL2 + PANEL3 + ["squat_depth_knee_deg"]))
    tf = [round(f / fps_front, 3) for f in front["frame"]]
    ts = [round(f / fps_side - off, 3) for f in side["frame"]]  # shift side onto the front clock
    reps = [[fused["front_reps"][i]["start"] / fps_front, fused["front_reps"][i]["end"] / fps_front]
            for i, _ in fused["pairs"]]
    table = fused["table"]
    tcols = [c for c in TABLE_COLS if c in table.columns]
    rows = [[None if (isinstance(v, float) and not np.isfinite(v)) else (round(v, 3) if isinstance(v, float) else v)
             for v in r] for r in table[tcols].itertuples(index=False, name=None)]
    data = {"tf": tf, "ts": ts, "front": _series(front, cols), "side": _series(side, cols),
            "reps": reps, "table_cols": tcols, "table": rows}
    sub = (f"{len(fused['front_reps'])} reps found in front view, {len(fused['side_reps'])} in side view, "
           f"{len(fused['pairs'])} matched. Clap offset: "
           + ("none (matched by order)" if offset_s is None else f"{offset_s:+.3f} s (side minus front)"))
    html = (TEMPLATE.replace("__TITLE__", title).replace("__SUB__", sub)
            .replace("__P2__", json.dumps(PANEL2)).replace("__P3__", json.dumps(PANEL3))
            .replace("__DATA__", json.dumps(data)))
    Path(out_html).write_text(html, encoding="utf-8")
    return out_html
