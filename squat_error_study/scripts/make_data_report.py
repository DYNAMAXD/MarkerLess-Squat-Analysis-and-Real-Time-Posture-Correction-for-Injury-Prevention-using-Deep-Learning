"""
make_data_report.py   (SCRIPT)
==============================
Per-person, per-recording, per-rep data-quality report.

Writes
  docs/data_quality_report.html     the visual report (open in a browser)
  data/recording_quality.csv        one row per recording (same information as a table)

Run it yourself (from the squat_error_study folder):
    C:\\squat_venv\\Scripts\\python.exe scripts\\make_data_report.py

A recording is
  CLEAN         included, Neck present, 9-12 reps, fewer than 3 odd-duration reps, no missing upper-body joints
                (one odd-duration rep is listed under "problem reps" but does not make a recording unclean)
  USABLE+NOTES  usable for training but has a note (13+ reps, 3+ odd-duration reps, Head/shoulder joints missing)
  NECK MISSING  trunk angles unreliable -> its reps are dropped from the main result
  EXCLUDED      left out on purpose (E_PANTS_MALE, KEY_INFO rep errors, no reps.csv)
  NOT PROCESSED listed in KEY_INFO but no processed folder exists in Temp_Dataset
"""

import csv
import glob
import html
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(r"C:\Current_Mini_Project_Working\Temp_Dataset")
CLASSES = ["correct", "insufficient_depth", "forward_lean", "knee_valgus", "knee_varus", "heel_lift", "abnormal_tempo"]
ODD_LOW, ODD_HIGH = 0.6, 1.6          # rep duration outside 0.6x-1.6x of the recording median = odd
UPPER = ["Head", "Nose", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist", "LWrist"]


# ----------------------------------------------------------------------------
# 1. Gather everything
# ----------------------------------------------------------------------------
def clean_label(raw):
    return raw.strip().strip('"').strip().lower().replace(" ", "_")


def main():
    ws = openpyxl.load_workbook(SRC / "Additional_Info/KEY_INFO.xlsx", data_only=True).active
    key = {str(r[0]).strip(): (str(r[1]).strip(), str(r[2] or "-").strip())
           for r in list(ws.iter_rows(values_only=True))[1:] if r[0]}
    labels = {}
    for f in glob.glob(str(SRC / "Additional_Info/MAPPINGS_INFO/*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            labels[r["person_id"].strip()] = clean_label(r["error_performed"])
    man = pd.read_csv(ROOT / "data/manifest.csv").set_index("recording_id")
    folders = set(os.listdir(SRC / "preprocessed_data"))
    d = np.load(ROOT / "data/processed/reps_T64.npz")
    meta = json.load(open(ROOT / "data/processed/reps_T64.json"))
    rep_df = pd.DataFrame({"rec": d["recording"], "rep_no": d["rep_no"], "dur": d["tempo"][:, 0],
                           "trunk": d["trunk_missing"], "miss": d["missing"]})

    rows = []
    for rid in sorted(key, key=lambda s: int(s[1:])):
        person, key_err = key[rid]
        label = labels.get(rid, "?")
        row = dict(recording=rid, person=person, label=label, reps=0, status="", problem_reps="", notes="",
                   usable_reps=0, clean_reps=0)
        notes = []
        if key_err not in ("-", ""):
            notes.append("KEY_INFO: " + key_err.lower())
        if rid not in folders:
            row["status"] = "not_processed"
            notes.append("no processed folder in Temp_Dataset (the batch log says it was processed: look on the F: drive)")
        elif rid in man.index and not man.loc[rid, "include"]:
            row["status"] = "excluded"
            row["reps"] = int(man.loc[rid, "n_reps"])
            notes.append(str(man.loc[rid, "exclude_reason"]))
        else:
            g = rep_df[rep_df.rec == rid].sort_values("rep_no")
            row["reps"] = len(g)
            trunk_bad = g[g.trunk > 0.5].rep_no.tolist()
            med = g.dur.median()
            odd = g[(g.dur < ODD_LOW * med) | (g.dur > ODD_HIGH * med)].rep_no.tolist() if label != "abnormal_tempo" else []
            miss_j = {k: v for k, v in meta["recordings"][rid]["missing_per_joint"].items()}
            upper = {k: v for k, v in miss_j.items() if k in UPPER and k != "Neck"}
            flagged_notes = []
            if len(g) > 12:
                flagged_notes.append(f"{len(g)} reps (more than 12: the last ones may be false splits)")
            if odd:
                flagged_notes.append("odd-duration reps: " + ", ".join(map(str, odd)))
            if upper:
                flagged_notes.append("upper-body joints missing: " + ", ".join(upper))
            if label == "abnormal_tempo":
                flagged_notes.append("tempo class (not used in the 6-class model)")
            if trunk_bad:
                row["status"] = "neck_missing"
                row["problem_reps"] = "all reps (" + ", ".join(map(str, trunk_bad)) + ")" if len(trunk_bad) == len(g) \
                    else ", ".join(map(str, trunk_bad))
                notes.append("Neck missing: trunk angles unreliable")
            elif len(g) > 12 or upper or len(odd) >= 3:
                # a single odd-duration rep (often the first or last) is only a rep-level note, not a recording problem
                row["status"] = "usable_notes"
            else:
                row["status"] = "clean"
            notes += flagged_notes
            if odd:
                row["problem_reps"] = ", ".join(map(str, odd)) if not row["problem_reps"] else row["problem_reps"]
            ok = ~g.rep_no.isin(trunk_bad) & ~g.rep_no.isin(odd)
            row["usable_reps"] = int((~g.rep_no.isin(trunk_bad)).sum())
            row["clean_reps"] = int(ok.sum())
        row["notes"] = "; ".join(notes)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/recording_quality.csv", index=False)
    write_html(df)
    print(df.groupby(["label", "status"]).size().unstack(fill_value=0).reindex(CLASSES).to_string())
    print("\nwritten:", ROOT / "docs/data_quality_report.html", "and", ROOT / "data/recording_quality.csv")


# ----------------------------------------------------------------------------
# 2. HTML
# ----------------------------------------------------------------------------
STATUS = {"clean": ("clean", "t-ok"), "usable_notes": ("usable + notes", "t-info"),
          "neck_missing": ("neck missing", "t-warn"), "excluded": ("excluded", "t-bad"),
          "not_processed": ("not processed", "t-bad")}

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d232b;--muted:#5d6875;--line:#dfe3e8;--blue:#2f6db5;--blue-soft:#dce9f7;
--orange:#d9822b;--amber-soft:#fbf0cf;--green:#2e8b57;--green-soft:#daf0e3;--red:#c0453b;--red-soft:#f8dcd9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14181d;--card:#1c2229;--ink:#e8ecf0;--muted:#9aa6b2;
--line:#2d3640;--blue:#6aa5ea;--blue-soft:#1f3550;--orange:#eba155;--amber-soft:#43391a;--green:#5cc58c;--green-soft:#1b3a2a;
--red:#e8776d;--red-soft:#4a2420}}
:root[data-theme="dark"]{--bg:#14181d;--card:#1c2229;--ink:#e8ecf0;--muted:#9aa6b2;--line:#2d3640;--blue:#6aa5ea;--blue-soft:#1f3550;
--orange:#eba155;--amber-soft:#43391a;--green:#5cc58c;--green-soft:#1b3a2a;--red:#e8776d;--red-soft:#4a2420}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.8rem;margin:0 0 4px}
h2{font-size:1.3rem;margin:36px 0 10px;padding-top:8px;border-top:2px solid var(--line)}.sub{color:var(--muted);margin:0 0 18px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.stat b{display:block;font-size:1.6rem;line-height:1.1}.stat span{color:var(--muted);font-size:.86rem}
.tw{overflow-x:auto}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);font-size:.9rem}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--bg);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:1px 9px;border-radius:999px;font-size:.76rem;font-weight:600;white-space:nowrap}
.t-ok{background:var(--green-soft);color:var(--green)}.t-info{background:var(--blue-soft);color:var(--blue)}
.t-warn{background:var(--amber-soft);color:var(--orange)}.t-bad{background:var(--red-soft);color:var(--red)}
.stack{display:flex;height:14px;border-radius:5px;overflow:hidden;border:1px solid var(--line);min-width:160px}
.stack i{display:block;height:100%}.s-ok{background:var(--green)}.s-info{background:var(--blue)}.s-warn{background:var(--orange)}.s-bad{background:var(--red)}
details{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:8px 0;padding:0 14px}
summary{cursor:pointer;padding:11px 0;font-weight:600;list-style:none}summary::-webkit-details-marker{display:none}
summary::before{content:"\\25B8  ";color:var(--muted)}details[open] summary::before{content:"\\25BE  "}
summary .sm{font-weight:400;color:var(--muted);font-size:.86rem;margin-left:8px}
details table{margin-bottom:12px}
.callout{border-left:4px solid var(--orange);background:var(--amber-soft);padding:10px 14px;border-radius:0 10px 10px 0;margin:14px 0}
.callout.bad{border-color:var(--red);background:var(--red-soft)}code{background:var(--bg);padding:1px 5px;border-radius:4px;border:1px solid var(--line);font-size:.85rem}
.leg{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0;font-size:.85rem;color:var(--muted)}
"""


def tag(status):
    name, cls = STATUS[status]
    return f'<span class="tag {cls}">{name}</span>'


def stack(c):
    tot = max(sum(c.values()), 1)
    parts = [("s-ok", c["clean"]), ("s-info", c["usable_notes"]), ("s-warn", c["neck_missing"]),
             ("s-bad", c["excluded"] + c["not_processed"])]
    return '<div class="stack">' + "".join(f'<i class="{k}" style="width:{100 * v / tot:.1f}%"></i>' for k, v in parts if v) + "</div>"


def write_html(df):
    esc = html.escape
    out = []
    n_rec, n_clean = len(df), int((df.status == "clean").sum())
    main6 = df[df.label != "abnormal_tempo"]
    out.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Data Quality Report</title><style>{CSS}</style></head>
<body><div class="wrap"><h1>Data Quality Report</h1>
<p class="sub">Every person, every recording, every problem. Generated by <code>scripts/make_data_report.py</code> from the manifest,
KEY_INFO and the compact dataset.</p>
<div class="grid">
<div class="card stat"><b>{n_rec}</b><span>recordings in KEY_INFO</span></div>
<div class="card stat"><b>{n_clean}</b><span>completely clean</span></div>
<div class="card stat"><b>{int((df.status == 'usable_notes').sum())}</b><span>usable, with a note</span></div>
<div class="card stat"><b>{int((df.status == 'neck_missing').sum())}</b><span>neck missing</span></div>
<div class="card stat"><b>{int(df.status.isin(['excluded', 'not_processed']).sum())}</b><span>excluded / not processed</span></div>
</div>
<div class="leg"><span><i class="tag t-ok">clean</i> no problem found</span><span><i class="tag t-info">usable + notes</i> used, but check the note</span>
<span><i class="tag t-warn">neck missing</i> dropped from the main result</span><span><i class="tag t-bad">excluded / not processed</i> not used</span></div>""")

    missing = df[df.status == "not_processed"]
    if len(missing):
        ids = ", ".join(missing.recording)
        out.append(f'<div class="callout bad"><b>{len(missing)} recordings are in KEY_INFO but have no processed folder in Temp_Dataset:</b> {ids} '
                   f'(persons {", ".join(sorted(set(missing.person)))}). The batch log says they were processed '
                   f'(p95 shows an error), so they may exist on the F: drive output folder. Copying them in would add data.</div>')

    # ---- per class
    out.append('<h2>1. Per class: how much is properly there</h2><div class="tw"><table><tr><th>Class</th><th class="n">In KEY_INFO</th>'
               '<th class="n">Not processed</th><th class="n">Excluded</th><th class="n">Neck missing</th>'
               '<th class="n">Usable + notes</th><th class="n">Clean</th><th>Split</th><th class="n">Usable reps</th><th class="n">Clean reps</th></tr>')
    totals = defaultdict(int)
    for c in CLASSES:
        g = df[df.label == c]
        cnt = g.status.value_counts().to_dict()
        cc = {k: cnt.get(k, 0) for k in STATUS}
        ur, cr = int(g.usable_reps.sum()), int(g.clean_reps.sum())
        note = " <small>(not in the 6-class model)</small>" if c == "abnormal_tempo" else ""
        out.append(f"<tr><td><b>{c}</b>{note}</td><td class='n'>{len(g)}</td><td class='n'>{cc['not_processed']}</td>"
                   f"<td class='n'>{cc['excluded']}</td><td class='n'>{cc['neck_missing']}</td><td class='n'>{cc['usable_notes']}</td>"
                   f"<td class='n'><b>{cc['clean']}</b></td><td>{stack(cc)}</td><td class='n'>{ur}</td><td class='n'><b>{cr}</b></td></tr>")
        if c != "abnormal_tempo":
            totals["rec"] += len(g); totals["clean"] += cc["clean"]; totals["ur"] += ur; totals["cr"] += cr
            totals["usable_rec"] += cc["clean"] + cc["usable_notes"]
    out.append(f"<tr><td><b>6-class total</b></td><td class='n'>{totals['rec']}</td><td></td><td></td><td></td><td class='n'>"
               f"{totals['usable_rec'] - totals['clean']}</td><td class='n'><b>{totals['clean']}</b></td><td></td>"
               f"<td class='n'>{totals['ur']}</td><td class='n'><b>{totals['cr']}</b></td></tr></table></div>")
    out.append("<p class='sub'><b>Usable reps</b> = reps with the Neck present (what the main 6-class result trains on). "
               "<b>Clean reps</b> = usable reps that also have a normal duration (0.6&times;&ndash;1.6&times; the recording median).</p>")

    # ---- per person
    out.append("<h2>2. Per person, per recording</h2>")
    for person in sorted(df.person.unique(), key=lambda p: (p == "E_PANTS_MALE", p)):
        g = df[df.person == person]
        cnt = g.status.value_counts().to_dict()
        bits = " &middot; ".join(f"{cnt[k]} {STATUS[k][0]}" for k in STATUS if cnt.get(k))
        openattr = " open" if (g.status != "clean").any() and person != "E_PANTS_MALE" else ""
        out.append(f'<details{openattr}><summary>Person {esc(person)}<span class="sm">{len(g)} recordings: {bits}</span></summary>'
                   '<div class="tw"><table><tr><th>Recording</th><th>Class</th><th class="n">Reps</th><th>Status</th><th>Problem reps</th><th>Notes</th></tr>')
        for _, r in g.iterrows():
            out.append(f"<tr><td><b>{r.recording}</b></td><td>{r.label}</td><td class='n'>{r.reps if r.reps else '-'}</td>"
                       f"<td>{tag(r.status)}</td><td>{esc(str(r.problem_reps or '-'))}</td><td>{esc(r.notes or '-')}</td></tr>")
        out.append("</table></div></details>")

    out.append('<h2>3. What this means</h2><ul><li>The six <code>correct &hellip; heel_lift</code> classes each keep a different number of '
               'recordings. The smallest classes limit how reliable their per-class scores are, so quote per-class F1 with care.</li>'
               '<li>Neck-missing recordings are dropped from the main result and kept for a robustness run '
               '(<code>data.max_trunk_missing: null</code>).</li>'
               '<li>&ldquo;Odd-duration&rdquo; reps are only flagged, not removed: a slow or fast rep can be a real part of the movement.</li>'
               '<li>If the six not-processed recordings can be recovered from the F: drive, run <code>build_manifest.py</code>, '
               '<code>build_dataset.py</code> and this report again.</li></ul></div></body></html>')
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/data_quality_report.html").write_text("".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
