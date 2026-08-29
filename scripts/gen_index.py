#!/usr/bin/env python3
"""Regenerate index.html and EXECUTION/LEDGER.md from EXECUTION/{topics,status}.json.
Single source of the course hub's design — daily runs call this so the page never drifts.
Run from repo root:  python3 scripts/gen_index.py
"""
import json, pathlib, html, datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "EXECUTION/topics.json").read_text())
STATUS = json.loads((ROOT / "EXECUTION/status.json").read_text())
GH = f"https://github.com/{DATA['repo']}/tree/{DATA['branch']}"

# ---- status vocabulary ----
STATE = {  # key -> (short label, css class)
    "done":            ("done",     "s-done"),
    "pass":            ("pass",     "s-done"),
    "in_progress":     ("wip",      "s-wip"),
    "pending":         ("pending",  "s-wip"),
    "awaiting-4070":   ("4070",     "s-gpu"),
    "deferred":        ("deferred", "s-defer"),
    "scheduled":       ("todo",     "s-todo"),
    "n/a":             ("—",        "s-na"),
}
def st(v):        return STATE.get(v, STATE["scheduled"])
def cell(v):      lab, cls = st(v); return f'<span class="pill {cls}">{lab}</span>'

def is_code_done(tid):
    return STATUS.get(tid, {}).get("code") in ("done", "pass")

# ---- progress ----
build_topics = [t for t in DATA["topics"] if not t["id"].startswith("P0")]
done_n = sum(1 for t in DATA["topics"] if is_code_done(t["id"]))
total_n = len(DATA["topics"])
pct = round(100 * done_n / total_n) if total_n else 0

phase_by_id = {p["id"]: p for p in DATA["phases"]}
def topics_in(pid): return [t for t in DATA["topics"] if t["phase"] == pid]

# ---------------- CSS (plain string; three-state theming via tokens) ----------------
CSS = """
:root{
  --bg:#f4f6f8; --panel:#ffffff; --panel-2:#eef1f4; --ink:#141b24; --muted:#5a6672;
  --line:#d9dfe5; --line-strong:#c3ccd4;
  --accent:#0e7490; --accent-ink:#ffffff; --accent-soft:#d7edf1;
  --brass:#946a10; --brass-soft:#f2e8cf;
  --done:#1f875a; --done-bg:#dcf0e6; --wip:#a9700a; --wip-bg:#f6ecd3;
  --gpu:#5b4bc4; --gpu-bg:#e6e2f7; --todo:#6b7683; --todo-bg:#e7ebef; --defer:#8a94a0; --defer-bg:#eceff2;
  --shadow:0 1px 2px rgba(20,27,36,.06),0 4px 14px rgba(20,27,36,.05);
  --radius:14px;
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#0e1622; --panel:#151f2c; --panel-2:#1b2735; --ink:#e7ecf2; --muted:#93a1b0;
    --line:#26323f; --line-strong:#33414f;
    --accent:#45c4d6; --accent-ink:#08222a; --accent-soft:#123642;
    --brass:#d9ad4e; --brass-soft:#33290f;
    --done:#43c489; --done-bg:#123026; --wip:#e0a83e; --wip-bg:#322612;
    --gpu:#9d90ee; --gpu-bg:#23204a; --todo:#8c99a7; --todo-bg:#1c2733; --defer:#6f7c8a; --defer-bg:#1a2430;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.28);
  }
}
:root[data-theme="dark"]{
  --bg:#0e1622; --panel:#151f2c; --panel-2:#1b2735; --ink:#e7ecf2; --muted:#93a1b0;
  --line:#26323f; --line-strong:#33414f;
  --accent:#45c4d6; --accent-ink:#08222a; --accent-soft:#123642;
  --brass:#d9ad4e; --brass-soft:#33290f;
  --done:#43c489; --done-bg:#123026; --wip:#e0a83e; --wip-bg:#322612;
  --gpu:#9d90ee; --gpu-bg:#23204a; --todo:#8c99a7; --todo-bg:#1c2733; --defer:#6f7c8a; --defer-bg:#1a2430;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.28);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:6px}

/* top bar */
header.bar{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 86%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line);
  display:flex;align-items:center;gap:16px;padding:11px clamp(16px,4vw,40px)}
.bar .brand{font-weight:600;letter-spacing:.02em;display:flex;align-items:center;gap:10px}
.spark{width:12px;height:18px;border-radius:2px;background:linear-gradient(var(--accent),var(--brass));display:inline-block;flex:0 0 auto}
.bar .grow{flex:1}
.meter{display:flex;align-items:center;gap:9px;font-size:13px;color:var(--muted)}
.meter .track{width:120px;height:7px;border-radius:99px;background:var(--panel-2);overflow:hidden;border:1px solid var(--line)}
.meter .fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--brass))}
.btn{font:inherit;font-size:13px;color:var(--ink);background:var(--panel);border:1px solid var(--line-strong);
  border-radius:9px;padding:6px 11px;cursor:pointer;display:inline-flex;gap:7px;align-items:center}
.btn:hover{border-color:var(--accent)}

.wrap{max-width:1120px;margin:0 auto;padding:0 clamp(16px,4vw,40px)}

/* hero */
.hero{padding:54px 0 22px}
.eyebrow{font:600 12px/1 "IBM Plex Mono",monospace;letter-spacing:.22em;text-transform:uppercase;color:var(--accent)}
.hero h1{font-weight:600;font-size:clamp(30px,5vw,50px);line-height:1.04;margin:16px 0 0;text-wrap:balance;letter-spacing:-.01em}
.hero .lede{font-family:"IBM Plex Serif",Georgia,serif;font-size:clamp(16px,2vw,20px);color:var(--muted);
  max-width:64ch;margin:18px 0 0;line-height:1.5}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:34px 0 8px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px 18px;box-shadow:var(--shadow)}
.stat .n{font:600 30px/1 "IBM Plex Mono",monospace;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat .k{font-size:12.5px;color:var(--muted);margin-top:7px;letter-spacing:.02em}
.stat.capstone{border-color:var(--brass);background:linear-gradient(180deg,var(--brass-soft),var(--panel))}
.stat.capstone .n{color:var(--brass)}

/* legend */
.legend{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin:18px 0 2px;font-size:12.5px;color:var(--muted)}
.legend b{color:var(--ink);font-weight:600;margin-right:2px}

/* pills */
.pill{font:600 11px/1.4 "IBM Plex Mono",monospace;padding:2px 7px;border-radius:6px;letter-spacing:.02em;white-space:nowrap;border:1px solid transparent}
.s-done{color:var(--done);background:var(--done-bg)} .s-wip{color:var(--wip);background:var(--wip-bg)}
.s-gpu{color:var(--gpu);background:var(--gpu-bg)} .s-todo{color:var(--todo);background:var(--todo-bg)}
.s-defer{color:var(--defer);background:var(--defer-bg)} .s-na{color:var(--muted);background:transparent;border-color:var(--line)}

/* phase band */
.phase{padding:30px 0;border-top:1px solid var(--line)}
.phase-head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.phase-num{font:600 13px/1 "IBM Plex Mono",monospace;color:var(--accent);border:1px solid var(--line-strong);
  border-radius:7px;padding:5px 8px;background:var(--panel)}
.phase-head h2{font-size:clamp(19px,2.4vw,25px);font-weight:600;margin:0;letter-spacing:-.01em}
.phase-head .wk{font:600 12px/1 "IBM Plex Mono",monospace;color:var(--muted);letter-spacing:.03em}
.phase-desc{color:var(--muted);margin:8px 0 18px;max-width:70ch}

/* topic grid */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:15px 16px 14px;
  box-shadow:var(--shadow);display:flex;flex-direction:column;gap:10px;transition:border-color .15s,transform .15s}
.card:hover{border-color:var(--accent);transform:translateY(-2px)}
.card .top{display:flex;align-items:center;gap:9px}
.card .tid{font:600 13px/1 "IBM Plex Mono",monospace;color:var(--accent);background:var(--accent-soft);
  padding:3px 7px;border-radius:6px}
.card .day{font:600 11px/1 "IBM Plex Mono",monospace;color:var(--muted);margin-left:auto}
.card h3{font-size:15.5px;font-weight:600;margin:0;line-height:1.3;letter-spacing:-.005em}
.card h3 a{color:var(--ink)} .card h3 a:hover{color:var(--accent)}
.card .hook{font-size:12.5px;color:var(--muted);line-height:1.45}
.card .hook .lbl{font:600 10px/1 "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--brass)}
.card .status{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto;padding-top:4px}
.card .status .k{font:600 10px/1.4 "IBM Plex Mono",monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;align-self:center}
.defer-tag{font:600 10px/1 "IBM Plex Mono",monospace;color:var(--defer);border:1px solid var(--line-strong);border-radius:5px;padding:2px 5px}

/* capstone band */
.capstone-band{margin:30px 0;padding:26px clamp(18px,3vw,30px);border:1px solid var(--brass);border-radius:18px;
  background:linear-gradient(180deg,var(--brass-soft),var(--panel))}
.capstone-band .eyebrow{color:var(--brass)}
.capstone-band h2{font-size:clamp(21px,2.6vw,27px);margin:12px 0 6px;font-weight:600}
.capstone-band p{color:var(--muted);max-width:74ch;margin:6px 0}
.surfaces{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:18px}
.surface{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.surface h4{margin:0 0 6px;font-size:14.5px}
.surface p{font-size:12.5px;margin:0}

/* nav links row */
.navlinks{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 6px}
.navlinks a{background:var(--panel);border:1px solid var(--line-strong);border-radius:9px;padding:8px 13px;font-size:13.5px;color:var(--ink)}
.navlinks a:hover{border-color:var(--accent);text-decoration:none}
.navlinks a .mono{color:var(--accent)}

footer{border-top:1px solid var(--line);margin-top:36px;padding:26px 0 60px;color:var(--muted);font-size:13px}
.disclaimer{background:var(--panel-2);border:1px solid var(--line);border-radius:11px;padding:12px 15px;margin:14px 0;font-size:12.5px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

def esc(s): return html.escape(str(s))

# ---------------- build cards ----------------
def card(t):
    s = STATUS.get(t["id"], {})
    day = t["day"]
    defer = '<span class="defer-tag" title="Deferrable on the accelerated core path">◇ optional</span>' if t["deferrable"] else ""
    folder = f'{GH}/phases/{t["folder"]}'
    return f"""      <article class="card">
        <div class="top"><span class="tid mono">{esc(t['id'])}</span>{defer}<span class="day mono">Day {day}</span></div>
        <h3><a href="{folder}">{esc(t['title'])}</a></h3>
        <div class="hook"><span class="lbl">AlphaDesk</span> {esc(t['hook'])}</div>
        <div class="status">
          <span class="k">code</span>{cell(s.get('code','scheduled'))}
          <span class="k">tests</span>{cell(s.get('tests','scheduled'))}
          <span class="k">bench</span>{cell(s.get('bench','scheduled'))}
          <span class="k">video</span>{cell(s.get('video','scheduled'))}
        </div>
      </article>"""

def phase_band(p):
    ts = topics_in(p["id"])
    cards = "\n".join(card(t) for t in ts)
    return f"""    <section class="phase" id="{p['id']}">
      <div class="phase-head">
        <span class="phase-num mono">{p['id'].upper()}</span>
        <h2>{esc(p['name'])}</h2>
        <span class="wk mono">{esc(p['weeks'])}</span>
      </div>
      <p class="phase-desc">{esc(p['desc'])}</p>
      <div class="grid">
{cards}
      </div>
    </section>"""

phases_html = "\n".join(phase_band(p) for p in DATA["phases"])
updated = datetime.date.today().isoformat()

HEAD_INNER = f"""<title>AI Systems Forge</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;1,400&display=swap">
<style>{CSS}</style>"""

BODY_INNER = f"""<header class="bar">
  <span class="brand"><span class="spark"></span> AI Systems Forge</span>
  <span class="grow"></span>
  <div class="meter" title="Topics with code complete">
    <span class="mono">{done_n}/{total_n}</span>
    <span class="track"><span class="fill" style="width:{pct}%"></span></span>
    <span class="mono">{pct}%</span>
  </div>
  <a class="btn" href="{GH}">Repo ↗</a>
  <button class="btn" id="themeBtn" aria-label="Toggle theme">◐ Theme</button>
</header>

<div class="wrap">
  <section class="hero">
    <div class="eyebrow">Build-your-own · {DATA['total_topics']} systems · one trading-desk capstone</div>
    <h1>Learn the modern AI stack by building every layer of it.</h1>
    <p class="lede">A hands-on program for a senior fintech engineer: implement {DATA['total_topics']} AI systems from the original papers — autograd to transformers to GPU kernels to agents — each with a slow-paced Remotion video lesson, step-laddered code with tests, and a home inside <strong>AlphaDesk</strong>, a paper-trading AI desk that mirrors an OMS/EMS platform.</p>

    <div class="stats">
      <div class="stat"><div class="n">{DATA['total_topics']}</div><div class="k">build-your-own topics · {DATA.get('total_modules',52)} modules</div></div>
      <div class="stat"><div class="n">{len(DATA['phases'])}</div><div class="k">dependency-ordered phases</div></div>
      <div class="stat"><div class="n">{DATA['total_eps']}</div><div class="k">planned video episodes</div></div>
      <div class="stat capstone"><div class="n">1</div><div class="k">capital-markets capstone · AlphaDesk</div></div>
    </div>

    <div class="legend">
      <span><b>Status per topic →</b></span>
      <span>{cell('done')} code / tests / bench complete</span>
      <span>{cell('in_progress')} in progress or pending</span>
      <span>{cell('awaiting-4070')} GPU bench awaiting your RTX&nbsp;4070</span>
      <span>{cell('scheduled')} scheduled, not yet generated</span>
      <span><span class="defer-tag">◇ optional</span> deferrable on the fast path</span>
    </div>

    <div class="navlinks">
      <a href="{GH}/blob/{DATA['branch']}/MASTER_PLAN.md">📘 <span class="mono">MASTER_PLAN.md</span> — full curriculum &amp; capsules</a>
      <a href="{GH}/blob/{DATA['branch']}/EXECUTION/DAY_PLAN.md">🗓 <span class="mono">DAY_PLAN.md</span> — 15-day schedule</a>
      <a href="{GH}/blob/{DATA['branch']}/EXECUTION/LEDGER.md">📊 <span class="mono">LEDGER.md</span> — live progress</a>
      <a href="{GH}/tree/{DATA['branch']}/phases">📂 phases/ — all topic folders</a>
    </div>
  </section>

  <section class="capstone-band">
    <div class="eyebrow">The thread through every topic</div>
    <h2>AlphaDesk — a fictional AI trading desk</h2>
    <p>Every one of the {DATA['total_topics']} topics plugs into one growing platform, so your learning compounds into a system you can demo. Paper-trading only — no real orders, money, brokerage systems, or market-data redistribution.</p>
    <div class="surfaces">
      <div class="surface"><h4>Research Copilot</h4><p>Cited, reasoned answers over filings, transcripts &amp; market data. <span class="mono">RAG · GraphRAG · KG · CoT · agent · text-to-SQL</span></p></div>
      <div class="surface"><h4>Order Workflow (paper)</h4><p>NL order ticket → grammar + risk validation → simulated fills. <span class="mono">function-calling · CFG order-DSL · guardrails · feature store</span></p></div>
      <div class="surface"><h4>Compliance &amp; Ops</h4><p>Guardrail perimeter, eval harness, interpretability, model gateway. <span class="mono">guardrails · evals · SAE · gateway</span></p></div>
    </div>
  </section>

{phases_html}

  <footer>
    <div class="disclaimer">⚠️ <strong>Educational simulation.</strong> AlphaDesk is fictional and paper-only — it never touches real orders, real money, real brokerage systems, or redistributes market data.</div>
    <p>Content is generated incrementally by a scheduled Claude Opus session (~3:00 AM IST daily), following <span class="mono">EXECUTION/DAILY_PROMPT.md</span>. This page regenerates from <span class="mono">EXECUTION/status.json</span> — last updated {updated}. Branch <span class="mono">{DATA['branch']}</span>.</p>
  </footer>
</div>

<script>
(function(){{
  var KEY="aisf-theme";
  try{{var s=localStorage.getItem(KEY); if(s) document.documentElement.setAttribute("data-theme",s);}}catch(e){{}}
  document.getElementById("themeBtn").addEventListener("click",function(){{
    var cur=document.documentElement.getAttribute("data-theme");
    var mql=window.matchMedia("(prefers-color-scheme: dark)");
    var next = cur? (cur==="dark"?"light":"dark") : (mql.matches?"light":"dark");
    document.documentElement.setAttribute("data-theme",next);
    try{{localStorage.setItem(KEY,next);}}catch(e){{}}
  }});
}})();
</script>"""

# Repo file: full standalone document (opened via GitHub/local/Pages).
DOCTYPE = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
(ROOT / "index.html").write_text(DOCTYPE + HEAD_INNER + "\n</head>\n<body>\n" + BODY_INNER + "\n</body>\n</html>\n")

# Artifact-review copy: skeleton-less (the Artifact tool adds the skeleton).
import os as _os
_art = _os.environ.get("ARTIFACT_OUT")
if _art:
    pathlib.Path(_art).write_text(HEAD_INNER + "\n" + BODY_INNER + "\n")

# ---------------- regenerate LEDGER.md from the same data ----------------
def L(v): return st(v)[0]
rows = []
for t in DATA["topics"]:
    s = STATUS.get(t["id"], {})
    link = f'phases/{t["folder"]}/'
    def m(k):
        lab = L(s.get(k, "scheduled"))
        return {"done":"✅","pass":"✅","wip":"🟡","pending":"🟡","4070":"🖥️","deferred":"⏭️","todo":"⬜","—":"—"}.get(lab,"⬜")
    rows.append(f'| {t["day"]} | {t["phase"].upper()} | [{t["id"]}]({link}) | {t["title"]} | {m("code")} | {m("tests")} | {m("bench")} | {m("video")} | {m("wired")} | {s.get("note","")} |')
ledger = f"""# Progress Ledger  ·  {done_n}/{total_n} code-complete ({pct}%)

Regenerated from `EXECUTION/status.json` by `scripts/gen_index.py`. Legend: ✅ done · 🟡 in progress/pending · 🖥️ awaiting-4070 · ⏭️ deferred · ⬜ scheduled.
A run picks the earliest row whose **Code** isn't ✅, finishes it, continues down. Video may lag (best-effort).

| Day | Phase | ID | Topic | Code | Tests | Bench | Video | Wired | Notes |
|----:|:------|:---|:------|:----:|:-----:|:-----:|:-----:|:-----:|:------|
{chr(10).join(rows)}

_Last updated: {updated}._
"""
(ROOT / "EXECUTION/LEDGER.md").write_text(ledger)
print(f"index.html + LEDGER.md regenerated · {done_n}/{total_n} code-complete ({pct}%)")
