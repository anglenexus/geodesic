#!/usr/bin/env python3
"""
progress_dashboard.py - from the laptop: a live HTML page for a pieces_kimi_k3.py run, built from S3 (what is
done, what is cached) and Modal (is a GPU container running). Rebuilt every 30 s; the page reloads itself.

    python progress_dashboard.py --run-id kimi-20261103-...      # writes .nebius/progress_<run-id>.html
    open .nebius/progress_<run-id>.html
"""
import argparse
import collections
import datetime
import html
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pieces_kimi_k3 import BUCKET, PIECES, PREFIX  # noqa: E402

CSS = """
:root{--bg:#f6f8fa;--card:#fff;--ink:#1f2328;--mute:#59636e;--line:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--run:#0969da;--warn:#9a6700}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--mute:#8d96a0;--line:#30363d;--ok:#3fb950;--bad:#f85149;--run:#58a6ff;--warn:#d29922}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.sub{color:var(--mute)}.now{background:var(--card);border:2px solid var(--run);border-radius:10px;padding:16px;margin-top:16px}
.now .big{font-size:26px;font-weight:700;color:var(--run)}.now .note{color:var(--mute)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}.tile b{display:block;font-size:20px}
.tile span{color:var(--mute);font-size:12px}section{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--mute);font-weight:600}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.run{color:var(--run);font-weight:700}.mute{color:var(--mute)}.warn{color:var(--warn);font-weight:600}
.bar{display:flex;gap:3px;margin-top:10px}.bar div{flex:1;height:14px;border-radius:3px;background:var(--line)}
.bar .ok{background:var(--ok)}.bar .run{background:var(--run)}.bar .bad{background:var(--bad)}
code{font-size:12px}
"""


def s3_ls(prefix):
    p = subprocess.run(["aws", "s3", "ls", "--recursive", f"s3://{BUCKET}/{prefix}"], capture_output=True, text=True)
    out = []
    for ln in p.stdout.splitlines():
        parts = ln.split(None, 3)
        if len(parts) == 4:
            out.append(dict(time=f"{parts[0]} {parts[1]}", size=int(parts[2]), key=parts[3][len(prefix):]))
    return out


def s3_json(key):
    p = subprocess.run(["aws", "s3", "cp", "--only-show-errors", f"s3://{BUCKET}/{key}", "-"],
                       capture_output=True, text=True)
    try:
        return json.loads(p.stdout)
    except Exception:
        return None


def modal_state():
    p = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True)
    try:
        apps = [a for a in json.loads(p.stdout) if "kimi-k3-serving" in json.dumps(a)]
    except Exception:
        return "unknown"
    if not apps:
        return "no app"
    a = apps[0]
    return f"{a.get('State', a.get('state', '?'))}, {a.get('Tasks', a.get('tasks', '?'))} container(s)"


def size(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:,.0f} {u}" if u == "B" else f"{n:,.1f} {u}"
        n /= 1024


def render(run_id, out):
    root = f"{PREFIX}/{run_id}/"
    objs = s3_ls(root)
    files = [o for o in objs if o["key"].startswith("files/")]
    done = {o["key"][5:-5]: o for o in objs if o["key"].startswith("done/")}
    attempts = sorted((o for o in objs if o["key"].startswith("attempts/")), key=lambda o: o["key"].split("-")[1])
    fails = collections.Counter(o["key"][9:].split("-")[0] for o in attempts
                                if not o["key"].endswith(("-rc0.json", "-rc99.json")))
    cur = s3_json(root + "status/current.json") or {}
    modal = modal_state()
    age = time.time() - cur["time"] if cur.get("time") else None
    next_piece = next((p for p, *_ in PIECES if p not in done and fails[p] < 2), None)

    # the "now" box
    if not objs:
        now_big, now_note = "not started", "no files for this run on S3 yet"
    elif cur.get("piece") == "finished":
        now_big, now_note = "finished", cur.get("note", "")
    else:
        now_big = f"{cur.get('piece', '?')}  ·  {cur.get('stage', '?')}"
        now_note = f"{cur.get('note', '')} — started {age / 60:.1f} min ago" if age is not None else ""
        if age is not None and age > 8 * 60 and cur.get("piece") != "server":
            now_note += "  ⚠ longer than a piece should take: preempted or stuck? (check Modal)"
    bar = "".join(f"<div class='{'ok' if p in done else 'run' if p == cur.get('piece') else 'bad' if fails[p] >= 2 else ''}' "
                  f"title='{p}'></div>" for p, *_ in PIECES)

    # pieces table
    rows = ""
    for i, (p, stage, what) in enumerate(PIECES, 1):
        if p in done:
            d = s3_json(root + f"done/{p}.json") or {}
            st, cls = f"done in {d.get('seconds', '?')} s", "ok"
        elif p == cur.get("piece"):
            st, cls = "running now", "run"
        elif fails[p] >= 2:
            st, cls = "skipped: failed twice", "bad"
        elif fails[p]:
            st, cls = f"pending ({fails[p]} failed try)", "warn"
        else:
            st, cls = ("next" if p == next_piece else "pending"), "mute"
        rows += f"<tr><td>{i}</td><td><code>{p}</code></td><td>{stage}</td><td class='{cls}'>{st}</td><td>{html.escape(what)}</td></tr>"
    if cur.get("piece") == "server":
        rows = (f"<tr><td>–</td><td><code>server</code></td><td>gate</td><td class='run'>running now</td>"
                f"<td>{html.escape(cur.get('note', ''))}</td></tr>") + rows

    # cached data, grouped by what it is
    groups = collections.defaultdict(list)
    for f in files:
        rel = f["key"][6:]
        top = rel.split("/")[0]
        g = ("Phase 4 kernels" if top.startswith("fi_") else "Phase 3 serving" if top.startswith("kimi-k3_") else
             "Export (for git)" if top == "export" else "Server starts" if top == "server_starts" else top)
        groups[g].append((rel, f))
    cached = ""
    for g in ["Phase 4 kernels", "Phase 3 serving", "Server starts", "Export (for git)"] + \
             sorted(k for k in groups if k not in ("Phase 4 kernels", "Phase 3 serving", "Server starts", "Export (for git)")):
        if g not in groups:
            continue
        items = sorted(groups[g])
        total = sum(f["size"] for _, f in items)
        cached += (f"<tr><td colspan='3'><b>{html.escape(g)}</b> — {len(items)} files, {size(total)}</td></tr>"
                   + "".join(f"<tr><td><code>{html.escape(rel)}</code></td><td>{size(f['size'])}</td><td>{f['time']}</td></tr>"
                             for rel, f in items))
    tiles = [("Pieces done", f"{len(done)} / {len(PIECES)}"), ("Next piece", next_piece or "–"),
             ("Files on S3", f"{len(files)} ({size(sum(f['size'] for f in files))})"),
             ("Attempts logged", len(attempts)), ("Modal", html.escape(modal))]
    t_html = "".join(f"<div class='tile'><span>{k}</span><b>{v}</b></div>" for k, v in tiles)
    att = "".join(f"<tr><td>{o['time']}</td><td><code>{html.escape(o['key'][9:])}</code></td></tr>" for o in reversed(attempts[-30:]))

    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<meta http-equiv='refresh' content='30'><title>Kimi K3 Progress</title><style>{CSS}</style></head><body><main>
<h1>Kimi K3 on 8x B300: run progress</h1>
<p class='sub'>Run <code>{html.escape(run_id)}</code> · S3 <code>s3://{BUCKET}/{root}</code> · rebuilt {datetime.datetime.now():%H:%M:%S}, reloads every 30 s.
Pieces are 5 min or less; each is uploaded to S3 the moment it finishes, and a restarted machine continues at the next one.</p>
<div class='now'><div class='sub'>Now processing</div><div class='big'>{html.escape(now_big)}</div>
<div class='note'>{html.escape(now_note)}</div><div class='bar'>{bar}</div></div>
<div class='tiles'>{t_html}</div>
<h2>Pieces and stages</h2><section><table><tr><th>#</th><th>Piece</th><th>Stage</th><th>Status</th><th>What</th></tr>{rows}</table></section>
<h2>Data already on S3</h2><section><table><tr><th>File</th><th>Size</th><th>Uploaded</th></tr>{cached or "<tr><td colspan='3' class='mute'>nothing yet</td></tr>"}</table></section>
<h2>Attempts (newest first)</h2><section><table><tr><th>Time</th><th>Piece, start, exit code</th></tr>{att or "<tr><td colspan='2' class='mute'>none yet</td></tr>"}</table></section>
</main></body></html>"""
    out.write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    out = HERE / ".nebius" / f"progress_{a.run_id}.html"
    out.parent.mkdir(exist_ok=True)
    while True:
        render(a.run_id, out)
        if a.once:
            print(out)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
