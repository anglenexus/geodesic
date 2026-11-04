#!/usr/bin/env python3
"""
capacity_dashboard.py - render .nebius/capacity_log.jsonl (written by capacity_watch.py) as a local HTML page.

    python capacity_dashboard.py            # rebuild .nebius/capacity.html every 60 s (the page reloads every 15 min)
    python capacity_dashboard.py --once
    open .nebius/capacity.html
"""
import argparse
import collections
import datetime
import html
import json
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
LOG = HERE / ".nebius" / "capacity_log.jsonl"
OUT = HERE / ".nebius" / "capacity.html"
REGIONS = ["us-north1", "uk-south1", "eu-west2"]
CSS = """
:root{--bg:#f6f8fa;--card:#fff;--ink:#1f2328;--mute:#59636e;--line:#d0d7de;--ok:#1a7f37;--bad:#cf222e;--warn:#9a6700}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--mute:#8d96a0;--line:#30363d;--ok:#3fb950;--bad:#f85149;--warn:#d29922}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.sub{color:var(--mute)}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}.tile b{display:block;font-size:20px}
.tile span{color:var(--mute);font-size:12px}section{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mute);font-weight:600}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad)}.warn{color:var(--warn)}
.grid{display:flex;gap:2px;flex-wrap:wrap;padding:12px}.cell{width:10px;height:22px;border-radius:2px;background:var(--line)}
.cell.ok{background:var(--ok)}.cell.bad{background:var(--bad);opacity:.55}
"""


def load():
    if not LOG.exists():
        return []
    return [json.loads(ln) for ln in LOG.read_text().splitlines() if ln.strip()]


def render():
    rows = load()
    neb = [r for r in rows if r.get("provider") == "nebius"]
    prime = [r for r in rows if r.get("provider") == "prime"]
    secured = [r for r in neb if r.get("result") == "SECURED"]
    last = rows[-1]["time"] if rows else "–"
    first = rows[0]["time"] if rows else "–"
    by_region = collections.defaultdict(list)
    for r in neb:
        by_region[r["region"]].append(r)
    prime_hits = [r for r in prime if r.get("result") == "offers"]
    others = [r for r in rows if r.get("provider") in ("vast", "shadeform")]
    other_hits = [r for r in others if r.get("result") == "offers"]
    prime_status = ("not configured (no PRIME_API_KEY)" if not prime else
                    f"{len(prime_hits)} of {len(prime)} checks had offers")
    status = (f"<span class='ok'>SECURED: {html.escape(secured[-1]['region'])} at {secured[-1]['time']}</span>"
              if secured else "<span class='warn'>watching: no regular 8x B300 yet</span>")

    tiles = [("Status", status), ("Checks (Nebius)", len(neb)), ("Watching since", first), ("Last check", last),
             ("Prime Intellect", prime_status),
             ("Vast.ai / Shadeform", f"{len(other_hits)} of {len(others)} checks had offers" if others else "–")]
    t_html = "".join(f"<div class='tile'><span>{k}</span><b>{v}</b></div>" for k, v in tiles)

    grid = ""
    for reg in REGIONS:
        cells = "".join(f"<div class='cell {'ok' if r['result'] == 'SECURED' else 'bad'}' title='{r['time']}: "
                        f"{html.escape(r.get('reason', r['result']))}'></div>" for r in by_region.get(reg, []))
        n_ok = sum(r["result"] == "SECURED" for r in by_region.get(reg, []))
        grid += (f"<tr><td>{reg}</td><td>{len(by_region.get(reg, []))}</td><td>{n_ok}</td>"
                 f"<td><div class='grid' style='padding:0'>{cells or '–'}</div></td></tr>")

    recent = ""
    for r in reversed(rows[-60:]):
        if r.get("provider") == "nebius":
            cls = "ok" if r["result"] == "SECURED" else "bad"
            recent += (f"<tr><td>{r['time']}</td><td>Nebius</td><td>{r['region']}</td><td class='{cls}'>{r['result']}</td>"
                       f"<td>{html.escape(r.get('reason', ''))}</td><td>{r.get('seconds', '')}</td></tr>")
        else:
            offers = r.get("offers") or []
            txt = "; ".join(f"{o['provider']} {o['region']} ${o['price']}/GPU-h ({o['stock']})" for o in offers) or \
                  html.escape(r.get("reason", "no in-stock, non-spot 8x B300"))
            name = {"prime": "Prime Intellect", "vast": "Vast.ai", "shadeform": "Shadeform"}.get(r.get("provider"), r.get("provider"))
            recent += (f"<tr><td>{r['time']}</td><td>{name}</td><td>all</td>"
                       f"<td class='{'ok' if offers else 'bad'}'>{r['result']}</td><td>{html.escape(txt)}</td><td></td></tr>")

    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<meta http-equiv='refresh' content='900'><title>B300 Capacity Watch</title><style>{CSS}</style></head><body><main>
<h1>8x B300 capacity watch</h1>
<p class='sub'>Regular (on-demand, not preemptible) 8x B300 for the Kimi K3 tests. Nebius: a create attempt per region every
15 min (failed attempts cost nothing and are deleted); on success the run starts automatically with a 150-min kill switch.
Prime Intellect, Vast.ai and Shadeform: in-stock, non-spot offers (reported only). Page rebuilt {datetime.datetime.now():%Y-%m-%d %H:%M:%S}; reloads every 15 min.</p>
<div class='tiles'>{t_html}</div>
<h2>Nebius by region</h2><section><table><tr><th>Region</th><th>Attempts</th><th>Secured</th><th>History (oldest left; hover for detail)</th></tr>{grid}</table></section>
<h2>Recent checks</h2><section><table><tr><th>Time</th><th>Provider</th><th>Region</th><th>Result</th><th>Detail</th><th>Seconds</th></tr>{recent}</table></section>
</main></body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        render()
        if a.once:
            print(OUT)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
