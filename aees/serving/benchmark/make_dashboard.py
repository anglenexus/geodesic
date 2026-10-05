#!/usr/bin/env python3
"""
make_dashboard.py - turn a run_bench.py results directory into one self-contained,
interactive dashboard (dashboard.html) plus summary.csv.

    python make_dashboard.py results/<run-id> [--slo-ttft-ms 500] [--slo-itl-ms 50] [--slo-pct 99]

All percentiles, including p99.9, are computed here from per-request data
(bench_serving --output-details), so every number traces back to raw measurements.
"""
import argparse
import html
import json
import pathlib
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PCTS = [50, 90, 99, 99.9]
ARM_ORDER = ["baseline", "no_radix", "fp8_kv", "page16", "triton"]
ARM_COLORS = {"baseline": "#1f6feb", "no_radix": "#d1495b", "fp8_kv": "#2a9d8f",
              "page16": "#e9a23b", "triton": "#6c757d"}
WL_ORDER = ["agent", "chat", "long"]
WL_COLORS = {"agent": "#1f6feb", "chat": "#2a9d8f", "long": "#e9a23b"}
PCT_STYLE = {50: ("#8ab4f8", "dot"), 90: ("#4c8df6", "dashdot"), 99: ("#1f6feb", "solid"),
             99.9: ("#0b2a6b", "dash")}
SLO_COLOR = "#d1495b"


def lab(p):
    return "p" + f"{p:g}".replace(".", "")


def plab(p):
    return f"p{p:g}"


def last_json(path):
    try:
        lines = [ln for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
        return json.loads(lines[-1])
    except Exception:
        return None


# ---------------------------------------------------------------- loading
def point_metrics(rec, res):
    ttfts = np.asarray(res.get("ttfts") or [], float)
    itls = res.get("itls") or []
    errs = res.get("errors") or []
    ok = np.ones(len(ttfts), bool)
    if len(errs) == len(ttfts):
        ok &= np.array([not e for e in errs], bool)
    ok &= ttfts > 0
    scale = 1000.0  # bench_serving stores per-request times in seconds
    if ok.any() and res.get("median_ttft_ms"):
        scale = 1000.0 if res["median_ttft_ms"] / np.median(ttfts[ok]) > 100 else 1.0
    ttft = ttfts[ok] * scale
    itl_lists = []
    if len(itls) == len(ttfts):
        itl_lists = [np.asarray(x, float) * scale for x, k in zip(itls, ok) if k]
    itl = np.concatenate(itl_lists) if itl_lists else np.array([])
    e2e = ttft + np.array([x.sum() for x in itl_lists]) if len(itl_lists) == len(ttft) else np.array([])

    n = rec.get("num_prompts") or len(ttfts)
    completed = int(res.get("completed", ok.sum()))
    dur = res.get("duration") or np.nan
    m = dict(arm=rec["arm"], workload=rec["workload"], mode=rec["mode"], value=rec["value"],
             num_prompts=n, completed=completed,
             failed=max(n - completed, 0),
             duration_s=dur,
             req_s=res.get("request_throughput") or (completed / dur if dur else np.nan),
             out_tok_s=res.get("output_throughput", np.nan),
             in_tok_s=res.get("input_throughput", np.nan),
             n_ttft=len(ttft), n_itl=len(itl))
    fallback = {"ttft": "ttft", "itl": "itl", "e2e": "e2e_latency"}
    for name, arr in (("ttft", ttft), ("itl", itl), ("e2e", e2e)):
        for p in PCTS:
            m[f"{name}_{lab(p)}"] = float(np.percentile(arr, p)) if arr.size else np.nan
        if not arr.size:  # no per-request detail: fall back to bench_serving's summary
            m[f"{name}_p50"] = res.get(f"median_{fallback[name]}_ms", np.nan)
            m[f"{name}_p99"] = res.get(f"p99_{fallback[name]}_ms", np.nan)
    return m, dict(ttft=ttft, itl=itl, e2e=e2e)


def load(run_dir):
    recs = {}
    man = run_dir / "manifest.jsonl"
    if man.exists():
        for line in man.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("rc") == 0:
                    recs[(r["arm"], r["workload"], r["mode"], str(r["value"]))] = r
    rows, raw = [], {}
    for k, r in recs.items():
        res = last_json(run_dir / r["file"])
        if res is not None:
            m, arrays = point_metrics(r, res)
            rows.append(m)
            raw[k] = arrays
    df = pd.DataFrame(rows)
    if not df.empty:
        df["arm"] = pd.Categorical(df["arm"], [a for a in ARM_ORDER if a in set(df["arm"])] +
                                   sorted(set(df["arm"]) - set(ARM_ORDER)))
        df = df.sort_values(["arm", "workload", "mode", "value"]).reset_index(drop=True)
    return df, raw


def server_facts(run_dir):
    cap, hit = {}, []
    for arm_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        log = arm_dir / "server.log"
        if log.exists():
            mt = re.search(r"max_total_num_tokens=(\d+)", log.read_text(errors="ignore"))
            if mt:
                cap[arm_dir.name] = int(mt.group(1))
        for mf in arm_dir.glob("metrics_*.txt"):
            mh = re.search(r"^sglang:cache_hit_rate(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
                           mf.read_text(errors="ignore"), re.M)
            if mh:
                hit.append(dict(arm=arm_dir.name, workload=mf.stem.split("_", 1)[1],
                                hit=float(mh.group(1))))
    return cap, pd.DataFrame(hit)


# ---------------------------------------------------------------- figures
def slo_trace(x=None, y=None, show=True):
    return go.Scatter(x=x, y=y, mode="lines", name="SLO", legendgroup="SLO", showlegend=show,
                      line=dict(color=SLO_COLOR, dash="dash", width=1.5), hoverinfo="skip")


def fig_capacity(df, slo):
    b = df[(df.arm == "baseline") & (df["mode"] == "conc")]
    if b.empty:
        return None
    P = plab(slo["pct"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Throughput vs TTFT {P}", f"Throughput vs ITL {P}"))
    ymax = b.out_tok_s.max() * 1.15
    for wl in [w for w in WL_ORDER if w in set(b.workload)]:
        d = b[b.workload == wl].sort_values("value")
        for col, met in ((1, "ttft"), (2, "itl")):
            fig.add_trace(go.Scatter(
                x=d[f"{met}_{lab(slo['pct'])}"], y=d.out_tok_s, mode="lines+markers+text",
                text=[f"c={int(v)}" for v in d.value], textposition="top left", textfont=dict(size=10),
                name=wl, legendgroup=wl, showlegend=col == 1, line=dict(color=WL_COLORS.get(wl)),
                hovertemplate=f"{wl} c=%{{text}}<br>{met.upper()} {P} %{{x:.1f}} ms<br>%{{y:,.0f}} tok/s<extra></extra>"),
                row=1, col=col)
    fig.add_trace(slo_trace([slo["ttft"]] * 2, [0, ymax]), row=1, col=1)
    fig.add_trace(slo_trace([slo["itl"]] * 2, [0, ymax], show=False), row=1, col=2)
    fig.update_xaxes(type="log", title_text=f"TTFT {P} (ms, log)", row=1, col=1)
    fig.update_xaxes(type="log", title_text=f"ITL {P} (ms, log)", row=1, col=2)
    fig.update_yaxes(title_text="Output throughput (tok/s)", range=[0, ymax])
    fig.update_layout(height=460)
    return fig


def fig_percentiles(df):
    b = df[(df.arm == "baseline") & (df["mode"] == "conc")]
    wls = [w for w in WL_ORDER if w in set(b.workload)]
    if not wls:
        return None
    mets = [("ttft", "TTFT"), ("itl", "ITL"), ("e2e", "End-to-end")]
    fig = make_subplots(rows=len(wls), cols=3, vertical_spacing=0.09, horizontal_spacing=0.07,
                        subplot_titles=[f"{wl}: {t} (ms)" for wl in wls for _, t in mets])
    for i, wl in enumerate(wls, 1):
        d = b[b.workload == wl].sort_values("value")
        for j, (m, _) in enumerate(mets, 1):
            for p in PCTS:
                color, dash = PCT_STYLE[p]
                fig.add_trace(go.Scatter(
                    x=d.value, y=d[f"{m}_{lab(p)}"], mode="lines+markers", name=plab(p),
                    legendgroup=plab(p), showlegend=(i == 1 and j == 1),
                    line=dict(color=color, dash=dash), marker=dict(size=5),
                    hovertemplate=f"c=%{{x}}<br>{plab(p)} %{{y:.1f}} ms<extra></extra>"), row=i, col=j)
    fig.update_xaxes(type="log", title_text="Concurrency (log)")
    fig.update_yaxes(type="log")
    fig.update_layout(height=300 * len(wls) + 80)
    return fig


def fig_rate(df, slo):
    d = df[(df.arm == "baseline") & (df["mode"] == "rate")].sort_values("value")
    if d.empty:
        return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Achieved vs offered load", "TTFT vs offered load"))
    top = max(d.value.max(), d.req_s.max()) * 1.1
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="ideal (achieved = offered)",
                             line=dict(color="#adb5bd", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.value, y=d.req_s, mode="lines+markers", name="achieved req/s",
                             line=dict(color=WL_COLORS["agent"])), row=1, col=1)
    for p in (99, 99.9):
        color, dash = PCT_STYLE[p]
        fig.add_trace(go.Scatter(x=d.value, y=d[f"ttft_{lab(p)}"], mode="lines+markers",
                                 name=f"TTFT {plab(p)}", line=dict(color=color, dash=dash)), row=1, col=2)
    fig.add_trace(slo_trace([0, top], [slo["ttft"]] * 2), row=1, col=2)
    fig.update_xaxes(title_text="Offered load (req/s, Poisson)")
    fig.update_yaxes(title_text="Achieved (req/s)", row=1, col=1)
    fig.update_yaxes(title_text="TTFT (ms, log)", type="log", row=1, col=2)
    fig.update_layout(height=420)
    return fig


def common_levels(c, wl, arms):
    sets = [set(c[(c.arm == a) & (c.workload == wl)].value) for a in arms]
    return sorted(set.intersection(*sets)) if sets else []


def fig_arm_bars(df, slo):
    c = df[df["mode"] == "conc"]
    arms = [a for a in ARM_ORDER if a in set(c.arm)]
    if len(arms) < 2:
        return None
    wls = [w for w in ["agent", "chat"] if len(common_levels(c, w, arms)) > 0]
    if not wls:
        return None
    P = plab(slo["pct"])
    cols = [("out_tok_s", "Output tok/s"), (f"ttft_{lab(slo['pct'])}", f"TTFT {P} (ms)"),
            (f"itl_{lab(slo['pct'])}", f"ITL {P} (ms)")]
    fig = make_subplots(rows=len(wls), cols=3, vertical_spacing=0.14,
                        subplot_titles=[f"{wl}: {t}" for wl in wls for _, t in cols])
    for i, wl in enumerate(wls, 1):
        lv = common_levels(c, wl, arms)
        for a in arms:
            d = c[(c.arm == a) & (c.workload == wl) & (c.value.isin(lv))].sort_values("value")
            for j, (m, _) in enumerate(cols, 1):
                fig.add_trace(go.Bar(x=[f"c={int(v)}" for v in d.value], y=d[m], name=a, legendgroup=a,
                                     showlegend=(i == 1 and j == 1), marker_color=ARM_COLORS.get(a),
                                     hovertemplate=f"{a} %{{x}}: %{{y:,.1f}}<extra></extra>"), row=i, col=j)
    fig.update_layout(barmode="group", height=330 * len(wls) + 80)
    return fig


def fig_arm_curves(df, slo):
    c = df[df["mode"] == "conc"]
    arms = [a for a in ARM_ORDER if a in set(c.arm)]
    wls = [w for w in WL_ORDER if w in set(c.workload) and c[c.workload == w].arm.nunique() > 1]
    if len(arms) < 2 or not wls:
        return None
    P = plab(slo["pct"])
    fig = make_subplots(rows=1, cols=len(wls), subplot_titles=[f"{wl}" for wl in wls])
    ymax = c[c.workload.isin(wls)].out_tok_s.max() * 1.15
    for j, wl in enumerate(wls, 1):
        for a in arms:
            d = c[(c.arm == a) & (c.workload == wl)].sort_values("value")
            if d.empty:
                continue
            fig.add_trace(go.Scatter(
                x=d[f"ttft_{lab(slo['pct'])}"], y=d.out_tok_s, mode="lines+markers", name=a, legendgroup=a,
                showlegend=j == 1, line=dict(color=ARM_COLORS.get(a)),
                text=[f"c={int(v)}" for v in d.value],
                hovertemplate=f"{a} %{{text}}<br>TTFT {P} %{{x:.1f}} ms<br>%{{y:,.0f}} tok/s<extra></extra>"),
                row=1, col=j)
        fig.add_trace(slo_trace([slo["ttft"]] * 2, [0, ymax], show=j == 1), row=1, col=j)
    fig.update_xaxes(type="log", title_text=f"TTFT {P} (ms, log)")
    fig.update_yaxes(title_text="Output tok/s", range=[0, ymax], col=1)
    fig.update_layout(height=420)
    return fig


def fig_tail(raw):
    key = next((k for k in raw if k[0] == "baseline" and k[2] == "p999"), None)
    if key is None:
        return None
    fig = make_subplots(rows=1, cols=3, subplot_titles=("TTFT tail", "ITL tail", "End-to-end tail"))
    for j, m in enumerate(["ttft", "itl", "e2e"], 1):
        arr = raw[key][m]
        if arr.size < 10:
            continue
        s = np.logspace(0, np.log10(1.0 / arr.size), 300)
        fig.add_trace(go.Scatter(x=np.quantile(arr, 1 - s), y=s, mode="lines", showlegend=False,
                                 line=dict(color="#1f6feb"),
                                 hovertemplate="%{x:.1f} ms exceeded by %{y:.2%} of samples<extra></extra>"),
                      row=1, col=j)
        for p in (99, 99.9):
            v = np.percentile(arr, p)
            fig.add_trace(go.Scatter(x=[v], y=[1 - p / 100], mode="markers+text", text=[f"{plab(p)} {v:.0f} ms"],
                                     textposition="middle right", showlegend=False,
                                     marker=dict(color=PCT_STYLE[p][0], size=9)), row=1, col=j)
    fig.update_xaxes(type="log", title_text="Latency (ms, log)")
    fig.update_yaxes(type="log", title_text="Share of samples slower (log)", col=1)
    fig.update_layout(height=400)
    return fig


def fig_gpu(run_dir):
    frames = []
    for arm_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        f = arm_dir / "gpu.csv"
        if not f.exists() or f.stat().st_size == 0:
            continue
        d = pd.read_csv(f, names=["ts", "util", "mem", "power"], skipinitialspace=True)
        d["ts"] = pd.to_datetime(d.ts, errors="coerce", format="mixed")
        for col in ("util", "mem", "power"):
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["ts"])
        if d.empty:
            continue
        d["minutes"] = (d.ts - d.ts.iloc[0]).dt.total_seconds() / 60
        d["arm"] = arm_dir.name
        frames.append(d)
    if not frames:
        return None
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("GPU memory used (GiB)", "GPU utilization (%)", "Power (W)"))
    for d in frames:
        a = d.arm.iloc[0]
        for i, (col, sc) in enumerate((("mem", 1 / 1024), ("util", 1), ("power", 1)), 1):
            fig.add_trace(go.Scatter(x=d.minutes, y=d[col] * sc, mode="lines", name=a, legendgroup=a,
                                     showlegend=i == 1, line=dict(color=ARM_COLORS.get(a), width=1)),
                          row=i, col=1)
    fig.update_xaxes(title_text="Minutes since arm start", row=3, col=1)
    fig.update_layout(height=620)
    return fig


def fig_kv(cap, hit):
    if not cap and hit.empty:
        return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=("KV-cache pool capacity (tokens)",
                                                        "Prefix-cache hit rate at end of each sweep"))
    arms = [a for a in ARM_ORDER if a in cap]
    fig.add_trace(go.Bar(x=arms, y=[cap[a] for a in arms], marker_color=[ARM_COLORS.get(a) for a in arms],
                         showlegend=False, hovertemplate="%{x}: %{y:,} tokens<extra></extra>"), row=1, col=1)
    if not hit.empty:
        for wl in [w for w in WL_ORDER if w in set(hit.workload)]:
            d = hit[hit.workload == wl].copy()
            d["order"] = d.arm.map({a: i for i, a in enumerate(ARM_ORDER)}).fillna(99)
            d = d.sort_values("order")
            fig.add_trace(go.Bar(x=d.arm, y=d.hit * 100, name=wl, marker_color=WL_COLORS.get(wl),
                                 hovertemplate=f"{wl} %{{x}}: %{{y:.1f}}%<extra></extra>"), row=1, col=2)
        fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_layout(barmode="group", height=380)
    return fig


# ---------------------------------------------------------------- summaries
def kpis(df, slo):
    out, P, L = [], plab(slo["pct"]), lab(slo["pct"])
    b = df[df.arm == "baseline"]
    ag = b[(b.workload == "agent") & (b["mode"] == "conc")].sort_values("value")
    if not ag.empty:
        ok = ag[ag.meets]
        if ok.empty:
            out.append(("Max agent concurrency within SLO", "none", "no tested level met the SLO"))
        else:
            r = ok.iloc[-1]
            out.append(("Max agent concurrency within SLO", f"{int(r.value)}",
                        f"{r.out_tok_s:,.0f} output tok/s · TTFT {P} {r[f'ttft_{L}']:,.0f} ms"))
        pk = ag.loc[ag.out_tok_s.idxmax()]
        out.append(("Peak output throughput", f"{pk.out_tok_s:,.0f} tok/s", f"agent workload at c={int(pk.value)}"))
    rt = b[(b.workload == "agent") & (b["mode"] == "rate")].sort_values("value")
    if not rt.empty:
        ok = rt[rt.meets & (rt.req_s >= 0.9 * rt.value)]
        if ok.empty:
            out.append(("Max sustained QPS within SLO", "none", "lowest offered rate already missed the SLO"))
        else:
            r = ok.iloc[-1]
            out.append(("Max sustained QPS within SLO", f"{r.req_s:.1f} req/s", f"offered {r.value:g} req/s, Poisson"))
    st = b[b["mode"] == "p999"]
    if not st.empty:
        r = st.iloc[0]
        out.append(("TTFT p50 / p99 / p99.9", f"{r.ttft_p50:,.0f} / {r.ttft_p99:,.0f} / {r.ttft_p999:,.0f} ms",
                    f"{int(r.n_ttft):,} requests, agent workload, c={int(r.value)}"))
        out.append(("ITL p50 / p99 / p99.9", f"{r.itl_p50:.1f} / {r.itl_p99:.1f} / {r.itl_p999:.1f} ms",
                    f"{int(r.n_itl):,} inter-token gaps"))
    bu = b[b["mode"] == "burst"]
    if not bu.empty:
        r = bu.iloc[0]
        out.append(("Burst: failed requests", f"{int(r.failed)} of {int(r.num_prompts)}",
                    f"all sent at once · TTFT p99 {r.ttft_p99:,.0f} ms"))
    tot = df.num_prompts.sum()
    out.append(("Errors across all tests", f"{int(df.failed.sum())} of {int(tot):,}",
                f"{df.failed.sum() / tot:.3%} of requests" if tot else ""))
    return out


def ablation_table(df, slo):
    c = df[df["mode"] == "conc"]
    arms = [a for a in ARM_ORDER if a in set(c.arm)]
    if "baseline" not in arms or len(arms) < 2:
        return None
    L, rows = lab(slo["pct"]), []
    for wl in ["agent", "chat", "long"]:
        for a in arms[1:]:
            lv = common_levels(c, wl, ["baseline", a])
            if not lv:
                continue
            v = max(lv)
            b0 = c[(c.arm == "baseline") & (c.workload == wl) & (c.value == v)].iloc[0]
            x = c[(c.arm == a) & (c.workload == wl) & (c.value == v)].iloc[0]

            def delta(col, b0=b0, x=x):
                return (x[col] / b0[col] - 1) * 100 if b0[col] else np.nan
            rows.append({"setting": a, "workload": wl, "concurrency": int(v),
                         "output tok/s": f"{x.out_tok_s:,.0f} ({delta('out_tok_s'):+.0f}%)",
                         f"TTFT {plab(slo['pct'])} ms": f"{x[f'ttft_{L}']:,.0f} ({delta(f'ttft_{L}'):+.0f}%)",
                         f"ITL {plab(slo['pct'])} ms": f"{x[f'itl_{L}']:.1f} ({delta(f'itl_{L}'):+.0f}%)"})
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------- page
CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
main{max-width:1280px;margin:0 auto;padding:24px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 4px}
.sub{color:#59636e;font-size:14px;margin:0 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.card{background:#fff;border:1px solid #d1d9e0;border-radius:8px;padding:12px 14px}
.card .k{font-size:12px;color:#59636e;text-transform:uppercase;letter-spacing:.03em}
.card .v{font-size:22px;font-weight:600;margin:4px 0}
.card .n{font-size:12px;color:#59636e}
section{background:#fff;border:1px solid #d1d9e0;border-radius:8px;padding:12px 16px;margin-top:12px}
.why{color:#59636e;font-size:14px;margin:4px 0 8px}
.tbl{border-collapse:collapse;font-size:12px;width:100%}
.tbl th,.tbl td{border-bottom:1px solid #eaeef2;padding:4px 6px;text-align:right;white-space:nowrap}
.tbl th{background:#f6f8fa;position:sticky;top:0}
.scroll{overflow-x:auto;max-height:520px}
pre{background:#f6f8fa;padding:8px;font-size:12px;overflow-x:auto}
"""


def build(run_dir, slo):
    df, raw = load(run_dir)
    if df.empty:
        raise SystemExit("No completed test points yet.")
    L = lab(slo["pct"])
    df["meets"] = ((df[f"ttft_{L}"] <= slo["ttft"]) & (df[f"itl_{L}"] <= slo["itl"]) & (df.failed == 0)).fillna(False)
    df.to_csv(run_dir / "summary.csv", index=False)
    cap, hit = server_facts(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    P = plab(slo["pct"])

    sections = [
        ("Capacity: throughput vs latency (baseline)",
         f"Each point is one concurrency level. The usable capacity is the rightmost point left of the SLO line "
         f"(TTFT {P} ≤ {slo['ttft']:g} ms, ITL {P} ≤ {slo['itl']:g} ms).", fig_capacity(df, slo)),
        ("Latency percentiles vs concurrency (baseline)",
         "p50 is the typical request; p99 and p99.9 are the tail users notice. Log scales on both axes.",
         fig_percentiles(df)),
        ("QPS sweep: Poisson arrivals (baseline, agent workload)",
         "Where achieved load falls below the ideal line, the server is saturated and queueing delay grows.",
         fig_rate(df, slo)),
        ("Tail latency at the operating point (p99.9 run)",
         "Survival curve: the share of samples slower than each latency. p99.9 TTFT rests on the number of "
         "requests in this run; ITL percentiles use every inter-token gap.", fig_tail(raw)),
        ("KV-cache and FlashInfer settings: head-to-head",
         "Same workload and concurrency for every setting; only the server flag changes.", fig_arm_bars(df, slo)),
        ("KV-cache and FlashInfer settings: capacity curves",
         "A curve further up and to the left serves more tokens at lower latency.", fig_arm_curves(df, slo)),
        ("KV-cache capacity and prefix-cache hit rate",
         "Pool capacity comes from the server log; hit rate from /metrics at the end of each sweep.",
         fig_kv(cap, hit)),
        ("GPU telemetry", "Sampled once per second during each setting's run.", fig_gpu(run_dir)),
    ]

    parts, first = [], True
    for title, why, fig in sections:
        if fig is None:
            continue
        fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=50, b=50),
                          legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0))
        div = fig.to_html(full_html=False, include_plotlyjs=True if first else False,
                          config={"responsive": True, "displaylogo": False})
        first = False
        parts.append(f"<h2>{html.escape(title)}</h2><section><p class='why'>{html.escape(why)}</p>{div}</section>")

    cards = "".join(f"<div class='card'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(v)}</div>"
                    f"<div class='n'>{html.escape(n)}</div></div>" for k, v, n in kpis(df, slo))
    abl = ablation_table(df, slo)
    abl_html = ""
    if abl is not None:
        abl_html = ("<h2>Settings vs baseline</h2><section><p class='why'>At the highest concurrency every setting "
                    "ran; change vs baseline in brackets. Lower is better for latency.</p><div class='scroll'>"
                    + abl.to_html(index=False, classes="tbl", border=0) + "</div></section>")

    cols = ["arm", "workload", "mode", "value", "num_prompts", "failed", "req_s", "out_tok_s"] + \
           [f"{m}_{lab(p)}" for m in ("ttft", "itl", "e2e") for p in PCTS] + ["n_ttft", "meets"]
    table = df[cols].copy()
    for col in table.columns:
        if table[col].dtype.kind == "f":
            table[col] = table[col].round(1)
    v = meta.get("versions", {})
    runs = meta.get("runs", [{}])
    subtitle = (f"Qwen3-4B · {meta.get('gpu', 'GPU n/a')} · SGLang {v.get('sglang', '?')} · FlashInfer "
                f"{v.get('flashinfer', '?')} · torch {v.get('torch', '?')} (CUDA {v.get('cuda', '?')}) · "
                f"started {runs[0].get('started', '?')} · SLO at {P}: TTFT ≤ {slo['ttft']:g} ms, ITL ≤ {slo['itl']:g} ms")
    flags = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(meta.get('arm_labels', {}).get(k, ''))} "
                    f"<code>{html.escape(f)}</code></li>" for k, f in meta.get("arm_flags", {}).items())
    cmds = ""
    for arm_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if (arm_dir / "server.cmd").exists():
            cmds += f"<details><summary>{arm_dir.name} launch command</summary><pre>" \
                    f"{html.escape((arm_dir / 'server.cmd').read_text())}</pre></details>"

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Qwen3-4B serving benchmark</title>
<style>{CSS}</style></head><body><main>
<h1>Qwen3-4B serving benchmark: SGLang + FlashInfer</h1><p class="sub">{html.escape(subtitle)}</p>
<div class="cards">{cards}</div>
{abl_html}
{''.join(parts)}
<h2>All test points</h2><section><p class="why">Latencies in ms, computed from per-request data. n_ttft is the
number of requests behind each TTFT percentile; treat p99.9 with fewer than 1,000 requests as indicative.</p>
<div class="scroll">{table.to_html(index=False, classes="tbl", border=0, na_rep="–")}</div></section>
<h2>Method</h2><section><ul>{flags}</ul>
<p class="why">Workloads: agent = 16 shared 4K-token system prompts + 128-token question, 256 output tokens;
chat = 1K in / 256 out; long = 8K in / 256 out. Fixed output length, prefix cache flushed before every point,
load from sglang.bench_serving on the same host.</p>{cmds}</section>
</main></body></html>"""
    (run_dir / "dashboard.html").write_text(page)
    return run_dir / "dashboard.html"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--slo-ttft-ms", type=float, default=500)
    ap.add_argument("--slo-itl-ms", type=float, default=50)
    ap.add_argument("--slo-pct", type=float, default=99, choices=[50, 90, 99, 99.9])
    a = ap.parse_args()
    out = build(a.run_dir, dict(ttft=a.slo_ttft_ms, itl=a.slo_itl_ms, pct=a.slo_pct))
    print(out)


if __name__ == "__main__":
    main()
