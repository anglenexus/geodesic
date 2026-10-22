#!/usr/bin/env python3
"""
export_results.py - shrink Phase 3 and Phase 4 results into a small folder that can be committed.

    python export_results.py --phase3 ~/qwen-serve/results/<run-id>
    python export_results.py --phase4 ~/qwen-serve/results/fi_qwen3.5-27b
    python export_results.py --phase3 <run dir> --phase4 <fi dir>         # both into one folder

Writes geodesic/aees/serving/bench_results/<name>/ (name = the Phase 3 run id, or <model>_<gpu>_fi
for Phase 4 alone; --name overrides). Raw per-request data and kernel sources stay on the GPU host;
what lands here is enough to rebuild every chart and table:

  meta.json          model, GPU, versions, settings and their flags, test plan, export time, git commit
  summary.md         the results in plain language and Markdown tables: start here, and give it to an
                     LLM (with p3_run_log.md) when drafting a write-up
  p3_run_log.md      run_bench.py's readable log: every test point as it ran, skips, server start facts
  p3_points.csv      one row per test point: throughput and TTFT / ITL / end-to-end percentiles to p99.9
  p3_tails.json      latency survival curves of the long p99.9 run (quantiles, not raw samples)
  p3_quality.json    tool-call quality per setting (summary plus a compact row per prompt)
  p3_server.json     per setting: KV pool size, prefix-cache hit rate, launch command, key startup log lines
  p3_timeseries.csv  SGLang /metrics gauges per setting, averaged over 5 s
  p3_gpu.csv         GPU memory, utilization and power per setting, averaged over 5 s
  p4_sparse.json, p4_balance.json, p4_precision.json   kernel microbenchmarks (as written by fi_bench.py)
  p4_jit.json        JIT timings, module names and sizes, and the source diff between two variants
  p4_trace.json      live KV-index snapshot statistics plus one binned index map

Needs pandas and numpy (the serving venv has them). Typical size: well under 1 MB.
"""
import argparse
import datetime
import difflib
import json
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STARTUP_PATTERNS = (r"attention.?backend|max_total_num_tokens|mamba|cuda graph|quantiz|marlin|speculative|accept"
                    r"|load weight end")
KEEP_METRICS = {"sglang:token_usage", "sglang:cache_hit_rate", "sglang:num_running_reqs", "sglang:num_queue_reqs",
                "sglang:gen_throughput", "sglang:spec_accept_length"}
BIN_S = 5


def write_json(path, data):
    path.write_text(json.dumps(data, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o)))


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


# ---------------------------------------------------------------- phase 3
def export_phase3(run, out):
    import make_dashboard as md

    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    df, raw = md.load(run)
    if df.empty:
        print(f"  phase 3: no completed points in {run}")
        return meta, None
    df.round(3).to_csv(out / "p3_points.csv", index=False)
    if (run / "run_log.md").exists():
        shutil.copy(run / "run_log.md", out / "p3_run_log.md")

    # Survival curves of the p99.9 run: latency at the points where 10^-k of samples are slower.
    tails = {}
    for key, arrays in raw.items():
        if key[2] != "p999":
            continue
        cur = {}
        for name, arr in arrays.items():
            if arr.size:
                frac = np.unique(np.concatenate([np.logspace(0, -np.log10(arr.size), 60), [0.5, 0.1, 0.01, 0.001]]))
                cur[name] = dict(n=int(arr.size), slower_than=[r3(f) for f in frac[::-1]],
                                 ms=[r3(np.quantile(arr, 1 - f)) for f in frac[::-1]])
        tails["/".join(key)] = cur
    write_json(out / "p3_tails.json", tails)

    quality = {}
    for qf in sorted(run.glob("*/quality.json")):
        q = json.loads(qf.read_text())
        quality[qf.parent.name] = dict(summary=q["summary"], cases=[
            {k: c.get(k) for k in ("prompt", "expected_tool", "tool", "args", "valid_json", "tool_ok", "args_ok", "error")}
            for c in q["cases"]])
    write_json(out / "p3_quality.json", quality)

    cap, hit = md.server_facts(run)
    server = {}
    for arm_dir in sorted(p for p in run.iterdir() if p.is_dir()):
        s = dict(kv_pool_tokens=cap.get(arm_dir.name))
        if not hit.empty:
            s["cache_hit_rate"] = {r.workload: r3(r.hit) for r in hit[hit.arm == arm_dir.name].itertuples()}
        if (arm_dir / "server.cmd").exists():
            s["launch"] = (arm_dir / "server.cmd").read_text().strip()
        if (arm_dir / "server.log").exists():
            lines = (arm_dir / "server.log").read_text(errors="ignore").splitlines()
            seen, keep = set(), []
            for ln in lines:
                m = re.search(STARTUP_PATTERNS, ln, re.I)
                if m and m.group(0).lower() not in seen:
                    seen.add(m.group(0).lower())
                    keep.append(ln[:300])
            s["startup_log"] = keep
            m = re.search(r"Load weight end.*?mem usage=([\d.]+)\s*GB", "\n".join(lines))
            if m:
                s["weights_gb"] = float(m.group(1))
        start = arm_dir / "server_start.txt"
        if start.exists():
            text = start.read_text(errors="ignore")
            m = re.search(r"Ready after ~(\d+) s", text)
            if m:
                s["ready_seconds"] = int(m.group(1))
            elif arm_dir.name not in cap:  # did not come up: keep the error
                s["start_failure_tail"] = text.splitlines()[-40:]
        server[arm_dir.name] = s
    write_json(out / "p3_server.json", server)

    frames = []
    for f in sorted(run.glob("*/metrics_ts.csv")):
        if f.stat().st_size:
            d = pd.read_csv(f, names=["t", "metric", "value"])
            d = d[d.metric.isin(KEEP_METRICS) | d.metric.str.contains("mamba", case=False)]
            if d.empty:
                continue
            d["t_s"] = ((d.t - d.t.min()) // BIN_S * BIN_S).astype(int)
            w = d.pivot_table(index="t_s", columns="metric", values="value", aggfunc="mean").reset_index()
            w.columns = [c.replace("sglang:", "") for c in w.columns]
            w.insert(0, "arm", f.parent.name)
            frames.append(w)
    if frames:
        pd.concat(frames).round(4).to_csv(out / "p3_timeseries.csv", index=False)

    frames = []
    for f in sorted(run.glob("*/gpu.csv")):
        if f.stat().st_size:
            d = pd.read_csv(f, names=["ts", "util_pct", "mem_mib", "power_w"], skipinitialspace=True)
            d["ts"] = pd.to_datetime(d.ts, errors="coerce", format="mixed")
            d = d.dropna(subset=["ts"])
            for c in ("util_pct", "mem_mib", "power_w"):
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d["t_s"] = ((d.ts - d.ts.iloc[0]).dt.total_seconds() // BIN_S * BIN_S).astype(int)
            g = d.groupby("t_s")[["util_pct", "mem_mib", "power_w"]].mean().reset_index()
            g.insert(0, "arm", f.parent.name)
            frames.append(g)
    if frames:
        pd.concat(frames).round(1).to_csv(out / "p3_gpu.csv", index=False)
    print(f"  phase 3: {len(df)} points, {len(quality)} quality checks, {len(server)} settings")
    return meta, df


# ---------------------------------------------------------------- phase 4
def export_phase4(fi, out):
    import fi_bench
    env = None
    for name in ("sparse", "balance", "precision"):
        f = fi / f"{name}.json"
        if f.exists():
            shutil.copy(f, out / f"p4_{name}.json")
            env = env or json.loads(f.read_text()).get("env")
    jf = fi / "jit.json"
    if jf.exists():
        jt = json.loads(jf.read_text())
        env = env or jt.get("env")
        variants = []
        for v in jt["variants"]:
            variants.append(dict(key=v["key"], desc=v["desc"], first_s=r3(v["first_s"]), cached_s=r3(v["cached_s"]),
                                 error=v["error"], new_files=len(v["new_files"]),
                                 modules=[[pathlib.Path(p).name, sz] for p, sz in v["new_so"]]))
        by_key = {v["key"]: v for v in jt["variants"]}
        diff = ""
        a, b = by_key.get("decode_base"), by_key.get("decode_softcap")
        if a and b and a["sources"] and b["sources"]:
            def pick(srcs):
                return sorted(srcs.items(), key=lambda kv: ("config" not in kv[0], len(kv[1])))[0]
            (pa, ta), (pb, tb) = pick(a["sources"]), pick(b["sources"])
            diff = "\n".join(difflib.unified_diff(ta.splitlines(), tb.splitlines(), pathlib.Path(pa).name,
                                                  pathlib.Path(pb).name, lineterm="", n=2))[:12000]
        write_json(out / "p4_jit.json", dict(env=jt.get("env"), roots=jt.get("roots"), variants=variants,
                                             diff_base_vs_softcap=diff))
    snaps = fi_bench.load_traces(fi / "trace") if (fi / "trace").exists() else []
    if snaps:
        stats = [{k: (r3(v) if isinstance(v, float) else v) for k, v in s.items() if k not in ("indptr", "indices")}
                 for s in snaps]
        dec = [s for s in snaps if s["kind"] == "decode"] or snaps
        s = max(dec, key=lambda x: (x["batch"], x["refs"]))
        ip, ix = s["indptr"], s["indices"]
        rows, nb = min(s["batch"], 64), 200
        edges = np.linspace(int(ix.min()), int(ix.max()) + 1, nb + 1)
        vals, counts = np.unique(ix, return_counts=True)
        shared = set(vals[counts > 1].tolist())
        cells = []  # [request row, bin, private count, shared count] for non-empty bins
        for r in range(rows):
            seg = ix[ip[r]:ip[r + 1]]
            sh = np.fromiter((v in shared for v in seg), bool, seg.size) if shared else np.zeros(seg.size, bool)
            hp, _ = np.histogram(seg[~sh], bins=edges)
            hs, _ = np.histogram(seg[sh], bins=edges)
            cells += [[r, b, int(hp[b]), int(hs[b])] for b in np.nonzero(hp + hs)[0]]
        write_json(out / "p4_trace.json", dict(snapshots=stats, index_map=dict(
            file=s["file"], kind=s["kind"], batch=s["batch"], page_size=s["page_size"], refs=s["refs"],
            unique=s["unique"], share=r3(s["share"]), mean_run=r3(s["mean_run"]), rows=rows,
            bin_edges=[int(e) for e in edges], cells=cells)))
    print(f"  phase 4: {', '.join(p.name for p in sorted(out.glob('p4_*.json'))) or 'nothing found'}"
          f"{f' ({len(snaps)} trace snapshots)' if snaps else ''}")
    return env


# ---------------------------------------------------------------- summary
def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join("–" if (isinstance(v, float) and not np.isfinite(v)) else
                                     (f"{v:,.0f}" if isinstance(v, float) and abs(v) >= 100 else
                                      f"{v:.1f}" if isinstance(v, float) else str(v)) for v in row) + " |")
    return "\n".join(out)


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(new, old):
    try:
        return f"{new / old - 1:+.0%}"
    except (TypeError, ZeroDivisionError):
        return "–"


def point(df, arm, wl, mode, value):
    d = df[(df.arm.astype(str) == arm) & (df.workload == wl) & (df["mode"] == mode) & (df.value.astype(float) == float(value))]
    return d.iloc[0] if len(d) else None


def summary_md(meta, df, env, out):
    v = meta.get("versions", {})
    env = env or {}
    gpu = (meta.get("gpu") or env.get("gpu", "GPU")).split(",")[0].replace("NVIDIA ", "")
    model = meta.get("model_label", meta.get("model", env.get("model", "model")))
    run = (meta.get("runs") or [{}])[-1]
    L = [f"# {model} on {gpu}", "",
         f"Measured {run.get('started', '?')[:10]} with SGLang {v.get('sglang', '?')}, FlashInfer "
         f"{v.get('flashinfer', env.get('flashinfer', '?'))}, torch {v.get('torch', env.get('torch', '?'))} "
         f"(CUDA {v.get('cuda', env.get('cuda', '?'))}). Plan: {run.get('mode', 'full')}"
         + (f", capped at {run['budget_min']:g} min" if run.get("budget_min") else "") + ".",
         "", "How to read this: **TTFT** (time to first token) is how long a user waits before the answer starts; "
         "**ITL** (inter-token latency) is the gap between streamed tokens, so 1000 / ITL is one user's reading speed "
         "in tokens/s; **throughput** is output tokens/s summed over all users. **p50** is the typical request, "
         "**p99** the slowest 1 in 100. **Concurrency** is how many requests are in flight at once.", ""]
    server = load_json(out / "p3_server.json") or {}
    quality = load_json(out / "p3_quality.json") or {}
    if df is not None and not df.empty:
        labels = meta.get("arm_labels", {})
        arms = [a for a in dict.fromkeys(df.arm.astype(str))]
        L += ["## Settings tested", ""] + [f"- **{a}**: {labels.get(a, '')} (`{meta.get('arm_flags', {}).get(a, '')}`)"
                                           for a in arms] + [""]
        conc = df[df["mode"] == "conc"]
        rows = []
        for a in arms:
            d = conc[conc.arm.astype(str) == a]
            one = point(df, a, "agent", "conc", 1)
            ag = d[d.workload == "agent"]
            top = ag.loc[ag.value.astype(float).idxmax()] if len(ag) else None
            lg = d[d.workload == "long"]
            lg1 = lg.loc[lg.value.astype(float).idxmin()] if len(lg) else None
            sv, q = server.get(a, {}), quality.get(a, {}).get("summary", {})
            rows.append({
                "setting": a,
                "1 user: tok/s": 1000 / one.itl_p50 if one is not None and one.itl_p50 else np.nan,
                "1 user: TTFT ms": one.ttft_p50 if one is not None else np.nan,
                "peak agent throughput tok/s": top.out_tok_s if top is not None else np.nan,
                "at concurrency": int(top.value) if top is not None else "–",
                "TTFT p99 there ms": top.ttft_p99 if top is not None else np.nan,
                "ITL p99 there ms": top.itl_p99 if top is not None else np.nan,
                "8K-prompt TTFT p50 ms": lg1.ttft_p50 if lg1 is not None else np.nan,
                "KV cache tokens": f"{sv['kv_pool_tokens']:,}" if sv.get("kv_pool_tokens") else "–",
                "weights GB": sv.get("weights_gb", "–"),
                "server start s": sv.get("ready_seconds", "–"),
                "tool calls right": f"{q['args_ok']:.0%}" if q else "–"})
        L += ["## Each setting at a glance", "",
              "Agent workload (4K-token shared system prompt, 128-token question, 256-token answer) unless noted. "
              "1 user = concurrency 1. Peak = the highest concurrency tested for that setting.", "",
              md_table(pd.DataFrame(rows)), ""]
        base_rows = []
        for wl in ("agent", "chat", "long"):
            d = conc[conc.workload == wl]
            if d.arm.astype(str).nunique() < 2 or "baseline" not in set(d.arm.astype(str)):
                continue
            for c in sorted(set(d.value.astype(float))):
                b = point(df, "baseline", wl, "conc", c)
                for a in arms:
                    r = point(df, a, wl, "conc", c)
                    if a == "baseline" or r is None or b is None:
                        continue
                    base_rows.append({"workload": wl, "concurrency": int(c), "setting": a,
                                      "throughput": pct(r.out_tok_s, b.out_tok_s), "TTFT p50": pct(r.ttft_p50, b.ttft_p50),
                                      "TTFT p99": pct(r.ttft_p99, b.ttft_p99), "ITL p50": pct(r.itl_p50, b.itl_p50),
                                      "ITL p99": pct(r.itl_p99, b.itl_p99)})
        if base_rows:
            L += ["## Change vs baseline at each concurrency", "",
                  "Positive throughput is better; negative latency is better. Differences under ~5% are within run-to-run noise.",
                  "", md_table(pd.DataFrame(base_rows)), ""]
        for a in arms:
            d = conc[conc.arm.astype(str) == a]
            if d.empty:
                continue
            t = d[["workload", "value", "num_prompts", "out_tok_s", "req_s", "ttft_p50", "ttft_p99", "itl_p50", "itl_p99",
                   "e2e_p50", "failed"]].rename(columns={
                "value": "concurrency", "num_prompts": "requests", "out_tok_s": "tok/s", "req_s": "req/s",
                "ttft_p50": "TTFT p50 ms", "ttft_p99": "TTFT p99 ms", "itl_p50": "ITL p50 ms", "itl_p99": "ITL p99 ms",
                "e2e_p50": "request p50 ms"})
            L += [f"## {a}: throughput and latency by concurrency", "", md_table(t), ""]
        other = df[df["mode"].isin(["cold", "rate", "p999", "burst"])]
        if not other.empty:
            name = {"cold": "cold start (empty prefix cache)", "rate": "Poisson arrivals, req/s", "p999": "long run for p99.9",
                    "burst": "burst (all requests at once)"}
            t = other.assign(test=other["mode"].map(name))[["arm", "test", "value", "num_prompts", "failed", "out_tok_s",
                                                             "ttft_p50", "ttft_p99", "ttft_p999", "itl_p99"]].rename(columns={
                "arm": "setting", "value": "concurrency or rate", "num_prompts": "requests", "out_tok_s": "tok/s", "ttft_p50": "TTFT p50 ms",
                "ttft_p99": "TTFT p99 ms", "ttft_p999": "TTFT p99.9 ms", "itl_p99": "ITL p99 ms"})
            L += ["## Stress tests (agent workload)", "", "p99.9 from fewer than 1,000 requests is indicative only.", "",
                  md_table(t), ""]
    pr = load_json(out / "p4_precision.json")
    if pr:
        d = pd.DataFrame(pr["cases"])
        d = d[d["error"].isna()] if "error" in d else d
        tot = d.assign(t=d.ms * d.layers).groupby(["tokens", "precision"]).t.sum().unstack()
        t = pd.DataFrame({"tokens per forward pass": tot.index})
        for prec in [x for x in ("bf16", "fp8", "awq") if x in tot]:
            t[f"{prec} ms"] = tot[prec].round(2).values
            if prec != "bf16" and "bf16" in tot:
                t[f"{prec} speedup"] = (tot["bf16"] / tot[prec]).map("{:.2f}x".format).values
        L += ["## Phase 4: weight precision in the linear layers", "",
              "Time the model's weight matrices take per forward pass (all layers), measured with the kernels SGLang "
              "uses: " + "; ".join(f"{pr['precisions'].get(k, k)} = {v}" for k, v in pr["kernels"].items())
              + ". 1-64 tokens is decoding (reading weights dominates), 1024+ is prefill (math dominates).", "",
              md_table(t), ""]
    sp = load_json(out / "p4_sparse.json")
    if sp:
        d = pd.DataFrame(sp["cases"])
        d = d[d["error"].isna()] if "error" in d else d
        k = d[(d.page_size.isin([1, 16]))].pivot_table(index=["batch", "kv_len", "page_size", "layout"],
                                                        columns="kv_dtype", values="ms").reset_index()
        for c in [c for c in ("bf16", "fp8_e5m2", "fp8_e4m3") if c in k]:
            k[c] = (k[c] * 1000).round(1)
        L += ["## Phase 4: attention decode kernel time by KV-cache format (µs)", "",
              f"FlashInfer batch decode at {env.get('model', '')} attention shapes {env.get('shape', '')}; "
              f"peak memory bandwidth {env.get('peak_gbs', '?')} GB/s.", "", md_table(k), ""]
    jt = load_json(out / "p4_jit.json")
    if jt:
        t = pd.DataFrame(jt["variants"])[["key", "desc", "first_s", "cached_s"]].rename(columns={
            "key": "variant", "desc": "what changes", "first_s": "first call s (compile)", "cached_s": "later calls s"})
        L += ["## Phase 4: JIT compile cost per attention variant", "", md_table(t), ""]
    bl = load_json(out / "p4_balance.json")
    if bl:
        d = pd.DataFrame(bl["cases"])
        d = d[d["error"].isna()] if "error" in d else d
        if d.variant.nunique() > 1:
            on = d[d.variant.str.startswith("balanced")].pivot_table(index="dist", columns="batch", values="ms")
            off = d[~d.variant.str.startswith("balanced")].pivot_table(index="dist", columns="batch", values="ms")
            sp_ = (off / on).round(2).reset_index()
            sp_.columns = ["length mix"] + [f"{c} requests" for c in sp_.columns[1:]]
            L += ["## Phase 4: speedup from FlashInfer's split-KV load balancing", "",
                  "Decode kernel time without split-KV divided by time with it (above 1 = balancing helps).", "",
                  md_table(sp_), ""]
    (out / "summary.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase3", type=pathlib.Path, help="run_bench.py results folder")
    ap.add_argument("--phase4", type=pathlib.Path, help="fi_bench.py results folder (results/fi or results/fi_<model>)")
    ap.add_argument("--name", help="output folder name under bench_results/")
    ap.add_argument("--dest", type=pathlib.Path, default=HERE / "bench_results")
    a = ap.parse_args()
    if not (a.phase3 or a.phase4):
        ap.error("give --phase3 and/or --phase4")
    p3 = a.phase3.expanduser().resolve() if a.phase3 else None
    p4 = a.phase4.expanduser().resolve() if a.phase4 else None
    name = a.name
    if not name and p3:
        name = p3.name
    if not name and p4:
        e = next((json.loads(f.read_text()).get("env", {}) for f in sorted(p4.glob("*.json"))), {})
        gpu = re.sub(r"[^a-z0-9.]+", "-", e.get("gpu", "gpu").lower().replace("nvidia", "")).strip("-")
        name = f"{e.get('model', 'qwen3-4b')}_{gpu}_fi"
    out = a.dest / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {out}")
    meta, df = export_phase3(p3, out) if p3 else ({}, None)
    env = export_phase4(p4, out) if p4 else None
    if out.joinpath("meta.json").exists():  # keep what an earlier export of the other phase wrote
        old = json.loads(out.joinpath("meta.json").read_text())
        meta = {**old.get("phase3", {}), **meta} if p3 else old.get("phase3", {})
        env = env or old.get("phase4_env")
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    write_json(out / "meta.json", dict(phase3=meta, phase4_env=env, exported=datetime.datetime.now().isoformat(timespec="seconds"),
                                       git_commit=commit, source=dict(phase3=str(p3) if p3 else None, phase4=str(p4) if p4 else None)))
    if df is None and out.joinpath("p3_points.csv").exists():
        df = pd.read_csv(out / "p3_points.csv")
    summary_md(meta, df, env, out)
    total = sum(f.stat().st_size for f in out.iterdir())
    for f in sorted(out.iterdir()):
        print(f"  {f.name:20s} {f.stat().st_size / 1024:8.1f} KB")
    print(f"Total {total / 1024:.0f} KB. Commit it with: git add {out.relative_to(HERE.parent.parent) if out.is_relative_to(HERE.parent.parent) else out}")
    if total > 5 * 2**20:
        print("WARNING: over 5 MB; check which file grew (p4_trace.json or p3_timeseries.csv are the usual suspects).")


if __name__ == "__main__":
    main()
