#!/usr/bin/env python3
"""
fi_bench.py - Phase 4 FlashInfer deep dive: microbenchmarks plus one HTML report.

    python fi_bench.py sparse     # 2a: page size x memory layout x KV dtype       (~5 min)
    python fi_bench.py balance    # 2c: skewed batches, split-KV on vs off, plan() cost (~5 min)
    python fi_bench.py jit        # 2b: run in the COLD venv from fi_jit_setup.sh   (~10-20 min)
    python fi_bench.py report     # results/fi/fi_report.html (also reads live traces) (needs pandas, plotly)

Attention shapes default to Qwen3-4B: 32 query heads, 8 KV heads, head_dim 128.
Results land in ~/qwen-serve/results/fi/*.json (override with FI_OUT=...), inside the folder that
`./serve.sh dashboard` serves on port 8000. Every step is safe to rerun.
Live KV-index traces from the running server are captured by fi_trace.py
(see the plan doc, Phase 4 step 4) and land in results/fi/trace/.
"""
import argparse
import difflib
import html
import importlib.metadata
import itertools
import json
import os
import pathlib
import statistics
import sys
import time

OUT = pathlib.Path(os.environ.get("FI_OUT") or pathlib.Path(os.environ.get(
    "RESULTS_DIR", pathlib.Path(os.environ.get("WORK_DIR", "~/qwen-serve")).expanduser() / "results")) / "fi")
QH, KVH, HD, LAYERS = 32, 8, 128, 36  # Qwen3-4B attention
PEAK_GBS = {"A100-SXM4-80GB": 2039, "A100 80GB PCIe": 1935, "A100": 1555, "H100": 3350, "L40S": 864}


def log(msg):
    print(msg, flush=True)


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(data, indent=1, default=str))
    log(f"saved {path}")


def env_info():
    import torch
    import flashinfer
    name = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k.lower() in name.lower()), None)
    return dict(gpu=name, peak_gbs=peak, sm=".".join(map(str, torch.cuda.get_device_capability())),
                torch=torch.__version__, cuda=torch.version.cuda, flashinfer=flashinfer.__version__,
                time=time.strftime("%Y-%m-%d %H:%M:%S"))


def bench(fn, iters=30, reps=5):
    """Median kernel time in ms, measured with CUDA events."""
    import torch
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    return statistics.median(out)


def build_batch(lens, page_size, layout, kv_dtype, head_dim=HD, seed=0):
    """Paged KV cache in FlashInfer's NHD layout: (num_pages, 2, page_size, kv_heads, head_dim)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    pages = [max(1, -(-int(n) // page_size)) for n in lens]
    total = sum(pages)
    if layout == "contiguous":
        pool, ids = total, torch.arange(total, dtype=torch.int32)
    else:  # pages scattered at random over a pool 25% larger than needed, like a churned allocator
        pool = int(total * 1.25) + 1
        ids = torch.randperm(pool, generator=g)[:total].to(torch.int32)
    indptr = torch.tensor([0] + list(itertools.accumulate(pages)), dtype=torch.int32)
    last = torch.tensor([int(n) - (p - 1) * page_size for n, p in zip(lens, pages)], dtype=torch.int32)
    cache = torch.randn(pool, 2, page_size, KVH, head_dim, dtype=torch.bfloat16, device="cuda")
    if kv_dtype != torch.bfloat16:
        cache = cache.to(kv_dtype)
    return dict(indptr=indptr.cuda(), indices=ids.cuda(), last=last.cuda(), cache=cache, entries=total)


def decode_case(lens, page_size, layout, kv_dtype, ws, plan_kw=None):
    import torch
    import flashinfer
    b = build_batch(lens, page_size, layout, kv_dtype)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    q = torch.randn(len(lens), QH, HD, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    w.plan(b["indptr"], b["indices"], b["last"], QH, KVH, HD, page_size,
           q_data_type=torch.bfloat16, kv_data_type=kv_dtype, **(plan_kw or {}))
    plan_ms = (time.perf_counter() - t0) * 1000
    ms = bench(lambda: w.run(q, b["cache"]))
    kv_bytes = sum(int(n) for n in lens) * 2 * KVH * HD * b["cache"].element_size()
    return dict(ms=ms, gbs=kv_bytes / ms / 1e6, kv_mb=kv_bytes / 1e6, index_entries=b["entries"], plan_ms=plan_ms)


def free_gpu():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- 2a sparse
def cmd_sparse(a):
    import torch
    ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    rows = []
    for B, L in [(64, 4096), (16, 16384), (256, 1024)]:  # same total KV tokens, different shapes
        for ps in [1, 4, 16, 64]:
            for layout in ["contiguous", "scattered"]:
                for dt_name, dt in [("bf16", torch.bfloat16), ("fp8_e5m2", torch.float8_e5m2)]:
                    if dt_name != "bf16" and ps not in (1, 16):
                        continue
                    rec = dict(batch=B, kv_len=L, page_size=ps, layout=layout, kv_dtype=dt_name)
                    try:
                        rec.update(decode_case([L] * B, ps, layout, dt, ws))
                        log(f"B={B:<4} L={L:<6} page={ps:<3} {layout:10s} {dt_name:8s} "
                            f"{rec['ms']:7.3f} ms  {rec['gbs']:7.0f} GB/s")
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {e}"[:300]
                        log(f"B={B} L={L} page={ps} {layout} {dt_name}: {rec['error']}")
                    rows.append(rec)
                    free_gpu()
    save("sparse", dict(env=env_info(), cases=rows))


# ----------------------------------------------------------------------------- 2c balance
def lens_for(dist, B, mean, rng):
    import numpy as np
    T = B * mean
    if dist == "uniform":
        lens = np.full(B, mean, float)
    elif dist == "one_long":  # one long agent session among short ones
        long = min(32768, T // 2) if B > 1 else T
        lens = np.array([long] + [(T - long) / max(B - 1, 1)] * (B - 1), float)
    elif dist == "tiered":  # 10% of requests carry 80% of the tokens
        nl = max(1, B // 10)
        lens = np.array([0.8 * T / nl] * nl + [0.2 * T / max(B - nl, 1)] * (B - nl), float)
    else:  # heavy-tailed lognormal
        x = rng.lognormal(0.0, 1.0, B)
        lens = x / x.sum() * T
    return [max(16, int(round(v))) for v in lens]


def cmd_balance(a):
    import inspect
    import numpy as np
    import torch
    import flashinfer
    W = flashinfer.BatchDecodeWithPagedKVCacheWrapper
    ws = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    variants = {"balanced (split-KV on)": {}}
    if "disable_split_kv" in inspect.signature(W.plan).parameters:
        variants["no split-KV"] = {"disable_split_kv": True}
    else:
        log("WARNING: this FlashInfer has no disable_split_kv; only the balanced variant runs.")
    rng = np.random.default_rng(0)
    cases, examples = [], {}
    for B in [4, 16, 64, 128]:
        for dist in ["uniform", "one_long", "tiered", "lognormal"]:
            lens = lens_for(dist, B, 2048, rng)
            if B == 64:
                examples[dist] = sorted(lens, reverse=True)
            for vname, kw in variants.items():
                rec = dict(batch=B, dist=dist, variant=vname, max_len=max(lens),
                           mean_len=sum(lens) / B, skew=max(lens) / (sum(lens) / B))
                try:
                    rec.update(decode_case(lens, 1, "contiguous", torch.bfloat16, ws, kw))
                    log(f"B={B:<4} {dist:10s} {vname:24s} {rec['ms']:7.3f} ms  {rec['gbs']:6.0f} GB/s")
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                    log(f"B={B} {dist} {vname}: {rec['error']}")
                cases.append(rec)
                free_gpu()
    plan_rows = []
    for B in [8, 32, 128, 512]:
        free_gpu()
        lens = [2048] * B
        b = build_batch(lens, 1, "contiguous", torch.bfloat16)
        w = W(ws, "NHD")
        q = torch.randn(B, QH, HD, dtype=torch.bfloat16, device="cuda")
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            w.plan(b["indptr"], b["indices"], b["last"], QH, KVH, HD, 1,
                   q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            times.append((time.perf_counter() - t0) * 1000)
        run_ms = bench(lambda: w.run(q, b["cache"]))
        med = statistics.median(times)
        plan_rows.append(dict(batch=B, plan_ms=med, plan_per_layer_ms=med / LAYERS, run_ms=run_ms))
        log(f"plan() B={B:<4} {med:6.3f} ms total = {med / LAYERS * 1000:6.1f} us per layer; run {run_ms:.3f} ms")
    save("balance", dict(env=env_info(), cases=cases, plan=plan_rows, examples_b64=examples))


# ----------------------------------------------------------------------------- 2b jit
def jit_roots():
    from flashinfer.jit import env as je
    home = pathlib.Path.home().resolve()
    cands = {pathlib.Path(getattr(je, n)) for n in dir(je)
             if n.isupper() and n.endswith("DIR") and isinstance(getattr(je, n), (str, pathlib.PurePath))}
    if os.environ.get("FLASHINFER_WORKSPACE_BASE"):
        cands.add(pathlib.Path(os.environ["FLASHINFER_WORKSPACE_BASE"]))
    roots = []
    for r in sorted((c.resolve() for c in cands), key=lambda p: len(str(p))):
        if r in (home, pathlib.Path("/")) or any(r == k or k in r.parents for k in roots):
            continue
        roots.append(r)
    return roots


def snapshot(roots):
    snap = {}
    for r in roots:
        if r.exists():
            for p in r.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    snap[str(p)] = (st.st_size, st.st_mtime)
    return snap


def jit_variants():
    import torch
    import flashinfer
    bf = torch.bfloat16

    def dec(page=16, kv=bf, hd=HD, tc=False, **pk):
        def f(ws):
            b = build_batch([512] * 4, page, "contiguous", kv, head_dim=hd)
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
            w.plan(b["indptr"], b["indices"], b["last"], QH, KVH, hd, page, q_data_type=bf, kv_data_type=kv, **pk)
            w.run(torch.randn(4, QH, hd, dtype=bf, device="cuda"), b["cache"])
        return f

    def prefill(ws):
        b = build_batch([512] * 4, 16, "contiguous", bf)
        qo = torch.tensor([0, 512, 1024, 1536, 2048], dtype=torch.int32, device="cuda")
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        w.plan(qo, b["indptr"], b["indices"], b["last"], QH, KVH, HD, 16, causal=True, q_data_type=bf, kv_data_type=bf)
        w.run(torch.randn(2048, QH, HD, dtype=bf, device="cuda"), b["cache"])

    def single_decode(ws):
        k = torch.randn(2048, KVH, HD, dtype=bf, device="cuda")
        flashinfer.single_decode_with_kv_cache(torch.randn(QH, HD, dtype=bf, device="cuda"), k, torch.randn_like(k))

    def single_prefill(ws):
        k = torch.randn(512, KVH, HD, dtype=bf, device="cuda")
        flashinfer.single_prefill_with_kv_cache(torch.randn(512, QH, HD, dtype=bf, device="cuda"), k,
                                                torch.randn_like(k), causal=True)

    return [
        ("decode_base", "Batch decode, BF16, head_dim 128 (Qwen3-4B)", dec()),
        ("decode_softcap", "+ logits soft-cap", dec(logits_soft_cap=30.0)),
        ("decode_rope", "+ fused RoPE (ROPE_LLAMA)", dec(pos_encoding_mode="ROPE_LLAMA")),
        ("decode_window", "+ sliding window (256)", dec(window_left=256)),
        ("decode_fp8kv", "FP8 e5m2 KV cache", dec(kv=torch.float8_e5m2)),
        ("decode_hd64", "head_dim 64", dec(hd=64)),
        ("decode_tensorcores", "Decode on tensor cores (prefill kernel)", dec(tc=True)),
        ("prefill_causal", "Batch prefill, causal, paged KV", prefill),
        ("single_decode", "Single-request decode", single_decode),
        ("single_prefill", "Single-request prefill, causal", single_prefill),
    ]


def cmd_jit(a):
    import torch
    for pkg in ("flashinfer-jit-cache", "flashinfer-cubin"):
        try:
            v = importlib.metadata.version(pkg)
            log(f"WARNING: {pkg} {v} is installed, so kernels come prebuilt and nothing compiles. "
                "Run this in the cold venv from fi_jit_setup.sh.")
        except importlib.metadata.PackageNotFoundError:
            pass
    roots = jit_roots()
    log("Watching JIT directories:\n  " + "\n  ".join(map(str, roots)))
    ws = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    rows = []
    for key, desc, fn in jit_variants():
        before = snapshot(roots)
        err = None
        t0 = time.perf_counter()
        try:
            fn(ws)
            torch.cuda.synchronize()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:400]
        first = time.perf_counter() - t0
        second = None
        if err is None:
            t1 = time.perf_counter()
            fn(ws)
            torch.cuda.synchronize()
            second = time.perf_counter() - t1
        after = snapshot(roots)
        new = sorted(p for p in after if before.get(p) != after[p])
        texts = {}
        for p in new:
            if pathlib.Path(p).suffix in (".cu", ".cuh", ".inc", ".h", ".json") and after[p][0] < 40_000 and len(texts) < 6:
                try:
                    texts[p] = pathlib.Path(p).read_text(errors="ignore")
                except Exception:
                    pass
        rec = dict(key=key, desc=desc, first_s=first, cached_s=second, error=err,
                   new_files=[(p, after[p][0]) for p in new],
                   new_so=[(p, after[p][0]) for p in new if p.endswith(".so")], sources=texts)
        rows.append(rec)
        log(f"{key:20s} first call {first:7.1f} s  cached {second if second is not None else float('nan'):6.3f} s  "
            f"new files {len(new):3d} (.so {len(rec['new_so'])})" + (f"  ERROR {err}" if err else ""))
    save("jit", dict(env=env_info(), roots=[str(r) for r in roots], variants=rows))


# ----------------------------------------------------------------------------- live traces
def load_traces(d):
    import numpy as np
    snaps = []
    for f in sorted(d.glob("*.npz")):
        try:
            z = np.load(f)
        except Exception:
            continue
        keys = z.files
        ip = next((z[k] for k in keys if k.endswith("indptr") and "qo" not in k), None)
        ix = next((z[k] for k in keys if k.endswith("indices")), None)
        if ip is None or ix is None or ip.size < 2:
            continue
        ip = ip.astype(np.int64)
        ix = ix.astype(np.int64)[: int(ip[-1])]
        B, refs = ip.size - 1, int(ip[-1])
        if refs == 0:
            continue
        uniq = np.unique(ix).size
        d = np.diff(ix)
        mask = np.ones(d.size, bool)
        cut = ip[1:-1] - 1
        mask[cut[(cut >= 0) & (cut < d.size)]] = False
        breaks = int((d[mask] != 1).sum())
        snaps.append(dict(file=f.name, kind=f.name.split("_")[0], t=float(z["t"]) if "t" in keys else 0.0,
                          page_size=int(z["page_size"]) if "page_size" in keys else 1, batch=B, refs=refs,
                          unique=uniq, share=refs / uniq, contig=float((d[mask] == 1).mean()) if mask.any() else 1.0,
                          mean_run=refs / (breaks + B), indptr=ip, indices=ix))
    return snaps


# ----------------------------------------------------------------------------- report
CSS = """body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
main{max-width:1200px;margin:0 auto;padding:24px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 4px}
.sub{color:#59636e;font-size:14px}section{background:#fff;border:1px solid #d1d9e0;border-radius:8px;padding:12px 16px;margin-top:12px}
.why{color:#59636e;font-size:14px;margin:4px 0 8px}.tbl{border-collapse:collapse;font-size:12px}
.tbl th,.tbl td{border-bottom:1px solid #eaeef2;padding:4px 8px;text-align:right}.tbl th{background:#f6f8fa}
pre{background:#f6f8fa;padding:8px;font-size:11px;overflow:auto;max-height:420px}.miss{color:#9a6700}"""
C = {"contiguous": "#1f6feb", "scattered": "#d1495b", "bf16": "#1f6feb", "fp8_e5m2": "#2a9d8f",
     "balanced (split-KV on)": "#1f6feb", "no split-KV": "#d1495b"}


def cmd_report(a):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    parts, first = [], [True]

    def add(title, why, fig=None, raw=""):
        div = ""
        if fig is not None:
            fig.update_layout(template="plotly_white", margin=dict(l=60, r=20, t=50, b=50),
                              legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0))
            div = fig.to_html(full_html=False, include_plotlyjs=first[0], config={"responsive": True, "displaylogo": False})
            first[0] = False
        parts.append(f"<h2>{html.escape(title)}</h2><section><p class='why'>{html.escape(why)}</p>{div}{raw}</section>")

    def missing(title, how):
        parts.append(f"<h2>{html.escape(title)}</h2><section><p class='why miss'>No data yet: {html.escape(how)}</p></section>")

    def load(name):
        p = OUT / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    env = None
    # ---- 2a sparse
    sp = load("sparse")
    if sp:
        env = sp["env"]
        df = pd.DataFrame(sp["cases"])
        df = df[df.get("error").isna()] if "error" in df else df
        bf = df[df.kv_dtype == "bf16"]
        shapes = sorted({(r.batch, r.kv_len) for r in bf.itertuples()})
        fig = make_subplots(rows=1, cols=len(shapes), subplot_titles=[f"{b} requests × {n:,} tokens" for b, n in shapes])
        for j, (b, n) in enumerate(shapes, 1):
            for lay in ["contiguous", "scattered"]:
                d = bf[(bf.batch == b) & (bf.kv_len == n) & (bf.layout == lay)].sort_values("page_size")
                fig.add_trace(go.Scatter(x=d.page_size, y=d.gbs, mode="lines+markers", name=lay, legendgroup=lay,
                                         showlegend=j == 1, line=dict(color=C[lay])), row=1, col=j)
            if env.get("peak_gbs"):
                fig.add_trace(go.Scatter(x=[1, 64], y=[env["peak_gbs"]] * 2, mode="lines", name="HBM peak",
                                         legendgroup="peak", showlegend=j == 1,
                                         line=dict(color="#adb5bd", dash="dash")), row=1, col=j)
        fig.update_xaxes(type="log", title_text="Page size (tokens, log)", tickvals=[1, 4, 16, 64])
        fig.update_yaxes(title_text="Achieved KV bandwidth (GB/s)", rangemode="tozero", col=1)
        fig.update_layout(height=420)
        add("2a · Sparse paged KV: does scattering pages cost bandwidth?",
            "Batch decode reading the whole KV cache, at Qwen3-4B shapes. Contiguous = pages laid out in order; "
            "scattered = pages placed at random across the pool, like a long-running allocator. Close lines mean "
            "FlashInfer's block-sparse indexing costs little.", fig)
        piv = bf.pivot_table(index=["batch", "kv_len", "page_size"], columns="layout", values="gbs").reset_index()
        if {"contiguous", "scattered"} <= set(piv.columns):
            piv["scattered / contiguous"] = (piv.scattered / piv.contiguous).map("{:.0%}".format)
            piv[["contiguous", "scattered"]] = piv[["contiguous", "scattered"]].round(0)
            parts.append("<section><p class='why'>Bandwidth (GB/s) by layout</p>"
                         + piv.to_html(index=False, classes="tbl", border=0) + "</section>")
        f8 = df[df.page_size.isin([1, 16])]
        if (f8.kv_dtype == "fp8_e5m2").any():
            f8 = f8.assign(case=f8.apply(lambda r: f"{r.batch}×{r.kv_len} p{r.page_size} {r.layout[:4]}", axis=1))
            fig = go.Figure()
            for dt in ["bf16", "fp8_e5m2"]:
                d = f8[f8.kv_dtype == dt]
                fig.add_trace(go.Bar(x=d.case, y=d.ms * 1000, name=dt, marker_color=C[dt]))
            fig.update_layout(barmode="group", height=400, yaxis_title="Kernel time (µs)")
            add("2a · FP8 vs BF16 KV: kernel time",
                "FP8 halves the bytes read; on the A100 each value is converted back to 16-bit in the kernel. "
                "If FP8 bars are not clearly shorter, conversion cost eats the bandwidth saving, matching Phase 3.", fig)
        ent = bf[bf.layout == "contiguous"].groupby("page_size").index_entries.max().reset_index()
        fig = go.Figure(go.Bar(x=ent.page_size.astype(str), y=ent.index_entries, marker_color="#1f6feb"))
        fig.update_layout(height=320, xaxis_title="Page size (tokens)", yaxis_title="Index entries per batch", yaxis_type="log")
        add("2a · Bookkeeping: index entries per batch", "Larger pages shrink the index arrays the scheduler builds and "
            "FlashInfer reads every step; this is the main thing page size 16 changes.", fig)
    else:
        missing("2a · Sparse paged KV", "run `python fi_bench.py sparse` on the GPU host.")

    # ---- live traces
    snaps = load_traces(OUT / "trace") if (OUT / "trace").exists() else []
    if snaps:
        dec = [s for s in snaps if s["kind"] == "decode"] or snaps
        s = max(dec, key=lambda x: (x["batch"], x["refs"]))
        ip, ix = s["indptr"], s["indices"]
        rows = min(s["batch"], 64)
        lo, hi = int(ix.min()), int(ix.max()) + 1
        nb = 300
        edges = np.linspace(lo, hi, nb + 1)
        vals, counts = np.unique(ix, return_counts=True)
        shared = set(vals[counts > 1].tolist())
        zp, zs = np.full((rows, nb), np.nan), np.full((rows, nb), np.nan)
        for r in range(rows):
            seg = ix[ip[r]:ip[r + 1]]
            sh = np.fromiter((v in shared for v in seg), bool, seg.size) if shared else np.zeros(seg.size, bool)
            for z, m in ((zp, ~sh), (zs, sh)):
                h, _ = np.histogram(seg[m], bins=edges)
                z[r] = np.where(h > 0, h, np.nan)
        x = (edges[:-1] + edges[1:]) / 2
        def layer(z, color, label):  # solid colour wherever a request uses a bin; counts on hover
            return go.Heatmap(z=np.where(np.isnan(z), np.nan, 1.0), x=x, text=np.nan_to_num(z).astype(int),
                              colorscale=[[0, color], [1, color]], zmin=0, zmax=1, showscale=False, name=label,
                              hovertemplate="request %{y}<br>slots ~%{x:.0f}: %{text} " + label + "<extra></extra>")
        fig = go.Figure([layer(zp, "#8ab4f8", "private"), layer(zs, "#d1495b", "shared")])
        fig.update_layout(height=max(300, 10 * rows + 120), xaxis_title=f"KV pool slot (page size {s['page_size']}), binned",
                          yaxis_title="Request in batch", yaxis_autorange="reversed")
        add(f"2a · Live KV index map ({s['kind']} batch of {s['batch']} requests)",
            f"Captured from the running server ({s['file']}). Each row is a request; blue = pages only it uses, "
            f"red = pages shared with other requests (prefix cache). {s['refs']:,} references to {s['unique']:,} "
            f"distinct pages, sharing ratio {s['share']:.1f}×, mean contiguous run {s['mean_run']:.0f} pages.", fig)
        tdf = pd.DataFrame([{k: v for k, v in x.items() if k not in ("indptr", "indices")} for x in snaps])
        tdf["sec"] = tdf.t - tdf.t.min()
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                            subplot_titles=("Requests per batch", "Sharing ratio (references / distinct pages)",
                                            "Mean contiguous run (pages, log)"))
        for kind, col in (("decode", "#1f6feb"), ("prefill", "#e9a23b")):
            d = tdf[tdf.kind == kind]
            for i, m in enumerate(["batch", "share", "mean_run"], 1):
                fig.add_trace(go.Scatter(x=d.sec, y=d[m], mode="markers", name=kind, legendgroup=kind,
                                         showlegend=i == 1, marker=dict(color=col, size=5)), row=i, col=1)
        fig.update_yaxes(type="log", row=3, col=1)
        fig.update_xaxes(title_text="Seconds since first capture", row=3, col=1)
        fig.update_layout(height=600)
        add("2a · KV layout over time", f"{len(snaps)} sampled plan() calls. Falling contiguous runs mean the "
            "allocator is fragmenting the pool; FlashInfer's indexing makes that harmless if 2a's bandwidth lines match.", fig)
    else:
        missing("2a · Live KV index map", "run `./run_phase4.sh trace`, which restarts the server with the fi_trace.py hook and sends traffic.")

    # ---- 2c balance
    bl = load("balance")
    if bl:
        env = env or bl["env"]
        df = pd.DataFrame(bl["cases"])
        df = df[df.get("error").isna()] if "error" in df else df
        Bs = sorted(df.batch.unique())
        fig = make_subplots(rows=1, cols=len(Bs), subplot_titles=[f"{b} requests" for b in Bs])
        for j, b in enumerate(Bs, 1):
            for v in df.variant.unique():
                d = df[(df.batch == b) & (df.variant == v)]
                fig.add_trace(go.Bar(x=d.dist, y=d.ms * 1000, name=v, legendgroup=v, showlegend=j == 1,
                                     marker_color=C.get(v, "#6c757d")), row=1, col=j)
        fig.update_layout(barmode="group", height=420)
        fig.update_yaxes(title_text="Decode kernel time (µs)", col=1)
        add("2c · Load balancing: skewed batches with split-KV on vs off",
            "Same total tokens per batch (2,048 per request on average), different length mixes. Without split-KV, "
            "a long request's work cannot be spread over idle SMs, so the batch waits for it.", fig)
        if df.variant.nunique() > 1:
            on = df[df.variant.str.startswith("balanced")].pivot_table(index="dist", columns="batch", values="ms")
            off = df[~df.variant.str.startswith("balanced")].pivot_table(index="dist", columns="batch", values="ms")
            sp_ = (off / on).round(2)
            fig = go.Figure(go.Heatmap(z=sp_.values, x=[f"{c} req" for c in sp_.columns], y=sp_.index,
                                       colorscale="Blues", text=sp_.values, texttemplate="%{text:.2f}×",
                                       hovertemplate="%{y}, %{x}: %{z:.2f}× faster with split-KV<extra></extra>"))
            fig.update_layout(height=320)
            add("2c · Speedup from split-KV", "Time without split-KV ÷ time with it. Above 1× means load balancing helps.", fig)
        ex = bl.get("examples_b64", {})
        if ex:
            fig = make_subplots(rows=1, cols=len(ex), subplot_titles=list(ex))
            for j, (k, v) in enumerate(ex.items(), 1):
                fig.add_trace(go.Bar(y=v, marker_color="#1f6feb", showlegend=False), row=1, col=j)
            fig.update_yaxes(title_text="KV tokens", col=1)
            fig.update_xaxes(title_text="requests, longest first")
            fig.update_layout(height=320)
            add("2c · What the 64-request batches look like", "Request lengths in each distribution.", fig)
        pl = pd.DataFrame(bl["plan"])
        fig = go.Figure([go.Bar(x=pl.batch.astype(str), y=pl.plan_ms * 1000 / LAYERS, name="plan() per layer (µs)",
                                marker_color="#e9a23b"),
                         go.Bar(x=pl.batch.astype(str), y=pl.run_ms * 1000, name="attention kernel per layer (µs)",
                                marker_color="#1f6feb")])
        fig.update_layout(barmode="group", height=360, xaxis_title="Requests in batch (2,048 tokens each)",
                          yaxis_title="µs per layer", yaxis_type="log")
        add("2c · Scheduling overhead: plan() vs the kernel",
            f"plan() runs once per step on the CPU and is reused by all {LAYERS} layers, so its cost is shown per layer.", fig)
    else:
        missing("2c · Load balancing", "run `python fi_bench.py balance` on the GPU host.")

    # ---- 2b jit
    jt = load("jit")
    if jt:
        env = env or jt["env"]
        df = pd.DataFrame(jt["variants"])
        fig = go.Figure([go.Bar(x=df.key, y=df.first_s, name="first call (compile + run)", marker_color="#d1495b",
                                text=df.desc, hovertemplate="%{text}<br>%{y:.1f} s<extra></extra>"),
                         go.Bar(x=df.key, y=df.cached_s, name="second call (cached)", marker_color="#1f6feb")])
        fig.update_layout(barmode="group", height=420, yaxis_title="Seconds (log)", yaxis_type="log")
        add("2b · JIT: what each new attention variant costs the first time",
            "Each bar is one kernel specialization. The first call instantiates the template, compiles it with nvcc "
            "and caches the module; later calls reuse it. Prebuilt wheels skip the red bars entirely.", fig)
        tab = pd.DataFrame([dict(variant=r["key"], description=r["desc"], first_s=round(r["first_s"], 1),
                                 cached_ms=round((r["cached_s"] or 0) * 1000, 1), new_files=len(r["new_files"]),
                                 modules=", ".join(pathlib.Path(p).name for p, _ in r["new_so"]) or "–",
                                 so_MB=round(sum(sz for _, sz in r["new_so"]) / 1e6, 1), error=r["error"] or "")
                            for r in jt["variants"]])
        parts.append("<section><p class='why'>Modules built per variant. Watched: "
                     + html.escape(", ".join(jt["roots"])) + "</p>" + tab.to_html(index=False, classes="tbl", border=0) + "</section>")
        base = next((r for r in jt["variants"] if r["key"] == "decode_base"), None)
        other = next((r for r in jt["variants"] if r["key"] == "decode_softcap"), None)
        if base and other and base["sources"] and other["sources"]:
            def pick(srcs):
                return sorted(srcs.items(), key=lambda kv: ("config" not in kv[0], len(kv[1])))[0]
            (pa, ta), (pb, tb) = pick(base["sources"]), pick(other["sources"])
            diff = "\n".join(difflib.unified_diff(ta.splitlines(), tb.splitlines(), pathlib.Path(pa).name,
                                                  pathlib.Path(pb).name, lineterm="", n=2))
            add("2b · What changes between two variants", "Generated source for the base decode kernel vs the soft-cap "
                "variant: only template parameters differ; the kernel body is the same template.",
                raw=f"<pre>{html.escape(diff[:12000] or 'No textual difference found in captured sources.')}</pre>")
    else:
        missing("2b · JIT compilation", "run fi_jit_setup.sh, then `python fi_bench.py jit` in the cold venv.")

    head = ""
    if env:
        head = (f"{env['gpu']} (sm_{env['sm'].replace('.', '')}) · FlashInfer {env['flashinfer']} · torch {env['torch']} "
                f"(CUDA {env['cuda']}) · Qwen3-4B attention shapes: {QH} query heads, {KVH} KV heads, head_dim {HD}")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>FlashInfer deep dive</title><style>{CSS}</style></head><body><main>"
            f"<h1>FlashInfer deep dive: sparse KV, JIT, load balancing</h1><p class='sub'>{html.escape(head)}</p>"
            + "".join(parts) + "</main></body></html>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fi_report.html").write_text(page)
    log(f"report: {OUT / 'fi_report.html'}\nWith ./serve.sh dashboard running and the tunnel open: "
        f"http://127.0.0.1:{os.environ.get('DASH_PORT', '8000')}/{OUT.resolve().name}/fi_report.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sparse", "balance", "jit", "report"])
    a = ap.parse_args()
    {"sparse": cmd_sparse, "balance": cmd_balance, "jit": cmd_jit, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
