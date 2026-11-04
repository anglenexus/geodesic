"""Figures for the Kimi K3 serving blog. Data: the run's exported CSVs (data/) and the server log facts."""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
PLAIN = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")

D = pathlib.Path(__file__).resolve().parent / "data"
OUT = pathlib.Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)

INK, MUTE, GRID, BG = "#1d2430", "#6b7280", "#e6e8ec", "#ffffff"
K3, K3L = "#0e7c86", "#7fc8cd"          # Kimi K3 accent (teal)
AMB, COR, IND, SLA, OLV = "#e39b2d", "#d1495b", "#4b5dbf", "#9aa3af", "#6a994e"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11, "text.color": INK,
    "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 14, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    "legend.frameon": False, "text.parse_math": False,
})


def rows(name):
    return list(csv.DictReader(open(D / name)))


def save(fig, name, note=None):
    if note:
        fig.text(0.0, -0.02, note, fontsize=8.5, color=MUTE, ha="left", va="top", transform=fig.transFigure)
    fig.savefig(OUT / f"{name}.png", facecolor=BG)
    plt.close(fig)
    print(name)


SRC = "Source: our run on one 8× B300 node (Modal), SGLang 0.5.23 + FlashInfer 0.7.2, Kimi K3 MXFP4, TP8 + DCP8."

# ------------------------------------------------------------------ F1 agentic benchmarks (model card)
def f1():
    benches = ["Terminal-Bench 2.1", "BrowseComp", "MCPMark-Verified", "OSWorld-Verified", "Toolathlon-Verified",
               "MCP-Atlas", "DeepSWE", "SWE-Marathon"]
    models = [("Kimi K3 (open)", K3, [88.3, 91.2, 94.5, 84.8, 76.5, 84.2, 67.5, 42.0]),
              ("Claude Fable 5", IND, [88.0, 88.0, 87.4, 85.0, 77.9, 84.7, 70.0, 35.0]),
              ("GPT-5.6 Sol", AMB, [88.8, 90.4, 92.9, 83.0, 74.9, 83.6, 73.0, 39.0]),
              ("Claude Opus 4.8", SLA, [84.6, 84.3, 76.4, 83.4, 76.2, 83.6, 59.0, 40.0]),
              ("GLM-5.2 (open)", OLV, [82.7, np.nan, np.nan, np.nan, 59.9, 82.6, 46.2, 13.0])]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(benches)); w = 0.16
    for i, (name, c, v) in enumerate(models):
        ax.bar(x + (i - 2) * w, v, w * 0.92, label=name, color=c, zorder=3)
        if i == 0:
            for xi, vi in zip(x, v):
                ax.text(xi - 2 * w, vi + 1.2, f"{vi:g}", ha="center", fontsize=8.5, color=K3, fontweight="bold")
    ax.set_xticks(x, benches, rotation=0, fontsize=9.5)
    for t in ax.get_xticklabels():
        t.set_text(t.get_text().replace(" ", "\n", 1))
    ax.set_xticklabels([b.replace("-", "-\n", 1) if len(b) > 14 else b for b in benches], fontsize=9.2)
    ax.set_ylim(0, 105); ax.set_ylabel("score")
    ax.set_title("Kimi K3 on agentic benchmarks: frontier-level, and open", pad=34)
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=9.5)
    ax.grid(axis="x", visible=False)
    save(fig, "f01_agentic_benchmarks",
         "Source: moonshotai/Kimi-K3 model card (Kimi K3 at max reasoning effort; competitor scores as reported there). "
         "Blank = not reported. Artificial Analysis Intelligence Index: Kimi K3 = 57, the leading open-weights model at release.")

# ------------------------------------------------------------------ F2 model anatomy + parameters
def f2():
    fig = plt.figure(figsize=(12, 5.6))
    ax = fig.add_axes([0.0, 0.0, 0.62, 1.0]); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(0, 97, "Kimi K3: 93 layers, a hybrid of linear and latent attention", fontsize=14, fontweight="bold", va="top")
    # layer strip
    x0, y0, cw, ch = 2, 72, 1.02, 10
    for i in range(93):
        is_mla = (i + 1) % 4 == 0 or i == 92
        is_dense = i == 0
        c = AMB if is_mla else K3L
        ax.add_patch(Rectangle((x0 + i * cw, y0), cw * 0.86, ch, color=c, lw=0))
        if is_dense:
            ax.add_patch(Rectangle((x0 + i * cw, y0 - 2.2), cw * 0.86, 1.6, color=INK, lw=0))
    ax.text(x0, y0 - 5.5, "layer 1 (dense FFN)", fontsize=8.5, color=MUTE)
    ax.text(x0 + 93 * cw, y0 - 5.5, "layer 93", fontsize=8.5, color=MUTE, ha="right")
    ax.add_patch(Rectangle((2, 56), 2.5, 3, color=K3L)); ax.text(5.5, 57.5, "69 × KDA (Kimi Delta Attention, linear): fixed recurrent state per request", fontsize=9.5, va="center")
    ax.add_patch(Rectangle((2, 51), 2.5, 3, color=AMB)); ax.text(5.5, 52.5, "24 × Gated MLA: latent KV cache, 512 + 64 values per token per layer", fontsize=9.5, va="center")
    # one layer block
    def box(x, y, w, h, title, sub, fc, ec=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec or fc, lw=1.2))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.3, color=INK)
    ax.text(2, 42, "Every layer = attention (KDA or MLA) + MoE feed-forward", fontsize=10.5, fontweight="bold")
    box(2, 18, 22, 18, "Attention", "KDA (69) or\nGated MLA (24)\nhidden 7168, 96 heads", "#eef6f7", K3L)
    ax.annotate("", xy=(28, 27), xytext=(24.8, 27), arrowprops=dict(arrowstyle="->", color=MUTE))
    box(28.5, 18, 22, 18, "Router", "picks 16 of 896\nrouted experts", "#f4f4f6", "#d5d8de")
    ax.annotate("", xy=(54.5, 27), xytext=(51.3, 27), arrowprops=dict(arrowstyle="->", color=MUTE))
    for j in range(3, 0, -1):
        ax.add_patch(FancyBboxPatch((55 + j * 1.6, 18 + j * 1.6), 26, 18, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#fdf3e3", ec=AMB, lw=1))
    ax.add_patch(FancyBboxPatch((55, 18), 26, 18, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#fdf3e3", ec=AMB, lw=1.4))
    ax.text(68, 31.5, "16 routed experts (MXFP4)", ha="center", fontsize=9.8, fontweight="bold")
    ax.text(68, 26.5, "each: latent 3584 → 3072 → 3584", ha="center", fontsize=8.4)
    ax.text(68, 22.0, "+ 2 shared experts (BF16)", ha="center", fontsize=8.4)
    ax.text(2, 9, "Context 1,048,576 tokens · vocabulary 160K · Attention Residuals · MoonViT-V2 vision encoder (401M)", fontsize=8.8, color=MUTE)
    # params donut
    ax2 = fig.add_axes([0.64, 0.12, 0.34, 0.74])
    vals = [2722.74, 57.18]
    wedges, _ = ax2.pie(vals, colors=[AMB, K3], startangle=90, counterclock=False, wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2))
    ax2.text(0, 0.1, "2.78T", ha="center", va="center", fontsize=24, fontweight="bold")
    ax2.text(0, -0.17, "parameters", ha="center", va="center", fontsize=10, color=MUTE)
    ax2.set_title("Where the parameters are", fontsize=12.5, loc="center")
    ax2.text(0, -1.32, "Routed experts: 2,722.7B (97.9%), MXFP4 → 1.45 TB\nEverything else: 57.2B, BF16 → 114 GB\n~105B active per token (16 of 896 experts + BF16 parts)",
             ha="center", va="top", fontsize=9.2, linespacing=1.5)
    save(fig, "f02_model_anatomy", "Sources: Kimi K3 config.json and model card; parameter split from the checkpoint's safetensors metadata (2,779,931,837,184 parameters).")

# ------------------------------------------------------------------ F3 node topology
def f3():
    fig, ax = plt.subplots(figsize=(12, 6.6)); ax.set_axis_off(); ax.set_xlim(0, 120); ax.set_ylim(0, 66)
    ax.text(0, 65.5, "One HGX B300 node: 8 GPUs, every pair one NVLink hop apart", fontsize=14, fontweight="bold", va="top")
    sw = (8, 27, 70, 12)
    ax.add_patch(FancyBboxPatch(sw[:2], sw[2], sw[3], boxstyle="round,pad=0.4,rounding_size=1.5", fc=INK, ec=INK))
    ax.text(43, 34.6, "NVLink 5 switch fabric (NVSwitch)", color="white", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(43, 30.2, "every GPU to every GPU · in-switch multicast and reduction (NVLS)", color="#c8d0dc", ha="center", fontsize=9)
    for g in range(8):
        top = g < 4
        x = 8 + (g % 4) * 18.2; y = 46 if top else 5
        ax.add_patch(FancyBboxPatch((x, y), 15, 11, boxstyle="round,pad=0.3,rounding_size=1", fc="#eef6f7", ec=K3, lw=1.4))
        ax.text(x + 7.5, y + 8.0, f"B300 #{g}", ha="center", fontsize=10, fontweight="bold")
        ax.text(x + 7.5, y + 4.9, "288 GB HBM3e", ha="center", fontsize=8.4)
        ax.text(x + 7.5, y + 2.0, "8 TB/s", ha="center", fontsize=8.4, color=MUTE)
        if top:
            ax.plot([x + 7.5, x + 7.5], [y, 39.4], color=K3, lw=3, alpha=0.8, solid_capstyle="round")
        else:
            ax.plot([x + 7.5, x + 7.5], [y + 11, 26.6], color=K3, lw=3, alpha=0.8, solid_capstyle="round")
    ax.text(43, 42.3, "NVLink 5: 1.8 TB/s per GPU (900 GB/s each way)", ha="center", fontsize=9.6, color=K3, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.add_patch(FancyBboxPatch((86, 4), 33, 54, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#f6f7f9", ec="#dde1e7"))
    ax.text(88, 55.5, "The machine", fontsize=11.5, fontweight="bold", va="top")
    specs = [("GPU memory", "8 × 288 GB = 2.3 TB"), ("Usable per GPU", "268.6 GiB"), ("HBM bandwidth", "8 × 8 TB/s = 64 TB/s"),
             ("GPU to GPU", "NVLink 5, 1.8 TB/s per GPU"), ("All-reduce", "SGLang custom (multicast) + NCCL"),
             ("Kimi K3 weights", "1.56 TB, 195 GB per GPU"), ("Price (Modal)", "8 × $7.10 = $56.80 per hour")]
    y = 49.5
    for k, v in specs:
        ax.text(88, y, k, fontsize=8.6, color=MUTE, va="top")
        ax.text(88, y - 2.6, v, fontsize=9.3, fontweight="bold", va="top")
        y -= 6.6
    save(fig, "f03_node_topology", "Sources: NVIDIA HGX B300 specifications (288 GB HBM3e and 8 TB/s per GPU; NVLink 5 at 1.8 TB/s per GPU); "
         "our server log (NCCL 2.30.7, custom all-reduce with multicast); Modal B300 price.")

# ------------------------------------------------------------------ F4 how the model is split
def f4():
    fig, ax = plt.subplots(figsize=(13, 7.6)); ax.set_axis_off(); ax.set_xlim(0, 130); ax.set_ylim(0, 76)
    ax.text(0, 75.5, "How one layer is split across the 8 GPUs (TP8 + DCP8; no expert or pipeline parallelism)", fontsize=14, fontweight="bold", va="top")
    gx = [36 + i * 11.6 for i in range(8)]
    for i, x in enumerate(gx):
        ax.text(x + 5, 67.5, f"GPU {i}", ha="center", fontsize=9.5, fontweight="bold", color=MUTE)
    rows_ = [
        (58, "Attention projections (BF16)", "96 heads → 12 per GPU", K3L),
        (45, "MLA KV cache in decode (DCP8)", "each request's tokens split 8 ways", AMB),
        (32, "KDA recurrent state", "12 of 96 heads' state per GPU", K3L),
        (19, "896 routed experts (MXFP4)", "each expert's 3072-wide FFN → 384 per GPU", "#f2c27a"),
        (6, "Shared experts, dense layer, lm_head", "column/row split (Megatron-style TP)", "#b9d7da"),
    ]
    for y, name, sub, c in rows_:
        ax.text(0, y + 4.2, name, fontsize=9.8, fontweight="bold", va="center")
        ax.text(0, y + 1.5, sub, fontsize=8.3, color=MUTE, va="center")
        for x in gx:
            ax.add_patch(FancyBboxPatch((x, y + 0.4), 10, 5.4, boxstyle="round,pad=0.2,rounding_size=0.8", fc=c, ec="white", lw=1))
    def link(y, color, label, ls="-"):
        ax.annotate("", xy=(gx[-1] + 10.2, y), xytext=(gx[0] - 0.2, y), arrowprops=dict(arrowstyle="<->", color=color, lw=1.6, ls=ls))
        ax.text((gx[0] + gx[-1] + 10) / 2, y + 0.9, label, ha="center", fontsize=8.6, color=color, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1.2))
    link(55.4, COR, "all-reduce after the attention output projection")
    link(42.4, IND, "DCP: gather the queries, then merge partial attention by log-sum-exp", ls="--")
    link(16.4, COR, "all-reduce after the MoE (fused with the residual add and RMSNorm)")
    ax.text(0, 1.2, "Per GPU: 195.3 GB of weights (1/8 of every matrix), 1/8 of each request's MLA cache and of its KDA state. "
                    "About 2 all-reduces per layer × 93 layers per forward pass, all over NVLink.", fontsize=9.3)
    save(fig, "f04_model_split", "From the server's settings: tp_size=8, dcp_size=8, ep_size=1, pp_size=1, moe_runner_backend=flashinfer_mxfp4, "
         "decode attention cutedsl_mla, prefill trtllm_mla. Collective counts are approximate.")

# ------------------------------------------------------------------ F5 memory per GPU
def f5():
    parts = [("Model weights", 195.32, K3), ("MLA KV cache (601,664 tokens)", 15.49, AMB), ("KDA state (265 slots)", 13.91, COR),
             ("CUDA graphs + runtime", 4.52, SLA), ("Unused at serving time", 35.36, "#e5e7eb")]
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    left = 0
    for name, v, c in parts:
        ax.barh(0, v, left=left, color=c, height=0.55, edgecolor="white", lw=1.5, zorder=3)
        if v > 20:
            ax.text(left + v / 2, 0, f"{v:.1f} GB", ha="center", va="center", color="white" if c not in ("#e5e7eb",) else INK, fontsize=10, fontweight="bold")
        left += v
    ax.set_xlim(0, 270); ax.set_ylim(-0.9, 0.9); ax.set_yticks([]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("GB on one GPU (of 264.6 GB free at start)")
    ax.axvline(0.85 * 268.6 * 1.0737, color=MUTE, ls=":", lw=1)
    ax.text(0.85 * 268.6 * 1.0737 + 1, 0.62, "mem-fraction-static 0.85", fontsize=8.5, color=MUTE)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    ax.legend(handles, [f"{n}: {v:.1f} GB" for n, v, _ in parts], ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=9.2)
    ax.set_title("GPU memory, one of eight GPUs: weights dominate, ~35 GB stayed unused", pad=52)
    save(fig, "f05_memory_per_gpu", "Source: server log on GPU 0 (Load weight end: 195.32 GB; KV cache 15.49 GB BF16; Mamba/KDA cache conv 0.47 + ssm 13.44 GB; "
         "35.36 GB available after CUDA-graph capture).")

# ------------------------------------------------------------------ F6 the cap: KV vs state capacity
def f6():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    kv_req = 4813312 / 4480  # agent request ~4,352 prompt + 128 question... use 4.48K tokens incl. answer
    vals = [("Requests the MLA KV cache could hold\n(4.81M tokens ÷ ~4.5K tokens per agent request)", kv_req, AMB),
            ("Requests the KDA state cache holds\n(265 slots ÷ 5 slots per request)", 53, COR),
            ("Requests we asked for\n(--max-running-requests)", 128, SLA)]
    for i, (n, v, c) in enumerate(vals):
        ax.barh(i, v, color=c, height=0.55, zorder=3)
        ax.text(v * 1.08, i, f"{v:,.0f}", va="center", fontsize=12, fontweight="bold", color=c)
    ax.set_xscale("log"); ax.set_xlim(10, 4000); ax.set_xticks([10, 30, 100, 300, 1000, 3000]); ax.xaxis.set_major_formatter(PLAIN); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(3), [v[0] for v in vals], fontsize=9.6); ax.invert_yaxis()
    ax.set_xlabel("concurrent requests (log scale)"); ax.grid(axis="y", visible=False)
    ax.set_title("The recurrent state, not the KV cache, sets the concurrency limit")
    save(fig, "f06_concurrency_cap", "Source: server log ('max_running_requests is capped to 53 by the mamba state cache (max_mamba_cache_size=265, 5 state slots per request)'; "
         "max_total_num_tokens=4,813,312).")

# ------------------------------------------------------------------ F7 decode step: bytes & comm
def f7():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    # left: bytes read per GPU per decode step at batch 1
    bf16 = 57.18e9 * 2 / 8 / 1e9 - 1.17e9 * 2 / 8 / 1e9  # minus the embedding table (one row read)
    exp = 2722.74e9 * 16 / 896 * (4.25 / 8) / 8 / 1e9
    parts = [("BF16 parts\n(attention, shared experts,\ndense layer, lm_head)", bf16, K3), ("16 routed experts\nper layer (MXFP4)", exp, AMB)]
    left = 0
    for n, v, c in parts:
        a1.bar(0, v, bottom=left, color=c, width=0.5, zorder=3); a1.text(0.32, left + v / 2, f"{n}: {v:.1f} GB", va="center", fontsize=9)
        left += v
    a1.set_xlim(-0.5, 2.2); a1.set_xticks([]); a1.set_ylabel("GB read per GPU per decode step (batch 1)")
    a1.set_title("Memory traffic for one token", fontsize=12.5)
    a1.text(-0.45, left + 0.8, f"≈ {left:.1f} GB ÷ 8 TB/s ≈ {left / 8 * 1000 / 1000:.2f} ms floor", fontsize=9.6, fontweight="bold")
    a1.set_ylim(0, left + 3); a1.grid(axis="x", visible=False)
    # right: measured ITL vs floor + comm estimate
    cats = ["Memory floor\n(17 GB @ 8 TB/s)", "All-reduces\n(~186 × ~5 µs)", "Measured ITL\n(1 user, p50)"]
    vals = [left / 8, 186 * 5 / 1000, 8.7]
    cols = [K3, COR, INK]
    a2.bar(range(3), vals, color=cols, width=0.55, zorder=3)
    for i, v in enumerate(vals):
        a2.text(i, v + 0.15, f"{v:.1f} ms", ha="center", fontsize=11, fontweight="bold")
    a2.set_xticks(range(3), cats, fontsize=9.5); a2.set_ylabel("milliseconds per token"); a2.grid(axis="x", visible=False)
    a2.set_title("Where 8.7 ms per token goes (estimate)", fontsize=12.5)
    a2.text(1.0, 7.3, "The rest: KDA recurrence, MoE routing,\nDCP combine, sampling, kernel gaps", ha="center", fontsize=9, color=MUTE)
    save(fig, "f07_decode_budget", "Estimates from the checkpoint's parameter counts (4.25 bits per MXFP4 weight incl. scales) and B300's 8 TB/s; all-reduce latency "
         "assumed ~5 µs for small messages on NVLink. Measured ITL from our 1-user test.")

# ------------------------------------------------------------------ F8 TTFT/ITL vs concurrency
def f8():
    P = {r["id"]: r for r in rows("points.csv")}
    c = [1, 16, 64]
    g = lambda f: [float(P[f"agent|conc|{x}"][f]) for x in c]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw=dict(wspace=0.38))
    for ax, (p50, p99, title, unit, log) in zip((a1, a2), [("ttft_p50", "ttft_p99", "Time to first token", "ms", True),
                                                          ("itl_p50", "itl_p99", "Time per output token", "ms", False)]):
        ax.plot(c, g(p50), "-o", color=K3, lw=2.4, ms=7, label="p50", zorder=3)
        ax.plot(c, g(p99), "-o", color=COR, lw=2.0, ms=6, label="p99", zorder=3)
        for x, v in zip(c, g(p50)):
            ax.annotate(f"{v:,.0f}" if v >= 100 else f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=K3, fontweight="bold")
        for x, v in zip(c, g(p99)):
            ax.annotate(f"{v:,.0f}" if v >= 100 else f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=COR)
        ax.set_xscale("log"); ax.set_xticks(c, [str(x) for x in c]); ax.minorticks_off()
        if log:
            ax.set_yscale("log"); ax.set_yticks([200, 500, 1000, 2000, 5000]); ax.yaxis.set_major_formatter(PLAIN); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("requests in flight"); ax.set_ylabel(unit); ax.set_title(title, fontsize=12.5); ax.legend(loc="upper left")
    tp = g("out_tok_s")
    a3.bar([str(x) for x in c], tp, color=[K3L, K3, K3], width=0.55, zorder=3)
    for i, v in enumerate(tp):
        a3.text(i, v + 40, f"{v:,.0f}", ha="center", fontsize=11, fontweight="bold")
    a3.set_xlabel("requests in flight"); a3.set_ylabel("output tokens/s, all requests"); a3.set_title("Throughput", fontsize=12.5); a3.grid(axis="x", visible=False)
    fig.suptitle("Agent workload: 4K shared system prompt + 128-token question, 256-token answers", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.04)
    save(fig, "f08_ttft_itl_concurrency", SRC)

# ------------------------------------------------------------------ F9 all tests throughput + TTFT
def f9():
    R = rows("points.csv")
    R.sort(key=lambda r: -float(r["out_tok_s"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    names = [r["test"] for r in R]
    y = np.arange(len(R))
    a1.barh(y, [float(r["out_tok_s"]) for r in R], color=[K3 if r["workload"] == "agent" and r["mode"] == "conc" else SLA for r in R], height=0.6, zorder=3)
    for i, r in enumerate(R):
        a1.text(float(r["out_tok_s"]) + 30, i, f"{float(r['out_tok_s']):,.0f}", va="center", fontsize=9.5)
    a1.set_yticks(y, names); a1.invert_yaxis(); a1.set_xlabel("output tokens/s"); a1.set_title("Throughput by test", fontsize=12.5); a1.grid(axis="y", visible=False)
    a2.barh(y - 0.17, [float(r["ttft_p50"]) / 1000 for r in R], height=0.32, color=K3, label="TTFT p50", zorder=3)
    a2.barh(y + 0.17, [float(r["ttft_p99"]) / 1000 for r in R], height=0.32, color=COR, label="TTFT p99", zorder=3)
    a2.set_xlabel("seconds to first token"); a2.set_title("Wait for the first token", fontsize=12.5); a2.legend(loc="lower right"); a2.grid(axis="y", visible=False)
    save(fig, "f09_all_tests", SRC + " No request failed in any test.")

# ------------------------------------------------------------------ F10 running vs queued timeline
def f10():
    T = rows("timeline.csv")
    t = np.array([float(r["t_s"]) for r in T]) / 60
    run = np.array([float(r["running"]) for r in T]); q = np.array([float(r["queued"]) for r in T])
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.fill_between(t, run, step=None, color=K3, alpha=0.18, lw=0)
    ax.plot(t, run, color=K3, lw=2, label="running")
    ax.plot(t, q, color=COR, lw=2, label="waiting in queue")
    ax.axhline(53, color=INK, ls="--", lw=1.2); ax.text(t.max() * 0.995, 55, "cap: 53 running (state slots)", ha="right", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("minutes of test traffic (pauses between tests removed)"); ax.set_ylabel("requests"); ax.set_ylim(0, 64)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("At 64 concurrent requests, 53 run and the rest wait")
    save(fig, "f10_running_queued", "Source: SGLang gauges sampled every 5 s during our tests.")

# ------------------------------------------------------------------ F11 tails
def f11():
    T = rows("tails.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (m, title, c, scale) in zip(axes, [("ttft", "First token", K3, 1000), ("itl", "Per output token", AMB, 1), ("e2e", "Whole request", IND, 1000)]):
        raw = sorted(((float(r["slower_than"]), float(r["ms"])) for r in T if r["metric"] == m and float(r["slower_than"]) > 0), key=lambda p: -p[0])
        pts, hi = [], 0.0
        for a, b in raw:  # from the median toward the tail, latency can only grow
            hi = max(hi, b); pts.append((a, hi))
        best = {}
        for a, b in pts:  # one point per share: the highest latency reported for it
            best[a] = max(best.get(a, 0.0), b)
        pts = sorted(best.items())
        x = [a * 100 for a, _ in pts]; yv = [b / scale for _, b in pts]
        ax.plot(x, yv, color=c, lw=2.4)
        ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xticks([50, 10, 1, 0.1], ["50%", "10%", "1%", "0.1%"]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("slowest share of requests" if m != "itl" else "slowest share of tokens")
        ax.set_ylabel("seconds" if scale == 1000 else "ms"); ax.set_title(title, fontsize=12.5)
        for pct in (0.5, 0.01, 0.001):
            near = min(pts, key=lambda p: abs(p[0] - pct))
            ax.plot(near[0] * 100, near[1] / scale, "o", color=c, ms=5)
            ax.annotate(f"{near[1] / scale:.1f}" if scale > 1 else f"{near[1]:.0f}", (near[0] * 100, near[1] / scale), textcoords="offset points", xytext=(6, 4), fontsize=9)
    fig.suptitle("Latency tails: 400 agent requests, 64 in flight (the p99.9 run)", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.04)
    save(fig, "f11_tails", SRC)

# ------------------------------------------------------------------ F12 MLA kernels
def f12():
    M = rows("mla.csv")
    shapes = ["1 × 8K", "16 × 8K", "64 × 4K", "16 × 32K", "4 × 128K"]
    keys = [("tp8", "bf16", K3L, "TP8, BF16 cache"), ("tp8", "fp8_e4m3", K3, "TP8, FP8 cache"), ("dcp8", "bf16", "#f2c27a", "DCP8, BF16 cache"), ("dcp8", "fp8_e4m3", AMB, "DCP8, FP8 cache")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    x = np.arange(len(shapes)); w = 0.2
    for i, (par, kv, c, lab) in enumerate(keys):
        v = [next(float(r["us"]) for r in M if r["shape"] == s and r["parallel"] == par and r["kv_dtype"] == kv and r["layout"] == "scattered") for s in shapes]
        a1.bar(x + (i - 1.5) * w, v, w * 0.92, color=c, label=lab, zorder=3)
    a1.set_xticks(x, [s + " tokens" for s in shapes], fontsize=9.3); a1.set_ylabel("µs per layer, one GPU"); a1.legend(ncol=2, fontsize=9)
    a1.set_title("MLA decode kernel time (requests × context)", fontsize=12.5); a1.grid(axis="x", visible=False)
    sp = [next(float(r["us"]) for r in M if r["shape"] == s and r["parallel"] == "tp8" and r["kv_dtype"] == "bf16" and r["layout"] == "scattered") /
          next(float(r["us"]) for r in M if r["shape"] == s and r["parallel"] == "dcp8" and r["kv_dtype"] == "bf16" and r["layout"] == "scattered") for s in shapes]
    a2.barh(range(len(shapes)), sp, color=AMB, height=0.55, zorder=3)
    for i, v in enumerate(sp):
        a2.text(v + 0.08, i, f"{v:.1f}×", va="center", fontsize=10.5, fontweight="bold")
    a2.set_yticks(range(len(shapes)), shapes); a2.invert_yaxis(); a2.axvline(1, color=MUTE, lw=1)
    a2.set_xlabel("TP8 time ÷ DCP8 time"); a2.set_title("DCP8 speed-up", fontsize=12.5); a2.grid(axis="y", visible=False)
    save(fig, "f12_mla_kernels", "Source: our Phase 4 microbenchmarks, FlashInfer trtllm-gen MLA decode on one B300, timed in CUDA graphs; 64-token pages scattered over the pool.")

# ------------------------------------------------------------------ F13 FP4 formats
def f13():
    F = rows("fp4.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (mat, title) in zip(axes, [("expert_gate_up", "Expert gate+up matrix (6144 × 3584)"), ("attn_o_tp8", "Attention output, per GPU (7168 × 1536)")]):
        base = {int(r["tokens"]): float(r["us"]) for r in F if r["matrix"] == mat and r["precision"] == "bf16"}
        for p, c, lab in [("fp8", IND, "FP8"), ("mxfp8", SLA, "MXFP8"), ("mxfp4", K3, "MXFP4 (Kimi's format)"), ("nvfp4", AMB, "NVFP4")]:
            pts = sorted((int(r["tokens"]), base[int(r["tokens"])] / float(r["us"])) for r in F if r["matrix"] == mat and r["precision"] == p)
            ax.plot([a for a, _ in pts], [b for _, b in pts], "-o", color=c, lw=2.4 if p == "mxfp4" else 1.6, ms=5, label=lab)
        ax.axhline(1, color=INK, lw=1, ls="--"); ax.text(1.05, 1.06, "BF16", fontsize=9, color=MUTE)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 16, 64, 256, 1024, 4096], ["1", "16", "64", "256", "1K", "4K"]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("tokens in the batch"); ax.set_title(title, fontsize=12.5); ax.minorticks_off()
    axes[0].set_ylabel("speed relative to BF16 (×)"); axes[0].legend(loc="upper left", fontsize=9.5)
    fig.suptitle("FP4 pays off once batches grow: Blackwell's FP4 tensor cores reach ~3× BF16 at prefill sizes", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.04)
    save(fig, "f13_fp4_formats", "Source: our Phase 4 microbenchmarks on one B300 (cuBLAS BF16/FP8, FlashInfer mm_mxfp8 / mm_fp4), timed in CUDA graphs.")

# ------------------------------------------------------------------ F14 cost
def f14():
    P = {r["id"]: r for r in rows("points.csv")}
    hourly = 8 * 7.10
    cases = [("1 user (agent)", "agent|conc|1"), ("16 agents", "agent|conc|16"), ("64 agents", "agent|conc|64"), ("64 chat", "chat|conc|64"), ("p99.9 run, 64", "agent|p999|64")]
    vals = [(n, hourly / (float(P[k]["out_tok_s"]) * 3600) * 1e6) for n, k in cases]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a1.barh(range(len(vals)), [v for _, v in vals], color=[SLA, K3L, K3, K3, K3], height=0.55, zorder=3)
    for i, (_, v) in enumerate(vals):
        a1.text(v * 1.08, i, f"${v:,.2f}", va="center", fontsize=10.5, fontweight="bold")
    a1.axvline(15, color=COR, ls="--", lw=1.4); a1.text(15 * 1.06, 4.45, "Kimi API list price: $15 / 1M output", color=COR, fontsize=9)
    a1.set_xscale("log"); a1.set_xticks([5, 10, 20, 50, 100, 200], ["$5", "$10", "$20", "$50", "$100", "$200"]); a1.xaxis.set_minor_formatter(NullFormatter())
    a1.set_xlim(4, 400); a1.set_yticks(range(len(vals)), [n for n, _ in vals]); a1.invert_yaxis()
    a1.set_xlabel("$ per 1M output tokens (8 × B300 at $56.80/h; log scale)"); a1.set_title("Cost per million output tokens", fontsize=12.5); a1.grid(axis="y", visible=False)
    spend = [("Weights download ×2 (CPU)", 2.0), ("Dev checks + S3 tests", 1.5), ("Preempted / stopped attempts", 13.0), ("The profiling run (~40 min GPU)", 32.5)]
    left = 0
    for n, v in spend:
        a2.barh(0, v, left=left, color={0: SLA, 1: "#c3c9d2", 2: COR, 3: K3}[spend.index((n, v))], height=0.5, edgecolor="white")
        left += v
    a2.barh(1, 200, color="#eef0f3", height=0.5); a2.text(100, 1, "$200 budget", ha="center", va="center", fontsize=10.5, fontweight="bold")
    a2.text(left + 3, 0, f"≈ ${left:.0f}", va="center", fontsize=11, fontweight="bold")
    a2.set_xlim(0, 210); a2.set_yticks([0, 1], ["spent", "budget"]); a2.set_xlabel("US$ (Modal, metered)"); a2.set_title("What the whole project cost", fontsize=12.5)
    a2.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (SLA, "#c3c9d2", COR, K3)], [n for n, _ in spend], loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=8.6, ncol=2)
    a2.grid(axis="y", visible=False)
    save(fig, "f14_cost", "Cost per token = hourly price ÷ measured output tokens/s. Spend split is approximate; Modal metered total ≈ $49 (of which $30 free credits).")

# ------------------------------------------------------------------ F15 resilient run pipeline
def f15():
    fig, ax = plt.subplots(figsize=(12.5, 4.6)); ax.set_axis_off(); ax.set_xlim(0, 125); ax.set_ylim(0, 46)
    ax.text(0, 45, "Profiling on preemptible GPUs: 5-minute pieces, each checkpointed to S3", fontsize=14, fontweight="bold", va="top")
    steps = [("recover", "read S3: which\npieces are done", SLA), ("P4 kernels", "3 pieces\n~70 s total", AMB), ("server", "load 1.56 TB\n6.8 min (prefetch)", "#9aa3af"),
             ("P3 tests", "10 pieces\n49–134 s each", K3), ("export", "102 KB\nfor git", IND)]
    x = 2
    for i, (t, s, c) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 18), 19, 13, boxstyle="round,pad=0.4,rounding_size=1.4", fc=c, ec=c))
        ax.text(x + 9.5, 27.5, t, ha="center", color="white", fontsize=11, fontweight="bold"); ax.text(x + 9.5, 22, s, ha="center", color="white", fontsize=8.7)
        if i < 4:
            ax.annotate("", xy=(x + 23.3, 24.5), xytext=(x + 19.6, 24.5), arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.4))
        x += 24
    ax.add_patch(FancyBboxPatch((26, 3), 95, 7.5, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#f6f7f9", ec="#dde1e7"))
    ax.text(73.5, 6.7, "S3: after every piece, its files, then a done marker  ·  a restarted machine resumes at the next piece", ha="center", fontsize=9.6)
    for xx in (35.5, 83.5, 107.5):
        ax.annotate("", xy=(xx, 10.8), xytext=(xx, 17.6), arrowprops=dict(arrowstyle="->", color=K3, lw=1.2))
    ax.annotate("", xy=(11.5, 17.6), xytext=(26.5, 7), arrowprops=dict(arrowstyle="->", color=COR, lw=1.2, connectionstyle="arc3,rad=-0.3"))
    ax.text(4, 9, "on restart", fontsize=8.8, color=COR)
    save(fig, "f15_resilient_pipeline", "13 of 14 pieces finished before the workspace hit its usage limit; nothing measured was lost.")


for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15):
    f()
