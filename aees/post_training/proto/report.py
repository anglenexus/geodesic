"""Overlap prototype report (PLAN.md §5b): reads one bench.sh run folder, writes report.md + report.json.

  uv run python -m proto.report data/proto/<RUN>
Input: <run>/<phase>/phase.json + agents.jsonl, train.jsonl, gpu.csv, dcgm.txt, hotload.jsonl (bench.sh),
and optionally <run>/ec2/calls.jsonl (llm_log_proxy.py on EC2: real Harbor trials).
Output: per-phase serving/training/GPU numbers; for each co-run phase, the slowdowns against the matching
alone phases, projected step times (alternating vs overlapped) and pass/fail against CRITERIA; and
dashboard.html, proto/dashboard.html with report.json embedded (charts, opens offline).
"""
import json
import statistics
import sys
from pathlib import Path

from proto.shapes import Shapes

CRITERIA = {"headroom_gb": 8.0, "decode_slowdown": 1.5, "train_slowdown": 1.5, "stall_p99_s": 10.0, "saving": 0.15}
TOKENS_PER_STEP = 1.9e6          # 20 tasks x 8 attempts, trained tokens per GRPO step (PLAN §5)
ROLLOUT_MIN = {"EC2 sandboxes, 32 at once": 40.0, "local sandboxes, 16 at once": 64.0}
SWITCH_MIN = 2.0                 # alternating: offload/onload + adapter sync per step
WARMUP_S = 60.0                  # first minute of each phase: agents ramp up, caches fill


def jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass                                       # a half-written last line
    return rows


def q(v: list, f: float):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(f * len(v)))] if v else None


def gpu_rows(path: Path) -> list[dict]:
    """bench.sh's nvidia-smi sampler: epoch,util %,mem util %,mem used MiB,mem total MiB,power W."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = [x.strip() for x in line.split(",")]
            try:
                rows.append({"t": float(p[0]), "util": float(p[1]), "mem_util": float(p[2]),
                             "used_gb": float(p[3]) / 1024, "total_gb": float(p[4]) / 1024})
            except (ValueError, IndexError):
                pass
    return rows


def dcgm_rows(path: Path) -> list[dict]:
    """`<epoch> GPU 0 SMACT TENSO DRAMA PCITX PCIRX` lines from dcgmi dmon (fields 1002,1004,1005,1009,1010)."""
    rows, keys = [], ("sm_active", "tensor_active", "dram_active", "pcie_tx", "pcie_rx")
    if path.exists():
        for line in path.read_text().split("\n"):
            p = line.split()
            if len(p) >= 8 and p[1] == "GPU":
                num = lambda s: None if s.upper().startswith("N/A") else float(s)
                try:
                    rows.append({"t": float(p[0]), **dict(zip(keys, map(num, p[3:8])))})
                except ValueError:
                    pass
    return rows


def load_run(run: Path) -> dict:
    phases = {}
    for meta in sorted(run.glob("*/phase.json")):
        d = meta.parent
        phases[d.name] = {"meta": json.loads(meta.read_text()), "agents": jsonl(d / "agents.jsonl"),
                          "train": jsonl(d / "train.jsonl"), "hot": jsonl(d / "hotload.jsonl"),
                          "gpu": gpu_rows(d / "gpu.csv"), "dcgm": dcgm_rows(d / "dcgm.txt")}
    return dict(sorted(phases.items(), key=lambda kv: kv[1]["meta"]["t0"]))          # in the order they ran


def train_window(train: list[dict]):
    steps = [r for r in train if r.get("type") == "step"]
    return (min(r["t"] - r["seconds"] for r in steps), max(r["t"] for r in steps)) if steps else None


def agent_stats(rows: list[dict], lo: float, hi: float, inside: bool, weight_bytes: float) -> dict:
    """Serving numbers for calls whose first token falls inside (or outside) [lo, hi]."""
    sel = [r for r in rows if r.get("t_first") and not r.get("error") and (lo <= r["t_first"] <= hi) == inside]
    errors = sum(1 for r in rows if r.get("error") and (lo <= r["t_send"] <= hi) == inside)
    if not sel:
        return {"calls": 0, "errors": errors}
    per_tok = [(r["t_end"] - r["t_first"]) / (r["out"] - 1) for r in sel if r["out"] >= 16]
    lat = [r["t_end"] - r["t_send"] for r in sel]
    span = (max(r["t_end"] for r in sel) - min(r["t_send"] for r in sel)) if not inside else hi - lo
    p50 = q(per_tok, 0.5)
    return {"calls": len(sel), "errors": errors,
            "out_tok_s": round(sum(r["out"] for r in sel) / max(span, 1e-9), 1),
            "per_tok_ms_p50": p50 and round(1000 * p50, 2), "per_tok_ms_p90": per_tok and round(1000 * q(per_tok, .9), 2),
            "ttft_s_p50": round(q([r["t_first"] - r["t_send"] for r in sel], .5), 3),
            "ttft_s_p99": round(q([r["t_first"] - r["t_send"] for r in sel], .99), 3),
            "latency_s_p99": round(q(lat, .99), 2), "stall_s_p99": round(q([r.get("max_gap", 0) for r in sel], .99), 3),
            "gen_share": round(sum(lat) / max(sum(lat) + sum(r.get("sleep", 0) for r in sel), 1e-9), 3),
            "weight_stream_gbs": p50 and round(weight_bytes / p50 / 1e9)}     # decode re-reads all weights per step


def train_stats(train: list[dict]) -> dict:
    steps = [r for r in train if r.get("type") == "step"]
    start = ([r for r in train if r.get("type") == "start"] or [{}])[-1]      # the 32k retry, after an OOM
    out = {"oom": any(r.get("type") == "oom" for r in train), "seq": start.get("seq"), "offload": start.get("offload")}
    timed = steps[1:] or steps                                  # the first micro-batch includes warm-up
    if timed:
        sec = statistics.median(r["seconds"] for r in timed)
        out.update({"steps": len(steps), "seconds": round(sec, 2), "tok_s": round(timed[0]["tokens"] / sec),
                    "tflops": statistics.median(r["tflops"] for r in timed), "peak_gb": max(r["peak_gb"] for r in steps)})
    return out


def window_mean(rows: list[dict], key: str, lo: float, hi: float):
    v = [r[key] for r in rows if lo <= r["t"] <= hi and r.get(key) is not None]
    return round(statistics.mean(v), 3) if v else None


def decode_weight_bytes(s: Shapes, weights: str) -> float:
    """Bytes read per decode step: decoder linears (1 B in FP8, 2 B in bf16) + the bf16 output layer."""
    head = s.hidden * s.vocab
    return (s.matmul_params() - head) * (1 if weights == "fp8" else 2) + head * 2


def timeline(p: dict) -> dict:
    """Raw series for dashboard.html, in seconds from the phase start: calls [sent, first token, end, output
    tokens, longest stall s], errors [sent], training steps [end, seconds], GPU [t, util %, used GB],
    DCGM [t, SM, tensor, DRAM active], adapter hot-loads [t, load s, ok]."""
    t0 = p["meta"]["t0"]
    r = lambda t: round(t - t0, 2)
    return {"calls": [[r(c["t_send"]), r(c["t_first"]), r(c["t_end"]), c["out"], c.get("max_gap")]
                      for c in p["agents"] if c.get("t_first") and not c.get("error")],
            "errors": [r(c["t_send"]) for c in p["agents"] if c.get("error")],
            "train": [[r(x["t"]), x["seconds"]] for x in p["train"] if x.get("type") == "step"],
            "gpu": [[r(g["t"]), g["util"], round(g["used_gb"], 1)] for g in p["gpu"]],
            "dcgm": [[r(d["t"]), d["sm_active"], d["tensor_active"], d["dram_active"]] for d in p["dcgm"]],
            "hot": [[r(h["t"]), h.get("load_s"), h.get("ok")] for h in p["hot"]]}


def phase_summary(p: dict, s: Shapes) -> dict:
    m = p["meta"]
    t0, t1 = m["t0"] + WARMUP_S, m["t1"]
    win = train_window(p["train"])
    wb = decode_weight_bytes(s, m.get("serve") or "bf16")
    out = {"meta": m, "train": train_stats(p["train"]) if p["train"] else None, "timeline": timeline(p)}
    if p["agents"]:
        lo, hi = win or (t1, t1)
        out["alone"] = agent_stats([r for r in p["agents"] if r["t_send"] >= t0], lo, hi, False, wb)
        out["with_train"] = agent_stats(p["agents"], lo, hi, True, wb) if win else None
    lo, hi = win or (t0, t1)
    out["gpu"] = {"mem_max_gb": max((r["used_gb"] for r in p["gpu"]), default=None),
                  "mem_total_gb": max((r["total_gb"] for r in p["gpu"]), default=None),
                  **{k: window_mean(p["dcgm"], k, lo, hi) for k in ("sm_active", "tensor_active", "dram_active")}}
    if p["hot"]:
        ok = [r["load_s"] for r in p["hot"] if r.get("ok")]
        out["hotload"] = {"tries": len(p["hot"]), "ok": len(ok), "load_s_p50": q(ok, .5), "load_s_max": max(ok, default=None)}
    return out


def ec2_stats(calls: list[dict], t0: float, t1: float) -> dict:
    """Real Harbor trials: model latency per call and the share of trial time spent waiting on the model."""
    sel = [c for c in calls if t0 <= c["t_send"] <= t1]
    ok = [c for c in sel if c.get("status") == 200]
    by_trial = {}
    for c in ok:
        by_trial.setdefault(c["trial"], []).append(c)
    lat, gap = 0.0, 0.0
    for turns in by_trial.values():
        turns.sort(key=lambda c: c["turn"])
        for a, b in zip(turns, turns[1:]):
            if b["turn"] == a["turn"] + 1:
                lat, gap = lat + a["t_end"] - a["t_send"], gap + b["t_send"] - a["t_end"]
    lats = [c["t_end"] - c["t_send"] for c in ok]
    return {"calls": len(sel), "errors": len(sel) - len(ok), "trials": len(by_trial),
            "latency_s_p50": q(lats, .5), "latency_s_p99": q(lats, .99),
            "gen_share": round(lat / (lat + gap), 3) if lat + gap > 0 else None}


def baseline(summ: dict, name: str, kind: str):
    """The alone phase matching a co-run phase: same server settings (kind=serve) or same training (kind=train)."""
    m = summ[name]["meta"]
    for other, o in summ.items():
        om = o["meta"]
        if other == name or om.get("co"):
            continue
        if kind == "serve" and not om.get("train") and om.get("serve") == m["serve"] \
                and om.get("mps") == m["mps"] and om.get("agents") == m["agents"]:
            return o
        if kind == "train" and not om.get("serve") and om.get("train") and o["train"] and summ[name]["train"] \
                and o["train"]["offload"] == summ[name]["train"]["offload"] and o["train"]["seq"] == summ[name]["train"]["seq"]:
            return o
    return None


def project(train_min: float, f_train: float, f_decode: float, gen_share: float) -> dict:
    """Step minutes per rollout scenario. Alternating: R + T + S. Overlapped: training (slowed by f_train) runs
    inside the rollout, which it slows by gen_share x (f_decode - 1) while they overlap."""
    out = {}
    for name, r in ROLLOUT_MIN.items():
        t_co = train_min * f_train
        alt = r + train_min + SWITCH_MIN
        co = max(r + gen_share * (f_decode - 1) * t_co, t_co + 1)
        out[name] = {"alternating_min": round(alt, 1), "overlapped_min": round(co, 1), "saving": round(1 - co / alt, 3),
                     "rollout_min": r, "train_min_overlapped": round(t_co, 1)}
    return out


def verdict(summ: dict, name: str, gen_share: float) -> dict:
    """Pass/fail of one co-run phase against CRITERIA, with the numbers behind each check."""
    co, serve_b, train_b = summ[name], baseline(summ, name, "serve"), baseline(summ, name, "train")
    wt, tr = co.get("with_train") or {}, co.get("train") or {}
    out = {"serve_baseline": serve_b and serve_b["meta"]["name"], "train_baseline": train_b and train_b["meta"]["name"]}
    mem, total = co["gpu"]["mem_max_gb"], co["gpu"]["mem_total_gb"]
    checks = {"no_oom": (not tr.get("oom") and tr.get("steps", 0) > 0, None)}
    if mem is not None and total:
        checks["headroom_gb"] = (round(total - mem, 1), CRITERIA["headroom_gb"])
    if serve_b and wt.get("per_tok_ms_p50") and serve_b["alone"].get("per_tok_ms_p50"):
        checks["decode_slowdown"] = (round(wt["per_tok_ms_p50"] / serve_b["alone"]["per_tok_ms_p50"], 2),
                                     CRITERIA["decode_slowdown"])
    if wt.get("stall_s_p99") is not None:
        checks["stall_p99_s"] = (wt["stall_s_p99"], CRITERIA["stall_p99_s"])
    if train_b and tr.get("seconds") and train_b["train"].get("seconds"):
        checks["train_slowdown"] = (round(tr["seconds"] / train_b["train"]["seconds"], 2), CRITERIA["train_slowdown"])
    if "decode_slowdown" in checks and "train_slowdown" in checks:
        t_min = TOKENS_PER_STEP / train_b["train"]["tok_s"] / 60
        out["train_min_alone"] = round(t_min, 1)
        out["projection"] = project(t_min, checks["train_slowdown"][0], checks["decode_slowdown"][0], gen_share)
        checks["saving"] = (out["projection"][next(iter(ROLLOUT_MIN))]["saving"], CRITERIA["saving"])   # design B: EC2
    out["checks"] = {k: {"value": v, "limit": lim,
                         "pass": bool(v) if lim is None else (v >= lim if k in ("headroom_gb", "saving") else v <= lim)}
                     for k, (v, lim) in checks.items()}
    expected = {"no_oom", "headroom_gb", "decode_slowdown", "stall_p99_s", "train_slowdown", "saving"}
    out["pass"] = expected <= set(out["checks"]) and all(c["pass"] for c in out["checks"].values())
    return out


def build(run: Path) -> dict:
    phases = load_run(run)
    cfg = run / "config.json"
    s = Shapes.from_config(cfg) if cfg.exists() else Shapes()
    summ = {n: phase_summary(p, s) for n, p in phases.items()}
    calls = jsonl(run / "ec2" / "calls.jsonl")
    for n, p in phases.items():
        if calls:
            summ[n]["ec2"] = ec2_stats(calls, p["meta"]["t0"], p["meta"]["t1"])
    real = [summ[n]["ec2"]["gen_share"] for n in summ if summ[n].get("ec2", {}).get("gen_share") is not None
            and not summ[n]["meta"].get("co")]
    synth = [summ[n]["alone"]["gen_share"] for n in summ if summ[n].get("alone", {}).get("gen_share")
             and not summ[n]["meta"].get("co")]
    gen_share = statistics.median(real) if real else statistics.median(synth) if synth else 0.4
    pcie = next((r for r in jsonl(run / "pcie" / "train.jsonl") if r.get("type") == "pcie"), None)
    co = [n for n in summ if summ[n]["meta"].get("co")]
    t0 = min((p["meta"]["t0"] for p in phases.values()), default=0.0)
    return {"run": run.name, "gen_share": gen_share, "gen_share_source": "real trials" if real else "synthetic agents",
            "pcie": pcie, "phases": summ, "verdicts": {n: verdict(summ, n, gen_share) for n in co},
            "constants": {"criteria": CRITERIA, "tokens_per_step": TOKENS_PER_STEP, "switch_min": SWITCH_MIN,
                          "warmup_s": WARMUP_S}, "t0": t0,
            "ec2_calls": [[round(c["t_send"] - t0, 2), round(c["t_end"] - c["t_send"], 2), c.get("status")]
                          for c in calls]}          # real trials over the whole run: [sent, latency s, HTTP status]


def fmt(v) -> str:
    return "–" if v is None else f"{v:g}" if isinstance(v, float) else str(v)


def markdown(rep: dict) -> str:
    L = [f"# Overlap prototype {rep['run']}", "",
         f"Generation share of trial time: {rep['gen_share']} ({rep['gen_share_source']}). "
         f"PCIe: {fmt((rep['pcie'] or {}).get('h2d_gbps'))} GB/s to GPU, {fmt((rep['pcie'] or {}).get('d2h_gbps'))} GB/s back.", "",
         "| phase | serve | MPS | agents | calls | out tok/s | ms/token p50 | p90 | TTFT p99 s | stall p99 s | weights GB/s"
         " | train tok/s | TFLOPS | peak GB | GPU mem max GB | SM / tensor / DRAM active |",
         "|" + "---|" * 16]
    for n, p in rep["phases"].items():
        m, tr, g = p["meta"], p.get("train") or {}, p["gpu"]
        for label, a in [("", p.get("alone")), (" (with training)", p.get("with_train"))]:
            if a is None and label:
                continue
            a = a or {}
            L.append(f"| {n}{label} | {fmt(m.get('serve'))} | {fmt(m.get('mps_pct') or m.get('mps'))} | {fmt(m.get('agents'))}"
                     f" | {fmt(a.get('calls'))} | {fmt(a.get('out_tok_s'))} | {fmt(a.get('per_tok_ms_p50'))}"
                     f" | {fmt(a.get('per_tok_ms_p90'))} | {fmt(a.get('ttft_s_p99'))} | {fmt(a.get('stall_s_p99'))}"
                     f" | {fmt(a.get('weight_stream_gbs'))} | {fmt(tr.get('tok_s'))} | {fmt(tr.get('tflops'))}"
                     f" | {fmt(tr.get('peak_gb'))}{' OOM' if tr.get('oom') else ''} | {fmt(g['mem_max_gb'] and round(g['mem_max_gb'], 1))}"
                     f" | {fmt(g['sm_active'])} / {fmt(g['tensor_active'])} / {fmt(g['dram_active'])} |")
        if p.get("hotload"):
            L.append(f"| {n} adapter hot-load | | | | {p['hotload']['ok']}/{p['hotload']['tries']} ok"
                     f" | p50 {fmt(p['hotload']['load_s_p50'])} s, max {fmt(p['hotload']['load_s_max'])} s |" + " |" * 10)
    for n, v in rep["verdicts"].items():
        L += ["", f"## {n}: {'PASS' if v['pass'] else 'FAIL'}",
              f"Baselines: serving {v['serve_baseline']}, training {v['train_baseline']}.", ""]
        L += [f"- {k}: {fmt(c['value'])} (limit {fmt(c['limit'])}) {'ok' if c['pass'] else 'FAIL'}" for k, c in v["checks"].items()]
        for sc, pr in (v.get("projection") or {}).items():
            L.append(f"- {sc}: alternating {pr['alternating_min']} min/step, overlapped {pr['overlapped_min']}"
                     f" min/step ({100 * pr['saving']:.0f}% saved; training alone {v['train_min_alone']} min/step)")
    return "\n".join(L) + "\n"


def dashboard(rep: dict) -> str:
    """proto/dashboard.html with the report embedded, so the saved page opens offline."""
    page = Path(__file__).with_name("dashboard.html").read_text()
    return page.replace("const EMBEDDED = null;", "const EMBEDDED = " + json.dumps(rep).replace("</", "<\\/") + ";", 1)


if __name__ == "__main__":
    run = Path(sys.argv[1])
    rep = build(run)
    (run / "report.json").write_text(json.dumps(rep, indent=1))
    (run / "report.md").write_text(markdown(rep))
    (run / "dashboard.html").write_text(dashboard(rep))
    print(markdown(rep))
    print(f"dashboard: {run / 'dashboard.html'}")
