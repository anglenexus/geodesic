"""Tests for proto/report.py on a synthetic bench run whose slowdowns and projections are known."""
import json
import re
import subprocess
from pathlib import Path

import pytest

from proto import report as rp
from proto.shapes import Shapes

T0 = 1_790_000_000.0
HERE = Path(__file__).resolve().parents[1]
TOTAL_MIB = 143771                                         # H200


def write_phase(run, name, meta, agents=(), train=(), gpu=(), dcgm=(), hot=()):
    d = run / name
    d.mkdir(parents=True)
    (d / "phase.json").write_text(json.dumps({"name": name, "mps_pct": None, "lora": True, **meta}))
    for fname, rows in [("agents.jsonl", agents), ("train.jsonl", train), ("hotload.jsonl", hot)]:
        if rows:
            (d / fname).write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"half')
    if gpu:
        (d / "gpu.csv").write_text("".join(f"{t},50,60,{used},{TOTAL_MIB},500\n" for t, used in gpu) + "garbage\n")
    if dcgm:
        (d / "dcgm.txt").write_text("1 #Entity SMACT TENSO DRAMA PCITX PCIRX\n" + "".join(
            f"{t} GPU 0 {sm} {tc} {dr} N/A 100\n" for t, sm, tc, dr in dcgm))


def calls(t_start, t_stop, per_tok, out=500, ttft=0.2, gen_share=0.4, gap=0.05):
    """Agent rows back to back from t_start: ttft + (out-1) x per_tok, then env sleep so gen share is exact."""
    rows, t = [], t_start
    lat = ttft + (out - 1) * per_tok
    sleep = lat * (1 - gen_share) / gen_share
    while t + lat < t_stop:
        rows.append({"agent": 0, "trial": "0.1", "turn": len(rows), "prompt": 1000, "t_send": t, "t_first": t + ttft,
                     "t_end": t + lat, "out": out, "cached": 900, "max_gap": gap, "sleep": sleep, "error": None})
        t += lat + 0.001
    return rows


def train_rows(t_start, n, seconds, seq=49152, offload=True, oom=False):
    rows = [{"t": t_start, "type": "start", "seq": seq, "offload": offload}]
    rows += [{"t": t_start + (k + 1) * seconds, "type": "step", "step": k, "seconds": seconds, "tokens": seq,
              "tflops": 400.0, "peak_gb": 80.0} for k in range(n)]
    return rows + ([{"t": t_start, "type": "oom"}] if oom else [])


@pytest.fixture
def run(tmp_path):
    r = tmp_path / "20261025_0100"
    r.mkdir()
    (r / "config.json").write_text(json.dumps({"text_config": {
        "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256, "attn_output_gate": True, "full_attention_interval": 4,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "vocab_size": 248320}}))
    (r / "pcie").mkdir()
    (r / "pcie" / "train.jsonl").write_text(json.dumps({"t": T0, "type": "pcie", "h2d_gbps": 52.1, "d2h_gbps": 50.3}) + "\n")
    write_phase(r, "train_alone_offload", {"serve": None, "mps": False, "agents": 0, "train": {"offload": True},
                                           "co": False, "t0": T0, "t1": T0 + 200}, train=train_rows(T0, 9, 20.0))
    write_phase(r, "train_alone", {"serve": None, "mps": False, "agents": 0, "train": {"offload": False},
                                   "co": False, "t0": T0, "t1": T0 + 200}, train=train_rows(T0, 9, 18.0, offload=False))
    a0 = T0 + 1000
    write_phase(r, "serve_fp8", {"serve": "fp8", "mps": False, "agents": 16, "train": None, "co": False,
                                 "t0": a0, "t1": a0 + 300}, agents=calls(a0, a0 + 300, 0.014),
                gpu=[(a0 + k, 60000) for k in range(300)], dcgm=[(a0 + 100, 0.9, 0.05, 0.85)])
    c0 = T0 + 2000                                         # co-run: training from +60 s for 5 min
    tw0, tw1 = c0 + 60, c0 + 60 + 12 * 26.0
    agents = calls(c0, tw0, 0.014) + calls(tw0 + 0.5, tw1 - 0.5, 0.021, gap=0.8) + calls(tw1 + 0.5, c0 + 420, 0.014)
    write_phase(r, "co_timeslice", {"serve": "fp8", "mps": False, "agents": 16, "train": {"offload": True}, "co": True,
                                    "t0": c0, "t1": c0 + 420}, agents=agents, train=train_rows(tw0, 12, 26.0),
                gpu=[(c0 + k, 130000) for k in range(420)],
                dcgm=[(tw0 + 10, 0.95, 0.4, 0.9), (tw0 + 20, 0.97, 0.5, 0.95), (c0 + 5, 0.5, 0.0, 0.8)],
                hot=[{"type": "hotload", "t": tw0 + 30, "load_s": 1.5, "unload_s": 0.1, "ok": True},
                     {"type": "hotload", "t": tw0 + 90, "load_s": 2.5, "unload_s": 0.1, "ok": True}])
    write_phase(r, "co_mps", {"serve": "fp8", "mps": True, "agents": 16, "train": {"offload": True}, "co": True,
                              "t0": c0 + 1000, "t1": c0 + 1420}, agents=calls(c0 + 1000, c0 + 1420, 0.015),
                train=train_rows(c0 + 1060, 1, 30.0, oom=True), gpu=[(c0 + 1000, 141000)])
    return r


def test_slowdowns_projection_and_verdict(run):
    rep = rp.build(run)
    assert rep["pcie"]["h2d_gbps"] == 52.1
    assert rep["gen_share"] == pytest.approx(0.4, abs=0.002) and rep["gen_share_source"] == "synthetic agents"
    co = rep["phases"]["co_timeslice"]
    assert co["with_train"]["per_tok_ms_p50"] == pytest.approx(21.0) and co["alone"]["per_tok_ms_p50"] == pytest.approx(14.0)
    assert co["with_train"]["stall_s_p99"] == 0.8 and co["train"]["seconds"] == 26.0
    assert co["gpu"]["dram_active"] == pytest.approx(0.925) and co["gpu"]["tensor_active"] == pytest.approx(0.45)
    assert co["hotload"] == {"tries": 2, "ok": 2, "load_s_p50": 2.5, "load_s_max": 2.5}
    s = Shapes()
    fp8_bytes = (s.matmul_params() - s.hidden * s.vocab) + 2 * s.hidden * s.vocab
    assert rep["phases"]["serve_fp8"]["alone"]["weight_stream_gbs"] == round(fp8_bytes / 0.014 / 1e9)

    v = rep["verdicts"]["co_timeslice"]
    assert (v["serve_baseline"], v["train_baseline"]) == ("serve_fp8", "train_alone_offload")
    c = {k: x["value"] for k, x in v["checks"].items()}
    assert c["decode_slowdown"] == 1.5 and c["train_slowdown"] == 1.3 and c["stall_p99_s"] == 0.8
    assert c["headroom_gb"] == round((TOTAL_MIB - 130000) / 1024, 1)
    t_min = 1.9e6 / round(49152 / 20.0) / 60
    assert v["train_min_alone"] == round(t_min, 1)
    g = rep["gen_share"]
    for name, r in rp.ROLLOUT_MIN.items():
        alt, co_min = r + t_min + rp.SWITCH_MIN, max(r + g * 0.5 * t_min * 1.3, t_min * 1.3 + 1)
        assert v["projection"][name] == {"alternating_min": round(alt, 1), "overlapped_min": round(co_min, 1),
                                         "saving": round(1 - co_min / alt, 3), "rollout_min": r,
                                         "train_min_overlapped": round(t_min * 1.3, 1)}
    assert c["saving"] == v["projection"]["EC2 sandboxes, 32 at once"]["saving"] > 0.15
    assert v["pass"] is True

    bad = rep["verdicts"]["co_mps"]                         # out of memory, no MPS serving baseline
    assert bad["pass"] is False and bad["checks"]["no_oom"]["pass"] is False and bad["serve_baseline"] is None
    assert bad["checks"]["headroom_gb"]["pass"] is False


def test_real_trials_set_the_generation_share(run):
    ec2 = run / "ec2"
    ec2.mkdir()
    a0, rows = T0 + 1000, []
    for trial in (1, 2):                                    # model 6 s, then 14 s of environment per turn
        t = a0 + 10 * trial
        for turn in range(4):
            rows.append({"t_send": t, "t_end": t + 6, "status": 200, "trial": trial, "turn": turn})
            t += 20
    rows.append({"t_send": a0 + 5, "t_end": a0 + 6, "status": 502, "trial": 3, "turn": 0, "error": "down"})
    (ec2 / "calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = rp.build(run)
    e = rep["phases"]["serve_fp8"]["ec2"]
    assert (e["calls"], e["errors"], e["trials"], e["latency_s_p50"]) == (9, 1, 2, 6)
    assert e["gen_share"] == 0.3 and rep["gen_share"] == 0.3 and rep["gen_share_source"] == "real trials"


def test_timelines_for_the_dashboard(run):
    rep = rp.build(run)
    assert list(rep["phases"]) == ["train_alone", "train_alone_offload", "serve_fp8", "co_timeslice", "co_mps"]   # run order
    assert rep["t0"] == T0 and rep["ec2_calls"] == [] and rep["constants"]["criteria"] == rp.CRITERIA
    tl = rep["phases"]["co_timeslice"]["timeline"]
    c0 = T0 + 2000
    assert tl["calls"][0] == [0.0, 0.2, round(0.2 + 499 * 0.014, 2), 500, 0.05]           # seconds from the phase start
    assert len(tl["calls"]) == sum(1 for r in jsonl_rows(run / "co_timeslice" / "agents.jsonl"))
    assert tl["train"][0] == [60 + 26.0, 26.0] and len(tl["train"]) == 12
    assert tl["gpu"][5] == [5.0, 50.0, round(130000 / 1024, 1)] and tl["dcgm"][0] == [70.0, 0.95, 0.4, 0.9]
    assert tl["hot"] == [[90.0, 1.5, True], [150.0, 2.5, True]] and tl["errors"] == []
    assert rep["phases"]["serve_fp8"]["timeline"]["train"] == [] and c0 > T0


def jsonl_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.endswith("}")]


def test_dashboard_embeds_the_report(run):
    rep = rp.build(run)
    rep["run"] = "x</script><b>"                                           # cannot end the script tag
    page = rp.dashboard(rep)
    head = (HERE / "proto" / "dashboard.html").read_text()
    assert page.count("const EMBEDDED = ") == 1 and "const EMBEDDED = null;" in head and "const EMBEDDED = null;" not in page
    data, end = json.JSONDecoder().raw_decode(page.split("const EMBEDDED = ", 1)[1])
    raw = page.split("const EMBEDDED = ", 1)[1][:end]
    assert "</script>" not in raw and data == rep                         # JSON reads <\\/ back as </


def test_markdown_and_cli(run):
    md = rp.markdown(rp.build(run))
    assert "## co_timeslice: PASS" in md and "## co_mps: FAIL" in md and "OOM" in md and "hot-load" in md
    rows = [line for line in md.splitlines() if line.startswith("| ")]
    assert len({line.count("|") for line in rows}) == 1                      # every table row has 17 cells
    out = subprocess.run(["python", "-m", "proto.report", str(run)], cwd=HERE, capture_output=True, text=True)
    assert out.returncode == 0 and (run / "report.md").exists()
    assert json.loads((run / "report.json").read_text())["verdicts"]["co_timeslice"]["pass"] is True
    assert "const EMBEDDED = {" in (run / "dashboard.html").read_text()


def test_parsers_skip_bad_lines(tmp_path):
    (tmp_path / "x.jsonl").write_text('{"a": 1}\n{bad\n{"a": 2}\n')
    assert rp.jsonl(tmp_path / "x.jsonl") == [{"a": 1}, {"a": 2}] and rp.jsonl(tmp_path / "missing") == []
    (tmp_path / "d.txt").write_text("5 GPU 0 0.5 N/A 0.7 1 2\n6 GPU 0 bad\n")
    assert rp.dcgm_rows(tmp_path / "d.txt") == [{"t": 5.0, "sm_active": 0.5, "tensor_active": None,
                                                 "dram_active": 0.7, "pcie_tx": 1.0, "pcie_rx": 2.0}]


def test_scripts_parse_and_reference_real_things():
    proto = HERE / "proto"
    for sh in sorted(proto.glob("*.sh")) + [HERE / "setup.sh"]:
        assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0, sh
    selected = set((HERE.parent / "harbor" / "state" / "tb21_selected_tasks.txt").read_text().split())
    assert set((proto / "ec2_tasks.txt").read_text().split()) <= selected          # training tasks only
    bench = (proto / "bench.sh").read_text()
    phases = re.search(r'PHASES=\$\{PHASES:-"([^"]+)"', bench).group(1).split()
    for p in phases:                                     # every default phase is run somewhere
        assert re.search(rf'(has {p}\b|for t in .*\b{p}\b)', bench), p


def test_live_mode_shows_the_running_phase(run):
    t = T0 + 5000
    for name in ("serve_bf16", "co_mps70"):                  # serve_bf16 failed (no phase.json); co_mps70 is running
        (run / name).mkdir()
        (run / name / "gpu.csv").write_text(f"{t},90,80,100000,{TOTAL_MIB},600\n{t + 30},91,80,101000,{TOTAL_MIB},610\n")
    (run / "co_mps70" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_rows(t, 2, 10.0)))
    (run / "bench.log").write_text("[01:00:00] phase serve_bf16\n[01:00:01] SGLang failed\n[01:10:00] phase co_mps70\n")
    assert "co_mps70" not in rp.build(run)["phases"]
    rep = rp.build(run, live=True)
    assert "serve_bf16" not in rep["phases"] and list(rep["phases"])[-1] == "co_mps70"
    p = rep["phases"]["co_mps70"]
    assert p["meta"] == {"name": "co_mps70", "running": True, "t0": t, "t1": t + 30} and p["train"]["steps"] == 2
    assert "co_mps70" not in rep["verdicts"] and rep["live"] and rep["bench_log"][-1] == "[01:10:00] phase co_mps70"
    assert '<meta http-equiv="refresh" content="60">' in rp.dashboard(rep)
    assert "http-equiv" not in rp.dashboard(rp.build(run))
    (run / "co_mps70" / "phase.json").write_text(json.dumps({"name": "co_mps70", "t0": t, "t1": t + 30}))
    assert rp.running_phase(run) is None
