"""Side-by-side tracker: the auxiliary-reward run against the vanilla GRPO run, per step (aees/reward/
auxiliary_reward_first_run_plan.md §6). Separate from monitor/server.py. Standard library only.

    python3 monitor/compare.py --build-vanilla             # once: monitor/vanilla_steps.json from the S3 train logs
    python3 monitor/compare.py --aux-host <user@h200> --aux-log /data/ckpt/<run>/train.log [--port 8790]
    python3 monitor/compare.py                             # a finished run: the aux block saved in --aux-json

Both runs are read from Miles' own per-step log lines, so the two curves use the same definitions:
    log_utils.py:N - rollout K: {'rollout/raw_reward': ..., ...}    rollout metrics of step K+1 (success = raw_reward)
    log_utils.py:N - step K: {'train/loss': ..., ...}               the update's metrics
    perf.py:N - perf K: {'perf/rollout_time': ..., ...}             seconds per phase (rollout, log-probs per pass,
                                                                    training, weight sync, sleep/wake), throughput
The vanilla run is three Miles runs in sequence (steps 1, 2-6, 7-10; Adam restarted at 2 and 7). The aux run's log is
fetched over ssh (only those lines) every --poll seconds, with the live rollout: the dispatcher's status (running,
waiting) and its per-trial rows ($DATA/trials/<run>/dispatch.jsonl: task, reward, exit status, queued/start/end).
select() replays how Miles picks each step's 20 groups of 8 out of what it sent (resubmitted batches, dropped and
cancelled groups) and the page checks the result against Miles' logged success. With --aux-host each refresh saves the
aux block to --aux-json (default monitor/aux_steps.json, auxr_1103_0033's); without it that file is served, so the
page outlives the H200. Serves monitor/compare.html with /data.json on 127.0.0.1.
"""
import argparse
import ast
import calendar
import json
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUCKET = "s3://aees-grpo-backups-611728564241"
VANILLA_RUNS = ("grpo10h_1026_1924", "grpo10h_r2", "grpoc_1027_2058")    # steps 1, 2-6, 7-10
LINE = re.compile(r"(?:log_utils|perf)\.py:\d+ - (rollout|step|perf) (\d+): (\{[^\n]*\})")
ROLLOUT_END = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]\n]*\] log_utils\.py:\d+ - rollout \d+: \{")
RESTART = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]\n]* main\] arguments\.py:\d+ - --tokenizer-model not set")


def parse(text: str, first_step: int = 1) -> dict[int, dict]:
    """step (1-based, from first_step) -> {metric: value}, from Miles' rollout and train lines."""
    steps: dict[int, dict] = {}
    for kind, k, body in LINE.findall(text):
        try:
            metrics = ast.literal_eval(body)
        except (ValueError, SyntaxError):
            continue
        if kind == "step" and not any(key.startswith("train/") for key in metrics):
            continue
        steps.setdefault(first_step + int(k), {}).update(
            {key: float(v) for key, v in metrics.items() if isinstance(v, (int, float))})
    return steps


def build_vanilla(out: Path) -> dict:
    steps, first = {}, 1
    for run in VANILLA_RUNS:
        text = subprocess.run(["aws", "s3", "cp", f"{BUCKET}/{run}/train.log", "-"], capture_output=True, text=True,
                              check=True).stdout
        part = parse(text, first)
        steps.update(part)
        first = max(part) + 1
    data = {"name": "vanilla GRPO", "runs": list(VANILLA_RUNS), "steps": {str(k): v for k, v in sorted(steps.items())}}
    out.write_text(json.dumps(data, indent=1))
    return data


def rollout_ends(text: str) -> list[float]:
    """Unix times of Miles' 'rollout K:' lines: rollout K (step K+1) closed then; step boundaries for the trial rows."""
    return [calendar.timegm(time.strptime(t, "%Y-%m-%d %H:%M:%S")) for t in ROLLOUT_END.findall(text)]


def restarts(text: str) -> list[float]:
    """Unix times Miles started (the run, and any resume from a checkpoint after a stop)."""
    return [calendar.timegm(time.strptime(t, "%Y-%m-%d %H:%M:%S")) for t in RESTART.findall(text)]


def aborted_keys(names: list[str]) -> set[tuple[str, int]]:
    """(task, start second) of trials Miles marks ABORTED: the agent made no model call (environment setup failed).
    names are trial folders '<task>__<YYYYmmdd-HHMMSS>__<id>' (UTC start), listed by the --aux-host command."""
    out = set()
    for n in names:
        task, stamp, _ = n.rsplit("__", 2)
        out.add((task, calendar.timegm(time.strptime(stamp, "%Y%m%d-%H%M%S"))))
    return out


def is_aborted(row: dict, aborted: set) -> bool:
    """The folder's stamp is the trial's start, to the second (allow one second either way). Several trials of one
    task start in the same second, so only a trial that ended in an error can be the aborted one."""
    if row.get("exit_status") in ("Submitted", "TimeLimitExceeded", "SequenceLengthLimitExceeded", "Flushed"):
        return False
    t = int(row.get("start") or 0)
    return any((row["task"], t + d) in aborted for d in (-1, 0, 1))


def select(rows: list[dict], ends: list[float], aborted: set, tasks: int = 20, group: int = 8,
           starts: list[float] = ()) -> list[dict]:
    """Replay Miles' choice of each step's batch (sglang_rollout.generate_rollout_async, check_no_aborted filter).

    A step submits tasks x group trials; Miles takes groups in completion order, drops a group with an ABORTED trial
    (and submits another tasks x group batch to replace it), keeps the first `tasks` clean groups, and cancels the
    trials still running ('Flushed'). Trials of step K were queued before rollout K's log line (ends[K-1]); a batch
    is a burst of queue times; a group is a batch's trials of one task (in queue order, `group` at a time).
    Group status: selected (trained on) | dropped (aborted trial) | unused (complete, after the batch filled) |
    cancelled (flushed) | running. Trial status: selected | usable (finished, not trained on) | unusable (aborted or
    flushed) | running | abandoned (queued before a restart inside the step's window: that attempt never trained).
    """
    steps, edges = [], [float("-inf"), *ends, float("inf")]
    for k in range(1, len(edges)):
        rs = sorted((r for r in rows if edges[k - 1] < r["queued"] <= edges[k]), key=lambda r: r["queued"])
        restart = max((t for t in starts if edges[k - 1] < t <= edges[k]), default=None)
        abandoned = [r for r in rs if restart is not None and r["queued"] < restart]
        for r in abandoned:
            r.update(step=k, batch=0, status="abandoned", group_status="abandoned")
        rs = [r for r in rs if r not in abandoned]
        if not rs and not abandoned:
            continue
        batches = []
        for r in rs:
            if not batches or r["queued"] - batches[-1][-1]["queued"] > 60:
                batches.append([])
            batches[-1].append(r)
        groups = []
        for b, batch in enumerate(batches):
            by_task = {}
            for r in batch:
                by_task.setdefault(r["task"], []).append(r)
            for task, trs in by_task.items():
                for i in range(0, len(trs), group):
                    groups.append({"task": task, "batch": b + 1, "trials": trs[i:i + group]})
        closed = k <= len(ends)
        for g in groups:
            trs = g["trials"]
            if any(r.get("exit_status") == "Flushed" for r in trs):
                g["status"] = "cancelled"
            elif any(is_aborted(r, aborted) for r in trs):
                g["status"] = "dropped"
            elif len(trs) < group:
                g["status"] = "cancelled" if closed else "running"
        kept = 0
        for g in sorted(groups, key=lambda g: max(r["end"] for r in g["trials"])):
            if "status" not in g:
                g["status"] = "selected" if kept < tasks else "unused"
                kept += g["status"] == "selected"
        for g in groups:
            for r in g["trials"]:
                bad = r.get("exit_status") == "Flushed" or is_aborted(r, aborted)
                r["status"] = ("selected" if g["status"] == "selected" else "unusable" if bad
                               else "running" if g["status"] == "running" else "usable")
                r.update(step=k, batch=g["batch"], group_status=g["status"])
            g["passed"] = sum((r.get("reward") or 0) >= 1 for r in g["trials"])
        sel = [g for g in groups if g["status"] == "selected"]
        per_task = {}                                    # the trajectory check: each task's selected 8, and reruns
        for g in sorted(groups, key=lambda g: g["batch"]):
            t = per_task.setdefault(g["task"], {"selected": [], "dropped": 0, "selected_groups": 0, "running": []})
            t["dropped"] += g["status"] == "dropped"
            if g["status"] == "running":                 # finished trials of a group still running
                t["running"] += [1 if (r.get("reward") or 0) >= 1 else 0 for r in g["trials"]]
            if g["status"] == "selected":
                t["selected_groups"] += 1
                t["selected"] += [1 if (r.get("reward") or 0) >= 1 else 0 for r in g["trials"]]
                t["batch"] = g["batch"]
        count = lambda key, v: sum(1 for x in (groups if key == "g" else rs) if x["status"] == v)
        steps.append({"step": k, "closed": closed, "sent": [len(b) for b in batches], "abandoned": len(abandoned),
                      "selected": len(sel), "passed": sum(g["passed"] for g in sel),
                      "groups": {s: count("g", s) for s in ("selected", "dropped", "unused", "cancelled", "running")},
                      "trials": {s: count("t", s) for s in ("selected", "usable", "unusable", "running")},
                      "per_task": per_task,
                      "missing": sorted(t for t, v in per_task.items() if not v["selected_groups"]) if closed or
                                 len(sel) == tasks else [],
                      "doubled": sorted(t for t, v in per_task.items() if v["selected_groups"] > 1),
                      "detail": [{"task": g["task"], "batch": g["batch"], "status": g["status"], "passed": g["passed"],
                                  "done": sum(r.get("exit_status") != "Flushed" for r in g["trials"])}
                                 for g in groups if g["status"] not in ("selected", "running")]})
    return steps


def rollout_summary(status: dict | None, rows: list[dict], ends: list[float], aborted: set,
                    success: dict[int, float], tasks: int = 20, group: int = 8, starts: list[float] = ()) -> dict:
    """The selection per step, checked against Miles' logged success (step -> rollout/raw_reward), plus live counts."""
    steps = select(rows, ends, aborted, tasks, group, starts)
    for st in steps:
        st["miles"] = success.get(st["step"])
        st["match"] = None if st["miles"] is None else abs(st["passed"] / (tasks * group) - st["miles"]) < 1e-9
    live = None
    if status:
        boxes = status.get("boxes", {})
        live = {"running": sum(b.get("running", 0) for b in boxes.values()),
                "free": sum(b.get("free", 0) for b in boxes.values()),
                "waiting": status.get("waiting", 0), "boxes": len(boxes)}
    return {"tasks": tasks, "group": group, "live": live, "steps": steps}


class Aux:
    def __init__(self, host: str, log: str, run: str, records: Path | None = None):
        self.host, self.log, self.run, self.records = host, log, run, records
        self.steps, self.rollout, self.updated, self.error = {}, None, None, None

    def refresh(self) -> None:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-x", self.host,
               f"grep -E '(log_utils|perf).py:[0-9]+ - (rollout|step|perf) [0-9]+: \\{{|main\\] arguments.py:[0-9]+ - "
               f"--tokenizer-model not set' {self.log} || true"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            self.error = r.stderr.strip()[-300:]
            return
        self.steps, self.updated, self.error = parse(r.stdout), time.time(), None
        ends, starts = rollout_ends(r.stdout), restarts(r.stdout)
        trials = f"/data/trials/{self.run}"
        # status line, then '##' + the aborted trials' folders (an exception and no agent trajectory), then the rows
        cmd[-1] = (f"curl -s -m 10 127.0.0.1:11000/status || echo '{{}}'; echo; cd {trials} && "
                   f"for f in $(grep -l '\"exception_type\"' */result.json 2>/dev/null); do d=${{f%/result.json}}; "
                   f"[ -e $d/agent/trajectory.json ] || echo \"##$d\"; done; cat dispatch.jsonl 2>/dev/null || true")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            lines = r.stdout.splitlines()
            status = json.loads(lines[0]) if lines and lines[0].startswith("{") else None
            aborted = aborted_keys([x[2:] for x in lines[1:] if x.startswith("##")])
            rows = [json.loads(x) for x in lines[1:] if x.startswith("{")]
            success = {k: m["rollout/raw_reward"] for k, m in self.steps.items() if "rollout/raw_reward" in m}
            self.rollout = rollout_summary(status or None, rows, ends, aborted, success, starts=starts)
            if self.records:                             # every trial, with its step, batch and selection status
                self.records.write_text("".join(json.dumps(x) + "\n" for x in rows))


def aux_block(aux: Aux) -> dict:
    return {"name": "GRPO + auxiliary reward (β·A^Q)", "run": aux.run,
            "steps": {str(k): v for k, v in sorted(aux.steps.items())},
            "updated": aux.updated, "error": aux.error, "rollout": aux.rollout}


def payload(vanilla: dict, aux: Aux | None, saved: dict | None = None) -> dict:
    block = aux_block(aux) if aux else saved or {"name": "GRPO + auxiliary reward (β·A^Q)", "steps": {},
                                                  "updated": None, "error": "no --aux-host", "rollout": None}
    pipeline = HERE / "pipeline.json"                    # the SFT / Q-function run's costs, for the pipeline comparison
    return {"vanilla": vanilla, "aux": block, "now": time.time(),
            "pipeline": json.loads(pipeline.read_text()) if pipeline.exists() else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-vanilla", action="store_true")
    p.add_argument("--vanilla", type=Path, default=HERE / "vanilla_steps.json")
    p.add_argument("--aux-host")
    p.add_argument("--aux-log")
    p.add_argument("--aux-run", help="run name for the trial rows (default: the log's folder name)")
    p.add_argument("--records", type=Path, help="write every aux trial with its selection status here (jsonl)")
    p.add_argument("--aux-json", type=Path, default=HERE / "aux_steps.json",
                   help="with --aux-host: saved after each refresh; without: the finished run served from here")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--poll", type=float, default=60)
    args = p.parse_args()
    if args.build_vanilla:
        data = build_vanilla(args.vanilla)
        print(f"wrote {args.vanilla}: steps {list(data['steps'])}")
        return
    vanilla = json.loads(args.vanilla.read_text())
    aux = Aux(args.aux_host, args.aux_log, args.aux_run or Path(args.aux_log).parent.name, args.records) \
        if args.aux_host else None
    saved = None if aux or not args.aux_json.exists() else json.loads(args.aux_json.read_text())

    def loop():
        while aux:
            try:
                aux.refresh()
                if aux.updated and not aux.error:        # a good read: keep it for when the host is gone
                    args.aux_json.write_text(json.dumps(aux_block(aux), indent=1))
            except Exception as e:                       # keep serving the last good data
                aux.error = repr(e)
            time.sleep(args.poll)

    threading.Thread(target=loop, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data.json"):
                body, kind = json.dumps(payload(vanilla, aux, saved)).encode(), "application/json"
            else:
                body, kind = (HERE / "compare.html").read_bytes(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"compare tracker: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
