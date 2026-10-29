"""Agent-load profile from real Harbor trials, replayed by agent_load.py (PLAN.md §5b). Runs on the laptop.

  uv run python -m proto.make_profile ~/code/jobs/tb21_qwen4b_k20 ~/code/jobs/tb21_n1024_mac_b1 -o proto/load_profile.json
Input: Harbor job folders (<trial>/agent/trajectory.json from terminus-2).
Output: {"sources": [...], "trials": [{"task", "prompt0", "turns": [[out, obs, env_s], ...]}]}, per turn:
  out    the call's completion tokens
  obs    tokens added to the next prompt beyond this prompt + output (the terminal output fed back)
  env_s  the turn's command durations summed (terminus-2 waits that long before reading the screen),
         plus ENV_OVERHEAD_S for sending keys and capturing the pane
Only token counts and durations are kept, no text.
"""
import argparse
import json
import statistics
from pathlib import Path

ENV_OVERHEAD_S = 0.5


def command_seconds(message) -> float:
    """Sum of "duration" over the commands of a terminus-2 JSON reply; 0 if it didn't parse."""
    try:
        cmds = json.loads(message).get("commands") or []
        return sum(float(c.get("duration") or 0) for c in cmds if isinstance(c, dict))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def trial_turns(traj: dict):
    """(first prompt tokens, [[out, obs, env_s], ...]) for one trajectory, or None without usable calls."""
    agent = [s for s in traj.get("steps", []) if s.get("source") == "agent"]
    metrics = [s.get("metrics") or {} for s in agent]
    prompts = [m.get("prompt_tokens") or 0 for m in metrics]
    outs = [m.get("completion_tokens") or 0 for m in metrics]
    if not agent or not prompts[0]:
        return None
    turns = []
    for i, step in enumerate(agent):
        nxt = prompts[i + 1] if i + 1 < len(agent) else 0
        obs = max(0, nxt - prompts[i] - outs[i]) if nxt else 0
        turns.append([max(1, outs[i]), obs, round(command_seconds(step.get("message")) + ENV_OVERHEAD_S, 2)])
    return prompts[0], turns


def build(job_dirs) -> dict:
    trials = []
    for job in job_dirs:
        for path in sorted(Path(job).expanduser().glob("*/agent/trajectory.json")):
            try:
                got = trial_turns(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
            if got:
                trials.append({"task": path.parent.parent.name.split("__")[0], "prompt0": got[0], "turns": got[1]})
    return {"sources": [Path(j).name for j in job_dirs], "trials": trials}


def summary(profile: dict) -> str:
    t = profile["trials"]
    turns = [x for tr in t for x in tr["turns"]]
    if not turns:
        return "no trials"
    q = lambda v, f: sorted(v)[int(f * (len(v) - 1))]
    n, out, env = [len(tr["turns"]) for tr in t], [x[0] for x in turns], [x[2] for x in turns]
    return (f"{len(t)} trials, {len(turns)} turns; turns/trial p50 {q(n, .5)} p90 {q(n, .9)}; "
            f"output tokens/turn mean {statistics.mean(out):.0f} p90 {q(out, .9)}; env s/turn p50 {q(env, .5)} p90 {q(env, .9)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", nargs="+")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    prof = build(a.jobs)
    Path(a.out).write_text(json.dumps(prof, separators=(",", ":")))
    print(summary(prof))
