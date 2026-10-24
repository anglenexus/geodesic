"""Export passing Harbor trajectories as chat-format SFT data (post_training/PLAN.md §3d, D15).

Reads trials from the aees_harbor ledger: one model, the given task list, reward == 1. Rebuilds each
linear piece of history exactly as terminus-2 sent it to the model:
    user (first prompt), assistant (reply), user (terminal output), ..., assistant (last reply).
A trial recorded with `linear_history` that was summarized has several pieces (trajectory.json,
trajectory.cont-1.json, ...); each becomes one SFT row. The summarization calls themselves are skipped.
Thinking (`reasoning_content`) is dropped: SFT targets thinking-off mode (D13). Replies the model didn't
make in the task loop (copied hand-off context) or that drew a parse error stay as context but get
`step_loss_mask: 0`. Pieces longer than --max-tokens are dropped, never truncated.

    uv run python sft/export_trajectories.py --ledger ../harbor/state/ledger.sqlite \
        --model hosted_vllm/qwen3.5-27b --tasks ../harbor/state/tb21_selected_tasks.txt \
        --out data/sft_qwen35_27b.jsonl
"""
import argparse
import json
import sqlite3
from pathlib import Path

PARSE_ERROR_PREFIX = "Previous response had parsing errors"   # terminus-2's reply to an unparseable turn


def passing_trials(ledger: Path, model: str, tasks: set[str]) -> list[tuple[str, Path]]:
    """(task, trial_dir) for every trial of `model` on `tasks` with reward 1, in recording order."""
    con = sqlite3.connect(ledger)
    rows = con.execute(
        "SELECT s.task, s.fields_json, t.trial_dir, t.facts_json FROM trials t "
        "JOIN scenarios s USING (scenario_key) ORDER BY t.recorded_at"
    ).fetchall()
    con.close()
    out = []
    for task, fields, trial_dir, facts in rows:
        name = task.split("/")[-1]                     # "terminal-bench/mailman" -> "mailman"
        if name not in tasks or json.loads(fields).get("model") != model:
            continue
        if json.loads(facts).get("reward") == 1.0:
            out.append((name, Path(trial_dir)))
    return out


def pieces(agent_dir: Path) -> list[dict]:
    """The trial's linear-history pieces in order: trajectory.json, then each continued_trajectory_ref."""
    out, path = [], agent_dir / "trajectory.json"
    while path.exists():
        piece = json.loads(path.read_text())
        out.append(piece)
        ref = piece.get("continued_trajectory_ref")
        if not ref:
            break
        path = agent_dir / ref
    return out


def peak_tokens(steps: list[dict]) -> int:
    """Largest prompt + reply of any model call in the piece, from the counts the server recorded."""
    totals = [0]
    for step in steps:
        metrics = step.get("metrics") or {}
        totals.append((metrics.get("prompt_tokens") or 0) + (metrics.get("completion_tokens") or 0))
    return max(totals)


def to_messages(steps: list[dict]) -> list[dict]:
    """Rebuild the chat terminus-2 sent from one piece's ATIF steps (recorded with raw_content=true).

    Each agent step is a reply; its observation is the next user message. A reply that is copied hand-off
    context, or that the next message answers with a parse error, gets step_loss_mask 0.
    """
    messages = []
    for step in steps:
        if step["source"] == "user":
            messages.append({"role": "user", "content": step["message"]})
        elif step["source"] == "agent":
            reply = {"role": "assistant", "content": step["message"].strip()}
            if step.get("is_copied_context"):
                reply["step_loss_mask"] = 0
            messages.append(reply)
            results = [r for r in (step.get("observation") or {}).get("results", []) if "content" in r]
            if len(results) > 1:
                raise ValueError("several observations in one step: export needs raw_content=true trajectories")
            messages += [{"role": "user", "content": r["content"]} for r in results]
    for reply, answer in zip(messages, messages[1:]):
        if reply["role"] == "assistant" and answer["content"].startswith(PARSE_ERROR_PREFIX):
            reply["step_loss_mask"] = 0
    while messages and messages[-1]["role"] == "user":
        messages.pop()
    return messages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--model", required=True, help="model string as stored in the ledger, e.g. hosted_vllm/qwen3.5-27b")
    p.add_argument("--tasks", type=Path, required=True, help="file with one task name per line")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=49152, help="drop pieces longer than this (D10: 48k)")
    args = p.parse_args()

    tasks = set(args.tasks.read_text().split())
    trials = passing_trials(args.ledger, args.model, tasks)
    kept = too_long = untrainable = missing = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for task, trial_dir in trials:
            trial_pieces = pieces(trial_dir / "agent")
            missing += not trial_pieces
            for i, piece in enumerate(trial_pieces):
                messages, peak = to_messages(piece["steps"]), peak_tokens(piece["steps"])
                if peak > args.max_tokens:
                    too_long += 1
                elif not any(m["role"] == "assistant" and m.get("step_loss_mask", 1) for m in messages):
                    untrainable += 1
                else:
                    meta = {"task": task, "trial": trial_dir.name, "piece": i, "model": args.model,
                            "peak_tokens": peak}
                    f.write(json.dumps({"messages": messages, "metadata": meta}) + "\n")
                    kept += 1
    print(f"{len(trials)} passing trials -> {kept} rows in {args.out}; dropped: {too_long} pieces over "
          f"{args.max_tokens} tokens, {untrainable} with no trainable reply, {missing} trials without trajectory")


if __name__ == "__main__":
    main()
