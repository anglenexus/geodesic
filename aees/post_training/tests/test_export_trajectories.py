"""Random-data tests for sft/export_trajectories.py: synthetic ledger + ATIF trajectory files."""
import json
import random
import sqlite3
import sys

import pytest

from sft import export_trajectories as ex

MODEL = "hosted_vllm/qwen3.5-27b"


def random_text(rng: random.Random, n: int = 12) -> str:
    return " ".join(rng.choice(["ls", "cat", "/app", "ok", "error", "{", "}", "x=1", "\n"]) for _ in range(n))


def agent_step(rng: random.Random, reply: str, observation: str | None, copied: bool = False) -> dict:
    step = {"source": "agent", "message": reply, "reasoning_content": random_text(rng)}
    if observation is not None:
        step["observation"] = {"results": [{"content": observation}]}
    if copied:
        step["is_copied_context"] = True
    else:
        step["metrics"] = {"prompt_tokens": rng.randint(100, 5000), "completion_tokens": rng.randint(10, 500)}
    return step


def random_piece(rng: random.Random, continuation: bool) -> tuple[list[dict], list[dict]]:
    """ATIF steps of one piece, and the chat messages the exporter should rebuild from them."""
    first = random_text(rng)
    steps, expected = [{"source": "user", "message": first}], [{"role": "user", "content": first}]
    if continuation:  # original prompt, hand-off question prompt, copied model questions, hand-off answers
        q, mq, ans = random_text(rng), random_text(rng), random_text(rng)
        steps += [{"source": "user", "message": q, "is_copied_context": True},
                  agent_step(rng, mq, None, copied=True), {"source": "user", "message": ans}]
        expected += [{"role": "user", "content": q}, {"role": "assistant", "content": mq.strip(), "step_loss_mask": 0},
                     {"role": "user", "content": ans}]
    for _ in range(rng.randint(1, 6)):
        reply = random_text(rng)
        bad = rng.random() < 0.3
        obs = (ex.PARSE_ERROR_PREFIX + ":\n" + random_text(rng)) if bad else random_text(rng)
        steps.append(agent_step(rng, "\n\n" + reply + "\n", obs))
        expected.append({"role": "assistant", "content": reply.strip(), **({"step_loss_mask": 0} if bad else {})})
        expected.append({"role": "user", "content": obs})
    expected.pop()  # the last observation is never sent back to the model
    return steps, expected


def write_trial(rng: random.Random, trial_dir, n_pieces: int) -> list[list[dict]]:
    agent = trial_dir / "agent"
    agent.mkdir(parents=True)
    expected_all = []
    for i in range(n_pieces):
        steps, expected = random_piece(rng, continuation=i > 0)
        if i < n_pieces - 1:
            steps.append({"source": "system", "message": "Performed context summarization and handoff."})
        doc = {"schema_version": "ATIF-v1.8", "steps": steps}
        if i < n_pieces - 1:
            doc["continued_trajectory_ref"] = f"trajectory.cont-{i + 1}.json"
        name = "trajectory.json" if i == 0 else f"trajectory.cont-{i}.json"
        (agent / name).write_text(json.dumps(doc))
        (agent / f"trajectory.summarization-{i + 1}-summary.json").write_text(json.dumps({"steps": []}))
        expected_all.append(expected)
    return expected_all


def make_ledger(path, rows: list[tuple[str, str, str, float]]) -> None:
    """rows: (task, model, trial_dir, reward)."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE scenarios (scenario_key TEXT PRIMARY KEY, fields_json TEXT, task TEXT)")
    con.execute("CREATE TABLE trials (scenario_key TEXT, trial_dir TEXT, facts_json TEXT, recorded_at REAL)")
    for i, (task, model, trial_dir, reward) in enumerate(rows):
        con.execute("INSERT INTO scenarios VALUES (?, ?, ?)", (f"s{i}", json.dumps({"model": model}), f"terminal-bench/{task}"))
        con.execute("INSERT INTO trials VALUES (?, ?, ?, ?)", (f"s{i}", trial_dir, json.dumps({"reward": reward}), i))
    con.commit()
    con.close()


@pytest.mark.parametrize("seed", range(20))
def test_to_messages_rebuilds_chat(seed):
    rng = random.Random(seed)
    steps, expected = random_piece(rng, continuation=rng.random() < 0.5)
    assert ex.to_messages(steps) == expected


def test_to_messages_rejects_split_observations():
    step = {"source": "agent", "message": "{}", "observation": {"results": [{"content": "a"}, {"content": "b"}]}}
    with pytest.raises(ValueError):
        ex.to_messages([{"source": "user", "message": "hi"}, step])


def test_pieces_follow_the_chain_past_ten(tmp_path):
    expected = write_trial(random.Random(0), tmp_path / "t", n_pieces=12)
    got = [ex.to_messages(p["steps"]) for p in ex.pieces(tmp_path / "t" / "agent")]
    assert got == expected


def test_peak_tokens():
    steps = [{"source": "user"}, {"source": "agent", "metrics": {"prompt_tokens": 10, "completion_tokens": 5}},
             {"source": "agent", "metrics": {"prompt_tokens": 30}}, {"source": "agent"}]
    assert ex.peak_tokens(steps) == 30
    assert ex.peak_tokens([]) == 0


def test_passing_trials_filters_model_task_reward(tmp_path):
    rows = [("mailman", MODEL, "/a", 1.0), ("mailman", MODEL, "/b", 0.0), ("mailman", "openai/gpt-5.6-luna", "/c", 1.0),
            ("fix-git", MODEL, "/d", 1.0), ("dna-insert", MODEL, "/e", 1.0)]
    make_ledger(tmp_path / "l.sqlite", rows)
    got = ex.passing_trials(tmp_path / "l.sqlite", MODEL, {"mailman", "dna-insert"})
    assert [(t, str(d)) for t, d in got] == [("mailman", "/a"), ("dna-insert", "/e")]


def test_main_end_to_end(tmp_path, monkeypatch):
    rng = random.Random(7)
    rows, expected_rows = [], []
    for i in range(8):
        trial = tmp_path / "jobs" / f"mailman__{i}"
        expected = write_trial(rng, trial, n_pieces=rng.randint(1, 3))
        rows.append(("mailman", MODEL, str(trial), 1.0))
        expected_rows += [e for e in expected if any(m["role"] == "assistant" and m.get("step_loss_mask", 1) for m in e)]
    rows.append(("mailman", MODEL, str(tmp_path / "jobs" / "deleted"), 1.0))   # no trajectory on disk
    make_ledger(tmp_path / "l.sqlite", rows)
    (tmp_path / "tasks.txt").write_text("mailman\n")
    out = tmp_path / "out" / "sft.jsonl"
    argv = ["x", "--ledger", str(tmp_path / "l.sqlite"), "--model", MODEL, "--tasks", str(tmp_path / "tasks.txt"),
            "--out", str(out), "--max-tokens", "100000"]
    monkeypatch.setattr(sys, "argv", argv)
    ex.main()
    got = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["messages"] for r in got] == expected_rows
    assert all(r["metadata"]["model"] == MODEL and r["messages"][-1]["role"] == "assistant" for r in got)

    monkeypatch.setattr(sys, "argv", argv[:-1] + ["1"])   # every piece is longer than 1 token
    ex.main()
    assert out.read_text() == ""
