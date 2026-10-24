"""Random-data tests for grpo/split_tasks.py: synthetic task lists and task folders."""
import json
import random
import sys

import pytest

from grpo import split_tasks as st


def make_setup(tmp_path, rng: random.Random, n=(20, 8, 20), overlap: str | None = None, missing: int = 0):
    """Writes three disjoint random task lists and their task folders; optionally breaks one rule."""
    names = rng.sample([f"task-{i:03d}" for i in range(500)], sum(n))
    sets = {"train": names[:n[0]], "val": names[n[0]:n[0] + n[1]], "test": names[n[0] + n[1]:]}
    if overlap:
        a, b = overlap.split("/")
        sets[b][0] = sets[a][0]
    state, tasks = tmp_path / "state", tmp_path / "tasks"
    state.mkdir()
    for name, file in st.LISTS.items():
        (state / file).write_text("\n".join(sets[name]) + "\n\n")
    for t in sorted({t for s in sets.values() for t in s})[missing:]:
        (tasks / t).mkdir(parents=True)
        (tasks / t / "task.toml").write_text("")
    return sets, state, tasks


def run(monkeypatch, state, tasks, out):
    monkeypatch.setattr(sys, "argv", ["x", "--tasks-dir", str(tasks), "--out-dir", str(out), "--state", str(state)])
    st.main()


@pytest.mark.parametrize("seed", range(20))
def test_writes_train_and_val_rows(tmp_path, monkeypatch, seed):
    rng = random.Random(seed)
    sets, state, tasks = make_setup(tmp_path, rng, n=(rng.randint(1, 30), rng.randint(1, 10), rng.randint(1, 30)))
    run(monkeypatch, state, tasks, tmp_path / "out")
    for name in ["train", "val"]:
        rows = [json.loads(line) for line in (tmp_path / "out" / f"grpo_{name}.jsonl").read_text().splitlines()]
        assert rows == [{"prompt": t, "metadata": {"instance_id": t, "agent_name": "terminus-2"}} for t in sets[name]]
    assert not (tmp_path / "out" / "grpo_test.jsonl").exists()


@pytest.mark.parametrize("overlap", ["train/val", "train/test", "val/test"])
def test_overlapping_sets_are_rejected(tmp_path, monkeypatch, overlap):
    _, state, tasks = make_setup(tmp_path, random.Random(0), overlap=overlap)
    with pytest.raises(AssertionError, match="overlap"):
        run(monkeypatch, state, tasks, tmp_path / "out")


def test_missing_task_folder_is_rejected(tmp_path, monkeypatch):
    _, state, tasks = make_setup(tmp_path, random.Random(0), missing=1)
    with pytest.raises(AssertionError, match="missing"):
        run(monkeypatch, state, tasks, tmp_path / "out")


def test_duplicate_in_a_list_is_rejected(tmp_path):
    (tmp_path / "l.txt").write_text("a\nb\na\n")
    with pytest.raises(AssertionError, match="duplicate"):
        st.read_list(tmp_path / "l.txt")


def test_real_task_lists_are_disjoint_and_sized():
    sets = {name: st.read_list(st.STATE / file) for name, file in st.LISTS.items()}
    assert [len(sets[k]) for k in ["train", "val", "test"]] == [20, 8, 20]
    assert len(set().union(*sets.values())) == 48
