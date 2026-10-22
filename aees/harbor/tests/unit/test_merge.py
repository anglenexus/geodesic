import dataclasses
import shutil

import pytest

from aees_harbor.ledger import Ledger, Scenario
from aees_harbor.merge import merge

A = Scenario(dataset="tb", dataset_ref="r", task="task-a", task_checksum="c", model="m",
             agent="terminus-2", agent_version="2.0.0", tool_mode="t", harbor_version="0.23.0")
B = dataclasses.replace(A, task="task-b")
F = {"reward": 1.0}


def _run(led, sc, trial_dir):
    key = led.register(sc, 2000)
    [i] = led.lease_next(key, 1)
    led.record_trial(key, i, trial_dir, F)
    return key, i


def _seed_and_split(tmp_path):
    """Mac seed with 1 run of A and B, copied to two instances; i1 runs A, i2 runs B."""
    seed = tmp_path / "seed.sqlite"
    led = Ledger(seed)
    _run(led, A, "/Users/me/jobs/b1/a__1")
    _run(led, B, "/Users/me/jobs/b1/b__1")
    led.close()
    paths = []
    for n, sc in ((1, A), (2, B)):
        p = tmp_path / f"i{n}.sqlite"
        shutil.copy(seed, p)
        led = Ledger(p)
        for k in range(3):
            _run(led, sc, f"/home/ubuntu/jobs/i{n}_b1/{sc.task}__{k}")
        led.close()
        paths.append(p)
    return seed, paths


def test_disjoint_tasks_from_shared_seed_merge_cleanly(tmp_path):
    seed, (i1, i2) = _seed_and_split(tmp_path)
    out = tmp_path / "merged.sqlite"
    shutil.copy(seed, out)
    report = merge(out, [i1, i2], [("/home/ubuntu/jobs", "/Users/me/jobs")])
    assert report["trials"] == {"inserted": 6, "identical": 4}  # 3 new per instance; seed rows seen twice
    led = Ledger(out)
    summary = {s["fields"]["task"]: s["held"] for s in led.scenarios()}
    assert summary == {"task-a": [1, 2, 3, 4], "task-b": [1, 2, 3, 4]}  # indices continue after the seed
    dirs = {t["trial_dir"] for s in led.scenarios() for t in led.trial_facts(s["key"])}
    assert "/Users/me/jobs/i1_b1/task-a__0" in dirs and not any(d.startswith("/home/ubuntu") for d in dirs)


def test_remerging_is_a_no_op(tmp_path):
    seed, sources = _seed_and_split(tmp_path)
    out = tmp_path / "merged.sqlite"
    merge(out, sources)
    again = merge(out, sources)
    assert all(v["inserted"] == 0 for v in again.values())


def test_overlapping_tasks_are_refused_and_nothing_is_written(tmp_path):
    seed = tmp_path / "seed.sqlite"
    Ledger(seed).close()
    i1, i2 = tmp_path / "i1.sqlite", tmp_path / "i2.sqlite"
    for p, d in ((i1, "/x/i1/a__0"), (i2, "/x/i2/a__0")):  # both instances ran task A -> both used index 1
        shutil.copy(seed, p)
        led = Ledger(p)
        _run(led, A, d)
        led.close()
    out = tmp_path / "merged.sqlite"
    with pytest.raises(ValueError, match="conflicting"):
        merge(out, [i1, i2])
    assert Ledger(out).scenarios() == []  # rolled back entirely
