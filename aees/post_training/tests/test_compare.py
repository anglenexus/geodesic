"""monitor/compare.py: per-step metrics from Miles' log lines, the vanilla file, and the served page."""
import importlib.util
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("compare", ROOT / "monitor" / "compare.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

LOG = """[2026-10-27 rank00000] log_utils.py:125 - rollout 0: {'rollout/raw_reward': 0.25, 'rollout/response_lengths': 15000.5, 'note': 'x'}
noise rollout 9: {'rollout/raw_reward': 9}
[..] model.py:882 - step 0: {'train/loss': 0.1}
[..] log_utils.py:554 - step 0: {'train/loss': 0.03, 'train/grad_norm': 0.02}
[..] log_utils.py:125 - rollout 1: {'rollout/raw_reward': 0.3125, 'train/opd_reverse_kl': 0.4}
[..] log_utils.py:125 - rollout 2: {broken
[..] log_utils.py:77 - step 3: {'perf/time': 5}
[..] perf.py:56 - perf 0: {'perf/rollout_time': 3000.5, 'perf/teacher_log_probs_time': 120.0}
"""


def test_parse_reads_miles_rollout_and_train_lines_only():
    steps = cmp.parse(LOG)
    assert steps[1] == {"rollout/raw_reward": 0.25, "rollout/response_lengths": 15000.5,
                        "train/loss": 0.03, "train/grad_norm": 0.02,
                        "perf/rollout_time": 3000.5, "perf/teacher_log_probs_time": 120.0}
    assert steps[2] == {"rollout/raw_reward": 0.3125, "train/opd_reverse_kl": 0.4}
    assert set(steps) == {1, 2}                              # unparsable and non-train 'step' lines skipped
    assert set(cmp.parse(LOG, first_step=7)) == {7, 8}


def test_vanilla_file_has_ten_steps_with_success():
    v = json.loads((ROOT / "monitor" / "vanilla_steps.json").read_text())
    assert list(v["steps"]) == [str(k) for k in range(1, 11)]
    assert all(0 <= s["rollout/raw_reward"] <= 1 for s in v["steps"].values())
    assert v["steps"]["1"]["rollout/raw_reward"] == 0.2375 and v["steps"]["10"]["rollout/raw_reward"] == 0.25
    assert all("perf/step_time" in s and "perf/train_wait_time" in s for s in v["steps"].values())


def serve(*args):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    p = subprocess.Popen([sys.executable, ROOT / "monitor" / "compare.py", "--port", str(port), *args],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(50):
        try:
            return p, port, json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/data.json").read())
        except OSError:
            time.sleep(0.1)
    p.terminate()
    raise AssertionError("compare.py did not serve")


def test_server_serves_page_and_data(tmp_path):
    p, port, d = serve("--aux-json", str(tmp_path / "none.json"))    # no host, no saved run
    try:
        assert len(d["vanilla"]["steps"]) == 10 and d["aux"]["steps"] == {} and d["aux"]["error"]
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "<title>Auxiliary Reward vs Vanilla</title>" in page and "rollout/raw_reward" in page
        assert "renderStages" in page and "perf/teacher_log_probs_time" in page
    finally:
        p.terminate()


def test_server_serves_the_saved_aux_run():
    p, _, d = serve()                                   # default --aux-json: auxr_1103_0033, all 10 steps
    try:
        assert d["aux"]["run"] == "auxr_1103_0033" and len(d["aux"]["steps"]) == 10 and not d["aux"]["error"]
        assert len(d["aux"]["rollout"]["steps"]) == 10 and all(st["match"] for st in d["aux"]["rollout"]["steps"])
        sft = d["pipeline"]["sft"]                      # the break-even section: SFT epoch measured in qfunc_1030_1950
        assert d["pipeline"]["sft_epochs"] == 2 and sft["steps"] == 93 and 3500 < sft["train_seconds"] < 4200
    finally:
        p.terminate()


def test_stage_timers_add_up_to_step_time():
    """The page's six stages: train_wait (all but the train block) = rollout + hand-off timers; train = teacher pass +
    policy log-probs + actor train + a few seconds (advantages, adapter swap); the two make perf/step_time."""
    for name in ("vanilla_steps.json", "aux_steps.json"):
        for k, m in json.loads((ROOT / "monitor" / name).read_text())["steps"].items():
            assert abs(m["perf/train_wait_time"] + m["perf/train_time"] - m["perf/step_time"]) < 1, (name, k)
            rest = m["perf/train_time"] - m.get("perf/teacher_log_probs_time", 0) - m["perf/log_probs_time"] \
                - m["perf/actor_train_time"]
            assert 0 <= rest < 10, (name, k, rest)
            assert ("perf/teacher_log_probs_time" in m) == (name == "aux_steps.json")


def trial(task, queued, end, reward=0.0, exit="Submitted"):
    return {"task": task, "queued": queued, "start": queued + 1, "end": end, "reward": reward, "exit_status": exit}


def test_rollout_ends_reads_rollout_line_times():
    text = ("[2026-11-03 10:07:50.317 actor_cell00000_rank00000] log_utils.py:125 - rollout 0: {'rollout/raw_reward': 0.2}\n"
            "[2026-11-03 10:07:51.000 x] log_utils.py:554 - step 0: {'train/loss': 0.1}\n")
    assert cmp.rollout_ends(text) == [1793700470]


def test_aborted_keys_from_trial_folders():
    assert cmp.aborted_keys(["dna-insert__20261103-170738__3kCjv3G", "a__b__20261103-000000__x"]) == \
        {("dna-insert", 1793725658), ("a__b", 1793664000)}


def test_select_replays_drop_resubmit_and_cancel():
    """2 tasks x groups of 2. Step 1: clean. Step 2: batch 1's task-a group has an aborted trial and is dropped, so a
    second batch is sent; its task-b group finishes first and fills the gap; its task-a group is flushed. Step 3 runs."""
    rows = [trial("a", 0, 100, 1.0), trial("a", 0, 110), trial("b", 0, 120, 1.0), trial("b", 0, 130, 1.0),
            trial("a", 1000, 1100), trial("a", 1000, 1050, 0.0, "AgentError"),    # step 2, batch 1: a aborted
            trial("b", 1000, 1200, 1.0), trial("b", 1000, 1210),
            trial("a", 1300, 1500, 1.0, "Flushed"), trial("a", 1300, 1500, 0.0, "Flushed"),   # batch 2
            trial("b", 1300, 1400), trial("b", 1300, 1450, 1.0),
            trial("a", 2000, 2100), trial("b", 2000, 2200, 1.0)]                 # step 3, in progress
    rows[5]["start"] = 1005
    aborted = {("a", 1005)}
    steps = cmp.select(rows, ends=[500, 1600], aborted=aborted, tasks=2, group=2)
    s1, s2, s3 = steps
    assert s1["sent"] == [4] and s1["selected"] == 2 and s1["passed"] == 3 and s1["closed"]
    assert s2["sent"] == [4, 4] and s2["selected"] == 2 and s2["passed"] == 2
    assert s2["groups"] == {"selected": 2, "dropped": 1, "unused": 0, "cancelled": 1, "running": 0}
    assert s2["trials"] == {"selected": 4, "usable": 1, "unusable": 3, "running": 0}
    assert sorted((g["task"], g["batch"], g["status"]) for g in s2["detail"]) == [("a", 1, "dropped"), ("a", 2, "cancelled")]
    assert [r["status"] for r in rows[4:6]] == ["usable", "unusable"]          # the aborted trial is unusable
    assert not s3["closed"] and s3["groups"]["running"] == 2 and s3["selected"] == 0
    assert s1["missing"] == [] and s1["doubled"] == [] and s1["per_task"]["a"]["selected"] == [1, 0]
    assert s2["missing"] == ["a"] and s2["doubled"] == ["b"]                   # the old refill: a out, b twice
    assert s2["per_task"]["a"] == {"selected": [], "dropped": 1, "selected_groups": 0, "running": []}
    assert s3["per_task"]["a"]["running"] == [0] and s3["per_task"]["b"]["running"] == [1]   # in progress, per trial
    assert s3["detail"] == []                                                  # running groups aren't "not trained on"
    assert s2["per_task"]["b"]["selected_groups"] == 2 and s3["missing"] == []  # open step: not judged yet


def test_only_an_errored_trial_matches_an_aborted_folder():
    """Trials of one task start in the same second; the Submitted one next to the aborted one stays usable."""
    rows = [trial("a", 0, 100, 1.0), trial("a", 0, 120, 0.0, "AgentError")]
    st = cmp.select(rows, ends=[500], aborted={("a", 1)}, tasks=1, group=2)[0]
    assert st["groups"]["dropped"] == 1 and [r["status"] for r in rows] == ["usable", "unusable"]


def test_trials_queued_before_a_restart_in_the_step_are_abandoned():
    """Step 2 was stopped mid-rollout and resumed from step 1's checkpoint: its first attempt never trained."""
    rows = [trial("a", 0, 100, 1.0), trial("a", 1000, 1100, 1.0), trial("a", 1000, 1200, 1.0),
            trial("a", 3000, 3100), trial("a", 3000, 3200, 1.0)]
    s1, s2 = cmp.select(rows, ends=[500], aborted=set(), tasks=1, group=2, starts=[-5, 2500])
    assert s1["abandoned"] == 0 and s1["sent"] == [1]
    assert s2["abandoned"] == 2 and s2["sent"] == [2] and s2["selected"] == 1 and s2["passed"] == 1
    assert [r["status"] for r in rows[1:3]] == ["abandoned", "abandoned"]
    text = "[2026-11-03 21:54:46.1 main] arguments.py:51 - --tokenizer-model not set, use --hf-checkpoint\n"
    assert cmp.restarts(text) == [1793742886]


def test_select_marks_groups_after_the_batch_filled_unused():
    rows = [trial("a", 0, 100), trial("b", 0, 300), trial("a", 30, 200, 1.0)]   # one burst; groups of 1
    st = cmp.select(rows, ends=[1000], aborted=set(), tasks=2, group=1)[0]
    assert st["sent"] == [3] and st["groups"]["selected"] == 2 and st["groups"]["unused"] == 1
    assert [r["status"] for r in rows] == ["selected", "usable", "selected"] and st["passed"] == 1


def test_rollout_summary_checks_against_miles_and_counts_live():
    rows = [trial("a", 0, 100, 1.0), trial("b", 0, 200)]
    status = {"boxes": {"x": {"running": 30, "free": 2}, "y": {"running": 32, "free": 0}}, "waiting": 7}
    r = cmp.rollout_summary(status, rows, [500], set(), {1: 0.5}, tasks=2, group=1)
    assert r["live"] == {"running": 62, "free": 2, "waiting": 7, "boxes": 2}
    assert r["steps"][0]["miles"] == 0.5 and r["steps"][0]["match"] is True
    assert cmp.rollout_summary(None, rows, [500], set(), {1: 0.25}, 2, 1)["steps"][0]["match"] is False
    assert cmp.rollout_summary(None, [], [], set(), {}) == {"tasks": 20, "group": 8, "live": None, "steps": []}
