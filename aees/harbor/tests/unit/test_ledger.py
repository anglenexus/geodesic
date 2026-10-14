import dataclasses
import time

import pytest

from aees_harbor.ledger import Ledger, Scenario

BASE = Scenario(
    dataset="terminal-bench/terminal-bench-2-1", dataset_ref="sha256:7d7b", task="extract-elf",
    task_checksum="abc", model="openai/gpt-5.6-luna", agent="terminus-2", agent_version="2.0.0",
    tool_mode="terminus-json-keystrokes", harbor_version="0.23.0",
    agent_kwargs={"store_all_messages": True},
)


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.sqlite", max_infra_retries=3)
    yield led
    led.close()


def test_scenario_key_changes_with_every_field():
    keys = {BASE.key}
    for f in dataclasses.fields(Scenario):
        value = {"x": 1} if f.name == "agent_kwargs" else "changed"
        keys.add(dataclasses.replace(BASE, **{f.name: value}).key)
    assert len(keys) == len(dataclasses.fields(Scenario)) + 1


def test_leases_start_at_one_and_are_sequential(ledger):
    key = ledger.register(BASE, target_runs=2000)
    assert ledger.lease_next(key, 3) == [1, 2, 3]
    assert ledger.lease_next(key, 2) == [4, 5]


def test_after_run_20_passes_next_session_starts_at_21(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    key = led.register(BASE, target_runs=2000)
    for i in led.lease_next(key, 20):
        led.record_outcome(key, i, reward=1.0 if i == 20 else 0.0)
    led.close()
    reopened = Ledger(tmp_path / "l.sqlite")  # a new session reads the saved cache
    assert reopened.summary(key)["passed"] == [20]
    assert reopened.lease_next(key, 1) == [21]


def test_second_outcome_for_same_index_is_rejected(ledger):
    key = ledger.register(BASE, target_runs=10)
    [i] = ledger.lease_next(key, 1)
    assert ledger.record_outcome(key, i, 1.0) is True
    assert ledger.record_outcome(key, i, 0.0) is False
    assert ledger.summary(key)["passed"] == [1] and ledger.summary(key)["failed"] == []


def test_infra_error_retries_the_same_index(ledger):
    key = ledger.register(BASE, target_runs=10)
    assert ledger.lease_next(key, 1) == [1]
    assert ledger.record_infra(key, 1, "DockerBuildError") == 1
    assert ledger.lease_next(key, 1) == [1]


def test_index_is_abandoned_after_max_infra_retries(ledger):
    key = ledger.register(BASE, target_runs=10)
    for _ in range(3):
        ledger.lease_next(key, 1)
        ledger.record_infra(key, 1, "NetworkDrop")
    assert ledger.lease_next(key, 1) == [2]
    assert ledger.summary(key)["abandoned"] == [1]


def test_target_caps_leases_and_counts_graded_runs(ledger):
    key = ledger.register(BASE, target_runs=3)
    assert ledger.lease_next(key, 10) == [1, 2, 3]
    for i in (1, 2, 3):
        ledger.record_outcome(key, i, 0.0)
    assert ledger.lease_next(key, 10) == []
    assert ledger.summary(key)["remaining"] == 0


def test_stale_lease_is_released_after_crash(ledger):
    key = ledger.register(BASE, target_runs=10)
    ledger.lease_next(key, 2)
    time.sleep(0.01)
    assert ledger.release_stale(max_age_s=0.0) == 2
    assert ledger.lease_next(key, 1) == [1]


def test_frozen_scenario_leases_nothing(ledger):
    key = ledger.register(BASE, target_runs=10)
    ledger.freeze(key)
    assert ledger.lease_next(key, 5) == []
    assert ledger.summary(key)["frozen"] is True


def test_changed_setting_gets_its_own_counter(ledger):
    a = ledger.register(BASE, target_runs=10)
    b = ledger.register(dataclasses.replace(BASE, tool_mode="openai-native-functions"), 10)
    ledger.lease_next(a, 5)
    assert ledger.lease_next(b, 1) == [1]


def test_reregister_keeps_runs_and_only_grows_target(ledger):
    key = ledger.register(BASE, target_runs=10)
    ledger.record_outcome(key, *ledger.lease_next(key, 1), 1.0)
    ledger.register(BASE, target_runs=5)
    assert ledger.summary(key)["target"] == 10 and ledger.summary(key)["passed"] == [1]


def test_unknown_scenario_raises(ledger):
    with pytest.raises(KeyError):
        ledger.lease_next("nope", 1)


def test_summary_next_index_fills_gap_after_released_lease(ledger):
    key = ledger.register(BASE, target_runs=10)
    for i in ledger.lease_next(key, 3):
        if i != 2:
            ledger.record_outcome(key, i, 1.0)
    ledger.release_stale(max_age_s=-1)  # index 2's runner died
    assert ledger.summary(key)["next_index"] == 2
    assert ledger.summary(key)["passed"] == [1, 3]


def test_concurrent_runners_never_lease_the_same_index(tmp_path):
    import threading
    path = tmp_path / "shared.sqlite"
    setup = Ledger(path)
    key = setup.register(BASE, target_runs=400)
    setup.close()
    got, errors = [], []

    def runner():
        led = Ledger(path)  # each runner has its own connection, like separate processes
        try:
            while idx := led.lease_next(key, 5):
                got.extend(idx)
        except Exception as e:  # pragma: no cover - surfaced by the assert below
            errors.append(e)
        finally:
            led.close()

    threads = [threading.Thread(target=runner) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert sorted(got) == list(range(1, 401))  # every index exactly once, none duplicated


def test_failed_transaction_rolls_back_leases(ledger):
    key = ledger.register(BASE, target_runs=10)
    def boom(*a, **k):
        raise RuntimeError("disk error mid-lease")

    ledger._db = _Proxy(ledger._db, executemany=boom)
    with pytest.raises(RuntimeError):
        ledger.lease_next(key, 3)
    ledger._db = ledger._db._inner
    assert ledger.summary(key)["leased"] == []  # nothing half-written
    assert ledger.lease_next(key, 1) == [1]


class _Proxy:
    """Wraps a sqlite3 connection and overrides selected methods (sqlite3 objects are immutable)."""

    def __init__(self, inner, **overrides):
        self._inner, self._overrides = inner, overrides

    def __getattr__(self, name):
        return self._overrides.get(name) or getattr(self._inner, name)


# ---- held / review workflow -------------------------------------------------------------

FACTS = {"reward": 0.0, "exception_type": None, "test_report": False, "platform": "arm64-rosetta"}


def test_finished_trial_is_held_not_counted_and_not_rerun(ledger):
    key = ledger.register(BASE, target_runs=10)
    [i] = ledger.lease_next(key, 1)
    assert ledger.record_trial(key, i, "jobs/j1/t1", FACTS) == 1
    s = ledger.summary(key)
    assert s["held"] == [1] and s["passed"] == [] and s["remaining"] == 10
    assert ledger.lease_next(key, 1) == [2]


def test_held_indices_occupy_slots_so_target_is_not_overshot(ledger):
    key = ledger.register(BASE, target_runs=3)
    for i in ledger.lease_next(key, 3):
        ledger.record_trial(key, i, f"jobs/j1/t{i}", FACTS)
    assert ledger.lease_next(key, 5) == []


def test_resolve_pass_is_final_and_cannot_be_changed(ledger):
    key = ledger.register(BASE, target_runs=10)
    [i] = ledger.lease_next(key, 1)
    ledger.record_trial(key, i, "jobs/j1/t1", FACTS)
    assert ledger.resolve(key, i, "pass") is True
    assert ledger.resolve(key, i, "fail") is False
    assert ledger.record_outcome(key, i, 0.0) is False
    assert ledger.summary(key)["passed"] == [1] and ledger.summary(key)["held"] == []


def test_resolve_retry_reruns_same_index_and_keeps_both_trials(ledger):
    key = ledger.register(BASE, target_runs=10)
    [i] = ledger.lease_next(key, 1)
    ledger.record_trial(key, i, "jobs/j1/t1", FACTS)
    assert ledger.resolve(key, i, "retry", reason="network drop") is True
    assert ledger.lease_next(key, 1) == [1]
    assert ledger.record_trial(key, 1, "jobs/j2/t1", {**FACTS, "reward": 1.0}) == 2
    assert [t["attempt"] for t in ledger.trial_facts(key, 1)] == [1, 2]
    assert ledger.trial_facts(key, 1)[0]["facts"] == FACTS


def test_record_trial_is_idempotent_per_trial_dir(ledger):
    key = ledger.register(BASE, target_runs=10)
    [i] = ledger.lease_next(key, 1)
    assert ledger.record_trial(key, i, "jobs/j1/t1", FACTS) == 1
    assert ledger.record_trial(key, i, "jobs/j1/t1", FACTS) == 1
    assert len(ledger.trial_facts(key)) == 1


def test_record_outcome_finalizes_a_held_index(ledger):
    key = ledger.register(BASE, target_runs=10)
    [i] = ledger.lease_next(key, 1)
    ledger.record_trial(key, i, "jobs/j1/t1", FACTS)
    assert ledger.record_outcome(key, i, 1.0) is True
    assert ledger.summary(key)["passed"] == [1]


def test_record_trial_never_downgrades_a_final_outcome(ledger):
    key = ledger.register(BASE, target_runs=10)
    [i] = ledger.lease_next(key, 1)
    ledger.record_outcome(key, i, 1.0)
    ledger.record_trial(key, i, "jobs/late/t1", FACTS)
    assert ledger.summary(key)["passed"] == [1] and ledger.summary(key)["held"] == []


def test_resolve_rejects_unknown_decisions_and_non_held_indices(ledger):
    key = ledger.register(BASE, target_runs=10)
    with pytest.raises(ValueError):
        ledger.resolve(key, 1, "maybe")
    assert ledger.resolve(key, 1, "pass") is False


def test_held_survives_reopen(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    key = led.register(BASE, target_runs=10)
    led.record_trial(key, *led.lease_next(key, 1), "jobs/j1/t1", FACTS)
    led.close()
    again = Ledger(tmp_path / "l.sqlite")
    assert again.summary(key)["held"] == [1] and again.lease_next(key, 1) == [2]


def test_retries_exhausted_by_review_abandon_the_index(ledger):
    key = ledger.register(BASE, target_runs=10)
    for n in range(3):
        assert ledger.lease_next(key, 1) == [1]
        ledger.record_trial(key, 1, f"jobs/j{n}/t1", FACTS)
        ledger.resolve(key, 1, "retry", reason="network drop")
    assert ledger.lease_next(key, 1) == [2] and ledger.summary(key)["abandoned"] == [1]
